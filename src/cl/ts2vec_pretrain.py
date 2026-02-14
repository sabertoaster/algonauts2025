import argparse
import sys
import os
import time
import json
import shutil
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from omegaconf import OmegaConf

# Add this import at the top
from torch.utils.data import default_collate

def custom_collate_fn(batch):
    """
    Custom collate to handle variable length fMRI if crop failed,
    or simply to debug which item is failing.
    """
    # Filter out None items if any
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
        
    # Check shapes before stacking
    first_shape = batch[0]['fmri'].shape
    for i, item in enumerate(batch):
        current_shape = item['fmri'].shape
        if current_shape != first_shape:
            # If lengths differ, force crop to minimum length in batch
            # This is a safety fallback for pretraining
            min_len = min(current_shape[1], first_shape[1])
            print(f"Resizing batch item {i} from {current_shape} to {first_shape} (len={min_len})")
            
            # Crop everyone to min_len
            for j in range(len(batch)):
                batch[j]['fmri'] = batch[j]['fmri'][:, :min_len, :]
            break
            
    return default_collate(batch)

# --- Path Setup to find sibling modules ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # points to 'src'
sys.path.append(parent_dir)

from data import (
    Algonauts2025Dataset,
    load_algonauts2025_friends_fmri,
    load_algonauts2025_movie10_fmri,
    episode_filter
)
from cl.models import TS2Vec

# --- Constants ---
# Default to all subjects for pretraining to maximize data diversity
SUBJECTS = (1, 2, 3, 5) 
DEFAULT_DATA_DIR = Path("/raid/nhdang01/fmri_encoder/algonauts2025/datasets/algonauts_2025.competitors")

def get_args():
    parser = argparse.ArgumentParser(description="Pretrain TS2Vec on fMRI Data")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to algonauts_2025.competitors")
    parser.add_argument("--out_dir", type=str, default="./results/ts2vec_pretrain", help="Where to save the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (per GPU)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--sample_length", type=int, default=200, help="Length of time window to crop for training")
    parser.add_argument("--temporal_unit", type=int, default=5, help="Relax temporal precision for fMRI HRF")
    parser.add_argument("--input_dim", type=int, default=1000, help="Number of fMRI parcels")
    parser.add_argument("--output_dim", type=int, default=192, help="Size of learned embedding")
    return parser.parse_args()

def make_fmri_only_loader(args):
    print("Loading fMRI data for pretraining...")
    data_dir = Path(args.data_dir)
    
    friends_fmri = load_algonauts2025_friends_fmri(data_dir, subjects=SUBJECTS)
    movie10_fmri = load_algonauts2025_movie10_fmri(data_dir, subjects=SUBJECTS, runs=[1, 2])
    
    all_fmri = {**friends_fmri, **movie10_fmri}
    all_episodes = list(all_fmri.keys())
    print(f"Found {len(all_episodes)} total fMRI episodes.")

    # FORCE sample_length to be an integer
    # Your error implies stacking failed, so we must guarantee uniform size.
    # The dataset class handles random cropping if sample_length is set.
    dataset = Algonauts2025Dataset(
        episode_list=all_episodes,
        fmri_data=all_fmri,
        feat_data=None, 
        sample_length=64,
        num_samples=2000,
        shuffle=True,
        seed=42
    )

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=custom_collate_fn, # Use safety collate
        drop_last=True # Drop incomplete batches to avoid shape issues at end
    )
    return loader

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup Output Directory
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print(f"Warning: {out_dir} exists. Overwriting...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    
    # Save Config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # 1. Get Data
    train_loader = make_fmri_only_loader(args)

    # 2. Initialize Model
    print(f"Initializing TS2Vec (In: {args.input_dim}, Out: {args.output_dim}, Temporal Unit: {args.temporal_unit})...")
    
    # We use the TS2Vec class wrapper from src/cl/models.py
    # It handles the optimizer and loss calculation internally in .fit(),
    # BUT since your dataloader is custom, we will extract the internal net 
    # and write a custom loop to handle the (Subjects, Time, Parcels) shape.
    
    model = TS2Vec(
        input_dims=args.input_dim,
        output_dims=args.output_dim,
        hidden_dims=64,
        depth=10,
        device=device,
        batch_size=args.batch_size,
        temporal_unit=args.temporal_unit, # Handles HRF lag
        lr=args.lr
    )
    
    # Access the internal optimizer created by TS2Vec init
    optimizer = torch.optim.AdamW(model._net.parameters(), lr=args.lr)
    
    # 3. Training Loop
    print("Starting Training...")
    model._net.train()
    
    step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0
        
        # Wrap loader with tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            # batch is a dict: {'fmri': tensor(B, Subjects, T, 1000), 'episode': ...}
            
            # 1. Prepare Data
            # Shape: (Batch, Subjects, Time, Parcels) -> (Batch * Subjects, Time, Parcels)
            # We treat every subject as an independent instance for contrastive learning
            fmri_data = batch['fmri']
            B, S, T, C = fmri_data.shape
            x = fmri_data.view(B * S, T, C).to(device) # Flatten subjects into batch
            
            # 2. TS2Vec Magic (Cropping & Loss)
            # The logic inside TS2Vec.fit is complex to extract, so we replicate the
            # core "cropping + forward + loss" steps here using the helper functions 
            # available in models.py (which imports soft_losses.py)
            
            optimizer.zero_grad()
            
            # Create two views by cropping (Standard TS2Vec augmentation)
            # The model expects input x. Using helper logic from TS2Vec.fit:
            ts_l = x.size(1)
            crop_l = np.random.randint(low=2 ** (args.temporal_unit + 1), high=ts_l+1)
            
            # Randomly crop two overlapping windows
            crop_left = np.random.randint(ts_l - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
            crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
            
            from cl.utils import take_per_row
            from cl.soft_losses import hier_CL_soft
            
            # Forward pass on two views
            out1_all = model._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
            out2_all = model._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
            
            out1 = out1_all[:, -crop_l:]
            out2 = out2_all[:, :crop_l]
            
            # Calculate Loss (Hard/Standard CL because soft_labels is None)
            loss = hier_CL_soft(
                out1,
                out2,
                None, # soft_labels_batch (None = standard self-supervised)
                lambda_=0.5,
                temporal_unit=args.temporal_unit
            )
            
            loss.backward()
            optimizer.step()
            model.net.update_parameters(model._net) # Update averaged model
            
            pbar.set_postfix({'loss': loss.item()})
            
            epoch_loss += loss.item()
            n_batches += 1
            step += 1
            
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/n_batches:.4f}")
        
        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = out_dir / f"checkpoint_epoch_{epoch+1}.pth"
            model.save(str(save_path))
            
    # 4. Final Save
    final_path = out_dir / "pretrained_fmri_encoder.pth"
    print(f"Saving final model to {final_path}")
    model.save(str(final_path))

if __name__ == "__main__":
    main()
