import torch
from torch import nn
from cl.models import TS2Vec, TSEncoder
from cl.soft_losses import hier_CL_soft, inst_CL_soft, temp_CL_soft

## Let's test the function here

# This file provides an overview and usage examples for the soft contrastive learning implementation.
# The implementation is divided into two main parts:
# 1. Soft Contrastive Losses: Functions for calculating instance-wise, temporal, and hierarchical contrastive losses.
# 2. Models: The TSEncoder and TS2Vec models that learn time series representations.

# -------------------------------------------------------------------------------------------------
# 0. Losses
# -------------------------------------------------------------------------------------------------
# The loss functions are the core of the contrastive learning framework.

# 0.1 Temporal Contrastive Loss (temp_CL_soft)
# This loss encourages the model to learn representations that are close in time for the same time series.
# It uses a soft assignment based on the time difference between timestamps.
#
# Usage:
# z1 (torch.Tensor): Representations of the first augmented view. Shape: (batch_size, T, features).
# z2 (torch.Tensor): Representations of the second augmented view. Shape: (batch_size, T, features).
# timelag_L (torch.Tensor): Soft assignment weights for the first view based on temporal distance.
# timelag_R (torch.Tensor): Soft assignment weights for the second view based on temporal distance.
# loss = temp_CL_soft(z1, z2, timelag_L, timelag_R)

# 0.2 Instance-wise Contrastive Loss (inst_CL_soft)
# This loss distinguishes between different time series instances in a batch.
# It uses a soft assignment based on the similarity between different time series in the original data space (e.g., using DTW distance).
#
# Usage:
# z1 (torch.Tensor): Representations of the first augmented view. Shape: (batch_size, T, features).
# z2 (torch.Tensor): Representations of the second augmented view. Shape: (batch_size, T, features).
# soft_labels_L (torch.Tensor): Pre-computed soft assignment weights for instance pairs for the first view.
# soft_labels_R (torch.Tensor): Pre-computed soft assignment weights for instance pairs for the second view.
# loss = inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)


# 0.3 Hierarchical Contrastive Loss (hier_CL_soft)
# This is the main loss function used in training. It combines the instance-wise and temporal losses
# and applies them hierarchically across different scales of the time series representation.
# The hierarchical application is achieved by repeatedly applying max-pooling to the representations.
#
# Usage:
# z1 (torch.Tensor): Representations of the first augmented view.
# z2 (torch.Tensor): Representations of the second augmented view.
# soft_labels (np.array): A matrix of pre-computed similarities between instances in the batch.
# lambda_ (float): Weight for the instance-wise loss vs. temporal loss.
# tau_temp (float): Sharpness parameter for temporal soft assignments.
# soft_temporal (bool): If True, use soft temporal contrastive loss.
# soft_instance (bool): If True, use soft instance-wise contrastive loss.
# loss = hier_CL_soft(z1, z2, soft_labels, lambda_=0.5, tau_temp=2, soft_temporal=True, soft_instance=True)


# -------------------------------------------------------------------------------------------------
# 1. TSEncoder
# -------------------------------------------------------------------------------------------------
# The TSEncoder is a deep neural network that maps a raw time series to a sequence of representations.
# It uses a series of dilated convolutional layers to capture temporal patterns at different scales.
#
# Usage:
# model = TSEncoder(input_dims=1, output_dims=320, hidden_dims=64, depth=10)
# x = torch.randn(16, 100, 1)  # Batch of 16 time series of length 100 with 1 feature
# representations = model(x) # Shape: (16, 100, 320)


# -------------------------------------------------------------------------------------------------
# 2. TS2Vec
# -------------------------------------------------------------------------------------------------
# The TS2Vec class is a wrapper that orchestrates the training of the TSEncoder using the
# hierarchical contrastive loss. It handles data loading, augmentation, optimization, and encoding.
#
# Usage:
# model = TS2Vec(
#     input_dims=train_data.shape[-1],
#     device=device,
#     soft_instance=True,
#     soft_temporal=True
# )
# model.fit(
#     train_data,
#     train_labels,
#     test_data,
#     test_labels,
#     soft_labels=precomputed_soft_labels,
#     run_dir='./training',
#     n_epochs=20
# )
# learned_representations = model.encode(data_to_encode)


# -------------------------------------------------------------------------------------------------
# 3. Total loss between contrastive and reconstruction loss
# -------------------------------------------------------------------------------------------------
# This function can be used to combine the self-supervised contrastive loss with a supervised
# reconstruction loss (e.g., MSE), allowing for semi-supervised or multi-task learning.
#
# Args:
#   cl_loss (torch.Tensor): The contrastive learning loss.
#   recon_loss (torch.Tensor): The reconstruction loss (e.g., from a decoder).
#   alpha (float): A weighting factor between 0 and 1. `alpha=1` means only CL loss is used.
#
# Returns:
#   torch.Tensor: The combined loss.
#
def total_loss(cl_loss, recon_loss, alpha=1.0):
    return alpha * cl_loss + (1 - alpha) * recon_loss


