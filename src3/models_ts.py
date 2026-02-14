from functools import partial
from typing import Literal, List, Optional
import torch
from torch import nn

from layers import DepthConv1d, LinearPoolLatent, AttentionPoolLatent


class HierarchicalConvBlock(nn.Module):
    """Multi-scale hierarchical convolutional block"""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        kernel_sizes: List[int] = [11, 7, 5],
        dilations: List[int] = [1, 2, 4],
        causal: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        current_dim = in_features
        
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            block = nn.Sequential(
                DepthConv1d(
                    current_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    causal=causal,
                    groups=current_dim,
                ),
                nn.LayerNorm(current_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(current_dim, hidden_features if i < len(kernel_sizes)-1 else out_features)
            )
            self.blocks.append(block)
            current_dim = hidden_features if i < len(kernel_sizes)-1 else out_features
        
        # Residual connection
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x: torch.Tensor):
        # x: (N, L, C)
        residual = self.residual(x)
        for block in self.blocks:
            x = block(x)
        return x + residual


class HierarchicalFeatureExtractor(nn.Module):
    """Extracts hierarchical features at multiple temporal resolutions"""
    def __init__(
        self,
        in_features: int,
        embed_dim: int = 256,
        num_layers: int = 3,
        kernel_sizes: List[int] = [33, 17, 9],
        causal: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        # Create hierarchical layers with downsampling
        current_dim = in_features
        for i in range(num_layers):
            # Hierarchical conv block at current scale
            layer = HierarchicalConvBlock(
                in_features=current_dim if i == 0 else embed_dim,
                hidden_features=embed_dim * 2,
                out_features=embed_dim,
                kernel_sizes=kernel_sizes,
                causal=causal,
                dropout=dropout,
            )
            self.layers.append(layer)
            
            # Downsample layer for next scale (except last layer)
            if i < num_layers - 1:
                downsample = nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
                self.downsample_layers.append(downsample)
        
    def forward(self, x: torch.Tensor):
        # x: (N, L, C)
        hierarchical_features = []
        current_x = x
        
        # Extract features at multiple scales
        for i, (layer, downsample) in enumerate(zip(self.layers, self.downsample_layers + [None])):
            # Apply hierarchical conv block
            feat = layer(current_x)
            hierarchical_features.append(feat)
            
            # Downsample for next scale
            if downsample is not None:
                current_x = downsample(feat.transpose(1, 2)).transpose(1, 2)
        
        return hierarchical_features[-1], hierarchical_features


class HierarchicalMultiSubjectEncoder(nn.Module):
    """Multi-subject encoder with hierarchical feature extraction"""
    def __init__(
        self,
        num_subjects: int = 4,
        feat_dims: tuple[int | tuple[int, int], ...] = (2048,),
        embed_dim: int = 256,
        target_dim: int = 1000,
        hidden_model: nn.Module | None = None,
        global_pool: Literal["avg", "linear", "attn"] = "avg",
        encoder_kernel_size: int = 33,
        decoder_kernel_size: int = 0,
        encoder_causal: bool = True,
        encoder_positive: bool = False,
        encoder_blockwise: bool = False,
        pool_num_heads: int = 4,
        with_shared_decoder: bool = True,
        with_subject_decoders: bool = True,
        # Hierarchical parameters
        num_hierarchical_layers: int = 3,
        hierarchical_kernel_sizes: List[int] = [33, 17, 9],
        hierarchical_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_subjects = num_subjects
        self.global_pool = global_pool
        
        # Convert feat_dims to list of (nfeats, dim)
        feat_dims = [(1, dim) if isinstance(dim, int) else dim for dim in feat_dims]
        total_feat_size = sum(dim[0] for dim in feat_dims)
        
        # Hierarchical feature extractors for each feature type
        self.hierarchical_extractors = nn.ModuleList()
        self.feat_projection = nn.ModuleList()
        
        for dim in feat_dims:
            feat_dim = dim[1]
            # Projection to common dimension
            projector = nn.Linear(feat_dim, embed_dim)
            self.feat_projection.append(projector)
            
            # Hierarchical feature extractor
            extractor = HierarchicalFeatureExtractor(
                in_features=embed_dim,
                embed_dim=embed_dim,
                num_layers=num_hierarchical_layers,
                kernel_sizes=hierarchical_kernel_sizes,
                causal=encoder_causal,
                dropout=hierarchical_dropout,
            )
            self.hierarchical_extractors.append(extractor)
        
        # Global pooling
        if global_pool == "avg":
            self.register_module("feat_pool", None)
        elif global_pool == "linear":
            self.feat_pool = LinearPoolLatent(embed_dim, total_feat_size)
        elif global_pool == "attn":
            self.feat_pool = AttentionPoolLatent(
                embed_dim, total_feat_size, num_heads=pool_num_heads
            )
        
        # Hidden model (Transformer or Conv1dNext)
        if hidden_model is not None:
            self.hidden_model = hidden_model
        else:
            self.register_module("hidden_model", None)
        
        # Decoders
        if with_shared_decoder:
            self.shared_decoder = nn.Linear(embed_dim, target_dim)
        else:
            self.register_module("shared_decoder", None)
        
        if decoder_kernel_size > 1:
            decoder_linear = partial(ConvLinear, kernel_size=decoder_kernel_size)
        else:
            decoder_linear = nn.Linear
        
        if with_subject_decoders:
            self.subject_decoders = nn.ModuleList(
                [decoder_linear(embed_dim, target_dim) for _ in range(num_subjects)]
            )
        else:
            self.register_module("subject_decoders", None)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: list[torch.Tensor]):
        # features: list of (N, T, D_i)
        
        embed_features: list[torch.Tensor] = []
        
        # Extract hierarchical features for each feature type
        for feat, proj, extractor in zip(features, self.feat_projection, self.hierarchical_extractors):
            # Project to common dimension
            feat_proj = proj(feat)  # (N, T, embed_dim)
            
            # Extract hierarchical features (use only the final scale)
            feat_embed, _ = extractor(feat_proj)
            embed_features.append(feat_embed)
        
        # Global pooling
        if self.global_pool == "avg":
            # Sum features from different feature types
            embed = sum(feat.mean(dim=1) for feat in embed_features)
        else:
            # Concatenate and pool
            embed = torch.cat(embed_features, dim=1)
            embed = embed.transpose(1, 2)
            embed = self.feat_pool(embed)
        
        # Apply hidden model if exists
        if self.hidden_model is not None:
            embed = self.hidden_model(embed)
        
        # Decoding
        if self.shared_decoder is not None:
            shared_output = self.shared_decoder(embed)
            shared_output = shared_output[:, None].expand(-1, self.num_subjects, -1, -1)
        else:
            shared_output = 0.0
        
        if self.subject_decoders is not None:
            subject_output = torch.stack(
                [decoder(embed) for decoder in self.subject_decoders],
                dim=1,
            )
        else:
            subject_output = 0.0
        
        return subject_output + shared_output
    
    def get_hierarchical_features(self, features: List[torch.Tensor]):
        """Extract hierarchical features for analysis"""
        all_hierarchical = []
        
        for feat, proj, extractor in zip(features, self.feat_projection, self.hierarchical_extractors):
            feat_proj = proj(feat)
            _, hierarchical_feats = extractor(feat_proj)
            all_hierarchical.append(hierarchical_feats)
        
        return all_hierarchical