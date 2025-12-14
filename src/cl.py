import torch
from torch import nn

# 0. Losses [x]
# 0.1 Temporal
# 0.2 Instace-wise
# 0.3 Hierarchical

# 1. TSEncoder [x]

# 2. TS2Vec [x]

# 3. Total loss between constrastive and MSE loss []
def total_loss(cl_loss, recon_loss, alpha=1.0):
    return alpha * cl_loss + (1 - alpha) * recon_loss


