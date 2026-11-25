import torch
# Small analytic fixtures should not oversubscribe a many-core workstation.
torch.set_num_threads(2)
