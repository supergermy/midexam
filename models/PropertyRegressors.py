import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, einsum

from models.DifferentialAttention import DiffAttn

class PropertyRegressor(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hid_dim: int, 
        out_dim: int,
    ):
        super().__init__()
        
        self.diff_attn = DiffAttn(in_dim, hid_dim, depth=0)
        self.diff_attn = nn.Linear(in_dim, hid_dim)
        self.to_property = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hid_dim,hid_dim),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim)
        )
    
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        
        x = self.diff_attn(x)
        return self.to_property(x)

class PropertyRegressors(nn.Module):
    def __init__(
        self,
        hid_dim,
        out_dim,
    ):
        super().__init__()
        
        self.rdkit = PropertyRegressor(210, hid_dim, out_dim)
        self.morgan = PropertyRegressor(2048, hid_dim, out_dim)
        self.chembert2a = PropertyRegressor(600, hid_dim, out_dim)
        self.molformer = PropertyRegressor(768, hid_dim, out_dim)
        
        self.lambdas = nn.Parameter(torch.ones([4,1]))
    
    def forward(
        self,
        rdkit: Tensor,
        morgan: Tensor,
        chembert2a: Tensor,
        molformer: Tensor,
    ) -> tuple[Tensor, Tensor]:
        
        rdkit_out = self.rdkit(rdkit)
        morgan_out = self.morgan(morgan)
        chembert2a_out = self.chembert2a(chembert2a)
        molformer_out = self.molformer(molformer)
        
        before_vote = torch.stack([
            rdkit_out,
            morgan_out,
            chembert2a_out,
            molformer_out,
        ], dim=1)  # shape: [batch_size, 5, out_dim]
        
        lambdas = repeat(self.lambdas / (self.lambdas.sum() + 1e-8), 'four one -> batch_size four one', batch_size = rdkit.shape[0])
        
        after_vote = einsum(lambdas, before_vote, 'b f o, b f o -> b f o') # shape: [batch_size, 5, out_dim]
        
        return reduce(after_vote, 'b f o -> b o', 'sum'), self.lambdas