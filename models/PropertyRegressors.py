import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, einsum
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        pe = torch.zeros(1, d_model)  # [1, 2048] for broadcasting
        position = torch.arange(d_model).unsqueeze(0)  # [1, 2048]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        div_term = div_term[:d_model//2]
        
        pe[0, 0::2] = torch.sin(position[0, ::2] * div_term)
        pe[0, 1::2] = torch.cos(position[0, 1::2] * div_term)
        
        self.register_buffer('pe', pe)  # [1, 2048]
        
    def forward(self, x):  # x: [batch_size, 2048]
        return x + self.pe  # pe will broadcast across batch dimension

class PropertyRegressor(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hid_dim: int, 
        out_dim: int,
    ):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Dropout(0.2),
            nn.GELU(),
            )
        self.to_property = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
        )
    
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        
        x = self.input_layer(x)
        return self.to_property(x)

class PropertyRegressors(nn.Module):
    def __init__(
        self,
        hid_dim,
        out_dim,
    ):
        super().__init__()
        
        self.rdkit = PropertyRegressor(20, hid_dim, out_dim)
        self.morgan = PropertyRegressor(2048, hid_dim, out_dim)
        self.chembert2a = PropertyRegressor(600, hid_dim, out_dim)
        self.molformer = PropertyRegressor(768, hid_dim, out_dim)
        
        self.lambdas = nn.Parameter(torch.ones([4,1]))
        
        self.morgan_pe = PositionalEncoding(2048)
        self.rdkit_pe = PositionalEncoding(20)
    
    def forward(
        self,
        rdkit: Tensor,
        morgan: Tensor,
        chembert2a: Tensor,
        molformer: Tensor,
    ) -> tuple[Tensor, Tensor]:
        
        morgan = self.morgan_pe(morgan)
        
        rdkit_out = self.rdkit(rdkit)
        morgan_out = self.morgan(morgan)
        chembert2a_out = self.chembert2a(chembert2a)
        molformer_out = self.molformer(molformer)
        
        before_vote = torch.stack([
            rdkit_out,
            morgan_out,
            chembert2a_out,
            molformer_out,
        ], dim=1)  # shape: [batch_size, 4, out_dim]
        
        normalized_lambdas = F.softmax(self.lambdas, dim=0)
        lambdas = repeat(normalized_lambdas, 'four one -> batch_size four one', batch_size=rdkit.shape[0])
        
        after_vote = torch.einsum('bfo,bfo->bo', lambdas, before_vote)
        
        return after_vote, normalized_lambdas