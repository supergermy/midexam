import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

import math

class RMSNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape, 
        eps=1e-8
    ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        x_norm = x / (rms + self.eps)
        return self.gamma * x_norm

class DiffAttn(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        depth,
        num_heads=4,
        dim_head=16,
    ):
        super().__init__()
        self.embed_dim = num_heads * dim_head * 2
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.out_dim = out_dim
        self.scaling = dim_head ** -0.5
        
        self.q = nn.Linear(in_dim, self.embed_dim, bias=False)
        self.kv = nn.Linear(in_dim, self.embed_dim*2, bias=False)
        self.out = nn.Linear(self.embed_dim, out_dim, bias=False)
        
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.dim_head, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.dim_head, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.dim_head, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.dim_head, dtype=torch.float32).normal_(mean=0,std=0.1))
        
        self.subln = RMSNorm(2*self.dim_head)
    
    def forward(
        self,
        x,
    ):
        bsz, embed_dim = x.size()
        
        q, k, v = (self.q(x), *self.kv(x).chunk(2, dim = -1))
        
        q = q.view(bsz, 2 * self.num_heads, self.dim_head)
        k = k.view(bsz, 2 * self.num_heads, self.dim_head)
        v = v.view(bsz, self.num_heads, 2 * self.dim_head)
        
        q *= self.scaling
        dots = torch.matmul(q, k)
        
        attn_weights = F.softmax(dots, dim=-1, dtype=torch.float32).type_as(
            dots
        )
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, tgt_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.dim_head)
        
        return self.out(attn)

