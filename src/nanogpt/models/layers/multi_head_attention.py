import torch
from torch import nn

from .self_attention import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int, num_embed: int, context_length: int, num_heads: int, mask: bool=True):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size, num_embed, context_length, mask) for _ in range(num_heads)])
        self.projection = nn.Linear(num_embed, num_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x =  torch.cat([head(x) for head in self.heads], dim=-1)
        return self.projection(x)