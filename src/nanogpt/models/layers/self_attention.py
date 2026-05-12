import torch
from torch import nn

import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, num_embed: int, context_length: int, mask:bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.num_embed = num_embed
        self.context_length = context_length
        self.mask = mask

        self.key_linear = nn.Linear(num_embed, num_heads, bias=False)
        self.query_linear = nn.Linear(num_embed, num_heads, bias=False)
        self.value_linear = nn.Linear(num_embed, num_heads, bias=False)

        if self.mask:
            self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        k = self.key_linear(x)
        q = self.query_linear(x)

        w = q@k.transpose(-2, -1) * c**-0.5

        if self.mask:
            w = w.masked_fill(self.tril[:t, :t] == 0, -np.inf)

        w = nn.functional.softmax(w, dim=-1)

        v = self.value_linear(x)

        out = w @ v
        return out