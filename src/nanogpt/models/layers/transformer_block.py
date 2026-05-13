import torch
from torch import nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int, num_embed:int, context_length: int, decoder_only: bool=True):
        super().__init__()
        head_size = num_embed // num_heads
        self.register_buffer('decoder_only', torch.tensor(decoder_only, dtype=torch.bool))
        self.masked_attention = MultiHeadAttention(head_size, num_embed, context_length, num_heads)
        self.ffwd = FeedForward(num_embed)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_embed) for _ in range(num_heads)])
        if not decoder_only:
            self.attention = MultiHeadAttention(head_size, num_embed, context_length, num_heads, mask=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.masked_attention(self.layer_norms[0](x))
        if not self.decoder_only:
            x = x + self.attention(self.layer_norms[1](x))
        return x + self.ffwd(self.layer_norms[-1](x))