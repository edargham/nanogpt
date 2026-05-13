"""Multi-head self-attention module."""

import torch
from torch import nn

from .self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    """Parallel multi-head self-attention with output projection.

    Runs ``num_heads`` independent ``SelfAttention`` heads, concatenates
    their outputs along the feature dimension, then projects back to
    ``num_embed`` via a linear layer followed by dropout.

    Args:
        head_size: Dimensionality of each individual attention head's
            key, query, and value projections.
        num_embed: Dimensionality of the input and output embeddings.
        context_length: Maximum sequence length passed to each head.
        num_heads: Number of parallel attention heads.
        mask: Whether to apply a causal mask in each head. Defaults to ``True``.
        dropout: Dropout probability applied after the output projection.
            Defaults to 0.2.
    """

    def __init__(self, head_size: int, num_embed: int, context_length: int, num_heads: int, mask: bool = True, dropout: float = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size, num_embed, context_length, mask) for _ in range(num_heads)])
        self.projection = nn.Linear(num_embed, num_embed)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention to a sequence of embeddings.

        Args:
            x: Input tensor of shape ``(B, T, num_embed)``.

        Returns:
            Output tensor of shape ``(B, T, num_embed)``.
        """
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(x))