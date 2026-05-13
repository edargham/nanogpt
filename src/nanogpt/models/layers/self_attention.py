"""Scaled dot-product self-attention head."""

import torch
from torch import nn
from torch.nn import functional as fnc

import numpy as np


class SelfAttention(nn.Module):
    """Single scaled dot-product self-attention head.

    Projects input embeddings into key, query, and value spaces, computes
    scaled attention weights, optionally applies a causal lower-triangular
    mask to prevent attending to future positions, and returns the weighted
    sum of values.

    Args:
        head_size: Dimensionality of the key, query, and value projections.
        num_embed: Dimensionality of the input embeddings.
        context_length: Maximum sequence length; determines the size of the
            causal mask buffer when ``mask`` is ``True``.
        mask: If ``True``, applies a causal mask so each position can only
            attend to earlier positions. Defaults to ``True``.
        dropout: Dropout probability applied to attention weights after
            softmax. Defaults to 0.2.
    """

    def __init__(self, head_size: int, num_embed: int, context_length: int, mask: bool = True, dropout: float = 0.2):
        super().__init__()
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.bool))

        self.key_linear = nn.Linear(num_embed, head_size, bias=False)
        self.query_linear = nn.Linear(num_embed, head_size, bias=False)
        self.value_linear = nn.Linear(num_embed, head_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        if self.mask:
            self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted values for a sequence of embeddings.

        Args:
            x: Input tensor of shape ``(B, T, num_embed)``.

        Returns:
            Output tensor of shape ``(B, T, head_size)``.
        """
        b, t, c = x.shape
        k = self.key_linear(x)
        q = self.query_linear(x)

        w = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        if self.mask:
            w = w.masked_fill(self.tril[:t, :t] == 0, -np.inf)

        w = fnc.softmax(w, dim=-1)
        w = self.dropout(w)

        v = self.value_linear(x)

        out = w @ v
        return out