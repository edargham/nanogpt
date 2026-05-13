"""Transformer block with pre-norm residual connections."""

import torch
from torch import nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """Pre-norm residual transformer block.

    Applies layer normalization before each sub-layer and adds the sub-layer
    output back to the input via a residual connection. In decoder-only mode
    the block contains a single masked multi-head self-attention layer followed
    by a position-wise feed-forward network. When ``decoder_only=False`` a
    second unmasked multi-head attention layer is inserted between the two,
    enabling an encoder-style cross-attention path.

    Args:
        num_heads: Number of attention heads in each multi-head attention layer.
            Also determines the number of ``LayerNorm`` buffers allocated.
        num_embed: Dimensionality of the input and output embeddings.
            Each head uses ``num_embed // num_heads`` as its head size.
        context_length: Maximum sequence length passed to the attention layers.
        decoder_only: If ``True``, only masked self-attention is used.
            If ``False``, an additional unmasked attention layer is added.
            Defaults to ``True``.
    """

    def __init__(self, num_heads: int, num_embed: int, context_length: int, decoder_only: bool = True):
        super().__init__()
        head_size = num_embed // num_heads
        self.register_buffer('decoder_only', torch.tensor(decoder_only, dtype=torch.bool))
        self.masked_attention = MultiHeadAttention(head_size, num_embed, context_length, num_heads)
        self.ffwd = FeedForward(num_embed)
        num_norms = 2 if decoder_only else 3
        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_embed) for _ in range(num_norms)])
        if not decoder_only:
            self.attention = MultiHeadAttention(head_size, num_embed, context_length, num_heads, mask=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformer block to a sequence of embeddings.

        Args:
            x: Input tensor of shape ``(B, T, num_embed)``.

        Returns:
            Output tensor of shape ``(B, T, num_embed)``.
        """
        x = x + self.masked_attention(self.layer_norms[0](x))
        if not self.decoder_only:
            x = x + self.attention(self.layer_norms[1](x))
        return x + self.ffwd(self.layer_norms[-1](x))