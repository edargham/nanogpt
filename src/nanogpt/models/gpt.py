"""Bigram language model."""

from .baselm import BaseLM
from .layers import TransformerBlock

import torch
import torch.nn as nn


class NanoGPT(BaseLM):
    """Character-level GPT-style decoder-only transformer.

    Combines token and positional embeddings, passes them through a stack of
    ``TransformerBlock`` layers followed by a final ``LayerNorm``, and projects
    to vocabulary logits via a linear head.

    Args:
        vocab_size: Number of unique tokens in the character vocabulary.
        num_embed: Dimensionality of token and positional embeddings.
        context_length: Maximum sequence length the model can process.
        num_heads: Number of attention heads per transformer block.
        num_layers: Number of stacked transformer blocks.
        device: Device on which positional indices are created during the
            forward pass.
    """
    def __init__(
        self,
        vocab_size: int,
        num_embed: int,
        context_length:int,
        num_heads: int,
        num_layers: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.register_buffer("context_length", torch.tensor(context_length, dtype=torch.int32))

        self.token_embedding_tbl = nn.Embedding(vocab_size, num_embed)
        self.position_embedding_tbl = nn.Embedding(context_length, num_embed)

        self.blocks = nn.Sequential()

        for _ in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    num_heads,
                    num_embed,
                    context_length,
                ),
            )

        self.blocks.append(nn.LayerNorm(num_embed))

        self.head = nn.Linear(num_embed, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for a batch of token sequences.

        Args:
            x: Integer token indices of shape ``(B, T)``.

        Returns:
            Raw logits of shape ``(B, T, vocab_size)``.
        """
        b, t = x.shape
        toks = self.token_embedding_tbl(x)
        pos = self.position_embedding_tbl(torch.arange(t, device=self.device))
        x = toks + pos
        x = self.blocks(x)
        x = self.head(x)

        return x

    def generate(self, x: torch.Tensor, max_new_tokens: int, scope_context: bool = True) -> torch.Tensor:
        """Autoregressively extend a token sequence.

        Delegates to ``BaseLM.generate`` with context scoping enabled by default,
        restricting each forward pass to the last ``context_length`` tokens.

        Args:
            x: Seed token indices of shape ``(B, T)``.
            max_new_tokens: Number of tokens to append to each sequence.
            scope_context: Whether to restrict the input to the last
                ``context_length`` tokens at each step. Defaults to ``True``.

        Returns:
            Extended token indices of shape ``(B, T + max_new_tokens)``.
        """
        return super().generate(x, max_new_tokens, scope_context)