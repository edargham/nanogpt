"""Bigram language model."""

from .baselm import BaseLM
from .layers import TransformerBlock

import torch
import torch.nn as nn


class NanoGPT(BaseLM):
    """Character-level bigram language model backed by a single embedding table.

    Each token's embedding directly represents the logit distribution over the
    next token, so the embedding dimension equals the vocabulary size.

    Args:
        vocab_size: Number of unique tokens in the character vocabulary.
    """
    def __init__(
        self,
        vocab_size: int,
        num_embed: int,
        context_length:int,
        num_heads: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.register_buffer("context_length", torch.tensor(context_length, dtype=torch.int32))

        self.token_embedding_tbl = nn.Embedding(vocab_size, num_embed)
        self.position_embedding_tbl = nn.Embedding(context_length, num_embed)

        self.blocks = nn.Sequential(
            TransformerBlock(
                num_heads,
                num_embed,
                context_length,
            ),
            TransformerBlock(
                num_heads,
                num_embed,
                context_length,
            ),
            TransformerBlock(
                num_heads,
                num_embed,
                context_length,
            ),
            nn.LayerNorm(num_embed),
        )

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
        """Autoregressively extend a token sequence using bigram probabilities.

        At each step takes the last timestep's logits, applies softmax, samples
        the next token via ``torch.multinomial``, and appends it to the sequence.

        Args:
            x: Seed token indices of shape ``(B, T)``.
            max_new_tokens: Number of tokens to append to each sequence.
            scope_context: Whether to scope to windows of context length.

        Returns:
            Extended token indices of shape ``(B, T + max_new_tokens)``.
        """
        return super().generate(x, max_new_tokens, scope_context)