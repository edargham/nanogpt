"""Bigram language model."""

from .baselm import BaseLM

import torch
import torch.nn as nn
from torch.nn import functional as fnc


class BiGramLM(BaseLM):
    """Character-level bigram language model backed by a single embedding table.

    Each token's embedding directly represents the logit distribution over the
    next token, so the embedding dimension equals the vocabulary size.

    Args:
        vocab_size: Number of unique tokens in the character vocabulary.
    """
    def __init__(
        self,
        vocab_size: int,
    ):
        super().__init__()
        self.token_embedding_tbl = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for a batch of token sequences.

        Args:
            x: Integer token indices of shape ``(B, T)``.

        Returns:
            Raw logits of shape ``(B, T, vocab_size)``.
        """
        return self.token_embedding_tbl(x)

    def generate(self, x: torch.Tensor, max_new_tokens: int, scope_context: bool = False) -> torch.Tensor:
        """Autoregressively extend a token sequence using bigram probabilities.

        At each step takes the last timestep's logits, applies softmax, samples
        the next token via ``torch.multinomial``, and appends it to the sequence.

        Args:
            x: Seed token indices of shape ``(B, T)``.
            max_new_tokens: Number of tokens to append to each sequence.
            scope_context: Whether or not to scope to windows of context length.

        Returns:
            Extended token indices of shape ``(B, T + max_new_tokens)``.
        """
        return super().generate(x, max_new_tokens, scope_context)
