"""Bigram language model."""

import torch
import torch.nn as nn


class BiGramLM(nn.Module):
    """Character-level bigram language model backed by a single embedding table.

    Each token's embedding directly represents the logit distribution over the
    next token, so the embedding dimension equals the vocabulary size.

    Args:
        vocab_size: Number of unique tokens in the character vocabulary.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        torch.manual_seed(1337)
        self.vocab_size = vocab_size

        self.token_embedding_tbl = nn.Embedding(self.vocab_size, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for a batch of token sequences.

        Args:
            x: Integer token indices of shape ``(B, T)``.

        Returns:
            Raw logits of shape ``(B, T, vocab_size)``.
        """
        x = self.token_embedding_tbl(x)
        return x