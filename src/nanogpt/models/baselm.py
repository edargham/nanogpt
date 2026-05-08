"""Abstract base class for all language models in this package."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseLM(nn.Module, ABC):
    """Abstract base class for character-level language models.

    All concrete models must subclass ``BaseLM`` and implement both
    ``forward`` and ``generate``.
    """

    def __init__(self):
        super(BaseLM, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for a batch of token sequences.

        Args:
            x: Integer token indices of shape ``(B, T)``.

        Returns:
            Raw logits of shape ``(B, T, vocab_size)``.
        """

    @abstractmethod
    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively extend a token sequence.

        Args:
            x: Seed token indices of shape ``(B, T)``.
            max_new_tokens: Number of tokens to append to each sequence.

        Returns:
            Extended token indices of shape ``(B, T + max_new_tokens)``.
        """