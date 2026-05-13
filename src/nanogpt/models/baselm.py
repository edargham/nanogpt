"""Abstract base class for all language models in this package."""

import torch
import torch.nn as nn
from torch.nn import functional as fnc

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

    def generate(self, x: torch.Tensor, max_new_tokens: int, scope_context: bool = False) -> torch.Tensor:
        """Autoregressively extend a token sequence.

        Args:
            x: Seed token indices of shape ``(B, T)``.
            max_new_tokens: Number of tokens to append to each sequence.
            scope_context: Whether or not to restrict to a window of context length.

        Returns:
            Extended token indices of shape ``(B, T + max_new_tokens)``.
        """
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.context_length:] if scope_context else x
            logits = self(x_cond)
            logits = logits[:, -1, :]
            probs = fnc.softmax(logits, dim=-1)
            x_nxt = torch.multinomial(probs, 1)
            x = torch.cat((x, x_nxt), dim=1)

        return x