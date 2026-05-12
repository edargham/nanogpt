"""Bigram language model."""

from .baselm import BaseLM
from .layers import SelfAttention

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
        num_embed: int,
        context_length:int,
        device: torch.device
    ):
        super().__init__()
        torch.manual_seed(1337)
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.context_length = context_length
        self.device = device

        self.token_embedding_tbl = nn.Embedding(self.vocab_size, self.num_embed)
        self.position_embedding_tbl = nn.Embedding(self.context_length, self.num_embed)
        self.attention = SelfAttention(32, num_embed=self.num_embed, context_length=self.context_length)
        self.head = nn.Linear(self.num_embed, self.vocab_size)

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
        x = self.attention(x) 
        x = self.head(x)

        return x

    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively extend a token sequence using bigram probabilities.

        At each step takes the last timestep's logits, applies softmax, samples
        the next token via ``torch.multinomial``, and appends it to the sequence.

        Args:
            x: Seed token indices of shape ``(B, T)``.
            max_new_tokens: Number of tokens to append to each sequence.

        Returns:
            Extended token indices of shape ``(B, T + max_new_tokens)``.
        """
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.context_length:]
            logits = self(x_cond)
            logits = logits[:, -1, :]
            probs = fnc.softmax(logits, dim=-1)
            x_nxt = torch.multinomial(probs, 1)
            x = torch.cat((x, x_nxt), dim=1)

        return x
