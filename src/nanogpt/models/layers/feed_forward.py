"""Position-wise feed-forward network."""

import torch
from torch import nn


class FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network with ReLU activation.

    Expands the embedding dimension by a factor of 4, applies ReLU, then
    projects back down with dropout. Applied independently to each position
    in the sequence.

    Args:
        num_embed: Dimensionality of the input and output embeddings.
        dropout: Dropout probability applied after the second linear layer.
            Defaults to 0.2.
    """

    def __init__(self, num_embed: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed),
            nn.ReLU(),
            nn.Linear(4 * num_embed, num_embed),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through the feed-forward network.

        Args:
            x: Input tensor of shape ``(B, T, num_embed)``.

        Returns:
            Output tensor of shape ``(B, T, num_embed)``.
        """
        return self.net(x)