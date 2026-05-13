import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, num_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed),
            nn.ReLU(),
            nn.Linear(4 * num_embed, num_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)