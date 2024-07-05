import torch
from torch import nn

from src.model.layer.gelu import GELU


class FeedForwardEqual(nn.Module):
    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=hidden_size),
            GELU(),
            nn.Linear(in_features=hidden_size, out_features=num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
