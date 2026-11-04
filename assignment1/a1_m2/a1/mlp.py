"""MLP with one hidden layer; ReLU makes its mapping nonlinear."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, 10))

    def forward(self, x):
        return self.layers(x)
