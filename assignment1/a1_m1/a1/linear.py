"""Linear/softmax classifier. Return logits, NOT softmax probabilities."""
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

    def forward(self, x):
        return self.layers(x)
