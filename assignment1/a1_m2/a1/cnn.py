"""Small custom CNN trained from scratch; no pretrained backbone."""
from torch import nn


class GlobalAveragePool2d(nn.Module):
    """Reduce spatial dimensions explicitly, avoiding adaptive-pooling CUDA backward."""

    def forward(self, x):
        return x.mean(dim=(-2, -1), keepdim=True)


class CustomCNN(nn.Module):
    def __init__(self, channels=(32, 64, 128), dropout=0.2):
        super().__init__()
        blocks, previous = [], 1
        for i, width in enumerate(channels):
            blocks.extend([nn.Conv2d(previous, width, 3, padding=1), nn.ReLU()])
            if i < 2:
                blocks.append(nn.MaxPool2d(2))
            previous = width
        self.features = nn.Sequential(*blocks, GlobalAveragePool2d())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(previous, 10))

    def forward(self, x):
        return self.classifier(self.features(x))
