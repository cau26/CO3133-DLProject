"""Patch Transformer trained from scratch with learned position embeddings."""
import torch
import math
from numbers import Real
from torch import nn


class PatchTransformer(nn.Module):
    def __init__(self, patch_size=4, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, use_position_embedding=True):
        super().__init__()
        for name, value in (("patch_size", patch_size), ("d_model", d_model), ("nhead", nhead),
                            ("num_layers", num_layers), ("dim_feedforward", dim_feedforward)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if 28 % patch_size or d_model % nhead:
            raise ValueError("patch_size must divide 28 and nhead must divide d_model")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if type(use_position_embedding) is not bool:
            raise ValueError("use_position_embedding must be a boolean")
        self.use_position_embedding = use_position_embedding
        self.patch_embedding = nn.Conv2d(1, d_model, patch_size, stride=patch_size)
        position = torch.zeros(1, (28 // patch_size) ** 2, d_model)
        # Consume identical RNG draws in both variants, keeping shared initial weights equal.
        nn.init.trunc_normal_(position, std=0.02)
        self.register_parameter("position", nn.Parameter(position) if use_position_embedding else None)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 10)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28) or x.shape[0] < 1:
            raise ValueError("Transformer input must have shape [N,1,28,28] with N > 0")
        tokens = self.patch_embedding(x).flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens + self.position if self.use_position_embedding else tokens)
        return self.classifier(self.norm(encoded.mean(dim=1)))
