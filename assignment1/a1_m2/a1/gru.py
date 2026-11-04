"""Treat 28 image rows (default) or columns as timesteps of 28 pixels."""
import math
from numbers import Real

from torch import nn


class RowGRU(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, dropout=0.0, sequence_axis="rows"):
        super().__init__()
        for name, value in (("hidden_size", hidden_size), ("num_layers", num_layers)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(sequence_axis, str) or sequence_axis not in ("rows", "columns"):
            raise ValueError("sequence_axis must be 'rows' or 'columns'")
        self.sequence_axis = sequence_axis
        self.gru = nn.GRU(28, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28) or x.shape[0] < 1:
            raise ValueError("GRU input must have shape [N,1,28,28] with N > 0")
        sequence = x.squeeze(1)
        if self.sequence_axis == "columns":
            sequence = sequence.transpose(1, 2).contiguous()
        _, hidden = self.gru(sequence)
        return self.classifier(hidden[-1])
