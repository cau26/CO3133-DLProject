"""Optional recipe changes without modifying old architectures or checkpoints."""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

from a1.gru import RowGRU
from a1.mlp import MLP
from a1.train import build_model as legacy_build_model, model_config


def _head_dropout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("head_dropout must be finite and in [0,1)")
    return float(value)


class HeadDropoutMLP(MLP):
    """Functional hidden dropout preserves every legacy Sequential key."""

    def __init__(self, hidden_size: int = 256, head_dropout: float = 0.0):
        self.head_dropout = _head_dropout(head_dropout)
        super().__init__(hidden_size)

    def forward(self, x):
        if self.head_dropout == 0.0:
            return super().forward(x)
        hidden = self.layers[2](self.layers[1](self.layers[0](x)))
        return self.layers[3](F.dropout(hidden, p=self.head_dropout, training=self.training))


class HeadDropoutGRU(RowGRU):
    """External hidden-state dropout; legacy inter-layer dropout is unchanged."""

    def __init__(self, hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.0, sequence_axis: str = "rows",
                 head_dropout: float = 0.0):
        self.head_dropout = _head_dropout(head_dropout)
        super().__init__(hidden_size, num_layers, dropout, sequence_axis)

    def forward(self, x):
        if self.head_dropout == 0.0:
            return super().forward(x)
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28) or x.shape[0] < 1:
            raise ValueError("GRU input must have shape [N,1,28,28] with N > 0")
        sequence = x.squeeze(1)
        if self.sequence_axis == "columns":
            sequence = sequence.transpose(1, 2).contiguous()
        _, hidden = self.gru(sequence)
        return self.classifier(F.dropout(hidden[-1], p=self.head_dropout,
                                         training=self.training))


def initialize_independent_encoder(model: nn.Module, training_seed: int) -> None:
    """Versioned encoder-only CPU init; do not consume the caller's RNG stream.

    A separate generator, seeded 303000+training_seed, visits layers in order:
    attention in-projection, out-projection, FFN linear1, FFN linear2. Matrices
    receive Xavier uniform gain 1; biases zero; LayerNorm scale 1 and bias 0.
    Embedding, position, output norm and classifier retain legacy initialization.
    """
    generator = torch.Generator(device="cpu").manual_seed(303000 + training_seed)
    with torch.no_grad():
        for layer in model.encoder.layers:
            matrices = (layer.self_attn.in_proj_weight, layer.self_attn.out_proj.weight,
                        layer.linear1.weight, layer.linear2.weight)
            for parameter in matrices:
                if parameter.device.type != "cpu":
                    raise ValueError("independent_v1 must initialize the model on CPU before device transfer")
                nn.init.xavier_uniform_(parameter, gain=1.0, generator=generator)
            for parameter in (layer.self_attn.in_proj_bias, layer.self_attn.out_proj.bias,
                              layer.linear1.bias, layer.linear2.bias,
                              layer.norm1.bias, layer.norm2.bias):
                if parameter is not None:
                    parameter.zero_()
            layer.norm1.weight.fill_(1.0)
            layer.norm2.weight.fill_(1.0)


def build_model(family: str, config: dict, training_seed: int) -> nn.Module:
    """Construct on CPU from a fixed training seed, preserving C0 initialization.

    The engine seeds Python/NumPy/data-loader state separately. Seeding Torch
    here makes this API's initialization explicit and identical for every recipe
    at a given training seed. Independent encoder initialization consumes only
    its private generator, so unrelated parameters and subsequent RNG are equal.
    """
    if type(training_seed) is not int or not 0 <= training_seed < 2 ** 32:
        raise ValueError("training_seed must be an integer in [0,2**32)")
    resolved = model_config(family, config)
    head_dropout = _head_dropout(resolved.get("head_dropout", 0.0))
    init_policy = resolved.get("encoder_init", "legacy")
    if init_policy not in ("legacy", "independent_v1"):
        raise ValueError(f"Unknown encoder_init policy: {init_policy}")
    torch.manual_seed(training_seed)
    if family == "mlp":
        model = HeadDropoutMLP(resolved.get("hidden_size", 256), head_dropout)
    elif family == "gru":
        model = HeadDropoutGRU(resolved.get("gru_hidden_size", 128),
                               resolved.get("gru_num_layers", 1),
                               resolved.get("dropout", 0.0),
                               resolved.get("sequence_axis", "rows"), head_dropout)
    else:
        model = legacy_build_model(family, resolved)
    if family == "transformer" and init_policy == "independent_v1":
        initialize_independent_encoder(model, training_seed)
    return model
