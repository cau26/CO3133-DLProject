"""Shared probability arithmetic for calibration, evaluation and serving."""
import math

import numpy as np


def log_probabilities(logits, temperature=1.0):
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not len(values) or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Logits must be finite nonempty N x K values.")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scaled = values / temperature
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            result = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    except FloatingPointError as error:
        raise ValueError("Temperature scaling produced numerically unusable probabilities.") from error
    if not np.isfinite(result).all():
        raise ValueError("Log probabilities are not finite.")
    return result
