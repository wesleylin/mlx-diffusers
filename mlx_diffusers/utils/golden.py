"""Load golden tensors and compare them by mean and max absolute difference."""

from pathlib import Path

import mlx.core as mx
import numpy as np

ArrayLike = np.ndarray | mx.array


def load_golden(path: str | Path) -> dict[str, np.ndarray]:
    """Load every array in a golden ``.npz`` file."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compare(actual: ArrayLike, expected: ArrayLike) -> tuple[float, float]:
    """Return ``(mean_abs, max_abs)`` after ``mx.eval`` of the difference."""
    actual_mx = actual if isinstance(actual, mx.array) else mx.array(np.asarray(actual))
    expected_mx = expected if isinstance(expected, mx.array) else mx.array(np.asarray(expected))
    diff = mx.abs(actual_mx - expected_mx)
    mean_abs = mx.mean(diff)
    max_abs = mx.max(diff)
    mx.eval(mean_abs, max_abs)
    return float(mean_abs.item()), float(max_abs.item())
