"""CPU normal samples matching PyTorch's contiguous float32 path.

PyTorch's CPU generator is MT19937. For a contiguous float32 tensor of at
least 16 elements, ``normal_fill`` writes uniforms first, then applies the
Box-Muller transform in blocks of 16 (``NormalFill16`` in
``ATen/native/cpu/DistributionTemplates.h``). This is the noise
``StableDiffusionPipeline`` draws for a 64x64 latent at seed 0.
"""

from __future__ import annotations

import math

import numpy as np

_TWO_PI = np.float32(2.0 * math.pi)


class MT19937:
    def __init__(self, seed: int):
        self._mt = [0] * 624
        self._index = 624
        self._mt[0] = int(seed) & 0xFFFFFFFF
        for index in range(1, 624):
            previous = self._mt[index - 1]
            self._mt[index] = (1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF

    def random(self) -> int:
        if self._index >= 624:
            self._twist()
        value = self._mt[self._index]
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        self._index += 1
        return value & 0xFFFFFFFF

    def _twist(self) -> None:
        mag01 = (0, 0x9908B0DF)
        mt = self._mt
        for index in range(624):
            mixed = (mt[index] & 0x80000000) | (mt[(index + 1) % 624] & 0x7FFFFFFF)
            mt[index] = mt[(index + 397) % 624] ^ (mixed >> 1) ^ mag01[mixed & 1]
        self._index = 0


def _uniform(generator: MT19937) -> np.float32:
    mask = (1 << 24) - 1
    divisor = np.float32(1.0 / (1 << 24))
    return np.float32(generator.random() & mask) * divisor


def _box_muller_16(data: np.ndarray) -> None:
    for index in range(8):
        u1 = np.float32(1.0) - data[index]
        u2 = data[index + 8]
        radius = np.sqrt(np.float32(-2.0) * np.log(u1)).astype(np.float32)
        theta = np.float32(_TWO_PI * u2)
        data[index] = np.float32(radius * np.cos(theta))
        data[index + 8] = np.float32(radius * np.sin(theta))


def randn(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Float32 normal samples for ``shape`` from ``seed``.

    Matches ``torch.randn(shape, generator=torch.Generator().manual_seed(seed))``
    when the tensor is contiguous, float32, and has at least 16 elements.
    """
    size = int(np.prod(shape))
    if size < 16:
        raise ValueError(f"cpu randn matches the torch vector path for numel >= 16, got {size}")
    generator = MT19937(seed)
    data = np.empty(size, dtype=np.float32)
    for index in range(size):
        data[index] = _uniform(generator)
    for index in range(0, size - 15, 16):
        _box_muller_16(data[index : index + 16])
    if size % 16 != 0:
        tail = data[size - 16 :]
        for index in range(16):
            tail[index] = _uniform(generator)
        _box_muller_16(tail)
    return data.reshape(shape)
