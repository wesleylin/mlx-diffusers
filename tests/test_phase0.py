import mlx.core as mx

import mlx_diffusers


def test_package_imports():
    assert mlx_diffusers.__version__ == "0.0.1"


def test_mlx_eval_smoke():
    out = mx.array([1.0])
    mx.eval(out)
    assert float(out.item()) == 1.0
