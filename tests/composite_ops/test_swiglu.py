# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test for swiglu composite op."""

import platform

import numpy as np
import pytest
import torch
import torch.nn.functional as F

if platform.system() == "Darwin":
    import mlx  # type: ignore[import-not-found, unused-ignore]
    import mlx.core  # type: ignore[import-not-found, unused-ignore]
    import mlx.nn  # type: ignore[import-not-found, unused-ignore]

from coreai_torch.composite_ops import SwiGLU, SwiGLUImpl

from ..utils import (
    _mlx_array_to_numpy_array,
    _torch_tensor_to_numpy_array,
)


class TestTorchSwiGLU:
    """Test that SwiGLU and SwiGLUImpl execute correctly and match expected reference."""

    @pytest.mark.parametrize("dim", [32, 64])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_swiglu_impl_single_tensor(self, dim: int, dtype: torch.dtype) -> None:
        """Test SwiGLUImpl when input is a concatenated (gate, val) tensor."""
        x = torch.randn(4, 16, dim * 2, dtype=dtype)
        out = SwiGLUImpl()(x)

        # Reference
        gate, val = torch.chunk(x, 2, dim=-1)
        ref = F.silu(gate) * val

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(out),
            _torch_tensor_to_numpy_array(ref),
            rtol=1e-3 if dtype != torch.bfloat16 else 5e-2,
            atol=1e-3 if dtype != torch.bfloat16 else 5e-2,
        )

    @pytest.mark.parametrize("dim", [32, 64])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_swiglu_impl_two_tensors(self, dim: int, dtype: torch.dtype) -> None:
        """Test SwiGLUImpl when gate and val are passed as two separate tensors."""
        gate = torch.randn(4, 16, dim, dtype=dtype)
        val = torch.randn(4, 16, dim, dtype=dtype)
        out = SwiGLUImpl()(gate, val)

        ref = F.silu(gate) * val

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(out),
            _torch_tensor_to_numpy_array(ref),
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.parametrize("dim", [32, 64])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_swiglu_module_export(self, dim: int, bias: bool, dynamic: bool) -> None:
        """Test SwiGLU nn.Module export and eager vs export parity."""
        module = SwiGLU(dim=dim, bias=bias).eval()
        x = torch.randn(2, 8, dim)

        out_eager = module(x)

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            export_dynamic_shapes = {"x": {0: batch_dim}}

        exported = torch.export.export(
            module, args=(x,), dynamic_shapes=export_dynamic_shapes
        )
        out_export = exported.module()(x)

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(out_eager),
            _torch_tensor_to_numpy_array(out_export),
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.skipif(
        platform.system() != "Darwin", reason="MLX is only available on Darwin"
    )
    @pytest.mark.parametrize("dim", [32, 64])
    def test_swiglu_mlx_parity(self, dim: int) -> None:
        """Test mathematical parity against MLX equivalent on macOS."""
        x_torch = torch.randn(2, 4, dim * 2, dtype=torch.float32)
        out_torch = SwiGLUImpl()(x_torch)

        x_mlx = mlx.core.array(_torch_tensor_to_numpy_array(x_torch))
        gate_mlx, val_mlx = mlx.core.split(x_mlx, 2, axis=-1)
        out_mlx = mlx.nn.silu(gate_mlx) * val_mlx

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(out_torch),
            _mlx_array_to_numpy_array(out_mlx),
            rtol=1e-4,
            atol=1e-4,
        )
