# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test for rms_norm composite op."""

import platform

import numpy as np
import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm as HFGemma3RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama4.modeling_llama4 import Llama4TextRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from transformers.models.mixtral.modeling_mixtral import MixtralRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

if platform.system() == "Darwin":
    # disable mypy checking so it would not complain on linux
    # also need to calm mypy on MacOS when it sees no error
    import mlx  # type: ignore[import-not-found, unused-ignore]
    import mlx.core  # type: ignore[import-not-found, unused-ignore]
    import mlx.nn  # type: ignore[import-not-found, unused-ignore]

from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
from coreai_torch.composite_ops import RMSNorm, RMSNormImpl

from ..utils import (
    _mlx_array_to_numpy_array,
    _torch_tensor_to_numpy_array,
    filecheck_pattern,
    validate_numerical_output,
)


class TestTorchRMSNorm:
    """Test that torch RMSNorm can be exported and numerics match MLX."""

    @pytest.mark.parametrize("n_heads", [None, 8])
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rms_norm_basic(
        n_heads: int | None,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test RMSNorm eager vs export parity, and eager vs MLX parity."""
        dim = 32
        eps = 1e-5

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = RMSNorm(dim, eps=eps, n_heads=n_heads)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x)

        model = WrapperModel().to(dtype).eval()

        if n_heads is not None:
            x = torch.rand(2, n_heads, 16, dim, dtype=dtype)
        else:
            x = torch.rand(2, dim, dtype=dtype)

        # Torch eager
        output_torch_eager = model(x)

        # Torch export
        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            export_dynamic_shapes = {"x": {0: batch_dim}}
        exported_program = torch.export.export(
            model, args=(x,), dynamic_shapes=export_dynamic_shapes
        )
        output_torch_export = exported_program.module()(x)

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _torch_tensor_to_numpy_array(output_torch_export),
            atol=1e-4,
            rtol=1e-4,
        )

        # MLX comparison (only for standard case — mlx.nn.RMSNorm has no multi-head)
        if n_heads is None:
            mlx_norm = mlx.nn.RMSNorm(dim, eps=eps)
            # Copy weights from our model
            weight_np = model.norm.weight.detach().float().numpy()
            mlx_norm.weight = mlx.core.array(weight_np).astype(
                {
                    torch.float32: mlx.core.float32,
                    torch.float16: mlx.core.float16,
                    torch.bfloat16: mlx.core.bfloat16,
                }[dtype]
            )
            x_mlx = mlx.core.array(_torch_tensor_to_numpy_array(x))
            if dtype == torch.bfloat16:
                x_mlx = x_mlx.astype(mlx.core.bfloat16)
            output_mlx = mlx_norm(x_mlx)

            np.testing.assert_allclose(
                _torch_tensor_to_numpy_array(output_torch_eager),
                _mlx_array_to_numpy_array(output_mlx),
                rtol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                    dtype
                ],
                atol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                    dtype
                ],
            )


class TestTorchRMSNormDetails:
    """Test torch implementation details that are vital for numerics."""

    @staticmethod
    def test_rms_norm_impl_type_handling() -> None:
        """
        Test the type handling logic in RMSNormImpl.forward.

        This test directly verifies the behavior in _rms_norm.py:
        1. When scale has fp32 type and input has different type (fp16), they are
           first multiplied in fp32 precision before converting to the input type.
        2. When they have the same type, the fp32 intermediate tensor is first
           converted to the input type and then multiplied with the scale.
        """
        # Create test data
        batch_size = 24
        seq_len = 3141
        hidden_size = 423
        eps = 1e-5

        # Case 1: scale has fp32 type and input has fp16 type
        input_fp16 = torch.rand(batch_size, seq_len, hidden_size, dtype=torch.float16)
        scale_fp32 = torch.rand(hidden_size, dtype=torch.float32)

        # Case 2: scale and input have the same type (both fp16)
        input_fp16_2 = input_fp16.clone()  # Use same input values for fair comparison
        scale_fp16 = scale_fp32.to(torch.float16)

        # Run both cases
        norm = RMSNormImpl(eps=eps)
        result_diff_types = norm(input_fp16, scale_fp32)
        result_same_types = norm(input_fp16_2, scale_fp16)

        # Verify output types
        assert result_diff_types.dtype == torch.float16, (
            "Output should be fp16 when input is fp16"
        )
        assert result_same_types.dtype == torch.float16, (
            "Output should be fp16 when input is fp16"
        )

        # For reference, manually compute what should happen in each case
        # Common steps for both cases
        axes = norm.axes
        input_f32 = input_fp16.to(torch.float32)
        square_f32 = input_f32 * input_f32
        mean_square_f32 = square_f32.mean(axes, keepdim=True)
        inv_rms_f32 = torch.rsqrt(mean_square_f32 + eps)
        input_normalized = input_fp16 * inv_rms_f32

        # Case 1: Different types - multiply in fp32 then convert to fp16
        expected_diff_types = (input_normalized * scale_fp32).to(torch.float16)

        # Case 2: Same types - convert to fp16 then multiply
        expected_same_types = input_normalized.to(torch.float16) * scale_fp16

        # Check that our results match the expected behavior
        assert torch.allclose(result_diff_types, expected_diff_types), (
            "Different types case should multiply in fp32 then convert to fp16"
        )
        assert torch.allclose(result_same_types, expected_same_types), (
            "Same types case should convert to fp16 then multiply"
        )

    @staticmethod
    def test_rms_norm_numerical_stability() -> None:
        """Test that fp32 intermediate computation prevents fp16 overflow.

        RMSNormImpl squares the input in fp32 to avoid overflow when fp16
        values are near the fp16 max (~65504).
        """
        hidden_size = 64
        eps = 1e-5

        # Values near fp16 max — squaring in fp16 would overflow
        input_fp16 = torch.full((2, hidden_size), 60000.0, dtype=torch.float16)
        scale_fp16 = torch.ones(hidden_size, dtype=torch.float16)

        norm = RMSNormImpl(eps=eps)
        result = norm(input_fp16, scale_fp16)

        assert result.dtype == torch.float16
        assert torch.isfinite(result).all(), (
            "Output should be finite — fp32 intermediate prevents overflow"
        )


class TestRMSNormHuggingFace:
    """Validate coreai_torch RMSNorm numerics against HuggingFace transformers implementations.

    Standard HF RMSNorms (Llama, Mistral, etc.) use weight*1 initialization and
    compute: weight * (x / sqrt(mean(x^2) + eps)).

    Gemma3/Qwen3Next use weight+1 variant with zero-initialized weights:
    (1 + weight) * (x / sqrt(mean(x^2) + eps)).
    """

    @staticmethod
    def _run_standard_test(
        hf_class: type,
        precision: torch.dtype,
    ) -> None:
        """Test standard RMSNorm (weight*1 init) against HF reference."""
        dim, eps = 64, 1e-5
        x = torch.randn(2, 10, dim)

        our_norm = RMSNorm(dim, eps=eps)
        hf_norm = hf_class(dim, eps=eps)

        # Set the same weights
        weight = torch.randn(dim)
        our_norm.weight = torch.nn.Parameter(weight.clone())
        hf_norm.weight = torch.nn.Parameter(weight.clone())

        our_norm = our_norm.to(precision)
        hf_norm = hf_norm.to(precision)
        x = x.to(precision)

        our_out = our_norm(x)
        hf_out = hf_norm(x)

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(our_out),
            _torch_tensor_to_numpy_array(hf_out),
            rtol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                precision
            ],
            atol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                precision
            ],
        )

    @staticmethod
    def _run_gemma3_test(
        hf_class: type,
        precision: torch.dtype,
    ) -> None:
        """Test Gemma3/Qwen3Next RMSNorm (weight+1, zero-init) against HF reference.

        These models use zero-initialized weights and add 1.0 during forward.
        Our RMSNorm uses ones-initialized weights, so we set our weight = hf_weight + 1.
        """
        dim, eps = 64, 1e-6
        x = torch.randn(2, 10, dim)

        hf_norm = hf_class(dim, eps=eps)
        # HF uses zero-init weight, forward does (1 + weight) * normalized
        hf_weight = torch.randn(dim) * 0.1  # small values like real models
        hf_norm.weight = torch.nn.Parameter(hf_weight.clone())

        # Our RMSNorm uses ones-init weight, forward does weight * normalized
        # To match: our_weight = 1 + hf_weight
        our_norm = RMSNorm(dim, eps=eps)
        our_norm.weight = torch.nn.Parameter((1.0 + hf_weight).clone())

        our_norm = our_norm.to(precision)
        hf_norm = hf_norm.to(precision)
        x = x.to(precision)

        our_out = our_norm(x)
        hf_out = hf_norm(x)

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(our_out),
            _torch_tensor_to_numpy_array(hf_out),
            rtol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                precision
            ],
            atol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                precision
            ],
        )

    @pytest.mark.parametrize(
        "precision", [torch.float32, torch.float16, torch.bfloat16]
    )
    @staticmethod
    def test_standard_llama(precision: torch.dtype) -> None:
        """Validate standard RMSNorm numerics match HF Llama."""
        TestRMSNormHuggingFace._run_standard_test(LlamaRMSNorm, precision)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "hf_class",
        [
            Llama4TextRMSNorm,
            MistralRMSNorm,
            MixtralRMSNorm,
            Qwen2RMSNorm,
            Qwen3RMSNorm,
            Qwen3MoeRMSNorm,
        ],
    )
    @pytest.mark.parametrize(
        "precision", [torch.float32, torch.float16, torch.bfloat16]
    )
    @staticmethod
    def test_standard_slow(hf_class: type, precision: torch.dtype) -> None:
        """Validate standard RMSNorm numerics match other HF models.

        Since they are the same as Llama, no need to run them in CI.
        """
        TestRMSNormHuggingFace._run_standard_test(hf_class, precision)

    @pytest.mark.parametrize(
        "precision", [torch.float32, torch.float16, torch.bfloat16]
    )
    @staticmethod
    def test_gemma3(precision: torch.dtype) -> None:
        """Validate Gemma3 RMSNorm (weight+1 variant) numerics match HF."""
        TestRMSNormHuggingFace._run_gemma3_test(HFGemma3RMSNorm, precision)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "precision", [torch.float32, torch.float16, torch.bfloat16]
    )
    @staticmethod
    def test_qwen3_next(precision: torch.dtype) -> None:
        """Validate Qwen3Next RMSNorm (weight+1 variant) numerics match HF."""
        TestRMSNormHuggingFace._run_gemma3_test(Qwen3NextRMSNorm, precision)


class TestTorchRMSNormConversion:
    """Tests for torch rms_norm conversion to Core AI."""

    @staticmethod
    def _make_externalize_spec() -> ExternalizeSpec:
        return ExternalizeSpec(
            target_class=RMSNormImpl,
            composite_op_name="rms_norm",
            composite_attrs=["axes", "eps"],
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rms_norm_basic_ir(dynamic: bool, dtype: torch.dtype) -> None:
        """Test the rms_norm composite operation IR."""

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = RMSNorm(4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x)

        model = WrapperModel().to(dtype).eval()
        x = torch.rand(2, 3, 4, dtype=dtype)

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            seq_dim = torch.export.Dim("seq_len", min=1, max=64)
            export_dynamic_shapes = {"x": {0: batch_dim, 1: seq_dim}}

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x,), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchRMSNormConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        b = "?" if dynamic else 2
        s = "?" if dynamic else 3
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @norm.rmsnorm_impl_[[S:.*]](%arg0: tensor<{b}x{s}x4x{dt}>
        // CHECK-SAME: %arg1: tensor<4x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm" =
        // CHECK-SAME: input_names = ["input", "scale"]
        // CHECK-SAME: op_attrs =
        // CHECK-SAME: axes = -1 : si64
        // CHECK-SAME: eps = 9.99999974E-6 : f32
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @norm.rmsnorm_impl_[[S]](%arg0, %{{{{[a-zA-Z0-9_.]+}}}})
        """
        filecheck_pattern(
            str(converted_program._mlir_module),
            check_file=truth,
        )

    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_rms_norm_basic(dynamic: bool, dtype: torch.dtype) -> None:
        """Test the rms_norm composite operation."""

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = RMSNorm(4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x)

        model = WrapperModel().to(dtype).eval()
        x = torch.rand(2, 3, 4, dtype=dtype)

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            seq_dim = torch.export.Dim("seq_len", min=1, max=64)
            export_dynamic_shapes = {"x": {0: batch_dim, 1: seq_dim}}

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x,), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchRMSNormConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        output_torch_eager = model(x)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            x=x,
        )

    @pytest.mark.ir
    @staticmethod
    def test_rms_norm_scalar_input_ir() -> None:
        """Test rms_norm with scalar input IR."""

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = RMSNorm(1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x)

        model = WrapperModel().eval()
        x = torch.ones(1, dtype=torch.float32)

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x,)
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchRMSNormConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        truth = """
        // CHECK: coreai.graph private noinline @norm.rmsnorm_impl_[[S:.*]](%arg0: tensor<1xf32>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm" =
        // CHECK-SAME: input_names = ["input", "scale"]
        // CHECK-SAME: op_attrs =
        // CHECK-SAME: axes = -1 : si64
        // CHECK-SAME: eps = 9.99999974E-6 : f32
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @norm.rmsnorm_impl_[[S]](%arg0, %{{[a-zA-Z0-9_.]+}})
        """
        filecheck_pattern(
            str(converted_program._mlir_module),
            check_file=truth,
        )

    @staticmethod
    async def test_rms_norm_scalar_input() -> None:
        """Test rms_norm with scalar input."""

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = RMSNorm(1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x)

        model = WrapperModel().eval()
        x = torch.ones(1, dtype=torch.float32)

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x,)
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchRMSNormConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        output_torch_eager = model(x)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol=1e-6,
            atol=1e-6,
            x=x,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_rms_norm_n_heads_ir(dynamic: bool, dtype: torch.dtype) -> None:
        """Test rms_norm with n_heads for fused query & key normalization IR."""

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = RMSNorm(dim=32, n_heads=8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x)

        model = WrapperModel().to(dtype).eval()
        # query_key shape: (batch, num_heads, seq_len, head_dim)
        x = torch.rand(2, 8, 16, 32, dtype=dtype)

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            seq_dim = torch.export.Dim("seq_len", min=1, max=64)
            export_dynamic_shapes = {"x": {0: batch_dim, 2: seq_dim}}

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x,), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchRMSNormConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        b = "?" if dynamic else 2
        s = "?" if dynamic else 16
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @norm.rmsnorm_impl_[[S:.*]](%arg0: tensor<{b}x8x{s}x32x{dt}>
        // CHECK-SAME: %arg1: tensor<8x1x32x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm" =
        // CHECK-SAME: input_names = ["input", "scale"]
        // CHECK-SAME: op_attrs =
        // CHECK-SAME: axes = -1 : si64
        // CHECK-SAME: eps = 9.99999974E-6 : f32
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @norm.rmsnorm_impl_[[S]](%arg0, %{{{{[a-zA-Z0-9_.]+}}}})
        """
        filecheck_pattern(
            str(converted_program._mlir_module),
            check_file=truth,
        )

    @pytest.mark.parametrize("dynamic", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_rms_norm_n_heads(dynamic: bool, dtype: torch.dtype) -> None:
        """Test rms_norm with n_heads for fused query & key normalization."""

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = RMSNorm(dim=32, n_heads=8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x)

        model = WrapperModel().to(dtype).eval()
        # query_key shape: (batch, num_heads, seq_len, head_dim)
        x = torch.rand(2, 8, 16, 32, dtype=dtype)

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            seq_dim = torch.export.Dim("seq_len", min=1, max=64)
            export_dynamic_shapes = {"x": {0: batch_dim, 2: seq_dim}}

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(x,), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchRMSNormConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        output_torch_eager = model(x)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            x=x,
        )
