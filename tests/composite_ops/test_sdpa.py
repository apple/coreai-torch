# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for sdpa composite op."""

import platform
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import transformers
import transformers.integrations.sdpa_attention

if platform.system() == "Darwin":
    # disable mypy checking so it would not complain on linux
    # also need to calm mypy on MacOS when it sees no error
    import mlx  # type: ignore[import-not-found, unused-ignore]
    import mlx.core  # type: ignore[import-not-found, unused-ignore]
    import mlx.core.fast  # type: ignore[import-not-found, unused-ignore]

from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
from coreai_torch.composite_ops import SDPA
from coreai_torch.composite_ops._sdpa import _maybe_construct_attn_mask

from ..utils import (
    _mlx_array_to_numpy_array,
    _torch_tensor_to_numpy_array,
    filecheck_pattern,
    validate_numerical_output,
)


class TestTorchSDPA:
    """Test that torch implementation of sdpa composite op can be exported and numerics match mlx.

    Covers MHA, MHA with explicit mask, GQA, and GQA with sinks.
    Causal mask and sliding window are included.
    """

    @pytest.mark.parametrize("scale", [None, 1.5])
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("window_size", [0, 1, 2, 11])
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_mha(  # noqa: PLR0913
        scale: float,
        is_causal: bool,
        window_size: int,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Validate Multi-Head Attention numerics match mlx."""
        batch_size = 3
        num_heads = 4
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_heads, seq_len, head_v_dim), dtype=dtype)

        model = SDPA(scale=scale, is_causal=is_causal, window_size=window_size)
        model.eval()
        if dynamic:
            batch_size_dim = torch.export.Dim(name="batch_size", min=1, max=32)
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            seq_len_dim = torch.export.Dim(name="seq_len", min=1, max=2048)
            dynamic_shapes = {
                "query": {0: batch_size_dim, len(query.shape) - 2: q_len_dim},
                "key": {0: batch_size_dim, len(key.shape) - 2: seq_len_dim},
                "value": {0: batch_size_dim, len(value.shape) - 2: seq_len_dim},
            }
        else:
            dynamic_shapes = None
        exported_program = torch.export.export(
            model,
            args=(query, key, value),
            dynamic_shapes=dynamic_shapes,
        )

        output_torch_eager = model(query, key, value)
        output_torch_export = exported_program.module()(query, key, value)
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _torch_tensor_to_numpy_array(output_torch_export),
            atol=1e-4,
            rtol=1e-4,
        )

        query_mlx = mlx.core.array(query)
        key_mlx = mlx.core.array(key)
        value_mlx = mlx.core.array(value)
        mask = _maybe_construct_attn_mask(
            query,
            key,
            is_causal=is_causal,
            window_size=window_size,
        )
        mask_mlx = None if mask is None else mlx.core.array(mask)
        output_mlx = mlx.core.fast.scaled_dot_product_attention(
            query_mlx,
            key_mlx,
            value_mlx,
            scale=head_qk_dim**-0.5 if scale is None else scale,
            mask=mask_mlx,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _mlx_array_to_numpy_array(output_mlx),
            rtol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_mha_with_mask(dynamic: bool, dtype: torch.dtype) -> None:
        """Validate MHA with explicit attn_mask numerics match mlx."""
        batch_size = 3
        num_heads = 4
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_heads, seq_len, head_v_dim), dtype=dtype)

        full_trues = torch.ones((q_len, seq_len), dtype=torch.bool)
        causal_mask = full_trues.tril(diagonal=seq_len - q_len)
        left_out_of_window = full_trues.tril(diagonal=seq_len - q_len - 7)
        attn_mask = torch.logical_xor(causal_mask, left_out_of_window)

        model = SDPA()
        model.eval()
        if dynamic:
            batch_size_dim = torch.export.Dim(name="batch_size", min=1, max=32)
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            seq_len_dim = torch.export.Dim(name="seq_len", min=1, max=2048)
            dynamic_shapes = {
                "query": {0: batch_size_dim, len(query.shape) - 2: q_len_dim},
                "key": {0: batch_size_dim, len(key.shape) - 2: seq_len_dim},
                "value": {0: batch_size_dim, len(value.shape) - 2: seq_len_dim},
                "attn_mask": {0: q_len_dim, 1: seq_len_dim},
            }
        else:
            dynamic_shapes = None
        exported_program = torch.export.export(
            model,
            args=(query, key, value),
            kwargs={"attn_mask": attn_mask},
            dynamic_shapes=dynamic_shapes,
        )

        output_torch_eager = model(query, key, value, attn_mask=attn_mask)
        output_torch_export = exported_program.module()(
            query, key, value, attn_mask=attn_mask
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _torch_tensor_to_numpy_array(output_torch_export),
            atol=1e-4,
            rtol=1e-4,
        )

        query_mlx = mlx.core.array(query)
        key_mlx = mlx.core.array(key)
        value_mlx = mlx.core.array(value)
        mask_mlx = mlx.core.array(attn_mask)
        output_mlx = mlx.core.fast.scaled_dot_product_attention(
            query_mlx,
            key_mlx,
            value_mlx,
            scale=head_qk_dim**-0.5,
            mask=mask_mlx,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _mlx_array_to_numpy_array(output_mlx),
            rtol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

    @pytest.mark.parametrize("scale", [None, 1.5])
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("window_size", [0, 1, 2, 11])
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_gqa(  # noqa: PLR0913
        scale: float,
        is_causal: bool,
        window_size: int,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Validate Grouped Query Attention numerics match mlx."""
        batch_size = 3
        num_heads = 4
        num_kv_heads = 2
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_kv_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_kv_heads, seq_len, head_v_dim), dtype=dtype)

        model = SDPA(scale=scale, is_causal=is_causal, window_size=window_size)
        model.eval()
        if dynamic:
            batch_size_dim = torch.export.Dim(name="batch_size", min=1, max=32)
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            seq_len_dim = torch.export.Dim(name="seq_len", min=1, max=2048)
            dynamic_shapes = {
                "query": {0: batch_size_dim, len(query.shape) - 2: q_len_dim},
                "key": {0: batch_size_dim, len(key.shape) - 2: seq_len_dim},
                "value": {0: batch_size_dim, len(value.shape) - 2: seq_len_dim},
            }
        else:
            dynamic_shapes = None
        exported_program = torch.export.export(
            model,
            args=(query, key, value),
            dynamic_shapes=dynamic_shapes,
        )

        output_torch_eager = model(query, key, value)
        output_torch_export = exported_program.module()(query, key, value)
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _torch_tensor_to_numpy_array(output_torch_export),
            atol=1e-4,
            rtol=1e-4,
        )

        query_mlx = mlx.core.array(query)
        key_mlx = mlx.core.array(key)
        value_mlx = mlx.core.array(value)
        mask = _maybe_construct_attn_mask(
            query,
            key,
            is_causal=is_causal,
            window_size=window_size,
        )
        mask_mlx = None if mask is None else mlx.core.array(mask)
        output_mlx = mlx.core.fast.scaled_dot_product_attention(
            query_mlx,
            key_mlx,
            value_mlx,
            scale=head_qk_dim**-0.5 if scale is None else scale,
            mask=mask_mlx,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _mlx_array_to_numpy_array(output_mlx),
            rtol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

    @pytest.mark.parametrize("scale", [None, 1.5])
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("window_size", [0, 1, 2, 11])
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_gqa_with_sinks(
        scale: float,
        is_causal: bool,
        window_size: int,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Validate that sinks shows up in the arguments if specified."""
        batch_size = 3
        num_heads = 4
        num_kv_heads = 2
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_kv_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_kv_heads, seq_len, head_v_dim), dtype=dtype)
        sinks = torch.zeros((num_heads,), dtype=dtype)

        model = SDPA(scale=scale, is_causal=is_causal, window_size=window_size)
        model.eval()
        if dynamic:
            batch_size_dim = torch.export.Dim(name="batch_size", min=1, max=32)
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            seq_len_dim = torch.export.Dim(name="seq_len", min=1, max=2048)
            dynamic_shapes = {
                "query": {0: batch_size_dim, len(query.shape) - 2: q_len_dim},
                "key": {0: batch_size_dim, len(key.shape) - 2: seq_len_dim},
                "value": {0: batch_size_dim, len(value.shape) - 2: seq_len_dim},
                "sinks": {},
            }
        else:
            dynamic_shapes = None
        exported_program = torch.export.export(
            model,
            args=(query, key, value),
            kwargs={"sinks": sinks},
            dynamic_shapes=dynamic_shapes,
        )

        output_torch_eager = model(query, key, value, sinks=sinks)
        output_torch_export = exported_program.module()(query, key, value, sinks=sinks)
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _torch_tensor_to_numpy_array(output_torch_export),
            atol=1e-4,
            rtol=1e-4,
        )

        query_mlx = mlx.core.array(query)
        key_mlx = mlx.core.array(key)
        value_mlx = mlx.core.array(value)
        mask = _maybe_construct_attn_mask(
            query,
            key,
            is_causal=is_causal,
            window_size=window_size,
        )
        mask_mlx = None if mask is None else mlx.core.array(mask)
        sinks_mlx = mlx.core.array(sinks)
        output_mlx = mlx.core.fast.scaled_dot_product_attention(
            query_mlx,
            key_mlx,
            value_mlx,
            scale=head_qk_dim**-0.5 if scale is None else scale,
            mask=mask_mlx,
            sinks=sinks_mlx,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            _mlx_array_to_numpy_array(output_mlx),
            rtol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
        )


class TestTorchSDPAHuggingFace:
    """Test that torch implementation of sdpa composite op numerics match transformers.

    Uses transformers' sdpa_attention_forward as an independent reference
    implementation, complementing the MLX-based checks in TestTorchSDPA.
    """

    @staticmethod
    def _hf_reference(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run HF sdpa_attention_forward and return output in (B, H, S, D) layout."""
        n_heads = query.shape[1]
        n_kv_heads = key.shape[1]

        @dataclass
        class MockModule:
            num_key_value_groups = n_heads // n_kv_heads
            training = False

        output_hf, _ = transformers.integrations.sdpa_attention.sdpa_attention_forward(
            module=MockModule(),
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            scaling=scale,
        )
        # HF returns (B, S, H, D), convert to (B, H, S, D)
        return output_hf.permute(0, 2, 1, 3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_basic_causal(dtype: torch.dtype) -> None:
        """Validate basic causal SDPA numerics match transformers (Llama/Qwen style)."""
        batch = 2
        seq = 10
        head_dim = 24
        num_heads = 8
        num_kv_heads = 4
        scale = 3.14159

        query = torch.rand(batch, num_heads, seq, head_dim, dtype=dtype)
        key = torch.rand(batch, num_kv_heads, seq, head_dim, dtype=dtype)
        value = torch.rand(batch, num_kv_heads, seq, head_dim, dtype=dtype)
        attention_mask = (
            torch.triu(torch.full((seq, seq), float("-inf"), dtype=dtype), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        model = SDPA(scale=scale, is_causal=True, _use_hf_impl=True)
        model.eval()
        output_ours = model(query, key, value)

        output_hf = TestTorchSDPAHuggingFace._hf_reference(
            query,
            key,
            value,
            scale,
            attention_mask,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_ours),
            _torch_tensor_to_numpy_array(output_hf),
            rtol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

    @pytest.mark.parametrize(
        "heads",
        [((1, 1)), ((8, 8)), ((8, 4))],
        ids=["mha-1", "mha-8", "gqa-8-4"],
    )
    @pytest.mark.parametrize("sliding_window", [1, 4, 1000])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_sliding_window(
        heads: tuple[int, int],
        sliding_window: int,
        dtype: torch.dtype,
    ) -> None:
        """Validate sliding window attention numerics match transformers (Gemma3 style)."""
        batch = 2
        seq = 15
        head_dim = 24
        num_heads, num_kv_heads = heads
        scale = 3.14159

        query = torch.rand(batch, num_heads, seq, head_dim, dtype=dtype)
        key = torch.rand(batch, num_kv_heads, seq, head_dim, dtype=dtype)
        value = torch.rand(batch, num_kv_heads, seq, head_dim, dtype=dtype)

        # Gemma3-style float -inf mask: causal + sliding window
        full_trues = torch.ones((seq, seq), dtype=torch.bool)
        causal_mask = full_trues.tril(diagonal=0)
        left_out_of_window = full_trues.tril(diagonal=-sliding_window)
        window_mask = torch.logical_xor(causal_mask, left_out_of_window)
        attention_mask = (
            torch.where(
                window_mask,
                torch.zeros((seq, seq), dtype=dtype),
                torch.full((seq, seq), float("-inf"), dtype=dtype),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        model = SDPA(
            scale=scale, is_causal=True, window_size=sliding_window, _use_hf_impl=True
        )
        model.eval()
        output_ours = model(query, key, value)

        output_hf = TestTorchSDPAHuggingFace._hf_reference(
            query,
            key,
            value,
            scale,
            attention_mask,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_ours),
            _torch_tensor_to_numpy_array(output_hf),
            rtol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
        )


class TestTorchSDPAConversion:
    """Tests for SDPA composite op conversion using ExternalizeSpec.

    Each test corresponds to a variant of the forward call.  All tests are
    parametrised over dynamic shapes, dtypes, scale, is_causal and window_size
    to provide comprehensive coverage.
    """

    @staticmethod
    def _make_externalize_spec() -> ExternalizeSpec:
        return ExternalizeSpec(
            target_class=SDPA,
            composite_op_name="scaled_dot_product_attention",
            composite_attrs=["scale", "is_causal", "window_size"],
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("scale", [None, 1.5])
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("window_size", [0, 1, 2, 11])
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_mha_ir(  # noqa: PLR0913
        scale: float,
        is_causal: bool,
        window_size: int,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Validate the Multi-Head Attention (MHA) use case of sdpa composite op IR."""
        batch_size = 3
        num_heads = 4
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_heads, seq_len, head_v_dim), dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sdpa = SDPA(
                    scale=scale, is_causal=is_causal, window_size=window_size
                )

            def forward(
                self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                return self.sdpa(query, key, value)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            seq_dim = torch.export.Dim("seq_len", min=1, max=2048)
            export_dynamic_shapes = {
                "query": {0: batch_dim, 2: q_dim},
                "key": {0: batch_dim, 2: seq_dim},
                "value": {0: batch_dim, 2: seq_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(query, key, value), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchSDPAConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        b = "?" if dynamic else batch_size
        q = "?" if dynamic else q_len
        s = "?" if dynamic else seq_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @sdpa_[[S:.*]](%arg0: tensor<{b}x{num_heads}x{q}x{head_qk_dim}x{dt}>
        // CHECK-SAME: %arg1: tensor<{b}x{num_heads}x{s}x{head_qk_dim}x{dt}>
        // CHECK-SAME: %arg2: tensor<{b}x{num_heads}x{s}x{head_v_dim}x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention" =
        // CHECK-SAME: input_names = ["query", "key", "value"]
        // CHECK-SAME: op_attrs =
        // CHECK-SAME: is_causal = {str(is_causal).lower()}
        {"// CHECK-SAME: scale = " if scale is not None else ""}
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: window_size = {window_size!r} : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @sdpa_[[S]](
        """
        filecheck_pattern(str(converted_program._mlir_module), check_file=truth)

    @pytest.mark.parametrize("scale", [None, 1.5])
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("window_size", [0, 1, 2, 11])
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_mha(  # noqa: PLR0913
        scale: float,
        is_causal: bool,
        window_size: int,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Validate the Multi-Head Attention (MHA) use case of sdpa composite op."""
        batch_size = 3
        num_heads = 4
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_heads, seq_len, head_v_dim), dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sdpa = SDPA(
                    scale=scale, is_causal=is_causal, window_size=window_size
                )

            def forward(
                self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                return self.sdpa(query, key, value)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            seq_dim = torch.export.Dim("seq_len", min=1, max=2048)
            export_dynamic_shapes = {
                "query": {0: batch_dim, 2: q_dim},
                "key": {0: batch_dim, 2: seq_dim},
                "value": {0: batch_dim, 2: seq_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(query, key, value), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchSDPAConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        output_torch_eager = model(query, key, value)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            query=query,
            key=key,
            value=value,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_mha_with_mask_ir(dynamic: bool, dtype: torch.dtype) -> None:
        """Validate that attn_mask shows up in the arguments if specified (IR)."""
        batch_size = 3
        num_heads = 4
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_heads, seq_len, head_v_dim), dtype=dtype)

        full_trues = torch.ones((q_len, seq_len), dtype=torch.bool)
        causal_mask = full_trues.tril(diagonal=seq_len - q_len)
        left_out_of_window = full_trues.tril(diagonal=seq_len - q_len - 7)
        attn_mask = torch.logical_xor(causal_mask, left_out_of_window)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sdpa = SDPA()

            def forward(
                self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: torch.Tensor,
            ) -> torch.Tensor:
                return self.sdpa(query, key, value, attn_mask=attn_mask)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            seq_dim = torch.export.Dim("seq_len", min=1, max=2048)
            export_dynamic_shapes = {
                "query": {0: batch_dim, 2: q_dim},
                "key": {0: batch_dim, 2: seq_dim},
                "value": {0: batch_dim, 2: seq_dim},
                "attn_mask": {0: q_dim, 1: seq_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(query, key, value, attn_mask),
                    dynamic_shapes=export_dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchSDPAConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        b = "?" if dynamic else batch_size
        q = "?" if dynamic else q_len
        s = "?" if dynamic else seq_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @sdpa_[[S:.*]](%arg0: tensor<{b}x{num_heads}x{q}x{head_qk_dim}x{dt}>
        // CHECK-SAME: %arg1: tensor<{b}x{num_heads}x{s}x{head_qk_dim}x{dt}>
        // CHECK-SAME: %arg2: tensor<{b}x{num_heads}x{s}x{head_v_dim}x{dt}>
        // CHECK-SAME: %arg3: tensor<{q}x{s}xi1>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention" =
        // CHECK-SAME: input_names = ["query", "key", "value", "attn_mask"]
        // CHECK-SAME: op_attrs =
        // CHECK-SAME: is_causal = false
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: window_size = 0 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @sdpa_[[S]](
        """
        filecheck_pattern(str(converted_program._mlir_module), check_file=truth)

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_mha_with_mask(dynamic: bool, dtype: torch.dtype) -> None:
        """Validate that attn_mask shows up in the arguments if specified."""
        batch_size = 3
        num_heads = 4
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_heads, seq_len, head_v_dim), dtype=dtype)

        full_trues = torch.ones((q_len, seq_len), dtype=torch.bool)
        causal_mask = full_trues.tril(diagonal=seq_len - q_len)
        left_out_of_window = full_trues.tril(diagonal=seq_len - q_len - 7)
        attn_mask = torch.logical_xor(causal_mask, left_out_of_window)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sdpa = SDPA()

            def forward(
                self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: torch.Tensor,
            ) -> torch.Tensor:
                return self.sdpa(query, key, value, attn_mask=attn_mask)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            seq_dim = torch.export.Dim("seq_len", min=1, max=2048)
            export_dynamic_shapes = {
                "query": {0: batch_dim, 2: q_dim},
                "key": {0: batch_dim, 2: seq_dim},
                "value": {0: batch_dim, 2: seq_dim},
                "attn_mask": {0: q_dim, 1: seq_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(query, key, value, attn_mask),
                    dynamic_shapes=export_dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchSDPAConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        output_torch_eager = model(query, key, value, attn_mask)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("scale", [None, 1.5])
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("window_size", [0, 1, 2, 11])
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_gqa_ir(  # noqa: PLR0913
        scale: float,
        is_causal: bool,
        window_size: int,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Validate the Grouped Query Attention (GQA) use case of sdpa composite op IR."""
        batch_size = 3
        num_heads = 4
        num_kv_heads = 2
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_kv_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_kv_heads, seq_len, head_v_dim), dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sdpa = SDPA(
                    scale=scale, is_causal=is_causal, window_size=window_size
                )

            def forward(
                self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                return self.sdpa(query, key, value)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            seq_dim = torch.export.Dim("seq_len", min=1, max=2048)
            export_dynamic_shapes = {
                "query": {0: batch_dim, 2: q_dim},
                "key": {0: batch_dim, 2: seq_dim},
                "value": {0: batch_dim, 2: seq_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(query, key, value), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchSDPAConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        b = "?" if dynamic else batch_size
        q = "?" if dynamic else q_len
        s = "?" if dynamic else seq_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @sdpa_[[S:.*]](%arg0: tensor<{b}x{num_heads}x{q}x{head_qk_dim}x{dt}>
        // CHECK-SAME: %arg1: tensor<{b}x{num_kv_heads}x{s}x{head_qk_dim}x{dt}>
        // CHECK-SAME: %arg2: tensor<{b}x{num_kv_heads}x{s}x{head_v_dim}x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention" =
        // CHECK-SAME: input_names = ["query", "key", "value"]
        // CHECK-SAME: op_attrs =
        // CHECK-SAME: is_causal = {str(is_causal).lower()}
        {"// CHECK-SAME: scale = " if scale is not None else ""}
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: window_size = {window_size!r} : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @sdpa_[[S]](
        """
        filecheck_pattern(str(converted_program._mlir_module), check_file=truth)

    @pytest.mark.parametrize("scale", [None, 1.5])
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("window_size", [0, 1, 2, 11])
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_gqa(  # noqa: PLR0913
        scale: float,
        is_causal: bool,
        window_size: int,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Validate the Grouped Query Attention (GQA) use case of sdpa composite op."""
        batch_size = 3
        num_heads = 4
        num_kv_heads = 2
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_kv_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_kv_heads, seq_len, head_v_dim), dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sdpa = SDPA(
                    scale=scale, is_causal=is_causal, window_size=window_size
                )

            def forward(
                self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                return self.sdpa(query, key, value)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            seq_dim = torch.export.Dim("seq_len", min=1, max=2048)
            export_dynamic_shapes = {
                "query": {0: batch_dim, 2: q_dim},
                "key": {0: batch_dim, 2: seq_dim},
                "value": {0: batch_dim, 2: seq_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=(query, key, value), dynamic_shapes=export_dynamic_shapes
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchSDPAConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        output_torch_eager = model(query, key, value)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            query=query,
            key=key,
            value=value,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("scale", [None, 1.5])
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("window_size", [0, 1, 2, 11])
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_gqa_with_sinks_ir(  # noqa: PLR0913
        scale: float,
        is_causal: bool,
        window_size: int,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Validate that sinks shows up in the arguments if specified (IR)."""
        batch_size = 3
        num_heads = 4
        num_kv_heads = 2
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_kv_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_kv_heads, seq_len, head_v_dim), dtype=dtype)
        sinks = torch.zeros((num_heads,), dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sdpa = SDPA(
                    scale=scale, is_causal=is_causal, window_size=window_size
                )

            def forward(
                self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                sinks: torch.Tensor,
            ) -> torch.Tensor:
                return self.sdpa(query, key, value, sinks=sinks)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            seq_dim = torch.export.Dim("seq_len", min=1, max=2048)
            export_dynamic_shapes = {
                "query": {0: batch_dim, 2: q_dim},
                "key": {0: batch_dim, 2: seq_dim},
                "value": {0: batch_dim, 2: seq_dim},
                "sinks": {},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(query, key, value, sinks),
                    dynamic_shapes=export_dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchSDPAConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        b = "?" if dynamic else batch_size
        q = "?" if dynamic else q_len
        s = "?" if dynamic else seq_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @sdpa_[[S:.*]](%arg0: tensor<{b}x{num_heads}x{q}x{head_qk_dim}x{dt}>
        // CHECK-SAME: %arg1: tensor<{b}x{num_kv_heads}x{s}x{head_qk_dim}x{dt}>
        // CHECK-SAME: %arg2: tensor<{b}x{num_kv_heads}x{s}x{head_v_dim}x{dt}>
        // CHECK-SAME: %arg3: tensor<{num_heads}x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention" =
        // CHECK-SAME: input_names = ["query", "key", "value", "sinks"]
        // CHECK-SAME: op_attrs =
        // CHECK-SAME: is_causal = {str(is_causal).lower()}
        {"// CHECK-SAME: scale = " if scale is not None else ""}
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: window_size = {window_size!r} : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @sdpa_[[S]](
        """
        filecheck_pattern(str(converted_program._mlir_module), check_file=truth)

    @pytest.mark.parametrize("scale", [None, 1.5])
    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("window_size", [0, 1, 2, 11])
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_gqa_with_sinks(  # noqa: PLR0913
        scale: float,
        is_causal: bool,
        window_size: int,
        dynamic: bool,
        dtype: torch.dtype,
    ) -> None:
        """Validate that sinks shows up in the arguments if specified."""
        batch_size = 3
        num_heads = 4
        num_kv_heads = 2
        q_len = 5
        seq_len = 7
        head_qk_dim = 11
        head_v_dim = 13
        query = torch.rand((batch_size, num_heads, q_len, head_qk_dim), dtype=dtype)
        key = torch.rand((batch_size, num_kv_heads, seq_len, head_qk_dim), dtype=dtype)
        value = torch.rand((batch_size, num_kv_heads, seq_len, head_v_dim), dtype=dtype)
        sinks = torch.zeros((num_heads,), dtype=dtype)

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sdpa = SDPA(
                    scale=scale, is_causal=is_causal, window_size=window_size
                )

            def forward(
                self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                sinks: torch.Tensor,
            ) -> torch.Tensor:
                return self.sdpa(query, key, value, sinks=sinks)

        model = WrapperModel().eval()

        export_dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_size", min=1, max=32)
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            seq_dim = torch.export.Dim("seq_len", min=1, max=2048)
            export_dynamic_shapes = {
                "query": {0: batch_dim, 2: q_dim},
                "key": {0: batch_dim, 2: seq_dim},
                "value": {0: batch_dim, 2: seq_dim},
                "sinks": {},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(query, key, value, sinks),
                    dynamic_shapes=export_dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[TestTorchSDPAConversion._make_externalize_spec()],
            )
            .to_coreai()
        )

        output_torch_eager = model(query, key, value, sinks)
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            rtol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            atol={torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
                dtype
            ],
            query=query,
            key=key,
            value=value,
            sinks=sinks,
        )
