# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test for gated_delta_update composite op."""
# ruff: noqa: F821, F841

import platform

import pytest

if platform.system() != "Darwin":
    pytest.skip(
        "MLX is stable only on MacOS",
        allow_module_level=True,
    )
else:
    # disable mypy checking so it would not complain on linux
    # also need to calm mypy on MacOS when it sees no error
    import mlx.core as mx  # type: ignore[import-not-found, unused-ignore]
    from mlx_lm.models.gated_delta import (
        gated_delta_ops,  # type: ignore[import-not-found, unused-ignore]
    )

import numpy as np
import torch
from torch.testing import assert_close

from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
from coreai_torch.composite_ops import GatedDeltaUpdate

from ..utils import (
    _mlx_array_to_numpy_array,
    _torch_tensor_to_numpy_array,
    filecheck_pattern,
    validate_numerical_output,
)


def _log_decay_gate(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """
    A decay gate, in the log space the op reads it in.

    `GatedDeltaUpdate` exponentiates `g` and multiplies the recurrent state by it
    once per timestep, so `g` has to be negative for the recurrence to contract --
    which is why models produce it through a log-sigmoid. Drawing it from `randn`
    instead lets `exp(g)` reach e**3, growing the state by ~20x per step: over eight
    steps that is a factor of 2.5e10, which leaves fp16 (max 65504) as infinities and
    the values just below it with no significant digits left to agree on.
    """
    return torch.nn.functional.logsigmoid(torch.randn(*shape)).to(dtype)


class TestGatedDeltaUpdate:
    """Tests for the gated_delta_update composite operation."""

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.parametrize(
        ("batch_size", "num_heads", "seq_len", "head_k_dim", "head_v_dim"),
        [
            (1, 2, 3, 8, 8),
            (2, 4, 5, 16, 16),
        ],
    )
    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    def test_gated_delta_update_basic(  # noqa: PLR0913
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_k_dim: int,
        head_v_dim: int,
        precision: torch.dtype,
    ) -> None:
        """Test basic gated_delta_update functionality."""
        # Create input tensors in [B, H, T, ...] layout
        query = torch.randn(batch_size, num_heads, seq_len, head_k_dim, dtype=precision)
        key = torch.randn(batch_size, num_heads, seq_len, head_k_dim, dtype=precision)
        value = torch.randn(batch_size, num_heads, seq_len, head_v_dim, dtype=precision)
        g = _log_decay_gate(batch_size, num_heads, seq_len, dtype=precision)
        beta = torch.randn(batch_size, num_heads, seq_len, dtype=precision)
        initial_state = torch.randn(
            batch_size,
            num_heads,
            head_k_dim,
            head_v_dim,
            dtype=precision,
        )

        # Run the operation
        output, final_state = GatedDeltaUpdate()(
            query,
            key,
            value,
            g,
            beta,
            initial_state,
        )

        # Check output shapes
        assert output.shape == (batch_size, seq_len, num_heads, head_v_dim)
        assert final_state.shape == (batch_size, num_heads, head_k_dim, head_v_dim)

        # Check output dtype matches input
        assert output.dtype == precision
        assert final_state.dtype == precision

        # Compare with MLX reference implementation
        # Note that, the gated_delta_ops in MLX does NOT contains the l2 norm and the scaling,
        # hence we need to do the transformation before passing it to MLX's API.

        def l2norm(x: torch.Tensor) -> torch.Tensor:
            return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)

        # L2 normalize query and key
        query_normalized = l2norm(query.float())
        key_normalized = l2norm(key.float())

        # Apply query scaling, then transpose [B, H, T, ...] -> [B, T, H, ...] for MLX
        query_transformed = (query_normalized * (head_k_dim**-0.5)).transpose(1, 2)
        key_transformed = key_normalized.transpose(1, 2)
        value_transformed = value.float().transpose(1, 2)
        g_transformed = g.float().transpose(1, 2)
        beta_transformed = beta.float().transpose(1, 2)

        # Exponential of g
        g_exp = g_transformed.exp()

        # Convert tensors to MLX arrays
        query_mlx = mx.array(query_transformed.numpy())
        key_mlx = mx.array(key_transformed.numpy())
        value_mlx = mx.array(value_transformed.numpy())
        g_mlx = mx.array(g_exp.numpy())
        beta_mlx = mx.array(beta_transformed.numpy())
        # MLX state shape is [B, Hv, Dv, Dk], Core AI state is [B, H, Dk, Dv]
        # Need to transpose from [B, H, Dk, Dv] to [B, H, Dv, Dk]
        initial_state_mlx = mx.array(
            initial_state.float().numpy().transpose(0, 1, 3, 2),
        )

        # Run MLX reference implementation
        output_mlx, final_state_mlx = gated_delta_ops(
            query_mlx,
            key_mlx,
            value_mlx,
            g_mlx,
            beta_mlx,
            initial_state_mlx,
        )

        # Convert MLX outputs back to numpy and compare
        # Transpose state back from [B, H, Dv, Dk] to [B, H, Dk, Dv]
        final_state_mlx_transposed = final_state_mlx.transpose(0, 1, 3, 2)

        # Set tolerance based on precision
        tolerance = {torch.float32: 1e-4, torch.float16: 5e-2}[precision]

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output),
            _mlx_array_to_numpy_array(output_mlx),
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(final_state),
            _mlx_array_to_numpy_array(final_state_mlx_transposed),
            rtol=tolerance,
            atol=tolerance,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_gated_delta_update_torch_export_ir(
        self,
        precision: torch.dtype,
        dynamic: bool,
    ) -> None:
        """Test that gated_delta_update produces correct composite op IR."""
        batch_size = 1
        num_heads = 2
        seq_len = 2
        head_k_dim = 8
        head_v_dim = 8

        class GatedDeltaUpdateModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gated_delta_update = GatedDeltaUpdate()

            def forward(  # noqa: PLR0913
                self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                initial_state: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return self.gated_delta_update(
                    query, key, value, g, beta, initial_state
                )

        model = GatedDeltaUpdateModel().eval()

        # Create sample inputs in [B, H, T, ...] layout
        query = torch.randn(batch_size, num_heads, seq_len, head_k_dim, dtype=precision)
        key = torch.randn(batch_size, num_heads, seq_len, head_k_dim, dtype=precision)
        value = torch.randn(batch_size, num_heads, seq_len, head_v_dim, dtype=precision)
        g = _log_decay_gate(batch_size, num_heads, seq_len, dtype=precision)
        beta = torch.randn(batch_size, num_heads, seq_len, dtype=precision)
        initial_state = torch.randn(
            batch_size,
            num_heads,
            head_k_dim,
            head_v_dim,
            dtype=precision,
        )

        # Build dynamic shapes for the export_fn if needed
        dynamic_shapes = None
        if dynamic:
            seq_len_dim = torch.export.Dim(name="seq_len", min=1, max=64)
            dynamic_shapes = {
                "query": {2: seq_len_dim},
                "key": {2: seq_len_dim},
                "value": {2: seq_len_dim},
                "g": {2: seq_len_dim},
                "beta": {2: seq_len_dim},
                "initial_state": None,
            }

        # Verify externalization produces a composite op declaration in the IR
        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    (query, key, value, g, beta, initial_state),
                    dynamic_shapes=dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    ExternalizeSpec(
                        target_class=GatedDeltaUpdate,
                        composite_op_name="gated_delta_update",
                        composite_attrs=["use_qk_l2_norm"],
                    )
                ],
            )
            .to_coreai()
        )

        s = "?" if dynamic else seq_len
        dt = {torch.float32: "f32", torch.float16: "f16"}[precision]
        truth = f"""
        // CHECK: coreai.graph private noinline @gated_delta_update_[[S:.*]](%arg0: tensor<{batch_size}x{num_heads}x{s}x{head_k_dim}x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"gated_delta_update" = {{input_names = ["query", "key", "value", "g", "beta", "initial_state"], op_attrs = {{use_qk_l2_norm = true, version = 1 : si64}}, output_names = ["output_0", "output_1"]}}>

        // CHECK: coreai.invoke @gated_delta_update_[[S]]
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)

    @pytest.mark.control_flow
    @pytest.mark.flaky(reruns=3)
    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    @pytest.mark.parametrize("seq_len", [1, 8, "dynamic"])
    @pytest.mark.parametrize("num_heads", [8, 16])
    @pytest.mark.parametrize("head_k_dim", [8, 16])
    @pytest.mark.parametrize("head_v_dim", [8, 16])
    @pytest.mark.parametrize("use_qk_l2_norm", [True, False])
    async def test_gated_delta_update_torch_export(
        self,
        precision: torch.dtype,
        seq_len: int | str,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        use_qk_l2_norm: bool,
    ) -> None:
        """Test that gated_delta_update can be exported and externalized as a composite op."""
        batch_size = 1
        runtime_seq_len = 8 if seq_len == "dynamic" else seq_len

        class GatedDeltaUpdateModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gated_delta_update = GatedDeltaUpdate(use_qk_l2_norm)

            def forward(  # noqa: PLR0913
                self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                initial_state: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return self.gated_delta_update(
                    query, key, value, g, beta, initial_state
                )

        model = GatedDeltaUpdateModel().eval()

        # Create sample inputs in [B, H, T, ...] layout
        query = torch.randn(
            batch_size, num_heads, runtime_seq_len, head_k_dim, dtype=precision
        )
        key = torch.randn(
            batch_size, num_heads, runtime_seq_len, head_k_dim, dtype=precision
        )
        value = torch.randn(
            batch_size, num_heads, runtime_seq_len, head_v_dim, dtype=precision
        )
        g = _log_decay_gate(batch_size, num_heads, runtime_seq_len, dtype=precision)
        beta = torch.randn(batch_size, num_heads, runtime_seq_len, dtype=precision)
        initial_state = torch.randn(
            batch_size,
            num_heads,
            head_k_dim,
            head_v_dim,
            dtype=precision,
        )

        # Build dynamic shapes for the export_fn if needed
        dynamic_shapes = None
        if seq_len == "dynamic":
            seq_len_dim = torch.export.Dim(name="seq_len", min=1, max=64)
            dynamic_shapes = {
                "query": {2: seq_len_dim},
                "key": {2: seq_len_dim},
                "value": {2: seq_len_dim},
                "g": {2: seq_len_dim},
                "beta": {2: seq_len_dim},
                "initial_state": None,
            }

        # Verify torch.export correctness
        exported_program = torch.export.export(
            model,
            (query, key, value, g, beta, initial_state),
            dynamic_shapes=dynamic_shapes,
        )
        output_exported, state_exported = exported_program.module()(
            query,
            key,
            value,
            g,
            beta,
            initial_state,
        )
        output_original, state_original = model(
            query,
            key,
            value,
            g,
            beta,
            initial_state,
        )
        export_tol = {torch.float32: 1e-3, torch.float16: 5e-2}[precision]
        assert_close(output_exported, output_original, atol=export_tol, rtol=export_tol)
        assert_close(state_exported, state_original, atol=export_tol, rtol=export_tol)

        # Verify externalization produces a composite op declaration in the IR
        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    (query, key, value, g, beta, initial_state),
                    dynamic_shapes=dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    ExternalizeSpec(
                        target_class=GatedDeltaUpdate,
                        composite_op_name="gated_delta_update",
                        composite_attrs=["use_qk_l2_norm"],
                    )
                ],
            )
            .to_coreai()
        )

        # Core AI runtime validation
        tolerance = {torch.float32: 1e-3, torch.float16: 5e-2}[precision]
        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=(output_original, state_original),
            rtol=tolerance,
            atol=tolerance,
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
        )
