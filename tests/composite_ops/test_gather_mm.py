# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for gather_mm composite op."""

import platform

import numpy as np
import pytest
import torch

if platform.system() == "Darwin":
    # disable mypy checking so it would not complain on linux
    # also need to calm mypy on MacOS when it sees no error
    import mlx  # type: ignore[import-not-found, unused-ignore]
    import mlx.core  # type: ignore[import-not-found, unused-ignore]

from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
from coreai_torch.composite_ops import GatherMM

from ..utils import (
    _mlx_array_to_numpy_array,
    _torch_tensor_to_numpy_array,
    filecheck_pattern,
    validate_numerical_output,
)


class TestTorchGatherMM:
    """Test that torch implementation of gather_mm composite op numerics match mlx.

    Covers rhs_indices only, lhs_indices only, and both indices.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "batch_size, q_len, num_experts, embed_dim, up_dims, num_active_experts",
        [
            (1, 1, 4, 16, 32, 1),  # minimal
            (1, 6, 8, 64, 128, 2),  # typical
            (2, 8, 8, 32, 64, 2),  # batch > 1
            (1, 6, 8, 64, 128, 4),  # more active experts
            (1, 16, 16, 128, 256, 4),  # larger dims
            (3, 4, 4, 32, 64, 1),  # odd batch size
        ],
    )
    @staticmethod
    def test_gather_mm_in_moe(
        batch_size: int,
        q_len: int,
        num_experts: int,
        embed_dim: int,
        up_dims: int,
        num_active_experts: int,
        dtype: torch.dtype,
    ) -> None:
        """Validate Mixture-of-Experts (rhs_indices only) numerics match mlx."""
        rng = np.random.default_rng()
        x = torch.rand((batch_size, q_len, 1, 1, embed_dim), dtype=dtype)
        weight_transpose = torch.rand((num_experts, embed_dim, up_dims), dtype=dtype)
        rhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        gather_mm = GatherMM(num_batch_axes=0)
        output_torch = gather_mm(x, weight_transpose, rhs_indices=rhs_indices)

        x_mlx = mlx.core.array(x)
        weight_transpose_mlx = mlx.core.array(weight_transpose)
        rhs_indices_mlx = mlx.core.array(rhs_indices)
        output_mlx = mlx.core.gather_mm(
            x_mlx,
            weight_transpose_mlx,
            lhs_indices=None,  # type: ignore[arg-type,unused-ignore]
            rhs_indices=rhs_indices_mlx,  # type: ignore[arg-type,unused-ignore]
        )

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch),
            _mlx_array_to_numpy_array(output_mlx),
            atol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            rtol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "num_experts, M, embed_dim, up_dims, batch_size, q_len, num_active_experts",
        [
            (4, 1, 16, 32, 1, 1, 1),  # minimal, M=1
            (8, 1, 64, 128, 1, 6, 2),  # typical, M=1
            (8, 2, 64, 128, 1, 6, 2),  # M=2
            (8, 4, 32, 64, 2, 4, 3),  # M=4, batch > 1
            (16, 1, 128, 256, 1, 8, 4),  # larger dims
            (4, 3, 32, 64, 3, 5, 2),  # odd M and batch
        ],
    )
    @staticmethod
    def test_gather_mm_with_lhs_indices(
        num_experts: int,
        M: int,
        embed_dim: int,
        up_dims: int,
        batch_size: int,
        q_len: int,
        num_active_experts: int,
        dtype: torch.dtype,
    ) -> None:
        """Validate lhs_indices gather numerics match mlx."""
        rng = np.random.default_rng()
        x = torch.rand((num_experts, M, embed_dim), dtype=dtype)
        weight = torch.rand((embed_dim, up_dims), dtype=dtype)
        lhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        gather_mm = GatherMM(num_batch_axes=0)
        output_torch = gather_mm(x, weight, lhs_indices=lhs_indices)

        x_mlx = mlx.core.array(x)
        weight_mlx = mlx.core.array(weight)
        lhs_indices_mlx = mlx.core.array(lhs_indices)
        output_mlx = mlx.core.gather_mm(
            x_mlx,
            weight_mlx,
            lhs_indices=lhs_indices_mlx,  # type: ignore[arg-type,unused-ignore]
            rhs_indices=None,  # type: ignore[arg-type,unused-ignore]
        )

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch),
            _mlx_array_to_numpy_array(output_mlx),
            atol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            rtol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "num_experts, M, embed_dim, up_dims, batch_size, q_len, num_active_experts",
        [
            (4, 1, 16, 32, 1, 1, 1),  # minimal, M=1
            (8, 1, 64, 128, 1, 6, 2),  # typical, M=1
            (8, 2, 64, 128, 1, 6, 2),  # M=2
            (8, 4, 32, 64, 2, 4, 3),  # M=4, batch > 1
            (16, 1, 128, 256, 1, 8, 4),  # larger dims
            (4, 3, 32, 64, 3, 5, 2),  # odd M and batch
        ],
    )
    @staticmethod
    def test_gather_mm_with_both_indices(
        num_experts: int,
        M: int,
        embed_dim: int,
        up_dims: int,
        batch_size: int,
        q_len: int,
        num_active_experts: int,
        dtype: torch.dtype,
    ) -> None:
        """Validate both lhs_indices and rhs_indices numerics match mlx."""
        rng = np.random.default_rng()
        x = torch.rand((num_experts, M, embed_dim), dtype=dtype)
        weight = torch.rand((num_experts, embed_dim, up_dims), dtype=dtype)
        lhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )
        rhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        gather_mm = GatherMM(num_batch_axes=0)
        output_torch = gather_mm(
            x,
            weight,
            lhs_indices=lhs_indices,
            rhs_indices=rhs_indices,
        )

        x_mlx = mlx.core.array(x)
        weight_mlx = mlx.core.array(weight)
        lhs_indices_mlx = mlx.core.array(lhs_indices)
        rhs_indices_mlx = mlx.core.array(rhs_indices)
        output_mlx = mlx.core.gather_mm(
            x_mlx,
            weight_mlx,
            lhs_indices=lhs_indices_mlx,  # type: ignore[arg-type,unused-ignore]
            rhs_indices=rhs_indices_mlx,  # type: ignore[arg-type,unused-ignore]
        )

        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch),
            _mlx_array_to_numpy_array(output_mlx),
            atol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            rtol={torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
        )


class TestTorchGatherMMConversion:
    """Tests for gather_mm composite op conversion using ExternalizeSpec.

    Each test corresponds to a variant of the forward call.  All tests are
    parametrised over dynamic shapes and dtypes to provide comprehensive coverage.
    """

    @staticmethod
    def _make_externalize_spec() -> ExternalizeSpec:
        return ExternalizeSpec(
            target_class=GatherMM,
            composite_op_name="gather_mm",
            composite_attrs=["num_batch_axes"],
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_gather_mm_in_moe_ir(dynamic: bool, dtype: torch.dtype) -> None:
        """Validate the Mixture-of-Experts (MoE) use case of gather_mm composite op IR."""
        batch_size = 1  # TODO: Test batch size > 1 if found use case
        q_len = 6
        embed_dim = 64
        up_dims = 128
        num_experts = 8
        num_active_experts = 2
        rng = np.random.default_rng()
        x = torch.rand((batch_size, q_len, 1, 1, embed_dim), dtype=dtype)
        weight_transpose = torch.rand((num_experts, embed_dim, up_dims), dtype=dtype)
        rhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gather_mm = GatherMM(num_batch_axes=0)

            def forward(
                self,
                lhs: torch.Tensor,
                rhs: torch.Tensor,
                rhs_indices: torch.Tensor,
            ) -> torch.Tensor:
                return self.gather_mm(lhs, rhs, rhs_indices=rhs_indices)

        model = WrapperModel().eval()

        dynamic_shapes = None
        if dynamic:
            # TODO: Test dynamic batch size if found use case
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "lhs": {1: q_len_dim},
                "rhs": {},
                "rhs_indices": {1: q_len_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(x, weight_transpose, rhs_indices),
                    dynamic_shapes=dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchGatherMMConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        q = "?" if dynamic else q_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @gather_mm_[[S:.*]](%arg0: tensor<{batch_size}x{q}x1x1x{embed_dim}x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"gather_mm" =
        // CHECK-SAME: input_names = ["lhs", "rhs", "rhs_indices"]
        // CHECK-SAME: num_batch_axes = 0 : si64
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @gather_mm_[[S]]
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_gather_mm_in_moe(dynamic: bool, dtype: torch.dtype) -> None:
        """Validate the Mixture-of-Experts (MoE) use case of gather_mm composite op."""
        batch_size = 1  # TODO: Test batch size > 1 if found use case
        q_len = 6
        embed_dim = 64
        up_dims = 128
        num_experts = 8
        num_active_experts = 2
        rng = np.random.default_rng()
        x = torch.rand((batch_size, q_len, 1, 1, embed_dim), dtype=dtype)
        weight_transpose = torch.rand((num_experts, embed_dim, up_dims), dtype=dtype)
        rhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gather_mm = GatherMM(num_batch_axes=0)

            def forward(
                self,
                lhs: torch.Tensor,
                rhs: torch.Tensor,
                rhs_indices: torch.Tensor,
            ) -> torch.Tensor:
                return self.gather_mm(lhs, rhs, rhs_indices=rhs_indices)

        model = WrapperModel().eval()

        output_torch_eager = model(x, weight_transpose, rhs_indices)

        dynamic_shapes = None
        if dynamic:
            # TODO: Test dynamic batch size if found use case
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "lhs": {1: q_len_dim},
                "rhs": {},
                "rhs_indices": {1: q_len_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(x, weight_transpose, rhs_indices),
                    dynamic_shapes=dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchGatherMMConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            atol={torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            rtol={torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            lhs=x,
            rhs=weight_transpose,
            rhs_indices=rhs_indices,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_gather_mm_with_lhs_indices_ir(dynamic: bool, dtype: torch.dtype) -> None:
        """Validate gather_mm with lhs_indices to gather expert activations (IR)."""
        num_experts = 8
        embed_dim = 64
        up_dims = 128
        batch_size = 1
        q_len = 6
        num_active_experts = 2
        rng = np.random.default_rng()
        x = torch.rand((num_experts, 1, embed_dim), dtype=dtype)
        weight = torch.rand((embed_dim, up_dims), dtype=dtype)
        lhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gather_mm = GatherMM(num_batch_axes=0)

            def forward(
                self,
                lhs: torch.Tensor,
                rhs: torch.Tensor,
                lhs_indices: torch.Tensor,
            ) -> torch.Tensor:
                return self.gather_mm(lhs, rhs, lhs_indices=lhs_indices)

        model = WrapperModel().eval()

        dynamic_shapes = None
        if dynamic:
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "lhs": {},
                "rhs": {},
                "lhs_indices": {1: q_len_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(x, weight, lhs_indices),
                    dynamic_shapes=dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchGatherMMConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @gather_mm_[[S:.*]](%arg0: tensor<{num_experts}x1x{embed_dim}x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"gather_mm" = {{input_names = ["lhs", "rhs", "lhs_indices"], op_attrs = {{num_batch_axes = 0 : si64, version = 1 : si64}}, output_names = ["output"]}}>
        // CHECK: coreai.invoke @gather_mm_[[S]]
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_gather_mm_with_lhs_indices(
        dynamic: bool, dtype: torch.dtype
    ) -> None:
        """Validate gather_mm with lhs_indices to gather expert activations."""
        num_experts = 8
        embed_dim = 64
        up_dims = 128
        batch_size = 1
        q_len = 6
        num_active_experts = 2
        rng = np.random.default_rng()
        x = torch.rand((num_experts, 1, embed_dim), dtype=dtype)
        weight = torch.rand((embed_dim, up_dims), dtype=dtype)
        lhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gather_mm = GatherMM(num_batch_axes=0)

            def forward(
                self,
                lhs: torch.Tensor,
                rhs: torch.Tensor,
                lhs_indices: torch.Tensor,
            ) -> torch.Tensor:
                return self.gather_mm(lhs, rhs, lhs_indices=lhs_indices)

        model = WrapperModel().eval()

        output_torch_eager = model(x, weight, lhs_indices)

        dynamic_shapes = None
        if dynamic:
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "lhs": {},
                "rhs": {},
                "lhs_indices": {1: q_len_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(x, weight, lhs_indices),
                    dynamic_shapes=dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchGatherMMConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            atol={torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            rtol={torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            lhs=x,
            rhs=weight,
            lhs_indices=lhs_indices,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_gather_mm_with_both_indices_ir(dynamic: bool, dtype: torch.dtype) -> None:
        """Validate gather_mm with both lhs_indices and rhs_indices (IR)."""
        num_experts = 8
        embed_dim = 64
        up_dims = 128
        batch_size = 1
        q_len = 6
        num_active_experts = 2
        rng = np.random.default_rng()
        x = torch.rand((num_experts, 1, embed_dim), dtype=dtype)
        weight = torch.rand((num_experts, embed_dim, up_dims), dtype=dtype)
        lhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )
        rhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gather_mm = GatherMM(num_batch_axes=0)

            def forward(
                self,
                lhs: torch.Tensor,
                rhs: torch.Tensor,
                lhs_indices: torch.Tensor,
                rhs_indices: torch.Tensor,
            ) -> torch.Tensor:
                return self.gather_mm(
                    lhs, rhs, lhs_indices=lhs_indices, rhs_indices=rhs_indices
                )

        model = WrapperModel().eval()

        dynamic_shapes = None
        if dynamic:
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "lhs": {},
                "rhs": {},
                "lhs_indices": {1: q_len_dim},
                "rhs_indices": {1: q_len_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(x, weight, lhs_indices, rhs_indices),
                    dynamic_shapes=dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchGatherMMConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @gather_mm_[[S:.*]](%arg0: tensor<{num_experts}x1x{embed_dim}x{dt}>
        // CHECK: composite_decl = #coreai.composite_declaration<"gather_mm" = {{input_names = ["lhs", "rhs", "lhs_indices", "rhs_indices"], op_attrs = {{num_batch_axes = 0 : si64, version = 1 : si64}}, output_names = ["output"]}}>
        // CHECK: coreai.invoke @gather_mm_[[S]]
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_gather_mm_with_both_indices(
        dynamic: bool, dtype: torch.dtype
    ) -> None:
        """Validate gather_mm with both lhs_indices and rhs_indices."""
        num_experts = 8
        embed_dim = 64
        up_dims = 128
        batch_size = 1
        q_len = 6
        num_active_experts = 2
        rng = np.random.default_rng()
        x = torch.rand((num_experts, 1, embed_dim), dtype=dtype)
        weight = torch.rand((num_experts, embed_dim, up_dims), dtype=dtype)
        lhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )
        rhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gather_mm = GatherMM(num_batch_axes=0)

            def forward(
                self,
                lhs: torch.Tensor,
                rhs: torch.Tensor,
                lhs_indices: torch.Tensor,
                rhs_indices: torch.Tensor,
            ) -> torch.Tensor:
                return self.gather_mm(
                    lhs, rhs, lhs_indices=lhs_indices, rhs_indices=rhs_indices
                )

        model = WrapperModel().eval()

        output_torch_eager = model(x, weight, lhs_indices, rhs_indices)

        dynamic_shapes = None
        if dynamic:
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "lhs": {},
                "rhs": {},
                "lhs_indices": {1: q_len_dim},
                "rhs_indices": {1: q_len_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(x, weight, lhs_indices, rhs_indices),
                    dynamic_shapes=dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchGatherMMConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            atol={torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            rtol={torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            lhs=x,
            rhs=weight,
            lhs_indices=lhs_indices,
            rhs_indices=rhs_indices,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    def test_gather_mm_in_moe_with_fused_proj_ir(
        dynamic: bool, dtype: torch.dtype
    ) -> None:
        """Validate the MoE with fused gate and up linear projections use case (IR).

        This is beyond MLX semantics so we compare our own fused vs separated.
        """
        batch_size = 1  # TODO: Test batch size > 1 if found use case
        q_len = 6
        embed_dim = 64
        up_dims = 128
        num_experts = 8
        num_active_experts = 2
        rng = np.random.default_rng()
        x = torch.rand((batch_size, q_len, 1, 1, embed_dim), dtype=dtype)
        weight_transpose0 = torch.rand((num_experts, embed_dim, up_dims), dtype=dtype)
        weight_transpose1 = torch.rand((num_experts, embed_dim, up_dims), dtype=dtype)
        rhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gather_mm = GatherMM(num_batch_axes=0)
                self.gather_mm_fused = GatherMM(num_batch_axes=1)

            def forward(
                self,
                lhs: torch.Tensor,
                rhs0: torch.Tensor,
                rhs1: torch.Tensor,
                rhs_indices: torch.Tensor,
            ) -> torch.Tensor:
                y0 = self.gather_mm(lhs, rhs0, rhs_indices=rhs_indices)
                y1 = self.gather_mm(lhs, rhs1, rhs_indices=rhs_indices)
                y = torch.stack((y0, y1))

                fused_rhs = torch.stack((rhs0, rhs1))
                fused_y = self.gather_mm_fused(lhs, fused_rhs, rhs_indices=rhs_indices)

                return y - fused_y

        model = WrapperModel().eval()

        dynamic_shapes = None
        if dynamic:
            # TODO: Test dynamic batch size if found use case
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "lhs": {1: q_len_dim},
                "rhs0": {},
                "rhs1": {},
                "rhs_indices": {1: q_len_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(x, weight_transpose0, weight_transpose1, rhs_indices),
                    dynamic_shapes=dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchGatherMMConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        q = "?" if dynamic else q_len
        dt = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
        truth = f"""
        // CHECK: coreai.graph private noinline @gather_mm_[[S0:.*]](%arg0: tensor<{batch_size}x{q}x1x1x{embed_dim}x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"gather_mm" =
        // CHECK-SAME: input_names = ["lhs", "rhs", "rhs_indices"]
        // CHECK-SAME: num_batch_axes = 0 : si64
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.graph private noinline @gather_mm_fused_[[S1:.*]](%arg0: tensor<{batch_size}x{q}x1x1x{embed_dim}x{dt}>
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"gather_mm" =
        // CHECK-SAME: input_names = ["lhs", "rhs", "rhs_indices"]
        // CHECK-SAME: num_batch_axes = 1 : si64
        // CHECK-SAME: version = 1 : si64
        // CHECK-SAME: output_names = ["output"]
        // CHECK: coreai.invoke @gather_mm_[[S0]]
        // CHECK: coreai.invoke @gather_mm_fused_[[S1]]
        """
        filecheck_pattern(str(converted_program._module._mlir_module), check_file=truth)

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @staticmethod
    async def test_gather_mm_in_moe_with_fused_proj(
        dynamic: bool, dtype: torch.dtype
    ) -> None:
        """Validate the MoE with fused gate and up linear projections use case.

        This is beyond MLX semantics so we compare our own fused vs separated.
        """
        batch_size = 1  # TODO: Test batch size > 1 if found use case
        q_len = 6
        embed_dim = 64
        up_dims = 128
        num_experts = 8
        num_active_experts = 2
        rng = np.random.default_rng()
        x = torch.rand((batch_size, q_len, 1, 1, embed_dim), dtype=dtype)
        weight_transpose0 = torch.rand((num_experts, embed_dim, up_dims), dtype=dtype)
        weight_transpose1 = torch.rand((num_experts, embed_dim, up_dims), dtype=dtype)
        rhs_indices = torch.tensor(
            rng.integers(
                low=0,
                high=num_experts,
                size=(batch_size, q_len, num_active_experts),
                dtype=np.uint16,
            )
        )

        class WrapperModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gather_mm = GatherMM(num_batch_axes=0)
                self.gather_mm_fused = GatherMM(num_batch_axes=1)

            def forward(
                self,
                lhs: torch.Tensor,
                rhs0: torch.Tensor,
                rhs1: torch.Tensor,
                rhs_indices: torch.Tensor,
            ) -> torch.Tensor:
                y0 = self.gather_mm(lhs, rhs0, rhs_indices=rhs_indices)
                y1 = self.gather_mm(lhs, rhs1, rhs_indices=rhs_indices)
                y = torch.stack((y0, y1))

                fused_rhs = torch.stack((rhs0, rhs1))
                fused_y = self.gather_mm_fused(lhs, fused_rhs, rhs_indices=rhs_indices)

                return y - fused_y

        model = WrapperModel().eval()

        # Numerical validation: unfused and fused approaches must produce identical results
        output_torch_eager = model(x, weight_transpose0, weight_transpose1, rhs_indices)
        np.testing.assert_allclose(
            _torch_tensor_to_numpy_array(output_torch_eager),
            0.0,
            atol={torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            rtol={torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
        )

        dynamic_shapes = None
        if dynamic:
            # TODO: Test dynamic batch size if found use case
            q_len_dim = torch.export.Dim(name="q_len", min=1, max=64)
            dynamic_shapes = {
                "lhs": {1: q_len_dim},
                "rhs0": {},
                "rhs1": {},
                "rhs_indices": {1: q_len_dim},
            }

        converted_program = (
            TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m,
                    args=(x, weight_transpose0, weight_transpose1, rhs_indices),
                    dynamic_shapes=dynamic_shapes,
                ).run_decompositions(get_decomp_table()),
                externalize_modules=[
                    TestTorchGatherMMConversion._make_externalize_spec()
                ],
            )
            .to_coreai()
        )

        await validate_numerical_output(
            coreai_program=converted_program,
            torch_out=output_torch_eager,
            atol={torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            rtol={torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-2}[
                dtype
            ],
            lhs=x,
            rhs0=weight_transpose0,
            rhs1=weight_transpose1,
            rhs_indices=rhs_indices,
        )
