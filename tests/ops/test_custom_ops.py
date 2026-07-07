# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for coreai compression custom op lowering to Core AI.

Each test creates a minimal module that invokes one compression op, exports
it via torch.export, imports it with TorchConverter, verifies that the IR
output contains the expected Core AI op mnemonic via filecheck, and validates
numerical correctness against the PyTorch eager implementation.
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# Importing custom_layers registers all coreai:: ops.
import coreai_torch._compression.custom_layers as _  # noqa: F401
from coreai_torch._compression.utils import inject_subbyte_tensors

from ..utils import (
    TorchConverter,
    _all_dims_dynamic,
    filecheck_pattern,
    validate_numerical_output,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _export_and_import(model: nn.Module, *args: Tensor) -> str:
    """Export model, import with TorchConverter, return IR string."""
    model.eval()
    exported = torch.export.export(model, args=args).run_decompositions()
    exported = inject_subbyte_tensors(exported)
    converter = TorchConverter().add_exported_program(exported)
    coreai_program = converter.to_coreai()
    return str(coreai_program)


# ---------------------------------------------------------------------------
# lut_to_dense → coreai.lut_to_dense
# ---------------------------------------------------------------------------


class TestLutToDense:
    """Tests for lut_to_dense → coreai.lut_to_dense lowering."""

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "indices_shape,num_palettes",
        [
            ((2, 3), 4),  # 2D indices, 2-bit palette  (nbits=2)
            ((4, 8), 16),  # 2D indices, 4-bit palette  (nbits=4)
            ((3, 5, 2), 64),  # 3D indices, 6-bit palette (nbits=6)
        ],
    )
    def test_ir(self, indices_shape: tuple[int, ...], num_palettes: int) -> None:
        """lut_to_dense is lowered to coreai.lut_to_dense with a si16 axis."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "indices",
                    torch.randint(0, num_palettes, indices_shape, dtype=torch.uint8),
                )
                lut_leading = (1,) * len(indices_shape)
                self.register_buffer("lut", torch.randn(*lut_leading, num_palettes, 1))

            def forward(self) -> Tensor:
                return torch.ops.coreai.lut_to_dense(self.indices, self.lut, 0)

        model = Model()
        ir = _export_and_import(model)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK: coreai.lut_to_dense
                // CHECK-NOT: TorchImport
            """,
        )

    @pytest.mark.parametrize(
        "indices_shape,num_palettes",
        [
            ((2, 3), 4),  # 2D indices, 2-bit palette  (nbits=2)
            ((4, 8), 16),  # 2D indices, 4-bit palette  (nbits=4)
            ((3, 5, 2), 64),  # 3D indices, 6-bit palette (nbits=6)
        ],
    )
    async def test_numerical(
        self, indices_shape: tuple[int, ...], num_palettes: int
    ) -> None:
        """lut_to_dense numerical validation against PyTorch eager."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "indices",
                    torch.randint(0, num_palettes, indices_shape, dtype=torch.uint8),
                )
                lut_leading = (1,) * len(indices_shape)
                self.register_buffer("lut", torch.randn(*lut_leading, num_palettes, 1))

            def forward(self) -> Tensor:
                return torch.ops.coreai.lut_to_dense(self.indices, self.lut, 0)

        model = Model()
        await validate_numerical_output(
            model=model, prepare_program=inject_subbyte_tensors
        )


# ---------------------------------------------------------------------------
# blockwise_shift_scale → coreai.blockwise_shift_scale
# ---------------------------------------------------------------------------


class TestBlockwiseShiftScale:
    """Tests for blockwise_shift_scale → coreai.blockwise_shift_scale lowering."""

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "data_shape,scale_shape",
        [
            ((4, 8), (1, 1)),
            ((8, 16), (2, 1)),
            ((6, 4), (1, 4)),
        ],
    )
    def test_symmetric_ir(
        self, data_shape: tuple[int, ...], scale_shape: tuple[int, ...]
    ) -> None:
        """2-arg blockwise_shift_scale is lowered to blockwise_shift_scale with zero offsets."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "data", torch.randint(-8, 8, data_shape, dtype=torch.int8)
                )
                self.register_buffer("scale", torch.randn(*scale_shape))

            def forward(self) -> Tensor:
                return torch.ops.coreai.constexpr_blockwise_shift_scale(
                    self.data, self.scale
                )

        model = Model()
        ir = _export_and_import(model)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK: coreai.blockwise_shift_scale
            """,
        )

    @pytest.mark.parametrize(
        "data_shape,scale_shape",
        [
            ((4, 8), (1, 1)),
            ((8, 16), (2, 1)),
            ((6, 4), (1, 4)),
        ],
    )
    async def test_symmetric_numerical(
        self, data_shape: tuple[int, ...], scale_shape: tuple[int, ...]
    ) -> None:
        """2-arg blockwise_shift_scale numerical validation."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "data", torch.randint(-8, 8, data_shape, dtype=torch.int8)
                )
                self.register_buffer("scale", torch.randn(*scale_shape))

            def forward(self) -> Tensor:
                return torch.ops.coreai.constexpr_blockwise_shift_scale(
                    self.data, self.scale
                )

        model = Model()
        await validate_numerical_output(
            model=model, prepare_program=inject_subbyte_tensors
        )

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "data_shape,scale_shape",
        [
            ((4, 8), (1, 1)),
            ((8, 16), (2, 1)),
            ((6, 4), (1, 4)),
        ],
    )
    def test_asymmetric_ir(
        self, data_shape: tuple[int, ...], scale_shape: tuple[int, ...]
    ) -> None:
        """3-arg blockwise_shift_scale (with zero_point) is lowered to blockwise_shift_scale."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "data", torch.randint(-8, 8, data_shape, dtype=torch.int8)
                )
                self.register_buffer("scale", torch.randn(*scale_shape))
                self.register_buffer(
                    "zero_point", torch.randint(-4, 4, scale_shape, dtype=torch.int8)
                )

            def forward(self) -> Tensor:
                return torch.ops.coreai.constexpr_blockwise_shift_scale(
                    self.data, self.scale, zero_point=self.zero_point
                )

        model = Model()
        ir = _export_and_import(model)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK: coreai.blockwise_shift_scale
            """,
        )

    @pytest.mark.parametrize(
        "data_shape,scale_shape",
        [
            ((4, 8), (1, 1)),
            ((8, 16), (2, 1)),
            ((6, 4), (1, 4)),
        ],
    )
    async def test_asymmetric_numerical(
        self, data_shape: tuple[int, ...], scale_shape: tuple[int, ...]
    ) -> None:
        """3-arg blockwise_shift_scale (with zero_point) numerical validation."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "data", torch.randint(-8, 8, data_shape, dtype=torch.int8)
                )
                self.register_buffer("scale", torch.randn(*scale_shape))
                self.register_buffer(
                    "zero_point", torch.randint(-4, 4, scale_shape, dtype=torch.int8)
                )

            def forward(self) -> Tensor:
                return torch.ops.coreai.constexpr_blockwise_shift_scale(
                    self.data, self.scale, zero_point=self.zero_point
                )

        model = Model()
        await validate_numerical_output(
            model=model, prepare_program=inject_subbyte_tensors
        )

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "data_shape,scale_shape",
        [
            ((4, 8), (1, 1)),
            ((8, 16), (2, 1)),
            ((6, 4), (1, 4)),
        ],
    )
    def test_zp_mode_ir(
        self, data_shape: tuple[int, ...], scale_shape: tuple[int, ...]
    ) -> None:
        """blockwise_shift_scale with zero_point is lowered to coreai.blockwise_shift_scale."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "data", torch.randint(-8, 8, data_shape, dtype=torch.int8)
                )
                self.register_buffer("scale", torch.randn(*scale_shape))
                self.register_buffer(
                    "zero_point", torch.randint(-4, 4, scale_shape, dtype=torch.int8)
                )

            def forward(self) -> Tensor:
                return torch.ops.coreai.constexpr_blockwise_shift_scale(
                    self.data, self.scale, zero_point=self.zero_point
                )

        model = Model()
        ir = _export_and_import(model)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK: coreai.blockwise_shift_scale
            """,
        )

    @pytest.mark.parametrize(
        "data_shape,scale_shape",
        [
            ((4, 8), (1, 1)),
            ((8, 16), (2, 1)),
            ((6, 4), (1, 4)),
        ],
    )
    async def test_zp_mode_numerical(
        self, data_shape: tuple[int, ...], scale_shape: tuple[int, ...]
    ) -> None:
        """blockwise_shift_scale with zero_point numerical validation."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "data", torch.randint(-8, 8, data_shape, dtype=torch.int8)
                )
                self.register_buffer("scale", torch.randn(*scale_shape))
                self.register_buffer(
                    "zero_point", torch.randint(-4, 4, scale_shape, dtype=torch.int8)
                )

            def forward(self) -> Tensor:
                return torch.ops.coreai.constexpr_blockwise_shift_scale(
                    self.data, self.scale, zero_point=self.zero_point
                )

        model = Model()
        await validate_numerical_output(
            model=model, prepare_program=inject_subbyte_tensors
        )


# ---------------------------------------------------------------------------
# quantize → coreai.quantize
# ---------------------------------------------------------------------------


class TestQuantize:
    """Tests for quantize → coreai.quantize lowering."""

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "x",
        [
            torch.randn(4, 8),
            torch.randn(8, 16),
            torch.randn(2, 3, 5),
        ],
    )
    def test_ir(self, x: Tensor) -> None:
        """quantize is lowered to coreai.quantize."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.tensor([0.1], dtype=torch.float32))
                self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int8))

            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.coreai.quantize(
                    x, self.scale, torch.int8, zero_point=self.zero_point, axis=0
                )

        model = Model()
        ir = _export_and_import(model, x)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK: coreai.quantize
            """,
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x",
        [
            torch.randn(4, 8),
            torch.randn(8, 16),
            torch.randn(2, 3, 5),
        ],
    )
    async def test_numerical(self, x: Tensor, dynamic: bool) -> None:
        """quantize numerical validation against PyTorch eager."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.tensor([0.1], dtype=torch.float32))
                self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int8))

            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.coreai.quantize(
                    x, self.scale, torch.int8, zero_point=self.zero_point, axis=0
                )

        model = Model()
        dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
        await validate_numerical_output(
            model=model,
            x=x,
            dynamic_shapes=dynamic_shapes,
            prepare_program=inject_subbyte_tensors,
        )

    # (2, 4, 4): equal dims keep a wrong axis silent (a value mismatch).
    # (2, 3, 4): distinct dims show -1 is the last dim (a wrong axis is a reshape error).
    @pytest.mark.parametrize("x", [torch.randn(2, 4, 4), torch.randn(2, 3, 4)])
    async def test_per_channel_negative_axis_numerical(self, x: Tensor) -> None:
        """quantize with a per-channel scale on a negative axis matches eager."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "scale", torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
                )
                self.register_buffer("zero_point", torch.zeros(4, dtype=torch.int8))

            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.coreai.quantize(
                    x, self.scale, torch.int8, zero_point=self.zero_point, axis=-1
                )

        model = Model()
        await validate_numerical_output(
            model=model, x=x, prepare_program=inject_subbyte_tensors
        )


# ---------------------------------------------------------------------------
# dequantize → coreai.dequantize
# ---------------------------------------------------------------------------


class TestDequantize:
    """Tests for dequantize → coreai.dequantize lowering."""

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "x",
        [
            torch.randint(-128, 127, (4, 8), dtype=torch.int8),
            torch.randint(-128, 127, (8, 16), dtype=torch.int8),
            torch.randint(-128, 127, (2, 3, 5), dtype=torch.int8),
        ],
    )
    def test_ir(self, x: Tensor) -> None:
        """dequantize is lowered to coreai.dequantize."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.tensor([0.1], dtype=torch.float32))
                self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int8))

            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.coreai.dequantize(
                    x, self.scale, zero_point=self.zero_point, axis=0
                )

        model = Model()
        ir = _export_and_import(model, x)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK: coreai.dequantize
            """,
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x",
        [
            torch.randint(-128, 127, (4, 8), dtype=torch.int8),
            torch.randint(-128, 127, (8, 16), dtype=torch.int8),
            torch.randint(-128, 127, (2, 3, 5), dtype=torch.int8),
        ],
    )
    async def test_numerical(self, x: Tensor, dynamic: bool) -> None:
        """dequantize numerical validation against PyTorch eager."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.tensor([0.1], dtype=torch.float32))
                self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int8))

            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.coreai.dequantize(
                    x, self.scale, zero_point=self.zero_point, axis=0
                )

        model = Model()
        dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
        await validate_numerical_output(
            model=model,
            x=x,
            dynamic_shapes=dynamic_shapes,
            prepare_program=inject_subbyte_tensors,
        )

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "x",
        [
            torch.randint(-128, 127, (4, 8), dtype=torch.int8),
            torch.randint(-128, 127, (8, 16), dtype=torch.int8),
        ],
    )
    def test_defaults_ir(self, x: Tensor) -> None:
        """dequantize with only input+scale defaults zero_point and axis to 0."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.tensor([0.1], dtype=torch.float32))

            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.coreai.dequantize(x, self.scale)

        model = Model()
        ir = _export_and_import(model, x)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK: coreai.dequantize
            """,
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x",
        [
            torch.randint(-128, 127, (4, 8), dtype=torch.int8),
            torch.randint(-128, 127, (8, 16), dtype=torch.int8),
        ],
    )
    async def test_defaults_numerical(self, x: Tensor, dynamic: bool) -> None:
        """dequantize with only input+scale defaults numerical validation."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.tensor([0.1], dtype=torch.float32))

            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.coreai.dequantize(x, self.scale)

        model = Model()
        dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
        await validate_numerical_output(
            model=model,
            x=x,
            dynamic_shapes=dynamic_shapes,
            prepare_program=inject_subbyte_tensors,
        )

    # (2, 4, 4): equal dims keep a wrong axis silent (a value mismatch).
    # (2, 3, 4): distinct dims show -1 is the last dim (a wrong axis is a reshape error).
    @pytest.mark.parametrize(
        "x",
        [
            torch.randint(-128, 127, (2, 4, 4), dtype=torch.int8),
            torch.randint(-128, 127, (2, 3, 4), dtype=torch.int8),
        ],
    )
    async def test_per_channel_negative_axis_numerical(self, x: Tensor) -> None:
        """dequantize with a per-channel scale on a negative axis matches eager."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "scale", torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
                )
                self.register_buffer("zero_point", torch.zeros(4, dtype=torch.int8))

            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.coreai.dequantize(
                    x, self.scale, zero_point=self.zero_point, axis=-1
                )

        model = Model()
        await validate_numerical_output(
            model=model, x=x, prepare_program=inject_subbyte_tensors
        )


# ---------------------------------------------------------------------------
# sparse_to_dense → coreai.build_sparse_with_bitmask + coreai.sparse_with_bitmask_to_dense
# ---------------------------------------------------------------------------


class TestSparseToDense:
    """Tests for sparse_to_dense → coreai.build_sparse_with_bitmask + sparse_with_bitmask_to_dense."""

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "nonzero_data,mask",
        [
            (
                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
                torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8),
            ),
            (
                torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
                torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.uint8),
            ),
            (
                torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32),
                torch.tensor([1, 1, 0, 1, 1, 0, 1, 1, 0, 0], dtype=torch.uint8),
            ),
        ],
    )
    def test_ir(self, nonzero_data: Tensor, mask: Tensor) -> None:
        """sparse_to_dense is lowered to coreai.build_sparse_with_bitmask + coreai.sparse_with_bitmask_to_dense."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("nonzero_data", nonzero_data)
                self.register_buffer("mask", mask)

            def forward(self) -> Tensor:
                return torch.ops.coreai.sparse_to_dense(self.nonzero_data, self.mask)

        model = Model()
        ir = _export_and_import(model)
        filecheck_pattern(
            ir,
            check_file="""
                // CHECK: coreai.build_sparse_with_bitmask
                // CHECK: coreai.sparse_with_bitmask_to_dense
            """,
        )

    @pytest.mark.parametrize(
        "nonzero_data,mask",
        [
            (
                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
                torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8),
            ),
            (
                torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
                torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.uint8),
            ),
            (
                torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32),
                torch.tensor([1, 1, 0, 1, 1, 0, 1, 1, 0, 0], dtype=torch.uint8),
            ),
        ],
    )
    async def test_numerical(self, nonzero_data: Tensor, mask: Tensor) -> None:
        """sparse_to_dense numerical validation against PyTorch eager."""

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("nonzero_data", nonzero_data)
                self.register_buffer("mask", mask)

            def forward(self) -> Tensor:
                return torch.ops.coreai.sparse_to_dense(self.nonzero_data, self.mask)

        model = Model()
        await validate_numerical_output(
            model=model, prepare_program=inject_subbyte_tensors
        )
