# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import math
import sys
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from ..utils import (
    TorchConverter,
    _all_dims_dynamic,
    filecheck_pattern,
    make_dynamic_shapes,
    validate_numerical_output,
)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        # Float tensors with mixed positive/negative values
        torch.rand(2, 3) * 4 - 2,  # 2D: Range [-2, 2]
        torch.rand(3, 4, 5) * 6 - 3,  # 3D: Range [-3, 3]
        torch.rand(10) * 10 - 5,  # 1D: Range [-5, 5]
        # Edge cases
        torch.tensor([0.0, -0.0, 1.0, -1.0, 2.5, -2.5]),
        torch.tensor([[-1.5, 2.0], [3.0, -4.0]]),
        # Integer tensors
        torch.tensor([-5, -2, 0, 3, 7], dtype=torch.int32),
        torch.tensor([[-10, -1], [0, 2], [5, 15]], dtype=torch.int32),
    ],
)
async def test_abs(x: Tensor, dynamic: bool) -> None:
    """Test absolute value operation with various tensor shapes and values."""

    class AbsModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.abs(x)

    model = AbsModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None

    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2) + 1.0])
async def test_acosh(x: Tensor, dynamic: bool) -> None:
    # writing this test seperately, as it has different domain
    # acosh domain is [1, ∞)
    class AcoshModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.acosh(x)

    model = AcoshModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
async def test_add(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class AddModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return x + y

    model = AddModel().eval()
    batch = torch.export.Dim("batch", min=1)
    dynamic_shapes = {"x": {0: batch}, "y": {0: batch}} if dynamic else None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("x", [torch.rand(2, 2)])
async def test_add_constant(x: Tensor) -> None:
    class AddConstantModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return x + 1

    model = AddConstantModel().eval()
    await validate_numerical_output(model=model, x=x)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,y,z",
    [
        (torch.rand(2, 3), torch.rand(2, 3), torch.rand(3, 3)),  # fp32
        (
            torch.rand(2, 3, dtype=torch.float16),
            torch.rand(2, 3, dtype=torch.float16),
            torch.rand(3, 3, dtype=torch.float16),
        ),  # fp16
    ],
)
async def test_addmm(x: Tensor, y: Tensor, z: Tensor, dynamic: bool) -> None:
    class AddMM(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
            return torch.addmm(x, y, z)

    model = AddMM().eval()
    batch = torch.export.Dim("batch", min=1)
    k = torch.export.Dim("k", min=1)
    dynamic_shapes = (
        {"x": {0: batch}, "y": {0: batch, 1: k}, "z": {0: k}} if dynamic else None
    )
    await validate_numerical_output(
        model=model, x=x, y=y, z=z, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,y,z",
    [
        (torch.rand(2, 3), torch.rand(2, 3), torch.rand(3, 3)),  # fp32
        (
            torch.rand(2, 3, dtype=torch.float16),
            torch.rand(2, 3, dtype=torch.float16),
            torch.rand(3, 3, dtype=torch.float16),
        ),  # fp16
    ],
)
async def test_addmm_with_alphbeta(
    x: Tensor, y: Tensor, z: Tensor, dynamic: bool
) -> None:
    class AddMM(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
            return torch.addmm(x, y, z, alpha=2.0, beta=3.0)

    model = AddMM().eval()
    batch = torch.export.Dim("batch", min=1)
    k = torch.export.Dim("k", min=1)
    dynamic_shapes = (
        {"x": {0: batch}, "y": {0: batch, 1: k}, "z": {0: k}} if dynamic else None
    )
    await validate_numerical_output(
        model=model, x=x, y=y, z=z, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3),  # 2D float tensor
        torch.rand(4, 5, 6),  # 3D float tensor
        torch.rand(10),  # 1D float tensor
        torch.randint(0, 10, (3, 4), dtype=torch.int32),  # 2D int tensor
    ],
)
async def test_alias(x: Tensor, dynamic: bool) -> None:
    """Test alias operation which creates a view without copying data."""

    class AliasModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.ops.aten.alias.default(x)

    model = AliasModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        # Boolean tensor - the main case that was failing
        torch.tensor([[True, False, True], [False, False, False]], dtype=torch.bool),
        # 3D Boolean tensor - multi-dimensional test case
        torch.tensor(
            [[[True, False], [False, True]], [[False, False], [True, True]]],
            dtype=torch.bool,
        ),
        # Integer tensor - to test non-boolean conversion
        torch.tensor([[1, 0, 2], [0, 0, 0]], dtype=torch.int32),
        # Float tensor - to test float conversion
        torch.tensor([[1.5, 0.0, 2.3], [0.0, 0.0, 0.0]], dtype=torch.float32),
    ],
)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
async def test_any_dim(x: Tensor, dim: int, keepdim: bool, dynamic: bool) -> None:
    class AnyDimModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.any(x, dim=dim, keepdim=keepdim)

    model = AnyDimModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        # Boolean tensors
        torch.tensor([[True, False, True], [False, False, False]], dtype=torch.bool),
        torch.tensor(
            [[[True, False], [False, True]], [[False, False], [True, True]]],
            dtype=torch.bool,
        ),
        # Integer tensor
        torch.tensor([[1, 0, 2], [0, 0, 0]], dtype=torch.int32),
        # Float tensor
        torch.tensor([[1.5, 0.0, 2.3], [0.0, 0.0, 0.0]], dtype=torch.float32),
        # All True case
        torch.tensor([[True, True], [True, True]], dtype=torch.bool),
        # All False case
        torch.tensor([[False, False], [False, False]], dtype=torch.bool),
    ],
)
async def test_any_default(x: Tensor, dynamic: bool) -> None:
    """Test any operation with no arguments - reduces across all dimensions."""

    class AnyDefaultModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.any(x)

    model = AnyDefaultModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        # 3D Boolean tensor for multi-dim reduction
        torch.tensor(
            [[[True, False], [False, True]], [[False, False], [True, True]]],
            dtype=torch.bool,
        ),
        # Integer tensor
        torch.tensor([[[1, 0], [0, 2]], [[0, 0], [3, 0]]], dtype=torch.int32),
        # Float tensor
        torch.tensor(
            [[[1.5, 0.0], [0.0, 2.3]], [[0.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
        ),
    ],
)
@pytest.mark.parametrize(
    "dims,keepdim",
    [
        ([0, 1], False),  # Reduce dims 0 and 1
        ([1, 2], True),  # Reduce dims 1 and 2 with keepdim
        (None, False),  # Reduce all dims (None case)
    ],
)
async def test_any_dims(
    x: Tensor, dims: list[int] | None, keepdim: bool, dynamic: bool
) -> None:
    """Test any.dims operation - reduces across multiple dimensions."""

    class AnyDimsModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.ops.aten.any.dims(x, dims, keepdim)

    model = AnyDimsModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestArange:
    """Test suite for aten.arange → coreai.range conversion."""

    @pytest.mark.parametrize(
        "start,end,step",
        [
            (0, 10, 1),  # Basic integer range
            (0, 10, 2),  # Integer range with step > 1
            (1, 5, 1),  # Non-zero start
            (0.0, 5.0, 0.5),  # Float range
            (1.0, 10.0, 2.0),  # Float range with step > 1
            (-5, 5, 1),  # Negative to positive range
            (10, 0, -2),  # Negative step (descending)
            (0, 10, None),  # Default step (None, should use 1)
        ],
    )
    async def test_start_step(
        self, start: int | float, end: int | float, step: int | float | None
    ) -> None:
        """Test arange with scalar start/end/step constants."""

        class ArangeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self) -> Tensor:
                if step is None:
                    return torch.arange(start, end, dtype=torch.float32)
                return torch.arange(start, end, step, dtype=torch.float32)

        model = ArangeModel().eval()
        await validate_numerical_output(model=model)

    async def test_end_from_tensor_shape(self) -> None:
        """Test arange where end is derived from a tensor shape dimension.

        This exercises the case where the end operand arrives as a 1D tensor
        (tensor<1xsi32>) rather than a scalar (tensor<si32>), which previously
        caused a 'coreai.range' verifier error.
        """

        class ArangeFromShapeModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                # x.shape[1] produces a rank-1 tensor in the traced graph.
                return torch.arange(0, x.shape[1], dtype=torch.int32)

        model = ArangeFromShapeModel().eval()
        x = torch.zeros(2, 8)
        await validate_numerical_output(model=model, x=x)

    async def test_symint_end_with_float_start_step(self) -> None:
        """Regression for ``replace_arange_start_step``: when ``end`` is
        SymInt-derived (carrying f32 element type from a sym_size cast)
        and ``start`` / ``step`` come in as scalar si32, ``coreai.range_``
        rejects the mismatched element types. The lowering must unify all
        three to the FX-meta target type before the range op."""

        class ArangeMixedModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.arange(0.0, x.shape[0], 0.5, dtype=torch.float32)

        model = ArangeMixedModel().eval()
        x = torch.zeros(6)
        dynamic_shapes = {"x": {0: torch.export.Dim("n", min=2, max=16)}}
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "kernel_size,stride,padding,divisor_override",
    [
        # Basic square kernel, no padding
        ((2, 2), (2, 2), (0, 0), None),
        # Different kernel/stride with padding
        ((3, 3), (2, 2), (1, 1), None),
        # Non-square kernel
        ((2, 3), (2, 3), (0, 0), None),
        # divisor_override: custom denominator instead of kernel_h * kernel_w
        ((3, 3), (2, 2), (1, 1), 4),
        ((2, 2), (2, 2), (0, 0), 3),
        ((2, 4), (2, 2), (0, 0), 6),
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 2, 8, 8),
        (2, 3, 8, 8),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
    ],
)
async def test_avg_pool2d(
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    divisor_override: int | None,
    input_shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    dynamic: bool,
) -> None:
    """Test avg_pool2d operation with various kernel, stride, padding configurations."""
    x = torch.rand(*input_shape, dtype=dtype)

    class AvgPool2dModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.nn.functional.avg_pool2d(
                x,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                divisor_override=divisor_override,
            )

    model = AvgPool2dModel().eval()
    dynamic_shapes = (
        {
            "x": {
                0: torch.export.Dim("batch", min=1),
                1: torch.export.Dim("channels", min=1),
            }
        }
        if dynamic
        else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "kernel_size,stride,padding,divisor_override",
    [
        # Basic cube kernel, no padding
        ((2, 2, 2), (2, 2, 2), (0, 0, 0), None),
        # Different kernel/stride with padding
        ((3, 3, 3), (2, 2, 2), (1, 1, 1), None),
        # Non-cube kernel
        ((2, 3, 2), (2, 3, 2), (0, 0, 0), None),
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 2, 8, 8, 8),
        (2, 3, 8, 8, 8),
    ],
)
async def test_avg_pool3d(
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    divisor_override: int | None,
    input_shape: tuple[int, int, int, int, int],
    dynamic: bool,
) -> None:
    """Test avg_pool3d operation with various kernel, stride, padding configurations.

    Note: PyTorch avg_pool3d does not support float16 on CPU, so only float32 is tested.
    """
    x = torch.rand(*input_shape, dtype=torch.float32)

    class AvgPool3dModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.nn.functional.avg_pool3d(
                x,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                divisor_override=divisor_override,
            )

    model = AvgPool3dModel().eval()
    dynamic_shapes = (
        {
            "x": {
                0: torch.export.Dim("batch", min=1),
                1: torch.export.Dim("channels", min=1),
            }
        }
        if dynamic
        else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "input_shape,output_size,dtype",
    [
        # Divisible cases (input evenly divisible by output)
        ((1, 1, 8, 8), (4, 4), torch.float32),  # 8/4=2, evenly divisible
        ((1, 3, 12, 12), (4, 4), torch.float32),  # 12/4=3, evenly divisible
        # Global average pooling (output_size = (1, 1))
        ((2, 16, 7, 7), (1, 1), torch.float32),  # Global avg pool
        # Non-divisible cases (requires slice-based adaptive pooling)
        ((2, 3, 13, 19), (3, 5), torch.float32),  # Non-divisible dimensions
        (
            (2, 3, 15, 24),
            (4, 8),
            torch.float32,
        ),  # Partially divisible (24/8=3, 15/4 not)
        (
            (2, 2, 13, 18),
            (3, 6),
            torch.float32,
        ),  # Height non-divisible, width divisible
        # Prime divisor cases
        ((2, 2, 13, 19), (7, 7), torch.float32),  # Prime-ish output sizes
        # FP16 precision
        ((1, 3, 8, 8), (4, 4), torch.float16),  # FP16 divisible
        ((1, 3, 13, 19), (3, 5), torch.float16),  # FP16 non-divisible
    ],
)
@pytest.mark.parametrize("dynamic_dims", [tuple(), (2,), (3,), (2, 3)])
async def test_adaptive_avg_pool2d(
    input_shape: tuple[int, int, int, int],
    output_size: tuple[int, int],
    dtype: torch.dtype,
    dynamic_dims: tuple[int],
) -> None:
    """Test adaptive_avg_pool2d operation which pools to a specified output size.

    aten._adaptive_avg_pool2d(input, output_size) -> Tensor

    Performs adaptive average pooling on 2D spatial dimensions (H, W) of NCHW tensor.
    The output has spatial dimensions equal to output_size, regardless of input size.

    Supports both divisible and non-divisible input/output size combinations.
    """
    x = torch.rand(input_shape, dtype=dtype)

    class AdaptiveAvgPool2dModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(output_size)

        def forward(self, x: Tensor) -> Tensor:
            return self.pool(x)

    model = AdaptiveAvgPool2dModel().eval()
    dynamic_shapes = {"x": [torch.export.Dim.STATIC for _ in range(len(x.shape))]}
    for d in dynamic_dims:
        dynamic_shapes["x"][d] = torch.export.Dim(f"dim_{d}")
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "shape,dim,keepdim,dtype",
    [
        # Global argmax (dim=None) - returns scalar index of max in flattened tensor
        ((3, 4), None, False, torch.float32),
        ((2, 3, 4), None, False, torch.float32),
        # argmax along specific dimensions with keepdim variations
        ((2, 5, 4), 1, True, torch.float32),
        ((2, 5, 4), 1, False, torch.float32),
        ((3, 4, 5), 0, True, torch.float32),
        ((3, 4, 5), -1, False, torch.float32),  # negative dim
        ((4, 6), 0, True, torch.float16),
        ((4, 6), 0, False, torch.float16),
        # 1D tensor
        ((10,), 0, False, torch.float32),
    ],
)
async def test_argmax(
    shape: tuple[int, ...],
    dim: int | None,
    keepdim: bool,
    dtype: torch.dtype,
    dynamic: bool,
) -> None:
    """Test argmax operation which returns indices of maximum values."""
    x = torch.rand(shape, dtype=dtype)

    class ArgmaxModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            if dim is None:
                return torch.argmax(x)
            return torch.argmax(x, dim=dim, keepdim=keepdim)

    model = ArgmaxModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3, 8, 8),
        torch.rand(2, 3, 8, 8, dtype=torch.float16),  # fp16
    ],
)
@pytest.mark.parametrize(
    "dynamic_dims", [tuple(), (0,), (2,), (3,), (0, 2), (0, 3), (0, 2, 3)]
)
async def test_batchnorm(x: Tensor, dynamic_dims: tuple[int]) -> None:
    class BatchNormModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x: Tensor) -> Tensor:
            return self.bn(x)

    model = BatchNormModel().eval()
    dim_names = {0: "batch", 1: "channels", 2: "height", 3: "width"}
    dynamic_shapes = make_dynamic_shapes(x={d: dim_names[d] for d in dynamic_dims})
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3) * 4 - 2,  # 2D: Range [-2, 2]
        torch.rand(3, 4, 5) * 6 - 3,  # 3D: Range [-3, 3]
        torch.rand(10) * 10 - 5,  # 1D: Range [-5, 5]
        torch.tensor(
            [0.0, 1.0, -1.0, 1.5, -1.5, 2.9, -2.9]
        ),  # Specific values including integers
        torch.tensor([[0.1, 0.9], [-0.1, -0.9]]),  # Values close to integers
    ],
)
async def test_ceil(x: Tensor, dynamic: bool) -> None:
    """Test ceil operation with various tensor shapes and values."""

    class CeilModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.ceil(x)

    model = CeilModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("x", [torch.rand(1, 128, 4)])
async def test_truediv_ceil_dynamic_shapes(x: Tensor) -> None:
    """operator.truediv on symbolic ints must use float division.

    Regression test for the Emformer concat shape mismatch:
    math.ceil(symint / const) needs true (float) division so the ceiling
    rounds up correctly (e.g. ceil(31/4)==8, not 7).

    The model derives a symbolic utterance_length via reshape + slice,
    then builds per-segment mask blocks whose widths depend on
    ceil(utterance_length / segment_length).  The outer concat along
    dim=0 fails at runtime when ceil produces the wrong value.
    """

    class TruedivCeilModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.segment_length = 4
            self.proj = nn.Linear(4, 16, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            # x: [B, T_raw, 4] → proj → [B, T_raw, 16]
            # reshape as time-reduction(stride=4): [B, T_raw//4, 64]
            proj = self.proj(x)
            B, T_raw, _ = proj.shape
            T = T_raw // self.segment_length  # symbolic
            proj = proj.reshape(B, T, 64)
            # derive utterance_length from T (drop 1 for right-context)
            utt_len = proj.size(1) - 1  # symbolic

            n = math.ceil(utt_len / self.segment_length)

            blocks: list[Tensor] = []
            for i in range(n):
                # Each block has width = n + utt_len (constant across i)
                pieces = [
                    torch.zeros(1, i, device=x.device),
                    torch.ones(1, 1, device=x.device),
                    torch.zeros(1, n - i - 1, device=x.device),
                    torch.ones(
                        1,
                        min((i + 1) * self.segment_length, utt_len),
                        device=x.device,
                    ),
                    torch.zeros(
                        1,
                        utt_len - min((i + 1) * self.segment_length, utt_len),
                        device=x.device,
                    ),
                ]
                blocks.append(torch.cat(pieces, dim=1))

            return torch.cat(blocks, dim=0)

    model = TruedivCeilModel().eval()
    _half = torch.export.Dim("_half", min=2, max=128)
    dynamic_shapes = {"x": {1: 2 * _half}}
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(3, 4, 5)])
@pytest.mark.parametrize("scalar_value", [2.0, 0.5, -1.5, 10.0])
@pytest.mark.parametrize(
    "binary_scalar_op",
    [
        lambda x, s: torch.ops.aten.add.Scalar(x, s),  # add.Scalar
        lambda x, s: torch.ops.aten.sub.Scalar(x, s),  # sub.Scalar
        lambda x, s: torch.ops.aten.mul.Scalar(x, s),  # mul.Scalar
        lambda x, s: torch.ops.aten.div.Scalar(x, s),  # div.Scalar
        lambda x, s: torch.ops.aten.fmod.Scalar(x, s),  # fmod.Scalar
    ],
)
async def test_binary_op_scalar_version(
    x: Tensor, scalar_value: float, binary_scalar_op: Any, dynamic: bool
) -> None:
    """Test scalar binary operations (add.Scalar, sub.Scalar, mul.Scalar, div.Scalar, fmod.Scalar) which use wrap_for_scalar_broadcast."""

    class BinaryScalarModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return binary_scalar_op(x, scalar_value)

    model = BinaryScalarModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.randint(0, 2, (2, 3, 4), dtype=torch.bool),
        torch.randint(0, 2, (2, 3, 4), dtype=torch.int8),
    ],
)
async def test_bitwise_not(x: Tensor, dynamic: bool) -> None:
    class BinaryBitwiseNotModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.bitwise_not(x)

    model = BinaryBitwiseNotModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.tensor([-1, -2, 3], dtype=torch.int8)])
@pytest.mark.parametrize("y", [torch.tensor([1, 0, 3], dtype=torch.int8)])
@pytest.mark.parametrize(
    "binary_bitwise_op",
    [
        torch.bitwise_and,
        torch.bitwise_or,
        torch.bitwise_xor,
    ],
)
async def test_binary_bitwise_ops(
    x: Tensor, y: Tensor, binary_bitwise_op: Any, dynamic: bool
) -> None:
    class BinaryBitwiseModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return binary_bitwise_op(x, y)

    model = BinaryBitwiseModel().eval()
    batch = torch.export.Dim("batch", min=1)
    dynamic_shapes = {"x": {0: batch}, "y": {0: batch}} if dynamic else None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
@pytest.mark.parametrize(
    "binary_comparison_op",
    [
        torch.eq,
        torch.ge,
        torch.gt,
        torch.le,
        torch.lt,
        torch.ne,
    ],
)
@pytest.mark.parametrize("dynamic", [True, False])
async def test_binary_comparison_ops(
    x: Tensor, y: Tensor, binary_comparison_op: Any, dynamic: bool
) -> None:
    class BinaryComparisonModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return binary_comparison_op(x, y)

    model = BinaryComparisonModel().eval()

    dyn_dims = {
        0: torch.export.Dim("batch", min=1),
        1: torch.export.Dim("features", min=1),
        2: torch.export.Dim("spatial", min=1),
    }
    dynamic_shapes = {
        "x": [
            torch.export.Dim.STATIC if not dynamic else dyn_dims[i]
            for i in range(len(x.shape))
        ],
        "y": [
            torch.export.Dim.STATIC if not dynamic else dyn_dims[i]
            for i in range(len(y.shape))
        ],
    }

    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("x", [torch.rand(3, 4, 5)])
@pytest.mark.parametrize("scalar_value", [0.5, 1.0, -0.5, 2.0])
@pytest.mark.parametrize(
    "binary_comparison_scalar_op",
    [
        lambda x, s: torch.ops.aten.eq.Scalar(x, s),  # eq.Scalar
        lambda x, s: torch.ops.aten.ge.Scalar(x, s),  # ge.Scalar
        lambda x, s: torch.ops.aten.gt.Scalar(x, s),  # gt.Scalar
        lambda x, s: torch.ops.aten.lt.Scalar(x, s),  # lt.Scalar
        lambda x, s: torch.ops.aten.le.Scalar(x, s),  # le.Scalar
        lambda x, s: torch.ops.aten.ne.Scalar(x, s),  # ne.Scalar
    ],
)
async def test_binary_comparison_ops_scalar_version(
    x: Tensor, scalar_value: float, binary_comparison_scalar_op: Any
) -> None:
    """Test scalar binary comparison operations (eq.Scalar, ge.Scalar, gt.Scalar, lt.Scalar, le.Scalar, ne.Scalar)."""

    class BinaryComparisonScalarModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return binary_comparison_scalar_op(x, scalar_value)

    model = BinaryComparisonScalarModel().eval()
    await validate_numerical_output(model=model, x=x)


class TestBmm:
    """Test suite for aten.bmm → coreai.batch_matmul conversion."""

    @pytest.mark.parametrize(
        "dynamic_shapes",
        [
            None,  # static
            make_dynamic_shapes(mat1=["batch"], mat2=["batch"]),  # batch only
            make_dynamic_shapes(
                mat1=["batch", "M", "K"], mat2=["batch", "K", "N"]
            ),  # all
        ],
    )
    @pytest.mark.parametrize(
        "mat1,mat2",
        [
            # fp32 - varied shapes (B, M, K) x (B, K, N)
            (torch.rand(4, 3, 5), torch.rand(4, 5, 2)),  # basic
            (torch.rand(2, 8, 16), torch.rand(2, 16, 4)),  # larger inner dim
            (torch.rand(3, 4, 4), torch.rand(3, 4, 4)),  # square matrices
            # fp16
            (
                torch.rand(4, 3, 5, dtype=torch.float16),
                torch.rand(4, 5, 2, dtype=torch.float16),
            ),
        ],
    )
    async def test_basic(
        self, mat1: Tensor, mat2: Tensor, dynamic_shapes: dict | None
    ) -> None:
        class BmmModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, mat1: Tensor, mat2: Tensor) -> Tensor:
                return torch.bmm(mat1, mat2)

        model = BmmModel().eval()
        await validate_numerical_output(
            model=model, mat1=mat1, mat2=mat2, dynamic_shapes=dynamic_shapes
        )

    async def test_mixed_dtypes(self) -> None:
        """Test bmm with mixed f32/f16 inputs.

        Reproduces the EfficientSam pattern: model.half() makes weights f16,
        but an explicit dtype=torch.float32 tensor creates f32 that flows
        into a bmm with f16 weights.
        """

        class MixedBmmModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 8, 4))

            def forward(self, x: Tensor) -> Tensor:
                # Explicit f32 creation contaminates x via add
                f32_val = torch.ones(1, device=x.device, dtype=torch.float32)
                x = x + f32_val  # promotes x(f16) to f32
                return torch.bmm(x, self.weight)  # f32 @ f16

        model = MixedBmmModel().eval().half()
        x = torch.randn(3, 4, 8, dtype=torch.float16)

        with torch.autocast(device_type="cpu", dtype=torch.float16):
            exported_program = torch.export.export(model, args=(), kwargs={"x": x})
        exported_program = exported_program.run_decompositions(
            torch.export.default_decompositions()
        )

        converter = TorchConverter().add_exported_program(exported_program)
        converter.to_coreai()


class TestCat:
    """Test suite for aten.cat → coreai.concat conversion."""

    @pytest.mark.parametrize(
        "tensors,dim",
        [
            # Concatenate along dim 0
            ([torch.rand(2, 3), torch.rand(3, 3)], 0),
            # Concatenate along dim 1
            ([torch.rand(2, 3), torch.rand(2, 4)], 1),
            # Concatenate 3D tensors along dim 2
            ([torch.rand(2, 3, 4), torch.rand(2, 3, 5)], 2),
            # Concatenate with negative dim=-1 on 2D (wraps to dim 1)
            ([torch.rand(2, 3), torch.rand(2, 5)], -1),
            # Concatenate 3D tensors with dim=-1 (wraps to dim 2)
            ([torch.rand(2, 3, 4), torch.rand(2, 3, 5)], -1),
            # Concatenate 3D tensors with dim=-2 (wraps to dim 1)
            ([torch.rand(2, 3, 4), torch.rand(2, 5, 4)], -2),
            # Concatenate 3D tensors with dim=-3 (wraps to dim 0)
            ([torch.rand(2, 3, 4), torch.rand(4, 3, 4)], -3),
            # Concatenate 4D tensors with dim=-2 (wraps to dim 2)
            ([torch.rand(2, 3, 4, 5), torch.rand(2, 3, 6, 5)], -2),
            # Concatenate 1D tensors
            ([torch.rand(5), torch.rand(3)], 0),
        ],
    )
    async def test_dim(self, tensors: list[Tensor], dim: int) -> None:
        """Concatenation along various positive and negative dimensions."""

        class CatModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return torch.cat([x, y], dim=dim)

        model = CatModel().eval()
        await validate_numerical_output(model=model, x=tensors[0], y=tensors[1])

    @pytest.mark.ir
    def test_skips_empty_tensors(self) -> None:
        """Empty (zero-element) tensors must be silently dropped by replace_cat."""

        class CatModel(nn.Module):
            def forward(self, x: Tensor, empty: Tensor, y: Tensor) -> Tensor:
                return torch.cat([x, empty, y], dim=0)

        model = CatModel().eval()
        x = torch.rand(2, 3)
        empty = torch.zeros(0, 3)
        y = torch.rand(3, 3)

        exported = torch.export.export(model, args=(x, empty, y)).run_decompositions()
        converter = TorchConverter().add_exported_program(exported)
        coreai_program = converter.to_coreai()

        # The empty tensor must not appear as a concat input in the IR
        check_file = """
            // CHECK: coreai.concat
            // CHECK-NOT: tensor<0x3xf32>
        """
        filecheck_pattern(str(coreai_program), check_file=check_file)

    async def test_dynamic_vs_static_non_concat_axis(self) -> None:
        """Regression for non-concat-axis promotion: when one cat input has a
        dynamic non-concat axis while a sibling has a known static size for
        that same axis, the verifier can't statically prove they match. The
        lowering reshapes the dynamic side to the known static size before
        the concat (needed by multi-resolution feature merges under dynamic
        shapes)."""

        class CatModel(nn.Module):
            def forward(self, a: Tensor, b: Tensor) -> Tensor:
                return torch.cat([a, b], dim=0)

        a = torch.rand(2, 4)
        b = torch.rand(3, 4)
        # Mark only a's dim 1 dynamic; b's dim 1 stays static at 4. Dim.AUTO
        # prevents torch.export from specializing it back to a constant.
        await validate_numerical_output(
            model=CatModel().eval(),
            a=a,
            b=b,
            dynamic_shapes=({1: torch.export.Dim.AUTO}, {}),
        )

    async def test_partial_static_promotion_with_dynamic_axes(self) -> None:
        """A cat input has one non-concat axis that is statically known
        via a sibling AND another non-concat axis that is dynamic on every
        input. After promoting the first axis to its static size, the
        second axis remains dynamic, so the lowering must build the
        reshape's shape vector at runtime."""

        class CatModel(nn.Module):
            def forward(self, a: Tensor, b: Tensor) -> Tensor:
                return torch.cat([a, b], dim=2)

        a = torch.rand(2, 4, 5, 6)
        b = torch.rand(2, 4, 7, 6)
        # Mark dim 1 of `a` dynamic (sibling `b` has static 4 there → must be
        # promoted) and dim 0 of both inputs dynamic (no static sibling →
        # remains dynamic post-promotion, forcing the runtime-shape path).
        await validate_numerical_output(
            model=CatModel().eval(),
            a=a,
            b=b,
            dynamic_shapes=(
                {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO},
                {0: torch.export.Dim.AUTO},
            ),
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3),
    ],
)
async def test_clone(x: Tensor, dynamic: bool) -> None:
    class CloneModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.clone(x)

    model = CloneModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestConstantPadNd:
    """Tests for aten.constant_pad_nd → coreai crop-then-pad conversion."""

    def _dynamic_shapes(self, x: Tensor) -> dict:
        # Use Dim.DYNAMIC so torch.export infers the valid range itself.
        # Keep size-1 dims STATIC — torch.export cannot make them symbolic.
        return {
            "x": [
                torch.export.Dim.DYNAMIC if x.shape[d] > 1 else torch.export.Dim.STATIC
                for d in range(x.dim())
            ]
        }

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x,padding,value",
        [
            # 2D tensor with positive padding on all sides
            (torch.rand(2, 3), (1, 2, 3, 2), 0.0),
            # 3D tensor with positive padding
            (torch.rand(2, 3, 4), (1, 1, 2, 2, 0, 0), 0.0),
            # 4D tensor (NCHW) with custom fill value
            (torch.rand(1, 3, 4, 4), (1, 1, 1, 1), 1.0),
            # 1D tensor with padding
            (torch.rand(5), (2, 3), 0.0),
            # Negative padding (cropping) on last dimension
            (torch.rand(3, 5), (-1, -1), 0.0),
            # Mixed positive and negative padding
            (torch.rand(3, 6), (1, -2), 0.0),
            # Asymmetric negative padding on multiple dimensions
            (torch.rand(4, 6, 8), (-1, -2, 0, -1), 0.0),
        ],
    )
    async def test_mixed(
        self, x: Tensor, padding: tuple[int, ...], value: float, dynamic: bool
    ) -> None:
        """Test constant_pad_nd with various padding configurations."""

        class PadModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.pad(x, padding, mode="constant", value=value)

        model = PadModel().eval()
        dynamic_shapes = self._dynamic_shapes(x) if dynamic else None
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x,padding,value",
        [
            # Crop last dim only
            (torch.rand(4, 8), (-1, -2), 0.0),
            # Crop both dims of a 2D tensor
            (torch.rand(6, 8), (-1, -1, -2, -2), 0.0),
            # Crop last two dims of a 3D tensor
            (torch.rand(2, 6, 10), (0, 0, -1, -2), 0.0),
            # Asymmetric crop
            (torch.rand(5, 12), (-3, -1), 0.0),
        ],
    )
    async def test_crop_only(
        self, x: Tensor, padding: tuple[int, ...], value: float, dynamic: bool
    ) -> None:
        """Test constant_pad_nd with all-negative padding (pure crop, no padding)."""

        class CropModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.pad(x, padding, mode="constant", value=value)

        model = CropModel().eval()
        dynamic_shapes = self._dynamic_shapes(x) if dynamic else None
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,min_bound,max_bound",
    [
        # Scalar bounds
        (torch.rand(2, 3) * 3 - 2, -1.0, 1.0),  # both
        (torch.rand(4, 5, 6) * 6 - 3, -1.0, None),  # min only
        (torch.rand(10) * 10 - 5, None, 1.0),  # max only
        # Tensor bounds
        (
            torch.rand(3, 4) * 4 - 2,
            torch.full((3, 4), -0.5),
            torch.full((3, 4), 0.5),
        ),  # both
        (torch.rand(2, 3) * 6 - 3, torch.full((2, 3), -1.0), None),  # tensor min only
        (torch.rand(2, 3) * 6 - 3, None, torch.full((2, 3), 1.0)),  # tensor max only
        # Broadcast: 1D tensor bounds over 2D input
        (
            torch.rand(3, 4) * 4 - 2,
            torch.tensor([-0.5, -0.3, -0.1, -0.2]),
            torch.tensor([0.2, 0.4, 0.6, 0.8]),
        ),
    ],
)
async def test_clamp(
    x: Tensor,
    min_bound: float | Tensor | None,
    max_bound: float | Tensor | None,
    dynamic: bool,
) -> None:
    """Test clamp with scalar or tensor min/max bounds (clamp.default and clamp.Tensor)."""
    min_is_tensor = isinstance(min_bound, Tensor)
    max_is_tensor = isinstance(max_bound, Tensor)

    # One Dim per dimension of x; bounds align from the right (broadcast rules).
    x_dims = _all_dims_dynamic(x) if dynamic else None

    def _dyn_for_bound(t: Tensor) -> dict:
        offset = x.dim() - t.dim()
        return {i: x_dims[offset + i] for i in range(t.dim())}

    if min_is_tensor or max_is_tensor:
        # Tensor-bound variants: each combo needs its own export signature
        if min_bound is not None and max_bound is not None:

            class ClampBothModel(nn.Module):
                def forward(self, x: Tensor, min_t: Tensor, max_t: Tensor) -> Tensor:
                    return torch.clamp(x, min=min_t, max=max_t)

            dynamic_shapes = (
                {
                    "x": x_dims,
                    "min_t": _dyn_for_bound(min_bound),
                    "max_t": _dyn_for_bound(max_bound),
                }
                if dynamic
                else None
            )
            await validate_numerical_output(
                model=ClampBothModel().eval(),
                x=x,
                min_t=min_bound,
                max_t=max_bound,
                dynamic_shapes=dynamic_shapes,
            )
        elif min_bound is not None:

            class ClampMinModel(nn.Module):
                def forward(self, x: Tensor, min_t: Tensor) -> Tensor:
                    return torch.clamp(x, min=min_t)

            dynamic_shapes = (
                {"x": x_dims, "min_t": _dyn_for_bound(min_bound)} if dynamic else None
            )
            await validate_numerical_output(
                model=ClampMinModel().eval(),
                x=x,
                min_t=min_bound,
                dynamic_shapes=dynamic_shapes,
            )
        else:

            class ClampMaxModel(nn.Module):
                def forward(self, x: Tensor, max_t: Tensor) -> Tensor:
                    return torch.clamp(x, max=max_t)

            dynamic_shapes = (
                {"x": x_dims, "max_t": _dyn_for_bound(max_bound)} if dynamic else None
            )
            await validate_numerical_output(
                model=ClampMaxModel().eval(),
                x=x,
                max_t=max_bound,
                dynamic_shapes=dynamic_shapes,
            )
    else:
        # Scalar-bound variant: min/max are captured as constants
        min_val, max_val = min_bound, max_bound

        class ClampScalarModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.clamp(x, min=min_val, max=max_val)

        dynamic_shapes = {"x": x_dims} if dynamic else None
        await validate_numerical_output(
            model=ClampScalarModel().eval(), x=x, dynamic_shapes=dynamic_shapes
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "in_ch,out_ch,kernel,stride,padding,dilation,groups,bias,input_len",
    [
        # Basic Conv1d, no padding
        (3, 16, 3, 1, 0, 1, 1, True, 8),
        # Stride=5, no padding, no bias
        (3, 16, 3, 5, 0, 1, 1, False, 25),
        # Non-zero padding (exercises the pad branch in replace_conv for rank==3)
        (3, 16, 3, 1, 1, 1, 1, True, 8),
        (4, 8, 5, 1, 2, 1, 1, False, 16),
        # Padding + dilation
        (3, 8, 3, 1, 2, 2, 1, True, 16),
    ],
)
async def test_conv1d(
    in_ch: int,
    out_ch: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    bias: bool,
    input_len: int,
    dynamic: bool,
) -> None:
    x = torch.rand(2, in_ch, input_len)

    class Conv1dModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.conv(x)

    model = Conv1dModel().eval()
    # Batch (dim 0) and spatial length (dim 2) can be dynamic; channel (dim 1) is fixed by the layer.
    # min_len ensures output_len >= 2 (avoids specialization when output == 1) and length >= kernel.
    # max=2**20 keeps us within the upper bound torch.export derives from conv arithmetic.
    if dynamic:
        min_len = max(kernel, stride + dilation * (kernel - 1) + 1 - 2 * padding)
        dynamic_shapes = {
            "x": {
                0: torch.export.Dim("batch", min=1),
                2: torch.export.Dim("length", min=min_len, max=2**20),
            }
        }
    else:
        dynamic_shapes = None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "in_ch,out_ch,kernel,padding,dilation,groups,bias,input_shape",
    [
        # Basic 2D conv with bias
        (3, 16, 3, 1, 1, 1, True, (1, 3, 8, 8)),
        # Without bias
        (3, 16, 3, 1, 1, 1, False, (1, 3, 8, 8)),
        # Depthwise: groups == in_channels
        (4, 4, 3, 1, 1, 4, True, (1, 4, 8, 8)),
        (8, 8, 3, 1, 1, 8, True, (2, 8, 6, 6)),
        # Dilation > 1 (atrous convolution)
        (3, 8, 3, 2, 2, 1, True, (1, 3, 16, 16)),
        (3, 8, 3, (2, 3), (2, 3), 1, True, (1, 3, 16, 16)),
    ],
)
async def test_conv2d(
    in_ch: int,
    out_ch: int,
    kernel: int,
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
    groups: int,
    bias: bool,
    input_shape: tuple[int, int, int, int],
    dynamic: bool,
) -> None:
    x = torch.rand(2, *input_shape[1:])

    class Conv2dModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.conv(x)

    model = Conv2dModel().eval()
    # Batch (dim 0) and spatial H/W (dims 2, 3) can be dynamic; channel (dim 1) is fixed by the layer.
    dynamic_shapes = (
        {
            "x": {
                0: torch.export.Dim("batch", min=1),
                2: torch.export.Dim("H", min=kernel),
                3: torch.export.Dim("W", min=kernel),
            }
        }
        if dynamic
        else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "in_ch,out_ch,kernel,stride,padding,dilation,groups,bias,input_shape",
    [
        # Basic 3D conv with bias, no padding
        (3, 8, 3, 1, 0, 1, 1, True, (2, 3, 6, 6, 6)),
        # Without bias
        (3, 8, 3, 1, 0, 1, 1, False, (2, 3, 6, 6, 6)),
        # Non-zero padding (exercises the pad branch in replace_conv for rank==5)
        (3, 8, 3, 1, 1, 1, 1, True, (2, 3, 6, 6, 6)),
        # Anisotropic kernel / padding / dilation across D, H, W
        (3, 8, (3, 3, 3), 1, (1, 2, 0), (1, 2, 1), 1, True, (2, 3, 6, 8, 6)),
        # Stride > 1
        (3, 8, 3, 2, 1, 1, 1, True, (2, 3, 8, 8, 8)),
        # Depthwise: groups == in_channels
        (4, 4, 3, 1, 1, 1, 4, True, (2, 4, 6, 6, 6)),
    ],
)
async def test_conv3d(
    in_ch: int,
    out_ch: int,
    kernel: int | tuple[int, int, int],
    stride: int,
    padding: int | tuple[int, int, int],
    dilation: int | tuple[int, int, int],
    groups: int,
    bias: bool,
    input_shape: tuple[int, int, int, int, int],
    dynamic: bool,
) -> None:
    x = torch.rand(*input_shape)

    class Conv3dModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.conv(x)

    model = Conv3dModel().eval()

    # Batch (dim 0) and spatial D/H/W (dims 2, 3, 4) can be dynamic; channel (dim 1) is fixed.
    # Per-axis min matches the conv1d logic: large enough that kernel fits AND output >= 2
    # (an output of 1 gets specialized, breaking the dynamic guard).
    def _triple(v: int | tuple[int, int, int]) -> tuple[int, int, int]:
        return v if isinstance(v, tuple) else (v, v, v)

    k = _triple(kernel)
    p = _triple(padding)
    d = _triple(dilation)
    s = _triple(stride)
    min_d = max(k[0], s[0] + d[0] * (k[0] - 1) + 1 - 2 * p[0])
    min_h = max(k[1], s[1] + d[1] * (k[1] - 1) + 1 - 2 * p[1])
    min_w = max(k[2], s[2] + d[2] * (k[2] - 1) + 1 - 2 * p[2])
    dynamic_shapes = (
        {
            "x": {
                0: torch.export.Dim("batch", min=1),
                2: torch.export.Dim("D", min=min_d, max=2**20),
                3: torch.export.Dim("H", min=min_h, max=2**20),
                4: torch.export.Dim("W", min=min_w, max=2**20),
            }
        }
        if dynamic
        else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestCopy:
    """Test suite for aten.copy.default → coreai.broadcast_to conversion."""

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "dest,src",
        [
            # Same shape and dtype - basic case
            (torch.rand(2, 3), torch.rand(2, 3)),
            # Different dtype - int to float
            (torch.rand(3, 4), torch.randint(0, 10, (3, 4), dtype=torch.int32)),
            # Different dtype - float to int
            (torch.randint(0, 10, (2, 2), dtype=torch.int32), torch.rand(2, 2)),
            # Broadcasting - scalar src to larger dest
            (torch.rand(3, 4), torch.rand(1, 1)),
            # Broadcasting - 1D src to 2D dest
            (torch.rand(2, 4), torch.rand(1, 4)),
            # Different dtype with broadcasting
            (torch.rand(3, 3), torch.randint(0, 5, (1, 3), dtype=torch.int32)),
        ],
    )
    async def test_cast_and_broadcast(
        self, dest: Tensor, src: Tensor, dynamic: bool
    ) -> None:
        """Test copy operation with casting and broadcasting."""

        class CopyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, dest: Tensor, src: Tensor) -> Tensor:
                return torch.ops.aten.copy.default(dest, src)

        model = CopyModel().eval()
        if dynamic:
            dest_dims = _all_dims_dynamic(dest, "dest_d")
            offset = dest.dim() - src.dim()
            # Only share dest Dim for equal-sized dims; broadcast dims stay static.
            src_dims = {
                i: dest_dims[offset + i] for i in range(src.dim()) if src.size(i) != 1
            }
            dynamic_shapes = {"dest": dest_dims, "src": src_dims}
        else:
            dynamic_shapes = None
        await validate_numerical_output(
            model=model, dest=dest, src=src, dynamic_shapes=dynamic_shapes
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
async def test_div(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class DivModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return x / y

    model = DivModel().eval()
    if dynamic:
        dims = _all_dims_dynamic(x)
        dynamic_shapes = {"x": dims, "y": dims}
    else:
        dynamic_shapes = None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize(
    "x,y",
    [
        # Float tensors - mixed positive/negative values
        (
            torch.tensor([[3.5, -7.2], [-2.8, 9.1]]),
            torch.tensor([[2.0, 3.0], [2.0, -4.0]]),
        ),
        # Larger tensors
        (
            torch.rand(3, 4) * 10 - 5,
            torch.rand(3, 4) * 4 + 0.5,
        ),  # Avoid division by values near zero
        # Broadcasting case
        (torch.rand(2, 3, 4) * 10 - 5, torch.rand(1, 3, 1) * 4 + 0.5),
    ],
)
@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
async def test_div_tensor_mode(x: Tensor, y: Tensor, rounding_mode: str | None) -> None:
    """Test division with different rounding modes.

    aten.div.Tensor_mode(input, other, rounding_mode) supports:
        - None: True division (standard floating-point division)
        - "floor": Floor division (rounds toward negative infinity)
        - "trunc": Truncated division (rounds toward zero)
    """

    class DivTensorModeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return torch.div(x, y, rounding_mode=rounding_mode)

    model = DivTensorModeModel().eval()
    await validate_numerical_output(model=model, x=x, y=y)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,y",
    [
        (torch.rand(2, 3) + 0.1, torch.rand(2, 3) + 0.1),
        (torch.rand(3, 4, 5) + 0.1, torch.rand(3, 4, 5) + 0.1),
        (torch.rand(4) + 0.1, torch.rand(4) + 0.1),
        # FP16
        (
            torch.rand(2, 3, dtype=torch.float16) + 0.1,
            torch.rand(2, 3, dtype=torch.float16) + 0.1,
        ),
    ],
)
async def test_true_divide(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class TrueDivideModel(nn.Module):
        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return torch.true_divide(x, y)

    model = TrueDivideModel().eval()
    if dynamic:
        dims = _all_dims_dynamic(x)
        dynamic_shapes = {"x": dims, "y": dims}
    else:
        dynamic_shapes = None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 3) + 0.1, torch.rand(3, 4, 5) + 0.1])
async def test_true_divide_scalar(x: Tensor, dynamic: bool) -> None:
    class TrueDivideScalarModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.true_divide(x, 2.0)

    model = TrueDivideScalarModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3) * 4 - 2,  # 2D: Range [-2, 2]
        torch.rand(3, 4, 5) * 6 - 3,  # 3D: Range [-3, 3]
        torch.rand(10) * 10 - 5,  # 1D: Range [-5, 5]
        torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0]),  # Specific values
        torch.tensor([[0.5, 1.5], [-0.5, -1.5]]),  # 2D with mixed signs
    ],
)
async def test_exp2(x: Tensor, dynamic: bool) -> None:
    """Test exp2 operation which computes 2^x element-wise."""

    class Exp2Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.exp2(x)

    model = Exp2Model().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3) * 2 - 1,  # 2D: Range [-1, 1] for numerical stability
        torch.rand(3, 4, 5) * 4 - 2,  # 3D: Range [-2, 2]
        torch.rand(10) * 6 - 3,  # 1D: Range [-3, 3]
        torch.tensor(
            [0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0]
        ),  # Specific values near zero
        torch.tensor(
            [[-0.01, 0.01], [0.001, -0.001]]
        ),  # Small values for numerical precision
    ],
)
async def test_expm1(x: Tensor, dynamic: bool) -> None:
    """Test expm1 operation (exp(x) - 1) with various tensor shapes and values."""

    class Expm1Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.expm1(x)

    model = Expm1Model().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dims",
    [
        (torch.rand(6), [0]),  # 1D, single dim
        (torch.rand(3, 4), [0]),  # 2D, flip rows
        (torch.rand(3, 4), [1]),  # 2D, flip cols
        (torch.rand(3, 4), [0, 1]),  # 2D, flip both dims
        (torch.rand(2, 3, 4), [2]),  # 3D, flip last dim
        (torch.rand(2, 3, 4), [0, 2]),  # 3D, flip multiple dims
        (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4), [1]),  # known values
        (torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32), [0]),  # int32
    ],
)
async def test_flip(x: Tensor, dims: list[int], dynamic: bool) -> None:
    """Test torch.flip with various tensor shapes, dtypes, and dim combinations."""

    class FlipModel(nn.Module):
        def __init__(self, dims: list[int]) -> None:
            super().__init__()
            self.dims = dims

        def forward(self, x: Tensor) -> Tensor:
            return torch.flip(x, self.dims)

    model = FlipModel(dims).eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,y",
    [
        (torch.tensor([7.0, 8.0, 9.0]), torch.tensor([2.0, 3.0, 4.0])),  # 1D float
        (
            torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
            torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        ),  # 2D float
        (
            torch.tensor([7, 8, 9], dtype=torch.int32),
            torch.tensor([2, 3, 4], dtype=torch.int32),
        ),  # int32
        (
            torch.tensor([-7.0, -8.0, 9.0]),
            torch.tensor([2.0, 3.0, 4.0]),
        ),  # negative numerator
        (
            torch.tensor([7.0, -8.0, 9.0]),
            torch.tensor([-2.0, 3.0, -4.0]),
        ),  # mixed signs
        (
            torch.rand(3, 4) * 10 + 1,
            torch.rand(3, 4) * 3 + 1,
        ),  # 2D broadcast-compatible
    ],
)
async def test_floor_divide(x: Tensor, y: Tensor, dynamic: bool) -> None:
    """Test floor division (x // y) with various shapes, dtypes, and sign combinations."""

    class FloorDivideModel(nn.Module):
        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return x // y

    model = FloorDivideModel().eval()
    dynamic_shapes = (
        {"x": _all_dims_dynamic(x), "y": _all_dims_dynamic(y)} if dynamic else None
    )
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3) * 4 - 2,  # 2D: Range [-2, 2]
        torch.rand(3, 4, 5) * 6 - 3,  # 3D: Range [-3, 3]
        torch.rand(10) * 10 - 5,  # 1D: Range [-5, 5]
        torch.tensor(
            [0.0, 1.0, -1.0, 1.5, -1.5, 2.9, -2.9]
        ),  # Specific values including integers
        torch.tensor([[0.1, 0.9], [-0.1, -0.9]]),  # Values close to integers
    ],
)
async def test_floor(x: Tensor, dynamic: bool) -> None:
    """Test floor operation with various tensor shapes and values."""

    class FloorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.floor(x)

    model = FloorModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((2, 3), torch.float32),
        ((4, 5, 6), torch.float32),
        ((10,), torch.float32),
        ((3, 4), torch.float16),
        ((2, 2), torch.int32),
    ],
)
async def test_empty(shape: tuple[int, ...], dtype: torch.dtype, dynamic: bool) -> None:
    """Test empty tensor creation by filling it immediately after creation.

    Since torch.empty returns uninitialized data, we fill the tensor with
    a known value to enable deterministic comparison between PyTorch and Core AI.
    The model takes x as input so its shape can be marked dynamic.
    """

    class EmptyFillModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            empty_tensor = torch.empty(x.shape, dtype=dtype)
            return empty_tensor.fill_(1.0)

    x = torch.zeros(shape, dtype=dtype)
    model = EmptyFillModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestExpand:
    @pytest.mark.ir
    @pytest.mark.parametrize(
        "x,size",
        [
            # Basic expansion tests
            (torch.rand(2, 1, 3), (2, 4, 3)),  # expand middle dimension
            (torch.rand(1, 3), (5, 3)),  # expand first dimension
            (torch.rand(2, 1), (2, 4)),  # expand last dimension
            (torch.rand(1, 1, 1), (3, 4, 5)),  # expand all dimensions
            # Test with -1 (keep original size)
            (torch.rand(2, 3), (-1, 3)),  # keep first dimension
            (torch.rand(2, 3), (2, -1)),  # keep second dimension
            (torch.rand(2, 1, 4), (-1, 5, -1)),  # keep first and last dimensions
            # Test expanding to higher dimensions
            (torch.rand(3), (1, 3)),  # 1D to 2D
            (torch.rand(2, 3), (1, 2, 3)),  # 2D to 3D
            (torch.rand(4), (2, 1, 4)),  # 1D to 3D
        ],
    )
    def test_expand_ir(self, x: Tensor, size: tuple[int, ...]) -> None:
        class ExpandModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor) -> Tensor:
                return x.expand(size)

        model = ExpandModel().eval()
        program = torch.export.export(model, args=(x,)).run_decompositions()
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        pattern = "CHECK-NOT: ?"
        filecheck_pattern(str(coreai_program), pattern)

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x,size",
        [
            # Basic expansion tests
            (torch.rand(2, 1, 3), (2, 4, 3)),  # expand middle dimension
            (torch.rand(1, 3), (5, 3)),  # expand first dimension
            (torch.rand(2, 1), (2, 4)),  # expand last dimension
            (torch.rand(1, 1, 1), (3, 4, 5)),  # expand all dimensions
            # Test with -1 (keep original size)
            (torch.rand(2, 3), (-1, 3)),  # keep first dimension
            (torch.rand(2, 3), (2, -1)),  # keep second dimension
            (torch.rand(2, 1, 4), (-1, 5, -1)),  # keep first and last dimensions
            # Test expanding to higher dimensions
            (torch.rand(3), (1, 3)),  # 1D to 2D
            (torch.rand(2, 3), (1, 2, 3)),  # 2D to 3D
            (torch.rand(4), (2, 1, 4)),  # 1D to 3D
        ],
    )
    async def test_expand(
        self, x: Tensor, size: tuple[int, ...], dynamic: bool
    ) -> None:
        class ExpandModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor) -> Tensor:
                return x.expand(size)

        model = ExpandModel().eval()
        # A dim can only be dynamic if the expand target for that dim is -1 (keep existing size).
        # Static targets constrain x.size(i) to a constant, and broadcast dims (size==1) are
        # also always specialized by torch.export.
        if dynamic:
            offset = len(size) - x.dim()
            dim_map = {
                i: torch.export.Dim(f"d{i}", min=1)
                for i in range(x.dim())
                if x.size(i) != 1 and size[offset + i] == -1
            }
            dynamic_shapes = {"x": dim_map} if dim_map else None
        else:
            dynamic_shapes = None
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "x,size",
        [
            (torch.rand(2, 1, 3), (2, 4, 3)),  # expand middle dimension
            (torch.rand(1, 3), (5, 3)),  # expand first dimension
            (torch.rand(1, 1, 1), (3, 4, 5)),  # expand all dimensions
            (torch.rand(2, 1, 4), (-1, 5, -1)),  # keep first and last dimensions
        ],
    )
    def test_expand_uses_broadcast_in_dims(
        self, x: Tensor, size: tuple[int, ...]
    ) -> None:
        """Expand with statically size-1 broadcast dims should emit broadcast_in_dims."""

        class ExpandModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x.expand(size)

        model = ExpandModel().eval()
        program = torch.export.export(model, args=(x,)).run_decompositions()
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        ir = str(coreai_program)
        pattern = "CHECK: coreai.broadcast_in_dims\nCHECK-NOT: coreai.broadcast_to"
        filecheck_pattern(ir, pattern)

    @pytest.mark.ir
    def test_expand_dynamic_uses_broadcast_in_dims(self) -> None:
        """Expand with dynamic non-broadcast dims should still use broadcast_in_dims."""

        class ExpandModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.token = torch.nn.Parameter(torch.rand(1, 1, 8))

            def forward(self, x: Tensor) -> Tensor:
                # x.shape[0] is dynamic, token has size-1 dims at 0 and 1
                return self.token.expand(x.shape[0], -1, -1)

        batch = torch.export.Dim("batch", min=1)
        x = torch.rand(2, 4, 8)
        model = ExpandModel().eval()
        program = torch.export.export(
            model, args=(x,), dynamic_shapes={"x": {0: batch}}
        ).run_decompositions()
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        ir = str(coreai_program)
        pattern = "CHECK: coreai.broadcast_in_dims\nCHECK-NOT: coreai.broadcast_to"
        filecheck_pattern(ir, pattern)

    async def test_symbolic_batch(self) -> None:
        """Test expand where the batch dim is symbolic (fx.Node in the size list).

        Mirrors the pattern produced by ViTB16's class-token expansion:
          %expand = aten.expand(%token, [%sym_size_int, -1, -1])
        where the first element of the size list is a sym_size.int node rather
        than a static integer.
        """

        class ExpandSymbolicModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.token = torch.nn.Parameter(torch.rand(1, 1, 8))

            def forward(self, x: Tensor) -> Tensor:
                return self.token.expand(x.shape[0], -1, -1)

        batch = torch.export.Dim("batch", min=1)
        x = torch.rand(2, 4, 8)
        model = ExpandSymbolicModel().eval()
        await validate_numerical_output(
            model=model, x=x, dynamic_shapes={"x": {0: batch}}
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
async def test_fmod(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class FmodModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return torch.fmod(x, y)

    model = FmodModel().eval()
    if dynamic:
        shared_dims = _all_dims_dynamic(x)
        dynamic_shapes = {"x": shared_dims, "y": shared_dims}
    else:
        dynamic_shapes = None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        # Float tensors
        torch.rand(2, 3, dtype=torch.float32),
        torch.rand(4, 5, 6, dtype=torch.float32),
        torch.rand(10, dtype=torch.float32),
        # Integer tensors
        torch.randint(0, 10, (2, 3), dtype=torch.int32),
        torch.randint(0, 10, (2, 2), dtype=torch.int8),
        # Boolean tensors
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
    ],
)
@pytest.mark.parametrize("fill_value", [0.0, 3.0, -1.5])
async def test_full_like(x: Tensor, fill_value: float, dynamic: bool) -> None:
    class FullLikeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.full_like(x, fill_value)

    model = FullLikeModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "shape,fill_value,dtype",
    [
        # Different shapes with float32
        ((2, 3), 0.0, torch.float32),
        ((4, 5, 6), 3.14, torch.float32),
        ((10,), -1.5, torch.float32),
        # Different dtypes
        ((3, 4), 2.0, torch.float16),
        ((2, 2), 5, torch.int32),
        # Edge cases
        ((2,), 42.0, torch.float32),
        ((2, 2, 2, 2), 1.0, torch.float32),
    ],
)
async def test_full(
    shape: tuple[int, ...], fill_value: float | int, dtype: torch.dtype, dynamic: bool
) -> None:
    """Test torch.full operation which creates a tensor filled with a scalar value."""

    class FullModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.full(x.shape, fill_value, dtype=dtype)

    x = torch.zeros(shape, dtype=dtype)
    model = FullModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2), torch.rand(3, 4, 5)])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
async def test_gelu(x: Tensor, approximate: str, dynamic: bool) -> None:
    class GeluModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.nn.functional.gelu(x, approximate=approximate)

    model = GeluModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestGroupNorm:
    """Tests for native_group_norm → coreai group_norm composite."""

    @pytest.mark.parametrize("channels,groups", [(6, 2), (8, 4)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("affine", [True, False])
    @pytest.mark.parametrize(
        "spatial_dims,dynamic_dims",
        [
            # 3D (batch, C, W)
            ((8,), ()),
            ((8,), (0,)),
            ((8,), (2,)),
            ((8,), (0, 2)),
            # 4D (batch, C, H, W)
            ((8, 9), ()),
            ((8, 9), (0,)),
            ((8, 9), (2,)),
            ((8, 9), (3,)),
            ((8, 9), (0, 2)),
            ((8, 9), (2, 3)),
            # 5D (batch, C, D, H, W)
            ((8, 9, 10), ()),
            ((8, 9, 10), (0,)),
            ((8, 9, 10), (2, 4)),
            ((8, 9, 10), (3, 4)),
            ((8, 9, 10), (2, 3, 4)),
            ((8, 9, 10), (0, 2)),
        ],
    )
    async def test_basic(
        self,
        channels: int,
        groups: int,
        spatial_dims: tuple[int],
        dtype: torch.dtype,
        affine: bool,
        dynamic_dims: tuple[int],
    ) -> None:
        class GroupNormModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.group_norm = nn.GroupNorm(
                    num_groups=groups, num_channels=channels, affine=affine
                )

            def forward(self, x: Tensor) -> Tensor:
                return self.group_norm(x)

        x = torch.randn(2, channels, *spatial_dims, dtype=dtype)
        model = GroupNormModel().eval()
        dynamic_shapes = {"x": [torch.export.Dim.STATIC for _ in range(x.dim())]}
        for d in dynamic_dims:
            dynamic_shapes["x"][d] = torch.export.Dim.DYNAMIC

        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 2) * 4 - 2,  # Range [-2, 2] to test clipping
        torch.rand(3, 3) * 6 - 3,  # Range [-3, 3] to test more extreme clipping
        torch.rand(2, 5) * 0.5,  # Range [0, 0.5] to test values within bounds
        torch.tensor([-2.5, -1.0, 0.0, 1.0, 2.5]),  # Specific test values
    ],
)
async def test_hard_tanh(x: Tensor, dynamic: bool) -> None:
    """Test hard tanh activation function with various input ranges."""

    class HardTanhModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.nn.functional.hardtanh(x)

    model = HardTanhModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3) * 8 - 4,  # 2D: Range [-4, 4] to test all regions
        torch.tensor(
            [-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0]
        ),  # Specific values at boundaries
        torch.rand(3, 4, 5) * 10 - 5,  # 3D: Range [-5, 5] for wider coverage
        torch.rand(10) * 6 - 3,  # 1D: Range [-3, 3] hitting boundaries
        (torch.rand(2, 3) * 8 - 4).to(torch.float32),  # Explicit float32
        torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float32
        ),  # Positive values within linear region
        torch.tensor(
            [-2.5, -1.2, 0.0, 1.8, 2.9], dtype=torch.float32
        ),  # More float32 values in linear region
    ],
)
async def test_hardsigmoid(x: Tensor, dynamic: bool) -> None:
    """Test hardsigmoid activation function: 0 if x<=-3, 1 if x>=3, x/6+0.5 otherwise."""

    class HardsigmoidModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.nn.functional.hardsigmoid(x)

    model = HardsigmoidModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(
        model=model,
        x=x,
        dynamic_shapes=dynamic_shapes,
        remove_decomps=[torch.ops.aten.hardsigmoid.default],
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.randn(2, 3),  # 2D basic
        torch.randn(3, 4, 5),  # 3D
        torch.randn(2, 3, 4, 5),  # 4D
        torch.tensor([-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0]),  # Boundary values
        torch.tensor(
            [-2.5, -1.2, 0.0, 1.8, 2.9], dtype=torch.float32
        ),  # Values in linear region
    ],
)
async def test_hardswish(x: Tensor, dynamic: bool) -> None:
    """Test hardswish activation function: x * hardsigmoid(x)."""

    class HardswishModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.nn.functional.hardswish(x)

    model = HardswishModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(
        model=model,
        x=x,
        dynamic_shapes=dynamic_shapes,
        remove_decomps=[torch.ops.aten.hardswish.default],
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,negative_slope",
    [
        # Default negative_slope (0.01) with various shapes
        (torch.rand(2, 3) * 4 - 2, 0.01),  # 2D: Range [-2, 2]
        (torch.rand(3, 4, 5) * 6 - 3, 0.01),  # 3D: Range [-3, 3]
        (torch.rand(10) * 10 - 5, 0.01),  # 1D: Range [-5, 5]
        # Custom negative_slope values
        (torch.rand(2, 3) * 4 - 2, 0.1),  # Larger slope
        (torch.rand(2, 3) * 4 - 2, 0.2),  # Even larger slope
        # Edge cases with specific values
        (
            torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
            0.01,
        ),  # Specific values including zero
        (torch.tensor([0.0, -0.0, 0.5, -0.5]), 0.1),  # Values around zero
    ],
)
async def test_leaky_relu(x: Tensor, negative_slope: float, dynamic: bool) -> None:
    """Test leaky_relu activation: max(0,x) + negative_slope*min(0,x)."""

    class LeakyReluModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

    model = LeakyReluModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestLayerNorm:
    """Tests for native_layer_norm → coreai layer_norm composite."""

    @pytest.mark.parametrize(
        "x",
        [
            torch.rand(2, 4, 8),
            torch.rand(2, 3, 16, 16),
            torch.rand(2, 4, 8, dtype=torch.float16),
            torch.rand(2, 3, 16, 16, dtype=torch.float16),
        ],
    )
    @pytest.mark.parametrize("elementwise_affine", [True, False])
    @pytest.mark.parametrize("dynamic_dims", [tuple(), (0,)])
    async def test_basic(
        self, x: Tensor, elementwise_affine: bool, dynamic_dims: tuple[int]
    ) -> None:
        class LayerNormModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                normalized_shape = x.shape[1:]
                self.layer_norm = nn.LayerNorm(
                    normalized_shape, elementwise_affine=elementwise_affine
                )

            def forward(self, x: Tensor) -> Tensor:
                return self.layer_norm(x)

        model = LayerNormModel().eval()
        dim_names = {0: "batch", 1: "channels", 2: "height", 3: "width"}
        dynamic_shapes = make_dynamic_shapes(x={d: dim_names[d] for d in dynamic_dims})
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)

    @pytest.mark.parametrize(
        "x",
        [
            torch.rand(2, 4, 8),
            torch.rand(2, 3, 16, 16),
            torch.rand(2, 4, 8, dtype=torch.float16),
        ],
    )
    @pytest.mark.parametrize("elementwise_affine", [True, False])
    async def test_multi_dynamic_dims_with_legalization(
        self, x: Tensor, elementwise_affine: bool
    ) -> None:
        """All leading dims dynamic + Core AI legalization passes."""

        class LayerNormModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Normalize only the last dim so leading dims can be dynamic.
                self.layer_norm = nn.LayerNorm(
                    [x.shape[-1]], elementwise_affine=elementwise_affine
                )

            def forward(self, x: Tensor) -> Tensor:
                return self.layer_norm(x)

        model = LayerNormModel().eval()
        dynamic_dims = {i: f"d{i}" for i in range(x.dim() - 1)}
        await validate_numerical_output(
            model=model,
            x=x,
            dynamic_shapes=make_dynamic_shapes(x=dynamic_dims),
            run_optimize_passes=True,
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3),  # 2D float tensor
        torch.rand(4, 5, 6),  # 3D float tensor
        torch.rand(10),  # 1D float tensor
        torch.randint(0, 10, (3, 4), dtype=torch.int32),  # 2D int tensor
    ],
)
async def test_lift_fresh_copy(x: Tensor, dynamic: bool) -> None:
    """Test lift_fresh_copy operation which creates a fresh copy of the tensor."""

    class LiftFreshCopyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.ops.aten.lift_fresh_copy.default(x)

    model = LiftFreshCopyModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,col",
    [
        (torch.rand(6, 4), torch.tensor(0, dtype=torch.int32)),
        (torch.rand(6, 4), torch.tensor(2, dtype=torch.int32)),
    ],
)
async def test_local_scalar_dense(x: Tensor, col: Tensor, dynamic: bool) -> None:
    """_local_scalar_dense: 0-dim int tensor extracted via .item() and used as a dynamic slice index."""

    class SelectColModel(nn.Module):
        def forward(self, x: Tensor, col: Tensor) -> Tensor:
            c = col.item()
            torch._check_is_size(c)
            torch._check(
                c + 1 <= x.shape[1]
            )  # shape[1]=4 is static, provable at export time
            return x[:, c : c + 1]

    model = SelectColModel().eval()
    N = torch.export.Dim("N", min=1)
    dynamic_shapes = {"x": {0: N}, "col": {}} if dynamic else None
    await validate_numerical_output(
        model=model, x=x, col=col, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize(
    "x,log_op",
    [
        # log: positive values only (domain is (0, inf))
        (torch.rand(2, 3) + 0.1, torch.log),  # 2D tensor
        (torch.rand(3, 4, 5) + 0.1, torch.log),  # 3D tensor
        (torch.rand(10) + 0.1, torch.log),  # 1D tensor
        # log2: positive values only (domain is (0, inf))
        (torch.rand(2, 3) + 0.1, torch.log2),  # 2D tensor
        (torch.rand(3, 4, 5) + 0.1, torch.log2),  # 3D tensor
        (torch.rand(10) + 0.1, torch.log2),  # 1D tensor
        # log10: positive values only
        (torch.rand(2, 3) + 0.1, torch.log10),  # 2D tensor
        (torch.rand(3, 4, 5) + 0.1, torch.log10),  # 3D tensor
        (torch.rand(10) + 0.1, torch.log10),  # 1D tensor
        # log1p: domain is (-1, inf), so x > -1
        (torch.rand(2, 3) * 2 - 0.5, torch.log1p),  # 2D tensor, range [-0.5, 1.5]
        (torch.rand(3, 4, 5) * 2 - 0.5, torch.log1p),  # 3D tensor
        (torch.rand(10) * 2 - 0.5, torch.log1p),  # 1D tensor
    ],
)
@pytest.mark.parametrize(
    "dynamic_dims", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2), tuple()]
)
async def test_log_ops(x: Tensor, log_op: Any, dynamic_dims: tuple[int]) -> None:
    """Test logarithm operations: log2, log10, log1p."""
    if max(dynamic_dims + (0,)) >= len(x.shape):
        pytest.skip("Invalid dynamic_dims/shape configuration")

    class LogModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return log_op(x)

    model = LogModel().eval()
    dynamic_shapes = {"x": [None for _ in range(len(x.shape))]}
    for d in dynamic_dims:
        dynamic_shapes["x"][d] = torch.export.Dim(f"dim_{d}")

    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "x,dim,ord,keepdim",
    [
        # L0 norm (count non-zero elements) - test our fix
        (torch.rand(5), None, 0.0, False),
        (torch.rand(3, 4), 0, 0.0, True),
        # L1 norm (sum of absolute values)
        (torch.rand(3, 4), None, 1.0, False),
        (torch.rand(2, 3, 4), 1, 1.0, True),
        # L2 norm (Euclidean norm) - most common
        (torch.rand(5), 0, 2.0, False),
        (torch.rand(3, 4), [0, 1], 2.0, True),
        # General p-norm
        (torch.rand(2, 3, 4), 2, 3.0, False),
        # Infinity norms
        (torch.rand(3, 4), None, float("inf"), False),
        (torch.rand(2, 3, 4), [1, 2], -float("inf"), True),
        # Large order (treated as infinity norm)
        (torch.rand(5), None, 15.0, False),
        (torch.rand(2, 3, 4), 2, -3.0, False),
    ],
)
@pytest.mark.parametrize(
    "dynamic_dims", [tuple(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
)
async def test_linalg_vector_norm(
    x: Tensor,
    dim: int | list[int] | None,
    ord: float,
    keepdim: bool,
    dynamic_dims: tuple[int],
) -> None:
    if max(dynamic_dims + (0,)) >= len(x.shape):
        pytest.skip("Invalid dynamic_dims/shape configuration")

    class LinalgVectorNormModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.linalg.vector_norm(x, ord=ord, dim=dim, keepdim=keepdim)

    model = LinalgVectorNormModel().eval()
    dynamic_shapes = {"x": [torch.export.Dim.STATIC for _ in range(len(x.shape))]}
    for d in dynamic_dims:
        dynamic_shapes["x"][d] = torch.export.Dim(f"dim_{d}")
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([True, False, True, False], dtype=torch.bool),
        torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        torch.tensor([1, 0, 2, -1], dtype=torch.int32),
        torch.tensor([1.0, 0.0, 2.5, -1.5], dtype=torch.float32),
        torch.tensor([0, 1, 0, 1], dtype=torch.int8),
    ],
)
@pytest.mark.parametrize("dynamic", [True, False])
async def test_logical_not(x: Tensor, dynamic: bool) -> None:
    class LogicalNotModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.logical_not(x)

    dynamic_shapes = {"x": [torch.export.Dim.STATIC for _ in range(len(x.shape))]}
    if dynamic:
        # Only set dim 0 to dynamic as for most of our inputs, `x`has rank 1
        # For one of the inputs `x` has rank 2 and so this will test the case
        # where only one dimension is dynamic.
        dynamic_shapes["x"][0] = torch.export.Dim("batch", min=1)
    model = LogicalNotModel().eval()
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "x,y",
    [
        # Boolean tensors - basic case
        (
            torch.tensor([True, False, True, False], dtype=torch.bool),
            torch.tensor([True, True, False, False], dtype=torch.bool),
        ),
        # Boolean 2D tensors
        (
            torch.tensor([[True, False], [False, True]], dtype=torch.bool),
            torch.tensor([[True, True], [False, False]], dtype=torch.bool),
        ),
        # Integer tensors (non-zero is truthy)
        (
            torch.tensor([1, 0, 2, -1], dtype=torch.int32),
            torch.tensor([1, 1, 0, 0], dtype=torch.int32),
        ),
        # Float tensors (non-zero is truthy)
        (
            torch.tensor([1.0, 0.0, 2.5, -1.5], dtype=torch.float32),
            torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        ),
    ],
)
@pytest.mark.parametrize(
    "logical_op",
    [
        torch.logical_and,
        torch.logical_or,
        torch.logical_xor,
    ],
)
@pytest.mark.parametrize("dynamic", [True, False])
async def test_binary_logical_ops(
    x: Tensor, y: Tensor, logical_op: Any, dynamic: bool
) -> None:
    """Test binary logical operations: logical_and, logical_or, logical_xor."""

    class BinaryLogicalModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return logical_op(x, y)

    model = BinaryLogicalModel().eval()

    # For n-ary ops, dynamic dimensions must match on each operand. Otherwise torch.export.export will fail
    # as it determines that a dimension you have marked dynamic must actually be the exact size of the dim
    # in the other operand (unless its being broadcasted).
    #
    # i.e (1x?) + (1x4)
    # Torch export sees you have marked dim 1 of the left-hand-side as dynamic, but concludes that the dim MUST
    # be size 4, and so you were wrong to mark it dynamic.
    #
    # So, we go with an all-dynamic or all-static approach for this test.

    dyn_dims = {
        0: torch.export.Dim("batch", min=1),
        1: torch.export.Dim("features", min=1),
        2: torch.export.Dim("spatial", min=1),
    }
    dynamic_shapes = {
        "x": [
            torch.export.Dim.STATIC if not dynamic else dyn_dims[i]
            for i in range(len(x.shape))
        ],
        "y": [
            torch.export.Dim.STATIC if not dynamic else dyn_dims[i]
            for i in range(len(y.shape))
        ],
    }

    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
async def test_maximum(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class MaximumModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return torch.maximum(x, y)

    model = MaximumModel().eval()
    if dynamic:
        shared_dims = _all_dims_dynamic(x)
        dynamic_shapes = {"x": shared_dims, "y": shared_dims}
    else:
        dynamic_shapes = None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize(
    "input_shape, dtype, kernel_size, stride, padding, dilation, ceil_mode, dynamic_dims",
    [
        # Static — all pool configs, multiple shapes and dtypes
        ((2, 4, 16, 16), torch.float32, 3, 2, 1, 1, False, tuple()),
        ((2, 4, 16, 16), torch.float32, 3, 2, 0, 1, True, tuple()),
        ((2, 4, 16, 16), torch.float32, 3, 2, 1, 1, True, tuple()),
        ((2, 4, 16, 16), torch.float32, 3, 1, 0, 2, False, tuple()),
        ((2, 4, 16, 16), torch.float32, 3, 1, 1, 2, False, tuple()),
        ((2, 4, 16, 16), torch.float32, 3, 2, 0, 2, True, tuple()),
        ((1, 4, 14, 20), torch.float32, 3, 2, 1, 1, False, tuple()),  # non-square
        ((2, 4, 16, 16), torch.float16, 3, 2, 1, 1, False, tuple()),  # float16
        (
            (2, 4, 16, 16),
            torch.float16,
            3,
            1,
            0,
            2,
            False,
            tuple(),
        ),  # float16 + dilation
        # Dynamic batch — all pool configs
        ((2, 4, 16, 16), torch.float32, 3, 2, 1, 1, False, (0,)),
        ((2, 4, 16, 16), torch.float32, 3, 2, 0, 1, True, (0,)),
        ((2, 4, 16, 16), torch.float32, 3, 2, 1, 1, True, (0,)),
        ((2, 4, 16, 16), torch.float32, 3, 1, 0, 2, False, (0,)),
        ((2, 4, 16, 16), torch.float32, 3, 1, 1, 2, False, (0,)),
        ((2, 4, 16, 16), torch.float32, 3, 2, 0, 2, True, (0,)),
        # Dynamic batch + channel
        ((2, 4, 16, 16), torch.float32, 3, 2, 1, 1, False, (0, 1)),
        ((2, 4, 16, 16), torch.float16, 3, 1, 0, 2, False, (0, 1)),
        # Dynamic spatial dims (stride=1 only — stride > 1 causes parity guards in
        # torch.export that cannot hold for an unbounded dynamic range)
        ((2, 4, 16, 16), torch.float32, 3, 1, 0, 2, False, (2,)),
        ((2, 4, 16, 16), torch.float32, 3, 1, 0, 2, False, (3,)),
        ((2, 4, 16, 16), torch.float32, 3, 1, 0, 2, False, (2, 3)),
        ((2, 3, 14, 20), torch.float32, 3, 1, 0, 2, False, (2, 3)),  # non-square
        ((2, 4, 16, 16), torch.float16, 3, 1, 1, 2, False, (2, 3)),
        # All dims dynamic (stride=1 only)
        ((2, 4, 16, 16), torch.float32, 3, 1, 0, 2, False, (0, 1, 2, 3)),
        ((2, 4, 16, 16), torch.float16, 3, 1, 1, 2, False, (0, 1, 2, 3)),
    ],
)
async def test_maxpool2d(
    input_shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool,
    dynamic_dims: tuple[int, ...],
) -> None:
    x = torch.rand(*input_shape, dtype=dtype)

    class MaxPool2dModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.nn.functional.max_pool2d(
                x,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            )

    model = MaxPool2dModel().eval()
    if not dynamic_dims:
        dynamic_shapes = None
    else:
        dim_names = {0: "batch", 1: "channels", 2: "height", 3: "width"}
        dynamic_shapes = make_dynamic_shapes(x={d: dim_names[d] for d in dynamic_dims})
        # Spatial dims need a higher min so output_h/w >= 2 for all H/W in the
        # dynamic range, satisfying torch.export's output-validity guards.
        min_spatial = max(2, dilation * (kernel_size - 1) + 2 - 2 * padding)
        for d in set(dynamic_dims) & {2, 3}:
            dynamic_shapes["x"][d] = torch.export.Dim(dim_names[d], min=min_spatial)
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "x,dim,keepdim",
    [
        # mean.default: global mean (dim=None)
        (torch.rand(2, 3), None, False),
        (torch.rand(3, 4, 5), None, False),
        (torch.rand(10), None, False),
        # mean.dim: reduce along specific dimensions with keepdim variations
        (torch.rand(2, 5, 4), 1, True),
        (torch.rand(2, 5, 4), 1, False),
        (torch.rand(3, 4, 5), [0, 2], True),
        (torch.rand(3, 4, 5), [0, 2], False),
        (torch.rand(4, 6), -1, True),
        (torch.rand(4, 6), -1, False),
    ],
)
@pytest.mark.parametrize(
    "dynamic_dims", [tuple(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
)
async def test_mean(
    x: Tensor, dim: int | list[int] | None, keepdim: bool, dynamic_dims: tuple[int]
) -> None:
    """Test mean operation with global mean (dim=None) and dim-specific reduction."""
    if max(dynamic_dims + (0,)) >= len(x.shape):
        pytest.skip("Invalid dynamic_dims/shape configuration")

    class MeanModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            if dim is None:
                return torch.mean(x)
            return torch.mean(x, dim=dim, keepdim=keepdim)

    model = MeanModel().eval()
    dynamic_shapes = {"x": [torch.export.Dim.STATIC for _ in range(len(x.shape))]}
    for d in dynamic_dims:
        dynamic_shapes["x"][d] = torch.export.Dim(f"dim_{d}")
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
async def test_minimum(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class MinimumModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return torch.minimum(x, y)

    model = MinimumModel().eval()
    if dynamic:
        shared_dims = _all_dims_dynamic(x)
        dynamic_shapes = {"x": shared_dims, "y": shared_dims}
    else:
        dynamic_shapes = None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3),  # 2D float tensor
        torch.rand(3, 4, 5),  # 3D float tensor
        torch.rand(10),  # 1D float tensor
        torch.rand(2, 3, 4, 5),  # 4D float tensor
        torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),  # 2D with mixed signs
        torch.tensor([0.5]),  # Single element tensor
    ],
)
async def test_min(x: Tensor, dynamic: bool) -> None:
    """Test global min reduction across all dimensions."""

    class MinModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.min(x)

    model = MinModel().eval()
    if dynamic:
        dim_map = {
            i: torch.export.Dim(f"d{i}", min=1) for i in range(x.dim()) if x.size(i) > 1
        }
        dynamic_shapes = {"x": dim_map} if dim_map else None
    else:
        dynamic_shapes = None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "shape,dim,keepdim,dtype",
    [
        # min.dim: reduce along specific dimensions with keepdim variations
        ((2, 5, 4), 1, True, torch.float32),
        ((2, 5, 4), 1, False, torch.float32),
        ((3, 4, 5), 0, True, torch.float32),
        ((3, 4, 5), -1, False, torch.float32),  # negative dim
        ((4, 6), 0, True, torch.float16),
        ((4, 6), 0, False, torch.float16),
    ],
)
async def test_min_dim(
    shape: tuple[int, ...], dim: int, keepdim: bool, dtype: torch.dtype, dynamic: bool
) -> None:
    """Test min.dim operation which returns (min_values, argmin_indices)."""
    x = torch.rand(shape, dtype=dtype)

    class MinDimModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return torch.min(x, dim=dim, keepdim=keepdim)

    model = MinDimModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "shape,dim,keepdim,dtype",
    [
        # max.default: global max (dim=None)
        ((3, 4, 5), None, False, torch.float32),
        ((3, 4, 5), None, False, torch.float16),
        # max.dim: reduce along specific dimensions with keepdim variations
        ((2, 5, 4), 1, True, torch.float32),
        ((2, 5, 4), 1, False, torch.float32),
        ((3, 4, 5), -1, False, torch.float16),  # negative dim
        ((4, 6), 0, True, torch.float16),
    ],
)
async def test_max(
    shape: tuple[int, ...],
    dim: int | None,
    keepdim: bool,
    dtype: torch.dtype,
    dynamic: bool,
) -> None:
    """Test max operation with global max (dim=None) and dim-specific reduction."""
    x = torch.rand(shape, dtype=dtype)
    model: Any = None
    if dim is None:

        class MaxModelGlobal(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor) -> Tensor:
                return torch.max(x)

        model = MaxModelGlobal().eval()
        dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)
    else:

        class MaxModelDim(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                return torch.max(x, dim=dim, keepdim=keepdim)

        model = MaxModelDim().eval()
        dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "mat1,mat2",
    [
        # Basic 2D matrix multiplication
        (torch.rand(2, 3), torch.rand(3, 4)),  # (2x3) @ (3x4) -> (2x4)
        (torch.rand(4, 5), torch.rand(5, 2)),  # (4x5) @ (5x2) -> (4x2)
        # Square matrices
        (torch.rand(3, 3), torch.rand(3, 3)),  # Square matrix multiplication
        # Different sizes
        (
            torch.rand(1, 10),
            torch.rand(10, 1),
        ),  # Vector-like matrices -> scalar-like result
        (torch.rand(8, 16), torch.rand(16, 4)),  # Larger matrices
    ],
)
async def test_mm(mat1: Tensor, mat2: Tensor, dynamic: bool) -> None:
    """Test 2D matrix multiplication operation."""

    class MmModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, mat1: Tensor, mat2: Tensor) -> Tensor:
            return torch.mm(mat1, mat2)

    model = MmModel().eval()
    # K (inner dim) must be shared; M and N are independent but skip if size == 1.
    if dynamic:
        K = torch.export.Dim("K", min=1)
        mat1_dims: dict[int, torch.export.Dim] = {1: K}
        mat2_dims: dict[int, torch.export.Dim] = {0: K}
        if mat1.size(0) > 1:
            mat1_dims[0] = torch.export.Dim("M", min=1)
        if mat2.size(1) > 1:
            mat2_dims[1] = torch.export.Dim("N", min=1)
        dynamic_shapes = {"mat1": mat1_dims, "mat2": mat2_dims}
    else:
        dynamic_shapes = None
    await validate_numerical_output(
        model=model, mat1=mat1, mat2=mat2, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
async def test_mod(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class ModModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return x % y

    model = ModModel().eval()
    if dynamic:
        shared_dims = _all_dims_dynamic(x)
        dynamic_shapes = {"x": shared_dims, "y": shared_dims}
    else:
        dynamic_shapes = None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
async def test_mul(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class MulModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return x * y

    model = MulModel().eval()
    if dynamic:
        shared_dims = _all_dims_dynamic(x)
        dynamic_shapes = {"x": shared_dims, "y": shared_dims}
    else:
        dynamic_shapes = None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3) * 4 - 2,  # 2D: Range [-2, 2]
        torch.rand(3, 4, 5) * 6 - 3,  # 3D: Range [-3, 3]
        torch.rand(10) * 10 - 5,  # 1D: Range [-5, 5]
        torch.tensor(
            [0.0, -0.0, 1.0, -1.0, 2.5, -2.5]
        ),  # Specific values including zero
        torch.tensor([[-1.5, 2.0], [3.0, -4.0]]),  # 2D with mixed signs
        torch.tensor([-5, -2, 0, 3, 7], dtype=torch.int32),  # int32
    ],
)
async def test_neg(x: Tensor, dynamic: bool) -> None:
    """Test negation operation which computes -x element-wise."""

    class NegModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.neg(x)

    model = NegModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        # 1D float32: mixed zeros and non-zeros
        torch.tensor([0.0, 1.0, 0.0, -2.0, 0.0, 3.0], dtype=torch.float32),
        # 1D bool
        torch.tensor([True, False, True, False, True], dtype=torch.bool),
        # 2D float32
        torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=torch.float32),
        # 2D int32
        torch.tensor([[0, 1, 0], [2, 0, 3]], dtype=torch.int32),
        # 2D bool
        torch.tensor([[True, False], [False, True], [True, True]], dtype=torch.bool),
        # 3D float32
        torch.tensor(
            [[[1.0, 0.0], [0.0, 2.0]], [[0.0, 3.0], [4.0, 0.0]]], dtype=torch.float32
        ),
        # All zeros — output is empty [0, ndim]
        torch.zeros(3, 4, dtype=torch.float32),
        # All non-zero
        torch.ones(2, 3, dtype=torch.float32),
    ],
)
async def test_nonzero(x: Tensor, dynamic: bool) -> None:
    """Test aten.nonzero — returns indices of all non-zero elements as [N, rank] int64."""

    class NonZeroModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.nonzero(x)

    model = NonZeroModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([0.0, 1.0, 0.0, -2.0, 0.0, 3.0], dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=torch.float32),
        torch.tensor([[True, False], [False, True], [True, True]], dtype=torch.bool),
        torch.zeros(3, 4, dtype=torch.float32),
        torch.ones(2, 3, dtype=torch.float32),
    ],
)
async def test_nonzero_as_tuple(x: Tensor, dynamic: bool) -> None:
    """Test aten.nonzero_numpy — as_tuple=True returns one 1D index tensor per dimension."""

    class NonZeroAsTupleModel(nn.Module):
        def forward(self, x: Tensor) -> tuple[Tensor, ...]:
            return torch.nonzero(x, as_tuple=True)

    model = NonZeroAsTupleModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
async def test_ne_scalar(x: Tensor, dynamic: bool) -> None:
    class NeScalarModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.ne(x, 0.5)

    model = NeScalarModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3) + 0.1,  # 2D: Range [0.1, 1.1] - avoid division by zero
        torch.rand(3, 4, 5) + 0.5,  # 3D: Range [0.5, 1.5]
        torch.rand(10) * 2 + 0.5,  # 1D: Range [0.5, 2.5]
        torch.tensor([0.5, 1.0, 2.0, 4.0, 10.0]),  # Specific positive values
        torch.tensor([[-2.0, -1.0], [1.0, 2.0]]),  # 2D with mixed signs (non-zero)
    ],
)
async def test_reciprocal(x: Tensor, dynamic: bool) -> None:
    """Test reciprocal operation which computes 1/x element-wise."""

    class ReciprocalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.reciprocal(x)

    model = ReciprocalModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
async def test_relu(x: Tensor, dynamic: bool) -> None:
    class Relu(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.nn.functional.relu(x)

    model = Relu().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
async def test_remainder(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class RemainderModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return torch.remainder(x, y)

    model = RemainderModel().eval()
    if dynamic:
        shared_dims = _all_dims_dynamic(x)
        dynamic_shapes = {"x": shared_dims, "y": shared_dims}
    else:
        dynamic_shapes = None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,repeats",
    [
        # Basic 2D tensor repeats
        (torch.rand(2, 3), (2, 2)),  # repeat 2x in each dim
        (torch.rand(2, 3), (1, 3)),  # repeat only along dim 1
        (torch.rand(2, 3), (3, 1)),  # repeat only along dim 0
        # 1D tensor repeats
        (torch.rand(4), (3,)),  # 1D with single repeat
        (torch.rand(5), (2, 3)),  # 1D with extra dimension
        # 3D tensor repeats
        (torch.rand(2, 3, 4), (2, 1, 2)),  # 3D with mixed repeats
        # Repeats with more dims than input (adds leading dimensions)
        (torch.rand(3, 4), (2, 2, 3)),  # 2D input with 3 repeats
    ],
)
async def test_repeat(x: Tensor, repeats: tuple[int, ...], dynamic: bool) -> None:
    """Test repeat operation which replicates tensor along dimensions."""

    class RepeatModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return x.repeat(*repeats)

    model = RepeatModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestRepeat:
    @pytest.mark.ir
    def test_symint_arg_lowers_ir(self) -> None:
        """``aten.repeat`` with a SymInt entry in the repeats list (i.e. a
        ``torch.fx.Node``, not a plain int) must lower to a dynamic
        ``coreai.tile`` whose dim vector is built at runtime."""

        class RepeatModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x.repeat(y.shape[0], 1)

        x = torch.rand(2, 3)
        y = torch.rand(4, 8)
        batch = torch.export.Dim("batch", min=1, max=16)
        program = torch.export.export(
            RepeatModel(), args=(x, y), dynamic_shapes=({}, {0: batch})
        ).run_decompositions()

        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK-LABEL: coreai.graph @main
                // CHECK-SAME:    %arg0: tensor<2x3xf32>
                // CHECK-SAME:    %arg1: tensor<?x8xf32>
                // CHECK:         %[[SHAPE:.+]] = coreai.get_shape %arg1 : tensor<?x8xf32> -> tensor<2xui32>
                // CHECK:         %[[SLICE:.+]] = coreai.slice %[[SHAPE]]
                // CHECK-SAME:      -> tensor<1xui32>
                // CHECK:         %[[ONE:.+]] = coreai.constant dense<1> : tensor<1xui32>
                // CHECK:         %[[DIMS:.+]] = coreai.concat {{.*}}, %{{.+}}, %[[ONE]]
                // CHECK-SAME:      -> tensor<2xui32>
                // CHECK:         %[[OUT:.+]] = coreai.tile %arg0, %[[DIMS]]
                // CHECK:         coreai.output %[[OUT]]
            """,
        )

    async def test_symint_arg_numerical(self) -> None:
        """Numerical validation: ``aten.repeat`` with a SymInt entry in the
        repeats list must produce the same result as ``torch.repeat``."""

        class RepeatModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return x.repeat(y.shape[0], 1)

        x = torch.rand(2, 3)
        y = torch.rand(4, 8)
        batch = torch.export.Dim("batch", min=1, max=16)
        await validate_numerical_output(
            model=RepeatModel().eval(),
            x=x,
            y=y,
            dynamic_shapes=({}, {0: batch}),
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3) * 4 - 2,  # 2D: Range [-2, 2]
        torch.rand(3, 4, 5) * 6 - 3,  # 3D: Range [-3, 3]
        torch.rand(10) * 10 - 5,  # 1D: Range [-5, 5]
        torch.tensor(
            [0.0, 1.0, -1.0, 1.5, -1.5, 2.9, -2.9]
        ),  # Specific values including integers
        torch.tensor([[0.1, 0.9], [-0.1, -0.9]]),  # Values close to integers
        torch.tensor(
            [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        ),  # Banker's rounding: halfway values
        torch.tensor(
            [[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]]
        ),  # 2D banker's rounding test
    ],
)
async def test_round(x: Tensor, dynamic: bool) -> None:
    """Test round operation which rounds to nearest integer using banker's rounding."""

    class RoundModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.round(x)

    model = RoundModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "decimals,x",
    [
        (2, torch.tensor([1.2345, -2.6789, 0.005], dtype=torch.float32)),
        (0, torch.tensor([1.5, -2.5, 0.5], dtype=torch.float32)),
        (-1, torch.tensor([123.4, 456.7, -78.9], dtype=torch.float32)),
        (-2, torch.tensor([1234.5, -6789.0], dtype=torch.float32)),
        (3, torch.tensor([[1.2345, 0.0005], [-9.9999, 3.1415]], dtype=torch.float32)),
    ],
)
async def test_round_decimals(x: Tensor, decimals: int, dynamic: bool) -> None:
    """Test aten.round.decimals — rounds to a given number of decimal places."""

    class RoundDecimalsModel(nn.Module):
        def __init__(self, decimals: int) -> None:
            super().__init__()
            self.decimals = decimals

        def forward(self, x: Tensor) -> Tensor:
            return torch.round(x, decimals=self.decimals)

    model = RoundDecimalsModel(decimals).eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(2, 3) + 0.1,  # 2D: Range [0.1, 1.1] - avoid division by zero
        torch.rand(3, 4, 5) + 0.5,  # 3D: Range [0.5, 1.5]
        torch.rand(10) * 2 + 0.5,  # 1D: Range [0.5, 2.5]
        torch.tensor(
            [0.25, 0.5, 1.0, 4.0, 9.0, 16.0]
        ),  # Specific values with known sqrt results
        torch.tensor([[1.0, 4.0], [9.0, 16.0]]),  # 2D with perfect squares
    ],
)
async def test_rsqrt(x: Tensor, dynamic: bool) -> None:
    """Test rsqrt operation which computes 1/sqrt(x) element-wise."""

    class RsqrtModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.rsqrt(x)

    model = RsqrtModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dims",
    [
        # 2D: basic transpose
        (torch.rand(2, 2), [1, 0]),
        # 3D
        (torch.rand(2, 3, 4), [2, 1, 0]),
        (torch.rand(2, 3, 4), [1, 2, 0]),
        # 4D: NCHW → NHWC
        (torch.rand(2, 3, 4, 5), [0, 2, 3, 1]),
        (torch.rand(2, 3, 4, 5), [3, 2, 1, 0]),
        # fp16
        (torch.rand(2, 3, 4, dtype=torch.float16), [2, 0, 1]),
    ],
)
async def test_permute(x: Tensor, dims: list[int], dynamic: bool) -> None:
    class PermuteModel(nn.Module):
        def __init__(self, dims: list[int]) -> None:
            super().__init__()
            self.dims = dims

        def forward(self, x: Tensor) -> Tensor:
            return torch.permute(x, self.dims)

    model = PermuteModel(dims).eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
async def test_pow(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class PowModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return torch.pow(x, y)

    model = PowModel().eval()
    dynamic_shapes = (
        {"x": _all_dims_dynamic(x), "y": _all_dims_dynamic(y)} if dynamic else None
    )
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
async def test_pow_constant(x: Tensor, dynamic: bool) -> None:
    class PowModelConstant(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.pow(x, 2.0)

    model = PowModelConstant().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestRound:
    """Regression suite for the bare ``aten.round`` OpOverloadPacket target.
    Same shape of bug as the pow case: torch.export rewrites can leave
    ``aten.round`` without a ``.default`` overload, which previously raised
    ``Unsupported ATen op: round`` / ``KeyError: 'round'``."""

    @staticmethod
    def _rewrite_to_overload_packet(program: object) -> None:
        for node in program.graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if getattr(node.target, "_overloadpacket", None) is torch.ops.aten.round:
                node.target = torch.ops.aten.round
        program.graph_module.recompile()

    def test_lowers_ir(self) -> None:
        class RoundModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.round(x)

        x = torch.tensor([[-1.5, 2.0], [3.0, -4.0]])
        program = torch.export.export(RoundModel(), args=(x,)).run_decompositions()
        self._rewrite_to_overload_packet(program)

        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        coreai_program.optimize()
        # The bare ``aten.round`` target must reach ``coreai.round`` —
        # not produce a different op or fail with ``Unsupported ATen op``.
        # Shape and element type must pass through unchanged.
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK-LABEL: coreai.graph @main
                // CHECK-SAME:    %[[ARG0:.*]]: tensor<2x2xf32>
                // CHECK-SAME:    -> (tensor<2x2xf32>
                // CHECK:         %[[OUT:.+]] = coreai.round %[[ARG0]] : tensor<2x2xf32> -> tensor<2x2xf32>
                // CHECK:         coreai.output %[[OUT]] : tensor<2x2xf32>
            """,
        )

    async def test_numerical(self) -> None:
        class RoundModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.round(x)

        x = torch.tensor([[-1.5, 2.0, 0.4], [3.6, -4.5, -0.5]])
        program = torch.export.export(RoundModel(), args=(x,)).run_decompositions()
        self._rewrite_to_overload_packet(program)

        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        torch_out = RoundModel().eval()(x)
        await validate_numerical_output(
            coreai_program=coreai_program, torch_out=torch_out, x=x
        )


class TestView:
    """Regression suite for ``get_operand``'s mixed-list path: a list arg
    that mixes ``fx.Node`` SymInt entries with plain ints (e.g. a view
    shape like ``[1, 1024, h, w]`` where h, w come from ``round(SymFloat)``)
    must produce concat operands with a uniform rank-1 int32 element type
    — otherwise the dialect rejects the dim-vector concat with ``expected
    the same element type for all inputs to concat``."""

    def test_view_with_round_symfloat_dims(self) -> None:
        """View shape mixes plain ints with SymInts that come from
        ``round(sym_float)`` shape arithmetic. Without the fix,
        ``coreai.concat`` rejects the mixed-element-type operands when
        building the dim vector for the reshape."""

        class ViewModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                # round(SymFloat) → SymInt: produces a Value whose type
                # doesn't match the int32 constants used for plain-int
                # entries in the same dim list.
                aspect = x.shape[3] / x.shape[2]
                h = round((1800 / aspect) ** 0.5)
                w = round((1800 * aspect) ** 0.5)
                # ``view([1, C, round_h, round_w])`` — the mixed-list path
                # in ``get_operand`` builds a dim-vector concat that
                # combines plain-int constants with SymInt Values whose
                # element type the fix normalises.
                flat = x.view(1, 1024, -1)
                return flat.view(1, 1024, h, w)

        x = torch.rand(1, 1024, 42, 42, dtype=torch.float16)
        dynamic_shapes = {
            "x": {
                2: torch.export.Dim("h", min=14),
                3: torch.export.Dim("w", min=14),
            }
        }
        program = torch.export.export(
            ViewModel().eval(), args=(x,), dynamic_shapes=dynamic_shapes
        ).run_decompositions()
        # Convert to Core AI — must not raise. Pre-fix this raised
        # ``ValueError: Operation creation failed`` from ``coreai.concat``
        # with the diagnostic ``expected the same element type for all
        # inputs to concat``.
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        coreai_program.optimize()
        # The fix's job is to normalise every entry of the view shape
        # ``[1, 1024, h, w]`` to the same canonical rank-1 si32 form
        # before the dim-vector concat. The check below pins:
        #   1. The two SymInt entries (h, w) carry rank-1 f32 from the
        #      round(SymFloat) chain and get explicitly cast to rank-1
        #      si32 (matching the int constants).
        #   2. All four operands of the dim-vector concat are rank-1
        #      si32, so the concat verifier accepts them.
        #   3. The resulting rank-1 si32 vector of length 4 feeds the
        #      reshape that builds the final 4-D output.
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK-LABEL: coreai.graph @main
                // CHECK-SAME:    %[[ARG0:.*]]: tensor<1x1024x?x?xf16>
                // CHECK-SAME:    -> (tensor<1x1024x?x?xf16>
                //
                // h and w land as rank-1 f32 from round(SymFloat) arithmetic;
                // the fix casts each to rank-1 si32 so it matches the int
                // constants in the same dim vector:
                // CHECK:         %[[H:.+]] = coreai.cast {{.*}} : tensor<1xf32> to tensor<1xsi32>
                // CHECK:         %[[W:.+]] = coreai.cast {{.*}} : tensor<1xf32> to tensor<1xsi32>
                //
                // Dim-vector concat: 4 entries [1, 1024, h, w] → all rank-1 si32:
                // CHECK:         %[[SHAPE:.+]] = coreai.concat {{.*}} : (tensor<si32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<4xsi32>
                //
                // Final reshape uses that shape vector to produce the 4-D output.
                // (After optimize() the inner ``view(1, 1024, -1)`` is fused
                // away, so the reshape applies directly to %arg0.):
                // CHECK:         %[[OUT:.+]] = coreai.reshape %[[ARG0]], %[[SHAPE]] : (tensor<1x1024x?x?xf16>, tensor<4xsi32>) -> tensor<1x1024x?x?xf16>
                // CHECK:         coreai.output %[[OUT]]
            """,
        )


_SCALAR_PARAMS = pytest.mark.parametrize(
    "scalar,tensor_shape,dtype",
    [
        (2.0, (2, 3), torch.float32),
        (3.0, (2, 3, 4), torch.float32),
        (2.5, (5,), torch.float32),
        (2, (3, 4), torch.float32),
        (2.0, (2, 3), torch.float16),
        (torch.tensor(2.0, dtype=torch.float32), (2, 3), torch.float16),
        (torch.tensor(2.0, dtype=torch.float32), 1, torch.float16),
        (0.5, (2, 2), torch.float32),
    ],
)


@_SCALAR_PARAMS
async def test_pow_scalar(
    scalar: int | float | Tensor, tensor_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    """Test pow.Scalar operation which computes scalar^tensor.

    aten.pow.Scalar(self, exponent) computes self^exponent where self is a
    scalar base and exponent is a tensor.
    """
    exponent = (
        torch.rand(tensor_shape, dtype=dtype) * 2 + 0.5
    )  # Range [0.5, 2.5] for numerical stability

    class PowScalarModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, exponent: Tensor) -> Tensor:
            return torch.pow(scalar, exponent)

    model = PowScalarModel().eval()
    await validate_numerical_output(model=model, exponent=exponent)


@_SCALAR_PARAMS
async def test_add_scalar(
    scalar: int | float | Tensor, tensor_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    x = torch.rand(tensor_shape, dtype=dtype)

    class Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x + scalar

    await validate_numerical_output(model=Model().eval(), x=x)


class TestFp16ScalarCasting:
    """Tests for fp16 scalar promotion: float scalars are cast to fp16 when all
    float tensor operands of the node are fp16."""

    @pytest.mark.parametrize(
        ("x", "scalar"),
        [
            (torch.rand(3, 2), 0.5),
            (torch.rand(1, 3, 2), 2.0),
            (torch.rand(2, 4), 0.25),
        ],
    )
    async def test_add(self, x: Tensor, scalar: float) -> None:
        """Test add.Scalar with fp16 tensor and float scalar."""

        class Model(nn.Module):
            def __init__(self, s: float) -> None:
                super().__init__()
                self.s = s

            def forward(self, x: Tensor) -> Tensor:
                return x + self.s

        await validate_numerical_output(
            model=Model(scalar).eval(),
            x=x.to(torch.float16),
        )

    @pytest.mark.parametrize(
        ("x", "scalar"),
        [
            (torch.rand(3, 2), 0.5),
            (torch.rand(1, 3, 2), 2.0),
        ],
    )
    async def test_mul(self, x: Tensor, scalar: float) -> None:
        """Test mul.Scalar with fp16 tensor and float scalar."""

        class Model(nn.Module):
            def __init__(self, s: float) -> None:
                super().__init__()
                self.s = s

            def forward(self, x: Tensor) -> Tensor:
                return x * self.s

        await validate_numerical_output(
            model=Model(scalar).eval(),
            x=x.to(torch.float16),
        )

    @pytest.mark.ir
    def test_ir(self) -> None:
        """Check that a float scalar is emitted as tensor<f16> when all tensor operands are fp16."""

        class Model(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x * 0.5

        x = torch.rand(3, 2, dtype=torch.float16)
        exported = torch.export.export(Model().eval(), args=(x,)).run_decompositions()
        converter = TorchConverter()
        converter.add_exported_program(exported)
        coreai_program = converter.to_coreai()

        check_file = """
        // CHECK: coreai.constant dense<5.000000e-01> : tensor<f16>
        """
        filecheck_pattern(str(coreai_program), check_file=check_file)


@_SCALAR_PARAMS
async def test_sub_scalar(
    scalar: int | float | Tensor, tensor_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    x = torch.rand(tensor_shape, dtype=dtype)

    class Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x - scalar

    await validate_numerical_output(model=Model().eval(), x=x)


@_SCALAR_PARAMS
async def test_mul_scalar(
    scalar: int | float | Tensor, tensor_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    x = torch.rand(tensor_shape, dtype=dtype)

    class Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x * scalar

    await validate_numerical_output(model=Model().eval(), x=x)


@_SCALAR_PARAMS
async def test_div_scalar(
    scalar: int | float | Tensor, tensor_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    x = torch.rand(tensor_shape, dtype=dtype)

    class Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x / scalar

    await validate_numerical_output(model=Model().eval(), x=x)


@_SCALAR_PARAMS
async def test_fmod_scalar(
    scalar: int | float | Tensor, tensor_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    x = (
        torch.rand(tensor_shape, dtype=dtype) + 0.5
    )  # avoid near-zero for fmod stability

    class Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.fmod(x, scalar)

    await validate_numerical_output(model=Model().eval(), x=x)


@pytest.mark.parametrize(
    "scalar,tensor_shape,dtype",
    [
        (2.0, (2, 3), torch.float32),
        (3.0, (2, 3, 4), torch.float32),
        (-1.0, (5,), torch.float32),
        (2.0, (2, 3), torch.float16),
    ],
)
async def test_maximum_scalar(
    scalar: float, tensor_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    x = torch.rand(tensor_shape, dtype=dtype)
    scalar_tensor = torch.tensor(scalar, dtype=dtype)

    class Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.maximum(x, scalar_tensor)

    await validate_numerical_output(model=Model().eval(), x=x)


@pytest.mark.parametrize(
    "scalar,tensor_shape,dtype",
    [
        (2.0, (2, 3), torch.float32),
        (3.0, (2, 3, 4), torch.float32),
        (-1.0, (5,), torch.float32),
        (2.0, (2, 3), torch.float16),
    ],
)
async def test_minimum_scalar(
    scalar: float, tensor_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    x = torch.rand(tensor_shape, dtype=dtype)
    scalar_tensor = torch.tensor(scalar, dtype=dtype)

    class Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.minimum(x, scalar_tensor)

    await validate_numerical_output(model=Model().eval(), x=x)


class TestSlice:
    """Test suite for aten.slice.Tensor → coreai.slice_ conversion."""

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x,dim,start,end,step",
        [
            (torch.rand(4, 6, 8), 1, 2, 5, 1),  # basic: middle dim, positive bounds
            (torch.rand(10, 8), 0, 0, 8, 2),  # step > 1
            (torch.rand(5, 7), 0, None, None, 1),  # None start/end defaults
            (torch.rand(6, 8), 0, 1, -1, 1),  # negative end
            (torch.rand(4, 6, 8), -1, 0, 4, 1),  # negative dim
            pytest.param(
                torch.rand(10),
                0,
                0,
                10,
                1,
            ),  # 1D full slice
        ],
    )
    async def test_static_bounds(
        self,
        x: Tensor,
        dim: int,
        start: int | None,
        end: int | None,
        step: int,
        dynamic: bool,
    ) -> None:
        """Static start/end/step; optionally non-sliced dims made symbolic.

        When dynamic=True only the dimensions other than the sliced one are made
        symbolic. Making the sliced dim dynamic with a finite static end would
        cause torch.export to derive a constraint (d >= end) that conflicts with
        the Dim's min=1 declaration.
        """

        class SliceModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.aten.slice.Tensor(x, dim, start, end, step)

        model = SliceModel().eval()
        if dynamic:
            actual_dim = (dim + x.dim()) % x.dim()
            names = [None if i == actual_dim else f"d{i}" for i in range(x.dim())]
            dynamic_shapes = make_dynamic_shapes(x=names) if any(names) else None
        else:
            dynamic_shapes = None
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)

    @pytest.mark.parametrize(
        "x_shape,y_shape,slice_dim,dtype",
        [
            # Slice x along dim 0 using y's batch size as end bound
            ((8, 4), (5, 4), 0, torch.float32),
            # slice_dim=1: x's sliced dim shares the same Dim as y's dim 0 so end is dynamic
            ((4, 6), (6, 4), 1, torch.float32),
            # 3D case: slice along dim 0
            ((6, 4, 3), (4, 4, 3), 0, torch.float32),
            # FP16
            ((8, 6), (5, 6), 0, torch.float16),
        ],
    )
    async def test_dynamic_bounds(
        self,
        x_shape: tuple[int, ...],
        y_shape: tuple[int, ...],
        slice_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """Dynamic end bound derived from another tensor's shape (fx.Node path).

        end = y.shape[0] becomes a sym_size.int fx.Node when exported with
        dynamic shapes, exercising the IR-Value path in replace_slice.

        For slice_dim=0: batch_x and batch_y are independent Dims (batch_y <= batch_x).
        For slice_dim != 0: x's sliced dim shares the same Dim as y's dim 0, allowing
        both to be dynamic without triggering upper-bound constraint violations.
        """
        x = torch.rand(x_shape, dtype=dtype)
        y = torch.rand(y_shape, dtype=dtype)

        class DynBoundSliceModel(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.dim = dim

            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                # end = y.shape[0] → becomes a sym_size.int fx.Node when dynamic
                end = y.shape[0]
                return torch.ops.aten.slice.Tensor(x, self.dim, 0, end, 1)

        model = DynBoundSliceModel(dim=slice_dim).eval()
        if slice_dim == 0:
            # Independent batch dims: batch_y (end) <= batch_x (x's sliced dim)
            dynamic_shapes = make_dynamic_shapes(
                x=["batch_x"] + [None] * (len(x_shape) - 1),
                y=["batch_y"] + [None] * (len(y_shape) - 1),
            )
        else:
            # Share a Dim between x's sliced dim and y's dim 0 so end stays dynamic
            x_names: list[str | None] = [None] * len(x_shape)
            x_names[slice_dim] = "shared"
            dynamic_shapes = make_dynamic_shapes(
                x=x_names,
                y=["shared"] + [None] * (len(y_shape) - 1),
            )
        await validate_numerical_output(
            model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x,dim,start",
        [
            (torch.rand(5, 3), 0, 2),  # 2D: slice dim 0 from index 2 to end
            (torch.rand(4, 6, 3), 1, 1),  # 3D: slice dim 1 from index 1 to end
            (torch.rand(8), 0, 3),  # 1D: slice from index 3 to end
        ],
    )
    async def test_slice_to_end(
        self, x: Tensor, dim: int, start: int, dynamic: bool
    ) -> None:
        """Slice where end = sys.maxsize (INT64_MAX) from Python x[start:] notation.

        ATen uses INT64_MAX as the 'slice to end' sentinel. Coreai indices are si32,
        so without clamping, INT64_MAX overflows to -1 and causes wrong output shapes.
        The fix in resolve_slice_arg clamps values above INT32_MAX to INT32_MAX.
        """

        class SliceToEndModel(nn.Module):
            def __init__(self, d: int, s: int) -> None:
                super().__init__()
                self.d = d
                self.s = s

            def forward(self, x: Tensor) -> Tensor:
                return torch.ops.aten.slice.Tensor(x, self.d, self.s, sys.maxsize, 1)

        model = SliceToEndModel(dim, start).eval()
        if dynamic:
            actual_dim = (dim + x.dim()) % x.dim()
            names = [None if i == actual_dim else f"d{i}" for i in range(x.dim())]
            dynamic_shapes = make_dynamic_shapes(x=names) if any(names) else None
        else:
            dynamic_shapes = None
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)

    @pytest.mark.ir
    def test_static_dims_preserved_with_dynamic_bounds(self) -> None:
        """replace_slice must preserve static dims when the sliced dim has a dynamic bound.

        coreai.slice_ infers *all* output dims as dynamic whenever any bound is a
        dynamic Value.  replace_slice fixes this by forwarding
        results=[get_tensor_type(node.meta["val"])] so that torch.export's static
        shape knowledge for the non-sliced dims is reflected in the IR type.

        Concretely: slicing a (1, seq, 64) tensor along dim=1 from 0 to a dynamic
        end must produce tensor<1x?x64xf32>, not tensor<?x?x?xf32>.
        """

        class DynEndSlice(nn.Module):
            def forward(self, x: Tensor, end: Tensor) -> Tensor:
                return torch.ops.aten.slice.Tensor(x, 1, 0, end.shape[0], 1)

        x = torch.rand(1, 4, 64, dtype=torch.float32)
        end = torch.rand(4, dtype=torch.float32)

        dynamic_shapes = make_dynamic_shapes(
            x=[None, "seq", None],
            end=["seq"],
        )
        program = torch.export.export(
            DynEndSlice(), args=(x, end), dynamic_shapes=dynamic_shapes
        ).run_decompositions()

        result = TorchConverter().add_exported_program(program).to_coreai()
        filecheck_pattern(
            str(result),
            check_file="""
                // CHECK: coreai.graph @main{{.*}}-> (tensor<1x?x64xf32>
                // CHECK: coreai.slice
            """,
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dim",
    [
        # Test different tensor dimensions with positive and negative dim
        (torch.rand(8), 0),  # 1D tensor, dim=0
        (torch.rand(8), -1),  # 1D tensor, dim=-1 (same as 0)
        (torch.rand(3, 5), 1),  # 2D tensor, dim=1
        (torch.rand(3, 5), -1),  # 2D tensor, dim=-1 (same as 1)
        (torch.rand(2, 4, 6), 2),  # 3D tensor, dim=2
        (torch.rand(2, 4, 6), -2),  # 3D tensor, dim=-2 (same as 1)
    ],
)
async def test_softmax(x: Tensor, dim: int, dynamic: bool) -> None:
    class SoftmaxModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.softmax(x, dim)

    model = SoftmaxModel().eval()
    dynamic_shapes = (
        make_dynamic_shapes(x=[f"d{i}" for i in range(x.dim())]) if dynamic else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dim",
    [
        # Test different tensor dimensions with positive and negative dim
        (torch.rand(8), 0),  # 1D tensor, dim=0
        (torch.rand(8), -1),  # 1D tensor, dim=-1 (same as 0)
        (torch.rand(3, 5), 1),  # 2D tensor, dim=1
        (torch.rand(3, 5), -1),  # 2D tensor, dim=-1 (same as 1)
        (torch.rand(2, 4, 6), 2),  # 3D tensor, dim=2
        (torch.rand(2, 4, 6), -2),  # 3D tensor, dim=-2 (same as 1)
        # FP16 test case for numerical stability path
        (torch.rand(3, 5, dtype=torch.float16), 1),
    ],
)
async def test_log_softmax(x: Tensor, dim: int, dynamic: bool) -> None:
    """Test log_softmax operation which computes log(softmax(x, dim))."""

    class LogSoftmaxModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.log_softmax(x, dim)

    model = LogSoftmaxModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize("y", [torch.rand(2, 2)])
@pytest.mark.parametrize("dynamic", [False, True])
async def test_sub(x: Tensor, y: Tensor, dynamic: bool) -> None:
    class SubModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return x - y

    model = SubModel().eval()
    dynamic_shapes = None
    if dynamic:
        n0 = torch.export.Dim(name="n1", min=1, max=32)
        n1 = torch.export.Dim(name="n2", min=1, max=64)
        dynamic_shapes = {"x": {0: n0, 1: n1}, "y": {0: n0, 1: n1}}
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 2)])
@pytest.mark.parametrize(
    "unary_op",
    [
        torch.acos,
        torch.asin,
        torch.asinh,
        torch.atan,
        torch.atanh,
        torch.cos,
        torch.cosh,
        torch.erf,
        torch.exp,
        torch.log,
        torch.relu,
        torch.sigmoid,
        torch.nn.functional.silu,
        torch.sin,
        torch.sinh,
        torch.sqrt,
        torch.tan,
        torch.tanh,
    ],
)
async def test_unary_ops(x: Tensor, unary_op: Any, dynamic: bool) -> None:
    class UnaryModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return unary_op(x)

    model = UnaryModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    remove_decomps = []
    if unary_op is torch.nn.functional.silu:
        remove_decomps.append(torch.ops.aten.silu.default)
    await validate_numerical_output(
        model=model,
        x=x,
        dynamic_shapes=dynamic_shapes,
        remove_decomps=remove_decomps or None,
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("x", [torch.rand(2, 3)])
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
async def test_unsqueeze(x: Tensor, dim: int, dynamic: bool) -> None:
    class UnsqueezeModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.unsqueeze(x, dim)

    model = UnsqueezeModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestWhere:
    """Test suite for aten.where.self → coreai.broadcasting_where conversion."""

    @pytest.mark.parametrize(
        "condition,x,y",
        [
            # 2D case with float32 tensors
            (
                torch.tensor(
                    [[True, False, True], [False, True, False]], dtype=torch.bool
                ),
                torch.rand(2, 3, dtype=torch.float32),
                torch.rand(2, 3, dtype=torch.float32),
            ),
            # 2D case with int32 tensors
            (
                torch.tensor([[True, True], [False, False]], dtype=torch.bool),
                torch.randint(0, 10, (2, 2), dtype=torch.int32),
                torch.randint(0, 10, (2, 2), dtype=torch.int32),
            ),
            # 1D case with float64 tensors
            (
                torch.tensor([True, False, True, False], dtype=torch.bool),
                torch.rand(4, dtype=torch.float64),
                torch.rand(4, dtype=torch.float64),
            ),
            # 3D case with float16 tensors
            (
                torch.tensor(
                    [[[True, False], [False, True]], [[False, True], [True, False]]],
                    dtype=torch.bool,
                ),
                torch.rand(2, 2, 2, dtype=torch.float16),
                torch.rand(2, 2, 2, dtype=torch.float16),
            ),
            # Broadcast: 1D condition over 2D inputs
            (
                torch.tensor([True, False, True], dtype=torch.bool),
                torch.rand(4, 3, dtype=torch.float32),
                torch.rand(4, 3, dtype=torch.float32),
            ),
            # Broadcast: scalar-like condition (shape [1]) over 2D
            (
                torch.tensor([True], dtype=torch.bool),
                torch.rand(3, 4, dtype=torch.float32),
                torch.rand(3, 4, dtype=torch.float32),
            ),
            # Broadcast: 2D condition [2,4] over 3D x/y [2,2,4]
            (
                torch.tensor(
                    [[True, False, True, False], [False, True, False, True]],
                    dtype=torch.bool,
                ),
                torch.rand(2, 2, 4, dtype=torch.float32),
                torch.rand(2, 2, 4, dtype=torch.float32),
            ),
        ],
    )
    async def test_static(self, condition: Tensor, x: Tensor, y: Tensor) -> None:
        class WhereModel(nn.Module):
            def forward(self, condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
                return torch.where(condition, x, y)

        model = WhereModel().eval()
        await validate_numerical_output(model=model, condition=condition, x=x, y=y)

    @pytest.mark.parametrize(
        "condition,x,y,condition_dims,x_dims,y_dims",
        [
            # Batch dim only dynamic — condition/x/y share same dynamic batch
            (
                torch.tensor([[True, False], [False, True]], dtype=torch.bool),
                torch.rand(2, 2, dtype=torch.float32),
                torch.rand(2, 2, dtype=torch.float32),
                ["batch", None],
                ["batch", None],
                ["batch", None],
            ),
            # All dims dynamic — 2D float32
            (
                torch.tensor(
                    [[True, False, True], [False, True, False]], dtype=torch.bool
                ),
                torch.rand(2, 3, dtype=torch.float32),
                torch.rand(2, 3, dtype=torch.float32),
                ["batch", "seq"],
                ["batch", "seq"],
                ["batch", "seq"],
            ),
            # All dims dynamic — 3D float16
            (
                torch.randint(0, 2, (2, 3, 4), dtype=torch.bool),
                torch.rand(2, 3, 4, dtype=torch.float16),
                torch.rand(2, 3, 4, dtype=torch.float16),
                ["batch", "H", "W"],
                ["batch", "H", "W"],
                ["batch", "H", "W"],
            ),
            # Broadcast: 1D condition (dynamic) over 2D x/y (both dims dynamic, sharing cols)
            (
                torch.tensor([True, False, True], dtype=torch.bool),
                torch.rand(4, 3, dtype=torch.float32),
                torch.rand(4, 3, dtype=torch.float32),
                ["cols"],
                ["rows", "cols"],
                ["rows", "cols"],
            ),
            # Broadcast: 2D condition over 3D x/y — x/y batch dim dynamic only
            (
                torch.tensor(
                    [[True, False, True, False], [False, True, False, True]],
                    dtype=torch.bool,
                ),
                torch.rand(2, 2, 4, dtype=torch.float32),
                torch.rand(2, 2, 4, dtype=torch.float32),
                [None, None],
                ["batch", None, None],
                ["batch", None, None],
            ),
        ],
    )
    async def test_dynamic(
        self,
        condition: Tensor,
        x: Tensor,
        y: Tensor,
        condition_dims: list[str | None],
        x_dims: list[str | None],
        y_dims: list[str | None],
    ) -> None:
        class WhereModel(nn.Module):
            def forward(self, condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
                return torch.where(condition, x, y)

        model = WhereModel().eval()
        dynamic_shapes = make_dynamic_shapes(
            condition=condition_dims,
            x=x_dims,
            y=y_dims,
        )
        await validate_numerical_output(
            model=model, condition=condition, x=x, y=y, dynamic_shapes=dynamic_shapes
        )


@pytest.mark.parametrize(
    "scalar_value,dtype",
    [
        # Float values with different dtypes
        (5.0, torch.float32),
        (3.14, torch.float16),
        (-2.5, torch.float32),
        # Integer values
        (42, torch.int32),
        (-10, torch.int32),
        # Boolean values
        (True, torch.bool),
        (False, torch.bool),
        # Edge cases
        (0.0, torch.float32),
        (0, torch.int32),
    ],
)
async def test_scalar_tensor(
    scalar_value: bool | int | float, dtype: torch.dtype
) -> None:
    """Test scalar_tensor operation which creates a 0-dimensional tensor from a scalar value."""

    class ScalarTensorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self) -> Tensor:
            return torch.scalar_tensor(scalar_value, dtype=dtype)

    model = ScalarTensorModel().eval()
    await validate_numerical_output(model=model)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dtype",
    [
        # Float output dtypes
        (torch.rand(3, 4), torch.float32),
        (torch.rand(5, 2), torch.float16),
        # Integer output dtype
        (torch.zeros(4, 3, dtype=torch.int32), torch.int32),
        # 1D input — batch dim only
        (torch.rand(6), torch.float32),
    ],
)
async def test_scalar_tensor_dynamic(
    x: Tensor, dtype: torch.dtype, dynamic: bool
) -> None:
    """Test scalar_tensor with a dynamic scalar value derived from a tensor dimension."""

    class ScalarTensorDynamicModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.scalar_tensor(x.size(0), dtype=dtype)

    model = ScalarTensorDynamicModel().eval()
    dynamic_shapes = make_dynamic_shapes(x=["batch"]) if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "dim,reduce_mode,fill_value",
    [
        # scatter.src: tensor source, no reduction
        (0, None, None),
        (1, None, None),
        # scatter.src with reduce=add
        (0, "add", None),
        (1, "add", None),
        # scatter.src with reduce=multiply
        (0, "multiply", None),
        # scatter.value: scalar fill value
        (0, None, 2.5),
        (1, None, -1.0),
        (0, None, 0.0),
    ],
)
async def test_scatter(
    dim: int, reduce_mode: str | None, fill_value: float | None, dynamic: bool
) -> None:
    """Test scatter.src (tensor source) and scatter.value (scalar fill) variants."""
    is_value = fill_value is not None
    # Use non-zero input for multiply mode
    x = (
        torch.ones(3, 4, dtype=torch.float32)
        if reduce_mode == "multiply"
        else torch.zeros(3, 4, dtype=torch.float32)
    )
    index_shape = [2, 4] if dim == 0 else [3, 2]
    max_idx = x.shape[dim]
    index = torch.randint(0, max_idx, index_shape, dtype=torch.int32)

    if is_value:

        class ScatterValueModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dim = dim
                self.fill_value = fill_value

            def forward(self, x: Tensor, index: Tensor) -> Tensor:
                return torch.scatter(x, self.dim, index, self.fill_value)

        model = ScatterValueModel().eval()
        dynamic_shapes = (
            # scatter enforces index.size(d) <= x.size(d) for all d, so x's min
            # per dimension must be >= the corresponding index dimension.
            {
                "x": {
                    d: torch.export.Dim(f"x{d}", min=max(index.shape[d], 1))
                    for d in range(x.dim())
                },
                "index": None,
            }
            if dynamic
            else None
        )
        await validate_numerical_output(
            model=model, x=x, index=index, dynamic_shapes=dynamic_shapes
        )
    else:
        src = torch.rand(index_shape, dtype=torch.float32)

        class ScatterSrcModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dim = dim
                self.reduce_mode = reduce_mode

            def forward(self, x: Tensor, index: Tensor, src: Tensor) -> Tensor:
                if self.reduce_mode is not None:
                    return torch.scatter(
                        x, self.dim, index, src, reduce=self.reduce_mode
                    )
                return torch.scatter(x, self.dim, index, src)

        model = ScatterSrcModel().eval()
        dynamic_shapes = (
            {
                "x": _all_dims_dynamic(x, "x"),
                "index": _all_dims_dynamic(index, "i"),
                "src": _all_dims_dynamic(src, "s"),
            }
            if dynamic
            else None
        )
        await validate_numerical_output(
            model=model, x=x, index=index, src=src, dynamic_shapes=dynamic_shapes
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dim,index",
    [
        # 2D float32, select along dim 0
        (torch.rand(3, 4, dtype=torch.float32), 0, 1),
        # 2D float32, select along dim 1
        (torch.rand(3, 4, dtype=torch.float32), 1, 2),
        # 3D float16, select along dim 1
        (torch.rand(2, 3, 4, dtype=torch.float16), 1, 2),
        # 3D int32, select along dim 2
        (torch.randint(0, 100, (2, 3, 4), dtype=torch.int32), 2, 3),
        # 2D int64, select along dim 0
        (torch.randint(-50, 50, (4, 5), dtype=torch.int64), 0, 2),
        # Negative dimension (dim=-1 is last dim, float32)
        (torch.rand(3, 4, 5, dtype=torch.float32), -1, 2),
        # Negative index (index from end, int32) — exercises dynamic path when dynamic=True
        (torch.randint(0, 100, (4, 5), dtype=torch.int32), 0, -1),
        # 1D tensor select (float32)
        (torch.rand(10, dtype=torch.float32), 0, 5),
        # 1D tensor, negative index — exercises 1D dynamic path when dynamic=True
        (torch.rand(8, dtype=torch.float32), 0, -2),
        # 3D float32, negative index on last dim — exercises dynamic path when dynamic=True
        (torch.rand(2, 3, 5, dtype=torch.float32), 2, -1),
    ],
)
async def test_select_int(x: Tensor, dim: int, index: int, dynamic: bool) -> None:
    """Test select.int operation which selects a single element along a dimension.

    When dynamic=True all dimensions of x are made symbolic, exercising:
    - Non-negative index with dynamic shapes (static path, dynamic tensor dims).
    - Negative index with dynamic shapes (dynamic path: runtime index resolution).
    """

    class SelectModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.select(x, dim, index)

    model = SelectModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dim,index",
    [
        # 2D tensor, gather along dim 0 (int32)
        (
            torch.rand(3, 4),
            0,
            torch.tensor([[0, 1, 2, 0], [2, 1, 0, 1]], dtype=torch.int32),
        ),
        # 2D tensor, gather along dim 1 (int32)
        (
            torch.rand(3, 4),
            1,
            torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.int32),
        ),
        # 3D tensor, gather along dim 0 (int32)
        (torch.rand(2, 3, 4), 0, torch.randint(0, 2, (2, 3, 4), dtype=torch.int32)),
        # 3D tensor, gather along dim 1 (int32)
        (torch.rand(2, 3, 4), 1, torch.randint(0, 3, (2, 2, 4), dtype=torch.int32)),
        # 3D tensor, gather along dim 2 (int32)
        (torch.rand(2, 3, 4), 2, torch.randint(0, 4, (2, 3, 2), dtype=torch.int32)),
        # Negative dim (same as dim=1 for 2D) (int32)
        (
            torch.rand(3, 4),
            -1,
            torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.int32),
        ),
        # 2D tensor, gather along dim 0 (int64)
        (
            torch.rand(3, 4),
            0,
            torch.tensor([[0, 1, 2, 0], [2, 1, 0, 1]], dtype=torch.int64),
        ),
        # 2D tensor, gather along dim 1 (int64)
        (
            torch.rand(3, 4),
            1,
            torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.int64),
        ),
        # 3D tensor, gather along dim 0 (int64)
        (torch.rand(2, 3, 4), 0, torch.randint(0, 2, (2, 3, 4), dtype=torch.int64)),
        # 3D tensor, gather along dim 1 (int64)
        (torch.rand(2, 3, 4), 1, torch.randint(0, 3, (2, 2, 4), dtype=torch.int64)),
        # 3D tensor, gather along dim 2 (int64)
        (torch.rand(2, 3, 4), 2, torch.randint(0, 4, (2, 3, 2), dtype=torch.int64)),
        # Negative dim (int64)
        (
            torch.rand(3, 4),
            -1,
            torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.int64),
        ),
    ],
)
async def test_gather(x: Tensor, dim: int, index: Tensor, dynamic: bool) -> None:
    """Test gather operation which gathers values along an axis specified by dim."""

    class GatherModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor, index: Tensor) -> Tensor:
            return torch.gather(x, dim, index)

    model = GatherModel().eval()
    dynamic_shapes = (
        {"x": _all_dims_dynamic(x, "x"), "index": _all_dims_dynamic(index, "i")}
        if dynamic
        else None
    )
    await validate_numerical_output(
        model=model, x=x, index=index, dynamic_shapes=dynamic_shapes
    )


class TestIndexSelect:
    """Test suite for aten.index_select → coreai.gather_along_axis conversion."""

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x,dim,index,dtype",
        [
            # 2D tensor, select along dim 0 with int32 indices
            (
                torch.rand(4, 5),
                0,
                torch.tensor([0, 2, 1], dtype=torch.int32),
                torch.float32,
            ),
            # 2D tensor, select along dim 1 with int64 indices
            (
                torch.rand(3, 6),
                1,
                torch.tensor([1, 3, 5, 0], dtype=torch.int64),
                torch.float32,
            ),
            # 3D tensor, select along dim 0
            (
                torch.rand(4, 3, 5),
                0,
                torch.tensor([0, 3, 1], dtype=torch.int32),
                torch.float32,
            ),
            # 3D tensor, select along dim 1
            (
                torch.rand(2, 5, 4),
                1,
                torch.tensor([2, 4, 0], dtype=torch.int32),
                torch.float32,
            ),
            # 3D tensor, select along dim 2
            (
                torch.rand(2, 3, 6),
                2,
                torch.tensor([0, 2, 5, 1], dtype=torch.int32),
                torch.float32,
            ),
            # Negative dimension (dim=-1 is last dim)
            (
                torch.rand(3, 4, 5),
                -1,
                torch.tensor([0, 2, 4], dtype=torch.int32),
                torch.float32,
            ),
            # FP16 test case
            (
                torch.rand(3, 4),
                0,
                torch.tensor([0, 2, 1, 2], dtype=torch.int32),
                torch.float16,
            ),
            # Single index (edge case)
            (torch.rand(5, 3), 0, torch.tensor([2], dtype=torch.int32), torch.float32),
            # int64 indices — exercises coreai.cast(index, np.int32) in replace_index_select
            (
                torch.rand(4, 5),
                0,
                torch.tensor([0, 2, 1], dtype=torch.int64),
                torch.float32,
            ),
            (
                torch.rand(3, 6),
                1,
                torch.tensor([1, 3, 5, 0], dtype=torch.int64),
                torch.float32,
            ),
        ],
    )
    async def test_dim(
        self, x: Tensor, dim: int, index: Tensor, dtype: torch.dtype, dynamic: bool
    ) -> None:
        """Test index_select operation which selects elements along a dimension using 1D indices."""
        x = x.to(dtype)

        class IndexSelectModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor, index: Tensor) -> Tensor:
                return torch.index_select(x, dim, index)

        model = IndexSelectModel().eval()
        dynamic_shapes = (
            {"x": _all_dims_dynamic(x, "x"), "index": None} if dynamic else None
        )
        await validate_numerical_output(
            model=model, x=x, index=index, dynamic_shapes=dynamic_shapes
        )

    @pytest.mark.parametrize(
        "x,dim",
        [
            (torch.rand(5), 0),  # 1D, dim=0
            (torch.rand(4, 5), 0),  # 2D, dim=0
            (torch.rand(4, 5), 1),  # 2D, dim=1
            (torch.rand(3, 4, 5), 1),  # 3D, dim=1
            (torch.rand(3, 4, 5), 0),  # 3D, dim=0
            (torch.rand(3, 4, 5), 2),  # 3D, dim=2
        ],
    )
    async def test_dynamic_index_size(self, x: Tensor, dim: int) -> None:
        """index_select with dynamic index size (indices_size < 0 branch)."""

        class IndexSelectModel(nn.Module):
            def forward(self, x: Tensor, index: Tensor) -> Tensor:
                return torch.index_select(x, dim, index)

        index = torch.tensor([0, 2, 1], dtype=torch.int64)
        model = IndexSelectModel().eval()
        # Make only the index size dynamic — x stays fully static.
        dynamic_shapes = {"x": {}, "index": {0: torch.export.Dim("N", min=1)}}
        await validate_numerical_output(
            model=model, x=x, index=index, dynamic_shapes=dynamic_shapes
        )

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "x,dim,index",
        [
            # Static: 2D, select along dim 0
            (torch.rand(4, 5), 0, torch.tensor([0, 2], dtype=torch.int32)),
            # Static: 3D, select along dim 1
            (torch.rand(2, 5, 4), 1, torch.tensor([1, 3], dtype=torch.int32)),
            # Static: 3D, select along dim 2
            (torch.rand(2, 3, 6), 2, torch.tensor([0, 2, 5], dtype=torch.int32)),
        ],
    )
    def test_uses_broadcast_in_dims(self, x: Tensor, dim: int, index: Tensor) -> None:
        """index_select should emit broadcast_in_dims instead of concat+broadcast_to."""

        class IndexSelectModel(nn.Module):
            def forward(self, x: Tensor, index: Tensor) -> Tensor:
                return torch.index_select(x, dim, index)

        model = IndexSelectModel().eval()
        program = torch.export.export(model, args=(x, index)).run_decompositions()
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        ir = str(coreai_program)
        pattern = "CHECK: coreai.broadcast_in_dims\nCHECK-NOT: coreai.broadcast_to"
        filecheck_pattern(ir, pattern)

    @pytest.mark.ir
    @pytest.mark.parametrize(
        "x,dim,index",
        [
            # Dynamic x: 2D, select along dim 0
            (torch.rand(4, 5), 0, torch.tensor([0, 2], dtype=torch.int32)),
            # Dynamic x: 3D, select along dim 1
            (torch.rand(2, 5, 4), 1, torch.tensor([1, 3], dtype=torch.int32)),
            # Dynamic x: 3D, select along dim 2
            (torch.rand(2, 3, 6), 2, torch.tensor([0, 2, 5], dtype=torch.int32)),
        ],
    )
    def test_uses_broadcast_in_dims_dynamic(
        self, x: Tensor, dim: int, index: Tensor
    ) -> None:
        """index_select with dynamic x should still emit broadcast_in_dims."""

        class IndexSelectModel(nn.Module):
            def forward(self, x: Tensor, index: Tensor) -> Tensor:
                return torch.index_select(x, dim, index)

        model = IndexSelectModel().eval()
        dynamic_shapes = {"x": _all_dims_dynamic(x, "x"), "index": None}
        program = torch.export.export(
            model, args=(x, index), dynamic_shapes=dynamic_shapes
        ).run_decompositions()
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        ir = str(coreai_program)
        pattern = "CHECK: coreai.broadcast_in_dims\nCHECK-NOT: coreai.broadcast_to"
        filecheck_pattern(ir, pattern)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        # Float tensors with mixed positive/negative values
        torch.rand(2, 3) * 4 - 2,  # 2D: Range [-2, 2]
        torch.rand(3, 4, 5) * 6 - 3,  # 3D: Range [-3, 3]
        torch.rand(10) * 10 - 5,  # 1D: Range [-5, 5]
        # Edge cases with specific values including zero
        torch.tensor(
            [0.0, -0.0, 1.0, -1.0, 2.5, -2.5]
        ),  # Specific values including zero
        torch.tensor([[-1.5, 2.0], [0.0, -4.0]]),  # 2D with mixed signs and zero
        # Integer tensors
        torch.tensor([-5, -2, 0, 3, 7], dtype=torch.int32),  # int32
        torch.tensor(
            [[-10, -1], [0, 2], [5, 15]], dtype=torch.int32
        ),  # 2D int32 with zero
    ],
)
async def test_sign(x: Tensor, dynamic: bool) -> None:
    """Test sign operation which returns -1 for negative, 0 for zero, 1 for positive values."""

    class SignModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.sign(x)

    model = SignModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,target_dtype",
    [
        # Same dtype (no-op case)
        pytest.param(
            torch.rand(2, 3),
            torch.float32,
        ),
        # Float32 to Float16
        (torch.rand(2, 3), torch.float16),
        # Float16 to Float32
        (torch.rand(2, 3, dtype=torch.float16), torch.float32),
        # Int32 to Float32
        (torch.randint(-10, 10, (3, 4), dtype=torch.int32), torch.float32),
        # Float32 to Int32
        (torch.rand(2, 3) * 10, torch.int32),
        # 3D tensor Float32 to Float16
        (torch.rand(2, 3, 4), torch.float16),
        # 1D tensor conversion
        (torch.rand(10), torch.float16),
    ],
)
async def test_to_copy(x: Tensor, target_dtype: torch.dtype, dynamic: bool) -> None:
    """Test _to_copy operation which performs dtype conversions."""

    class ToCopyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.ops.aten._to_copy.default(x, dtype=target_dtype)

    model = ToCopyModel().eval()
    dynamic_shapes = (
        make_dynamic_shapes(x=["batch"] + [None] * (x.dim() - 1)) if dynamic else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,target_dtype",
    [
        # Same dtype (no-op case)
        pytest.param(
            torch.rand(2, 3),
            torch.float32,
        ),
        # Float32 to Float16
        (torch.rand(2, 3), torch.float16),
        # Float16 to Float32
        (torch.rand(2, 3, dtype=torch.float16), torch.float32),
        # Int32 to Float32
        (torch.randint(-10, 10, (3, 4), dtype=torch.int32), torch.float32),
        # Float32 to Int32
        (torch.rand(2, 3) * 10, torch.int32),
        # 3D tensor Float32 to Float16
        (torch.rand(2, 3, 4), torch.float16),
        # 1D tensor conversion
        (torch.rand(10), torch.float16),
    ],
)
async def test_to_dtype(x: Tensor, target_dtype: torch.dtype, dynamic: bool) -> None:
    """Test to.dtype operation which performs dtype conversions."""

    class ToDtypeModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x.to(dtype=target_dtype)

    model = ToDtypeModel().eval()
    dynamic_shapes = (
        make_dynamic_shapes(x=["batch"] + [None] * (x.dim() - 1)) if dynamic else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestIndexPut:
    """Test suite for index_put operation."""

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_contiguous_2d(self, dtype: torch.dtype, dynamic: bool) -> None:
        """Test index_put with contiguous indices on 2D tensor."""
        x = torch.rand(4, 6, dtype=dtype)
        idx0 = torch.tensor([0, 2, 1], dtype=torch.int32)
        idx1 = torch.tensor([1, 3, 2], dtype=torch.int32)
        values = torch.rand(3, dtype=dtype)

        class IndexPutModel(nn.Module):
            def forward(
                self, x: Tensor, values: Tensor, idx0: Tensor, idx1: Tensor
            ) -> Tensor:
                x_copy = x.clone()
                x_copy[idx0, idx1] = values
                return x_copy

        model = IndexPutModel().eval()
        # index_put only supports dynamic batch dim (dim 0) on x and values
        dynamic_shapes = (
            {
                "x": {0: torch.export.Dim("x0", min=1)},
                "values": {0: torch.export.Dim("v0", min=1)},
                "idx0": _all_dims_dynamic(idx0, "i0"),
                "idx1": _all_dims_dynamic(idx1, "i1"),
            }
            if dynamic
            else None
        )
        await validate_numerical_output(
            model=model,
            x=x,
            values=values,
            idx0=idx0,
            idx1=idx1,
            dynamic_shapes=dynamic_shapes,
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_contiguous_3d(self, dtype: torch.dtype, dynamic: bool) -> None:
        """Test index_put with contiguous indices on 3D tensor."""
        x = torch.rand(3, 4, 5, dtype=dtype)
        idx0 = torch.tensor([0, 2], dtype=torch.int32)
        idx1 = torch.tensor([1, 3], dtype=torch.int32)
        values = torch.rand(2, 5, dtype=dtype)

        class IndexPutModel(nn.Module):
            def forward(
                self, x: Tensor, values: Tensor, idx0: Tensor, idx1: Tensor
            ) -> Tensor:
                x_copy = x.clone()
                x_copy[idx0, idx1, :] = values
                return x_copy

        model = IndexPutModel().eval()
        # index_put only supports dynamic batch dim (dim 0) on x and values
        dynamic_shapes = (
            {
                "x": {0: torch.export.Dim("x0", min=1)},
                "values": {0: torch.export.Dim("v0", min=1)},
                "idx0": _all_dims_dynamic(idx0, "i0"),
                "idx1": _all_dims_dynamic(idx1, "i1"),
            }
            if dynamic
            else None
        )
        await validate_numerical_output(
            model=model,
            x=x,
            values=values,
            idx0=idx0,
            idx1=idx1,
            dynamic_shapes=dynamic_shapes,
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_contiguous_with_broadcast(
        self, dtype: torch.dtype, dynamic: bool
    ) -> None:
        """Test index_put with broadcasting contiguous indices."""
        x = torch.rand(4, 6, dtype=dtype)
        idx0 = torch.tensor([[0], [2], [1]], dtype=torch.int32)  # shape (3, 1)
        idx1 = torch.tensor([[1, 3]], dtype=torch.int32)  # shape (1, 2)
        values = torch.rand(3, 2, dtype=dtype)

        class IndexPutModel(nn.Module):
            def forward(
                self, x: Tensor, values: Tensor, idx0: Tensor, idx1: Tensor
            ) -> Tensor:
                x_copy = x.clone()
                x_copy[idx0, idx1] = values
                return x_copy

        model = IndexPutModel().eval()
        # idx0/idx1 have size-1 dims so are kept static; index_put only supports dynamic
        # batch dim (dim 0) on x and values
        dynamic_shapes = (
            {
                "x": {0: torch.export.Dim("x0", min=1)},
                "values": {0: torch.export.Dim("v0", min=1)},
                "idx0": None,
                "idx1": None,
            }
            if dynamic
            else None
        )
        await validate_numerical_output(
            model=model,
            x=x,
            values=values,
            idx0=idx0,
            idx1=idx1,
            dynamic_shapes=dynamic_shapes,
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_non_contiguous_1_3(self, dtype: torch.dtype, dynamic: bool) -> None:
        """Test index_put with non-contiguous indices at dims 1 and 3."""
        x = torch.rand(4, 5, 6, 7, dtype=dtype)
        idx1 = torch.tensor([0, 3], dtype=torch.int32)
        idx3 = torch.tensor([2, 5], dtype=torch.int32)
        values = torch.rand(2, 4, 6, dtype=dtype)

        class IndexPutModel(nn.Module):
            def forward(
                self, x: Tensor, values: Tensor, idx1: Tensor, idx3: Tensor
            ) -> Tensor:
                x_copy = x.clone()
                x_copy[:, idx1, :, idx3] = values
                return x_copy

        model = IndexPutModel().eval()
        # x.dim(0) becomes a non-indexed "remaining" dim after transposing indexed dims
        # [1,3] to the front; making it dynamic causes scatter_nd to receive a dynamic
        # remaining dim it can't handle at runtime.  Keep x static and share a single
        # Dim across values.dim(0), idx1.dim(0), idx3.dim(0) (they must all be equal).
        n = torch.export.Dim("n", min=1)
        dynamic_shapes = (
            {
                "x": None,
                "values": {0: n},
                "idx1": {0: n},
                "idx3": {0: n},
            }
            if dynamic
            else None
        )
        await validate_numerical_output(
            model=model,
            x=x,
            values=values,
            idx1=idx1,
            idx3=idx3,
            dynamic_shapes=dynamic_shapes,
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_non_contiguous_0_2(self, dtype: torch.dtype, dynamic: bool) -> None:
        """Test index_put with non-contiguous indices at dims 0 and 2."""
        x = torch.rand(3, 4, 5, 6, dtype=dtype)
        idx0 = torch.tensor([0, 2], dtype=torch.int32)
        idx2 = torch.tensor([1, 3], dtype=torch.int32)
        values = torch.rand(2, 4, 6, dtype=dtype)

        class IndexPutModel(nn.Module):
            def forward(
                self, x: Tensor, values: Tensor, idx0: Tensor, idx2: Tensor
            ) -> Tensor:
                x_copy = x.clone()
                x_copy[idx0, :, idx2, :] = values
                return x_copy

        model = IndexPutModel().eval()
        # index_put only supports dynamic batch dim (dim 0) on x and values
        dynamic_shapes = (
            {
                "x": {0: torch.export.Dim("x0", min=1)},
                "values": {0: torch.export.Dim("v0", min=1)},
                "idx0": _all_dims_dynamic(idx0, "i0"),
                "idx2": _all_dims_dynamic(idx2, "i2"),
            }
            if dynamic
            else None
        )
        await validate_numerical_output(
            model=model,
            x=x,
            values=values,
            idx0=idx0,
            idx2=idx2,
            dynamic_shapes=dynamic_shapes,
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_contiguous_not_from_dim0(
        self, dtype: torch.dtype, dynamic: bool
    ) -> None:
        """Test index_put with contiguous indices not starting from dim 0."""
        x = torch.rand(5, 6, 7, 8, dtype=dtype)
        idx2 = torch.tensor([0, 3, 5], dtype=torch.int32)
        idx3 = torch.tensor([1, 4, 6], dtype=torch.int32)
        values = torch.rand(5, 6, 3, dtype=dtype)

        class IndexPutModel(nn.Module):
            def forward(
                self, x: Tensor, values: Tensor, idx2: Tensor, idx3: Tensor
            ) -> Tensor:
                x_copy = x.clone()
                x_copy[:, :, idx2, idx3] = values
                return x_copy

        model = IndexPutModel().eval()
        # values.shape[0] = x.dim(0) and values.shape[1] = x.dim(1), so they're
        # constrained by x.  x.dim(0) is a non-indexed remaining dim, making it
        # dynamic causes scatter_nd to fail at runtime.  There is no safe dynamic
        # dim here; run the same static compilation for both dynamic variants.
        dynamic_shapes = None
        await validate_numerical_output(
            model=model,
            x=x,
            values=values,
            idx2=idx2,
            idx3=idx3,
            dynamic_shapes=dynamic_shapes,
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_non_contiguous_with_broadcast(
        self, dtype: torch.dtype, dynamic: bool
    ) -> None:
        """Test index_put with non-contiguous indices and broadcasting."""
        x = torch.rand(4, 5, 6, 7, dtype=dtype)
        idx1 = torch.tensor([[0], [2], [4]], dtype=torch.int32)  # shape (3, 1)
        idx3 = torch.tensor([[1, 3]], dtype=torch.int32)  # shape (1, 2)
        values = torch.rand(3, 2, 4, 6, dtype=dtype)

        class IndexPutModel(nn.Module):
            def forward(
                self, x: Tensor, values: Tensor, idx1: Tensor, idx3: Tensor
            ) -> Tensor:
                x_copy = x.clone()
                x_copy[:, idx1, :, idx3] = values
                return x_copy

        model = IndexPutModel().eval()
        # idx1/idx3 are static (size-1 dims); values dims all derive from x (static)
        # or idx shapes (static).  x.dim(0) is a non-indexed remaining dim so making
        # it dynamic breaks scatter_nd at runtime.  Run static for both variants.
        dynamic_shapes = None
        await validate_numerical_output(
            model=model,
            x=x,
            values=values,
            idx1=idx1,
            idx3=idx3,
            dynamic_shapes=dynamic_shapes,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_simple_boolean_indexing_assignment(self, dtype: torch.dtype) -> None:
        """Test simple boolean indexing with assignment pattern."""

        class SimpleBoolIndexModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                # Simple boolean mask: zero out values greater than 0.5
                mask = x > 0.5
                x = x.clone()  # Clone to avoid mutation issues
                x[mask] = 0.0
                return x

        model = SimpleBoolIndexModel().eval()
        x = torch.randn(2, 4, 8, dtype=dtype)

        await validate_numerical_output(model=model, x=x)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "input_shape,dims",
    [
        # Squeeze single dimension at various positions
        ((2, 1, 3), [1]),  # Squeeze middle dimension
        ((1, 3, 4), [0]),  # Squeeze first dimension
        ((2, 3, 1), [2]),  # Squeeze last dimension
        # Squeeze multiple dimensions
        ((1, 3, 1, 4), [0, 2]),  # Squeeze first and third dimensions
        ((1, 1, 3, 4), [0, 1]),  # Squeeze first two dimensions
        # Negative dimension indices
        ((2, 1, 3), [-2]),  # Negative index (same as dim=1)
        ((1, 3, 1), [0, -1]),  # Mix of positive and negative indices
        # Higher rank tensors
        ((1, 2, 1, 3, 1), [0, 2, 4]),  # 5D tensor, squeeze dims 0, 2, 4
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
async def test_squeeze_dims(
    input_shape: tuple[int, ...], dims: list[int], dtype: torch.dtype, dynamic: bool
) -> None:
    """Test squeeze.dims operation which removes dimensions of size 1 at specified indices."""
    x = torch.rand(input_shape, dtype=dtype)

    class SqueezeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.squeeze(x, dims)

    model = SqueezeModel().eval()
    # Squeezed dims must be size 1, so only non-squeezed dims can be dynamic.
    ndim = len(input_shape)
    squeezed = {d % ndim for d in dims}
    dynamic_shapes = (
        {
            "x": {
                d: torch.export.Dim(f"x{d}", min=1)
                for d in range(ndim)
                if d not in squeezed
            }
        }
        if dynamic
        else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestIndexTensor:
    """Test suite for index.Tensor operation."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_contiguous_single(self, dtype: torch.dtype) -> None:
        """Test index.Tensor with single contiguous index at dim 0."""
        x = torch.rand(4, 5, 6, dtype=dtype)
        idx0 = torch.tensor([0, 2, 1], dtype=torch.int32)

        class IndexModel(nn.Module):
            def forward(self, x: Tensor, idx0: Tensor) -> Tensor:
                return x[idx0, :, :]

        model = IndexModel().eval()
        await validate_numerical_output(model=model, x=x, idx0=idx0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_contiguous_two(self, dtype: torch.dtype) -> None:
        """Test index.Tensor with two contiguous indices at dims 0 and 1."""
        x = torch.rand(4, 5, 6, dtype=dtype)
        idx0 = torch.tensor([0, 2, 1], dtype=torch.int32)
        idx1 = torch.tensor([1, 3, 2], dtype=torch.int32)

        class IndexModel(nn.Module):
            def forward(self, x: Tensor, idx0: Tensor, idx1: Tensor) -> Tensor:
                return x[idx0, idx1, :]

        model = IndexModel().eval()
        await validate_numerical_output(model=model, x=x, idx0=idx0, idx1=idx1)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_non_contiguous_0_2(self, dtype: torch.dtype) -> None:
        """Test index.Tensor with non-contiguous indices at dims 0 and 2."""
        x = torch.rand(3, 4, 5, dtype=dtype)
        idx0 = torch.tensor([0, 2], dtype=torch.int32)
        idx2 = torch.tensor([1, 3], dtype=torch.int32)

        class IndexModel(nn.Module):
            def forward(self, x: Tensor, idx0: Tensor, idx2: Tensor) -> Tensor:
                return x[idx0, :, idx2]

        model = IndexModel().eval()
        await validate_numerical_output(model=model, x=x, idx0=idx0, idx2=idx2)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_non_contiguous_1_3(self, dtype: torch.dtype) -> None:
        """Test index.Tensor with non-contiguous indices at dims 1 and 3."""
        x = torch.rand(4, 5, 6, 7, dtype=dtype)
        idx1 = torch.tensor([0, 3], dtype=torch.int32)
        idx3 = torch.tensor([2, 5], dtype=torch.int32)

        class IndexModel(nn.Module):
            def forward(self, x: Tensor, idx1: Tensor, idx3: Tensor) -> Tensor:
                return x[:, idx1, :, idx3]

        model = IndexModel().eval()
        await validate_numerical_output(model=model, x=x, idx1=idx1, idx3=idx3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    async def test_non_contiguous_with_broadcast(self, dtype: torch.dtype) -> None:
        """Test index.Tensor with non-contiguous indices and broadcasting."""
        x = torch.rand(4, 5, 6, 7, dtype=dtype)
        idx1 = torch.tensor([[0], [2], [4]], dtype=torch.int32)  # shape (3, 1)
        idx3 = torch.tensor([[1, 3]], dtype=torch.int32)  # shape (1, 2)

        class IndexModel(nn.Module):
            def forward(self, x: Tensor, idx1: Tensor, idx3: Tensor) -> Tensor:
                return x[:, idx1, :, idx3]

        model = IndexModel().eval()
        await validate_numerical_output(model=model, x=x, idx1=idx1, idx3=idx3)

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x,idx",
        [
            # 1D: select elements by position
            (torch.rand(8, dtype=torch.float32), torch.tensor([2, 5, 0, 7])),
            # 2D: select rows (index into dim 0)
            (torch.rand(5, 4, dtype=torch.float32), torch.tensor([0, 3, 1])),
            # 3D: select slices along dim 0
            (torch.rand(4, 3, 6, dtype=torch.float32), torch.tensor([1, 0, 3])),
            # fp16 input
            (torch.rand(6, 4, dtype=torch.float16), torch.tensor([0, 5, 2, 4])),
            # Repeated indices (same row selected multiple times)
            (torch.rand(4, 5, dtype=torch.float32), torch.tensor([0, 0, 2, 2])),
        ],
    )
    async def test_single_index(self, x: Tensor, idx: Tensor, dynamic: bool) -> None:
        """Advanced indexing into the first dimension."""

        class IndexTensorModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor, idx: Tensor) -> Tensor:
                return x[idx]

        model = IndexTensorModel().eval()
        dynamic_shapes = (
            {"x": _all_dims_dynamic(x, "x"), "idx": None} if dynamic else None
        )
        await validate_numerical_output(
            model=model, x=x, idx=idx, dynamic_shapes=dynamic_shapes
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x,row_idx,col_idx",
        [
            # 2D: element-wise selection via two index tensors
            (
                torch.rand(4, 6, dtype=torch.float32),
                torch.tensor([0, 2, 1, 3]),
                torch.tensor([5, 1, 4, 0]),
            ),
            # int32 input
            (
                torch.randint(0, 100, (5, 5), dtype=torch.int32),
                torch.tensor([0, 1, 4]),
                torch.tensor([4, 2, 0]),
            ),
            # fp16 input
            (
                torch.rand(3, 8, dtype=torch.float16),
                torch.tensor([0, 2, 1]),
                torch.tensor([7, 3, 5]),
            ),
        ],
    )
    async def test_multi_dim(
        self, x: Tensor, row_idx: Tensor, col_idx: Tensor, dynamic: bool
    ) -> None:
        """Two index tensors for element-wise selection (same shape, no broadcasting)."""

        class IndexTensorMultiDimModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor, row_idx: Tensor, col_idx: Tensor) -> Tensor:
                return x[row_idx, col_idx]

        model = IndexTensorMultiDimModel().eval()
        dynamic_shapes = (
            {"x": _all_dims_dynamic(x, "x"), "row_idx": None, "col_idx": None}
            if dynamic
            else None
        )
        await validate_numerical_output(
            model=model,
            x=x,
            row_idx=row_idx,
            col_idx=col_idx,
            dynamic_shapes=dynamic_shapes,
        )

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "x,row_idx,col_idx",
        [
            # (1,3) x (4,1) → broadcast shape (4,3)
            (
                torch.rand(5, 6, dtype=torch.float32),
                torch.tensor([[0, 1, 2]]),  # shape (1, 3)
                torch.tensor([[0], [1], [2], [3]]),  # shape (4, 1)
            ),
            # (1,4) x (3,1) → broadcast shape (3,4)
            (
                torch.rand(4, 8, dtype=torch.float32),
                torch.tensor([[0, 1, 2, 3]]),  # shape (1, 4)
                torch.tensor([[0], [1], [2]]),  # shape (3, 1)
            ),
            # fp16 input: (1,2) x (3,1) → broadcast shape (3,2)
            (
                torch.rand(4, 5, dtype=torch.float16),
                torch.tensor([[0, 1]]),  # shape (1, 2)
                torch.tensor([[0], [1], [2]]),  # shape (3, 1)
            ),
        ],
    )
    async def test_broadcast_shapes(
        self, x: Tensor, row_idx: Tensor, col_idx: Tensor, dynamic: bool
    ) -> None:
        """Index tensors with different shapes are broadcast to a common shape.

        Exercises process_indices_with_transpose: the fix passes a static Python list
        for shape_arg when all broadcast dims are known, so gather_nd can infer a
        statically-shaped output instead of a fully-dynamic one.
        """

        class BroadcastIndexModel(nn.Module):
            def forward(self, x: Tensor, row_idx: Tensor, col_idx: Tensor) -> Tensor:
                return x[row_idx, col_idx]

        model = BroadcastIndexModel().eval()
        dynamic_shapes = (
            {"x": _all_dims_dynamic(x, "x"), "row_idx": None, "col_idx": None}
            if dynamic
            else None
        )
        await validate_numerical_output(
            model=model,
            x=x,
            row_idx=row_idx,
            col_idx=col_idx,
            dynamic_shapes=dynamic_shapes,
        )

    @pytest.mark.ir
    def test_broadcast_static_output_shape(self) -> None:
        """Broadcast index shapes produce a statically-typed IR result.

        Before the process_indices_with_transpose fix, shape_val = coreai.constant(...)
        caused gather_nd to infer fully-dynamic output shapes (INT64_MIN). After the
        fix, the Python list is passed directly so the IR produces tensor<4x3xf32>.
        """

        class BroadcastIndexModel(nn.Module):
            def forward(self, x: Tensor, row_idx: Tensor, col_idx: Tensor) -> Tensor:
                return x[row_idx, col_idx]

        model = BroadcastIndexModel().eval()
        x = torch.rand(5, 6)
        row_idx = torch.tensor([[0, 1, 2]])  # shape (1, 3)
        col_idx = torch.tensor([[0], [1], [2], [3]])  # shape (4, 1)

        exported = torch.export.export(
            model, args=(x, row_idx, col_idx)
        ).run_decompositions()
        converter = TorchConverter().add_exported_program(exported)
        coreai_program = converter.to_coreai()

        # After the fix, the output shape must be the static broadcast result (4, 3).
        check_file = """
            // CHECK: tensor<4x3xf32>
        """
        filecheck_pattern(str(coreai_program), check_file=check_file)

    @pytest.mark.parametrize(
        "idx_shape",
        [
            (4,),  # 1D indices
            (4, 3),  # 2D indices
        ],
    )
    async def test_dynamic_broadcast_shapes(self, idx_shape: tuple[int, ...]) -> None:
        """Index tensors with dynamic sizes exercise the runtime broadcast_shapes path."""

        class BroadcastIndexModel(nn.Module):
            def forward(self, x: Tensor, row_idx: Tensor, col_idx: Tensor) -> Tensor:
                return x[row_idx, col_idx]

        x = torch.rand(5, 6)
        row_idx = torch.randint(0, 5, idx_shape, dtype=torch.int32)
        col_idx = torch.randint(0, 6, idx_shape, dtype=torch.int32)
        model = BroadcastIndexModel().eval()
        await validate_numerical_output(
            model=model,
            x=x,
            row_idx=row_idx,
            col_idx=col_idx,
            dynamic_shapes=make_dynamic_shapes(
                x={}, row_idx={0: "N"}, col_idx={0: "N"}
            ),
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding,dilation,output_padding,groups,is_1d",
    [
        # ConvTranspose2d: Basic cases with stride=2 (upsampling)
        (1, 1, 3, (2, 2), (1, 1), (1, 1), (0, 0), 1, False),
        # ConvTranspose2d: Different kernel/stride combinations
        (3, 6, 3, (1, 1), (0, 0), (1, 1), (0, 0), 1, False),
        # ConvTranspose2d: With padding and stride
        (2, 4, 3, (2, 2), (1, 1), (1, 1), (0, 0), 1, False),
        # ConvTranspose2d: With dilation
        (2, 4, 3, (1, 1), (2, 2), (2, 2), (0, 0), 1, False),
        # ConvTranspose2d: With output_padding
        (1, 1, 3, (2, 2), (1, 1), (1, 1), (1, 1), 1, False),
        # ConvTranspose2d: Grouped convolution
        (4, 4, 3, (2, 2), (1, 1), (1, 1), (0, 0), 2, False),
        # ConvTranspose1d: Basic case
        (1, 1, 3, (2,), (1,), (1,), (0,), 1, True),
        # ConvTranspose1d: With stride and padding
        (2, 4, 3, (2,), (1,), (1,), (0,), 1, True),
        # ConvTranspose1d: With output_padding
        (2, 4, 3, (2,), (1,), (1,), (1,), 1, True),
    ],
)
async def test_conv_transpose(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: Any,
    padding: Any,
    dilation: Any,
    output_padding: Any,
    groups: int,
    is_1d: bool,
    dynamic: bool,
) -> None:
    """Test conv_transpose1d and conv_transpose2d operations."""
    conv_layer: Any = None
    if is_1d:
        x = torch.rand(2, in_channels, 8)
        conv_layer = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
            output_padding=output_padding[0],
            groups=groups,
            bias=True,
        )
    else:
        x = torch.rand(2, in_channels, 8, 8)
        conv_layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            groups=groups,
            bias=True,
        )

    class ConvTransposeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_transpose = conv_layer

        def forward(self, x: Tensor) -> Tensor:
            return self.conv_transpose(x)

    model = ConvTransposeModel().eval()
    # Batch (dim 0) and spatial dimensions can be dynamic; channel (dim 1) is fixed by the layer.
    if dynamic:
        dynamic_shapes = (
            {
                "x": {
                    0: torch.export.Dim("batch", min=1),
                    2: torch.export.Dim("L", min=kernel_size, max=2**20),
                }
            }
            if is_1d
            else {
                "x": {
                    0: torch.export.Dim("batch", min=1),
                    2: torch.export.Dim("H", min=kernel_size),
                    3: torch.export.Dim("W", min=kernel_size),
                }
            }
        )
    else:
        dynamic_shapes = None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "input_shape,split_sizes,dim,dtype",
    [
        # Split along dim 0 (default)
        ((10, 4), [2, 3, 5], 0, torch.float32),
        # Split along dim 1
        ((3, 10), [2, 3, 5], 1, torch.float32),
        # Split along negative dim
        ((4, 12), [3, 4, 5], -1, torch.float32),
        # 3D tensor split along dim 0
        ((6, 4, 3), [2, 4], 0, torch.float32),
        # 3D tensor split along dim 1
        ((2, 8, 3), [3, 2, 3], 1, torch.float32),
        # FP16 split
        ((8, 4), [3, 5], 0, torch.float16),
        # Single split (edge case)
        pytest.param(
            (5, 3),
            [5],
            0,
            torch.float32,
        ),
        # Two equal splits
        ((6, 4), [3, 3], 0, torch.float32),
    ],
)
async def test_split_with_sizes(
    input_shape: tuple[int, ...],
    split_sizes: list[int],
    dim: int,
    dtype: torch.dtype,
    dynamic: bool,
) -> None:
    """Test split_with_sizes operation which splits a tensor into chunks with specified sizes."""
    x = torch.rand(input_shape, dtype=dtype)

    class SplitWithSizesModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> tuple[Tensor, ...]:
            return torch.split(x, split_sizes, dim=dim)

    model = SplitWithSizesModel().eval()
    # The split dim size equals sum(split_sizes), so only non-split dims can be dynamic.
    norm_dim = dim % len(input_shape)
    dynamic_shapes = (
        {
            "x": {
                d: torch.export.Dim(f"x{d}", min=1)
                for d in range(len(input_shape))
                if d != norm_dim
            }
        }
        if dynamic
        else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "num_embeddings,embedding_dim,indices_shape,dtype,indices_dtype",
    [
        # Basic 1D indices with float32
        (100, 64, (10,), torch.float32, torch.int64),
        # 2D indices (batch of sequences) with float32
        (100, 64, (4, 8), torch.float32, torch.int32),
        # 3D indices with float32
        (50, 32, (2, 4, 6), torch.float32, torch.int64),
        # Smaller embedding table with float16
        (20, 16, (5,), torch.float16, torch.int32),
        # Larger batch with float16
        (100, 64, (8, 16), torch.float16, torch.int64),
        # Single index (edge case)
        (10, 8, (2,), torch.float32, torch.int32),
    ],
)
async def test_embedding(
    num_embeddings: int,
    embedding_dim: int,
    indices_shape: tuple[int, ...],
    dtype: torch.dtype,
    indices_dtype: torch.dtype,
    dynamic: bool,
) -> None:
    """Test embedding operation which looks up embeddings from a weight table.

    aten.embedding(weight, indices) performs table lookup where:
    - weight: 2D tensor of shape [num_embeddings, embedding_dim]
    - indices: Tensor of any shape containing indices into the embedding table
    - Returns: Tensor of shape [*indices.shape, embedding_dim]
    """

    class EmbeddingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings, embedding_dim).to(dtype)

        def forward(self, indices: Tensor) -> Tensor:
            return self.embedding(indices)

    # Create indices tensor with valid indices into the embedding table
    indices = torch.randint(0, num_embeddings, indices_shape, dtype=indices_dtype)

    model = EmbeddingModel().eval()
    dynamic_shapes = {"indices": _all_dims_dynamic(indices)} if dynamic else None
    await validate_numerical_output(
        model=model, indices=indices, dynamic_shapes=dynamic_shapes
    )


class TestSliceScatter:
    """Test suite for aten.slice_scatter → coreai.slice_update conversion."""

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "input_shape,src_shape,dim,start,end,step,dtype",
        [
            ((8, 4), (3, 4), 0, 2, 5, 1, torch.float32),  # basic: dim 0
            ((4, 8), (4, 3), 1, 1, 4, 1, torch.float32),  # dim 1
            ((10, 4), (3, 4), 0, 1, 7, 2, torch.float32),  # step > 1
            ((6, 4, 3), (2, 4, 3), 0, 2, 4, 1, torch.float32),  # 3D tensor
            ((4, 8), (4, 3), -1, 1, 4, 1, torch.float32),  # negative dim
            ((8, 4), (3, 4), 0, 2, 5, 1, torch.float16),  # FP16
            ((8, 4), (8, 4), 0, None, None, 1, torch.float32),  # None defaults
        ],
    )
    async def test_static_bounds(
        self,
        input_shape: tuple[int, ...],
        src_shape: tuple[int, ...],
        dim: int,
        start: int | None,
        end: int | None,
        step: int,
        dtype: torch.dtype,
        dynamic: bool,
    ) -> None:
        """Static start/end/step; optionally non-scattered dims made symbolic.

        When dynamic=True only dimensions other than the scatter dim are made
        symbolic, for the same reason as TestSlice.test_static_bounds.
        """
        x = torch.rand(input_shape, dtype=dtype)
        src = torch.rand(src_shape, dtype=dtype)

        class SliceScatterModel(nn.Module):
            def forward(self, x: Tensor, src: Tensor) -> Tensor:
                return torch.slice_scatter(
                    x, src, dim=dim, start=start, end=end, step=step
                )

        model = SliceScatterModel().eval()
        if dynamic:
            rank_x = len(input_shape)
            rank_src = len(src_shape)
            actual_dim = (dim + rank_x) % rank_x
            x_names = [None if i == actual_dim else f"x{i}" for i in range(rank_x)]
            src_names = [None if i == actual_dim else f"s{i}" for i in range(rank_src)]
            has_dynamic = any(x_names) or any(src_names)
            dynamic_shapes = (
                make_dynamic_shapes(x=x_names, src=src_names) if has_dynamic else None
            )
        else:
            dynamic_shapes = None
        await validate_numerical_output(
            model=model, x=x, src=src, dynamic_shapes=dynamic_shapes
        )

    @pytest.mark.parametrize(
        "input_shape,src_shape,slice_dim,dtype",
        [
            # Scatter src into input along dim 0, using src's batch size as end bound
            ((8, 4), (5, 4), 0, torch.float32),
            # Scatter along dim 1 using src's dim 1 size as the end bound
            ((4, 10), (4, 6), 1, torch.float32),
            # 3D case
            ((6, 4, 3), (4, 4, 3), 0, torch.float32),
            # FP16
            ((8, 6), (5, 6), 0, torch.float16),
        ],
    )
    async def test_dynamic_bounds(
        self,
        input_shape: tuple[int, ...],
        src_shape: tuple[int, ...],
        slice_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """Dynamic end bound derived from src's shape (fx.Node path).

        end = src.shape[slice_dim] becomes a sym_size.int fx.Node when exported
        with dynamic shapes, exercising the IR-Value path in replace_slice_scatter.
        """
        x = torch.rand(input_shape, dtype=dtype)
        src = torch.rand(src_shape, dtype=dtype)

        class DynBoundSliceScatterModel(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.dim = dim

            def forward(self, x: Tensor, src: Tensor) -> Tensor:
                # end = src.shape[dim] → becomes a sym_size.int fx.Node when dynamic
                end = src.shape[self.dim]
                return torch.slice_scatter(
                    x, src, dim=self.dim, start=0, end=end, step=1
                )

        model = DynBoundSliceScatterModel(dim=slice_dim).eval()
        src_dim_names = [f"src{i}" for i in range(len(src_shape))]
        src_dim_names[slice_dim] = "k"  # slice dim is the constrained bound
        dynamic_shapes = make_dynamic_shapes(
            x=["batch_x"] + [None] * (len(input_shape) - 1),
            src=src_dim_names,
        )
        await validate_numerical_output(
            model=model, x=x, src=src, dynamic_shapes=dynamic_shapes
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "input_shape,dim,dtype",
    [
        # 1D tensor, cumsum along dim 0
        ((10,), 0, torch.float32),
        # 2D tensor, cumsum along dim 0
        ((4, 6), 0, torch.float32),
        # 2D tensor, cumsum along dim 1
        ((4, 6), 1, torch.float32),
        # 3D tensor, cumsum along dim 0
        ((3, 4, 5), 0, torch.float32),
        # 3D tensor, cumsum along dim 2
        ((3, 4, 5), 2, torch.float32),
        # Negative dimension (cumsum along last dim)
        ((4, 6), -1, torch.float32),
        # FP16 cumsum
        ((4, 6), 0, torch.float16),
        # Integer cumsum
        ((4, 6), 1, torch.int32),
    ],
)
async def test_cumsum(
    input_shape: tuple[int, ...],
    dim: int,
    dtype: torch.dtype,
    dynamic: bool,
) -> None:
    """Test cumsum operation which computes cumulative sum along a dimension.

    aten.cumsum(input, dim) -> Tensor

    Computes the cumulative sum of elements along the given dimension.
    Output has the same shape as input.
    """
    if dtype == torch.int32:
        x = torch.randint(0, 10, input_shape, dtype=dtype)
    else:
        x = torch.rand(input_shape, dtype=dtype)

    class CumsumModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.cumsum(x, dim=dim)

    model = CumsumModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestUpsampleNearest2d:
    """Test suite for aten.upsample_nearest2d.vec → coreai.interpolate (nearest_neighbor)."""

    @pytest.mark.parametrize(
        "input_shape,output_size,scale_factor,dtype,dynamic_dims",
        [
            # Static (no dynamic dims)
            ((1, 3, 4, 4), (8, 8), None, torch.float32, ()),
            # Output size specified - non-integer scale
            ((1, 3, 4, 4), (10, 10), None, torch.float32, ()),
            # Scale factor specified - 2x upsampling
            ((1, 3, 4, 4), None, (2.0, 2.0), torch.float32, ()),
            # Scale factor specified - different scales for H and W
            ((1, 3, 4, 6), None, (2.0, 3.0), torch.float32, ()),
            # Scale factor specified - downsampling
            ((1, 3, 8, 8), None, (0.5, 0.5), torch.float32, ()),
            # Single channel
            ((1, 1, 4, 4), (8, 8), None, torch.float32, ()),
            # FP16 precision
            ((1, 3, 4, 4), (8, 8), None, torch.float16, ()),
            # Dynamic batch with output_size
            ((2, 3, 4, 4), (8, 8), None, torch.float32, (0,)),
            # Dynamic batch + channel with output_size
            ((2, 3, 4, 4), (8, 8), None, torch.float32, (0, 1)),
            # Dynamic spatial with output_size
            ((2, 3, 4, 4), (8, 8), None, torch.float32, (2, 3)),
            # All dims dynamic with output_size
            ((2, 3, 4, 4), (8, 8), None, torch.float32, (0, 1, 2, 3)),
            # Dynamic batch with scale_factors
            ((2, 3, 4, 4), None, (2.0, 2.0), torch.float32, (0,)),
            # Dynamic batch + channel with scale_factors
            ((2, 3, 4, 4), None, (2.0, 2.0), torch.float32, (0, 1)),
            # Dynamic spatial with scale_factors
            ((2, 3, 4, 4), None, (2.0, 2.0), torch.float32, (2, 3)),
            # All dims dynamic with scale_factors
            ((2, 3, 4, 4), None, (2.0, 2.0), torch.float32, (0, 1, 2, 3)),
            # Dynamic spatial with asymmetric scale_factors
            ((2, 3, 4, 6), None, (2.0, 3.0), torch.float32, (2, 3)),
        ],
    )
    async def test_basic(
        self,
        input_shape: tuple[int, ...],
        output_size: tuple[int, int] | None,
        scale_factor: tuple[float, float] | None,
        dtype: torch.dtype,
        dynamic_dims: tuple[int, ...],
    ) -> None:
        """Test upsample_nearest2d with static output_size or scale_factor."""
        x = torch.rand(input_shape, dtype=dtype)

        class UpsampleNearest2dModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.interpolate(
                    x, size=output_size, scale_factor=scale_factor, mode="nearest"
                )

        model = UpsampleNearest2dModel().eval()
        if not dynamic_dims:
            dynamic_shapes = None
        else:
            dim_names = {0: "batch", 1: "channels", 2: "height", 3: "width"}
            dynamic_shapes = make_dynamic_shapes(
                x={d: dim_names[d] for d in dynamic_dims}
            )
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)

    async def test_dynamic_output_size(self) -> None:
        """Test upsample_nearest2d when output_size comes from dynamic tensor shapes.

        When F.interpolate is called with size derived from another tensor's shape
        (e.g. target.shape[2:]), the exported FX graph passes Node objects rather than
        static ints as the output_size. The input x has static spatial dims.
        """

        class DynamicUpsampleNearest(nn.Module):
            def forward(self, x: Tensor, target: Tensor) -> Tensor:
                return torch.nn.functional.interpolate(
                    x, size=(target.shape[2], target.shape[3]), mode="nearest"
                )

        model = DynamicUpsampleNearest().eval()
        x = torch.rand(1, 32, 8, 8)
        target = torch.rand(1, 32, 16, 16)
        dynamic_shapes = make_dynamic_shapes(
            x={},
            target={2: "h_out", 3: "w_out"},
        )
        await validate_numerical_output(
            model=model, x=x, target=target, dynamic_shapes=dynamic_shapes
        )

    def test_round_symfloat_size(self) -> None:
        """Regression for ``upsample_build_output_shape_dynamic``: when
        the output_size operands are SymInts that come from
        ``round(SymFloat)`` shape arithmetic — e.g.
        ``round((num_tokens / aspect_ratio) ** 0.5) * 14`` — the lowering
        must reshape/cast each (out_h, out_w) operand to rank-1 int32
        before the concat that builds the output shape. Without the fix,
        ``coreai.concat`` raises ``ValueError: Operation creation failed``
        on the mixed-rank / mixed-element-type inputs."""

        class RoundSymFloatUpsample(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                aspect = x.shape[3] / x.shape[2]
                h = round((1800 / aspect) ** 0.5)
                w = round((1800 * aspect) ** 0.5)
                return torch.nn.functional.interpolate(
                    x, size=(h * 14, w * 14), mode="nearest"
                )

        x = torch.rand(1, 3, 28, 28, dtype=torch.float16)
        dynamic_shapes = {
            "x": {
                2: torch.export.Dim("h", min=14),
                3: torch.export.Dim("w", min=14),
            }
        }
        program = torch.export.export(
            RoundSymFloatUpsample().eval(), args=(x,), dynamic_shapes=dynamic_shapes
        ).run_decompositions()
        # Convert must not raise. Pre-fix this raised ``ValueError:
        # Operation creation failed`` from ``coreai.concat``.
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        coreai_program.optimize()
        # The output_shape concat must take three rank-1 si32 operands —
        # the (N, C) slice from x's get_shape, plus normalised out_h and
        # out_w. Pre-fix, out_h/out_w arrived as rank-1 f32 (from
        # round(SymFloat) arithmetic) which broke the concat verifier.
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK-LABEL: coreai.graph @main
                // CHECK-SAME:    %[[ARG0:.*]]: tensor<1x3x?x?xf16>
                //
                // out_h, out_w come in as rank-1 f32 from round-of-symfloat
                // arithmetic; the fix casts each to rank-1 si32 to match the
                // (N, C) slice of get_shape (which is also rank-1 si32):
                // CHECK:         %[[H:.+]] = coreai.cast {{.*}} : tensor<1xf32> to tensor<1xsi32>
                // CHECK:         %[[W:.+]] = coreai.cast {{.*}} : tensor<1xf32> to tensor<1xsi32>
                //
                // Output-shape concat: non_spatial (rank-1 si32, length 2)
                // + out_h (rank-1 si32) + out_w (rank-1 si32) → rank-1 si32 length 4:
                // CHECK:         %[[OUT_SHAPE:.+]] = coreai.concat {{.*}} : (tensor<si32>, tensor<2xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<4xsi32>
                //
                // The shape feeds nearest-neighbor interpolate (whole op
                // matched on one line because the mode attribute is part of
                // the same SSA statement):
                // CHECK:         coreai.interpolate {{.+}}, %[[OUT_SHAPE]], {{.+}} {interpolation_mode = #coreai.interpolation_mode<nearest_neighbor>}
            """,
        )


class TestUpsampleBilinear2d:
    """Test suite for aten.upsample_bilinear2d.vec → coreai.interpolate (linear)."""

    @pytest.mark.parametrize(
        "input_shape,output_size,scale_factor,align_corners,dtype,dynamic_dims",
        [
            # Static (no dynamic dims)
            ((1, 3, 4, 4), (8, 8), None, False, torch.float32, ()),
            # Output size specified - 2x upsampling with align_corners=True
            ((1, 3, 4, 4), (8, 8), None, True, torch.float32, ()),
            # Scale factor specified - 2x upsampling with align_corners=False
            ((1, 3, 4, 4), None, (2.0, 2.0), False, torch.float32, ()),
            # Scale factor specified - 2x upsampling with align_corners=True
            ((1, 3, 4, 4), None, (2.0, 2.0), True, torch.float32, ()),
            # Different scales for H and W
            ((1, 3, 4, 6), None, (2.0, 3.0), False, torch.float32, ()),
            # Downsampling
            ((1, 3, 8, 8), None, (0.5, 0.5), False, torch.float32, ()),
            # FP16 precision
            ((1, 3, 4, 4), (8, 8), None, False, torch.float16, ()),
            # Dynamic batch with output_size, align_corners=False
            ((2, 3, 4, 4), (8, 8), None, False, torch.float32, (0,)),
            # Dynamic batch with output_size, align_corners=True
            ((2, 3, 4, 4), (8, 8), None, True, torch.float32, (0,)),
            # Dynamic batch + channel with output_size
            ((2, 3, 4, 4), (8, 8), None, False, torch.float32, (0, 1)),
            # Dynamic spatial with output_size, align_corners=False
            ((2, 3, 4, 4), (8, 8), None, False, torch.float32, (2, 3)),
            # Dynamic spatial with output_size, align_corners=True
            ((2, 3, 4, 4), (8, 8), None, True, torch.float32, (2, 3)),
            # All dims dynamic with output_size
            ((2, 3, 4, 4), (8, 8), None, False, torch.float32, (0, 1, 2, 3)),
            # Dynamic batch with scale_factors
            ((2, 3, 4, 4), None, (2.0, 2.0), False, torch.float32, (0,)),
            # Dynamic batch + channel with scale_factors
            ((2, 3, 4, 4), None, (2.0, 2.0), False, torch.float32, (0, 1)),
            # Dynamic spatial with scale_factors, align_corners=False
            ((2, 3, 4, 4), None, (2.0, 2.0), False, torch.float32, (2, 3)),
            # Dynamic spatial with scale_factors, align_corners=True
            ((2, 3, 4, 4), None, (2.0, 2.0), True, torch.float32, (2, 3)),
            # All dims dynamic with scale_factors
            ((2, 3, 4, 4), None, (2.0, 2.0), False, torch.float32, (0, 1, 2, 3)),
            # Dynamic spatial with asymmetric scale_factors
            ((2, 3, 4, 6), None, (2.0, 3.0), False, torch.float32, (2, 3)),
        ],
    )
    async def test_basic(
        self,
        input_shape: tuple[int, ...],
        output_size: tuple[int, int] | None,
        scale_factor: tuple[float, float] | None,
        align_corners: bool,
        dtype: torch.dtype,
        dynamic_dims: tuple[int, ...],
    ) -> None:
        """Test upsample_bilinear2d with static output_size or scale_factor."""
        x = torch.rand(input_shape, dtype=dtype)

        class UpsampleBilinear2dModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor) -> Tensor:
                return torch.nn.functional.interpolate(
                    x,
                    size=output_size,
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=align_corners,
                )

        model = UpsampleBilinear2dModel().eval()
        if not dynamic_dims:
            dynamic_shapes = None
        else:
            dim_names = {0: "batch", 1: "channels", 2: "height", 3: "width"}
            dynamic_shapes = make_dynamic_shapes(
                x={d: dim_names[d] for d in dynamic_dims}
            )
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)

    async def test_dynamic_output_size(self) -> None:
        """Test upsample_bilinear2d when output_size comes from dynamic tensor shapes.

        When F.interpolate is called with size derived from another tensor's shape
        (e.g. target.shape[2:]), the exported FX graph passes Node objects rather than
        static ints as the output_size. The input x has static spatial dims.
        """

        class DynamicUpsampleBilinear(nn.Module):
            def forward(self, x: Tensor, target: Tensor) -> Tensor:
                return torch.nn.functional.interpolate(
                    x,
                    size=(target.shape[2], target.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

        model = DynamicUpsampleBilinear().eval()
        x = torch.rand(1, 32, 8, 8)
        target = torch.rand(1, 32, 16, 16)
        dynamic_shapes = make_dynamic_shapes(
            x={},
            target={2: "h_out", 3: "w_out"},
        )
        await validate_numerical_output(
            model=model, x=x, target=target, dynamic_shapes=dynamic_shapes
        )

    def test_round_symfloat_size(self) -> None:
        """Regression for ``upsample_build_output_shape_dynamic``: when
        the output_size operands are SymInts that come from
        ``round(SymFloat)`` shape arithmetic — e.g.
        ``round((num_tokens / aspect_ratio) ** 0.5) * 14`` — the lowering
        must reshape/cast each (out_h, out_w) operand to rank-1 int32
        before the concat that builds the output shape. Without the fix,
        ``coreai.concat`` raises ``ValueError: Operation creation failed``
        on the mixed-rank / mixed-element-type inputs."""

        class RoundSymFloatUpsample(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                aspect = x.shape[3] / x.shape[2]
                h = round((1800 / aspect) ** 0.5)
                w = round((1800 * aspect) ** 0.5)
                return torch.nn.functional.interpolate(
                    x,
                    size=(h * 14, w * 14),
                    mode="bilinear",
                    align_corners=False,
                )

        x = torch.rand(1, 3, 28, 28, dtype=torch.float16)
        dynamic_shapes = {
            "x": {
                2: torch.export.Dim("h", min=14),
                3: torch.export.Dim("w", min=14),
            }
        }
        program = torch.export.export(
            RoundSymFloatUpsample().eval(), args=(x,), dynamic_shapes=dynamic_shapes
        ).run_decompositions()
        # Convert must not raise. Pre-fix this raised ``ValueError:
        # Operation creation failed`` from ``coreai.concat``.
        coreai_program = TorchConverter().add_exported_program(program).to_coreai()
        coreai_program.optimize()
        # Same shape concat as the nearest case; only the interpolation
        # mode tag differs. The fix is in the shape-build path so it
        # applies identically here.
        filecheck_pattern(
            str(coreai_program),
            check_file="""
                // CHECK-LABEL: coreai.graph @main
                // CHECK-SAME:    %[[ARG0:.*]]: tensor<1x3x?x?xf16>
                //
                // out_h, out_w cast from rank-1 f32 to rank-1 si32:
                // CHECK:         %[[H:.+]] = coreai.cast {{.*}} : tensor<1xf32> to tensor<1xsi32>
                // CHECK:         %[[W:.+]] = coreai.cast {{.*}} : tensor<1xf32> to tensor<1xsi32>
                //
                // Output-shape concat: all rank-1 si32 → rank-1 si32 length 4:
                // CHECK:         %[[OUT_SHAPE:.+]] = coreai.concat {{.*}} : (tensor<si32>, tensor<2xsi32>, tensor<1xsi32>, tensor<1xsi32>) -> tensor<4xsi32>
                //
                // The shape feeds bilinear interpolate (whole op matched on
                // one line because the mode attribute is part of the same
                // SSA statement):
                // CHECK:         coreai.interpolate {{.+}}, %[[OUT_SHAPE]], {{.+}} {interpolation_mode = #coreai.interpolation_mode<linear>}
            """,
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dims,keepdim,dtype",
    [
        # Single dimension reduction
        (torch.rand(3, 4, 5), [1], False, None),
        (torch.rand(3, 4, 5), [1], True, None),
        # Multiple dimensions reduction
        (torch.rand(2, 3, 4), [0, 2], False, None),
        (torch.rand(2, 3, 4), [0, 2], True, None),
        # Negative dimension
        (torch.rand(2, 3, 4), [-1], False, None),
        (torch.rand(3, 4, 5), [-2, -1], False, None),
        # All dimensions reduction (explicitly specified)
        (torch.rand(2, 3, 4), [0, 1, 2], False, None),
        # 2D tensor
        (torch.rand(4, 5), [0], True, None),
        # Integer tensor with implicit dtype conversion
        (torch.randint(0, 10, (3, 4), dtype=torch.int32), [1], False, None),
        # Reduce all dims via empty dim list (torch.sum(x))
        (torch.rand(3, 4), [], False, None),
        (torch.rand(2, 3, 5), [], False, None),
        (torch.rand(6), [], False, None),
        (torch.rand(2, 3, 4, dtype=torch.float16), [], False, None),
        # Explicit dtype upcast: int32 input → float32 output
        (torch.randint(0, 10, (3, 4), dtype=torch.int32), [0], False, torch.float32),
        (
            torch.randint(0, 10, (2, 3, 4), dtype=torch.int32),
            [0, 2],
            False,
            torch.float32,
        ),
    ],
)
async def test_sum_dim_intlist(
    x: Tensor, dims: list[int], keepdim: bool, dtype: torch.dtype | None, dynamic: bool
) -> None:
    """Test sum operation: per-dim, all-dims, keepdim, negative dims, dtype upcast."""

    class SumModel(nn.Module):
        def __init__(
            self, dims: list[int], keepdim: bool, dtype: torch.dtype | None
        ) -> None:
            super().__init__()
            self.dims = dims
            self.keepdim = keepdim
            self.dtype = dtype

        def forward(self, x: Tensor) -> Tensor:
            if not self.dims:
                return torch.sum(x)
            if self.dtype is not None:
                return torch.sum(
                    x, dim=self.dims, keepdim=self.keepdim, dtype=self.dtype
                )
            return torch.sum(x, dim=self.dims, keepdim=self.keepdim)

    model = SumModel(dims, keepdim, dtype).eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dim,keepdim",
    [
        # Single dim reduction along each axis
        (torch.rand(3, 4, 5), 0, False),
        (torch.rand(3, 4, 5), 1, False),
        (torch.rand(3, 4, 5), 2, False),
        # keepdim=True
        (torch.rand(3, 4, 5), 1, True),
        # Negative dim
        (torch.rand(2, 3, 4), -1, False),
        (torch.rand(2, 3, 4), -2, True),
        # 2D tensor
        (torch.rand(4, 5), 0, False),
        (torch.rand(4, 5), 1, True),
        # FP16
        (torch.rand(3, 4, dtype=torch.float16), 0, False),
    ],
)
async def test_prod_dim_int(x: Tensor, dim: int, keepdim: bool, dynamic: bool) -> None:
    class ProdModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.prod(x, dim=dim, keepdim=keepdim)

    model = ProdModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.rand(3, 4, 5),
        torch.rand(4, 5),
        torch.rand(10),
        torch.rand(2, 3, dtype=torch.float16),
    ],
)
async def test_prod_default(x: Tensor, dynamic: bool) -> None:
    class ProdModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.prod(x)

    model = ProdModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dims",
    [
        # Same rank: len(dims) == x.ndim
        (torch.rand(2, 3), (2, 3)),
        (torch.rand(3, 4, 5), (1, 2, 3)),
        # keepdim-style: tile by 1 along some axes
        (torch.rand(2, 3), (1, 4)),
        # len(dims) < x.ndim: dims padded with leading 1s
        (torch.rand(2, 3), (3,)),
        (torch.rand(3, 4, 5), (2,)),
        # len(dims) > x.ndim: x prepended with size-1 axes
        (torch.rand(3), (2, 3)),
        (torch.rand(2, 3), (1, 2, 3)),
        # 1D input
        (torch.rand(5), (4,)),
        # FP16
        (torch.rand(2, 3, dtype=torch.float16), (2, 2)),
    ],
)
async def test_tile(x: Tensor, dims: tuple[int, ...], dynamic: bool) -> None:
    class TileModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.tile(x, dims)

    model = TileModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,k,dim,largest",
    [
        # 1D input
        (torch.rand(8), 3, 0, True),
        # 2D: top-k along last dim
        (torch.rand(4, 6), 3, 1, True),
        # 2D: top-k along first dim
        (torch.rand(4, 6), 2, 0, True),
        # 2D: smallest-k (largest=False)
        (torch.rand(4, 6), 3, -1, False),
        # 3D: middle dim, k=1
        (torch.rand(3, 5, 4), 1, 1, True),
        # 3D: top-k along last dim
        (torch.rand(3, 5, 4), 2, 2, True),
        # 3D: negative dim
        (torch.rand(3, 5, 4), 2, -1, False),
        # FP16 — seeded to avoid flaky ties in low-precision sort
        (
            torch.rand(
                4, 6, dtype=torch.float16, generator=torch.Generator().manual_seed(42)
            ),
            3,
            1,
            True,
        ),
    ],
)
async def test_topk(x: Tensor, k: int, dim: int, largest: bool, dynamic: bool) -> None:
    """Test topk operation returning (values, indices)."""

    class TopkModel(nn.Module):
        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return torch.topk(x, k, dim=dim, largest=largest)

    model = TopkModel().eval()
    if dynamic:
        norm_dim = dim % x.dim()
        dynamic_shapes = {
            "x": {
                d: torch.export.Dim(f"x{d}", min=k if d == norm_dim else 1)
                for d in range(x.dim())
            }
        }
    else:
        dynamic_shapes = None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "x",
    [
        # Float32 tensors with infinity values
        torch.tensor([1.0, float("inf"), -float("inf"), 0.0, float("nan"), 2.5]),
        # 2D tensor with mixed values
        torch.tensor([[float("inf"), 1.0], [-float("inf"), 0.0]], dtype=torch.float32),
        # 3D tensor with infinities
        torch.tensor(
            [
                [[float("inf"), 1.0], [2.0, -float("inf")]],
                [[0.0, 3.0], [float("nan"), float("inf")]],
            ],
            dtype=torch.float32,
        ),
        # Float16 tensor with infinities
        torch.tensor([1.0, float("inf"), -float("inf"), 0.0], dtype=torch.float16),
        # 2D Float16 tensor
        torch.tensor([[float("inf"), -float("inf")], [1.0, 2.0]], dtype=torch.float16),
        # Regular tensor without infinities (all False expected)
        torch.rand(3, 4, dtype=torch.float32),
    ],
)
@pytest.mark.parametrize("dynamic_dims", [tuple(), (0,)])
async def test_isinf(x: Tensor, dynamic_dims: tuple[int]) -> None:
    """Test isinf operation which detects positive and negative infinity values.

    Returns True where input is positive or negative infinity, False otherwise.
    """

    class IsInfModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return torch.isinf(x)

    model = IsInfModel().eval()
    dynamic_shapes = make_dynamic_shapes(x={d: f"d{d}" for d in dynamic_dims})
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "x,dims",
    [
        (torch.randn(8, 32), [0]),
        (torch.randn(8, 32), [1]),
        (torch.randn(8, 32), [0, 1]),
        (torch.randn(8, 5, 14, 16), [0]),
        (torch.randn(8, 5, 14, 16), [1]),
        (torch.randn(8, 5, 14, 16), [2]),
        (torch.randn(8, 5, 14, 16), [3]),
        (torch.randn(8, 5, 14, 16), [0, 3]),
        (torch.randn(8, 5, 14, 16), [1, 0]),
        (torch.randn(8, 5, 14, 16), [0, 1, 2, 3]),
        (torch.randn(8, 5, 14, 16), [0, 2]),
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
async def test_amax(x: Tensor, dims: list[int], keepdim: bool):
    class AmaxModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            # Add second output without `keepdim` arg to ensure logic surrounding the length of the node args list is sound
            return torch.amax(x, dims, keepdim=keepdim), torch.amax(x, dims)

    model = AmaxModel()
    await validate_numerical_output(model=model, x=x)


@pytest.mark.parametrize(
    "x,dims",
    [
        (torch.randn(8, 32), [0]),
        (torch.randn(8, 32), [1]),
        (torch.randn(8, 32), [0, 1]),
        (torch.randn(8, 5, 14, 16), [0]),
        (torch.randn(8, 5, 14, 16), [1]),
        (torch.randn(8, 5, 14, 16), [2]),
        (torch.randn(8, 5, 14, 16), [3]),
        (torch.randn(8, 5, 14, 16), [0, 3]),
        (torch.randn(8, 5, 14, 16), [1, 0]),
        (torch.randn(8, 5, 14, 16), [0, 1, 2, 3]),
        (torch.randn(8, 5, 14, 16), [0, 2]),
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
async def test_amin(x: Tensor, dims: list[int], keepdim: bool):
    class AminModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            # Add second output without `keepdim` arg to ensure logic surrounding the length of the node args list is sound
            return torch.amin(x, dims, keepdim=keepdim), torch.amax(x, dims)

    model = AminModel()
    await validate_numerical_output(model=model, x=x)


@pytest.mark.parametrize(
    "x,shape",
    [
        # Flatten 2D to 1D
        (torch.rand(2, 6), [12]),
        # Unflatten 1D to 2D
        (torch.rand(12), [3, 4]),
        # Reshape 3D to 2D
        (torch.rand(2, 3, 4), [6, 4]),
        # Reshape 3D to different 3D
        (torch.rand(2, 3, 4), [4, 2, 3]),
        # Flatten all dims to 1D
        (torch.rand(2, 3, 4), [24]),
        # Add a trailing size-1 dim
        (torch.rand(3, 4), [3, 4, 1]),
        # -1 infers the last dim
        (torch.rand(2, 3, 4), [6, -1]),
        # -1 infers the first dim
        (torch.rand(2, 3, 4), [-1, 4]),
        # -1 infers a middle dim
        (torch.rand(2, 3, 4), [2, -1, 2]),
        # -1 flattens everything into one dim
        (torch.rand(2, 3, 4), [-1]),
    ],
)
async def test_view(x: Tensor, shape: list[int]) -> None:
    class ViewModel(nn.Module):
        def __init__(self, shape: list[int]) -> None:
            super().__init__()
            self.shape = shape

        def forward(self, x: Tensor) -> Tensor:
            return x.view(self.shape)

    model = ViewModel(shape).eval()
    await validate_numerical_output(model=model, x=x)


@pytest.mark.parametrize(
    "x, shape_fn, dynamic_dims",
    [
        # Keep dynamic batch dim, flatten static trailing dims
        (torch.rand(4, 3, 5), lambda x: [x.shape[0], 15], {0}),
        # Keep dynamic batch dim, flatten more static trailing dims
        (torch.rand(4, 2, 3, 5), lambda x: [x.shape[0], 30], {0}),
        # Dynamic batch, reshape remaining static dims into different shape
        (torch.rand(4, 4, 6), lambda x: [x.shape[0], 8, 3], {0}),
        # FP16
        (torch.rand(4, 2, 6, dtype=torch.float16), lambda x: [x.shape[0], 12], {0}),
        # -1 infers the product of static trailing dims (resolved to 15 by torch.export)
        (torch.rand(4, 3, 5), lambda x: [x.shape[0], -1], {0}),
        # -1 in the middle infers a static dim (resolved to 6 by torch.export)
        (torch.rand(4, 2, 3, 5), lambda x: [x.shape[0], -1, 5], {0}),
        # Multiple sym_size refs: split static middle dims while keeping both dynamic dims
        (torch.rand(4, 6, 5), lambda x: [x.shape[0], 3, 2, x.shape[-1]], {0, 2}),
        # Multiple sym_size refs with -1: -1 infers the split factor (resolved to 2)
        (torch.rand(4, 6, 5), lambda x: [x.shape[0], 3, -1, x.shape[-1]], {0, 2}),
    ],
)
async def test_view_dynamic(x: Tensor, shape_fn: Any, dynamic_dims: set[int]) -> None:
    fn = shape_fn

    class ViewDynamicModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x.view(fn(x))

    model = ViewDynamicModel().eval()
    dynamic_shapes = {"x": {d: torch.export.Dim(f"x{d}", min=1) for d in dynamic_dims}}
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x,dim",
    [
        (torch.rand(3, 4), 0),
        (torch.rand(3, 4), 1),
        (torch.rand(2, 5, 6), 2),
    ],
)
async def test_sym_size_int(x: Tensor, dim: int, dynamic: bool) -> None:
    """Test aten.sym_size.int — reading a tensor's runtime size along a given dimension.

    x.size(dim) produces a sym_size.int node in the exported graph when shapes are dynamic.
    The result is used to construct a zeros tensor, forcing the size to be materialised.
    """

    class SymSizeModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.zeros(x.size(dim))

    model = SymSizeModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "x_shape,y_shape",
    [
        ((5, 3), (4, 3)),
        ((3, 4), (7, 4)),
        ((6, 2, 3), (4, 2, 3)),
    ],
)
async def test_sym_min(x_shape: tuple[int, ...], y_shape: tuple[int, ...]) -> None:
    """Test torch.sym_min — element-wise minimum of two symbolic integer sizes.

    min(x.shape[0], y.shape[0]) produces a sym_min node in the exported graph
    when both batch dimensions are dynamic.
    """

    class SymMinModel(nn.Module):
        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            n = min(x.shape[0], y.shape[0])
            return x[:n] + y[:n]

    model = SymMinModel().eval()
    x = torch.rand(x_shape)
    y = torch.rand(y_shape)

    batch_x = torch.export.Dim("batch_x", min=1)
    batch_y = torch.export.Dim("batch_y", min=1)
    dynamic_shapes = {"x": {0: batch_x}, "y": {0: batch_y}}

    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.parametrize(
    "x_shape",
    [
        (3, 4),
        (5, 8),
        (2, 3, 6),
    ],
)
async def test_sym_float(x_shape: tuple[int, ...]) -> None:
    """Test torch.sym_float — converting a SymInt (dynamic dim) to a SymFloat scalar.

    Python's float() on a SymInt forces specialization, so the model uses
    torch.sym_float explicitly to keep the dim symbolic and emit a sym_float
    node in the exported graph.
    """

    class SymFloatModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x * torch.sym_float(x.shape[-1])

    model = SymFloatModel().eval()
    x = torch.rand(x_shape)
    dynamic_shapes = {"x": _all_dims_dynamic(x)}
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("x", [torch.randn(1, 4 * 9 * 16, 32, 32)])
@pytest.mark.parametrize("upscale_factor", [2, 3, 4])
@pytest.mark.parametrize("dynamic_dims", [tuple(), (2,), (3,), (2, 3)])
async def test_pixel_shuffle(x: Tensor, upscale_factor: int, dynamic_dims: tuple[int]):
    class PixelShuffleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ps = nn.PixelShuffle(upscale_factor)

        def forward(self, x):
            return self.ps(x)

    model = PixelShuffleModel().eval()
    dynamic_shapes = {"x": [torch.export.Dim.STATIC for _ in range(len(x.shape))]}
    for d in dynamic_dims:
        dynamic_shapes["x"][d] = torch.export.Dim(f"dim_{d}")
    await validate_numerical_output(
        model=model,
        x=x,
        dynamic_shapes=dynamic_shapes,
        remove_decomps=[torch.ops.aten.pixel_shuffle.default],
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (3, 4),
        (2, 3, 4),
    ],
)
async def test_complex(
    shape: tuple[int, ...], dtype: torch.dtype, dynamic: bool
) -> None:
    """Test torch.complex creating a complex tensor from real and imaginary parts."""

    class ComplexModel(nn.Module):
        def forward(self, real: Tensor, imag: Tensor) -> Tensor:
            c = torch.complex(real, imag)
            return torch.view_as_real(c)

    model = ComplexModel().eval()
    real = torch.randn(shape, dtype=dtype)
    imag = torch.randn(shape, dtype=dtype)
    dynamic_shapes = (
        {"real": _all_dims_dynamic(real), "imag": _all_dims_dynamic(imag)}
        if dynamic
        else None
    )
    await validate_numerical_output(
        model=model, real=real, imag=imag, dynamic_shapes=dynamic_shapes
    )


class TestPolar:
    """Tests for torch.polar converting polar coordinates to complex tensors."""

    class PolarModel(nn.Module):
        def forward(self, abs_val: Tensor, angle: Tensor) -> Tensor:
            return torch.view_as_real(torch.polar(abs_val, angle))

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "shape",
        [
            (4,),
            (3, 4),
            (2, 3, 4),
        ],
    )
    async def test_basic(self, shape: tuple[int, ...], dynamic: bool) -> None:
        model = self.PolarModel().eval()
        abs_val = torch.rand(shape)
        angle = torch.randn(shape)
        dynamic_shapes = (
            {
                "abs_val": _all_dims_dynamic(abs_val),
                "angle": _all_dims_dynamic(angle),
            }
            if dynamic
            else None
        )
        await validate_numerical_output(
            model=model,
            abs_val=abs_val,
            angle=angle,
            dynamic_shapes=dynamic_shapes,
        )

    async def test_zeros_abs(self) -> None:
        model = self.PolarModel().eval()
        abs_val = torch.zeros(3, 4)
        angle = torch.randn(3, 4)
        await validate_numerical_output(model=model, abs_val=abs_val, angle=angle)

    async def test_zeros_angle(self) -> None:
        model = self.PolarModel().eval()
        abs_val = torch.rand(3, 4)
        angle = torch.zeros(3, 4)
        await validate_numerical_output(model=model, abs_val=abs_val, angle=angle)

    async def test_large_angle(self) -> None:
        model = self.PolarModel().eval()
        abs_val = torch.rand(5)
        angle = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
        await validate_numerical_output(model=model, abs_val=abs_val, angle=angle)

    async def test_negative_angle(self) -> None:
        model = self.PolarModel().eval()
        abs_val = torch.rand(4)
        angle = torch.tensor([-math.pi, -math.pi / 2, 0.0, math.pi / 4])
        await validate_numerical_output(model=model, abs_val=abs_val, angle=angle)

    async def test_broadcast_shapes(self) -> None:
        model = self.PolarModel().eval()
        abs_val = torch.rand(3, 4)
        angle = torch.randn(4)
        await validate_numerical_output(model=model, abs_val=abs_val, angle=angle)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (3, 4),
        (2, 3, 4),
    ],
)
async def test_view_as_real(
    shape: tuple[int, ...], dtype: torch.dtype, dynamic: bool
) -> None:
    """Test torch.view_as_real converting a complex tensor to a real tensor with a trailing size-2 dimension."""

    class ViewAsRealModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.view_as_real(x)

    model = ViewAsRealModel().eval()
    x = torch.complex(torch.randn(shape, dtype=dtype), torch.randn(shape, dtype=dtype))
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    "x,shape",
    [
        (torch.rand(2, 12), [2, 3, 4]),
        (torch.rand(3, 4, 5), [3, 20]),
        (torch.rand(6, 4), [24]),
    ],
)
async def test_unsafe_view(x: Tensor, shape: list[int]) -> None:
    """Test aten._unsafe_view — equivalent to view but skips safety checks."""

    class UnsafeViewModel(nn.Module):
        def __init__(self, shape: list[int]) -> None:
            super().__init__()
            self.shape = shape

        def forward(self, x: Tensor) -> Tensor:
            return torch.ops.aten._unsafe_view(x, self.shape)

    model = UnsafeViewModel(shape).eval()
    await validate_numerical_output(model=model, x=x)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (3, 4),
        (2, 3, 4),
    ],
)
async def test_view_as_complex(
    shape: tuple[int, ...], dtype: torch.dtype, dynamic: bool
) -> None:
    """Test torch.view_as_complex converting a real tensor [..., 2] to a complex tensor [...]。

    float16 input produces complex<f16> (torch.complex32); check_result_type must
    accept complex<f16> when the FX metadata records complex<f32> (torch.complex64),
    which happens when an f16-cast ExportedProgram is imported via TorchConverter.
    """

    class ViewAsComplexModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.view_as_complex(x)

    class ViewAsComplexF16Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            # view_as_real roundtrip: keeps output as float so numpy can compare.
            # float16 input produces complex<f16> internally, exercising the
            # complex64->complex32 narrowing in check_result_type.
            return torch.view_as_real(torch.view_as_complex(x))

    model = (
        ViewAsComplexModel() if dtype == torch.float32 else ViewAsComplexF16Model()
    ).eval()
    x = torch.randn(*shape, 2, dtype=dtype)
    dynamic_shapes = (
        make_dynamic_shapes(x={i: f"d{i}" for i in range(x.dim() - 1)})
        if dynamic
        else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (3, 4),
        (2, 3, 4),
    ],
)
async def test_view_as_real_copy(
    shape: tuple[int, ...], dtype: torch.dtype, dynamic: bool
) -> None:
    """Test aten.view_as_real_copy — same as view_as_real but produces a copy."""

    class ViewAsRealCopyModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.ops.aten.view_as_real_copy(x)

    model = ViewAsRealCopyModel().eval()
    x = torch.complex(torch.randn(shape, dtype=dtype), torch.randn(shape, dtype=dtype))
    dynamic_shapes = (
        make_dynamic_shapes(x={i: f"d{i}" for i in range(x.dim())}) if dynamic else None
    )
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.control_flow
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
@pytest.mark.parametrize("positive", [True, False])
@pytest.mark.parametrize("multi_output", [False, True])
async def test_cond(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    positive: bool,
    dynamic: bool,
    multi_output: bool,
) -> None:
    """Test torch.cond higher-order control flow."""
    sign = 1.0 if positive else -1.0
    x = torch.full(shape, sign, dtype=dtype)
    y = torch.full(shape, 0.5, dtype=dtype)

    if multi_output:

        class CondModel(nn.Module):
            def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
                def true_fn(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
                    return x + y, x * y

                def false_fn(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
                    return x - y, x / y

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x, y])

    else:

        class CondModel(nn.Module):  # type: ignore[no-redef]
            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                def true_fn(x: Tensor, y: Tensor) -> Tensor:
                    return x + y

                def false_fn(x: Tensor, y: Tensor) -> Tensor:
                    return x - y

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x, y])

    model = CondModel().eval()
    dims = _all_dims_dynamic(x)
    dynamic_shapes = {"x": dims, "y": dims} if dynamic else None
    await validate_numerical_output(
        model=model, x=x, y=y, dynamic_shapes=dynamic_shapes
    )


@pytest.mark.control_flow
@pytest.mark.parametrize(
    ("x", "dynamic_shapes"),
    [
        (torch.rand(5), None),  # Static shape
        (
            torch.rand(10),
            {"x": {0: torch.export.Dim("B", min=1, max=100)}},
        ),  # Dynamic batch dimension
    ],
)
async def test_while_loop(
    x: Tensor,
    dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None,
) -> None:
    """Test torch.ops.higher_order.while_loop with static and dynamic dimensions."""

    class WhileLoopModel(nn.Module):
        @staticmethod
        def _cond(iteration: Tensor, _value: Tensor, max_iterations: Tensor) -> Tensor:
            return iteration < max_iterations

        @staticmethod
        def _body(
            iteration: Tensor, value: Tensor, _max_iterations: Tensor
        ) -> tuple[Tensor, Tensor]:
            return iteration + 1, value + 1

        def forward(self, x: Tensor) -> Tensor:
            max_iterations = torch.tensor(x.shape[0])
            init_state = (torch.tensor(0), x)
            _, final_value = torch.ops.higher_order.while_loop(
                self._cond, self._body, init_state, (max_iterations,)
            )
            return final_value

    model = WhileLoopModel().eval()
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize(
    ("input_shape", "spatial_dims"),
    [
        # rank 2: (C, L)
        ((4, 16), 1),
        # rank 3: (N, C, L)
        ((2, 4, 16), 1),
        # rank 3: (C, H, W)
        ((16, 32, 64), 2),
        # rank 4: (N, C, H, W)
        ((1, 3, 8, 8), 2),
        ((2, 3, 8, 8), 2),
        ((1, 8, 16, 16), 2),
        # rank 4: (C, D, H, W)
        ((8, 8, 8, 8), 3),
        # rank 5: (N, C, D, H, W)
        ((2, 3, 4, 8, 8), 3),
    ],
)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("track_running_stats", [True, False])
@pytest.mark.parametrize("dynamic_type", ["dynamic_spatial", "static"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
async def test_instance_norm(
    input_shape: tuple[int, ...],
    spatial_dims: int,
    affine: bool,
    track_running_stats: bool,
    dynamic_type: tuple[int],
    dtype: torch.dtype,
) -> None:
    class InstanceNormModule(nn.Module):
        instance_norm: (
            torch.nn.InstanceNorm1d | torch.nn.InstanceNorm2d | torch.nn.InstanceNorm3d
        )

        def __init__(
            self,
            num_features: int,
            nd: int,
            affine: bool = False,
            track_running_stats: bool = False,
            eps: float = 1e-5,
        ) -> None:
            super().__init__()
            if nd == 1:
                self.instance_norm = torch.nn.InstanceNorm1d(
                    num_features=num_features,
                    affine=affine,
                    track_running_stats=track_running_stats,
                    eps=eps,
                )
            elif nd == 2:
                self.instance_norm = torch.nn.InstanceNorm2d(
                    num_features=num_features,
                    affine=affine,
                    track_running_stats=track_running_stats,
                    eps=eps,
                )
            elif nd == 3:
                self.instance_norm = torch.nn.InstanceNorm3d(
                    num_features=num_features,
                    affine=affine,
                    track_running_stats=track_running_stats,
                    eps=eps,
                )
            else:
                raise ValueError(f"Invalid `nd`, must be in `[1, 2, 3], got {nd}")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.instance_norm(x)

    model = (
        InstanceNormModule(
            input_shape[-spatial_dims - 1], spatial_dims, affine, track_running_stats
        )
        .eval()
        .to(dtype)
    )
    x = torch.randn(*input_shape, dtype=dtype)
    dynamic_shapes = [torch.export.Dim.STATIC] * len(input_shape)
    if dynamic_type == "dynamic_spatial":
        for i in range(spatial_dims):
            dynamic_shapes[-i - 1] = torch.export.Dim.DYNAMIC
    await validate_numerical_output(
        model=model,
        x=x,
        dynamic_shapes=[dynamic_shapes],
        remove_decomps=[torch.ops.aten.instance_norm.default],
    )


class TestTrunc:
    @pytest.mark.parametrize(
        "x",
        [
            # Float tensors - mixed positive/negative values
            torch.tensor([[3.5, -7.2], [-2.8, 9.1]]),
            # Larger tensors
            torch.rand(3, 4) * 10 - 5,
        ],
    )
    @pytest.mark.parametrize(
        "dynamic_dims",
        [
            tuple(),
            (0,),
            (1,),
            (0, 1),
        ],
    )
    async def test_trunc(self, x: Tensor, dynamic_dims: tuple[int]) -> None:
        class TruncModule(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.trunc(x)

        model = TruncModule()
        dynamic_shapes = make_dynamic_shapes(x={d: f"d{d}" for d in dynamic_dims})
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)

    @pytest.mark.parametrize(
        "x_shape",
        [
            (3, 4),
            (5, 8),
            (2, 3, 6),
        ],
    )
    async def test_sym_trunc(self, x_shape: tuple[int, ...]) -> None:
        """Test aten.trunc on a SymFloat scalar (bare 'trunc' overload, not 'trunc.default').

        math.trunc on a SymFloat derived from a dynamic dim emits a bare aten.trunc
        node in the exported graph; using the result as a scale forces the cast
        to be materialised in the output.
        """

        import math

        class SymTruncModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x * math.trunc(torch.sym_float(x.shape[-1]) * 0.5 + 0.25)

        model = SymTruncModel().eval()
        x = torch.rand(x_shape)
        dynamic_shapes = {"x": _all_dims_dynamic(x)}
        await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


class TestSDPA:
    @pytest.mark.parametrize("batch_size", [2])
    @pytest.mark.parametrize("hidden_size", [128])
    @pytest.mark.parametrize("kv_heads", [2, 4])
    @pytest.mark.parametrize("q_heads", [8, 12])
    @pytest.mark.parametrize("max_ctx_len", [128])
    @pytest.mark.parametrize("q_len", [8, 16])
    @pytest.mark.parametrize("scale", [0.1, None])
    @pytest.mark.parametrize("mask_cfg", ["gen_causal", "pass_mask", "maskless"])
    @pytest.mark.parametrize("batch_leading_dims", [tuple(), (1, 1)])
    @pytest.mark.parametrize(
        "dynamic_dims",
        [
            "static",
            "batch",
            "q_len",
            "max_ctx_len",
            "batch+q_len",
            "batch+max_ctx_len",
            "q_len+max_ctx_len",
            "batch+q_len+max_ctx_len",
        ],
    )
    async def test_sdpa(
        self,
        batch_size,
        hidden_size,
        kv_heads,
        q_heads,
        max_ctx_len,
        q_len,
        scale,
        mask_cfg,
        batch_leading_dims,
        dynamic_dims,
    ) -> None:
        is_causal = mask_cfg == "gen_causal"

        class SDPAModule(nn.Module):
            def forward(self, query, key, value, attn_mask=None):
                if attn_mask is not None:
                    return nn.functional.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask,
                        is_causal=is_causal,
                        scale=scale,
                        enable_gqa=True,
                    )
                else:
                    return nn.functional.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        is_causal=is_causal,
                        scale=scale,
                        enable_gqa=True,
                    )

        q = torch.randn(*batch_leading_dims, batch_size, q_heads, q_len, hidden_size)
        k = torch.randn(
            *batch_leading_dims, batch_size, kv_heads, max_ctx_len, hidden_size
        )
        v = torch.randn(
            *batch_leading_dims, batch_size, kv_heads, max_ctx_len, hidden_size
        )
        attn_mask = (
            None
            if is_causal or mask_cfg == "maskless"
            else (
                torch.ones(*batch_leading_dims, 1, q_len, max_ctx_len) * float("-inf")
            ).tril()
        )

        model = SDPAModule().eval()

        # Build dynamic_shapes from the "+"-separated dim names.
        # Dim indices are offset by the number of prepended leading batch dims:
        #   batch       lives at dim offset   in query/key/value (not present in attn_mask)
        #   q_len       lives at dim offset+2 in query, and dim offset+1 in attn_mask
        #   max_ctx_len lives at dim offset+2 in key/value, and dim offset+2 in attn_mask
        dynamic_shapes = None
        if dynamic_dims != "static":
            active = set(dynamic_dims.split("+"))
            offset = len(batch_leading_dims)

            query_spec: dict[int, str | None] = {}
            key_spec: dict[int, str | None] = {}
            value_spec: dict[int, str | None] = {}
            mask_spec: dict[int, str | None] = {}

            if "batch" in active:
                query_spec[offset] = torch.export.Dim.DYNAMIC
                key_spec[offset] = torch.export.Dim.DYNAMIC
                value_spec[offset] = torch.export.Dim.DYNAMIC
                # attn_mask has shape (*batch_leading_dims, 1, q_len, max_ctx_len):
                # batch_size is not a separate dimension there, so no entry needed.
            if "q_len" in active:
                query_spec[offset + 2] = torch.export.Dim.DYNAMIC
                mask_spec[offset + 1] = torch.export.Dim.DYNAMIC
            if "max_ctx_len" in active:
                key_spec[offset + 2] = torch.export.Dim.DYNAMIC
                value_spec[offset + 2] = torch.export.Dim.DYNAMIC
                mask_spec[offset + 2] = torch.export.Dim.DYNAMIC

            if attn_mask is not None:
                dynamic_shapes = {
                    "query": query_spec,
                    "key": key_spec,
                    "value": value_spec,
                    "attn_mask": mask_spec,
                }
            else:
                dynamic_shapes = {
                    "query": query_spec,
                    "key": key_spec,
                    "value": value_spec,
                }

        if attn_mask is not None:
            await validate_numerical_output(
                model=model,
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                dynamic_shapes=dynamic_shapes,
                remove_decomps=[torch.ops.aten.scaled_dot_product_attention.default],
            )
        else:
            await validate_numerical_output(
                model=model,
                query=q,
                key=k,
                value=v,
                dynamic_shapes=dynamic_shapes,
                remove_decomps=[torch.ops.aten.scaled_dot_product_attention.default],
            )

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize(
        ("q_len", "seq_len"),
        [(5, 7), (8, 18)],
    )
    @pytest.mark.parametrize("embed_dim", [16, 32])
    @pytest.mark.parametrize("with_mask", [False, True])
    @pytest.mark.parametrize("dynamic", [False, True])
    async def test_sdpa_rank3(
        self,
        batch_size: int,
        q_len: int,
        seq_len: int,
        embed_dim: int,
        with_mask: bool,
        dynamic: bool,
    ) -> None:
        """Rank-3 (B, S, E) SDPA — PyTorch allows omitting the head dim."""

        class SDPAModule(nn.Module):
            def forward(
                self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Tensor | None = None,
            ) -> Tensor:
                if attn_mask is not None:
                    return nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask
                    )
                return nn.functional.scaled_dot_product_attention(query, key, value)

        query = torch.rand(batch_size, q_len, embed_dim)
        key = torch.rand(batch_size, seq_len, embed_dim)
        value = torch.rand(batch_size, seq_len, embed_dim)
        # diagonal = seq_len - q_len with seq_len >= q_len keeps at least one True
        # per row; fully-masked rows diverge because the decomp uses -1e4 rather
        # than -inf.
        attn_mask = (
            torch.tril(
                torch.ones((q_len, seq_len), dtype=torch.bool),
                diagonal=seq_len - q_len,
            )
            if with_mask
            else None
        )

        model = SDPAModule().eval()

        dynamic_shapes: dict[str, dict[int, Any]] | None = None
        if dynamic:
            q_dim = torch.export.Dim("q_len", min=1, max=64)
            s_dim = torch.export.Dim("seq_len", min=1, max=64)
            dynamic_shapes = {
                "query": {1: q_dim},
                "key": {1: s_dim},
                "value": {1: s_dim},
            }
            if with_mask:
                dynamic_shapes["attn_mask"] = {0: q_dim, 1: s_dim}

        kwargs: dict[str, Any] = {"query": query, "key": key, "value": value}
        if with_mask:
            kwargs["attn_mask"] = attn_mask

        await validate_numerical_output(
            model=model,
            dynamic_shapes=dynamic_shapes,
            remove_decomps=[torch.ops.aten.scaled_dot_product_attention.default],
            **kwargs,
        )
