# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowering handlers for coreai compression custom ops → Core AI ops.

Each handler mirrors the corresponding legalization in TorchImportToCore.cpp:
  - coreai::lut_to_dense                           → coreai.lut_to_dense
  - coreai::constexpr_blockwise_shift_scale        → coreai.blockwise_shift_scale
  - coreai::quantize                               → coreai.quantize
  - coreai::dequantize                             → coreai.dequantize
  - coreai::sparse_to_dense                        → coreai.build_sparse_with_bitmask + coreai.sparse_with_bitmask_to_dense

The zero_point / minval → offset1 / offset2 conversion follows the C++ lowering:
  - zero_point mode (arg is not None): offset1 = zero_point, offset2 = zeros(float_dtype)
  - minval mode     (arg is not None): offset1 = q_min(quant_dtype), offset2 = minval
  - no-offset mode:                    offset1 = zeros(quant_dtype), offset2 = zeros(float_dtype)
"""

import math
from typing import Callable

import numpy as np
import torch.fx as fx
from coreai._compiler.dialects import coreai
from coreai._compiler.ir import (
    DenseElementsAttr,
    DenseResourceElementsAttr,
    Float4E2M1FNType,
    Float8E4M3FNType,
    Float8E5M2Type,
    Float8E8M0FNUType,
    FloatAttr,
    IntegerType,
    Location,
    RankedTensorType,
    Value,
)
from coreai._compiler.type_mapping import _MLIR_TO_NUMPY_DTYPE as MLIR_TO_NUMPY_DTYPE

from ._type_mapping import TORCH_TO_COREAI_DTYPE
from ._utils import get_operand as _get_operand
from ._utils import get_operands as _get_operands

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zeros_constant(shape: list[int], elem_type: object) -> Value:
    """Create a splat zero constant tensor with the given shape and element type."""
    if isinstance(
        elem_type,
        (Float4E2M1FNType, Float8E4M3FNType, Float8E5M2Type, Float8E8M0FNUType),
    ):
        # Core AI compiler can only check if a DenseElementsAttr is splat,
        # i.e. cannot check the opaque DenseResourceElementsAttr.
        # As of 2026-04-10 coreai.constant uses DenseResourceElementsAttr
        # for reduced-precision float dtypes (fp4/fp8), so we have to
        # explicitly call DenseElementsAttr.get_splat + ConstantOp for them.
        # TODO: Once coreai.constant correctly handles fp4/fp8, migrate to it.
        tensor_type = RankedTensorType.get(shape, elem_type)
        zero_attr = DenseElementsAttr.get_splat(
            tensor_type,
            FloatAttr.get(elem_type, 0.0),
        )
        return coreai.ConstantOp(value=zero_attr).result
    else:
        numpy_dtype = MLIR_TO_NUMPY_DTYPE[str(elem_type)]
        return coreai.constant(
            np.zeros(shape, dtype=numpy_dtype),
            dtype=elem_type,  # type: ignore[arg-type]
        )


def _qmin_constant(shape: list[int], elem_type: object) -> Value:
    """Create a constant tensor filled with q_min for the given integer element type.

    Computes q_min from the IntegerType bit width directly, because
    MLIR_TO_NUMPY_DTYPE maps all sub-byte types (si4, ui4, …) to int8/uint8
    containers whose np.iinfo().min would be wrong (e.g. -128 instead of -8).

    For sub-byte types, builds pre-packed bytes matching the C++ lowering in
    TorchImportToCore::createShapedQMinTensorConstant — this avoids relying on
    create_elements_attr to pack non-zero sub-byte values correctly.
    """
    int_type = IntegerType(elem_type)
    bit_width = int_type.width
    qmin = -(1 << (bit_width - 1)) if int_type.is_signed else 0

    if bit_width < 8:  # noqa: PLR2004
        # Build the byte pattern: replicate the bitWidth-wide q_min across
        # each byte, matching the C++ lowering.
        element = qmin & ((1 << bit_width) - 1)  # mask to bitWidth
        splat_byte = 0
        for shift in range(0, 8, bit_width):
            splat_byte |= element << shift

        num_elements = math.prod(shape)
        num_bits = bit_width * num_elements
        num_bytes = (num_bits + 7) // 8
        packed = np.full(num_bytes, splat_byte, dtype=np.uint8)

        tensor_type = RankedTensorType.get(shape, elem_type)
        value_attr = DenseResourceElementsAttr.get_from_buffer(
            packed,  # type: ignore[arg-type]
            "qmin_constant",
            tensor_type,
        )
        return coreai.ConstantOp(value=value_attr).result

    numpy_dtype = MLIR_TO_NUMPY_DTYPE[str(elem_type)]
    return coreai.constant(
        np.full(shape, qmin, dtype=numpy_dtype),
        dtype=elem_type,  # type: ignore[arg-type]
    )


def _extract_int_arg(node: fx.Node, idx: int) -> int:
    """Extract a constant integer from node.args[idx] (literal or FX node with val)."""
    arg = node.args[idx]
    if isinstance(arg, fx.Node):
        return int(arg.meta["val"])
    return int(arg)  # type: ignore[arg-type]


def _get_optional_tensor_arg(node: fx.Node, idx: int) -> fx.Node | None:
    """Return the FX Node at args[idx] if it exists and is a tensor Node, else None."""
    if len(node.args) <= idx:
        return None
    arg = node.args[idx]
    if arg is None or not isinstance(arg, fx.Node):
        return None
    return arg


def _get_optional_int_arg(node: fx.Node, idx: int, default: int = 0) -> int:
    """Return int at args[idx] if present and is an int, else default."""
    if len(node.args) <= idx:
        return default
    arg = node.args[idx]
    if isinstance(arg, fx.Node):
        return int(arg.meta["val"])
    if isinstance(arg, int):
        return arg
    return default


def _build_offsets(
    values_map: dict[str, Value],
    node: fx.Node,
    zp_idx: int,
    mv_idx: int,
    quant_elem_type: object,
    float_elem_type: object,
    offset_shape: list[int],
    loc: Location,
) -> tuple[Value, Value]:
    """Build (offset1, offset2) from the zero_point / minval arg pattern.

    - zero_point mode: offset1 = zero_point tensor, offset2 = zeros(float_dtype)
    - minval mode:     offset1 = q_min(quant_dtype),  offset2 = minval tensor
    - no-offset mode:  offset1 = zeros(quant_dtype),  offset2 = zeros(float_dtype)
    """
    has_zp = _get_optional_tensor_arg(node, zp_idx) is not None
    has_mv = _get_optional_tensor_arg(node, mv_idx) is not None

    if has_zp:
        offset1 = _get_operand(values_map, node, zp_idx, loc)
        offset2 = _zeros_constant(offset_shape, float_elem_type)
    elif has_mv:
        offset1 = _qmin_constant(offset_shape, quant_elem_type)
        offset2 = _get_operand(values_map, node, mv_idx, loc)
    else:
        offset1 = _zeros_constant(offset_shape, quant_elem_type)
        offset2 = _zeros_constant(offset_shape, float_elem_type)

    return offset1, offset2


# ---------------------------------------------------------------------------
# lut_to_dense
# ---------------------------------------------------------------------------


def replace_lut_to_dense(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Lower coreai::lut_to_dense → coreai.lut_to_dense.

    Normalizes axis to a scalar si16 tensor, as required by LutToDenseOp.

    Note: indices are expected to already carry the correct sub-byte element type
    (ui1/ui2/ui4/ui6) because ``inject_subbyte_tensors`` narrows them from uint8
    before the import.  If that step was skipped, the MLIR verifier will report a
    NUM_PALETTES / bitwidth mismatch.
    """
    indices, lut = _get_operands(values_map, node, [0, 1], loc)
    axis_val = _extract_int_arg(node, 2)
    axis = coreai.constant(np.array(axis_val, dtype=np.int16), loc=loc)
    return coreai.lut_to_dense(indices, lut, axis, loc=loc)


# ---------------------------------------------------------------------------
# constexpr_blockwise_shift_scale
# ---------------------------------------------------------------------------


def replace_constexpr_blockwise_shift_scale(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Lower coreai::constexpr_blockwise_shift_scale → coreai.blockwise_shift_scale.

    After torch.export the arg layout is:
        (input, scale, zero_point?, minval?, input_dtype?, output_dtype?)
         [0]    [1]      [2]         [3]       [4]          [5]

    input_dtype / output_dtype are torch.dtype metadata (not tensors); they are
    reflected in the input / result element types and skipped here.
    """
    input_val = _get_operand(values_map, node, 0, loc)
    scale = _get_operand(values_map, node, 1, loc)

    input_elem_type = input_val.type.element_type
    input_rank = len(input_val.type.shape)
    result_elem_type = TORCH_TO_COREAI_DTYPE[node.meta["val"].dtype]()

    # quantElemType = input (blockwise is always dequantization: int→float)
    quant_elem_type = input_elem_type
    float_elem_type = result_elem_type

    scale_shape = list(scale.type.shape)

    offset1, offset2 = _build_offsets(
        values_map,
        node,
        zp_idx=2,
        mv_idx=3,
        quant_elem_type=quant_elem_type,
        float_elem_type=float_elem_type,
        offset_shape=scale_shape,
        loc=loc,
    )

    # Reshape scalar scale/offset1 to [1]*input_rank so BlockwiseShiftScaleOp
    # receives operands of the same rank as input.
    ones_shape = [1] * input_rank
    if len(scale.type.shape) == 0:
        scale = coreai.reshape(scale, ones_shape)
    if len(offset1.type.shape) == 0:
        offset1 = coreai.reshape(offset1, ones_shape)
    if len(offset2.type.shape) == 0:
        offset2 = coreai.reshape(offset2, ones_shape)

    return coreai.blockwise_shift_scale(input_val, scale, offset1, offset2, loc=loc)


# ---------------------------------------------------------------------------
# quantize / dequantize
# ---------------------------------------------------------------------------


def _replace_quantize_or_dequantize(
    values_map: dict[str, Value],
    node: fx.Node,
    loc: Location,
    *,
    is_quantize: bool,
) -> Value:
    """Shared lowering for quantize and dequantize ops.

    After torch.export the arg layouts are:
        quantize:   (input, scale, output_dtype, zero_point?, minval?, axis?)
                      [0]   [1]       [2]           [3]        [4]    [5]
        dequantize: (input, scale, zero_point?, minval?, axis?, input_dtype?, output_dtype?)
                      [0]   [1]      [2]         [3]    [4]       [5]           [6]

    output_dtype / input_dtype are torch.dtype metadata and skipped.
    """
    input_val = _get_operand(values_map, node, 0, loc)
    scale = _get_operand(values_map, node, 1, loc)

    input_type = input_val.type
    scale_type = scale.type
    result_elem_type = TORCH_TO_COREAI_DTYPE[node.meta["val"].dtype]()

    # Arg index mapping differs: quantize has output_dtype at [2] shifting the rest.
    if is_quantize:
        zp_idx, mv_idx, axis_idx = 3, 4, 5
        quant_elem_type = result_elem_type  # quantize: float→quant
        float_elem_type = input_type.element_type
    else:
        zp_idx, mv_idx, axis_idx = 2, 3, 4
        quant_elem_type = input_type.element_type  # dequantize: quant→float
        float_elem_type = result_elem_type

    # Extract axis; normalize a negative axis the same way the eager op does.
    axis_val = _get_optional_int_arg(node, axis_idx, default=0)
    input_rank = len(input_type.shape)
    if axis_val < 0:
        axis_val = axis_val + input_rank

    axis = coreai.constant(np.array(axis_val, dtype=np.int32), loc=loc)

    # Determine target shape for scale/offsets: per-tensor→[], per-channel→[dim].
    n_scale_elements = math.prod(scale_type.shape) if scale_type.shape else 1

    # Check zero_point cardinality too (it may be per-channel even if scale is scalar).
    zp_node = _get_optional_tensor_arg(node, zp_idx)
    if zp_node is not None:
        zp_shape = zp_node.meta["val"].shape
        n_zp_elements = math.prod(zp_shape) if zp_shape else 1
    else:
        n_zp_elements = 0

    if n_scale_elements > 1 or n_zp_elements > 1:
        target_shape: list[int] = [input_type.shape[axis_val]]
    else:
        target_shape = []

    # Reshape scale.
    scale = coreai.reshape(scale, target_shape)

    # Build offset1/offset2 from zero_point / minval pattern.
    offset1, offset2 = _build_offsets(
        values_map,
        node,
        zp_idx=zp_idx,
        mv_idx=mv_idx,
        quant_elem_type=quant_elem_type,
        float_elem_type=float_elem_type,
        offset_shape=target_shape,
        loc=loc,
    )

    # Reshape offsets to match target_shape (they may come from the graph with
    # broadcast-compatible shapes like Nx1x1x1).
    offset1 = coreai.reshape(offset1, target_shape)
    offset2 = coreai.reshape(offset2, target_shape)

    if is_quantize:
        return coreai.quantize(input_val, scale, offset1, offset2, axis, loc=loc)
    return coreai.dequantize(input_val, scale, offset1, offset2, axis, loc=loc)


def replace_quantize(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Lower coreai::quantize → coreai.quantize."""
    return _replace_quantize_or_dequantize(values_map, node, loc, is_quantize=True)


def replace_dequantize(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Lower coreai::dequantize → coreai.dequantize."""
    return _replace_quantize_or_dequantize(values_map, node, loc, is_quantize=False)


# ---------------------------------------------------------------------------
# sparse_to_dense
# ---------------------------------------------------------------------------


def replace_sparse_to_dense(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Lower coreai::sparse_to_dense → coreai.build_sparse_with_bitmask + coreai.sparse_with_bitmask_to_dense."""
    nonzero_data, mask = _get_operands(values_map, node, [0, 1], loc)
    sparse = coreai.build_sparse_with_bitmask(nonzero_data, mask, loc=loc)
    return coreai.sparse_with_bitmask_to_dense(sparse, loc=loc)


# ---------------------------------------------------------------------------
# Resolver: op target name → handler.
# ---------------------------------------------------------------------------

_custom_to_core_resolver: dict[
    str, Callable[[dict[str, Value], fx.Node, Location], Value]
] = {
    "lut_to_dense": replace_lut_to_dense,
    "constexpr_blockwise_shift_scale": replace_constexpr_blockwise_shift_scale,
    "quantize": replace_quantize,
    "dequantize": replace_dequantize,
    "sparse_to_dense": replace_sparse_to_dense,
}
