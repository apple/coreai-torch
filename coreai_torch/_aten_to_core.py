# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Any, Callable

import numpy as np
import torch
import torch.fx as fx
from coreai._compiler.dialects import coreai
from coreai._compiler.ir import (
    Attribute,
    F32Type,
    F64Type,
    IntegerType,
    Location,
    OpResultList,
    RankedTensorType,
    ShapedType,
    Value,
)

from ._composite_declaration import generate_composite_decl
from ._type_mapping import _get_coreai_to_numpy_dtype
from ._utils import (
    _sdpa_build_causal_mask,
    _sdpa_decompose,
    _sdpa_flatten_leading_batch_dims,
    build_hard_sigmoid_composite,
    build_shape_tensor,
    build_slice_index_array,
    convert_branch_subgraph,
    expand_boolean_indices,
    get_output_element_type_from_node,
    get_promoted_type,
    get_target,
    get_tensor_shape_at_index,
    get_tensor_type,
    prepare_compute_type_for_norm,
    process_expanded_indices,
    process_indices_with_transpose,
    resolve_slice_arg,
    to_uint32_perm,
    upsample_align_corners_scale_offset,
    upsample_build_output_shape_dynamic,
    upsample_halfpixel_scale,
    upsample_runtime_output_hw_from_scale_dynamic,
)
from ._utils import (
    get_operand as _get_operand,
)
from ._utils import (
    get_operands as _get_operands,
)
from ._utils import (
    replace_pad_with_mode as _replace_pad_with_mode,
)

INT32_MAX: int = 2147483647


def replace_abs(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    return coreai.abs_(_get_operand(values_map, node, 0))


def replace_addmm(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x, y, z = _get_operands(values_map, node, [0, 1, 2])

    alpha = node.kwargs.get("alpha", 1.0)
    beta = node.kwargs.get("beta", 1.0)
    # Use lhs (y) element type as the canonical type, matching replaceAddMM in
    # TorchImportToCore.cpp which casts rhs and bias to lhs's element type.
    ele_type = y.type.element_type

    if z.type.element_type != ele_type:
        z = coreai.cast(z, ele_type)

    mm = coreai.broadcasting_batch_matmul(y, z)

    if alpha != 1.0:
        mm = coreai.broadcasting_mul(mm, coreai.cast(alpha, ele_type))

    if x.type.element_type != ele_type:
        x = coreai.cast(x, ele_type)

    if beta != 1.0:
        x = coreai.broadcasting_mul(x, coreai.cast(beta, ele_type))

    return coreai.broadcasting_add(mm, x)


def replace_alias(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)
    return x


def replace_amax_default(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    dims = node.args[1]
    keepdim = len(node.args) > 2 and bool(node.args[2])
    result = coreai.reduce_max(x, dims)
    return result if keepdim else coreai.shrink_dims(result, dims)


def replace_amin_default(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    dims = node.args[1]
    keepdim = len(node.args) > 2 and bool(node.args[2])
    result = coreai.reduce_min(x, dims)
    return result if keepdim else coreai.shrink_dims(result, dims)


def replace_avg_pool2d(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.avg_pool2d to a composite op using coreai.graph.

    aten.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    Decomposition (as composite op):
        1. Pad the input with zeros according to padding parameter
        2. Apply sum_pool_2d on padded input
        3. Divide by divisor (kernel_h * kernel_w or divisor_override)
    """
    x = _get_operand(values_map, node, 0)
    kernel_size = node.args[1]
    stride = (
        node.args[2] if len(node.args) > 2 and node.args[2] is not None else kernel_size
    )
    padding = (
        node.args[3] if len(node.args) > 3 and node.args[3] is not None else [0, 0]
    )
    ceil_mode = (
        node.args[4] if len(node.args) > 4 and node.args[4] is not None else False
    )
    count_include_pad = (
        node.args[5] if len(node.args) > 4 and node.args[4] is not None else True
    )
    divisor_override_val = (
        int(node.args[6]) if len(node.args) > 6 and node.args[6] is not None else 0
    )
    divisor_f = (
        float(node.args[6])
        if len(node.args) > 6 and node.args[6] is not None
        else float(kernel_size[0] * kernel_size[1])
    )
    x_element_type = x.type.element_type

    input_names = [
        "input",
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]
    output_names = ["output"]
    op_attributes: dict[str, Any] = {"version": 1}
    composite_decl = generate_composite_decl(
        x.context, "avg_pool_2d", input_names, output_names, op_attributes
    )

    dyn = RankedTensorType.get_dynamic_size()
    n = x.type.shape[0] if x.type.shape[0] > 0 else dyn
    c = x.type.shape[1] if x.type.shape[1] > 0 else dyn

    # Spatial dims are dynamic inside the composite because padding is a
    # composite argument, not a compile-time constant.
    padded_shape = [n, c, dyn, dyn]

    # Pooled output shape is known from outer-scope values.
    actual_padded_h = x.type.shape[2] + 2 * padding[0] if x.type.shape[2] > 0 else -1
    actual_padded_w = x.type.shape[3] + 2 * padding[1] if x.type.shape[3] > 0 else -1
    pooled_shape = [
        n,
        c,
        (actual_padded_h - kernel_size[0]) // stride[0] + 1
        if actual_padded_h > 0
        else dyn,
        (actual_padded_w - kernel_size[1]) // stride[1] + 1
        if actual_padded_w > 0
        else dyn,
    ]

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def avg_pool2d_composite(
        input_tensor: Value,
        kernel_size: Value,
        stride: Value,
        padding: Value,
        ceil_mode: Value,
        count_include_pad: Value,
        divisor_override: Value,
    ) -> Value:
        # The old converter generates the body of the composite as if count_include_pad = True
        # and ceil_mode = False at all times. This is an inaccuracy. For now we are targeting
        # parity with the old converter and so the composite body will remain the same.
        padding_ui32 = coreai.cast(padding, np.uint32)
        pad_h = coreai.slice_(padding_ui32, [0], [1], [1])
        pad_w = coreai.slice_(padding_ui32, [1], [2], [1])
        four_zeros = coreai.constant([0, 0, 0, 0], dtype=np.uint32)

        padding_4d = coreai.concat(0, [four_zeros, pad_h, pad_h, pad_w, pad_w])

        pad_value = coreai.constant(0.0, dtype=x_element_type)
        padded_input = coreai.PadOp(
            input_tensor,
            padding_4d,
            pad_value,
            Attribute.parse("#coreai.padding_mode<constant>"),
            results=[RankedTensorType.get(padded_shape, x_element_type)],
        ).result

        kernel_size_ui32 = coreai.cast(kernel_size, np.uint32)
        stride_ui32 = coreai.cast(stride, np.uint32)

        divisor_const = coreai.constant(divisor_f, dtype=x_element_type)
        return coreai.broadcasting_divide(
            coreai.SumPool2dOp(
                padded_input,
                kernel_size=kernel_size_ui32,
                strides=stride_ui32,
                # TODO: We should still use numpy array, need a fix in coreai
                dilation=coreai.constant([1, 1], dtype=np.uint32),
                results=[RankedTensorType.get(pooled_shape, x_element_type)],
            ).result,
            divisor_const,
        )

    kernel_size = coreai.constant(kernel_size, dtype=np.int32)
    stride = coreai.constant(stride, dtype=np.int32)
    padding = coreai.constant(padding, dtype=np.int32)
    ceil_mode = coreai.constant(ceil_mode)
    count_include_pad = coreai.constant(count_include_pad)
    divisor_override = coreai.constant(divisor_override_val, dtype=np.int32)

    return avg_pool2d_composite(
        x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    )


def replace_adaptive_avg_pool2d_dynamic(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Lowering for adaptive_avg_pool2d when input H/W dimensions are dynamic.

    Uses slice-based pooling with dynamically-computed window boundaries.
    The outer loops are unrolled over the static output_h/output_w dimensions,
    while the window start/end indices are computed at runtime from the input shape.

    For each output position (oh, ow):
        hstart = (oh * H) // output_h          -- floor division
        hend   = ((oh + 1) * H + output_h - 1) // output_h   -- ceiling division
        wstart = (ow * W) // output_w
        wend   = ((ow + 1) * W + output_w - 1) // output_w
        out[:, :, oh, ow] = mean(x[:, :, hstart:hend, wstart:wend])

    Non-divisible case:
        When H is not evenly divisible by output_h, the floor/ceiling pair
        produces windows of unequal size (differing by one input element).
        Adjacent windows may share a boundary element; this matches PyTorch's
        adaptive pooling semantics exactly.

        Example — H=7, output_h=3:
            oh=0: hstart=0,  hend=3  → input rows [0, 1, 2]      (3 elements)
            oh=1: hstart=2,  hend=5  → input rows [2, 3, 4]      (3 elements, row 2 shared)
            oh=2: hstart=4,  hend=7  → input rows [4, 5, 6]      (3 elements, row 4 shared)

        Example — H=7, output_h=4:
            oh=0: hstart=0,  hend=2  → input rows [0, 1]         (2 elements)
            oh=1: hstart=1,  hend=4  → input rows [1, 2, 3]      (3 elements, row 1 shared)
            oh=2: hstart=3,  hend=6  → input rows [3, 4, 5]      (3 elements, row 3 shared)
            oh=3: hstart=5,  hend=7  → input rows [5, 6]         (2 elements, row 5 shared)

        The ceiling division for hend ensures every input row is covered by at
        least one output window, so no input information is lost.
    """
    x = _get_operand(values_map, node, 0)
    output_size = node.args[1]
    output_h, output_w = output_size[0], output_size[1]

    # Extract all shape dimensions at runtime since N and/or C may be dynamic.
    input_shape = coreai.cast(coreai.get_shape(x), dtype=np.int32)
    input_n = coreai.slice_(input_shape, [0], [1], [1])
    input_c = coreai.slice_(input_shape, [1], [2], [1])
    input_h = coreai.slice_(input_shape, [2], [3], [1])
    input_w = coreai.slice_(input_shape, [3], [4], [1])

    # 2-element begin/end prefixes reused every iteration to build the
    # 4-element begin/end vectors for coreai.slice_.
    nc_begin = [0, 0]
    nc_end = coreai.concat(0, [input_n, input_c])

    # Build target shape dynamically: [N, C, output_h, output_w]
    output_shape = coreai.concat(0, [input_n, input_c, [output_h], [output_w]])
    result = coreai.broadcast_to(coreai.cast(0.0, x.type.element_type), output_shape)

    for oh in range(output_h):
        hstart = coreai.broadcasting_floor_divide(
            coreai.broadcasting_mul(oh, input_h), output_h
        )
        hend = coreai.broadcasting_floor_divide(
            coreai.broadcasting_add(
                coreai.broadcasting_mul(oh + 1, input_h), output_h - 1
            ),
            output_h,
        )
        for ow in range(output_w):
            wstart = coreai.broadcasting_floor_divide(
                coreai.broadcasting_mul(ow, input_w), output_w
            )
            wend = coreai.broadcasting_floor_divide(
                coreai.broadcasting_add(
                    coreai.broadcasting_mul(ow + 1, input_w), output_w - 1
                ),
                output_w,
            )
            sliced = coreai.slice_(
                x,
                coreai.concat(0, [nc_begin, hstart, wstart]),
                coreai.concat(0, [nc_end, hend, wend]),
                [1, 1, 1, 1],
            )
            slice_update_end = coreai.concat(0, [input_n, input_c, [oh + 1], [ow + 1]])
            result = coreai.slice_update(
                result,
                [0, 0, oh, ow],
                slice_update_end,
                [1, 1, 1, 1],
                coreai.reduce_mean(sliced, [2, 3]),
            )

    return result


def replace_adaptive_avg_pool2d(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten._adaptive_avg_pool2d to Core AI operations.

    aten._adaptive_avg_pool2d(input, output_size) -> Tensor

    Input is a 4D tensor of shape (N, C, H, W).
    Output_size is a 2-element list [output_H, output_W].

    Supports two paths:
    1. Optimized path (divisible): When output_size evenly divides input_size,
       uses sum_pool_2d with fixed stride/kernel for better performance.
    2. General path (non-divisible): Uses slice_update-based approach similar to
       Core AI's C++ implementation. Iterates over output positions, computing adaptive
       boundaries and updating the result tensor in-place.

    Algorithm for non-divisible cases (following Core AI's C++ pattern):
        result = zeros([N, C, output_h, output_w])
        for oh in range(output_h):  # Outer loop (height)
            for ow in range(output_w):  # Inner loop (width)
                start_h = (oh * input_h) // output_h
                end_h = ((oh + 1) * input_h + output_h - 1) // output_h
                start_w = (ow * input_w) // output_w
                end_w = ((ow + 1) * input_w + output_w - 1) // output_w
                sliced = input[:, :, start_h:end_h, start_w:end_w]
                mean_val = reduce_mean(sliced, axes=[2, 3], keep_dims=True)
                result[:, :, oh:oh+1, ow:ow+1] = mean_val  # Using slice_update

    Note: Structured to facilitate future conversion to coreai.while_op for dynamic shapes.
    """
    x = _get_operand(values_map, node, 0)
    output_size = node.args[1]

    if x.type.rank != 4:
        raise ValueError(
            f"adaptive_avg_pool2d requires 4D input (NCHW), got rank {x.type.rank}"
        )
    if len(output_size) != 2:
        raise ValueError(f"output_size must contain 2 values, got {len(output_size)}")

    if not x.type.has_static_shape:
        return replace_adaptive_avg_pool2d_dynamic(values_map, node, loc)

    x_shape = x.type.shape
    x_element_type = x.type.element_type
    input_h, input_w = x_shape[2], x_shape[3]
    output_h, output_w = output_size[0], output_size[1]

    if input_h < output_h or input_w < output_w:
        raise ValueError(
            f"Input size ({input_h}, {input_w}) is smaller than output size ({output_h}, {output_w}). "
            "No interpolation or padding is supported."
        )

    if (input_h % output_h == 0) and (input_w % output_w == 0):
        # Optimized path: fixed-stride sum pooling for evenly divisible dimensions.
        stride_h = input_h // output_h
        stride_w = input_w // output_w
        kernel_h = input_h - (output_h - 1) * stride_h
        kernel_w = input_w - (output_w - 1) * stride_w

        def adaptive_avg_pool2d_divisible(input_tensor: Value) -> Value:
            return coreai.broadcasting_divide(
                coreai.sumpool2d(
                    input_tensor,
                    kernel_size=np.array([kernel_h, kernel_w], dtype=np.uint32),
                    strides=np.array([stride_h, stride_w], dtype=np.uint32),
                    # TODO: fix this in coreai
                    dilation=coreai.constant([1, 1], dtype=np.uint32),
                ),
                coreai.cast(float(kernel_h * kernel_w), x_element_type),
            )

        return adaptive_avg_pool2d_divisible(x)
    else:
        # General path: slice-update based adaptive pooling for non-divisible cases.

        def adaptive_avg_pool2d_general(input_tensor: Value) -> Value:
            result = coreai.broadcast_to(
                coreai.cast(0.0, x_element_type),
                [x_shape[0], x_shape[1], output_h, output_w],
            )
            for oh in range(output_h):
                for ow in range(output_w):
                    start_h = (oh * input_h) // output_h
                    end_h = ((oh + 1) * input_h + output_h - 1) // output_h
                    start_w = (ow * input_w) // output_w
                    end_w = ((ow + 1) * input_w + output_w - 1) // output_w
                    mean_val = coreai.reduce_mean(
                        coreai.slice_(
                            input_tensor,
                            start_indices=[0, 0, start_h, start_w],
                            end_indices=[INT32_MAX, INT32_MAX, end_h, end_w],
                            strides=[1, 1, 1, 1],
                        ),
                        [2, 3],
                    )
                    result = coreai.slice_update(
                        result,
                        start_indices=[0, 0, oh, ow],
                        end_indices=[INT32_MAX, INT32_MAX, oh + 1, ow + 1],
                        strides=[1, 1, 1, 1],
                        update=mean_val,
                    )
            return result

        return adaptive_avg_pool2d_general(x)


def replace_avg_pool3d(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.avg_pool3d to a composite op using coreai.graph.

    aten.avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    Input is a 5D tensor of shape (N, C, D, H, W) - batch, channels, depth, height, width.

    Decomposition (as composite op):
        1. Pad the input with zeros according to padding parameter
        2. Apply sum_pool_3d on padded input
        3. Divide by divisor (kernel_d * kernel_h * kernel_w or divisor_override)
    """
    x = _get_operand(values_map, node, 0)
    kernel_size = node.args[1]
    stride = (
        node.args[2] if len(node.args) > 2 and node.args[2] is not None else kernel_size
    )
    padding = (
        node.args[3] if len(node.args) > 3 and node.args[3] is not None else [0, 0, 0]
    )
    ceil_mode = (
        node.args[4] if len(node.args) > 4 and node.args[4] is not None else False
    )
    count_include_pad = (
        node.args[5] if len(node.args) > 4 and node.args[4] is not None else True
    )
    divisor_override_val = (
        int(node.args[6]) if len(node.args) > 6 and node.args[6] is not None else 0
    )
    divisor_f = (
        float(node.args[6])
        if len(node.args) > 6 and node.args[6] is not None
        else float(kernel_size[0] * kernel_size[1] * kernel_size[2])
    )

    x_element_type = x.type.element_type

    input_names = [
        "input",
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]
    output_names = ["output"]
    op_attributes: dict[str, Any] = {"version": 1}
    composite_decl = generate_composite_decl(
        x.context, "avg_pool_3d", input_names, output_names, op_attributes
    )

    dyn = RankedTensorType.get_dynamic_size()
    n = x.type.shape[0] if x.type.shape[0] > 0 else dyn
    c = x.type.shape[1] if x.type.shape[1] > 0 else dyn

    # Spatial dims are dynamic inside the composite because padding is a
    # composite argument, not a compile-time constant.
    padded_shape = [n, c, dyn, dyn, dyn]

    # Pooled output shape is known from outer-scope values.
    actual_padded_d = x.type.shape[2] + 2 * padding[0] if x.type.shape[2] > 0 else -1
    actual_padded_h = x.type.shape[3] + 2 * padding[1] if x.type.shape[3] > 0 else -1
    actual_padded_w = x.type.shape[4] + 2 * padding[2] if x.type.shape[4] > 0 else -1
    pooled_shape = [
        n,
        c,
        (actual_padded_d - kernel_size[0]) // stride[0] + 1
        if actual_padded_d > 0
        else dyn,
        (actual_padded_h - kernel_size[1]) // stride[1] + 1
        if actual_padded_h > 0
        else dyn,
        (actual_padded_w - kernel_size[2]) // stride[2] + 1
        if actual_padded_w > 0
        else dyn,
    ]

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def avg_pool3d_composite(
        input_tensor: Value,
        kernel_size: Value,
        stride: Value,
        padding: Value,
        ceil_mode: Value,
        count_include_pad: Value,
        divisor_override: Value,
    ) -> Value:
        # The old converter generates the body of the composite as if count_include_pad = True
        # and ceil_mode = False at all times. This is an inaccuracy. For now we are targeting
        # parity with the old converter and so the composite body will remain the same.
        padding_ui32 = coreai.cast(padding, np.uint32)
        pad_d = coreai.slice_(padding_ui32, [0], [1], [1])
        pad_h = coreai.slice_(padding_ui32, [1], [2], [1])
        pad_w = coreai.slice_(padding_ui32, [2], [3], [1])
        four_zeros = coreai.constant([0, 0, 0, 0], dtype=np.uint32)

        padding_5d = coreai.concat(
            0, [four_zeros, pad_d, pad_d, pad_h, pad_h, pad_w, pad_w]
        )

        pad_value = coreai.constant(0.0, dtype=x_element_type)
        padded_input = coreai.PadOp(
            input_tensor,
            padding_5d,
            pad_value,
            Attribute.parse("#coreai.padding_mode<constant>"),
            results=[RankedTensorType.get(padded_shape, x_element_type)],
        ).result

        kernel_size_ui32 = coreai.cast(kernel_size, np.uint32)
        stride_ui32 = coreai.cast(stride, np.uint32)

        divisor_const = coreai.constant(divisor_f, dtype=x_element_type)
        return coreai.broadcasting_divide(
            coreai.SumPool3dOp(
                padded_input,
                kernel_size=kernel_size_ui32,
                strides=stride_ui32,
                # TODO: We should still use numpy array, need a fix in coreai
                dilation=coreai.constant([1, 1, 1], dtype=np.uint32),
                results=[RankedTensorType.get(pooled_shape, x_element_type)],
            ).result,
            divisor_const,
        )

    kernel_size = coreai.constant(kernel_size, dtype=np.int32)
    stride = coreai.constant(stride, dtype=np.int32)
    padding = coreai.constant(padding, dtype=np.int32)
    ceil_mode = coreai.constant(ceil_mode)
    count_include_pad = coreai.constant(count_include_pad)
    divisor_override = coreai.constant(divisor_override_val, dtype=np.int32)

    return avg_pool3d_composite(
        x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    )


def _any_impl(x: Value, axes: list[int], keepdim: bool) -> Value:
    """Shared implementation for any operations across different variants."""
    x_type = x.type

    if x_type.element_type != IntegerType.get_signless(1):
        np_dtype = _get_coreai_to_numpy_dtype()[x_type.element_type]
        x = coreai.broadcasting_not_equal(x, np.zeros((), dtype=np_dtype))

    result = coreai.any_(x, axes)

    if not keepdim:
        result = coreai.shrink_dims(result, axes)

    return result


def replace_any_default(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Reduce across all dimensions, returning a scalar tensor."""
    x = _get_operand(values_map, node, 0)
    return _any_impl(x, list(range(x.type.rank)), keepdim=False)


def replace_any_dim(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Reduce across a single dimension."""
    x = _get_operand(values_map, node, 0)
    dim = node.args[1]
    keepdim = len(node.args) >= 3 and bool(node.args[2])
    return _any_impl(x, [dim], keepdim)


def replace_any_dims(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Reduce across multiple dimensions. If dims is None, reduce all."""
    x = _get_operand(values_map, node, 0)
    dims = (
        node.args[1]
        if len(node.args) > 1 and node.args[1] is not None
        else list(range(x.type.rank))
    )
    keepdim = len(node.args) > 2 and bool(node.args[2])
    return _any_impl(x, dims, keepdim)


def replace_arange_start_step(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    start, end = _get_operands(values_map, node, [0, 1])
    step = (
        _get_operand(values_map, node, 2)
        if len(node.args) > 2
        else coreai.constant(1, dtype=start.type.element_type)
    )

    # When ALL operands are integer-typed, keep them as si32 so coreai.range_
    # can infer a static output shape, then cast the result to the requested
    # dtype. If any operand is float (e.g. arange(0, 5, 0.5)), fall back to
    # target_type — truncating a float step to si32 would corrupt the values.
    target_type = get_output_element_type_from_node(node)
    si32 = IntegerType.get_signed(32)
    all_integer = all(
        isinstance(v.type.element_type, IntegerType) for v in (start, end, step)
    )
    range_type = si32 if all_integer else target_type

    def to_scalar(v: Value) -> Value:
        if v.type.rank > 0:
            v = coreai.shrink_dims(v, list(range(v.type.rank)))
        if v.type.element_type != range_type:
            v = coreai.cast(v, range_type)
        return v

    result = coreai.range_(to_scalar(start), to_scalar(end), to_scalar(step))
    if result.type.element_type != target_type:
        result = coreai.cast(result, target_type)
    return result


def replace_batch_norm(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> OpResultList:
    x = _get_operand(values_map, node, 0)
    weight = (
        _get_operand(values_map, node, 1)
        if node.args[1] is not None
        else coreai.constant(1.0)
    )
    bias = (
        _get_operand(values_map, node, 2)
        if node.args[2] is not None
        else coreai.constant(0.0)
    )
    running_mean, running_var, _momentum, eps = _get_operands(
        values_map, node, [3, 4, 5, 6]
    )

    x_type = x.type
    if not (2 <= x_type.rank <= 5):
        raise ValueError("batch norm only works for 2d to 5d inputs")

    ele_type = x_type.element_type

    input_names = ["input", "gamma", "beta", "mean", "variance"]
    output_names = ["output"]
    op_attributes = {"eps": node.args[6], "version": 1}
    composite_decl = generate_composite_decl(
        x.context, "batch_norm", input_names, output_names, op_attributes
    )

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def batch_norm(
        input: Value, gamma: Value, beta: Value, mean: Value, variance: Value
    ) -> Value:
        eps_casted = coreai.constant(node.args[6])

        eps_casted = coreai.cast(eps_casted, ele_type)

        expand_dims = [0] + list(
            range(2, x_type.rank)
        )  # expand [C] to [1, C, 1, ...] for broadcasting

        def expand_and_cast(v: Value) -> Value:
            return coreai.cast(coreai.expand_dims(v, expand_dims), ele_type)

        weight, bias, running_mean, running_var = (
            expand_and_cast(v) for v in [gamma, beta, mean, variance]
        )

        # ((x - mean) / sqrt(var + eps)) * weight + bias
        std = coreai.sqrt(coreai.broadcasting_add(running_var, eps_casted))
        normalized = coreai.broadcasting_divide(
            coreai.broadcasting_sub(input, running_mean), std
        )

        return coreai.broadcasting_add(
            coreai.broadcasting_mul(normalized, weight), bias
        )

    # In inference mode, outputs[1] and [2] are empty placeholder tensors (shape [0]).
    np_dtype = _get_coreai_to_numpy_dtype()[ele_type]
    empty = coreai.constant(np.array([], dtype=np_dtype))
    return batch_norm(x, weight, bias, running_mean, running_var)[0], empty, empty


def replace_binary_bitwise_ops(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    _op_map = {
        "bitwise_and.Tensor": coreai.broadcasting_bitwise_and,
        "bitwise_or.Tensor": coreai.broadcasting_bitwise_or,
        "bitwise_xor.Tensor": coreai.broadcasting_bitwise_xor,
    }
    x, y = _get_operands(values_map, node, [0, 1])
    promoted_type = get_promoted_type(x.type, y.type)
    return _op_map[get_target(node)](
        coreai.cast(x, promoted_type),
        coreai.cast(y, promoted_type),
    )


def replace_binary_comparision_ops(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    def ge(x, y):
        return coreai.not_(coreai.broadcasting_greater(y, x))

    def lt(x, y):
        return coreai.broadcasting_greater(y, x)

    def le(x, y):
        return coreai.not_(coreai.broadcasting_greater(x, y))

    _op_map = {
        "eq.Scalar": coreai.broadcasting_equal,
        "eq.Tensor": coreai.broadcasting_equal,
        "ge.Scalar": ge,
        "ge.Tensor": ge,
        "gt.Scalar": coreai.broadcasting_greater,
        "gt.Tensor": coreai.broadcasting_greater,
        "ne.Scalar": coreai.broadcasting_not_equal,
        "ne.Tensor": coreai.broadcasting_not_equal,
        "lt.Scalar": lt,
        "lt.Tensor": lt,
        "le.Scalar": le,
        "le.Tensor": le,
    }

    x, y = _get_operands(values_map, node, [0, 1])
    promoted_type = get_promoted_type(x.type, y.type)
    return _op_map[get_target(node)](
        coreai.cast(x, promoted_type),
        coreai.cast(y, promoted_type),
    )


def replace_binary_logical_ops(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    _op_map = {
        "logical_and.default": coreai.broadcasting_and,
        "logical_or.default": coreai.broadcasting_or,
        "logical_xor.default": coreai.broadcasting_xor,
    }
    x, y = _get_operands(values_map, node, [0, 1])
    return _op_map[get_target(node)](
        coreai.cast(x, np.bool_),
        coreai.cast(y, np.bool_),
    )


def replace_binary_ops(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    _op_map = {
        "add.Tensor": coreai.broadcasting_add,
        "add.Scalar": coreai.broadcasting_add,
        "add": coreai.broadcasting_add,
        "div.Tensor": coreai.broadcasting_divide,
        "div.Scalar": coreai.broadcasting_divide,
        "maximum.default": coreai.broadcasting_maximum,
        "minimum.default": coreai.broadcasting_minimum,
        "fmod.Tensor": coreai.broadcasting_modulo,
        "fmod.Scalar": coreai.broadcasting_modulo,
        "floordiv.Scalar": coreai.broadcasting_floor_divide,
        "floordiv.Tensor": coreai.broadcasting_floor_divide,
        "floordiv": coreai.broadcasting_floor_divide,
        "mod.Tensor": coreai.broadcasting_modulo,
        "mod.Scalar": coreai.broadcasting_modulo,
        "mod": coreai.broadcasting_modulo,
        "mul.Tensor": coreai.broadcasting_mul,
        "mul.Scalar": coreai.broadcasting_mul,
        "mul": coreai.broadcasting_mul,
        "pow.Scalar": coreai.broadcasting_pow,
        "pow.Tensor_Tensor": coreai.broadcasting_pow,
        "pow.Tensor_Scalar": coreai.broadcasting_pow,
        "pow": coreai.broadcasting_pow,
        "sub.Tensor": coreai.broadcasting_sub,
        "sub.Scalar": coreai.broadcasting_sub,
        "sub": coreai.broadcasting_sub,
    }

    x, y = _get_operands(values_map, node, [0, 1])
    promoted_type = get_promoted_type(x.type, y.type)
    result = _op_map[get_target(node)](
        x=coreai.cast(x, promoted_type),
        y=coreai.cast(y, promoted_type),
    )
    if isinstance(node.meta.get("val"), torch.Tensor):
        target_type = get_output_element_type_from_node(node)
        if result.type.element_type != target_type:
            result = coreai.cast(result, target_type)
    return result


def replace_truediv(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x, y = _get_operands(values_map, node, [0, 1])
    result_type = get_output_element_type_from_node(node)
    return coreai.broadcasting_divide(
        x=coreai.cast(x, result_type),
        y=coreai.cast(y, result_type),
    )


def replace_div_tensor_mode(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """aten.div.Tensor_mode(x, y, rounding_mode) -> Tensor

    rounding_mode:
        None    -> x / y
        "floor" -> floor_divide(x, y)
        "trunc" -> sign(x/y) * floor(abs(x/y))
    """
    x, y = _get_operands(values_map, node, [0, 1])

    rounding_mode = (
        node.kwargs.get("rounding_mode")
        if "rounding_mode" in node.kwargs
        else (node.args[2] if len(node.args) > 2 else None)
    )

    promoted_type = get_promoted_type(x.type, y.type)
    casted_x = coreai.cast(x, promoted_type)
    casted_y = coreai.cast(y, promoted_type)

    if rounding_mode is None:
        return coreai.broadcasting_divide(casted_x, casted_y)
    elif rounding_mode == "floor":
        return coreai.broadcasting_floor_divide(casted_x, casted_y)
    elif rounding_mode == "trunc":
        # Integer division already truncates toward zero, so a plain divide
        # is sufficient.  The full sign/abs/floor decomposition is only needed
        # for floating-point operands.
        if isinstance(promoted_type, IntegerType):
            return coreai.broadcasting_divide(casted_x, casted_y)
        # trunc(q) = sign(q) * floor(|q|), sign(q) = cast(q>0) - cast(q<0)
        quotient = coreai.broadcasting_divide(casted_x, casted_y)
        zero = coreai.constant(0, dtype=quotient.type.element_type)
        sign_q = coreai.broadcasting_sub(
            coreai.cast(coreai.broadcasting_greater(quotient, zero), promoted_type),
            coreai.cast(coreai.broadcasting_greater(zero, quotient), promoted_type),
        )
        return coreai.broadcasting_mul(
            sign_q,
            coreai.broadcasting_floor_divide(
                coreai.abs_(quotient),
                coreai.constant(1, dtype=quotient.type.element_type),
            ),
        )
    else:
        err = f"Unsupported rounding_mode: {rounding_mode!r}. Expected None, 'floor', or 'trunc'."
        raise ValueError(err)


def replace_bitwise_not(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    if x.type.element_type == IntegerType.get_signless(1):
        return coreai.not_(x)
    # -(x + 1): cast to float32, negate, cast back
    casted_x = coreai.cast(x, F32Type.get())
    return coreai.cast(
        coreai.broadcasting_mul(coreai.broadcasting_add(casted_x, 1.0), -1.0),
        x.type.element_type,
    )


def replace_bmm(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    mat1, mat2 = _get_operands(values_map, node, [0, 1])
    promoted_type = get_promoted_type(mat1.type, mat2.type)
    result = coreai.broadcasting_batch_matmul(
        coreai.cast(mat1, promoted_type),
        coreai.cast(mat2, promoted_type),
    )
    target_type = get_output_element_type_from_node(node)
    if result.type.element_type != target_type:
        result = coreai.cast(result, target_type)
    return result


def replace_cat(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    tensors_arg = node.args[0]
    assert isinstance(tensors_arg, (list, tuple)), (
        f"Expected list/tuple of tensors, got {type(tensors_arg)}"
    )
    dim = node.args[1] if len(node.args) > 1 else 0
    target_element_type = get_output_element_type_from_node(node)

    inputs = []
    for tensor_node in tensors_arg:
        assert isinstance(tensor_node, fx.Node), (
            f"Expected fx.Node, got {type(tensor_node)}"
        )
        input_val = values_map[tensor_node.name]
        # skip empty tensors (represent Python None or zero-element tensors)
        if input_val.type.has_static_shape and np.prod(input_val.type.shape) == 0:
            continue
        if input_val.type.element_type != target_element_type:
            input_val = coreai.cast(input_val, target_element_type)
        inputs.append(input_val)

    rank = inputs[0].type.rank
    dim = dim + rank if dim < 0 else dim

    # coreai.concat requires all non-concat dims to be provably equal across
    # inputs. Under dynamic shapes, one branch can carry a dynamic non-concat
    # axis while a sibling has a static size for the same axis — the dynamic
    # side must in fact equal that static size, but the type system doesn't
    # know it. Reshape such inputs to the known static size before the concat.
    # Multiple distinct static sizes on one axis is a real mismatch and is
    # left for the dialect verifier to reject.
    dyn = ShapedType.get_dynamic_size()

    def known_static(axis: int) -> int | None:
        if axis == dim:
            return None
        sizes = {inp.type.shape[axis] for inp in inputs if inp.type.shape[axis] != dyn}
        return next(iter(sizes)) if len(sizes) == 1 else None

    statics = [known_static(a) for a in range(rank)]
    promoted: list[Value] = []
    for inp in inputs:
        new_shape = [
            statics[a]
            if statics[a] is not None and inp.type.shape[a] == dyn
            else inp.type.shape[a]
            for a in range(rank)
        ]
        if new_shape != list(inp.type.shape):
            if all(s != dyn for s in new_shape):
                # All axes static post-promotion: list-form reshape packs
                # the shape into an int32 constant tensor.
                inp = coreai.reshape(inp, new_shape)
            else:
                # Mixed static / dynamic post-promotion: build the shape
                # vector at runtime by mixing the input's actual sizes
                # (via coreai.get_shape) for the still-dynamic axes with
                # constants for the promoted axes.
                runtime_shape = coreai.cast(coreai.get_shape(inp), dtype=np.int32)
                parts = [
                    coreai.constant([s], dtype=np.int32)
                    if s != dyn
                    else coreai.slice_(runtime_shape, [a], [a + 1], [1])
                    for a, s in enumerate(new_shape)
                ]
                result_type = RankedTensorType.get(new_shape, inp.type.element_type)
                inp = coreai.ReshapeOp(
                    inp, coreai.concat(0, parts), results=[result_type]
                ).result
        promoted.append(inp)
    inputs = promoted

    return coreai.concat(dim, inputs)


def replace_ceil(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    # ceil(x) = -floor(-x)
    x = _get_operand(values_map, node, 0)
    one = coreai.constant(1, dtype=x.type.element_type)
    neg_one = coreai.constant(-1, dtype=x.type.element_type)
    return coreai.broadcasting_mul(
        coreai.broadcasting_floor_divide(coreai.broadcasting_mul(x, neg_one), one),
        neg_one,
    )


def replace_clamp(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)

    def _get_bound(arg_idx: int) -> Value:
        arg = node.args[arg_idx]
        if isinstance(arg, (int, float)):
            return coreai.constant(float(arg), dtype=x.type.element_type)
        val = _get_operand(values_map, node, arg_idx)
        return coreai.cast(val, x.type.element_type)

    result = x
    if len(node.args) > 1 and node.args[1] is not None:
        result = coreai.broadcasting_maximum(result, _get_bound(1))
    if len(node.args) > 2 and node.args[2] is not None:
        result = coreai.broadcasting_minimum(result, _get_bound(2))
    return result


def replace_clone(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)
    return x


def replace_complex(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    real, imag = _get_operands(values_map, node, [0, 1])
    return coreai.create_complex(real, imag)


def replace_polar(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    abs_val, angle = _get_operands(values_map, node, [0, 1])
    real = coreai.broadcasting_mul(abs_val, coreai.cos(angle))
    imag = coreai.broadcasting_mul(abs_val, coreai.sin(angle))
    return coreai.create_complex(real, imag)


def replace_constant_pad_nd(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """aten.constant_pad_nd(input, pad, value=0) -> Tensor

    PyTorch pad format: [pad_left_lastdim, pad_right_lastdim, ...]  (reversed)
    Core AI pad format:  [pad_before_dim0, pad_after_dim0, ...]

    Negative padding values crop via coreai.slice_ before applying coreai.pad.
    """
    x = _get_operand(values_map, node, 0)
    inverted_padding = node.args[1]
    pad_value = (
        float(node.args[2]) if len(node.args) > 2 and node.args[2] is not None else 0.0
    )
    x_rank = x.type.rank

    # slice_start is always static: max(0, -pad_before).
    # slice_end entries: INT32_MAX = slice to end; >= 0 = static end;
    # < 0 = dynamic crop offset (dim_size + offset resolved at runtime).
    slice_start = [0] * x_rank
    slice_end = [INT32_MAX] * x_rank
    positive_padding = [0] * (2 * x_rank)
    has_negative = has_positive = False
    for i in range(0, len(inverted_padding), 2):
        dim = x_rank - (i // 2) - 1
        pad_before, pad_after = inverted_padding[i], inverted_padding[i + 1]
        slice_start[dim] = max(0, -pad_before)
        if pad_after < 0:
            dim_size = x.type.shape[dim]
            # Static dim: compute end now; dynamic dim: store raw offset (negative).
            slice_end[dim] = dim_size + pad_after if dim_size >= 0 else pad_after
        positive_padding[2 * dim] = max(0, pad_before)
        positive_padding[2 * dim + 1] = max(0, pad_after)
        has_negative = has_negative or pad_before < 0 or pad_after < 0
        has_positive = has_positive or pad_before > 0 or pad_after > 0

    result = x
    if has_negative:
        if any(e < 0 for e in slice_end):
            x_shape = coreai.cast(coreai.get_shape(x), dtype=np.int32)
            end_parts = []
            for d in range(x_rank):
                if slice_end[d] < 0:
                    dim_val = coreai.slice_(x_shape, [d], [d + 1], [1])
                    end_parts.append(coreai.broadcasting_add(dim_val, [slice_end[d]]))
                else:
                    end_parts.append([slice_end[d]])
            end = coreai.concat(0, end_parts) if x_rank > 1 else end_parts[0]
            result = coreai.slice_(result, slice_start, end, [1] * x_rank)
        else:
            result = coreai.slice_(result, slice_start, slice_end, [1] * x_rank)
    if has_positive:
        result = coreai.pad(
            result,
            np.array(positive_padding, dtype=np.uint32),
            coreai.cast(pad_value, x.type.element_type),
        )
    return result


def replace_reflection_pad(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """aten.reflection_pad{1,2,3}d.default -> coreai.pad<reflect>."""
    return _replace_pad_with_mode(values_map, node, loc, "reflect")


def replace_replication_pad(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """aten.replication_pad{1,2,3}d.default -> coreai.pad<replicate>."""
    return _replace_pad_with_mode(values_map, node, loc, "replicate")


def _conv_transpose(
    x: Value,
    weight: Value,
    bias: Value | None,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    output_padding: list[int],
    groups: Value,
    loc: Location,
) -> Value:
    """Handles transposed convolution (conv_transpose1d and conv_transpose2d).

    For 1D, expands to 2D, performs conv_transpose2d, then shrinks back.
    Handles output_padding via pre-padding input and post-cropping output.
    """
    is_1d = x.type.rank == 3
    if is_1d:
        # Expand 1D → 2D: [N,C,W] → [N,C,W,1]
        x = coreai.expand_dims(x, [-1])
        weight = coreai.expand_dims(weight, [-1])
        stride = stride + [1]
        padding = padding + [0]
        dilation = dilation + [1]
        output_padding = output_padding + [0]

    x_rank = x.type.rank
    effective_padding = padding
    pre_pad_amt = [0] * (x_rank * 2)
    post_crop_amt = [0] * (x_rank * 2)

    if any(p > 0 for p in output_padding):
        effective_padding = [0] * len(padding)
        pre_pad_amt = [0] * (x_rank * 2)
        post_crop_amt = [0] * (x_rank * 2)
        # For each spatial dim: initialize symmetric crop from padding,
        # then shift the output_padding amount from crop → pre-pad if needed
        for i, (p, op) in enumerate(zip(padding, output_padding)):
            before = 4 + 2 * i
            after = 4 + 2 * i + 1
            post_crop_amt[before] = p
            post_crop_amt[after] = p
            if post_crop_amt[after] >= op:
                post_crop_amt[after] -= op
            else:
                pre_pad_amt[after] = op - post_crop_amt[after]
                post_crop_amt[after] = 0

    if any(p > 0 for p in pre_pad_amt):
        x = coreai.pad(
            x,
            np.array(pre_pad_amt, dtype=np.uint32),
            coreai.constant(0, dtype=x.type.element_type),
        )
    stride = coreai.constant(stride, np.uint32)
    effective_padding = coreai.constant(effective_padding, np.uint32)
    dilation = coreai.constant(dilation, np.uint32)
    output_padding = coreai.constant([0, 0], dtype=np.uint32)
    groups = coreai.constant(groups, np.uint32)
    result = coreai.conv_transpose2d(
        input=x,
        weight=weight,
        stride=stride,
        padding=effective_padding,
        dilation=dilation,
        output_pad=output_padding,
        groups=groups,
    )

    if any(p > 0 for p in post_crop_amt):
        stop_val = coreai.sub(
            coreai.cast(coreai.get_shape(result), dtype=np.int32),
            [post_crop_amt[2 * d + 1] for d in range(x_rank)],
        )
        result = coreai.slice_(
            result,
            [post_crop_amt[2 * d] for d in range(x_rank)],
            stop_val,
            [1] * x_rank,
        )

    if is_1d:
        # Shrink back to 3D: [N,C,W,1] → [N,C,W]
        result = coreai.reshape(
            result, coreai.slice_(coreai.get_shape(result), [0], [3], [1])
        )

    if bias is not None:
        bias_shape = (
            [1, bias.type.shape[0], 1] if is_1d else [1, bias.type.shape[0], 1, 1]
        )
        result = coreai.broadcasting_add(result, coreai.reshape(bias, bias_shape))

    return result


def replace_conv(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x, weight = _get_operands(values_map, node, [0, 1])
    bias = None if node.args[2] is None else _get_operand(values_map, node, 2)
    stride = node.args[3]
    padding = node.args[4]
    dilation = node.args[5]
    is_transposed = bool(node.args[6])
    output_padding = node.args[7]
    groups = node.args[8]

    # Cast weight/bias to match input dtype
    if x.type.element_type != weight.type.element_type:
        weight = coreai.cast(weight, x.type.element_type)
    if bias is not None and x.type.element_type != bias.type.element_type:
        bias = coreai.cast(bias, x.type.element_type)

    rank = x.type.rank
    if rank not in (3, 4, 5):
        raise ValueError(
            f"Only conv1d, conv2d and conv3d are supported, at node: {node}, name: {node.name}"
        )

    if is_transposed:
        if rank == 5:
            raise ValueError(
                f"Transposed conv3d is not yet supported, at node: {node}, name: {node.name}"
            )
        return _conv_transpose(
            x, weight, bias, stride, padding, dilation, output_padding, groups, loc
        )

    if any(p != 0 for p in output_padding):
        raise ValueError(
            f"non-zero output padding is not supported in convolution, got {output_padding}"
        )

    if rank == 3:
        # Conv1d: expand to 2D by adding a dummy height dimension
        x = coreai.expand_dims(x, [2])
        weight = coreai.expand_dims(weight, [2])
        if any(p != 0 for p in padding):
            x = coreai.pad(
                x,
                np.array([0, 0, 0, 0, 0, 0, padding[0], padding[0]], dtype=np.uint32),
                coreai.constant(0, dtype=x.type.element_type),
                "constant",
            )
        result = coreai.shrink_dims(
            coreai.conv2d(x, weight, [1] + stride, [1] + dilation, groups), [2]
        )
        if bias is not None:
            result = coreai.broadcasting_add(
                result, coreai.reshape(bias, [1, bias.type.shape[0], 1])
            )
    elif rank == 4:
        # Conv2d
        if any(p != 0 for p in padding):
            pad_h, pad_w = padding[0], padding[1]
            x = coreai.pad(
                x,
                np.array([0, 0, 0, 0, pad_h, pad_h, pad_w, pad_w], dtype=np.uint32),
                coreai.constant(0, dtype=x.type.element_type),
                "constant",
            )
        result = coreai.conv2d(x, weight, stride, dilation, groups)
        if bias is not None:
            result = coreai.broadcasting_add(
                result, coreai.reshape(bias, [1, bias.type.shape[0], 1, 1])
            )
    elif rank == 5:
        # Conv3d
        if any(p != 0 for p in padding):
            pad_d, pad_h, pad_w = padding[0], padding[1], padding[2]
            x = coreai.pad(
                x,
                np.array(
                    [0, 0, 0, 0, pad_d, pad_d, pad_h, pad_h, pad_w, pad_w],
                    dtype=np.uint32,
                ),
                coreai.constant(0, dtype=x.type.element_type),
                "constant",
            )
        result = coreai.conv3d(
            x,
            weight,
            coreai.constant(stride, np.uint32),
            coreai.constant(dilation, np.uint32),
            coreai.constant(groups, np.uint32),
        )
        if bias is not None:
            result = coreai.broadcasting_add(
                result, coreai.reshape(bias, [1, bias.type.shape[0], 1, 1, 1])
            )
    else:
        raise ValueError(
            f"Only conv1d, conv2d and conv3d are supported in Core AI compiler, at the node: {node}, name: {node.name}"
        )
    return result


def replace_copy(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    dest, src = _get_operands(values_map, node, [0, 1])
    casted = coreai.cast(src, dest.type.element_type)
    target_type = get_tensor_type(node.meta["val"])
    if any(d < 0 for d in dest.type.shape):
        return coreai.BroadcastToOp(
            casted,
            coreai.get_shape(dest),
            results=[target_type],
        ).result
    return coreai.broadcast_to(casted, dest.type.shape)


def replace_embedding(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """aten.embedding(weight, indices) -> gather_nd(weight, indices[..., None])"""
    weight, indices = _get_operands(values_map, node, [0, 1])
    return coreai.gather_nd(
        input=weight, indices=coreai.expand_dims(indices, [indices.type.rank])
    )


def replace_empty(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    # torch.empty is lowered to zeros for deterministic behavior
    shape = node.args[0]
    np_dtype = _get_coreai_to_numpy_dtype()[get_output_element_type_from_node(node)]
    # With dynamic shapes, dims come from sym_size.int nodes rather than static ints.
    if any(isinstance(s, fx.Node) for s in shape):
        shape_tensor = build_shape_tensor(values_map, shape)
        # broadcast_to requires input rank == output rank; use all-ones static shape.
        rank = len(shape)
        # Explicit result type preserves static dims in the IR type.
        return coreai.BroadcastToOp(
            coreai.constant(np.zeros([1] * rank, np_dtype)),
            coreai.cast(shape_tensor, np.uint32),
            results=[get_tensor_type(node.meta["val"])],
        ).result
    return coreai.constant(np.zeros(shape, np_dtype))


def replace_sym_size_int(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """aten.sym_size.int(tensor, dim) -> size of tensor along dim as a shape-[1] tensor."""
    tensor = values_map[node.args[0].name]
    dim = node.args[1]
    return coreai.cast(
        coreai.slice_(coreai.get_shape(tensor), [dim], [dim + 1], [1]), dtype=np.int32
    )


def replace_sym_min(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    a, b = _get_operands(values_map, node, [0, 1])
    return coreai.broadcasting_minimum(a, b)


def replace_sym_float(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """aten.sym_float(sym_int) -> cast a SymInt scalar tensor to a SymFloat scalar tensor."""
    x = _get_operand(values_map, node, 0)
    return coreai.cast(x, dtype=np.float32)


def replace_exp2(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    # exp2(x) = 2^x
    x = _get_operand(values_map, node, 0)
    return coreai.broadcasting_pow(coreai.constant(2, dtype=x.type.element_type), x)


def replace_expand(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)

    fx_size_attr = node.args[1]
    for _ in range(x.type.rank, len(fx_size_attr)):
        x = coreai.expand_dims(x, [0])

    target_type = get_tensor_type(node.meta["val"])
    if x.type.element_type != target_type.element_type:
        x = coreai.cast(x, target_type.element_type)

    # Prefer broadcast_in_dims over concat+broadcast_to when possible.
    # Only axes where input is statically size 1 and target differs are listed.
    broadcast_axes: list[int] = []
    broadcast_sizes: list[Value] = []

    for i, dim in enumerate(fx_size_attr):
        if isinstance(dim, int) and (dim == -1 or x.type.shape[i] == dim):
            continue
        if x.type.shape[i] != 1:
            continue
        broadcast_axes.append(i)
        if isinstance(dim, int):
            broadcast_sizes.append(coreai.constant([dim], dtype=np.uint32))
        elif isinstance(dim, fx.Node):
            dim_val = coreai.cast(values_map[dim.name], dtype=np.uint32)
            if dim_val.type.rank == 0:
                dim_val = coreai.reshape(dim_val, [1])
            broadcast_sizes.append(dim_val)
        else:
            assert False, f"Unknown dimension argument type: {type(dim)}"

    if broadcast_axes:
        axes = coreai.constant(broadcast_axes, dtype=np.int32)
        dim_sizes = (
            coreai.concat(0, broadcast_sizes)
            if len(broadcast_sizes) > 1
            else broadcast_sizes[0]
        )
        return coreai.BroadcastInDimsOp(
            x, dim_sizes, axes, results=[target_type]
        ).result

    # Fallback: build full shape tensor via concat → BroadcastToOp.
    size = []
    for i, dim in enumerate(fx_size_attr):
        if isinstance(dim, int):
            if dim == -1:
                dim = get_tensor_shape_at_index(x, i)
            else:
                dim = coreai.constant(dim)
        elif isinstance(dim, fx.Node):
            dim = values_map[dim.name]
        else:
            assert False, f"Unknown dimension argument type: {type(dim)}"
        size.append(coreai.cast(dim, dtype=np.uint32))

    size = coreai.concat(0, size)
    return coreai.BroadcastToOp(x, size, results=[target_type]).result


def replace_expm1(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    # expm1(x) = exp(x) - 1
    x = _get_operand(values_map, node, 0)
    return coreai.broadcasting_sub(
        coreai.exp(x), coreai.constant(1, dtype=x.type.element_type)
    )


def replace_flip(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)
    dims = np.array(node.args[1], dtype=np.int32)
    return coreai.reverse(x, dims)


def replace_floor(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    # floor(x) = floor_divide(x, 1)
    x = _get_operand(values_map, node, 0)
    return coreai.broadcasting_floor_divide(
        x, coreai.constant(1, dtype=x.type.element_type)
    )


def replace_floor_divide(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x, y = _get_operands(values_map, node, [0, 1])
    promoted_type = get_promoted_type(x.type, y.type)
    return coreai.broadcasting_floor_divide(
        coreai.cast(x, promoted_type),
        coreai.cast(y, promoted_type),
    )


def replace_full(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    shape = node.args[0]
    np_dtype = _get_coreai_to_numpy_dtype()[get_output_element_type_from_node(node)]
    if any(isinstance(s, fx.Node) for s in shape):
        shape_tensor = build_shape_tensor(values_map, shape)
        rank = len(shape)
        return coreai.BroadcastToOp(
            coreai.constant(np.full([1] * rank, node.args[1], np_dtype)),
            coreai.cast(shape_tensor, np.uint32),
            results=[get_tensor_type(node.meta["val"])],
        ).result
    return coreai.constant(np.full(shape, node.args[1], np_dtype))


def replace_full_like(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    target_type = get_tensor_type(node.meta["val"])
    # Use target element type to avoid mismatched int64 input / int32 output.
    fill_val = coreai.constant(node.args[1], dtype=target_type.element_type)
    if any(d < 0 for d in x.type.shape):
        return coreai.BroadcastToOp(
            fill_val,
            coreai.get_shape(x),
            results=[target_type],
        ).result
    return coreai.broadcast_to(fill_val, x.type.shape)


def replace_argmax(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    """Converts aten.argmax to coreai.argmax."""
    x = _get_operand(values_map, node, 0)
    dim = node.args[1] if len(node.args) > 1 else None
    keepdim = node.args[2] if len(node.args) > 2 else False

    if dim is None:
        shape = x.type.shape
        if any(d < 0 for d in shape):
            numel = coreai.reduce_product(coreai.get_shape(x), [0])
        else:
            numel = coreai.constant([int(np.prod(shape))])
        flat = coreai.reshape(x, numel)
        return coreai.cast(coreai.shrink_dims(coreai.argmax(flat, 0), [0]), np.int32)

    dim = dim + x.type.rank if dim < 0 else dim
    result = coreai.cast(coreai.argmax(x, dim), np.int32)
    return result if keepdim else coreai.shrink_dims(result, [dim])


def replace_atan2(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    """Lower atan2(y, x) using atan(y/x) with quadrant correction.

    CoreAI has no native atan2, so it is decomposed as:
      - x != 0, finite: atan(y/x) adjusted by ±π for the correct quadrant.
      - x == +0: ±π/2 for non-zero y, 0 for y = 0.
      - x == -0: ±π for all y (including ±0 → ±π per IEEE-754).
      - both infinite: ±π/4 or ±3π/4 per IEEE-754.

    Signed-zero handling: IEEE-754 treats -0.0 as distinct from +0.0 for atan2
    (e.g. atan2(-0, -1) = -π, not +π). The 1/v trick — 1/-0.0 = -inf — is used
    to detect the sign bit of zero inputs so that y_neg and x_neg are correct
    for -0.0 inputs without misclassifying ±inf (which use the strict > path).

    When x=0, x is replaced with 1 before the divide solely to avoid NaN/inf; that
    intermediate result is discarded by the final where-select.
    atan2(0, 0) = 0 by convention.
    """
    y, x = _get_operands(values_map, node, [0, 1])
    ele_type = x.type.element_type

    zero = coreai.constant(0.0, dtype=ele_type)
    one = coreai.constant(1.0, dtype=ele_type)
    pi = coreai.constant(np.pi, dtype=ele_type)
    neg_pi = coreai.constant(-np.pi, dtype=ele_type)
    half_pi = coreai.constant(np.pi / 2.0, dtype=ele_type)
    neg_half_pi = coreai.constant(-np.pi / 2.0, dtype=ele_type)
    quarter_pi = coreai.constant(np.pi / 4.0, dtype=ele_type)
    neg_quarter_pi = coreai.constant(-np.pi / 4.0, dtype=ele_type)
    three_quarter_pi = coreai.constant(3.0 * np.pi / 4.0, dtype=ele_type)
    neg_three_quarter_pi = coreai.constant(-3.0 * np.pi / 4.0, dtype=ele_type)

    # ── signed-zero-aware sign predicates ─────────────────────────────────────
    # 1 / -0.0 = -inf (IEEE-754), so (0 > 1/v) is True iff v = -0.0. Combine with
    # the strict > predicate (handles ±inf and non-zero finites) via OR.
    y_is_zero = coreai.broadcasting_equal(y, zero)
    x_is_zero = coreai.broadcasting_equal(x, zero)
    y_neg = coreai.broadcasting_or(
        coreai.broadcasting_greater(zero, y),
        coreai.broadcasting_and(
            y_is_zero,
            coreai.broadcasting_greater(zero, coreai.broadcasting_divide(one, y)),
        ),
    )
    x_neg = coreai.broadcasting_or(
        coreai.broadcasting_greater(zero, x),
        coreai.broadcasting_and(
            x_is_zero,
            coreai.broadcasting_greater(zero, coreai.broadcasting_divide(one, x)),
        ),
    )
    x_is_neg_zero = coreai.broadcasting_and(
        x_is_zero,
        coreai.broadcasting_greater(zero, coreai.broadcasting_divide(one, x)),
    )

    # ── both-infinite branch ──────────────────────────────────────────────────
    # atan(inf/inf) = atan(NaN) = NaN; handle before the divide.
    pos_inf = coreai.constant(float("inf"), dtype=ele_type)
    neg_inf = coreai.constant(float("-inf"), dtype=ele_type)
    x_is_inf = coreai.broadcasting_or(
        coreai.broadcasting_equal(x, pos_inf), coreai.broadcasting_equal(x, neg_inf)
    )
    y_is_inf = coreai.broadcasting_or(
        coreai.broadcasting_equal(y, pos_inf), coreai.broadcasting_equal(y, neg_inf)
    )
    both_inf = coreai.broadcasting_and(x_is_inf, y_is_inf)
    inf_result = coreai.broadcasting_where(
        y_neg,
        coreai.broadcasting_where(x_neg, neg_three_quarter_pi, neg_quarter_pi),
        coreai.broadcasting_where(x_neg, three_quarter_pi, quarter_pi),
    )

    # ── x = 0 branch ──────────────────────────────────────────────────────────
    # x = +0: ±π/2 for strictly ±y, 0 when y = 0.
    # x = -0: ±π for all y (y_neg covers y = -0.0 via the 1/y trick above).
    y_pos_strict = coreai.broadcasting_greater(y, zero)
    y_neg_strict = coreai.broadcasting_greater(zero, y)
    pos_x_zero_result = coreai.broadcasting_where(
        y_pos_strict,
        half_pi,
        coreai.broadcasting_where(y_neg_strict, neg_half_pi, zero),
    )
    neg_x_zero_result = coreai.broadcasting_where(y_neg, neg_pi, pi)
    zero_result = coreai.broadcasting_where(
        x_is_neg_zero, neg_x_zero_result, pos_x_zero_result
    )

    # ── finite nonzero x branch ────────────────────────────────────────────────
    # Avoid division by zero: substitute x = 1 when x = 0; result discarded by
    # the outer where-select.
    x_safe = coreai.broadcasting_where(x_is_zero, one, x)
    base = coreai.atan(coreai.broadcasting_divide(y, x_safe))
    correction = coreai.broadcasting_where(
        y_neg,
        coreai.broadcasting_sub(base, pi),
        coreai.broadcasting_add(base, pi),
    )
    nonzero_result = coreai.broadcasting_where(x_neg, correction, base)

    # ── combine ────────────────────────────────────────────────────────────────
    result = coreai.broadcasting_where(x_is_zero, zero_result, nonzero_result)
    return coreai.broadcasting_where(both_inf, inf_result, result)


def replace_gather(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    """Converts aten.gather to coreai.gather_along_axis."""
    x, index = _get_operands(values_map, node, [0, 2])
    index = coreai.cast(index, np.int32)
    return coreai.gather_along_axis(x, index, node.args[1])


def replace_gelu(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)
    return coreai.gelu(x, approximate=node.kwargs.get("approximate", "none"))


def replace_getitem(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    item_idx_name = f"{node.args[0].name}#{node.args[1]}"
    if item_idx_name in values_map:
        return values_map[item_idx_name]
    return values_map[node.args[0].name]


def replace_group_norm(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> OpResultList:
    """Group norm: divides C channels into G groups and normalizes each independently."""
    x = _get_operand(values_map, node, 0)
    weight = None if node.args[1] is None else _get_operand(values_map, node, 1)
    bias = None if node.args[2] is None else _get_operand(values_map, node, 2)
    N, C, HW, group, eps = node.args[3:8]
    input_ele_type = x.type.element_type

    if weight is None:
        weight = coreai.constant([1.0] * C, dtype=np.float32)
    if bias is None:
        bias = coreai.constant([0.0] * C, dtype=np.float32)

    input_names = ["input", "weight", "bias"]
    output_names = ["output"]
    op_attributes = {"num_groups": group, "num_channels": C, "eps": eps, "version": 1}
    composite_decl = generate_composite_decl(
        x.context, "group_norm", input_names, output_names, op_attributes
    )

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def group_norm(input: Value, weight: Value, bias: Value) -> Value:
        # Reshape to [B, G, C//G, -1] for per-group normalization.
        batch_dim = coreai.cast(
            coreai.slice_(coreai.get_shape(input), [0], [1], [1]), dtype=np.int32
        )
        reshape_target = coreai.concat(0, [batch_dim, [group, C // group, -1]])
        reshaped_input = coreai.reshape(input, reshape_target)
        reshaped_input_compute, compute_type, use_fp32_stats = (
            prepare_compute_type_for_norm(reshaped_input, input_ele_type)
        )

        # Mean and variance over [C//G, H*W] dims using stable formula E((x-mean)^2)
        mean = coreai.reduce_mean(reshaped_input_compute, [2, 3])
        centered = coreai.broadcasting_sub(reshaped_input_compute, mean)
        var = coreai.reduce_mean(
            coreai.broadcasting_mul(centered, centered),
            [2, 3],
        )

        # rstd = 1 / sqrt(var + eps)
        rstd = coreai.rsqrt(
            coreai.broadcasting_add(
                var, coreai.constant(eps, dtype=var.type.element_type)
            )
        )

        # Multiply in compute precision (fp32 if use_fp32_stats), reshape, then cast once
        norm = coreai.broadcasting_mul(centered, rstd)
        norm = coreai.reshape(norm, coreai.get_shape(input))
        if use_fp32_stats:
            norm = coreai.cast(norm, input_ele_type)
            mean = coreai.cast(mean, input_ele_type)
            rstd = coreai.cast(rstd, input_ele_type)

        # Affine transform — reshape weight/bias to [C, 1, ...] for broadcasting.
        bcast_shape = [C] + [1] * (x.type.rank - 2)

        weight = coreai.cast(weight, input_ele_type)
        bias = coreai.cast(bias, input_ele_type)

        result = coreai.broadcasting_add(
            coreai.broadcasting_mul(norm, coreai.reshape(weight, bcast_shape)),
            coreai.reshape(bias, bcast_shape),
        )
        assert result.type == x.type, (
            f"Result type must be identical to input type, got result: {result.type}, input: {x.type}"
        )
        return [result]

    return group_norm(x, weight, bias)


def replace_index_select(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.index_select to coreai.gather_along_axis."""
    x, indices = _get_operands(values_map, node, [0, 2])
    dim = node.args[1]
    dim = dim + x.type.rank if dim < 0 else dim

    # Reshape 1D index [N] to [1,...,N,...,1] then broadcast to input shape at dim
    indices_size = indices.type.shape[0]
    bcast_shape = list(x.type.shape)
    bcast_shape[dim] = indices_size

    idx_shape = [1] * x.type.rank
    idx_shape[dim] = indices_size

    indices_size_tensor = coreai.get_shape(indices)
    if indices_size < 0:
        # Dynamic index size: build reshape target [1,...,N,...,1] at runtime.

        parts_idx: list = []
        if dim > 0:
            parts_idx.append(coreai.constant([1] * dim, dtype=np.uint32))
        parts_idx.append(indices_size_tensor)
        if dim < x.type.rank - 1:
            parts_idx.append(
                coreai.constant([1] * (x.type.rank - dim - 1), dtype=np.uint32)
            )
        idx_shape_val = (
            coreai.concat(0, parts_idx) if len(parts_idx) > 1 else parts_idx[0]
        )
        reshaped = coreai.reshape(indices, idx_shape_val)
    else:
        reshaped = coreai.reshape(indices, idx_shape)

    # The reshaped indices have shape [1,...,N,...,1] — size 1 on every axis
    # except `dim`. Use broadcast_in_dims to expand those size-1 axes to match
    # x's shape, avoiding a concat → broadcast_to pattern.
    broadcast_axes = [i for i in range(x.type.rank) if i != dim]

    if not broadcast_axes:
        # 1D input: no axes to broadcast, reshaped indices are already correct.
        broadcasted = reshaped
    elif any(d < 0 for d in bcast_shape):
        # x has dynamic dims: extract the sizes for the broadcast axes from
        # x's runtime shape.
        x_shape_tensor = coreai.get_shape(x)
        dim_size_parts = []
        for i in broadcast_axes:
            dim_size_parts.append(coreai.slice_(x_shape_tensor, [i], [i + 1], [1]))
        dim_sizes = (
            coreai.concat(0, dim_size_parts)
            if len(dim_size_parts) > 1
            else dim_size_parts[0]
        )
        axes = coreai.constant(broadcast_axes, dtype=np.int32)
        bcast_result_type = RankedTensorType.get(
            bcast_shape, reshaped.type.element_type
        )
        broadcasted = coreai.BroadcastInDimsOp(
            reshaped, dim_sizes, axes, results=[bcast_result_type]
        ).result
    else:
        dim_sizes = coreai.constant(
            [bcast_shape[i] for i in broadcast_axes], dtype=np.uint32
        )
        axes = coreai.constant(broadcast_axes, dtype=np.int32)
        bcast_result_type = RankedTensorType.get(
            bcast_shape, reshaped.type.element_type
        )
        broadcasted = coreai.BroadcastInDimsOp(
            reshaped, dim_sizes, axes, results=[bcast_result_type]
        ).result

    return coreai.gather_along_axis(x, broadcasted, dim)


def replace_index_put(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.index_put to coreai.scatter_nd.

    Implements NumPy-style advanced indexing for in-place updates. Handles boolean masks,
    integer indices, None dimensions, broadcasting, and non-contiguous indexing via transpose.
    Follows Core AI's approach (DecompositionUtils.cpp).

    Limitation: Dynamic boolean masks that are all False at runtime cause broadcast_to crash.
    """
    input_val = _get_operand(values_map, node, 0)
    updates = _get_operand(values_map, node, 2)

    if len(node.args) > 3 and node.args[3]:
        raise ValueError(
            "index_put with accumulate=True is not supported. "
            "The coreai dialect does not support this operation configuration."
        )

    if any(d < 0 for t in (input_val, updates) for d in t.type.shape[1:]):
        raise ValueError(
            "index_put currently only supports dynamic batch dimension (dim 0); "
            "all other dimensions must be static"
        )

    expanded_indices = expand_boolean_indices(input_val, values_map, node.args[1], loc)

    # Fast path: all indices are None
    if all(idx is None for idx in expanded_indices):
        return coreai.broadcast_to(updates, list(input_val.type.shape))

    stacked_info = process_expanded_indices(input_val, expanded_indices, loc)

    # Early return for statically-known empty indices (boolean mask with no True values)
    stacked_indices_shape = stacked_info.stacked_indices.type.shape
    if stacked_indices_shape and stacked_indices_shape[0] == 0:
        return input_val

    num_indexed_dims = stacked_indices_shape[-1]
    broadcast_shape = stacked_indices_shape[:-1]
    remaining_dims = list(stacked_info.base.type.shape[num_indexed_dims:])
    updates_shape = updates.type.shape

    # Transpose updates for contiguous indices not starting from dim 0
    if stacked_info.inverse_result_permutation is not None:
        if len(updates_shape) == stacked_info.inverse_result_permutation.type.shape[0]:
            updates = coreai.transpose(updates, stacked_info.inverse_result_permutation)
            updates_shape = updates.type.shape

    expected_updates_shape = broadcast_shape + remaining_dims

    if any(d < 0 for d in expected_updates_shape):
        # Dynamic shape: broadcast scalar/single-element updates to runtime shape
        if len(updates_shape) == 0 or (
            len(updates_shape) == 1 and updates_shape[0] == 1
        ):
            num_indices_tensor = coreai.slice_(
                coreai.get_shape(stacked_info.stacked_indices), [0], [1], [1]
            )
            target_shape = (
                coreai.concat(
                    0,
                    [
                        num_indices_tensor,
                        coreai.constant(remaining_dims, dtype=np.uint32),
                    ],
                )
                if remaining_dims
                else num_indices_tensor
            )
            if len(updates_shape) == 0:
                updates = coreai.reshape(updates, [1])
            updates = coreai.broadcast_to(updates, target_shape)
    elif updates_shape != expected_updates_shape:
        if len(updates_shape) < len(expected_updates_shape):
            updates = coreai.expand_dims(
                updates, list(range(len(expected_updates_shape) - len(updates_shape)))
            )
        updates = coreai.broadcast_to(updates, expected_updates_shape)

    updates = coreai.cast(updates, stacked_info.base.type.element_type)

    result = coreai.scatter_nd(
        input=stacked_info.base,
        indices=stacked_info.stacked_indices,
        updates=updates,
    )

    if stacked_info.base_inverse_permutation is not None:
        result = coreai.transpose(result, stacked_info.base_inverse_permutation)

    return result


def replace_index_tensor(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.index.Tensor to coreai.gather_nd."""
    input_val = _get_operand(values_map, node, 0)
    stacked_info = process_indices_with_transpose(
        input_val, values_map, node.args[1], loc
    )
    result = coreai.gather_nd(
        input=stacked_info.base, indices=stacked_info.stacked_indices
    )
    if stacked_info.result_permutation is not None:
        result = coreai.transpose(result, stacked_info.result_permutation)
    return result


def replace_hard_tanh(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    min_val = coreai.constant(node.args[1], dtype=x.type.element_type)
    max_val = coreai.constant(node.args[2], dtype=x.type.element_type)
    return coreai.broadcasting_minimum(coreai.broadcasting_maximum(x, min_val), max_val)


def replace_hardsigmoid(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """hardsigmoid(x) = min(max(x + 3, 0), 6) / 6"""
    x = _get_operand(values_map, node, 0)
    hard_sigmoid = build_hard_sigmoid_composite(x.context)
    return hard_sigmoid(x)


def replace_hardswish(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """hardswish(x) = x * hardsigmoid(x)"""
    x = _get_operand(values_map, node, 0)
    hard_sigmoid = build_hard_sigmoid_composite(x.context)
    hsigmoid = hard_sigmoid(x)[0]
    return coreai.broadcasting_mul(x, hsigmoid)


def replace_layer_norm(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> OpResultList:
    """Layer norm: normalizes over the last N dimensions specified by normalized_shape."""
    x = _get_operand(values_map, node, 0)
    normalized_shape = node.args[1]
    weight = None if node.args[2] is None else _get_operand(values_map, node, 2)
    bias = None if node.args[3] is None else _get_operand(values_map, node, 3)
    eps = node.args[4]

    numel = int(np.prod(normalized_shape))
    if weight is None:
        weight = coreai.constant([1.0] * numel, dtype=np.float32)
    if bias is None:
        bias = coreai.constant([0.0] * numel, dtype=np.float32)

    input_rank = x.type.rank
    input_ele_type = x.type.element_type
    # Normalize over the last len(normalized_shape) dims
    dim_list = list(range(input_rank - len(normalized_shape), input_rank))

    input_names = ["input", "gamma", "beta"]
    output_names = ["output"]
    op_attributes = {
        "eps": eps,
        "axes": dim_list,
        "version": 1,
    }
    composite_decl = generate_composite_decl(
        x.context, "layer_norm", input_names, output_names, op_attributes
    )

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def layer_norm(input: Value, gamma: Value, beta: Value) -> Value:
        x_compute, _, use_fp32_stats = prepare_compute_type_for_norm(
            input, input_ele_type, loc
        )

        # Mean and variance using stable formula E((x-mean)^2)
        mean = coreai.reduce_mean(x_compute, dim_list)
        centered = coreai.broadcasting_sub(x_compute, mean)
        var = coreai.reduce_mean(coreai.broadcasting_mul(centered, centered), dim_list)
        rstd = coreai.rsqrt(
            coreai.broadcasting_add(
                var, coreai.constant(eps, dtype=var.type.element_type)
            )
        )

        # Multiply in compute precision then cast once
        norm = coreai.broadcasting_mul(centered, rstd)
        if use_fp32_stats:
            norm = coreai.cast(norm, input_ele_type)
            mean = coreai.cast(mean, input_ele_type)
            rstd = coreai.cast(rstd, input_ele_type)

        # Affine transform with runtime broadcasting.
        if gamma is None and beta is None:
            return [norm, mean, rstd]

        weight = coreai.cast(gamma, input_ele_type)
        bias = coreai.cast(beta, input_ele_type)

        # Reshape for multi-dim normalized_shape (e.g. [32] → [4, 8]).
        if len(normalized_shape) > 1:
            weight = coreai.reshape(weight, list(normalized_shape))
            bias = coreai.reshape(bias, list(normalized_shape))

        # Broadcast gamma/beta to match the norm output shape.
        norm_shape = coreai.get_shape(norm)
        w_shape = coreai.constant(list(normalized_shape), dtype=np.uint32)
        target_shape = coreai.broadcast_shapes(norm_shape, w_shape)
        weight = coreai.BroadcastToOp(weight, target_shape, results=[norm.type]).result
        bias = coreai.BroadcastToOp(bias, target_shape, results=[norm.type]).result

        result = coreai.broadcasting_add(
            coreai.broadcasting_mul(norm, weight),
            bias,
        )
        return result

    return layer_norm(x, weight, bias)


def replace_leaky_relu(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """leaky_relu(x, negative_slope) = max(0, x) + negative_slope * min(0, x)"""
    x = _get_operand(values_map, node, 0)
    negative_slope = node.args[1] if len(node.args) > 1 else 0.01
    zero = coreai.constant(0, dtype=x.type.element_type)
    slope = coreai.constant(negative_slope, dtype=x.type.element_type)
    return coreai.broadcasting_add(
        coreai.broadcasting_maximum(x, zero),
        coreai.broadcasting_mul(coreai.broadcasting_minimum(x, zero), slope),
    )


def replace_lift_fresh_copy(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    return x


def replace_local_scalar_dense(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """aten._local_scalar_dense: return the 0-dim input tensor as-is."""
    return _get_operand(values_map, node, 0)


def replace_linalg_vector_norm(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Linalg vector norm: L0/L1/L2/Linf/Lp norms along specified dims."""
    x = _get_operand(values_map, node, 0)
    args = node.args
    ord_val = args[1] if len(args) > 1 and args[1] is not None else 2.0
    dim = args[2] if len(args) > 2 and args[2] is not None else None
    keepdim = args[3] if len(args) > 3 and args[3] is not None else False

    input_names = ["input"]
    output_names = ["output"]
    op_attributes = {
        "ord": ord_val,
        "keep_dim": keepdim,
        "axes": list(range(len(x.type.shape))) if dim is None else dim,
        "version": 1,
    }
    composite_decl = generate_composite_decl(
        x.context, "linalg_vector_norm", input_names, output_names, op_attributes
    )

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def linalg_vector_norm(input: Value) -> Value:
        dims = dim if dim is not None else list(range(input.type.rank))
        if ord_val == 0.0:
            result = coreai.reduce_sum(
                coreai.cast(
                    coreai.broadcasting_not_equal(
                        input, coreai.constant(0, dtype=input.type.element_type)
                    ),
                    input.type.element_type,
                ),
                dims,
            )
        elif ord_val == 1.0:
            result = coreai.reduce_sum(coreai.abs_(input), dims)
        elif ord_val == 2.0:
            result = coreai.sqrt(
                coreai.reduce_sum(coreai.broadcasting_mul(input, input), dims)
            )
        elif ord_val == float("inf"):
            result = coreai.reduce_max(coreai.abs_(input), dims)
        elif ord_val == float("-inf"):
            result = coreai.reduce_min(coreai.abs_(input), dims)
        else:
            # General p-norm: (sum(|x|^p))^(1/p)
            result = coreai.broadcasting_pow(
                coreai.reduce_sum(
                    coreai.broadcasting_pow(coreai.abs_(input), ord_val), dims
                ),
                1.0 / ord_val,
            )

        return result if keepdim else coreai.shrink_dims(result, dims)

    return linalg_vector_norm(x)


def _log_base_n(x: Value, n: int, loc: Location) -> Value:
    """Compute log base n: log_n(x) = ln(x) / ln(n)"""
    return coreai.broadcasting_divide(
        coreai.log(x),
        coreai.log(coreai.constant(n, dtype=x.type.element_type)),
    )


def replace_log10(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    return _log_base_n(_get_operand(values_map, node, 0), 10, loc)


def replace_log1p(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)
    return coreai.log(
        coreai.broadcasting_add(x, coreai.constant(1, dtype=x.type.element_type))
    )


def replace_log2(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    return _log_base_n(_get_operand(values_map, node, 0), 2, loc)


def replace_logical_not(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    if x.type.element_type != IntegerType.get_signless(1):
        x = coreai.broadcasting_not_equal(
            x, coreai.constant(0, dtype=x.type.element_type)
        )
    return coreai.not_(x)


def replace_masked_scatter(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Lowers aten.masked_scatter.default to:

        out_flat = where(mask, source[cumsum(mask) - 1], self)

    Out-of-range indices at False positions are harmless — where()
    discards them.
    """
    self_, mask, source = _get_operands(values_map, node, [0, 1, 2])

    self_flat = coreai.reshape(self_, [-1])
    if mask.type.shape != self_.type.shape:
        target_shape = (
            coreai.get_shape(self_)
            if any(d < 0 for d in self_.type.shape)
            else list(self_.type.shape)
        )
        mask = coreai.broadcast_to(mask, target_shape)
    mask_flat = coreai.reshape(mask, [-1])
    source_flat = coreai.reshape(source, [-1])

    if source_flat.type.element_type != self_flat.type.element_type:
        source_flat = coreai.cast(source_flat, self_flat.type.element_type)

    mask_int = coreai.cast(mask_flat, np.int32)
    idx = coreai.scan(mask_int, np.uint32(0), False, combiner="sum")
    idx = coreai.broadcasting_sub(idx, coreai.constant(1, dtype=np.int32))
    gathered = coreai.gather_along_axis(source_flat, idx, 0)

    out_flat = coreai.broadcasting_where(mask_flat, gathered, self_flat)
    out_shape = (
        coreai.get_shape(self_)
        if any(d < 0 for d in self_.type.shape)
        else list(self_.type.shape)
    )
    return coreai.reshape(out_flat, out_shape)


def replace_maxpool2d_with_indices(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> OpResultList:
    x = _get_operand(values_map, node, 0)
    args = node.args
    kernel_size = args[1]
    if isinstance(args[2], fx.Node):
        raise ValueError(
            f"Encountered dynamic stride at maxpool2d: node: {node}, name: {node.name}"
        )
    stride = args[2]
    padding = args[3] if len(args) >= 4 else [0, 0]
    dilation = args[4] if len(args) >= 5 else [1, 1]
    ceil_mode = args[5] if len(args) >= 6 else False

    # Perform padding manually — coreai.maxpool2d does not support padding
    x = coreai.pad(
        x,
        np.array(4 * [0] + 2 * [padding[0]] + 2 * [padding[1]], dtype=np.uint32),
        coreai.cast(np.float32("-inf"), x.type.element_type),
    )

    batch_size, channels, padded_height, padded_width = x.type.shape
    round_fn = np.ceil if ceil_mode else np.floor
    dyn = ShapedType.get_dynamic_size()
    output_hw = [
        dyn if sz < 0 else int(round_fn((sz - d * (k - 1) - 1) / s)) + 1
        for sz, k, d, s in zip(
            (padded_height, padded_width), kernel_size, dilation, stride
        )
    ]

    result = coreai.maxpool2d(
        output=RankedTensorType.get(
            [batch_size, channels, *output_hw], x.type.element_type
        ),
        input=x,
        kernel_size=np.array(kernel_size, np.uint32),
        stride=np.array(stride, np.uint32),
        dilation=np.array(dilation, np.uint32),
        ceil_mode=ceil_mode,
    )

    # coreai.maxpool2d does not return indices — return a dummy tensor matching
    # the output shape (same spatial dims as the pooled result), filled with zeros.
    dummy_zero = coreai.constant(np.zeros([1] * result.type.rank, dtype=np.int32))
    if any(d < 0 for d in result.type.shape):
        dummy_indices = coreai.BroadcastToOp(
            dummy_zero,
            coreai.get_shape(result),
            results=[get_tensor_type(node.meta["val"][1])],
        ).result
    else:
        dummy_indices = coreai.broadcast_to(dummy_zero, result.type.shape)
    return [result, dummy_indices]


def replace_mean_default(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Computes global mean across all dimensions, returning a scalar tensor."""
    x = _get_operand(values_map, node, 0)
    all_dims = list(range(x.type.rank))
    return coreai.shrink_dims(coreai.reduce_mean(x, all_dims), all_dims)


def replace_mean_dim(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Computes mean along specified dimensions."""
    x, axes = _get_operands(values_map, node, [0, 1])
    keepdim = len(node.args) >= 3 and bool(node.args[2])
    result = coreai.reduce_mean(x, axes)
    return result if keepdim else coreai.shrink_dims(result, axes)


def replace_max_default(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Computes global max across all dimensions, returning a scalar tensor."""
    x = _get_operand(values_map, node, 0)
    all_dims = list(range(x.type.rank))
    return coreai.shrink_dims(coreai.reduce_max(x, all_dims), all_dims)


def replace_max_dim(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> OpResultList:
    """Computes max along a specific dimension, returning (values, indices)."""
    x = _get_operand(values_map, node, 0)
    dim = node.args[1]
    keepdim = len(node.args) >= 3 and bool(node.args[2])
    dim = dim + x.type.rank if dim < 0 else dim

    max_values = coreai.reduce_max(x, [dim])
    argmax_indices = coreai.cast(coreai.argmax(x, dim), np.int32)

    if not keepdim:
        max_values = coreai.shrink_dims(max_values, [dim])
        argmax_indices = coreai.shrink_dims(argmax_indices, [dim])

    return [max_values, argmax_indices]


def replace_min_default(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Computes global min across all dimensions, returning a scalar tensor."""
    x = _get_operand(values_map, node, 0)
    all_dims = list(range(x.type.rank))
    return coreai.shrink_dims(coreai.reduce_min(x, all_dims), all_dims)


def replace_min_dim(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> OpResultList:
    """Computes min along a specific dimension, returning (values, indices).

    Argmin is computed as argmax(-x), since Core AI has argmax but not argmin.
    """
    x = _get_operand(values_map, node, 0)
    dim = node.args[1]
    keepdim = len(node.args) >= 3 and bool(node.args[2])
    dim = dim + x.type.rank if dim < 0 else dim

    min_values = coreai.reduce_min(x, [dim])
    argmin_indices = coreai.cast(
        coreai.argmax(
            coreai.broadcasting_mul(x, coreai.constant(-1, dtype=x.type.element_type)),
            dim,
        ),
        np.int32,
    )

    if not keepdim:
        min_values = coreai.shrink_dims(min_values, [dim])
        argmin_indices = coreai.shrink_dims(argmin_indices, [dim])

    return [min_values, argmin_indices]


def replace_mm(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    mat1, mat2 = _get_operands(values_map, node, [0, 1])
    assert mat1.type.rank == 2 and mat2.type.rank == 2, (
        f"mm requires 2D tensors, got mat1 rank: {mat1.type.rank}, mat2 rank: {mat2.type.rank}"
    )
    if mat1.type.element_type != mat2.type.element_type:
        mat2 = coreai.cast(mat2, mat1.type.element_type)
    return coreai.broadcasting_batch_matmul(mat1, mat2)


def replace_neg(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)
    return coreai.broadcasting_mul(x, coreai.constant(-1, dtype=x.type.element_type))


def replace_nonzero(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    result = coreai.non_zero(x)
    return coreai.cast(result, get_output_element_type_from_node(node))


def replace_nonzero_numpy(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> list[Value]:
    x = _get_operand(values_map, node, 0)
    nz = coreai.cast(
        coreai.non_zero(x), get_output_element_type_from_node(node, index=0)
    )
    return [
        coreai.shrink_dims(coreai.slice_(nz, [0, d], [INT32_MAX, d + 1], [1, 1]), [1])
        for d in range(x.type.rank)
    ]


def replace_permute(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    perm = to_uint32_perm(node.args[1], x.type.rank)
    return coreai.transpose(x, perm)


def replace_reciprocal(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    return coreai.broadcasting_divide(coreai.constant(1, dtype=x.type.element_type), x)


def replace_remainder(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    # decomposition: x - floor_divide(x, y) * y
    x, y = _get_operands(values_map, node, [0, 1])
    promoted_type = get_promoted_type(x.type, y.type)
    x = coreai.cast(x, promoted_type)
    y = coreai.cast(y, promoted_type)
    return coreai.broadcasting_sub(
        x, coreai.broadcasting_mul(coreai.broadcasting_floor_divide(x, y), y)
    )


def replace_repeat(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)
    repeat_args = list(node.args[1])
    extra_dims = len(repeat_args) - x.type.rank
    if extra_dims > 0:
        x = coreai.expand_dims(x, list(range(extra_dims)))

    if all(isinstance(r, int) for r in repeat_args):
        return coreai.tile(x, np.array(repeat_args, dtype=np.uint32))

    # At least one repeat is a SymInt fx.Node — build a rank-1 uint32 dim
    # vector at runtime, with per-axis constants for plain ints and the
    # resolved Value (cast to uint32, lifted to rank-1 if scalar) for
    # SymInts. coreai.tile accepts a runtime Value for its dims.
    chunks: list[Value] = []
    for r in repeat_args:
        if isinstance(r, int):
            chunks.append(coreai.constant([r], dtype=np.uint32))
        else:
            assert isinstance(r, fx.Node)
            v = coreai.cast(values_map[r.name], dtype=np.uint32)
            if v.type.rank == 0:
                v = coreai.reshape(v, [1])
            chunks.append(v)
    return coreai.tile(x, coreai.concat(0, chunks))


def replace_round_decimals(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """round(x, decimals=d) == round_(x * 10^d) / 10^d."""
    x = _get_operand(values_map, node, 0)
    scale = coreai.constant(10.0 ** node.kwargs["decimals"], dtype=x.type.element_type)
    return coreai.broadcasting_divide(
        coreai.round_(coreai.broadcasting_mul(x, scale)),
        scale,
    )


def replace_scatter(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Replaces aten.scatter.src, scatter.value, scatter.reduce, scatter.value_reduce.

    ATen scatter operation variants:
        scatter.src(input, dim, index, src) -> Tensor
        scatter.value(input, dim, index, value) -> Tensor
        scatter.reduce(input, dim, index, src, reduce) -> Tensor
        scatter.value_reduce(input, dim, index, value, reduce) -> Tensor

    Maps to coreai.scatter_along_axis with appropriate scatter_mode.
    """
    input_val = _get_operand(values_map, node, 0)
    dim = node.args[1]
    index = coreai.cast(_get_operand(values_map, node, 2), np.int32)
    dim = dim if dim >= 0 else dim + input_val.type.rank

    target = get_target(node)
    is_value = "value" in target
    is_reduce = "reduce" in target

    src = (
        coreai.broadcast_to(node.args[3], index.type.shape)
        if is_value
        else _get_operand(values_map, node, 3)
    )
    if src.type.element_type != input_val.type.element_type:
        src = coreai.cast(src, input_val.type.element_type)

    scatter_kwargs = {
        "output": input_val.type,
        "input": input_val,
        "indices": index,
        "updates": src,
        "axis": dim,
        "loc": loc,
    }
    if is_reduce:
        reduce_mode = str(node.kwargs.get("reduce", "add"))
        scatter_kwargs["scatter_mode"] = Attribute.parse(
            f"#coreai.scatter_mode<{'mul' if reduce_mode == 'multiply' else 'add'}>"
        )

    return coreai.scatter_along_axis(**scatter_kwargs)


def replace_scalar_tensor(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts a scalar value to a 0-dimensional tensor."""
    scalar = _get_operand(values_map, node, 0, loc)
    # Dynamic scalars (SymInt) arrive as rank-1 tensors from shape slices; squeeze to 0D.
    if isinstance(scalar.type, RankedTensorType) and scalar.type.rank > 0:
        scalar = coreai.shrink_dims(scalar, list(range(scalar.type.rank)))
    return coreai.cast(scalar, get_output_element_type_from_node(node))


def replace_slice(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    """Converts aten.slice.Tensor to coreai.slice_."""
    x = _get_operand(values_map, node, 0)
    args = node.args
    rank = x.type.rank
    dim = args[1] if len(args) > 1 else 0
    dim = dim + rank if dim < 0 else dim

    raw_start = args[2] if len(args) > 2 else None
    raw_end = args[3] if len(args) > 3 else None
    raw_stride = args[4] if len(args) > 4 else 1

    start_val = resolve_slice_arg(raw_start, 0, values_map)
    end_val = resolve_slice_arg(raw_end, INT32_MAX, values_map)
    stride_val = resolve_slice_arg(raw_stride, 1, values_map)

    start_indices = build_slice_index_array(rank, dim, 0, start_val)
    end_indices = build_slice_index_array(rank, dim, INT32_MAX, end_val)
    strides = build_slice_index_array(rank, dim, 1, stride_val)

    # coreai.slice_ drops static shape info on dims with dynamic bounds; use the
    # output type from torch.export (node.meta["val"]) which correctly preserves
    # static dims whose bounds are statically known.
    result_type = get_tensor_type(node.meta["val"])
    return coreai.SliceOp(
        x, start_indices, end_indices, strides, results=[result_type]
    ).result


def replace_slice_scatter(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.slice_scatter to coreai.slice_update."""
    input_val, src = _get_operands(values_map, node, [0, 1])
    args = node.args
    rank = input_val.type.rank

    dim = args[2] if len(args) > 2 and args[2] is not None else 0
    dim = dim + rank if dim < 0 else dim

    raw_start = args[3] if len(args) > 3 else None
    raw_end = args[4] if len(args) > 4 else None
    raw_stride = args[5] if len(args) > 5 else 1

    start_val = resolve_slice_arg(raw_start, 0, values_map)
    end_val = resolve_slice_arg(raw_end, INT32_MAX, values_map)
    stride_val = resolve_slice_arg(raw_stride, 1, values_map)

    start_indices = build_slice_index_array(rank, dim, 0, start_val)
    end_indices = build_slice_index_array(rank, dim, INT32_MAX, end_val)
    strides = build_slice_index_array(rank, dim, 1, stride_val)

    return coreai.slice_update(input_val, start_indices, end_indices, strides, src)


def replace_select_int(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.select.int to coreai.slice + dim removal.

    aten.select.int(input, dim, index) -> Tensor
    Selects a single element along a dimension, removing that dimension.
    """
    x = _get_operand(values_map, node, 0)
    dim, index = node.args[1], node.args[2]
    rank = x.type.rank
    dim = dim + rank if dim < 0 else dim

    if isinstance(index, torch.SymInt):
        raise ValueError(
            f"aten.select.int: symbolic int for index is not supported, got {index!r}"
        )

    if index < 0 and x.type.shape[dim] < 0:
        # Negative index with unknown dim size: resolve the actual index at runtime.
        dim_size = coreai.cast(
            coreai.slice_(coreai.get_shape(x), [dim], [dim + 1], [1]), dtype=np.int32
        )
        actual = coreai.broadcasting_add(dim_size, coreai.constant([index]))
        actual_p1 = coreai.broadcasting_add(actual, coreai.constant([1]))
        start_parts = [actual if i == dim else [0] for i in range(rank)]
        end_parts = [actual_p1 if i == dim else [INT32_MAX] for i in range(rank)]
        start = coreai.concat(0, start_parts) if rank > 1 else start_parts[0]
        end = coreai.concat(0, end_parts) if rank > 1 else end_parts[0]
    else:
        # Non-negative index, or negative index with statically-known dim size.
        index = index + x.type.shape[dim] if index < 0 else index
        start = [index if i == dim else 0 for i in range(rank)]
        end = [index + 1 if i == dim else INT32_MAX for i in range(rank)]

    # When start/end are compile-time Python lists (not dynamic Values),
    # the selected dim is exactly size 1. Use SliceOp directly with a
    # refined result type (same precedent as replace_slice_tensor) so
    # shrink_dims works even when the input dim was dynamic.
    if isinstance(start, list):
        out_shape = list(x.type.shape)
        out_shape[dim] = 1
        out_type = RankedTensorType.get(out_shape, x.type.element_type)
        sliced = coreai.SliceOp(
            x,
            coreai.constant(start),
            coreai.constant(end),
            coreai.constant([1] * rank),
            results=[out_type],
        ).result
    else:
        sliced = coreai.slice_(x, start, end, [1] * rank)

    if sliced.type.shape[dim] == 1:
        return coreai.shrink_dims(sliced, [dim])

    # Negative index with dynamic dim: dim size is unknown at compile time,
    # fall back to reshape-based dim removal.
    if rank == 1:
        return coreai.shrink_dims(coreai.reshape(sliced, coreai.constant([1])), [0])
    shape_t = coreai.get_shape(x)
    parts = [coreai.slice_(shape_t, [i], [i + 1], [1]) for i in range(rank) if i != dim]
    return coreai.reshape(
        sliced, coreai.concat(0, parts) if len(parts) > 1 else parts[0]
    )


def replace_sign(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    # sign(x) = cast(x > 0) - cast(x < 0)
    x = _get_operand(values_map, node, 0)
    zero = coreai.constant(0, dtype=x.type.element_type)
    e_type = x.type.element_type
    gt = coreai.cast(coreai.broadcasting_greater(x, zero), e_type)
    lt = coreai.cast(coreai.broadcasting_greater(zero, x), e_type)
    return coreai.broadcasting_sub(gt, lt)


def replace_isinf(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    """Converts aten.isinf to (x == +inf) | (x == -inf)."""
    x = _get_operand(values_map, node, 0)
    return coreai.broadcasting_or(
        coreai.broadcasting_equal(
            x, coreai.constant(float("inf"), dtype=x.type.element_type)
        ),
        coreai.broadcasting_equal(
            x, coreai.constant(float("-inf"), dtype=x.type.element_type)
        ),
    )


def replace_to_copy(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten._to_copy to identity or coreai.cast."""
    x = _get_operand(values_map, node, 0)
    target_element_type = get_output_element_type_from_node(node)
    if x.type.element_type == target_element_type:
        return x
    return coreai.cast(x, target_element_type)


def replace_to_dtype(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.to.dtype to identity or coreai.cast."""
    x = _get_operand(values_map, node, 0)
    target_element_type = get_output_element_type_from_node(node)
    if x.type.element_type == target_element_type:
        return x
    return coreai.cast(x, target_element_type)


def replace_softmax(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    dim = node.args[1]
    return coreai.softmax(x, dim + x.type.rank if dim < 0 else dim)


def replace_squeeze_dims(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.squeeze.dims to coreai.shrink_dims."""
    x = _get_operand(values_map, node, 0)
    return coreai.shrink_dims(
        x, [d + x.type.rank if d < 0 else d for d in node.args[1]]
    )


def replace_sum_dim_intlist(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.sum.dim_IntList to coreai.reduce_sum."""
    x = _get_operand(values_map, node, 0)
    args = node.args

    target_type = get_output_element_type_from_node(node)
    if x.type.element_type != target_type:
        x = coreai.cast(x, target_type)

    rank = x.type.rank
    raw_dims = args[1] if len(args) > 1 and args[1] is not None else None
    keepdim = args[2] if len(args) > 2 and args[2] is not None else False
    axes = (
        list(range(rank))
        if raw_dims is None or len(raw_dims) == 0
        else [d + rank if d < 0 else d for d in raw_dims]
    )

    result = coreai.reduce_sum(x, axes)
    return result if keepdim else coreai.shrink_dims(result, axes)


def replace_tile(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)
    dims = list(node.args[1])
    ndim, rank = len(dims), x.type.rank
    if ndim < rank:
        dims = [1] * (rank - ndim) + dims
    elif ndim > rank:
        x = coreai.expand_dims(x, list(range(ndim - rank)))
    dims = np.array(dims, dtype=np.uint32)
    return coreai.tile(x, dims)


def replace_topk(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> list[Value]:
    x = _get_operand(values_map, node, 0)
    k = node.args[1]
    dim = node.args[2] if len(node.args) > 2 else -1
    largest = node.args[3] if len(node.args) > 3 else True
    sorted_ = node.args[4] if len(node.args) > 4 else True
    rank = x.type.rank
    axis = dim % rank

    values = coreai.sort(x, axis, largest, sorted_)
    indices = coreai.argsort(x, axis, largest, sorted_)

    start = [0] * rank
    end = [k if i == axis else INT32_MAX for i in range(rank)]
    values = coreai.slice_(values, start, end, [1] * rank)
    indices = coreai.slice_(indices, start, end, [1] * rank)
    indices = coreai.cast(indices, np.int32)

    return [values, indices]


def replace_prod_default(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    target_type = get_output_element_type_from_node(node)
    if x.type.element_type != target_type:
        x = coreai.cast(x, target_type)
    all_dims = list(range(x.type.rank))
    return coreai.shrink_dims(coreai.reduce_product(x, all_dims), all_dims)


def replace_prod_dim_int(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    target_type = get_output_element_type_from_node(node)
    if x.type.element_type != target_type:
        x = coreai.cast(x, target_type)
    dim: int = node.args[1]
    keepdim = node.args[2] if len(node.args) > 2 and node.args[2] is not None else False
    axis = dim % x.type.rank
    result = coreai.reduce_product(x, [axis])
    return result if keepdim else coreai.shrink_dims(result, [axis])


def replace_log_softmax(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Numerically stable log_softmax: x - max(x) - log(sum(exp(x - max(x))))."""
    x = _get_operand(values_map, node, 0)
    dim = node.args[1]
    dim = dim + x.type.rank if dim < 0 else dim
    input_element_type = x.type.element_type

    input_names = ["input"]
    output_names = ["output"]
    op_attributes = {
        "axis": dim,
        "version": 1,
    }
    composite_decl = generate_composite_decl(
        x.context, "log_softmax", input_names, output_names, op_attributes
    )

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def log_softmax(input: Value) -> Value:
        x_compute, _, use_fp32 = prepare_compute_type_for_norm(
            input, input_element_type, loc
        )

        max_x = coreai.reduce_max(x_compute, [dim])
        x_shifted = coreai.broadcasting_sub(x_compute, max_x)
        exp_shifted = coreai.exp(x_shifted)
        sum_exp = coreai.reduce_sum(exp_shifted, [dim])
        log_sum_exp = coreai.log(sum_exp)
        result = coreai.broadcasting_sub(x_shifted, log_sum_exp)

        if use_fp32:
            result = coreai.cast(result, input_element_type)

        return result

    return log_softmax(x)


def replace_unary_ops(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    unary_ops = {
        "acos.default": coreai.acos,
        "acosh.default": coreai.acosh,
        "asin.default": coreai.asin,
        "asinh.default": coreai.asinh,
        "atan.default": coreai.atan,
        "atanh.default": coreai.atanh,
        "cos.default": coreai.cos,
        "cosh.default": coreai.cosh,
        "erf.default": coreai.erf,
        "exp.default": coreai.exp,
        "log.default": coreai.log,
        "relu.default": coreai.relu,
        "round.default": coreai.round_,
        "round": coreai.round_,
        "rsqrt.default": coreai.rsqrt,
        "sigmoid.default": coreai.sigmoid,
        "silu.default": coreai.silu,
        "sin.default": coreai.sin,
        "sinh.default": coreai.sinh,
        "sqrt.default": coreai.sqrt,
        "tan.default": coreai.tan,
        "tanh.default": coreai.tanh,
    }
    return unary_ops[get_target(node)](_get_operand(values_map, node, 0))


def replace_unsqueeze(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    return coreai.expand_dims(_get_operand(values_map, node, 0), [node.args[1]])


def replace_split_with_sizes(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> list[Value]:
    """Converts aten.split_with_sizes to coreai.split."""
    x = _get_operand(values_map, node, 0)
    dim = node.args[2] if len(node.args) > 2 else 0
    dim = dim + x.type.rank if dim < 0 else dim
    split_sizes = np.array(node.args[1], dtype=np.uint32)
    results = coreai.split(x, split_sizes, np.int32(dim))
    if isinstance(results, OpResultList):
        return list(results)
    return [results]


def replace_view(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x, shape = _get_operands(values_map, node, [0, 1])
    # for the dynamic input shape cases, coreai reshape op cannot infer the static dimension,
    # so we resort to the ground truth output shape from the fx node.
    result_type = get_tensor_type(node.meta["val"])
    # Cast to match result element type (reshape requires matching input/output types).
    if x.type.element_type != result_type.element_type:
        x = coreai.cast(x, result_type.element_type)
    return coreai.ReshapeOp(x, shape, results=[result_type]).result


def replace_where(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    condition, self_, other = _get_operands(values_map, node, [0, 1, 2])
    target_type = get_output_element_type_from_node(node)
    # where doesn't support fp64 inputs
    if target_type != F64Type.get():
        if self_.type.element_type != target_type:
            self_ = coreai.cast(self_, target_type)
        if other.type.element_type != target_type:
            other = coreai.cast(other, target_type)
    return coreai.broadcasting_where(condition, self_, other)


def replace_cumsum(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    """Converts aten.cumsum to coreai.scan with sum combiner."""
    x = _get_operand(values_map, node, 0)
    dim = node.args[1]
    target_type = get_output_element_type_from_node(node)
    if x.type.element_type != target_type:
        x = coreai.cast(x, target_type)
    dim = dim + x.type.rank if dim < 0 else dim
    return coreai.scan(x, np.uint32(dim), False, combiner="sum")


def replace_upsample_bilinear2d_vec(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.upsample_bilinear2d.vec to coreai.interpolate with linear mode.

    Either output_size or scale_factors must be provided. align_corners controls
    sampling: False (HalfPixel) or True (AlignCorners).
    """
    x = _get_operand(values_map, node, 0)
    if x.type.rank != 4:
        err = f"upsample_bilinear2d requires 4D input (NCHW), got rank {x.type.rank}"
        raise ValueError(err)

    args = node.args
    output_size = args[1] if len(args) > 1 and args[1] is not None else None
    align_corners = args[2] if len(args) > 2 and args[2] is not None else False
    scale_factors = args[3] if len(args) > 3 and args[3] is not None else None

    if (output_size is None) == (scale_factors is None):
        err = "Exactly one of output_size or scale_factors must be provided"
        raise ValueError(err)

    input_n, input_c, input_h, input_w = x.type.shape
    any_dynamic = any(d < 0 for d in x.type.shape)
    spatial_dynamic = input_h < 0 or input_w < 0

    if output_size is not None:
        # output_size elements can be static ints or dynamic fx.Nodes
        output_size_dynamic = any(isinstance(s, fx.Node) for s in output_size)
        if output_size_dynamic:
            out_h_val = values_map[output_size[0].name]
            out_w_val = values_map[output_size[1].name]
            output_shape_val = upsample_build_output_shape_dynamic(
                x, out_h_val, out_w_val
            )
            if align_corners:
                out_h_f32 = coreai.cast(out_h_val, np.float32)
                out_w_f32 = coreai.cast(out_w_val, np.float32)
                scale_val, offset_val = upsample_align_corners_scale_offset(
                    x, out_h_f32, out_w_f32
                )
            else:
                scale_val = upsample_halfpixel_scale(x, out_h_val, out_w_val)
                offset_val = [0.0, 0.0, 0.0, 0.0]
        elif any_dynamic:
            # Static output_size but dynamic input dims — use helpers
            out_h, out_w = int(output_size[0]), int(output_size[1])
            output_shape_val = upsample_build_output_shape_dynamic(x, out_h, out_w)
            if align_corners:
                out_h_f32 = coreai.constant(np.array([float(out_h)], dtype=np.float32))
                out_w_f32 = coreai.constant(np.array([float(out_w)], dtype=np.float32))
                scale_val, offset_val = upsample_align_corners_scale_offset(
                    x, out_h_f32, out_w_f32
                )
            else:
                scale_val = upsample_halfpixel_scale(x, out_h, out_w)
                offset_val = [0.0, 0.0, 0.0, 0.0]
        else:
            # Fully static — Python math
            out_h, out_w = int(output_size[0]), int(output_size[1])
            output_shape_val = [input_n, input_c, out_h, out_w]
            if align_corners:
                scale_h = (out_h - 1.0) / (input_h - 1.0) if input_h > 1 else 1.0
                scale_w = (out_w - 1.0) / (input_w - 1.0) if input_w > 1 else 1.0
                offset_h = 0.5 * (1.0 - scale_h)
                offset_w = 0.5 * (1.0 - scale_w)
                scale_val = [1.0, 1.0, scale_h, scale_w]
                offset_val = [0.0, 0.0, offset_h, offset_w]
            else:
                scale_val = [1.0, 1.0, out_h / input_h, out_w / input_w]
                offset_val = [0.0, 0.0, 0.0, 0.0]
    else:
        scale_h_f = float(scale_factors[0])
        scale_w_f = float(scale_factors[1])
        if spatial_dynamic:
            out_h_f32, out_w_f32, out_h_int, out_w_int = (
                upsample_runtime_output_hw_from_scale_dynamic(x, scale_h_f, scale_w_f)
            )
            output_shape_val = upsample_build_output_shape_dynamic(
                x, out_h_int, out_w_int
            )
            if align_corners:
                scale_val, offset_val = upsample_align_corners_scale_offset(
                    x, out_h_f32, out_w_f32
                )
            else:
                scale_val = [1.0, 1.0, scale_h_f, scale_w_f]
                offset_val = [0.0, 0.0, 0.0, 0.0]
        else:
            out_h = int(input_h * scale_h_f)
            out_w = int(input_w * scale_w_f)
            if any_dynamic:
                output_shape_val = upsample_build_output_shape_dynamic(x, out_h, out_w)
            else:
                output_shape_val = [input_n, input_c, out_h, out_w]
            if align_corners:
                scale_h = (out_h - 1.0) / (input_h - 1.0) if input_h > 1 else 1.0
                scale_w = (out_w - 1.0) / (input_w - 1.0) if input_w > 1 else 1.0
                offset_h = 0.5 * (1.0 - scale_h)
                offset_w = 0.5 * (1.0 - scale_w)
                scale_val = [1.0, 1.0, scale_h, scale_w]
                offset_val = [0.0, 0.0, offset_h, offset_w]
            else:
                scale_val = [1.0, 1.0, scale_h_f, scale_w_f]
                offset_val = [0.0, 0.0, 0.0, 0.0]

    return coreai.interpolate(
        x,
        output_shape_val,
        scale_val,
        offset_val,
        interpolation_mode=Attribute.parse("#coreai.interpolation_mode<linear>"),
    )


def replace_upsample_nearest2d_vec(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    """Converts aten.upsample_nearest2d.vec to coreai.interpolate with nearest_neighbor mode.
    Either output_size or scale_factors must be provided.
    """
    x = _get_operand(values_map, node, 0)
    if x.type.rank != 4:
        err = f"upsample_nearest2d requires 4D input (NCHW), got rank {x.type.rank}"
        raise ValueError(err)

    args = node.args
    output_size = args[1] if len(args) > 1 and args[1] is not None else None
    scale_factors = args[2] if len(args) > 2 and args[2] is not None else None

    if (output_size is None) == (scale_factors is None):
        raise ValueError("Exactly one of output_size or scale_factors must be provided")

    input_n, input_c, input_h, input_w = x.type.shape
    any_dynamic = any(d < 0 for d in x.type.shape)
    spatial_dynamic = input_h < 0 or input_w < 0

    if output_size is not None:
        # output_size elements can be static ints or dynamic fx.Nodes
        output_size_dynamic = any(isinstance(s, fx.Node) for s in output_size)
        if output_size_dynamic:
            out_h_val = values_map[output_size[0].name]
            out_w_val = values_map[output_size[1].name]
            output_shape_val = upsample_build_output_shape_dynamic(
                x, out_h_val, out_w_val
            )
            scale_val = upsample_halfpixel_scale(x, out_h_val, out_w_val)
        elif any_dynamic:
            # Static output_size but dynamic input dims — use helpers
            out_h, out_w = int(output_size[0]), int(output_size[1])
            output_shape_val = upsample_build_output_shape_dynamic(x, out_h, out_w)
            scale_val = upsample_halfpixel_scale(x, out_h, out_w)
        else:
            # Fully static — Python math
            out_h, out_w = int(output_size[0]), int(output_size[1])
            output_shape_val = [input_n, input_c, out_h, out_w]
            scale_val = [1.0, 1.0, out_h / input_h, out_w / input_w]
    else:
        scale_h_f = float(scale_factors[0])
        scale_w_f = float(scale_factors[1])
        if spatial_dynamic:
            _, _, out_h_int, out_w_int = upsample_runtime_output_hw_from_scale_dynamic(
                x, scale_h_f, scale_w_f
            )
            output_shape_val = upsample_build_output_shape_dynamic(
                x, out_h_int, out_w_int
            )
        else:
            out_h = int(input_h * scale_h_f)
            out_w = int(input_w * scale_w_f)
            if any_dynamic:
                output_shape_val = upsample_build_output_shape_dynamic(x, out_h, out_w)
            else:
                output_shape_val = [input_n, input_c, out_h, out_w]
        scale_val = [1.0, 1.0, scale_h_f, scale_w_f]

    return coreai.interpolate(
        x,
        output_shape_val,
        scale_val,
        [0.0, 0.0, 0.0, 0.0],
        interpolation_mode=Attribute.parse(
            "#coreai.interpolation_mode<nearest_neighbor>"
        ),
    )


def replace_pixel_shuffle(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    r = node.args[1]

    input_names = ["input"]
    output_names = ["output"]
    op_attributes = {"upscale_factor": r, "version": 1}
    composite_decl = generate_composite_decl(
        x.context, "pixel_shuffle", input_names, output_names, op_attributes
    )

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def pixel_shuffle(input: Value) -> Value:
        split_channels_shape = input.type.shape
        split_channels_shape.insert(-2, r)
        split_channels_shape.insert(-2, r)
        split_channels_shape[-5] = split_channels_shape[-5] // r**2

        shp = coreai.cast(coreai.get_shape(input), dtype=np.int32)
        lhs = coreai.slice_(shp, (0,), (input.type.rank - 3,), (1,))
        rhs = coreai.slice_(shp, (input.type.rank - 2,), (input.type.rank,), (1,))

        new_lhs = coreai.concat(0, [lhs, (input.type.shape[-3] // r**2, r, r)])
        split_channels_shape_val = coreai.concat(0, [new_lhs, rhs])
        split_channels = coreai.ReshapeOp(
            input,
            split_channels_shape_val,
            results=[
                RankedTensorType.get(split_channels_shape, input.type.element_type)
            ],
        ).result

        permute_dims = list(range(split_channels.type.rank))
        permute_dims[-4] = len(permute_dims) - 2
        permute_dims[-3] = len(permute_dims) - 4
        permute_dims[-2] = len(permute_dims) - 1
        permute_dims[-1] = len(permute_dims) - 3
        permuted = coreai.transpose(
            split_channels,
            to_uint32_perm(permute_dims, split_channels.type.rank),
        )

        final_shape = input.type.shape
        final_shape[-3] = final_shape[-3] // r**2

        for i in range(-2, 0):
            if final_shape[i] < 0:
                continue
            final_shape[i] = final_shape[i] * r

        lhs = coreai.slice_(
            coreai.cast(coreai.get_shape(permuted), dtype=np.int32),
            (0,),
            (permuted.type.rank - 4,),
            (1,),
        )
        rhs = coreai.slice_(shp, (input.type.rank - 2,), (input.type.rank,), (1,))

        rhs = coreai.broadcasting_mul(rhs, coreai.constant(r))
        final_shape_value = coreai.concat(0, [lhs, rhs])
        final = coreai.ReshapeOp(
            permuted,
            final_shape_value,
            results=[
                RankedTensorType.get(final_shape, element_type=input.type.element_type)
            ],
        ).result
        return final

    return pixel_shuffle(x)


def replace_view_as_real(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    real = coreai.real_part(x)
    imag = coreai.imaginary_part(x)
    rank = real.type.rank
    real_expanded = coreai.expand_dims(real, [rank])
    imag_expanded = coreai.expand_dims(imag, [rank])
    return coreai.concat(rank, [real_expanded, imag_expanded])


def replace_view_as_complex(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    x = _get_operand(values_map, node, 0)
    rank = x.type.rank
    start = [0] * rank
    end_all = [INT32_MAX] * rank

    end_real = list(end_all)
    end_real[-1] = 1
    real_sliced = coreai.slice_(x, start, end_real, [1] * rank)

    start_imag = list(start)
    start_imag[-1] = 1
    imag_sliced = coreai.slice_(x, start_imag, end_all, [1] * rank)

    real = coreai.shrink_dims(real_sliced, [rank - 1])
    imag = coreai.shrink_dims(imag_sliced, [rank - 1])
    return coreai.create_complex(real, imag)


def replace_instance_norm(
    values_map: dict[str, Value], node: fx.Node, loc: Location
) -> Value:
    use_input_stats = node.args[5]

    x = _get_operand(values_map, node, 0)
    weight = _get_operand(values_map, node, 1) if node.args[1] is not None else None
    bias = _get_operand(values_map, node, 2) if node.args[2] is not None else None
    mean = _get_operand(values_map, node, 3) if node.args[3] is not None else None
    var = _get_operand(values_map, node, 4) if node.args[4] is not None else None
    use_input_stats = node.args[5]

    num_features = x.type.shape[1]
    element_type = x.type.element_type

    n_spatial = x.type.rank - 2

    param_shape = [num_features, *(1,) * n_spatial]

    if weight is None:
        weight = coreai.constant([1.0] * num_features, dtype=element_type)
    weight = coreai.reshape(weight, param_shape)

    if bias is None:
        bias = coreai.constant([0.0] * num_features, dtype=element_type)
    bias = coreai.reshape(bias, param_shape)

    # If running mean/var are provided as input then we can just inline this
    if not use_input_stats:
        assert mean is not None and var is not None

        mean = coreai.reshape(mean, param_shape)
        var = coreai.reshape(var, param_shape)

        numerator = coreai.broadcasting_sub(x, mean)

        eps = coreai.constant(node.args[7], dtype=element_type)
        denominator = coreai.broadcasting_add(var, eps)
        denominator = coreai.rsqrt(denominator)

        norm = coreai.broadcasting_mul(numerator, denominator)
        norm = coreai.broadcasting_mul(norm, weight)
        res = coreai.broadcasting_add(norm, bias)
        return res

    input_names = ["input", "gamma", "beta"]
    output_names = ["output"]
    op_attributes = {"eps": node.args[7], "version": 1}

    composite_decl = generate_composite_decl(
        x.context, "instance_norm", input_names, output_names, op_attributes
    )

    @coreai.graph(composite_decl=composite_decl, private=True, no_inline=True)
    def instance_norm(input: Value, gamma: Value, beta: Value) -> Value:
        input, _, use_fp32_stats = prepare_compute_type_for_norm(
            input, element_type, loc
        )
        reduction_dims = list(range(input.type.rank))[2:]
        mean = coreai.reduce_mean(input, reduction_dims)
        numerator = coreai.broadcasting_sub(input, mean)
        var = coreai.reduce_mean(
            coreai.broadcasting_mul(numerator, numerator), reduction_dims
        )

        eps = coreai.constant(node.args[7], dtype=var.type.element_type)
        denominator = coreai.rsqrt(coreai.broadcasting_add(var, eps))

        norm = coreai.broadcasting_mul(numerator, denominator)
        if use_fp32_stats:
            norm = coreai.cast(norm, element_type)

        norm = coreai.broadcasting_mul(norm, gamma)
        res = coreai.broadcasting_add(norm, beta)
        return res

    return instance_norm(x, weight, bias)


def replace_cond(
    values_map: dict[str, Value],
    node: fx.Node,
    *,
    graph_module: fx.GraphModule,
) -> list[Value]:
    condition_node, true_graph_node, false_graph_node, branch_operands = node.args

    condition = values_map[condition_node.name]
    if hasattr(condition.type, "rank") and condition.type.rank > 0:
        condition = coreai.shrink_dims(condition, list(range(condition.type.rank)))

    output_meta = node.meta["val"]
    result_types = [
        get_tensor_type(t)
        for t in (
            output_meta if isinstance(output_meta, (tuple, list)) else [output_meta]
        )
    ]

    operand_values = [
        values_map[op.name] for op in branch_operands if isinstance(op, fx.Node)
    ]

    if_op: coreai.IfOutputType = coreai.if_(results=result_types, condition=condition)
    if_result, [then_builder, else_builder] = if_op

    for builder, branch_mod in [
        (then_builder, getattr(graph_module, true_graph_node.target)),
        (else_builder, getattr(graph_module, false_graph_node.target)),
    ]:
        with builder:
            coreai.yield_(
                convert_branch_subgraph(
                    branch_mod,
                    operand_values,
                    graph_module,
                    _aten_to_core_resolver,
                    {"cond": replace_cond},
                )
            )

    return [if_result] if isinstance(if_result, Value) else list(if_result)


def replace_while_loop(
    values_map: dict[str, Value],
    node: fx.Node,
    *,
    graph_module: fx.GraphModule,
) -> list[Value]:
    cond_fn_node, body_fn_node, carried_inputs, additional_inputs = node.args

    carried_values = [
        values_map[op.name] for op in carried_inputs if isinstance(op, fx.Node)
    ]
    additional_values = [
        values_map[op.name] for op in additional_inputs if isinstance(op, fx.Node)
    ]

    output_meta = node.meta["val"]
    result_types = [
        get_tensor_type(t)
        for t in (
            output_meta if isinstance(output_meta, (tuple, list)) else [output_meta]
        )
    ]

    while_result, [before_builder, after_builder] = coreai.while_(
        results=result_types,
        inits=carried_values,
    )

    cond_module = getattr(graph_module, cond_fn_node.target)
    body_module = getattr(graph_module, body_fn_node.target)

    with before_builder:
        before_args = list(before_builder.arguments)
        cond_val = convert_branch_subgraph(
            cond_module,
            before_args + additional_values,
            graph_module,
            _aten_to_core_resolver,
            _higher_order_resolver,
        )[0]
        if hasattr(cond_val.type, "rank") and cond_val.type.rank > 0:
            cond_val = coreai.shrink_dims(cond_val, list(range(cond_val.type.rank)))
        coreai.condition(cond_val, *before_args)

    with after_builder:
        after_args = list(after_builder.arguments)
        coreai.yield_(
            convert_branch_subgraph(
                body_module,
                after_args + additional_values,
                graph_module,
                _aten_to_core_resolver,
                _higher_order_resolver,
            )
        )

    return [while_result] if isinstance(while_result, Value) else list(while_result)


def replace_trunc(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    x = _get_operand(values_map, node, 0)

    zero = coreai.constant(0, dtype=x.type.element_type)
    sign_q = coreai.broadcasting_sub(
        coreai.cast(coreai.broadcasting_greater(x, zero), x.type.element_type),
        coreai.cast(coreai.broadcasting_greater(zero, x), x.type.element_type),
    )

    return coreai.broadcasting_mul(
        sign_q,
        coreai.broadcasting_floor_divide(
            coreai.abs_(x), coreai.constant(1, x.type.element_type)
        ),
    )


def replace_yield(
    values_map: dict[str, Value],
    node: fx.Node,
    *,
    graph_module: fx.GraphModule,
) -> list[Value]:
    if len(node.users) != 0:
        raise ValueError(f"expect 0 results for _yield op, got {len(node.users)}")
    operands = [values_map[arg.name] for arg in node.args if isinstance(arg, fx.Node)]
    coreai.yield_(operands)
    return []


def replace_sdpa(values_map: dict[str, Value], node: fx.Node, loc: Location) -> Value:
    """Converts aten.scaled_dot_product_attention to a Core AI composite op.

    Decomposition (inside composite):
        1. (GQA) repeat-interleave key/value heads to match query head count
        2. scaled_query = query * scale
        3. key^T = transpose(key, [0, 1, 3, 2])
        4. attn_scores = matmul(scaled_query, key^T)
        5. (mask) attn_scores += float_mask
        6. attn_weights = softmax(attn_scores, dim=-1)
        7. output = matmul(attn_weights, value)
    """

    args = node.args
    kwargs = node.kwargs

    query = _get_operand(values_map, node, 0)
    key = _get_operand(values_map, node, 1)
    value = _get_operand(values_map, node, 2)

    # attn_mask: optional tensor (fx.Node) or None
    attn_mask_arg = args[3] if len(args) > 3 else kwargs.get("attn_mask", None)
    attn_mask: Value | None = (
        values_map[attn_mask_arg.name] if isinstance(attn_mask_arg, fx.Node) else None
    )

    # Scalar/bool args — read as Python values, not Core AI Values.
    is_causal = (
        bool(args[5])
        if len(args) > 5 and args[5] is not None
        else bool(kwargs.get("is_causal", False))
    )
    scale_raw = (args[6] if len(args) > 6 else None) or kwargs.get("scale", None)
    scale: float | None = float(scale_raw) if scale_raw is not None else None
    enable_gqa = (
        bool(args[7])
        if len(args) > 7 and args[7] is not None
        else bool(kwargs.get("enable_gqa", False))
    )

    ele_type = query.type.element_type
    query_rank = query.type.rank

    if query_rank < 3:
        raise ValueError(f"SDPA expects query rank >= 3, got {query_rank}")

    original_query = query

    # PyTorch SDPA accepts rank-3 (B, S, E) inputs with an implicit single
    # head. Unsqueeze a singleton head dim at position 1 to reuse the rank-4
    # lowering; result is squeezed back at the end.
    if query_rank == 3:
        query = coreai.expand_dims(query, [1])
        key = coreai.expand_dims(key, [1])
        value = coreai.expand_dims(value, [1])

    # Flatten rank > 4 inputs to rank-4 for the lowering.
    if query_rank > 4:
        query = _sdpa_flatten_leading_batch_dims(query)
        key = _sdpa_flatten_leading_batch_dims(key)
        value = _sdpa_flatten_leading_batch_dims(value)

    # is_causal: build the causal float mask outside the composite and pass it
    # as attn_mask. The composite interface is always mask-based.
    if is_causal:
        if attn_mask is not None:
            raise ValueError(
                "scaled_dot_product_attention: attn_mask and is_causal=True cannot both be set"
            )
        attn_mask = _sdpa_build_causal_mask(query, key, ele_type)

    if attn_mask is not None and attn_mask.type.rank > 4:
        attn_mask = _sdpa_flatten_leading_batch_dims(attn_mask)

    # Build composite inputs. Scale is NOT an input — it is embedded in
    # op_attributes when a compile-time constant, otherwise defaulted to
    # 1/sqrt(head_dim) inside the decomposition body.
    input_names = ["query", "key", "value"]
    if attn_mask is not None:
        input_names.append("attn_mask")

    op_attributes: dict[str, Any] = {"is_causal": False, "window_size": 0, "version": 1}
    if scale is not None:
        op_attributes["scale"] = scale

    composite_decl = generate_composite_decl(
        query.context,
        "scaled_dot_product_attention",
        input_names,
        ["output"],
        op_attributes,
    )

    # Capture shape/type info for the composite body closure.
    q_shape = query.type.shape
    k_shape = key.type.shape
    v_shape = value.type.shape

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def sdpa(q: Value, k: Value, v: Value, m: Value) -> Value:
        return _sdpa_decompose(
            q, k, v, m, scale, enable_gqa, ele_type, q_shape, k_shape, v_shape
        )

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def sdpa_maskless(q: Value, k: Value, v: Value) -> Value:
        return _sdpa_decompose(
            q, k, v, None, scale, enable_gqa, ele_type, q_shape, k_shape, v_shape
        )

    result = (
        sdpa(query, key, value, attn_mask)
        if attn_mask is not None
        else sdpa_maskless(query, key, value)
    )[0]

    # Restore original leading batch dims if inputs were rank > 4.
    if query_rank == 4:
        assert result.type == original_query.type, (
            "Result type and original query type must be identical"
        )
        return result
    if query_rank == 3:
        result = coreai.shrink_dims(result, [1])
        assert result.type == original_query.type, (
            "Result type and original query type must be identical"
        )
        return result
    orig_shape = coreai.get_shape(original_query)
    result = coreai.reshape(result, orig_shape)
    assert result.type == original_query.type, (
        "Result type and original query type must be identical"
    )
    return result


_aten_to_core_resolver: dict[str, Callable[..., Any]] = {
    "_local_scalar_dense.default": replace_local_scalar_dense,
    "_log_softmax.default": replace_log_softmax,
    "_native_batch_norm_legit_no_training.default": replace_batch_norm,
    "_softmax.default": replace_softmax,
    "_to_copy.default": replace_to_copy,
    "abs.default": replace_abs,
    "acos.default": replace_unary_ops,
    "acosh.default": replace_unary_ops,
    "add.Scalar": replace_binary_ops,
    "add.Tensor": replace_binary_ops,
    "add": replace_binary_ops,
    "addmm.default": replace_addmm,
    "alias.default": replace_alias,
    "amax.default": replace_amax_default,
    "amin.default": replace_amin_default,
    "any.default": replace_any_default,
    "any.dim": replace_any_dim,
    "any.dims": replace_any_dims,
    "arange.start_step": replace_arange_start_step,
    "argmax.default": replace_argmax,
    "asin.default": replace_unary_ops,
    "asinh.default": replace_unary_ops,
    "atan.default": replace_unary_ops,
    "atan2.default": replace_atan2,
    "atanh.default": replace_unary_ops,
    "_adaptive_avg_pool2d.default": replace_adaptive_avg_pool2d,
    "_unsafe_view.default": replace_view,
    "avg_pool2d.default": replace_avg_pool2d,
    "avg_pool3d.default": replace_avg_pool3d,
    "bitwise_and.Tensor": replace_binary_bitwise_ops,
    "bitwise_not.default": replace_bitwise_not,
    "bitwise_or.Tensor": replace_binary_bitwise_ops,
    "bitwise_xor.Tensor": replace_binary_bitwise_ops,
    "bmm.default": replace_bmm,
    "cat.default": replace_cat,
    "ceil.default": replace_ceil,
    "ceil": replace_ceil,
    "clamp.Tensor": replace_clamp,
    "clamp.default": replace_clamp,
    "clone.default": replace_clone,
    "complex.default": replace_complex,
    "constant_pad_nd.default": replace_constant_pad_nd,
    "convolution.default": replace_conv,
    "copy.default": replace_copy,
    "cos.default": replace_unary_ops,
    "cosh.default": replace_unary_ops,
    "cumsum.default": replace_cumsum,
    "div.Scalar": replace_binary_ops,
    "div.Tensor": replace_binary_ops,
    "div.Tensor_mode": replace_div_tensor_mode,
    "embedding.default": replace_embedding,
    "empty.default": replace_empty,
    "empty.memory_format": replace_empty,
    "eq.Scalar": replace_binary_comparision_ops,
    "eq.Tensor": replace_binary_comparision_ops,
    "erf.default": replace_unary_ops,
    "exp.default": replace_unary_ops,
    "exp2.default": replace_exp2,
    "expand.default": replace_expand,
    "expm1.default": replace_expm1,
    "flip.default": replace_flip,
    "floor.default": replace_floor,
    "floor_divide.default": replace_floor_divide,
    "floordiv.Scalar": replace_binary_ops,
    "floordiv.Tensor": replace_binary_ops,
    "floordiv": replace_binary_ops,
    "fmod.Scalar": replace_binary_ops,
    "fmod.Tensor": replace_binary_ops,
    "full.default": replace_full,
    "full_like.default": replace_full_like,
    "ge.Scalar": replace_binary_comparision_ops,
    "ge.Tensor": replace_binary_comparision_ops,
    "gelu.default": replace_gelu,
    "gather.default": replace_gather,
    "getitem": replace_getitem,
    "gt.Scalar": replace_binary_comparision_ops,
    "gt.Tensor": replace_binary_comparision_ops,
    "hardtanh.default": replace_hard_tanh,
    "hardsigmoid.default": replace_hardsigmoid,
    "hardswish.default": replace_hardswish,
    "index_put.default": replace_index_put,
    "index_select.default": replace_index_select,
    "index.Tensor": replace_index_tensor,
    "isinf.default": replace_isinf,
    "instance_norm.default": replace_instance_norm,
    "le.Scalar": replace_binary_comparision_ops,
    "le.Tensor": replace_binary_comparision_ops,
    "leaky_relu.default": replace_leaky_relu,
    "lift_fresh_copy.default": replace_lift_fresh_copy,
    "linalg_vector_norm.default": replace_linalg_vector_norm,
    "log.default": replace_unary_ops,
    "log10.default": replace_log10,
    "log1p.default": replace_log1p,
    "log2.default": replace_log2,
    "logical_and.default": replace_binary_logical_ops,
    "logical_not.default": replace_logical_not,
    "logical_or.default": replace_binary_logical_ops,
    "logical_xor.default": replace_binary_logical_ops,
    "lt.Scalar": replace_binary_comparision_ops,
    "lt.Tensor": replace_binary_comparision_ops,
    "max_pool2d_with_indices.default": replace_maxpool2d_with_indices,
    "max.default": replace_max_default,
    "max.dim": replace_max_dim,
    "maximum.default": replace_binary_ops,
    "masked_scatter.default": replace_masked_scatter,
    "mean.default": replace_mean_default,
    "mean.dim": replace_mean_dim,
    "min.default": replace_min_default,
    "min.dim": replace_min_dim,
    "minimum.default": replace_binary_ops,
    "mm.default": replace_mm,
    "mod.Scalar": replace_binary_ops,
    "mod.Tensor": replace_binary_ops,
    "mod": replace_binary_ops,
    "mul": replace_binary_ops,
    "mul.Scalar": replace_binary_ops,
    "mul.Tensor": replace_binary_ops,
    "native_group_norm.default": replace_group_norm,
    "native_layer_norm.default": replace_layer_norm,
    "ne.Scalar": replace_binary_comparision_ops,
    "ne.Tensor": replace_binary_comparision_ops,
    "neg.default": replace_neg,
    "neg": replace_neg,
    "nonzero.default": replace_nonzero,
    "nonzero_numpy.default": replace_nonzero_numpy,
    "permute.default": replace_permute,
    "pixel_shuffle.default": replace_pixel_shuffle,
    "polar.default": replace_polar,
    "pow.Scalar": replace_binary_ops,
    "pow.Tensor_Scalar": replace_binary_ops,
    "pow.Tensor_Tensor": replace_binary_ops,
    "pow": replace_binary_ops,
    "prod.default": replace_prod_default,
    "prod.dim_int": replace_prod_dim_int,
    "reciprocal.default": replace_reciprocal,
    "reflection_pad1d.default": replace_reflection_pad,
    "reflection_pad2d.default": replace_reflection_pad,
    "reflection_pad3d.default": replace_reflection_pad,
    "relu.default": replace_unary_ops,
    "remainder.Tensor": replace_remainder,
    "replication_pad1d.default": replace_replication_pad,
    "replication_pad2d.default": replace_replication_pad,
    "replication_pad3d.default": replace_replication_pad,
    "round.default": replace_unary_ops,
    "round.decimals": replace_round_decimals,
    "round": replace_unary_ops,
    "repeat.default": replace_repeat,
    "rsqrt.default": replace_unary_ops,
    "scaled_dot_product_attention.default": replace_sdpa,
    "scalar_tensor.default": replace_scalar_tensor,
    "scatter.reduce": replace_scatter,
    "scatter.src": replace_scatter,
    "scatter.value": replace_scatter,
    "scatter.value_reduce": replace_scatter,
    "select.int": replace_select_int,
    "sigmoid.default": replace_unary_ops,
    "silu.default": replace_unary_ops,
    "sign.default": replace_sign,
    "sin.default": replace_unary_ops,
    "sinh.default": replace_unary_ops,
    "slice.Tensor": replace_slice,
    "slice_scatter.default": replace_slice_scatter,
    "split_with_sizes.default": replace_split_with_sizes,
    "squeeze.dims": replace_squeeze_dims,
    "sqrt.default": replace_unary_ops,
    "sub.Scalar": replace_binary_ops,
    "sub.Tensor": replace_binary_ops,
    "sub": replace_binary_ops,
    "sum.dim_IntList": replace_sum_dim_intlist,
    "sym_size.int": replace_sym_size_int,
    "sym_min": replace_sym_min,
    "sym_float": replace_sym_float,
    "tan.default": replace_unary_ops,
    "tanh.default": replace_unary_ops,
    "tile.default": replace_tile,
    "truediv": replace_truediv,
    "to.dtype": replace_to_dtype,
    "topk.default": replace_topk,
    "true_divide.Tensor": replace_binary_ops,
    "trunc.default": replace_trunc,
    "trunc": replace_trunc,
    "unsqueeze.default": replace_unsqueeze,
    "upsample_bilinear2d.vec": replace_upsample_bilinear2d_vec,
    "upsample_nearest2d.vec": replace_upsample_nearest2d_vec,
    "view.default": replace_view,
    "view_as_complex.default": replace_view_as_complex,
    "view_as_complex_copy.default": replace_view_as_complex,
    "view_as_real.default": replace_view_as_real,
    "view_as_real_copy.default": replace_view_as_real,
    "where.self": replace_where,
}


_higher_order_resolver: dict[str, Callable[..., Any]] = {
    "_yield": replace_yield,
    "cond": replace_cond,
    "while_loop": replace_while_loop,
}
