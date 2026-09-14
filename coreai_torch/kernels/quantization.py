# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""High-performance Metal Shading Language (MSL) sub-byte quantization kernels.

Optimized for Apple Silicon Unified Memory Architecture (UMA) to eliminate
intermediate round-trips and memory overhead when converting between FP32/FP16
and 4-bit affine quantized tensors.
"""

from __future__ import annotations

import math

import torch
from coreai.authoring import MetalParameter

from coreai_torch._torch_metal_kernel import TorchMetalKernel

# ---------------------------------------------------------------------------
# Metal Shading Language (MSL) Kernel Bodies
# ---------------------------------------------------------------------------

# Fused INT4 -> FP16/FP32 Affine Dequantization in hardware registers.
# Reads 2 int4 values per packed uint8 byte, applies branchless sign-extension,
# performs fused affine multiply-accumulate, and writes out directly.
FUSED_DEQUANTIZE_INT4_MSL = """
    // Guard against out-of-bounds dispatch
    uint total_unpacked = output.get_extent(0);
    uint byte_idx = id;
    uint out_idx = byte_idx * 2;
    if (out_idx >= total_unpacked) return;

    // Read packed byte containing two 4-bit nibbles (low nibble = out[2k], high nibble = out[2k+1])
    uint8_t packed_val = packed_data[byte_idx];

    // Branchless 4-bit signed integer sign-extension from [-8, 7]
    int8_t nibble0 = static_cast<int8_t>(packed_val & 0x0F);
    nibble0 = static_cast<int8_t>((nibble0 ^ 0x08) - 0x08);

    int8_t nibble1 = static_cast<int8_t>((packed_val >> 4) & 0x0F);
    nibble1 = static_cast<int8_t>((nibble1 ^ 0x08) - 0x08);

    // Fetch scale and zero-point
    TYPE s = scale[0];
    TYPE zp = zero_point[0];

    // Fused affine dequantization in GPU registers: output = (val - zero_point) * scale
    output[out_idx] = (static_cast<TYPE>(nibble0) - zp) * s;
    if (out_idx + 1 < total_unpacked) {
        output[out_idx + 1] = (static_cast<TYPE>(nibble1) - zp) * s;
    }
"""

# Fused FP32/FP16 -> INT4 Affine Quantization.
# Clamps values to [-8, 7], rounds to nearest integer, and bit-packs pairs of
# 4-bit nibbles into a single uint8 byte in a single kernel launch.
FUSED_QUANTIZE_INT4_MSL = """
    uint total_unpacked = input.get_extent(0);
    uint byte_idx = id;
    uint in_idx = byte_idx * 2;
    if (in_idx >= total_unpacked) return;

    TYPE s = scale[0];
    TYPE zp = zero_point[0];

    // Quantize first element
    float v0 = static_cast<float>(input[in_idx]);
    int q0 = static_cast<int>(round(v0 / static_cast<float>(s)) + static_cast<float>(zp));
    q0 = max(-8, min(7, q0));
    uint8_t u0 = static_cast<uint8_t>(q0 & 0x0F);

    // Quantize second element (if within bounds)
    uint8_t u1 = 0;
    if (in_idx + 1 < total_unpacked) {
        float v1 = static_cast<float>(input[in_idx + 1]);
        int q1 = static_cast<int>(round(v1 / static_cast<float>(s)) + static_cast<float>(zp));
        q1 = max(-8, min(7, q1));
        u1 = static_cast<uint8_t>(q1 & 0x0F);
    }

    // Bit-pack nibbles: low nibble is element 0, high nibble is element 1
    packed_output[byte_idx] = static_cast<uint8_t>((u1 << 4) | u0);
"""


# ---------------------------------------------------------------------------
# PyTorch Reference Implementations for Shape Inference & Validation
# ---------------------------------------------------------------------------


def _ref_dequantize_int4(
    packed_data: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Reference eager implementation of int4 affine dequantization."""
    # Each byte unpacks to 2 elements
    num_packed = packed_data.numel()
    flat_packed = packed_data.reshape(-1)

    low_nibble = (flat_packed & 0x0F).to(torch.int8)
    high_nibble = ((flat_packed >> 4) & 0x0F).to(torch.int8)

    # Sign extend from 4-bit [-8, 7]
    low_int4 = torch.where(low_nibble > 7, low_nibble - 16, low_nibble)
    high_int4 = torch.where(high_nibble > 7, high_nibble - 16, high_nibble)

    interleaved = torch.empty(
        num_packed * 2, dtype=scale.dtype, device=packed_data.device
    )
    interleaved[0::2] = (low_int4.to(scale.dtype) - zero_point) * scale
    interleaved[1::2] = (high_int4.to(scale.dtype) - zero_point) * scale
    return interleaved


def _ref_quantize_int4(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Reference eager implementation of int4 affine quantization and packing."""
    flat_in = input.reshape(-1)
    numel = flat_in.numel()
    packed_len = (numel + 1) // 2

    # Pad if odd length
    if numel % 2 != 0:
        flat_in = torch.cat(
            [flat_in, torch.zeros(1, dtype=flat_in.dtype, device=flat_in.device)]
        )

    q = torch.clamp(
        torch.round(flat_in / scale) + zero_point,
        min=-8,
        max=7,
    ).to(torch.int32)
    q_masked = (q & 0x0F).to(torch.uint8)

    low = q_masked[0::2]
    high = q_masked[1::2]
    packed = (high << 4) | low
    return packed[:packed_len]


# ---------------------------------------------------------------------------
# TorchMetalKernel Registrations
# ---------------------------------------------------------------------------

fused_dequantize_int4_kernel = TorchMetalKernel(
    name="fused_dequantize_int4",
    input_names=["packed_data", "scale", "zero_point"],
    result_names=["output"],
    src=FUSED_DEQUANTIZE_INT4_MSL,
    torch_defn=_ref_dequantize_int4,
    metal_params=[
        MetalParameter("id", "uint", "thread_position_in_grid"),
    ],
)

fused_quantize_int4_kernel = TorchMetalKernel(
    name="fused_quantize_int4",
    input_names=["input", "scale", "zero_point"],
    result_names=["packed_output"],
    src=FUSED_QUANTIZE_INT4_MSL,
    torch_defn=_ref_quantize_int4,
    metal_params=[
        MetalParameter("id", "uint", "thread_position_in_grid"),
    ],
)


# ---------------------------------------------------------------------------
# User-facing helper functions
# ---------------------------------------------------------------------------


def dequantize_int4_metal(
    packed_data: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    output_shape: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Dequantize packed 4-bit weights to target shape using optimized Metal kernel.

    Args:
        packed_data: 1D or ND tensor of packed uint8 data (2 INT4 elements per byte).
        scale: Scale factor (scalar or per-channel tensor).
        zero_point: Zero-point offset tensor matching scale dtype.
        output_shape: Target shape for the dequantized output tensor.

    Returns:
        Dequantized tensor in float16/float32 matching scale.dtype.
    """
    total_elements = math.prod(output_shape)
    num_bytes = (total_elements + 1) // 2

    # Thread dispatch: 1 thread per packed byte
    threads_per_grid = (num_bytes, 1, 1)
    threads_per_threadgroup = (
        (min(256, num_bytes), 1, 1) if num_bytes > 0 else (1, 1, 1)
    )

    flat_out = fused_dequantize_int4_kernel(
        packed_data.contiguous().view(-1),
        scale.contiguous().view(-1),
        zero_point.contiguous().view(-1),
        threads_per_grid=threads_per_grid,
        threads_per_thread_group=threads_per_threadgroup,
        result_shapes=[[total_elements]],
    )
    return flat_out.view(*output_shape)


def quantize_int4_metal(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Quantize FP32/FP16 tensor to packed INT4 uint8 bytes using optimized Metal kernel.

    Args:
        input_tensor: Float tensor to quantize.
        scale: Quantization scale factor.
        zero_point: Quantization zero-point offset.

    Returns:
        Packed uint8 tensor with 2 INT4 elements per byte.
    """
    flat_in = input_tensor.contiguous().view(-1)
    total_elements = flat_in.numel()
    num_bytes = (total_elements + 1) // 2

    threads_per_grid = (num_bytes, 1, 1)
    threads_per_threadgroup = (
        (min(256, num_bytes), 1, 1) if num_bytes > 0 else (1, 1, 1)
    )

    return fused_quantize_int4_kernel(
        flat_in,
        scale.contiguous().view(-1),
        zero_point.contiguous().view(-1),
        threads_per_grid=threads_per_grid,
        threads_per_thread_group=threads_per_threadgroup,
        result_shapes=[[num_bytes]],
    )
