# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Conversion tables for NumPy, Torch, and Core AI dtypes."""

from collections.abc import Callable

import numpy as np
import torch
from coreai._compiler.ir import (
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    Float4E2M1FNType,
    Float8E4M3FNType,
    Float8E5M2Type,
    Float8E8M0FNUType,
    IntegerType,
    Type,
)

_REDUCED_PRECISION_FLOAT_TYPES = (
    Float4E2M1FNType,
    Float8E4M3FNType,
    Float8E5M2Type,
    Float8E8M0FNUType,
)


def is_reduced_precision_float(elem_type: Type) -> bool:
    """Whether ``elem_type`` is an fp4/fp8 float type."""
    return isinstance(elem_type, _REDUCED_PRECISION_FLOAT_TYPES)


# Mapping of Torch dtypes to Core AI types
TORCH_TO_COREAI_DTYPE: dict[torch.dtype, Callable[[], Type]] = {
    torch.bool: lambda: IntegerType.get_signless(1),
    torch.uint1: lambda: IntegerType.get_unsigned(1),  # type: ignore[attr-defined]
    torch.uint2: lambda: IntegerType.get_unsigned(2),  # type: ignore[attr-defined]
    torch.uint3: lambda: IntegerType.get_unsigned(3),  # type: ignore[attr-defined]
    torch.uint4: lambda: IntegerType.get_unsigned(4),  # type: ignore[attr-defined]
    torch.uint6: lambda: IntegerType.get_unsigned(6),  # type: ignore[attr-defined]
    torch.int8: lambda: IntegerType.get_signed(8),
    torch.uint8: lambda: IntegerType.get_unsigned(8),
    torch.int16: lambda: IntegerType.get_signed(16),
    torch.uint16: lambda: IntegerType.get_unsigned(16),
    torch.int32: lambda: IntegerType.get_signed(32),
    torch.uint32: lambda: IntegerType.get_unsigned(32),
    torch.int64: lambda: IntegerType.get_signed(64),
    torch.float32: lambda: F32Type.get(),
    torch.float64: lambda: F64Type.get(),
    torch.float16: lambda: F16Type.get(),
    torch.bfloat16: lambda: BF16Type.get(),
    torch.float8_e5m2: lambda: Float8E5M2Type.get(),
    torch.float8_e4m3fn: lambda: Float8E4M3FNType.get(),
    torch.float8_e8m0fnu: lambda: Float8E8M0FNUType.get(),
    # Torch calls complex<f32> as complex64
    torch.complex32: lambda: ComplexType.get(F16Type.get()),
    torch.complex64: lambda: ComplexType.get(F32Type.get()),
}
if hasattr(torch, "int4"):
    TORCH_TO_COREAI_DTYPE[torch.int2] = lambda: IntegerType.get_signed(2)  # type: ignore[attr-defined]
    TORCH_TO_COREAI_DTYPE[torch.int4] = lambda: IntegerType.get_signed(4)
if hasattr(torch, "float4_e2m1fn_x2"):
    TORCH_TO_COREAI_DTYPE[torch.float4_e2m1fn_x2] = lambda: Float4E2M1FNType.get()  # type: ignore[attr-defined]


def _get_coreai_to_torch_dtype() -> dict[Type, torch.dtype]:
    """Get the Core AI-to-Torch dtype mapping, creating Core AI types lazily."""
    mapping = {
        IntegerType.get_signless(1): torch.bool,
        IntegerType.get_unsigned(1): torch.uint1,  # type: ignore[attr-defined]
        IntegerType.get_unsigned(2): torch.uint2,  # type: ignore[attr-defined]
        IntegerType.get_unsigned(3): torch.uint3,  # type: ignore[attr-defined]
        IntegerType.get_unsigned(4): torch.uint4,  # type: ignore[attr-defined]
        IntegerType.get_unsigned(6): torch.uint6,  # type: ignore[attr-defined]
        IntegerType.get_signed(8): torch.int8,
        IntegerType.get_unsigned(8): torch.uint8,
        IntegerType.get_signed(16): torch.int16,
        IntegerType.get_unsigned(16): torch.uint16,
        IntegerType.get_signed(32): torch.int32,
        IntegerType.get_unsigned(32): torch.uint32,
        IntegerType.get_signed(64): torch.int64,
        F16Type.get(): torch.float16,
        F32Type.get(): torch.float32,
        F64Type.get(): torch.float64,  # Note: TORCH_TO_COREAI_DTYPE maps both float32 and float64 to F32Type
        BF16Type.get(): torch.bfloat16,
        Float8E5M2Type.get(): torch.float8_e5m2,
        Float8E4M3FNType.get(): torch.float8_e4m3fn,
        Float8E8M0FNUType.get(): torch.float8_e8m0fnu,
        ComplexType.get(F16Type.get()): torch.complex32,
        ComplexType.get(F32Type.get()): torch.complex64,
    }

    # Add conditional mappings for newer torch dtypes if they exist
    if hasattr(torch, "int4"):
        mapping[IntegerType.get_signed(2)] = torch.int2  # type: ignore[attr-defined]
        mapping[IntegerType.get_signed(4)] = torch.int4
    if hasattr(torch, "float4_e2m1fn_x2"):
        mapping[Float4E2M1FNType.get()]: torch.float4_e2m1fn_x2  # type: ignore[attr-defined]

    return mapping


def _get_coreai_to_numpy_dtype() -> dict[Type, np.dtype]:
    """Get the Core AI-to-NumPy dtype mapping, creating Core AI types lazily."""
    mapping = {
        IntegerType.get_signless(1): np.dtype(np.bool_),
        IntegerType.get_unsigned(8): np.dtype(np.uint8),
        IntegerType.get_signed(8): np.dtype(np.int8),
        IntegerType.get_unsigned(16): np.dtype(np.uint16),
        IntegerType.get_signed(16): np.dtype(np.int16),
        IntegerType.get_unsigned(32): np.dtype(np.uint32),
        IntegerType.get_signed(32): np.dtype(np.int32),
        IntegerType.get_signed(64): np.dtype(np.int64),
        F16Type.get(): np.dtype(np.float16),
        F32Type.get(): np.dtype(np.float32),
        F64Type.get(): np.dtype(np.float64),
        BF16Type.get(): np.dtype(
            np.float32
        ),  # NumPy doesn't have bfloat16, use float32 as fallback
        ComplexType.get(F32Type.get()): np.dtype(np.complex64),
        ComplexType.get(F64Type.get()): np.dtype(np.complex128),
    }

    # Note: NumPy doesn't have equivalents for some torch-specific dtypes like:
    # - uint1, uint2, uint3, uint4, uint6 (sub-byte unsigned integers)
    # - float8_e5m2, float8_e4m3fn (8-bit floating point formats)
    # - int2, int4 (sub-byte signed integers)
    # - complex32 (half-precision complex)
    # These would need special handling or conversion if needed
    return mapping
