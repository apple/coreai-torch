# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Optimized Metal Shading Language (MSL) GPU kernels for Apple Silicon."""

from .quantization import (
    dequantize_int4_metal,
    fused_dequantize_int4_kernel,
    fused_quantize_int4_kernel,
    quantize_int4_metal,
)

__all__ = [
    "fused_dequantize_int4_kernel",
    "fused_quantize_int4_kernel",
    "dequantize_int4_metal",
    "quantize_int4_metal",
]
