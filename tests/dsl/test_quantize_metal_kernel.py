# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for high-performance sub-byte Metal quantization kernels."""

from __future__ import annotations

import sys

import pytest
import torch

from coreai_torch.kernels.quantization import (
    _ref_dequantize_int4,
    _ref_quantize_int4,
    fused_dequantize_int4_kernel,
    fused_quantize_int4_kernel,
)


class TestSubbyteMetalQuantization:
    """Test suite for 4-bit Metal quantization and dequantization operations."""

    def test_ref_quantize_dequantize_roundtrip(self) -> None:
        """Ensure reference quantize and dequantize roundtrip with minimal error."""
        torch.manual_seed(42)
        # Test 1D tensor
        x = torch.tensor([-3.5, -1.2, 0.0, 1.8, 4.2, 6.7], dtype=torch.float32)
        scale = torch.tensor([0.5], dtype=torch.float32)
        zero_point = torch.tensor([0.0], dtype=torch.float32)

        packed = _ref_quantize_int4(x, scale, zero_point)
        # 6 elements -> 3 bytes
        assert packed.numel() == 3
        assert packed.dtype == torch.uint8

        dequant = _ref_dequantize_int4(packed, scale, zero_point)
        assert dequant.numel() == 6
        # Check that dequantized elements are close to original values within quantization step
        abs_err = torch.abs(dequant - x)
        assert torch.all(abs_err <= scale)

    def test_signed_int4_range(self) -> None:
        """Verify signed INT4 bounds [-8, 7] are respected in quantization."""
        x = torch.tensor([-100.0, -8.0, 0.0, 7.0, 100.0], dtype=torch.float32)
        scale = torch.tensor([1.0], dtype=torch.float32)
        zero_point = torch.tensor([0.0], dtype=torch.float32)

        packed = _ref_quantize_int4(x, scale, zero_point)
        dequant = _ref_dequantize_int4(packed, scale, zero_point)

        # -100 clamps to -8, +100 clamps to 7
        assert dequant[0].item() == -8.0
        assert dequant[1].item() == -8.0
        assert dequant[2].item() == 0.0
        assert dequant[3].item() == 7.0
        assert dequant[4].item() == 7.0

    @pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests run only on Mac")
    def test_metal_quantize_kernel_attributes(self) -> None:
        """Verify TorchMetalKernel metadata and parameter setup for Apple Silicon."""
        assert fused_dequantize_int4_kernel.name == "fused_dequantize_int4"
        assert fused_dequantize_int4_kernel.input_names == [
            "packed_data",
            "scale",
            "zero_point",
        ]
        assert fused_dequantize_int4_kernel.result_names == ["output"]

        assert fused_quantize_int4_kernel.name == "fused_quantize_int4"
        assert fused_quantize_int4_kernel.input_names == [
            "input",
            "scale",
            "zero_point",
        ]
        assert fused_quantize_int4_kernel.result_names == ["packed_output"]
