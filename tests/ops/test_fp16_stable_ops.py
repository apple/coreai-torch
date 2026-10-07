# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for numerically stable softplus, mish, and logsumexp converters (#21).

These tests verify that the stable decompositions produce correct results
for inputs in the fp16 overflow range (x > ~10.4), where the naive forms
(log(1+exp(x)), etc.) would overflow to inf.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from ..utils import (
    _all_dims_dynamic,
    validate_numerical_output,
)


# --- Softplus ---


class SoftplusModel(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x)


class SoftplusBetaModel(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x, beta=2.0)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([0.0, 1.0, -1.0, 5.0, -5.0]),
        # Inputs in the fp16 overflow range — naive softplus fails here
        torch.tensor([10.0, 11.0, 15.0, 20.0, 50.0]),
        torch.randn(3, 4),
    ],
)
async def test_softplus(x: Tensor, dynamic: bool) -> None:
    """Test softplus with inputs spanning the fp16 overflow threshold."""
    model = SoftplusModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


@pytest.mark.parametrize("dynamic", [False])
@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([0.0, 5.0, 10.0, 15.0]),
    ],
)
async def test_softplus_beta(x: Tensor, dynamic: bool) -> None:
    """Test softplus with beta != 1."""
    model = SoftplusBetaModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


# --- Mish ---


class MishModel(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.mish(x)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([0.0, 1.0, -1.0, 5.0, -5.0]),
        # Inputs in the fp16 overflow range — naive mish fails here
        torch.tensor([10.0, 11.0, 15.0, 20.0, 50.0]),
        torch.randn(3, 4),
    ],
)
async def test_mish(x: Tensor, dynamic: bool) -> None:
    """Test mish with inputs spanning the fp16 overflow threshold."""
    model = MishModel().eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


# --- Logsumexp ---


class LogsumexpModel(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        return torch.logsumexp(x, dim=self.dim, keepdim=self.keepdim)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        # Inputs in the fp16 overflow range — naive logsumexp fails here
        torch.tensor([[8.0, 9.0, 10.0], [11.0, 12.0, 15.0]]),
        torch.randn(3, 4),
    ],
)
async def test_logsumexp(x: Tensor, keepdim: bool, dynamic: bool) -> None:
    """Test logsumexp with inputs spanning the fp16 overflow threshold."""
    model = LogsumexpModel(dim=1, keepdim=keepdim).eval()
    dynamic_shapes = {"x": _all_dims_dynamic(x)} if dynamic else None
    await validate_numerical_output(model=model, x=x, dynamic_shapes=dynamic_shapes)


# --- Numeric fp16 overflow proof ---


def test_fp16_overflow_proof() -> None:
    """Verify naive fp16 softplus overflows while stable form does not."""
    x_val = np.float16(15.0)

    # Naive: log(1 + exp(fp16(15))) overflows
    naive = np.float16(np.log(np.float16(1.0) + np.exp(x_val)))
    assert not np.isfinite(naive), f"Naive fp16 softplus should overflow, got {naive}"

    # Stable: max(15,0) + log(1 + exp(-15)) ≈ 15.0
    stable = np.float16(
        np.maximum(x_val, np.float16(0))
        + np.log(np.float16(1.0) + np.exp(-np.abs(x_val)))
    )
    assert np.isfinite(stable), f"Stable fp16 softplus should not overflow, got {stable}"
    assert abs(float(stable) - 15.0) < 0.5, f"Expected ~15.0, got {stable}"
