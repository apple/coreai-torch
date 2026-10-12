# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test uintx tensors."""

import math

import pytest
import torch

from coreai_torch._compression._intx import (  # type: ignore [attr-defined]
    IntxTensor,
    UintxTensor,
)


@pytest.mark.parametrize(
    "tensor_shape",
    [(3,), (2, 4), (3, 3, 3), (3, 4, 3, 3), (13, 39, 19, 111)],
)
@pytest.mark.parametrize(
    "nbits",
    [1, 2, 4, 6],
)
def test_uintx_tensor(tensor_shape: tuple[int, ...], nbits: int) -> None:
    """Test uintx tensor."""
    unpacked = torch.randint(0, 1 << nbits, tensor_shape).to(torch.uint8)
    uintx_tensor = UintxTensor.from_unpacked(unpacked, nbits)
    assert uintx_tensor.shape == torch.Size(tensor_shape)
    assert torch.equal(uintx_tensor.to(torch.uint8), unpacked)
    assert uintx_tensor.elem.numel() == math.ceil(math.prod(tensor_shape) * nbits / 8)


@pytest.mark.parametrize(
    "nbits",
    [1, 2, 4, 6],
)
def test_uintx_scalar(nbits: int) -> None:
    """Test uintx tensor for scalar input."""
    target_scalar = 2 if nbits > 1 else 1
    uintx_tensor = UintxTensor.from_unpacked(unpacked=target_scalar, nbits=nbits)
    assert uintx_tensor.shape == torch.Size((1,))
    assert torch.equal(uintx_tensor.to(torch.uint8), torch.Tensor([target_scalar]))
    # The packed elements could be padded when the number of bits is not aligned with
    # byte. For example, for 6-bit, 4 elements take 24 bits (3 bytes), so a single
    # 6-bit element will have two more padding values in packed `elem`.
    if nbits == 6:
        assert uintx_tensor.elem.numel() == 3
    else:
        assert uintx_tensor.elem.numel() == 1


@pytest.mark.parametrize(
    "tensor_shape",
    [(3,), (2, 4), (3, 3, 3), (3, 4, 3, 3), (13, 39, 19, 111)],
)
@pytest.mark.parametrize(
    "nbits",
    [4],
)
def test_intx_tensor(tensor_shape: tuple[int, ...], nbits: int) -> None:
    """Test intx tensor."""
    unpacked = torch.randint(
        -1 << (nbits - 1),
        (1 << (nbits - 1)) - 1,
        tensor_shape,
    ).to(torch.int8)
    intx_tensor = IntxTensor.from_unpacked(unpacked, nbits)
    assert intx_tensor.shape == torch.Size(tensor_shape)
    assert torch.equal(intx_tensor.to(torch.int8), unpacked)
    assert intx_tensor.elem.numel() == math.ceil(math.prod(tensor_shape) * nbits / 8)


@pytest.mark.parametrize(
    "nbits",
    [4],
)
def test_intx_scalar(nbits: int) -> None:
    """Test intx tensor with scalar input."""
    intx_tensor = IntxTensor.from_unpacked(unpacked=2, nbits=nbits)
    assert intx_tensor.shape == torch.Size((1,))
    assert torch.equal(intx_tensor.to(torch.int8), torch.Tensor([2]))
    assert intx_tensor.elem.numel() == 1


@pytest.mark.parametrize(
    "dim",
    [0, 1, -1],
)
def test_uintx_cat(dim: int) -> None:
    """Test uintx tensor concatenation along a given dimension."""
    unpacked = [
        torch.tensor([[3, 0, 14, 1], [0, 7, 15, 2]], dtype=torch.uint8),
        torch.tensor([[1, 4, 5, 6], [2, 3, 6, 1]], dtype=torch.uint8),
    ]
    uintx_tensors = [UintxTensor.from_unpacked(x, 4) for x in unpacked]
    concatenated = torch.cat(uintx_tensors, dim=dim)
    expected = torch.cat(unpacked, dim=dim)
    assert concatenated.shape == expected.shape
    assert torch.equal(concatenated.to(torch.uint8), expected)


@pytest.mark.parametrize(
    "dim",
    [0, 1, -1],
)
def test_intx_cat(dim: int) -> None:
    """Test intx tensor concatenation along a given dimension."""
    unpacked = [
        torch.tensor([[3, 0, -2, 1], [0, 7, -8, 2]], dtype=torch.int8),
        torch.tensor([[-1, 4, 5, -6], [2, -3, 6, 1]], dtype=torch.int8),
    ]
    intx_tensors = [IntxTensor.from_unpacked(x, 4) for x in unpacked]
    concatenated = torch.cat(intx_tensors, dim=dim)
    expected = torch.cat(unpacked, dim=dim)
    assert concatenated.shape == expected.shape
    assert torch.equal(concatenated.to(torch.int8), expected)


def test_uint4_packed_value() -> None:
    """Test uint4 packed values."""
    unpacked = torch.tensor([3, 0, 2, 0, 9, 14, 15], dtype=torch.uint8)
    uintx_tensor = UintxTensor.from_unpacked(unpacked, 4)
    # We use little end bitorder: https://numpy.org/doc/stable/reference/generated/numpy.packbits.html
    # [9, 14]: [(1, 0, 0, 1), (1, 1, 1, 0)] -> [1, 1, 1, 0, 1, 0, 0, 1] -> 233
    assert torch.equal(
        uintx_tensor.elem,
        torch.tensor([3, 2, 233, 15], dtype=torch.uint8),
    )


def test_uint2_packed_value() -> None:
    """Test uint2 packed values."""
    unpacked = torch.tensor([3, 0, 0, 0, 2, 0, 0, 0, 1, 0], dtype=torch.uint8)
    # Every 4 elements are packed into one byte.
    # [3, 0, 0, 0]: [(1, 1), (0, 0), (0, 0), (0, 0)] -> [00 00 00 11] -> 3
    # [2, 0, 0, 0]: [(1, 0), (0, 0), (0, 0), (0, 0)] -> [00 00 00 10] -> 2
    # [1, 0]: [(0, 1), (0, 0), padded(0, 0), padded(0, 0)] -> [00 00 00 01] -> 1
    uintx_tensor = UintxTensor.from_unpacked(unpacked, 2)
    assert torch.equal(uintx_tensor.elem, torch.tensor([3, 2, 1], dtype=torch.uint8))


def test_uint6_packed_value() -> None:
    """Test uint6 packed values."""
    unpacked = torch.tensor([21, 0, 35], dtype=torch.uint8)
    uintx_tensor = UintxTensor.from_unpacked(unpacked, 6)
    # We use little end bitorder: https://numpy.org/doc/stable/reference/generated/numpy.packbits.html
    # unpacked values are [010101], [0000-00], [10-0011]
    # it will gets packed to [00-010101], [0011-0000], [000000-10]
    assert torch.equal(
        uintx_tensor.elem,
        torch.tensor([21, 48, 2], dtype=torch.uint8),
    )

    unpacked = torch.tensor([21, 0, 35, 8], dtype=torch.uint8)
    uintx_tensor = UintxTensor.from_unpacked(unpacked, 6)
    # unpacked values are [010101], [0000-00], [10-0011], [001000]
    # it will gets packed to [00-010101], [0011-0000], [001000-10]
    assert torch.equal(
        uintx_tensor.elem,
        torch.tensor([21, 48, 34], dtype=torch.uint8),
    )


def test_uint1_packed_value() -> None:
    """Test uint1 packed values."""
    unpacked = torch.tensor([1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1], dtype=torch.uint8)
    # Every 8 elements are packed into one byte.
    # [1, 0, 1, 0, 1, 0, 0, 0] -> 00010101 -> 21
    # [1, 0, 1] -> padded(00000) 101 -> 5
    uintx_tensor = UintxTensor.from_unpacked(unpacked, 1)
    assert torch.equal(uintx_tensor.elem, torch.tensor([21, 5], dtype=torch.uint8))


def test_int4_packed_value() -> None:
    """Test int4 packed values."""
    unpacked = torch.tensor([[3, 0, -2], [0, 7, -8]], dtype=torch.int8)
    intx_tensor = IntxTensor.from_unpacked(unpacked, 4)
    # We use little end bitorder: https://numpy.org/doc/stable/reference/generated/numpy.packbits.html
    # [7, -8]: [(0, 1, 1, 1), (1, 0, 0, 0)] -> [1, 0, 0, 0, 0, 1, 1, 1] -> 135
    assert torch.equal(
        intx_tensor.elem,
        torch.tensor([3, 14, 135], dtype=torch.uint8),
    )
