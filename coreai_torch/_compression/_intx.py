# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# Most codes of _uintx.py are inspired from pytorch ao uint4 tensor
# implementation before torch has complete implementation for
# uint2/uint4/uint6/int4. We ignore the type checking for this file.

# type: ignore

import torch
import torch._prims_common as utils


def fill_defaults(args, n, defaults_tail):
    """
    __torch_dispatch__ doesn't guarantee the number of arguments you are
    passed (e.g., defaulted arguments are not passed); but usually it is
    convenient to pad out the arguments list with defaults.  This function
    helps you do that.

    Args:
    ----
        args: the list of positional arguments passed to __torch_dispatch__
        n: the number of arguments you are expecting to get
        defaults_tail: default values for the arguments, starting from the
            end of the list

    """
    if n - len(defaults_tail) > len(args):
        raise RuntimeError("not enough defaults to fill arguments")
    r = list(args)
    for i in range(len(args), n):
        r.append(defaults_tail[i - n + len(defaults_tail)])
    return r


def _unpack_uint4(uint8_data: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    data1 = (uint8_data >> 4).to(torch.uint8)
    data0 = (uint8_data & 0b1111).to(torch.uint8)
    stacked = torch.stack([data0, data1], dim=-1).view(-1)
    size = shape.numel()
    assert size <= len(stacked) <= size + 1
    stacked = stacked[:size]
    return stacked.view(shape)


def _pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    uint8_data = uint8_data.contiguous().view(-1)
    data0 = uint8_data[::2]
    data1 = uint8_data[1::2]
    if len(data1) != len(data0):
        data1 = torch.cat((data1, torch.tensor([0], dtype=torch.uint8)))
    return (data1 << 4 | data0).view(-1)


def _unpack_uint2(uint8_data: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    data3 = (uint8_data >> 6) & 0b11
    data2 = (uint8_data >> 4) & 0b11
    data1 = (uint8_data >> 2) & 0b11
    data0 = uint8_data & 0b11
    stacked = torch.stack((data0, data1, data2, data3), dim=-1).view(-1)
    size = shape.numel()
    assert size <= len(stacked) <= size + 3
    stacked = stacked[:size]
    return stacked.view(shape)


def _pack_uint2(uint8_data: torch.Tensor) -> torch.Tensor:
    uint8_data = uint8_data.contiguous().view(-1)
    data0 = uint8_data[::4]
    data1 = uint8_data[1::4]
    data2 = uint8_data[2::4]
    data3 = uint8_data[3::4]
    if len(data1) != len(data0):
        data1 = torch.cat((data1, torch.tensor([0], dtype=torch.uint8)))
    if len(data2) != len(data0):
        data2 = torch.cat((data2, torch.tensor([0], dtype=torch.uint8)))
    if len(data3) != len(data0):
        data3 = torch.cat((data3, torch.tensor([0], dtype=torch.uint8)))
    return (data3 << 6 | data2 << 4 | data1 << 2 | data0).view(-1)


def _unpack_uint6(uint8_data: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    # Every 3 bytes contain 4 uint6 values (as per little-endian order).
    bytes0 = uint8_data[::3]
    bytes1 = uint8_data[1::3]
    bytes2 = uint8_data[2::3]
    # Padding if length is not a multiple of 3
    if len(bytes1) != len(bytes0):
        bytes1 = torch.cat((bytes1, torch.tensor([0], dtype=torch.uint8)))
    if len(bytes2) != len(bytes0):
        bytes2 = torch.cat((bytes2, torch.tensor([0], dtype=torch.uint8)))
    data3 = bytes2 >> 2
    data2 = ((bytes2 & 0b11) << 4) | (bytes1 >> 4)
    data1 = ((bytes1 & 0b1111) << 2) | (bytes0 >> 6)
    data0 = bytes0 & 0b111111
    stacked = torch.stack((data0, data1, data2, data3), dim=-1).view(-1)
    size = shape.numel()
    assert size <= len(stacked) <= size + 3
    stacked = stacked[:size]
    return stacked.view(shape)


def _pack_uint6(uint8_data: torch.Tensor) -> torch.Tensor:
    uint8_data = uint8_data.contiguous().view(-1)
    data0 = uint8_data[::4]
    data1 = uint8_data[1::4]
    data2 = uint8_data[2::4]
    data3 = uint8_data[3::4]
    # Padding if length is not a multiple of 4
    if len(data1) != len(data0):
        data1 = torch.cat((data1, torch.tensor([0], dtype=torch.uint8)))
    if len(data2) != len(data0):
        data2 = torch.cat((data2, torch.tensor([0], dtype=torch.uint8)))
    if len(data3) != len(data0):
        data3 = torch.cat((data3, torch.tensor([0], dtype=torch.uint8)))
    packed = (data0 | (data1 << 6)).view(-1, 1)
    packed = torch.cat((packed, (data1 >> 2 | (data2 << 4)).view(-1, 1)), dim=1)
    packed = torch.cat((packed, (data2 >> 4 | (data3 << 2)).view(-1, 1)), dim=1)
    return packed.view(-1)


def _unpack_uint1(uint8_data: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    # Extract 8 bits from each byte and stack them into a flat tensor.
    bits = torch.stack([(uint8_data >> i) & 1 for i in range(8)], dim=-1).reshape(-1)
    # Trim to the desired number of elements and reshape to the target shape.
    return bits[: shape.numel()].view(shape)


def _pack_uint1(uint8_data: torch.Tensor) -> torch.Tensor:
    bits = uint8_data.contiguous().view(-1)
    # Pad with zeros if the total number of bits isn't a multiple of 8.
    pad = (-bits.numel()) % 8
    if pad:
        bits = torch.cat((bits, torch.zeros(pad, dtype=torch.uint8)))
    # Reshape into groups of 8 bits.
    bits = bits.view(-1, 8)
    # Create weights for each bit (bit0 as LSB up to bit7 as MSB).
    weights = 2 ** torch.arange(8, dtype=torch.uint8)
    # Multiply each group by the weights and sum them to form packed bytes.
    return (bits * weights).sum(dim=1).to(torch.uint8)


def unpack_uintx(
    uint8_data: torch.Tensor,
    shape: torch.Size,
    nbits: int,
) -> torch.Tensor:
    if nbits == 1:
        return _unpack_uint1(uint8_data, shape)
    if nbits == 2:
        return _unpack_uint2(uint8_data, shape)
    if nbits == 4:
        return _unpack_uint4(uint8_data, shape)
    if nbits == 6:
        return _unpack_uint6(uint8_data, shape)
    error_message = f"only uint2|uint4|uint6 is supported, but {nbits=} is given."
    raise RuntimeError(error_message)


def pack_uintx(uint8_data: torch.Tensor, nbits: int) -> torch.Tensor:
    """Pack uintx data into compact representation."""
    if uint8_data.dtype != torch.uint8:
        if not torch.all(uint8_data == uint8_data.to(torch.uint8)):
            err_msg = f"The input has to be uint8 data, but got {uint8_data.dtype}."
            raise AssertionError(err_msg)
        uint8_data = uint8_data.to(torch.uint8)

    if nbits == 1:
        return _pack_uint1(uint8_data)
    if nbits == 2:
        return _pack_uint2(uint8_data)
    if nbits == 4:
        return _pack_uint4(uint8_data)
    if nbits == 6:
        return _pack_uint6(uint8_data)
    error_message = f"only uint2|uint4|uint6 is supported, but {nbits=} is given."
    raise RuntimeError(error_message)


def _unpack_int2(uint8_data: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    uint2_data = _unpack_uint2(uint8_data, shape)
    uint2_data = uint2_data.to(torch.int8)
    # If the value is in the range 2-3, interpret it as a negative int2.
    int2_data = torch.where(uint2_data > 1, uint2_data - 4, uint2_data)
    if int2_data.dim() == 0:
        int2_data = int2_data.unsqueeze(0)
    return int2_data


def _pack_int2(int8_data: torch.Tensor) -> torch.Tensor:
    int8_data = int8_data.contiguous().view(-1)
    # Reinterpret int2 bits as uint2 bits represented in uint8.
    # For example, -1 has bits 11111111 in int8, -2 in int2 should have bits 11.
    # It would be 3(00000011) in uint8.
    uint8_data = (int8_data & 0x3).to(torch.uint8)
    return _pack_uint2(uint8_data)


def _unpack_int4(uint8_data: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    uint4_data = _unpack_uint4(uint8_data, shape)
    uint4_data = uint4_data.to(torch.int8)
    # If the value is in the range 8-15, interpret it as a negative int4.
    int4_data = torch.where(uint4_data > 7, uint4_data - 16, uint4_data)
    if int4_data.dim() == 0:
        int4_data = int4_data.unsqueeze(0)
    return int4_data


def _pack_int4(int8_data: torch.Tensor) -> torch.Tensor:
    int8_data = int8_data.contiguous().view(-1)
    # Reinterpret int4 bits as uint4 bits represented in uint8.
    # For example, -1 has bits 11111111 in int8, -8 in int4 should have bits 1111.
    # It would be 15(00001111) in uint8.
    uint8_data = (int8_data & 0xF).to(torch.uint8)
    return _pack_uint4(uint8_data)


def unpack_intx(
    uint8_data: torch.Tensor,
    shape: torch.Size,
    nbits: int,
) -> torch.Tensor:
    """
    The uint8_data is the packed data. For example, when nbits is 4, it means each
    uint8 element in the uint8_data represents two int4 elements.
    """
    if nbits == 2:
        return _unpack_int2(uint8_data, shape)
    if nbits == 4:
        return _unpack_int4(uint8_data, shape)
    error_message = f"only int2/4 is supported, but {nbits=} is given."
    raise RuntimeError(error_message)


def pack_intx(int8_data: torch.Tensor, nbits: int) -> torch.Tensor:
    """Pack intx data into compact representation."""
    if int8_data.dtype != torch.int8:
        if not torch.all(int8_data == int8_data.to(torch.int8)):
            err_msg = f"The input has to be int8 data, but got {int8_data.dtype}."
            raise AssertionError(err_msg)
        int8_data = int8_data.to(torch.int8)
    if nbits == 2:
        return _pack_int2(int8_data)
    if nbits == 4:
        return _pack_int4(int8_data)
    err_msg = f"only int2/4 is supported, but {nbits=} is given."
    raise RuntimeError(err_msg)


class SubbyteTensor(torch.Tensor):
    """
    SubbyteTensor is our own subclass implementation to support subbyte tensor classes with pytorch native torch.uint4 and
    torch.uint2, torch.int4 dtype. We didn't use torch.ao provided uint4Tensor(https://github.com/pytorch/ao/blob/ed4c405afdffeb3b1f594db625086d60a650e6be/torchao/dtypes/uint4.py#L96)
    or UintxTensor(https://github.com/pytorch/ao/blob/ed4c405afdffeb3b1f594db625086d60a650e6be/torchao/dtypes/uintx/Uintx.py#L18)
    because they are either prototype or wip with some shape constraints that only work for some cases.

    We plan to switch to ao once that gets more stable and usable.
    """

    def __init__(
        self, elem: torch.Tensor, tensor_shape: torch.Size, nbits: int, signed: bool
    ):
        """
        The sub-byte tensor represented by packed data, where `elem` stores the packed
        elements, `tensor_shape` stores the original unpacked shape, and `nbits` together
        with `signed` represents the dtype.
        """
        self.elem = elem
        self.tensor_shape = torch.Size(tensor_shape)
        self.nbits = nbits
        self.signed = signed
        self.unpack_func = unpack_intx if signed else unpack_uintx

        if any(dim_size < 0 for dim_size in tensor_shape):
            # When the tensor shape has negative values, use the original elements shape.
            self.tensor_shape = self.elem.shape

    def __tensor_flatten__(self):
        return ["elem"], {
            "tensor_shape": self.tensor_shape,
            "nbits": self.nbits,
            "signed": self.signed,
        }

    @staticmethod
    def __tensor_unflatten__(flattened, meta, outer_size, outer_stride):
        elem = flattened["elem"]
        tensor_shape = meta["tensor_shape"]
        nbits = meta["nbits"]
        signed = meta["signed"]
        if signed:
            return IntxTensor(elem, tensor_shape, nbits)
        else:
            return UintxTensor(elem, tensor_shape, nbits)

    def __hash__(self):
        return hash((self.elem, self.tensor_shape, self.nbits, self.signed))

    def __eq__(self, other):
        if not isinstance(other, SubbyteTensor):
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return unpacked == other
        return (
            torch.equal(self.elem, other.elem)
            and torch.equal(self.tensor_shape, other.tensor_shape)
            and self.nbits == other.nbits
            and self.signed == other.signed
        )

    @classmethod
    def __torch_dispatch__(cls, func, _types, args, kwargs=None):
        if func is torch.ops.aten.reshape.default:
            self, size = args
            return cls(self.elem, size, self.nbits)
        if func is torch.ops.aten.view.default:
            self, size = args
            size = utils.infer_size(size, self.numel())
            assert not kwargs
            # WARNING: views not preserved
            return cls(self.elem, size, self.nbits)
        if func is torch.ops.aten._to_copy.default:
            (self,) = args
            if kwargs == {"dtype": torch.uint8} or kwargs == {"dtype": torch.int8}:
                return self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            error_message = f"_to_copy {kwargs}"
            raise NotImplementedError(error_message)
        if func is torch.ops.aten.detach.default:
            (self,) = args
            return cls(func(self.elem), self.tensor_shape, self.nbits)
        if func is torch.ops.aten.select.int:
            self, dim, index = args
            selected_uint8 = torch.ops.aten.select.int(
                self.unpack_func(self.elem, self.tensor_shape, self.nbits),
                dim,
                index,
            )
            return cls.from_unpacked(selected_uint8, self.nbits)
        if func is torch.ops.aten.unbind.int:
            self, dim = fill_defaults(args, 2, [0])
            return torch.ops.aten.unbind.int(
                self.unpack_func(self.elem, self.tensor_shape, self.nbits),
                dim,
            )
        if func is torch.ops.aten.stack.default:
            self, dim = fill_defaults(args, 2, [0])
            stacked_uint8 = torch.ops.aten.stack.default(
                [x.unpack_func(x.elem, x.tensor_shape, x.nbits) for x in self],
                dim,
            )
            return cls.from_unpacked(stacked_uint8, self[0].nbits)
        if func is torch.ops.aten.eq.Tensor:
            self, other = args
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            if isinstance(other, SubbyteTensor):
                other = self.unpack_func(other.elem, other.tensor_shape, other.nbits)
            return func(unpacked, other)
        if func is torch.ops.aten.abs.default:
            (self,) = args
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return cls.from_unpacked(func(unpacked), self.nbits)
        if func is torch.ops.aten.ne.Scalar:
            self, scalar = args
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return func(unpacked, scalar)
        if func is torch.ops.aten.masked_select.default:
            self, mask = args
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return func(unpacked, mask)
        if func is torch.ops.aten.equal.default:
            self, other = args
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return func(unpacked, other)
        if func is torch.ops.aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return cls.from_unpacked(func(unpacked, dim, start, end, step), self.nbits)
        if func is torch.ops.aten.cat.default:
            self, dim = fill_defaults(args, 2, [0])
            unpacked = [x.unpack_func(x.elem, x.tensor_shape, x.nbits) for x in self]
            return cls.from_unpacked(func(unpacked, dim), self[0].nbits)
        if func is torch.ops.aten.min.default:
            (self,) = args
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return func(unpacked)
        if func is torch.ops.aten.max.default:
            (self,) = args
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return func(unpacked)
        if func is torch.ops.aten.bitwise_and.Scalar:
            self, other = args
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return func(unpacked, other)
        if func.name() == "coreai::lut_to_dense":
            self = args[0]
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return func(unpacked, *args[1:])
        if func.name() == "coreai::constexpr_blockwise_shift_scale":
            self = args[0]
            unpacked = self.unpack_func(self.elem, self.tensor_shape, self.nbits)
            return func(unpacked, *args[1:])

        error_message = (f"{func} is not implemented in intx __torch_dispatch__",)
        raise NotImplementedError(error_message)

    __torch_function__ = torch._C._disabled_torch_function_impl


class UintxTensor(SubbyteTensor):
    """
    UIntX is our own subclass implementation to support subbyte tensor classes with pytorch native torch.uint4 and
    torch.uint2 data type.
    """

    nbits_to_dtype = {
        1: torch.uint1,  # type: ignore[attr-defined]
        2: torch.uint2,  # type: ignore[attr-defined]
        4: torch.uint4,  # type: ignore[attr-defined]
        6: torch.uint6,  # type: ignore[attr-defined]
    }

    @staticmethod
    def __new__(cls, elem, tensor_shape, nbits, **kwargs):
        assert elem.dtype is torch.uint8, f"elem.dtype={elem.dtype}"
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False

        return torch.Tensor._make_wrapper_subclass(
            cls,
            torch.Size(tensor_shape),
            dtype=cls.nbits_to_dtype[nbits],
            **kwargs,
        )

    def __init__(self, elem, tensor_shape, nbits):
        if nbits not in [1, 2, 4, 6]:
            error_message = f"only uint1/2/4/6 is supported yet, {nbits=} is given."
            raise ValueError(error_message)
        super().__init__(elem, tensor_shape, nbits, signed=False)

    @classmethod
    def from_unpacked(cls, unpacked, nbits):
        """The unpacked data has elements within nbits value range but represented by uint8."""
        if nbits not in [1, 2, 4, 6]:
            error_message = f"only uint1/2/4/6 is supported yet, {nbits=} is given."
            raise ValueError(error_message)
        if isinstance(unpacked, int):
            unpacked = torch.tensor([unpacked], dtype=torch.uint8)
        return cls(pack_uintx(unpacked, nbits), unpacked.shape, nbits)

    def tolist(self):
        return self.to(torch.uint8).tolist()


class IntxTensor(SubbyteTensor):
    """
    IntxTensor is our own subclass implementation to support subbyte tensor classes with pytorch native
    torch.int4 data type.
    """

    if hasattr(torch, "int4"):
        int4_dtype = torch.int4  # type: ignore[attr-defined]
        int2_dtype = torch.int2  # type: ignore[attr-defined]
    else:
        int4_dtype = torch.int8
        int2_dtype = torch.int8
    nbits_to_dtype = {
        2: int2_dtype,
        4: int4_dtype,
    }

    _SUPPORTED_NBITS = {2, 4}

    @staticmethod
    def __new__(
        cls, elem: torch.Tensor, tensor_shape: torch.Size, nbits: int, **kwargs
    ):
        assert elem.dtype is torch.uint8, f"elem.dtype={elem.dtype}"
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False

        return torch.Tensor._make_wrapper_subclass(
            cls,
            torch.Size(tensor_shape),
            dtype=cls.nbits_to_dtype[nbits],
            **kwargs,
        )

    def __init__(self, elem: torch.Tensor, tensor_shape: torch.Size, nbits: int):
        if nbits not in self._SUPPORTED_NBITS:
            error_message = f"only nbits in {self._SUPPORTED_NBITS} is supported yet, {nbits=} is given."
            raise ValueError(error_message)
        super().__init__(elem, tensor_shape, nbits, signed=True)

    @classmethod
    def from_unpacked(cls, unpacked: int | torch.Tensor, nbits: int):
        """The unpacked data has elements within nbits value range but represented by int8."""
        if nbits not in cls._SUPPORTED_NBITS:
            error_message = f"only nbits in {cls._SUPPORTED_NBITS} is supported yet, {nbits=} is given."
            raise ValueError(error_message)
        if isinstance(unpacked, int):
            unpacked = torch.tensor([unpacked], dtype=torch.int8)
        if not isinstance(unpacked, torch.Tensor):
            raise AssertionError(f"unpacked must be torch.Tensor, got {type(unpacked)}")
        return cls(pack_intx(unpacked, nbits), unpacked.shape, nbits)

    def tolist(self):
        return unpack_intx(self.elem, self.tensor_shape, self.nbits).tolist()
