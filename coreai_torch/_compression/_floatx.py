# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, cast

import torch
from typing_extensions import Self, override


def byte_shape_to_fp4_shape(byte_shape: torch.Size) -> torch.Size:
    shape_list = list(byte_shape)
    shape_list[-1] *= 2
    return torch.Size(shape_list)


def fp4_shape_to_byte_shape(fp4_shape: torch.Size) -> torch.Size:
    shape_list = list(fp4_shape)
    shape_list[-1] //= 2
    return torch.Size(shape_list)


# in principle we can use this formula
#     mantissa = uint4_bits & 1  # 1 = 0001
#     exponent = (uint4_bits >> 1) & 3  # 3 = 0011
#     sign = (uint4_bits >> 3) & 1  # 1 = 0001
#     fp4value = (-1.0)**(sign) * 2.0**(exponent - 1) * (1.0 + 0.5 * mantissa)
#     subnormal_indices = torch.where(exponent == 0)
#     fp4value[subnormal_indices] = (-1.0)**(sign[subnormal_indices]) * 2.0**exponent[subnormal_indices] * 0.5 * mantissa[subnormal_indices]
# in practise it fails torch.export.export
#     torch._dynamo.exc.InternalTorchDynamoError: PendingUnbackedSymbolNotFound: Pending unbacked symbols {u2, u3, u0, u1} not in returned outputs FakeTensor(..., size=(1, 32, 2880, 2880), dtype=torch.float16) ((265420800, 8294400, 2880, 1), 0).
#     Did you accidentally call new_dynamic_size() or item() more times than you needed to in your fake implementation?
# so we have to use a lookup table
FP4_VALUES = torch.tensor(
    [
        +0.0,  # 0000
        +0.5,  # 0001
        +1.0,  # 0010 -> 2**(1 - 1) * 1
        +1.5,  # 0011 -> 2**(1 - 1) * (1 + 0.5)
        +2.0,  # 0100 -> 2**(2 - 1) * 1
        +3.0,  # 0101 -> 2**(2 - 1) * (1 + 0.5)
        +4.0,  # 0110 -> 2**(3 - 1) * 1
        +6.0,  # 0111 -> 2**(3 - 1) * (1 + 0.5)
        -0.0,  # 1000 -> -0.0 per OCP MX E2M1FN standard
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def unpack_fp4(uint8_byte: torch.Tensor) -> torch.Tensor:
    torch._check(  # type: ignore[no-untyped-call]
        uint8_byte.dtype == torch.uint8,
        message=lambda: "byte is represented as uint8",
    )
    # in principle low / high bits only need to be uint4
    # in practise .to(torch.uint4) would fail with
    #     NotImplementedError: "copy_" not implemented for 'UInt4'
    low_bits = uint8_byte & 15  # 15 = 00001111
    high_bits = uint8_byte >> 4
    # without .long call torch index errors out with
    #     IndexError: too many indices for tensor of dimension 1
    # Move FP4_VALUES to input's device so the gather works for any
    # device (CPU or non-CPU).
    fp4_values = FP4_VALUES.to(uint8_byte.device)
    low = fp4_values[low_bits.long()]
    high = fp4_values[high_bits.long()]
    return torch.stack((low, high), dim=-1).flatten(-2)


def fill_defaults(
    args: tuple[Any, ...],
    num_args_needed: int,
    defaults_tail: list[Any],
) -> tuple[Any, ...]:
    """
    Pad default values to args.

    __torch_dispatch__ doesn't guarantee the number of arguments you are
    passed (e.g., defaulted arguments are not passed); but usually it is
    convenient to pad out the arguments list with defaults.  This function
    helps you do that.

    Args:
    ----
        args: the list of positional arguments passed to __torch_dispatch__
        num_args_needed: the number of arguments you are expecting to get
        defaults_tail: default values for the arguments, starting from the
            end of the list

    """
    if len(args) >= num_args_needed:
        # already have enough args
        return args
    if len(args) + len(defaults_tail) < num_args_needed:
        message = "not enough defaults to fill arguments"
        raise ValueError(message)
    padded_args = list(args) + defaults_tail[len(args) - num_args_needed :]
    return tuple(padded_args)


class SubbyteTensor(torch.Tensor):
    """
    Minimal base class for subbyte tensor implementations.

    This class provides only the essential interface that all subbyte tensors
    must implement. Concrete subclasses must implement the 4 abstract methods.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore[assignment]

    def __init__(self: Self, elem: torch.Tensor):
        """Initialize the subbyte tensor given packed uint8 tensor containing the compressed data bytes."""
        self.elem = elem

    @property
    @abstractmethod
    def tensor_shape(self: Self) -> torch.Size: ...

    @property
    @abstractmethod
    def nbits(self: Self) -> int: ...

    @abstractmethod
    def __new__(
        cls: type[Self],
        elem: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> Self:
        """Create a new wrapper subclass instance."""
        ...

    @abstractmethod
    def __tensor_flatten__(self: Self) -> tuple[list[str], dict[str, Any]]:
        """Serialize the tensor for torch's serialization system."""
        ...

    @classmethod
    @abstractmethod
    def __tensor_unflatten__(
        cls: type[Self],
        flattened: dict[str, torch.Tensor],
        meta: dict[str, Any],
        outer_size: torch.Size,
        outer_stride: torch.Size,
    ) -> Self:
        """Deserialize the tensor from torch's serialization system."""
        ...

    @classmethod
    @abstractmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls: type[Self],
        func: Callable[..., Any],
        _types: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch aten ops to usual tensor ops."""
        ...

    @abstractmethod
    def unpack(self) -> torch.Tensor:
        """Unpack the compressed data to full precision."""
        ...


class Float4Tensor(SubbyteTensor):
    """Subbyte float tensor for FP4 quantization."""

    def __init__(self: Self, elem: torch.Tensor):
        """Initialize the subbyte tensor given packed uint8 tensor containing the compressed data bytes."""
        super().__init__(elem)
        self.elem.future_dtype = torch.float4_e2m1fn_x2

    @property
    @override
    def tensor_shape(self: Self) -> torch.Size:
        return byte_shape_to_fp4_shape(self.elem.shape)

    @property
    @override
    def nbits(self: Self) -> int:
        return 4

    @staticmethod
    @override
    def __new__(
        cls: type[Self],
        elem: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> Self:
        assert elem.dtype is torch.uint8, f"elem.dtype={elem.dtype}"
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False  # type: ignore[assignment]
        result = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            torch.Size(byte_shape_to_fp4_shape(elem.shape)),
            dtype=torch.uint8,
            **kwargs,
        )
        return cast("Self", result)

    @override
    def __tensor_flatten__(self: Self) -> tuple[list[str], dict[str, Any]]:
        return ["elem"], {
            "tensor_shape": self.tensor_shape,
            "nbits": self.nbits,
        }

    @classmethod
    @override
    def __tensor_unflatten__(
        cls: type[Self],
        flattened: dict[str, torch.Tensor],
        meta: dict[str, Any],
        outer_size: torch.Size,
        outer_stride: torch.Size,
    ) -> Self:
        elem = flattened["elem"]
        return cls(elem)

    @classmethod
    @override
    def __torch_dispatch__(  # type: ignore[override]
        cls: type[Self],
        func: Callable[..., Any],
        _types: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func in (
            torch.ops.aten.detach.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.contiguous.default,
        ):
            self = args[0]
            new_elem = func(self.elem)
            return cls(new_elem)
        elif func in (torch.ops.aten.reshape.default, torch.ops.aten.view.default):
            self, fp4_shape = args
            byte_shape = fp4_shape_to_byte_shape(fp4_shape)
            new_elem = func(self.elem, byte_shape)
            return cls(new_elem)
        elif func is torch.ops.aten.unsqueeze.default:
            self, dim = args
            new_elem = self.elem.unsqueeze(dim)
            return cls(new_elem)
        elif func is torch.ops.aten.select.int:
            self, dim, index = args
            return func(self.unpack(), dim, index)
        elif func is torch.ops.aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            torch._check(  # type: ignore[no-untyped-call]
                not (dim != -1 and (dim != (len(self.shape) - 1)))
                or end >= len(self.shape),
                message=lambda: "cannot slice the packed fp4 dim",
            )
            sliced_elem = func(self.elem, dim, start, end, step)
            return cls(sliced_elem)
        elif func.name() == "coreai::constexpr_blockwise_shift_scale":  # type: ignore[attr-defined]
            self, scale, zero_point, minval, input_dtype, output_dtype = args
            if zero_point is not None:
                zero_point = zero_point.elem
            return func(self.elem, scale, zero_point, minval, input_dtype, output_dtype)
        else:
            error_message = (
                f"{func} is not implemented in Float4Tensor __torch_dispatch__"
            )
            raise NotImplementedError(error_message)

    @override
    def unpack(self: Self) -> torch.Tensor:
        """
        Warning: Do not use this method if want core aten torch.export graph.

        Using this method can pass torch.export.export
        but would fail torch.export.ExportedProgram.run_decomposition
            torch._export.verifier.SpecViolationError: Constant tensor... is not in the constants dictionary.
        because torch.export.export loses location info of constants
        (unless they are torch.nn.Module parameter or registered buffer)
        so torch.export.ExportedProgram.run_decomposition cannot find them.
        Here the lookup table FP4_VALUES is such constant
        """
        return unpack_fp4(self.elem)
