# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import math
import os
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.fx as fx
from coreai._compiler.dialects import coreai
from coreai._compiler.ir import (
    F16Type,
    F32Type,
    IntegerType,
    Location,
    OpResult,
    RankedTensorType,
    ShapedType,
    Type,
    Value,
)
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text
from torch import Tensor
from torch.export import Dim
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import ExportGraphSignature
from torch.fx.node import Argument

from ._composite_declaration import generate_composite_decl
from ._type_mapping import (
    TORCH_TO_COREAI_DTYPE,
    _get_coreai_to_torch_dtype,
)


class _BarColumn(BarColumn):
    """``BarColumn`` that renders empty for indeterminate (``total=None``) tasks.

    Suppresses rich's pulsing-bar animation, which is laggy when the host
    loop calls ``advance()`` frequently. Determinate tasks still get a
    normal bar.
    """

    def render(self, task: Any) -> Any:
        if task.total is None:
            return Text("")
        return super().render(task)


class _ProgressBar:
    """Progress display backed by ``rich.progress.Progress``.

    Two construction modes:

    - ``_ProgressBar()`` — empty multi-task host. Iterate sub-sequences
      via :meth:`track`; pass ``transient=True`` to have a sub-bar
      disappear when its iteration finishes.

    - ``_ProgressBar(total=N, description=...)`` — starts a single
      streaming task driven by :meth:`update` / :meth:`set_postfix`.

    Usable as a context manager.
    """

    def __init__(self, *, total: int | None = None, description: str = "") -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            _BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[postfix]}"),
            disable=not sys.stdout.isatty(),
        )
        self._progress.start()
        self._task_id = (
            self._progress.add_task(description, total=total, postfix="")
            if total is not None
            else None
        )

    def track(
        self,
        items: Iterable[Any],
        *,
        description: str,
        transient: bool = False,
    ) -> Iterator[Any]:
        task_id = self._progress.add_task(
            description,
            total=len(items),
            postfix="",  # type: ignore[arg-type]
        )
        for item in items:
            yield item
            self._progress.advance(task_id)
        if transient:
            self._progress.remove_task(task_id)

    def update(self, n: int = 1) -> None:
        assert self._task_id is not None, (
            "update() requires _ProgressBar(total=..., description=...)"
        )
        self._progress.advance(self._task_id, n)

    def set_postfix(self, fields: dict[str, Any]) -> None:
        assert self._task_id is not None, (
            "set_postfix() requires _ProgressBar(total=..., description=...)"
        )
        text = ", ".join(f"{k}={v}" for k, v in fields.items())
        self._progress.update(self._task_id, postfix=text)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print through the bar's console so output stays in one stream."""
        self._progress.console.print(*args, **kwargs)

    @contextmanager
    def stream(self, description: str) -> Iterator[Callable[[], None]]:
        """Yield an ``advance()`` callback for a transient indeterminate task.

        Use when the iterable can't be materialized upfront (e.g. a stateful
        generator whose body must run between yields). The caller calls
        ``advance()`` after each item is processed; the task is removed when
        the block exits. The bar widget is suppressed (see
        :class:`_BarColumn`); the user sees spinner + description + ``n/?``
        count + elapsed time.
        """
        task_id = self._progress.add_task(description, total=None, postfix="")
        yield lambda: self._progress.advance(task_id)
        self._progress.remove_task(task_id)

    @contextmanager
    def status(self, description: str) -> Iterator[None]:
        """Show a transient spinner-only display during a slow opaque step.

        The existing progress bars are paused (and their display cleared,
        not left in scrollback) for the duration of the block — rich
        permits only one live display at a time — and resume when the
        block exits. The spinner clears with no scrollback.
        """
        live = self._progress.live
        prev_transient = live.transient
        live.transient = True
        self._progress.stop()
        live.transient = prev_transient
        with self._progress.console.status(description):
            yield
        self._progress.start()

    def close(self) -> None:
        self._progress.stop()

    def __enter__(self) -> "_ProgressBar":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def to_rank1_int32(v: Value) -> Value:
    """Coerce a SymInt-derived Value to canonical rank-1 si32 form.

    Dim-vector concats (used to build shape operands for ``coreai.reshape``,
    ``coreai.interpolate``, etc.) require all inputs to share rank and
    element type. SymInt values can arrive rank-0 (e.g. from
    ``aten._local_scalar_dense``) or with a different int variant. This
    helper produces the form that aligns with ``coreai.constant([i],
    dtype=np.int32)`` and ``replace_sym_size_int``.
    """
    if v.type.rank == 0:
        v = coreai.reshape(v, [1])
    if v.type.element_type != IntegerType.get_signed(32):
        v = coreai.cast(v, np.int32)
    return v


def upsample_build_output_shape_dynamic(
    x: Value, out_h: int | Value, out_w: int | Value
) -> Value:
    """Build 4D [N, C, H, W] output_shape as a runtime Value.

    Used when either input dims or output sizes are dynamic (runtime Values).
    """
    assert (
        any(d < 0 for d in x.type.shape)
        or isinstance(out_h, Value)
        or isinstance(out_w, Value)
    ), (
        "upsample_build_output_shape_dynamic called with fully static input and output; "
        "use a plain list [N, C, H, W] instead"
    )
    shape = coreai.cast(coreai.get_shape(x), dtype=np.int32)
    non_spatial = coreai.slice_(shape, [0], [2], [1])
    h = [out_h] if isinstance(out_h, int) else to_rank1_int32(out_h)
    w = [out_w] if isinstance(out_w, int) else to_rank1_int32(out_w)
    return coreai.concat(0, [non_spatial, h, w])


def upsample_halfpixel_scale(x: Value, out_h: int | Value, out_w: int | Value) -> Value:
    """Compute HalfPixel scale vector [1, 1, out_h/in_h, out_w/in_w] at runtime.
    Works for both static and dynamic input spatial dims.
    out_h / out_w can be static ints or runtime Values (shape [1]).
    """
    shape = coreai.get_shape(x)
    in_h_f32 = coreai.cast(coreai.slice_(shape, [2], [3], [1]), np.float32)
    in_w_f32 = coreai.cast(coreai.slice_(shape, [3], [4], [1]), np.float32)
    out_h_f32 = (
        coreai.cast(out_h, np.float32)
        if isinstance(out_h, Value)
        else coreai.constant(np.array([float(out_h)], dtype=np.float32))
    )
    out_w_f32 = (
        coreai.cast(out_w, np.float32)
        if isinstance(out_w, Value)
        else coreai.constant(np.array([float(out_w)], dtype=np.float32))
    )
    scale_h = coreai.broadcasting_divide(out_h_f32, in_h_f32)
    scale_w = coreai.broadcasting_divide(out_w_f32, in_w_f32)
    return coreai.concat(0, [[1.0, 1.0], scale_h, scale_w])


def upsample_align_corners_scale_offset(
    x: Value, out_h_f32: Value, out_w_f32: Value
) -> tuple[Value, Value]:
    """Compute AlignCorners scale/offset vectors at runtime.
    scale_h = (out_h - 1) / (in_h - 1), offset_h = 0.5 * (1 - scale_h).
    Works for both static and dynamic input spatial dims.
    out_h_f32 / out_w_f32 are float32 rank-1 Values (shape [1]).
    """
    shape = coreai.get_shape(x)
    in_h_f32 = coreai.cast(coreai.slice_(shape, [2], [3], [1]), np.float32)
    in_w_f32 = coreai.cast(coreai.slice_(shape, [3], [4], [1]), np.float32)
    one = coreai.constant(np.array([1.0], dtype=np.float32))
    half = coreai.constant(np.array([0.5], dtype=np.float32))
    scale_h = coreai.broadcasting_divide(
        coreai.broadcasting_sub(out_h_f32, one),
        coreai.broadcasting_sub(in_h_f32, one),
    )
    scale_w = coreai.broadcasting_divide(
        coreai.broadcasting_sub(out_w_f32, one),
        coreai.broadcasting_sub(in_w_f32, one),
    )
    offset_h = coreai.broadcasting_mul(half, coreai.broadcasting_sub(one, scale_h))
    offset_w = coreai.broadcasting_mul(half, coreai.broadcasting_sub(one, scale_w))
    return (
        coreai.concat(0, [[1.0, 1.0], scale_h, scale_w]),
        coreai.concat(0, [[0.0, 0.0], offset_h, offset_w]),
    )


def upsample_runtime_output_hw_from_scale_dynamic(
    x: Value, scale_h_f: float, scale_w_f: float
) -> tuple[Value, Value, Value, Value]:
    """Compute output H/W at runtime by multiplying input dims by scale factors.

    Returns (out_h_f32, out_w_f32, out_h_int, out_w_int) — all rank-1 Values.
    Must only be called when the spatial dims of x are dynamic. For static
    spatial dims, compute output H/W in Python directly as int(input_h * scale).
    """
    assert x.type.shape[2] < 0 or x.type.shape[3] < 0, (
        "upsample_runtime_output_hw_from_scale_dynamic called on input with static "
        "spatial dims; compute output H/W in Python instead"
    )
    shape = coreai.get_shape(x)
    in_h_f32 = coreai.cast(coreai.slice_(shape, [2], [3], [1]), np.float32)
    in_w_f32 = coreai.cast(coreai.slice_(shape, [3], [4], [1]), np.float32)
    out_h_f32 = coreai.broadcasting_mul(
        in_h_f32, coreai.constant(np.array([scale_h_f], dtype=np.float32))
    )
    out_w_f32 = coreai.broadcasting_mul(
        in_w_f32, coreai.constant(np.array([scale_w_f], dtype=np.float32))
    )
    out_h_int = coreai.cast(out_h_f32, IntegerType.get_signed(32))
    out_w_int = coreai.cast(out_w_f32, IntegerType.get_signed(32))
    return out_h_f32, out_w_f32, out_h_int, out_w_int


def get_promoted_type(type1: RankedTensorType, type2: RankedTensorType) -> Type:
    """Return the promoted element type for two tensor types using torch promotion rules."""
    m = _get_coreai_to_torch_dtype()
    return TORCH_TO_COREAI_DTYPE[
        torch.promote_types(m[type1.element_type], m[type2.element_type])
    ]()


# Narrow int64/fp64 to int32/fp32 since coreai does not handle 64-bit types.
_NARROW_TORCH_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.int64: torch.int32,
    torch.float64: torch.float32,
}


def get_tensor_type(
    tensor: Tensor, loc: Location | None = None, dtype: torch.dtype | None = None
) -> RankedTensorType:
    """Convert a torch.Tensor to a RankedTensorType.

    int64 and float64 are automatically narrowed to int32 and float32 respectively,
    since coreai does not handle 64-bit types well.

    Float4Tensor packs 2 fp4 values per uint8 byte.  After ``torch.export`` the
    subclass is lost and the tensor has ``dtype=uint8`` with packed shape.  When
    the caller passes the real FP4 dtype (via ``dtype`` or ``future_dtype``), we
    double the last dimension to recover the logical shape — matching Core AI's
    ``get_tensor_type`` in ``importer/torch/_components/types.py``.
    """
    effective_dtype = (
        dtype if dtype is not None else getattr(tensor, "future_dtype", tensor.dtype)
    )
    if isinstance(effective_dtype, str):
        effective_dtype = getattr(torch, effective_dtype)
    effective_dtype = _NARROW_TORCH_DTYPE.get(effective_dtype, effective_dtype)

    # Detect packed FP4: caller overrode dtype to float4_e2m1fn_x2 but the
    # tensor's physical dtype is still uint8 (packed, 2 values per byte).
    is_packed_fp4 = (
        effective_dtype == torch.float4_e2m1fn_x2 and tensor.dtype == torch.uint8
    )

    dims = tensor.size()
    shape = []
    for idx, s in enumerate(dims):
        dim = ShapedType.get_dynamic_size() if isinstance(s, torch.SymInt) else s
        # Packed FP4: double the last dimension to get logical element count.
        if is_packed_fp4 and idx == len(dims) - 1 and not isinstance(s, torch.SymInt):
            dim = 2 * dim
        shape.append(dim)

    return RankedTensorType.get(
        shape,
        TORCH_TO_COREAI_DTYPE[effective_dtype](),
        loc=loc,
    )


def get_result_types(node: fx.Node) -> list[RankedTensorType]:
    """Return result types for an FX node.

    Handles both single-result nodes (``node.meta["val"]`` is a Tensor)
    and multi-result nodes (``node.meta["val"]`` is a list/tuple of Tensors).
    """
    val = node.meta["val"]
    if isinstance(val, (list, tuple)):
        return [get_tensor_type(v) for v in val]
    return [get_tensor_type(val)]


def check_result_type(result: Value, expected: object, node: fx.Node, idx: int) -> None:
    if not isinstance(expected, Tensor):
        return
    actual = result.type
    if not isinstance(actual, RankedTensorType):
        raise ValueError(f"{node.name}[{idx}]: expected RankedTensorType, got {actual}")

    # coreai cannot handle int64 / fp64 well
    narrow_expected_dtypes = []
    if expected.dtype == torch.int64:
        narrow_expected_dtypes = [
            TORCH_TO_COREAI_DTYPE[torch.int64](),
            TORCH_TO_COREAI_DTYPE[torch.int32](),
            TORCH_TO_COREAI_DTYPE[torch.uint32](),
        ]
    elif expected.dtype == torch.float64:
        narrow_expected_dtypes = [
            TORCH_TO_COREAI_DTYPE[torch.float64](),
            TORCH_TO_COREAI_DTYPE[torch.float32](),
        ]
    elif expected.dtype == torch.complex64:
        # After f16 casting, view_as_complex produces complex<f16> (complex32).
        # Accept both, analogous to float64 -> float32 above.
        narrow_expected_dtypes = [
            TORCH_TO_COREAI_DTYPE[torch.complex64](),
            TORCH_TO_COREAI_DTYPE[torch.complex32](),
        ]

    expected_type = get_tensor_type(expected)
    if actual.rank != expected_type.rank:
        raise ValueError(
            f"{node.name}[{idx}]: rank {actual.rank} vs {expected_type.rank}"
        )

    if len(narrow_expected_dtypes) == 0:
        if actual.element_type != expected_type.element_type:
            raise ValueError(
                f"{node.name}[{idx}]: dtype {actual.element_type} vs {expected_type.element_type}"
            )
    else:
        if not any([actual.element_type == val for val in narrow_expected_dtypes]):
            raise ValueError(
                f"{node.name}[{idx}]: dtype {actual.element_type} vs {narrow_expected_dtypes}"
            )

    dyn = ShapedType.get_dynamic_size()
    for i, (a, e) in enumerate(zip(actual.shape, expected_type.shape)):
        if a != dyn and e != dyn and a != e:
            raise ValueError(f"{node.name}[{idx}]: dim[{i}] {a} vs {e}")


def get_target(node: fx.Node) -> str:
    """Return the target name from an FX node."""
    return node.target.__name__ if callable(node.target) else str(node.target)


def get_namespace(node: fx.Node) -> str | None:
    """Return the namespace of the FX node's target, or None if it has none."""
    if callable(node.target) and hasattr(node.target, "namespace"):
        return str(node.target.namespace)
    return None


def strip_variant_from_target(target: str) -> str:
    """Strip a known variant suffix from an op target string (e.g. "add.Tensor" -> "add")."""
    if "." not in target:
        return target
    op_name, variant = target.split(".", 1)
    if variant in {"default", "Tensor", "Scalar", "dim"}:
        return op_name
    return target


def prepare_compute_type_for_norm(
    input_value: Value,
    input_ele_type: Type,
    loc: Location | None = None,
) -> tuple[Value, Type, bool]:
    """Upcast fp16 input to fp32 for numerical stability in norm ops; pass other types through."""
    if input_ele_type == F16Type.get():
        fp32 = F32Type.get()
        return coreai.cast(input_value, fp32, loc=loc), fp32, True
    return input_value, input_ele_type, False


def get_output_element_type_from_node(node: fx.Node, index: int | None = None) -> Type:
    """Return the element type for a node's output.

    Handles Tensor, SymInt, SymFloat, SymBool, and plain Python scalar meta-values.
    int64 and float64 are narrowed to int32 and float32.
    """
    val = node.meta["val"]
    if index is not None:
        val = val[index]

    if isinstance(val, torch.Tensor):
        dtype = val.dtype
    elif isinstance(val, (float, torch.SymFloat)):
        dtype = torch.float32
    elif isinstance(val, (int, torch.SymInt)):
        dtype = torch.int32
    elif isinstance(val, (bool, torch.SymBool)):
        dtype = torch.bool
    else:
        dtype = val.dtype  # fall back to original behaviour

    dtype = _NARROW_TORCH_DTYPE.get(dtype, dtype)
    return TORCH_TO_COREAI_DTYPE[dtype]()


@dataclass
class _StackedIndexInfo:
    """Stacked indices and permutations for gather_nd/scatter_nd.

    The three permutation fields are None when no base transposition is needed.

    Attributes:
        result_permutation: Permutation to reorder gather_nd output to match PyTorch's expected layout.
        inverse_result_permutation: Inverse of result_permutation; used for index_put scatter updates.
        base_inverse_permutation: Permutation to restore base to its original axis order after scatter_nd.
    """

    base: Value
    stacked_indices: Value
    result_permutation: Value | None
    inverse_result_permutation: Value | None
    base_inverse_permutation: Value | None


def _are_indices_contiguous(indices: list[Value | None]) -> bool:
    """Return True if the non-None entries in indices form a contiguous span with no gaps."""
    positions = [i for i, idx in enumerate(indices) if idx is not None]
    return len(positions) <= 1 or positions[-1] - positions[0] + 1 == len(positions)


def _stack_indices(index_tensors: list[Value], loc: Location) -> Value:
    """Unsqueeze each index tensor along a new trailing dim, then concat to form [..., num_indices]."""
    if not index_tensors:
        raise ValueError("Cannot stack empty list of indices")
    rank = index_tensors[0].type.rank
    return coreai.concat(
        rank,
        [coreai.expand_dims(idx, [rank], loc=loc) for idx in index_tensors],
        loc=loc,
    )


def _invert_perm(perm: list[int]) -> list[int]:
    """Return the inverse permutation of perm (i.e. inv[perm[i]] == i)."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def to_uint32_perm(
    perm: Value | list[int] | tuple[int, ...],
    rank: int,
) -> Value:
    """Resolve negative indices and return a uint32 permutation for TransposeOp."""
    if isinstance(perm, Value):
        elem = RankedTensorType(perm.type).element_type
        if isinstance(elem, IntegerType) and elem.is_unsigned:
            return perm
        # Resolve negative indices: (perm + rank) % rank turns e.g. -1 → rank-1.
        rank_val = coreai.constant([rank], dtype=np.int32)
        resolved = coreai.broadcasting_modulo(
            coreai.broadcasting_add(perm, rank_val), rank_val
        )
        return coreai.cast(resolved, np.uint32)

    resolved = [p if p >= 0 else p + rank for p in perm]
    return coreai.constant(resolved, dtype=np.uint32)


def _transpose_to_front_and_stack(
    base: Value,
    indices: list[Value | None],
    loc: Location,
    require_transpose: bool,
) -> _StackedIndexInfo:
    """Move indexed dimensions to the front of base, then stack the index tensors.

    Example (no result permutation needed):
        base=[3,4,5,6], indices=[None, idx1, None, idx2]  (indexed dims: 1, 3)
        perm=[1,3,0,2] -> transposed_base=[4,6,3,5]
        base_inverse_perm=[2,0,3,1]  (restores original order after scatter_nd)

    Example (result permutation needed — contiguous indices not starting at dim 0):
        base=[3,4,5,6], indices=[None, None, idx2, idx3]  (indexed dims: 2, 3)
        gather_nd output: [broadcast, d0, d1]
        PyTorch expects:  [d0, d1, broadcast]
        result_perm=[1,2,0], inverse_result_perm=[2,0,1]
    """
    non_none_indices = [idx for idx in indices if idx is not None]
    indexed_dims = [i for i, idx in enumerate(indices) if idx is not None]

    if not non_none_indices:
        raise ValueError("Cannot transpose with no non-None indices")

    if not require_transpose:
        return _StackedIndexInfo(
            base=base,
            stacked_indices=_stack_indices(non_none_indices, loc),
            result_permutation=None,
            inverse_result_permutation=None,
            base_inverse_permutation=None,
        )

    non_indexed_dims = [i for i in range(len(indices)) if i not in indexed_dims]
    remaining_dims = list(range(len(indices), base.type.rank))
    perm = indexed_dims + non_indexed_dims + remaining_dims
    perm_val = to_uint32_perm(perm, base.type.rank)
    transposed_base = coreai.transpose(base, perm_val, loc=loc)
    base_inv_perm = to_uint32_perm(_invert_perm(perm), base.type.rank)

    # For contiguous indices not starting at dim 0, gather_nd places the broadcast
    # dims first but PyTorch expects them at the first indexed position.
    if _are_indices_contiguous(indices) and indexed_dims[0] > 0:
        B = non_none_indices[0].type.rank  # broadcast rank
        num_before = sum(1 for d in non_indexed_dims if d < indexed_dims[0])
        N = len(non_indexed_dims)
        result_perm = (
            list(range(B, B + num_before))  # non_indexed before
            + list(range(B))  # broadcast
            + list(range(B + num_before, B + N))  # non_indexed after
            + list(range(B + N, B + N + len(remaining_dims)))
        )
        return _StackedIndexInfo(
            base=transposed_base,
            stacked_indices=_stack_indices(non_none_indices, loc),
            result_permutation=to_uint32_perm(result_perm, len(result_perm)),
            inverse_result_permutation=to_uint32_perm(
                _invert_perm(result_perm), len(result_perm)
            ),
            base_inverse_permutation=base_inv_perm,
        )

    return _StackedIndexInfo(
        base=transposed_base,
        stacked_indices=_stack_indices(non_none_indices, loc),
        result_permutation=None,
        inverse_result_permutation=None,
        base_inverse_permutation=base_inv_perm,
    )


def process_expanded_indices(
    base: Value,
    indices: list[Value | None],
    loc: Location,
) -> _StackedIndexInfo:
    """Broadcast already-expanded indices to a common shape, then transpose and stack.

    Unlike process_indices_with_transpose, indices are Core AI Values (not FX nodes) and
    may have dynamic shapes (e.g. from boolean mask expansion via non_zero).
    Dynamic-shape indices are assumed to already match and are passed through without
    broadcasting.
    """
    non_none_indices = [idx for idx in indices if idx is not None]
    if not non_none_indices:
        raise ValueError("Index operation requires at least one non-None index tensor")

    # Align ranks by prepending leading dims to lower-rank indices.
    max_rank = max(idx.type.rank for idx in non_none_indices)
    aligned = [
        coreai.expand_dims(idx, list(range(max_rank - idx.type.rank)), loc=loc)
        if idx.type.rank < max_rank
        else idx
        for idx in non_none_indices
    ]

    # For static shapes, broadcast to the common shape; dynamic shapes already match.
    if not any(dim < 0 for dim in aligned[0].type.shape):
        broadcast_shape = coreai.constant(
            list(np.broadcast_shapes(*[idx.type.shape for idx in aligned])), loc=loc
        )
        aligned = [
            coreai.broadcast_to(idx, broadcast_shape, loc=loc) for idx in aligned
        ]

    # Reattach None slots.
    it = iter(aligned)
    broadcasted_indices = [next(it) if idx is not None else None for idx in indices]

    contiguous = _are_indices_contiguous(broadcasted_indices)
    first_indexed_dim = next(
        i for i, idx in enumerate(broadcasted_indices) if idx is not None
    )

    if contiguous and first_indexed_dim > 0:
        broadcasted_indices = _fill_leading_nones_with_ranges(
            base, broadcasted_indices, loc
        )
        require_transpose = False
    else:
        require_transpose = not (contiguous and first_indexed_dim == 0)

    return _transpose_to_front_and_stack(
        base, broadcasted_indices, loc, require_transpose
    )


def expand_boolean_indices(
    base: Value,
    values_map: dict[str, Value],
    indices_arg: list | tuple,
    loc: Location,
) -> list[Value | None]:
    """Convert indices to int32 Values (or None), expanding boolean masks via nonzero.

    Example: base=[10,20], indices=[bool_mask(shape=[10,20])]
      nonzero(mask) -> [N,2]; extract columns -> [idx0[N], idx1[N]]
    """
    base_shape = base.type.shape
    expanded: list[Value | None] = []
    INT32_MAX = 2**31 - 1

    for idx_node in indices_arg:
        if idx_node is None:
            expanded.append(None)
            continue

        assert isinstance(idx_node, fx.Node), f"Expected fx.Node, got {type(idx_node)}"
        idx_val = values_map[idx_node.name]

        # Integer index: ensure int32.
        if idx_node.meta["val"].dtype != torch.bool:
            if idx_val.type.element_type != IntegerType.get_signed(32):
                idx_val = coreai.cast(idx_val, IntegerType.get_signed(32), loc=loc)
            expanded.append(idx_val)
            continue

        # Boolean mask: validate each dimension against base, then extract columns via nonzero.
        idx_shape = idx_node.meta["val"].shape
        base_offset = len(expanded)
        for d, size in enumerate(idx_shape):
            dim = base_offset + d
            if dim >= len(base_shape):
                raise ValueError(
                    f"Boolean index at position {base_offset} has {len(idx_shape)} dimensions, "
                    f"but base tensor only has {len(base_shape)} dimensions"
                )
            if (
                base_shape[dim] >= 0
                and isinstance(size, int)
                and base_shape[dim] != size
            ):
                raise ValueError(
                    f"Boolean mask shape {idx_shape} does not match base shape "
                    f"{base_shape} at dim {dim}: expected {base_shape[dim]}, got {size}"
                )
            # nonzero returns [num_true, num_dims]; slice column d -> [num_true, 1] -> [num_true]
            nz = coreai.non_zero(idx_val, loc=loc)
            column = coreai.slice_(
                nz,
                coreai.constant([0, d], loc=loc),
                coreai.constant([INT32_MAX, d + 1], loc=loc),
                coreai.constant([1, 1], loc=loc),
                loc=loc,
            )
            expanded.append(
                coreai.shrink_dims(column, coreai.constant([1], loc=loc), loc=loc)
            )

    return expanded


def _fill_leading_nones_with_ranges(
    base: Value,
    indices: Sequence[Value | None],
    loc: Location,
) -> list[Value | None]:
    """Fill leading None index slots with range tensors so gather_nd/scatter_nd can
    operate on the untransposed base.
    For each leading None at position i, creates range(0, base.shape[i]) reshaped to
    [dim_size, 1, 1, ...] so it broadcasts with the actual index tensors.
    After filling, all indices are broadcast to a common shape.
    """
    num_leading_nones = next((i for i, idx in enumerate(indices) if idx is not None), 0)
    if num_leading_nones == 0:
        return indices

    non_none = [idx for idx in indices if idx is not None]
    max_index_rank = max(idx.type.rank for idx in non_none)
    indices = list(indices)

    for i in range(num_leading_nones):
        dim_size = base.type.shape[i]

        if dim_size >= 0:
            dim_range = coreai.constant(list(range(dim_size)), dtype=np.int32)
        else:
            dim_len_ui32 = coreai.slice_(
                coreai.get_shape(base),
                [i],
                [i + 1],
                [1],
            )
            dim_len = coreai.cast(
                coreai.shrink_dims(dim_len_ui32, [0]),
                np.int32,
            )
            dim_range = coreai.range_(
                coreai.constant(0, dtype=np.int32),
                dim_len,
                coreai.constant(1, dtype=np.int32),
            )

        # Reshape to [dim_size, 1, 1, ...] for broadcasting.
        num_trailing_ones = (num_leading_nones - i - 1) + max_index_rank
        if num_trailing_ones > 0:
            dim_range = coreai.reshape(dim_range, [-1] + [1] * num_trailing_ones)

        indices[i] = dim_range

    # Broadcast all indices (ranges + originals) to a common shape.
    all_non_none = [idx for idx in indices if idx is not None]
    target_rank = max(idx.type.rank for idx in all_non_none)
    aligned = [
        coreai.expand_dims(idx, list(range(target_rank - idx.type.rank)))
        if idx.type.rank < target_rank
        else idx
        for idx in all_non_none
    ]

    shapes = [tuple(idx.type.shape) for idx in aligned]
    if not any(d < 0 for s in shapes for d in s):
        bcast_shape: list[int] | Value = list(np.broadcast_shapes(*shapes))
    else:
        shape_tensors = [
            coreai.cast(coreai.get_shape(idx), np.uint32) for idx in aligned
        ]
        bcast_shape = shape_tensors[0]
        for s in shape_tensors[1:]:
            bcast_shape = coreai.broadcast_shapes(bcast_shape, s)

    broadcasted = [coreai.broadcast_to(idx, bcast_shape) for idx in aligned]

    it = iter(broadcasted)
    return [next(it) if idx is not None else None for idx in indices]


def process_indices_with_transpose(
    base: Value,
    values_map: dict[str, Value],
    indices_arg: list | tuple,
    loc: Location,
) -> _StackedIndexInfo:
    """Cast FX-node indices to int32, broadcast to a common shape, then transpose and stack.

    Example: base=[3,4,5,6], indices=[None, None, node1, node2]
      broadcast shape (3,2); transpose base -> [5,6,3,4]; inverse perm [2,3,0,1]
    """
    assert isinstance(indices_arg, (list, tuple)), (
        f"Expected list/tuple of indices, got {type(indices_arg)}"
    )

    # Cast each FX node to an int32 Value; keep None slots.
    indices: list[Value | None] = [
        None
        if idx_node is None
        else coreai.cast(values_map[idx_node.name], IntegerType.get_signed(32), loc=loc)
        for idx_node in indices_arg
    ]

    non_none_indices = [idx for idx in indices if idx is not None]
    if not non_none_indices:
        raise ValueError("Index operation requires at least one non-None index tensor")

    # Check contiguity early so we can skip the pre-broadcast when
    # _fill_leading_nones_with_ranges will handle broadcasting for us.
    contiguous = _are_indices_contiguous(indices)
    first_indexed_dim = next(i for i, idx in enumerate(indices) if idx is not None)

    if contiguous and first_indexed_dim > 0:
        # Defer broadcasting to _fill_leading_nones_with_ranges which will
        # expand all indices (ranges + originals) to the full target rank and
        # broadcast in one step.  This avoids an intermediate broadcast that
        # collapses shape information (e.g. 1024x1 → 1024x1024) followed by
        # a reshape to re-add leading 1-dims.
        broadcasted_indices = _fill_leading_nones_with_ranges(base, list(indices), loc)
        return _transpose_to_front_and_stack(
            base, broadcasted_indices, loc, require_transpose=False
        )

    # Align ranks then broadcast all indices to the common shape.
    # Dynamic dims (negative values) are propagated manually since
    # np.broadcast_shapes rejects them.
    def _broadcast_shapes_with_dynamic(*shapes: tuple[int, ...]) -> list[int]:
        max_rank = max(len(s) for s in shapes)
        padded = [(1,) * (max_rank - len(s)) + tuple(s) for s in shapes]
        result: list[int] = []
        for dims in zip(*padded):
            if any(d < 0 for d in dims):
                result.append(-1)
            else:
                result.append(int(np.broadcast_shapes(*[(d,) for d in dims])[0]))
        return result

    broadcast_shape = _broadcast_shapes_with_dynamic(
        *[tuple(idx.type.shape) for idx in non_none_indices]
    )
    target_rank = len(broadcast_shape)
    # Use static list for known shapes; use runtime broadcast_shapes for dynamic dims.
    # (coreai.constant([-1]) would produce UINT32_MAX after si32→ui32 cast.)
    if not any(d < 0 for d in broadcast_shape):
        shape_arg: list[int] | Value = broadcast_shape
    else:
        # Compute broadcast shape at runtime via coreai.broadcast_shapes.
        shape_tensors = [coreai.get_shape(idx, loc=loc) for idx in non_none_indices]
        shape_arg = shape_tensors[0]
        for s in shape_tensors[1:]:
            shape_arg = coreai.broadcast_shapes(shape_arg, s, loc=loc)

    broadcasted = [
        coreai.broadcast_to(
            coreai.expand_dims(idx, list(range(target_rank - idx.type.rank)), loc=loc)
            if idx.type.rank < target_rank
            else idx,
            shape_arg,
            loc=loc,
        )
        for idx in non_none_indices
    ]

    # Reattach None slots.
    it = iter(broadcasted)
    broadcasted_indices = [next(it) if idx is not None else None for idx in indices]

    require_transpose = not (contiguous and first_indexed_dim == 0)

    return _transpose_to_front_and_stack(
        base, broadcasted_indices, loc, require_transpose
    )


def get_invoke_from_graph(
    values_map: dict[str, Value], node: fx.Node, loc: Location, graph_op: coreai.GraphOp
) -> list[OpResult]:
    """Return the results of a coreai.invoke call targeting graph_op."""
    operands = [
        get_operand(values_map, node, i, loc)
        for i, arg in enumerate(node.args)
        if arg is not None
    ]
    result = coreai.invoke(
        results=[output.type for output in graph_op.outputs],
        callee=graph_op.symbol_name,
        operands=operands,
        loc=loc,
    )
    return list(result)


def resolve_slice_arg(
    raw: Any, default_val: int, values_map: dict[str, Value]
) -> int | Value:
    """Resolve a raw slice argument (start, end, or stride) from node.args to a static int or dynamic IR Value."""
    if raw is None:
        return default_val
    if isinstance(raw, fx.Node):
        return values_map[raw.name]
    if isinstance(raw, torch.SymInt):
        raise ValueError(
            f"Symbolic SymInt slice argument is not supported: {raw!r}. "
            "Use fx.Node references (e.g. results of aten.sym_size.int)."
        )
    val = int(raw)
    SLICE_INT32_MAX: int = 2**31 - 1

    # ATen uses INT64_MAX (~9.2e18) to mean "slice to end". Core AI indices are
    # si32, so values above INT32_MAX overflow to negative (e.g. INT64_MAX → -1),
    # causing coreai.slice_ to compute a wrong output shape. Clamp to INT32_MAX.
    return min(val, SLICE_INT32_MAX)


def build_slice_index_array(
    rank: int, dim: int, default_val: int, value: int | Value
) -> list[int] | Value:
    """Build a rank-length 1-D index array with default_val everywhere except at dim."""
    if not isinstance(value, Value):
        result = [default_val] * rank
        result[dim] = value
        return coreai.constant(result)

    val_1d = coreai.reshape(value, [1]) if value.type.rank == 0 else value
    parts: list[Value] = []
    if dim > 0:
        parts.append([default_val] * dim)
    parts.append(val_1d)
    if dim < rank - 1:
        parts.append([default_val] * (rank - 1 - dim))
    return coreai.concat(0, parts) if len(parts) > 1 else parts[0]


def _is_float_in_float16_range(val: float) -> bool:
    """Return True if val can be represented as float16 without precision loss."""
    casted_val: float = torch.tensor(val, dtype=torch.float16).item()
    eps = torch.finfo(torch.float16).eps
    return math.isclose(val, casted_val, rel_tol=eps, abs_tol=eps)


def _all_float_operands_are_fp16(node: fx.Node) -> bool:
    """Return True if every float tensor arg of node is fp16.

    Non-tensor and non-float args are ignored. Returns True when there
    are no float tensor args.
    """
    for arg in (*node.args, *node.kwargs.values()):
        if isinstance(arg, fx.Node):
            dtype = getattr(arg.meta.get("val"), "dtype", None)
        elif isinstance(arg, torch.nn.Parameter | torch.Tensor):
            dtype = arg.dtype
        else:
            continue
        if dtype is not None and dtype.is_floating_point and dtype != torch.float16:
            return False
    return True


def get_operand(
    values_map: dict[str, Value],
    node: fx.Node,
    idx: int,
    loc: Location | None = None,
) -> Value:
    """Return the Core AI Value for node.args[idx], converting scalars/tensors/lists to constants.

    When arg is a Python float and every float tensor operand of the node is fp16,
    the scalar is promoted to an fp16 constant (provided no precision loss) to keep
    operand types consistent.
    """
    assert 0 <= idx < len(node.args), (
        f"get_operand: idx {idx} out of range for node {node} with {len(node.args)} args"
    )
    arg: Argument = node.args[idx]
    if isinstance(arg, fx.Node):
        return values_map[arg.name]
    if isinstance(arg, list) and any(isinstance(e, fx.Node) for e in arg):
        # Mixed list: SymInt fx.Nodes + plain ints. Concat the two sources
        # into a single rank-1 si32 dim vector. Both branches must produce
        # the same canonical form so the concat verifier accepts them.
        dim_vals = [
            to_rank1_int32(values_map[e.name])
            if isinstance(e, fx.Node)
            else coreai.constant([e], dtype=np.int32)
            for e in arg
        ]
        return coreai.concat(0, dim_vals) if len(dim_vals) > 1 else dim_vals[0]
    if isinstance(arg, float) and not isinstance(arg, bool):
        if _all_float_operands_are_fp16(node) and _is_float_in_float16_range(arg):
            data = torch.tensor(arg, dtype=torch.float16).numpy()
            return coreai.constant(data)
        return coreai.constant(arg)
    if isinstance(arg, bool | int | float | Tensor | list):
        data = arg.detach().cpu().numpy() if isinstance(arg, Tensor) else arg
        return coreai.constant(data)
    raise ValueError(f"Unsupported arg type {type(arg)} in node {node}: {arg}")


def get_operands(
    values_map: dict[str, Value],
    node: fx.Node,
    indices: list[int],
    loc: Location | None = None,
) -> list[Value]:
    """Get multiple operand Values from an FX node's arguments by index."""
    return [get_operand(values_map, node, i, loc) for i in indices]


def build_shape_tensor(
    values_map: dict[str, Value],
    shape: list[int | fx.Node],
) -> Value:
    """Build a 1-D int32 shape tensor from a mix of static ints and dynamic fx.Nodes.

    Dynamic values are cast to int32 to ensure a uniform element type for
    ``coreai.concat``.  Callers that feed the result into ``BroadcastToOp``
    should cast to uint32 themselves.
    """
    dim_vals: list[Value | list[int]] = [
        coreai.cast(values_map[s.name], np.int32) if isinstance(s, fx.Node) else [s]
        for s in shape
    ]
    return coreai.concat(0, dim_vals) if len(dim_vals) > 1 else dim_vals[0]


class _ModuleInstanceRegistry:
    """Assign stable per-type instance counts to module instances.

    Each module instance name is assigned a count the first time it is seen for
    a given module type. Later lookups for the same instance return the same
    count.

    Attributes:
        module_type_to_next_count: Maps a module type name, such as "Linear",
            to the next instance count to assign for that type.
        module_instance_to_count: Maps a concrete module instance name, such as
            "encoder.layers.0.self_attn", to its assigned per-type count.
    """

    def __init__(self) -> None:
        """Initialize empty module bookkeeping state."""
        self.module_type_to_next_count: dict[str, int] = {}
        self.module_instance_to_count: dict[str, int] = {}

    def get_instance_count(
        self,
        module_instance_name: str,
        module_type: str,
    ) -> int:
        """Return the stable count assigned to a module instance.

        If the instance was already seen, return its existing count. Otherwise,
        assign the next count for the given module type, store it, and return it.

        Args:
            module_instance_name: Unique module instance identifier.
            module_type: Module type name, for example "Linear".

        Returns:
            The stable per-type instance count for this module instance.
        """
        existing_count = self.module_instance_to_count.get(module_instance_name)
        if existing_count is not None:
            return existing_count

        next_count = self.module_type_to_next_count.get(module_type, 0) + 1
        self.module_type_to_next_count[module_type] = next_count
        self.module_instance_to_count[module_instance_name] = next_count
        return next_count


def _get_module_hierarchy(
    node: torch.fx.Node,
    registry: _ModuleInstanceRegistry,
) -> list[str]:
    """Return module class names for this node with stable instance counts.

    Each module type name is suffixed with its per-type instance count in the
    form ``$<count>``. For example, ``Linear`` may become ``Linear$1``.
    """
    stack: dict[str, tuple[str, str]] = node.meta.get("nn_module_stack", {})
    result: list[str] = []

    for module_instance_name, (_, type_str) in stack.items():
        module_type = type_str.split(".")[-1]
        instance_count = registry.get_instance_count(
            module_instance_name=module_instance_name,
            module_type=module_type,
        )
        result.append(f"{module_type}${instance_count}")

    result.reverse()
    return result


@dataclass
class _TracebackEntry:
    """A single frame from a Python traceback (file, line number, method name)."""

    file_path: str
    line_number: int
    method: str


def parse_traceback(traceback_str: str) -> list[_TracebackEntry]:
    """Parse a Python traceback string; innermost frame is last."""
    traceback_re = re.compile(r'\s*File "(.*)", line (\d+), in (.+)')
    return [
        _TracebackEntry(
            file_path=m.group(1), line_number=int(m.group(2)), method=m.group(3)
        )
        for line in traceback_str.strip().split("\n")
        if (m := traceback_re.match(line))
    ]


def preprocess_graph(graph_module: fx.GraphModule) -> fx.GraphModule:
    """Remove assertion nodes from graph_module and eliminate dead code."""
    assert_ops = {
        torch.ops.aten._assert_async.msg,
        torch.ops.aten._assert_scalar.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten.sym_constrain_range.default,
        torch.ops.aten._assert_tensor_metadata.default,
    }
    for node in list(graph_module.graph.nodes):
        if node.op == "call_function" and node.target in assert_ops:
            graph_module.graph.erase_node(node)
    graph_module.recompile()
    graph_module.graph.eliminate_dead_code()
    return graph_module


def convert_branch_subgraph(
    branch_module: fx.GraphModule,
    operand_values: list[Value],
    graph_module: fx.GraphModule,
    aten_resolver: dict[str, Callable[..., Any]],
    higher_order_handlers: dict[str, Callable[..., list[Value]]],
) -> list[Value]:
    """Convert one branch of a torch.cond/while_loop subgraph into a list of Core AI Values.

    A torch.cond or while_loop node stores its branches as separate GraphModules
    whose nodes need to be converted into Core AI ops inside a region.
    This function walks one such branch GraphModule and maps every node to its
    Core AI equivalent using the provided resolvers.

    Example:
        Given a model::

            def forward(x):
                return torch.cond(x.sum() > 0, lambda x: x * 2, lambda x: x * -1, [x])

        The true-branch GraphModule has three nodes::

            %x        : placeholder          # branch input
            %mul      : call_function        # aten.mul.Scalar(%x, 2)
            %output   : output((%mul,))

        convert_branch_subgraph maps them as:
            placeholder %x   -> operand_values[0]   (the outer Core AI Value for x)
            call_function    -> coreai.broadcasting_mul via aten_resolver
            output           -> collects [coreai_mul_value]

        The returned list is then passed directly to coreai.yield_.
    """
    branch_values: dict[str, Value] = {}
    output_values: list[Value] = []
    operand_idx = 0

    for bnode in branch_module.graph.nodes:
        if bnode.op == "placeholder":
            branch_values[bnode.name] = operand_values[operand_idx]
            operand_idx += 1
        elif bnode.op == "call_function":
            btarget = get_target(bnode)
            bnamespace = get_namespace(bnode)
            if bnamespace is None or bnamespace == "aten":
                bresults = aten_resolver[btarget](
                    branch_values, bnode, Location.unknown()
                )
            elif bnamespace == "higher_order" and btarget in higher_order_handlers:
                bresults = higher_order_handlers[btarget](
                    branch_values, bnode, graph_module=graph_module
                )
            else:
                raise ValueError(
                    f"unsupported op in branch: {btarget} (namespace: {bnamespace})"
                )
            if not isinstance(bresults, (list, tuple)):
                bresults = [bresults]
            for i, r in enumerate(bresults):
                key = bnode.name if len(bresults) == 1 else f"{bnode.name}#{i}"
                branch_values[key] = r
        elif bnode.op == "get_attr":
            pass
        elif bnode.op == "output":
            # args[0] is a tuple of output nodes for multi-value returns, but a bare
            # fx.Node for single-value returns (e.g. while_loop's condition subgraph
            # returns a single bool tensor). Normalise to a tuple in both cases.
            raw = bnode.args[0]
            out_nodes = (raw,) if isinstance(raw, fx.Node) else tuple(raw)
            for out_node in out_nodes:
                if isinstance(out_node, fx.Node):
                    output_values.append(branch_values[out_node.name])
        else:
            raise ValueError(f"unsupported node op in branch: {bnode.op}")

    return output_values


def print_graph(ep: Any) -> None:
    """Print an exported program's graph with per-node inputs and output dtype/shape."""

    def _fmt_val(val: object) -> str:
        if isinstance(val, torch.Tensor):
            return f"{val.dtype} {list(val.shape)}"
        if isinstance(val, (list, tuple)):
            return "[" + ", ".join(_fmt_val(v) for v in val) + "]"
        return repr(val)

    def _fmt_arg(arg: object) -> str:
        if isinstance(arg, fx.Node):
            return arg.name
        if isinstance(arg, (list, tuple)):
            inner = [_fmt_arg(a) for a in arg]
            return "[" + ", ".join(inner) + "]"
        if isinstance(arg, (int, float, bool, str)):
            return repr(arg)
        return ""

    for node in ep.graph.nodes:
        val = node.meta.get("val")
        type_str = _fmt_val(val) if val is not None else ""
        args_str = ", ".join(s for a in node.args if (s := _fmt_arg(a)))
        print(f"{node.op:20} {node.name:40} ({args_str})  ->  {type_str}")


def _sdpa_flatten_leading_batch_dims(v: Value) -> Value:
    """Flatten [..., heads, seq, head_dim] to [B_flat, heads, seq, head_dim]."""
    rank = v.type.rank
    if rank == 4:
        return v
    shape = v.type.shape
    dyn = RankedTensorType.get_dynamic_size()

    # Best-effort static output type — any dynamic input dim stays dynamic.
    leading = shape[: rank - 3]
    if any(d < 0 for d in leading):
        b_static = dyn
    else:
        b_static = 1
        for d in leading:
            b_static *= d
    heads_s = shape[rank - 3] if shape[rank - 3] >= 0 else dyn
    seq_s = shape[rank - 2] if shape[rank - 2] >= 0 else dyn
    hdim_s = shape[rank - 1] if shape[rank - 1] >= 0 else dyn
    flat4d_type = RankedTensorType.get(
        [b_static, heads_s, seq_s, hdim_s], v.type.element_type
    )

    # Build the runtime reshape target from actual dimension values so that
    # dynamic dims are handled correctly.
    shape_val = coreai.get_shape(v)
    leading_shape = coreai.slice_(shape_val, [0], [rank - 3], [1])
    trailing_shape = coreai.slice_(shape_val, [rank - 3], [rank], [1])
    b_flat = coreai.reduce_product(leading_shape, [0])  # 1-element 1D tensor
    flat4d_shape = coreai.concat(0, [b_flat, trailing_shape])

    return coreai.ReshapeOp(v, flat4d_shape, results=[flat4d_type]).result


def _sdpa_build_causal_mask(query: Value, key: Value, ele_type: Any) -> Value:
    """Build a float additive causal mask of shape [q_len, k_len].

    Attended positions (q_idx >= k_idx) → 0.0, masked positions → -1e4.
    Built outside the composite op and passed as attn_mask.
    """
    q_static = query.type.shape[2]
    k_static = key.type.shape[2]
    dyn = RankedTensorType.get_dynamic_size()
    si32 = IntegerType.get_signed(32)

    # Use a static constant when the seq dim is known; otherwise extract at
    # runtime. Passing a static end value to range_ lets it produce a
    # statically-shaped result, which propagates known shape through all
    # downstream ops so the final mask type reflects [q_len, k_len].
    def _seq_len_scalar(v: Value, seq_dim_static: int) -> Value:
        if seq_dim_static > 0:
            return coreai.constant(seq_dim_static, dtype=np.int32)
        return coreai.cast(
            coreai.shrink_dims(coreai.slice_(coreai.get_shape(v), [2], [3], [1]), [0]),
            dtype=np.int32,
        )

    zero = coreai.constant(0, dtype=np.int32)
    one = coreai.constant(1, dtype=np.int32)
    q_end = _seq_len_scalar(query, q_static)
    k_end = _seq_len_scalar(key, k_static)

    q_indices = coreai.RangeOp(
        zero,
        q_end,
        one,
        results=[RankedTensorType.get([q_static if q_static > 0 else dyn], si32)],
    ).result  # [q_len]
    k_indices = coreai.RangeOp(
        zero,
        k_end,
        one,
        results=[RankedTensorType.get([k_static if k_static > 0 else dyn], si32)],
    ).result  # [k_len]

    q_indices_2d = coreai.expand_dims(q_indices, [1])  # [q_len, 1]
    k_indices_2d = coreai.expand_dims(k_indices, [0])  # [1, k_len]
    # causal_bool: q >= k — attended positions
    causal_bool = coreai.not_(coreai.broadcasting_greater(k_indices_2d, q_indices_2d))
    not_attended = coreai.not_(causal_bool)
    neg_large = coreai.constant(-1e4, dtype=ele_type)
    return coreai.broadcasting_mul(coreai.cast(not_attended, ele_type), neg_large)


def _sdpa_expand_heads(
    query: Value, kv: Value, n_q_heads: int, n_kv_heads: int, kv_shape: list
) -> Value:
    """Repeat-interleave kv heads along dim=1 by n_q_heads/n_kv_heads (GQA)."""
    factor = n_q_heads // n_kv_heads
    B, _, S, D = kv_shape
    dyn = RankedTensorType.get_dynamic_size()
    # unsqueeze at dim=2: [B, n_kv, 1, S, D]
    kv_unsq = coreai.expand_dims(kv, [2])
    # tile by factor along dim=2: [B, n_kv, factor, S, D]
    kv_tiled = coreai.tile(kv_unsq, np.array([1, 1, factor, 1, 1], dtype=np.uint32))
    # reshape to [B, n_q_heads, S, D]
    result_type = RankedTensorType.get(
        [B if B >= 0 else dyn, n_q_heads, S if S >= 0 else dyn, D if D >= 0 else dyn],
        kv.type.element_type,
    )
    # Build runtime reshape target — B, S, D may be dynamic so we cannot use
    # a static constant (multiple -1 entries in a reshape are invalid).
    kv_shape_val = coreai.get_shape(kv)
    B_1d = coreai.slice_(kv_shape_val, [0], [1], [1])
    S_1d = coreai.slice_(kv_shape_val, [2], [3], [1])
    D_1d = coreai.slice_(kv_shape_val, [3], [4], [1])
    nq_1d = coreai.constant([n_q_heads], dtype=np.uint32)
    reshape_target = coreai.concat(0, [B_1d, nq_1d, S_1d, D_1d])
    return coreai.ReshapeOp(kv_tiled, reshape_target, results=[result_type]).result


def _sdpa_decompose(
    query: Value,
    key: Value,
    value: Value,
    attn_mask: Value | None,
    scale: float | None,
    enable_gqa: bool,
    ele_type: Any,
    q_shape: list,
    k_shape: list,
    v_shape: list,
) -> Value:
    """SDPA decomposition body emitted inside the composite GraphOp."""
    import math

    # GQA: expand key/value heads to match query head count.
    if enable_gqa:
        n_q = q_shape[1]
        n_k = k_shape[1]
        n_v = v_shape[1]
        if n_q >= 0 and n_k >= 0 and n_q != n_k:
            key = _sdpa_expand_heads(query, key, n_q, n_k, k_shape)
        if n_q >= 0 and n_v >= 0 and n_q != n_v:
            value = _sdpa_expand_heads(query, value, n_q, n_v, v_shape)

    # Scale: compile-time constant or default 1/sqrt(head_dim).
    head_dim = q_shape[3]
    if scale is not None:
        scale_val = coreai.constant(float(scale), dtype=ele_type)
    elif head_dim > 0:
        scale_val = coreai.constant(1.0 / math.sqrt(float(head_dim)), dtype=ele_type)
    else:
        # Dynamic head_dim: compute at runtime.
        head_dim_1d = coreai.slice_(coreai.get_shape(query), [3], [4], [1])
        head_dim_float = coreai.cast(head_dim_1d, ele_type)
        scale_val = coreai.broadcasting_divide(
            coreai.constant(1.0, dtype=ele_type),
            coreai.sqrt(head_dim_float),
        )

    scaled_query = coreai.broadcasting_mul(query, scale_val)

    # Transpose key: [B, heads, head_dim, T_kv]
    key_t = coreai.transpose(key, to_uint32_perm([0, 1, 3, 2], key.type.rank))

    # attn_scores = matmul(scaled_query, key^T): [B, heads, q_len, k_len]
    attn_scores = coreai.broadcasting_batch_matmul(
        coreai.cast(scaled_query, ele_type),
        coreai.cast(key_t, ele_type),
    )

    # Apply attention mask: bool mask → -1e4 * ~mask; float mask → cast.
    if attn_mask is not None:
        if attn_mask.type.element_type == IntegerType.get_signless(1):
            float_mask = coreai.broadcasting_mul(
                coreai.cast(coreai.not_(attn_mask), ele_type),
                coreai.constant(-1e4, dtype=ele_type),
            )
        else:
            float_mask = coreai.cast(attn_mask, ele_type)
        attn_scores = coreai.broadcasting_add(attn_scores, float_mask)

    # Softmax over last dim, then weighted sum with value.
    attn_weights = coreai.softmax(attn_scores, attn_scores.type.rank - 1)
    result = coreai.broadcasting_batch_matmul(
        coreai.cast(attn_weights, ele_type),
        coreai.cast(value, ele_type),
    )
    assert result.type == query.type, "Result type and query type must be identical"
    return result


def get_tensor_shape_at_index(x: Value, idx: int):
    if x.type.shape[idx] > 0:
        return coreai.constant(x.type.shape[idx], dtype=np.uint32)
    shape = coreai.get_shape(x)

    begins = [idx]
    ends = [idx + 1]
    strides = [1]
    return coreai.slice_(shape, begins, ends, strides)


# ---------------------------------------------------------------------------
# Externalize helpers
# ---------------------------------------------------------------------------

_EXTERNALIZE_NAMESPACE = "coreai_torch_ext"
"""Fixed namespace used by all externalize helpers."""


def _sanitize_op_name(name: str) -> str:
    """Convert a dotted module path to a valid torch op name (no dots)."""
    return name.replace(".", "_")


def _resolve_name(
    model: torch.nn.Module,
    submodule: torch.nn.Module,
) -> str:
    """Find the dotted module path of *submodule* within *model*."""
    for name, mod in model.named_modules():
        if mod is submodule and name:
            return name
    raise ValueError(
        "submodule not found in model; "
        "pass a submodule obtained via model.<attr> (e.g. model.block.norm)"
    )


def _find_custom_op_node(
    exported_program: ExportedProgram,
    op_name: str,
) -> fx.Node | None:
    """Find the FX node for a custom op in the exported graph, or ``None``."""
    target_str = f"{_EXTERNALIZE_NAMESPACE}.{op_name}.default"
    for node in exported_program.graph.nodes:
        if node.op == "call_function" and target_str in str(node.target):
            return node
    return None


def _find_all_custom_op_nodes(
    exported_program: ExportedProgram,
    op_name: str,
) -> list[fx.Node]:
    """Return **all** FX nodes matching the custom op target, not just the first."""
    target_str = f"{_EXTERNALIZE_NAMESPACE}.{op_name}.default"
    return [
        node
        for node in exported_program.graph.nodes
        if node.op == "call_function" and target_str in str(node.target)
    ]


def _fake_inputs_from_node(node: fx.Node) -> tuple[torch.Tensor, ...]:
    """Extract concrete example tensors from a node's FakeTensor args.

    Creates fresh concrete tensors with the same shape/dtype rather than
    reusing FakeTensors from the parent graph.  This avoids two classes of
    ``torch.export`` failures when re-exporting the submodule in isolation:

    1. View metadata (e.g. storage_offset from ``.narrow()``) that carries
       derived symbolic expressions unresolvable outside the parent graph.
    2. SymInt objects bound to the parent ``ShapeEnv`` whose ``var_to_range``
       conflicts with guards generated by the standalone export.
    """
    inputs = []
    for i, arg in enumerate(node.args):
        if arg is None:
            continue  # skip unset optional tensor arguments
        val = arg.meta["val"]
        if not isinstance(val, torch.Tensor):
            raise TypeError(
                f"Expected argument {i} of custom op node '{node.target}' to be a "
                f"Tensor, but got {type(val).__name__}. Only Tensor inputs are "
                f"supported for externalized submodules."
            )
        concrete_shape = [int(s) for s in val.shape]
        inputs.append(torch.empty(concrete_shape, dtype=val.dtype, device=val.device))
    return tuple(inputs)


def _dim_for_sym(s: torch.SymInt, cache: dict[str, Dim]) -> Dim:
    """Get or create a Dim for a SymInt, reusing Dims for shared symbols."""
    key = str(s.node.expr)
    if key not in cache:
        # Dim names must be valid Python identifiers; derived expressions
        # like "s0 + s1" need sanitisation.
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", key)
        if not safe_name[0:1].isalpha() and not safe_name.startswith("_"):
            safe_name = f"d_{safe_name}"
        r = s.node.shape_env.var_to_range.get(s.node.expr)
        cache[key] = (
            Dim(safe_name, min=int(r.lower), max=int(r.upper))
            if r
            else Dim(safe_name, min=1)
        )
    return cache[key]


def _dynamic_shapes_from_node(node: fx.Node) -> tuple[dict[int, Dim] | None, ...]:
    """Reconstruct a positional dynamic_shapes tuple from a custom op node's FakeTensors."""
    cache: dict[str, Dim] = {}
    result = []
    for i, arg in enumerate(node.args):
        if arg is None:
            continue  # skip unset optional tensor arguments
        val = arg.meta["val"]
        if not isinstance(val, torch.Tensor):
            raise TypeError(
                f"Expected argument {i} of custom op node '{node.target}' to be a "
                f"Tensor, but got {type(val).__name__}. Only Tensor inputs are "
                f"supported for externalized submodules."
            )
        dims = {
            j: _dim_for_sym(s, cache)
            for j, s in enumerate(val.shape)
            if isinstance(s, torch.SymInt) and not s.node.expr.is_number
        }
        result.append(dims or None)
    return tuple(result)


def _default_decompose(ep: ExportedProgram) -> ExportedProgram:
    return ep.run_decompositions()


def _ancestor_paths(name: str) -> list[str]:
    """Return ancestor paths from nearest to root.

    ``"block.norm.scale"`` → ``["block.norm", "block", ""]``
    """
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(len(parts) - 1, 0, -1)] + [""]


def _find_program_for(
    name: str,
    op_name: str,
    programs: dict[str, ExportedProgram],
) -> ExportedProgram | None:
    """Find the nearest ancestor program that contains *op_name*, or ``None``."""
    for ancestor in _ancestor_paths(name):
        if (
            ancestor in programs
            and _find_custom_op_node(programs[ancestor], op_name) is not None
        ):
            return programs[ancestor]
    return None


def _reverse_lookup(mapping: dict[str, str], value: str) -> str | None:
    """Return the first key in *mapping* whose value equals *value*, or None."""
    for k, v in mapping.items():
        if v == value:
            return k
    return None


def _get_mutation_output_name(
    orig_name: str,
    inputs_to_buffers: dict[str, str],
    buffers_to_mutate: dict[str, str],
    user_inputs_to_mutate: dict[str, str],
) -> str | None:
    """Return the output node name if *orig_name* is a mutated input, else None.

    Checks both buffer mutations (``buffers_to_mutate``) and user input
    mutations (``user_inputs_to_mutate``).
    """
    buffer_name = inputs_to_buffers.get(orig_name)
    if buffer_name is not None and buffer_name in buffers_to_mutate.values():
        return _reverse_lookup(buffers_to_mutate, buffer_name)
    return _reverse_lookup(user_inputs_to_mutate, orig_name)


def _validate_input_names(
    user_input_names: list[str],
    graph_input_names: list[str],
) -> None:
    """Validate that user-provided input names match the graph live-in count.

    Raises:
        ValueError: If the lengths differ.
    """
    if user_input_names and len(user_input_names) != len(graph_input_names):
        raise ValueError(
            f"Graph has {len(graph_input_names)} live inputs "
            f"({graph_input_names}), but input_names has "
            f"{len(user_input_names)} entries ({list(user_input_names)})."
        )


def _validate_state_input_names(
    user_state_names: list[str],
    graph_state_names: list[str],
) -> None:
    """Validate that user-provided state input names match the stateful input count.

    Raises:
        ValueError: If the lengths differ.
    """
    if user_state_names and len(user_state_names) != len(graph_state_names):
        raise ValueError(
            f"Graph has {len(graph_state_names)} stateful inputs "
            f"({graph_state_names}), but state_names has "
            f"{len(user_state_names)} entries ({list(user_state_names)})."
        )


def _classify_stateful_inputs(
    graph_signature: ExportGraphSignature,
    original_input_names: list[str],
) -> tuple[list[int], list[int]]:
    """Classify graph inputs into stateful and non-stateful indices.

    Stateful inputs are those with a mutation output:
    - Mutable buffers (from register_buffer with in-place ops)
    - Mutated user inputs (user_inputs_to_mutate)

    Returns:
        A tuple of (state_indices, non_state_indices).
    """
    inputs_to_buffers = graph_signature.inputs_to_buffers
    buffers_to_mutate = graph_signature.buffers_to_mutate
    user_inputs_to_mutate = graph_signature.user_inputs_to_mutate

    state_indices: list[int] = []
    non_state_indices: list[int] = []

    for i, orig_name in enumerate(original_input_names):
        mutation_output = _get_mutation_output_name(
            orig_name, inputs_to_buffers, buffers_to_mutate, user_inputs_to_mutate
        )
        if mutation_output is not None:
            state_indices.append(i)
        else:
            non_state_indices.append(i)

    return state_indices, non_state_indices


def _classify_stateful_outputs(
    graph_signature: ExportGraphSignature,
    graph_output_names: list[str],
) -> tuple[list[int], list[int]]:
    """Classify graph outputs into state-mutation outputs and non-state outputs.

    Returns:
        A tuple of (state_output_indices, non_state_output_indices).
    """
    mutation_outputs = set(graph_signature.buffers_to_mutate.keys()) | set(
        graph_signature.user_inputs_to_mutate.keys()
    )

    state_indices: list[int] = []
    non_state_indices: list[int] = []
    for i, name in enumerate(graph_output_names):
        if name in mutation_outputs:
            state_indices.append(i)
        else:
            non_state_indices.append(i)
    return state_indices, non_state_indices


def _get_default_state_names(
    original_input_names: list[str],
    state_indices: list[int],
) -> list[str]:
    """Derive default state names from FX placeholder names.

    Keeps the original FX names (e.g. 'b_kv_cache' for buffers, 'y' for
    mutated user inputs) to avoid potential collisions with non-state inputs.
    """
    return [original_input_names[i] for i in state_indices]


def _resolve_io_names(
    graph_signature: ExportGraphSignature,
    original_input_names: list[str],
    graph_output_names: list[str],
    user_input_names: Sequence[str],
    user_output_names: Sequence[str],
    user_state_names: Sequence[str],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Resolve final input/output names for the graph.

    Always classifies inputs/outputs into state vs non-state:
    - input_names covers non-state inputs only
    - output_names covers non-state outputs only
    - state_names (if provided) overrides state IO names;
      otherwise FX placeholder defaults are used

    Returns:
        A tuple of:
        - graph_input_names: resolved names for all graph inputs (in order)
        - resolved_output_names: resolved names for all graph outputs (in order)
        - fx_to_output: mapping from FX output node name to resolved output
          name (used for MutableBuffers.buffer_mutation attr resolution)
    """
    state_in_idx, non_state_in_idx = _classify_stateful_inputs(
        graph_signature, original_input_names
    )
    state_out_idx, non_state_out_idx = _classify_stateful_outputs(
        graph_signature, graph_output_names
    )

    # State inputs and outputs must be 1:1 — each mutated input has exactly
    # one corresponding mutation output. This relies on PyTorch's
    # ExportGraphSignature emitting mutation outputs in the same order as
    # their corresponding inputs (buffers first, then mutated user inputs).
    assert len(state_in_idx) == len(state_out_idx), (
        f"State input/output count mismatch: {len(state_in_idx)} state inputs "
        f"but {len(state_out_idx)} state outputs. This may indicate an "
        f"unsupported graph signature layout."
    )

    # Verify the documented "mutable buffers first, then mutated user inputs"
    # ordering. If PyTorch ever interleaves them, fail loudly here so
    # state_names don't get silently mapped to the wrong buffer/arg.
    inputs_to_buffers = graph_signature.inputs_to_buffers
    buffers_to_mutate_vals = set(graph_signature.buffers_to_mutate.values())
    buffer_positions: list[int] = []
    user_input_positions: list[int] = []
    for i in state_in_idx:
        buffer_name = inputs_to_buffers.get(original_input_names[i])
        if buffer_name is not None and buffer_name in buffers_to_mutate_vals:
            buffer_positions.append(i)
        else:
            user_input_positions.append(i)
    assert buffer_positions + user_input_positions == state_in_idx, (
        "FX placeholder order violates the 'mutable buffers first, then "
        "mutated user inputs' invariant. State input positions: "
        f"{state_in_idx}, buffer positions: {buffer_positions}, mutated "
        f"user input positions: {user_input_positions}. This breaks the "
        "documented state_names ordering — pass state_names explicitly "
        "matched to your buffer/arg names, or check PyTorch version "
        "compatibility."
    )

    # Resolve state names
    resolved_state: list[str] = (
        list(user_state_names)
        if user_state_names
        else _get_default_state_names(original_input_names, state_in_idx)
    )

    # Validate counts
    if user_state_names:
        _validate_state_input_names(
            list(user_state_names),
            [original_input_names[i] for i in state_in_idx],
        )
    _validate_input_names(
        list(user_input_names),
        [original_input_names[i] for i in non_state_in_idx],
    )
    _validate_output_names(
        user_output_names,
        [graph_output_names[i] for i in non_state_out_idx],
    )

    # Build resolved input names
    graph_input_names = list(original_input_names)
    for i, state_name in zip(state_in_idx, resolved_state):
        graph_input_names[i] = state_name
    if user_input_names:
        for i, inp_name in zip(non_state_in_idx, user_input_names):
            graph_input_names[i] = inp_name

    # Build resolved output names + fx→output mapping
    resolved_output_names = list(graph_output_names)
    fx_to_output: dict[str, str] = {}

    for i, state_name in zip(state_out_idx, resolved_state):
        resolved_output_names[i] = state_name
        fx_to_output[graph_output_names[i]] = state_name

    if user_output_names:
        for i, out_name in zip(non_state_out_idx, user_output_names):
            resolved_output_names[i] = out_name
            fx_to_output[graph_output_names[i]] = out_name

    return graph_input_names, resolved_output_names, fx_to_output


def _get_debug_info_enabled() -> bool:
    """Get debug info enable flag from ENABLE_DEBUG_INFO environment variable.

    By default, debug info is not enabled for performance reasons.
    Set ENABLE_DEBUG_INFO=true to enable debug information generation.

    Returns:
        True if debug info should be enabled, False otherwise.
        Defaults to False if environment variable is not set.
    """
    return os.getenv("ENABLE_DEBUG_INFO", "false").lower() in ("true", "1", "yes", "on")


def _get_verify_debuginfo_locations_enabled() -> bool:
    """Get debuginfo location verification flag from VERIFY_DEBUGINFO_LOCATIONS environment variable.

    By default, debuginfo location verification is not enabled for performance reasons.

    Returns:
        True if debuginfo location verification should be enabled, False otherwise.
        Defaults to False if environment variable is not set.
    """
    return os.getenv("VERIFY_DEBUGINFO_LOCATIONS", "false").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )


def _validate_output_names(
    user_output_names: list[str],
    graph_output_names: list[str],
) -> None:
    """Validate that user-provided output names match the graph live-out count.

    Raises:
        ValueError: If the lengths differ.
    """
    if user_output_names and len(user_output_names) != len(graph_output_names):
        raise ValueError(
            f"Graph has {len(graph_output_names)} live outputs "
            f"({graph_output_names}), but output_names has "
            f"{len(user_output_names)} entries ({list(user_output_names)})."
        )


def build_hard_sigmoid_composite(context: Any) -> coreai.GraphOp:
    """Build a hard_sigmoid composite graph: min(max(x + 3, 0), 6) / 6."""
    composite_decl = generate_composite_decl(
        context, "hard_sigmoid", ["input"], ["output"], {}
    )

    @coreai.graph(private=True, no_inline=True, composite_decl=composite_decl)
    def hard_sigmoid(input: Value) -> Value:
        dtype = input.type.element_type
        three = coreai.constant(3.0, dtype=dtype)
        zero = coreai.constant(0.0, dtype=dtype)
        six = coreai.constant(6.0, dtype=dtype)
        add_three = coreai.broadcasting_add(input, three)
        max_val = coreai.broadcasting_maximum(add_three, zero)
        min_val = coreai.broadcasting_minimum(max_val, six)
        return coreai.broadcasting_divide(min_val, six)

    return hard_sigmoid


def validate_and_cast_numpy_array(
    arr: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Validate and cast a numpy array.

    Only ranked arrays are accepted.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.shape == ():
        arr = arr[None]
    arr = np.require(
        arr,
        requirements=["C_CONTIGUOUS"],
    )
    arr.flags.writeable = False

    return arr
