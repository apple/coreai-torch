# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Operation traversal shared by the debugging tools.

The tools each need every operation in a program, including those nested inside
composite graph bodies, so the walk lives here rather than being repeated per
module.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
from coreai._compiler.ir import Operation
from coreai.authoring import AIProgram
from coreai.runtime import SpecializationOptions


def _with_debug(
    specialization_options: SpecializationOptions | None,
) -> SpecializationOptions:
    """
    The caller's options with debug enabled, defaulting when they passed none.

    Args:
        specialization_options: What the caller asked for, or None.

    Returns:
        Those options with debug enabled, or the defaults with debug enabled.

    """
    if specialization_options is None:
        specialization_options = SpecializationOptions.default()
    return specialization_options.with_debug(enabled=True)


def _plain(value: Any) -> Any:
    """
    Normalise one leaf value for a report's ``to_dict``.

    Args:
        value: Leaf, or a container of leaves.

    Returns:
        The same value as `json.dumps` accepts it. A non-finite float becomes None,
        which is the convention `IntermediateComparison.max_diff` already uses.

    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _walk_operation(
    operation: Operation, depth: int = 0
) -> Iterator[tuple[Operation, int]]:
    """
    Walk *operation* and everything nested inside its regions.

    Args:
        operation: Operation to start from.
        depth: Region nesting depth of *operation*, used for the values yielded
            for its children.

    Yields:
        Each operation and its region nesting depth, parents before children.

    """
    resolved = getattr(operation, "operation", operation)
    yield resolved, depth
    for region in resolved.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk_operation(child, depth + 1)


def _walk_operations(coreai_program: AIProgram) -> list[Operation]:
    """
    Collect every operation in a program.

    Args:
        coreai_program: AIProgram to walk.

    Returns:
        The operations, parents before children.

    """
    module = coreai_program._mlir_module.operation
    return [operation for operation, _ in _walk_operation(module)]


def split_module_frame(frame: str) -> tuple[str, int | None]:
    """
    Split a stack-trace module frame into its type and instance number.

    A frame names an instance, not a type -- ``Linear$3`` is the third ``Linear``.
    Splitting is what lets a caller group instances of one type together without
    re-parsing the name itself.

    Args:
        frame: Module frame as the stack trace gives it, e.g. ``"Linear$3"``.

    Returns:
        The type name and the instance number, the latter None when the frame
        carries no number (``<unknown>``, or a name with no ``$``).

    """
    type_name, _, suffix = frame.partition("$")
    return type_name, int(suffix) if suffix.isdigit() else None


@dataclass(frozen=True)
class LocationInfo:
    """
    A source location attributed to a Core AI operation.
    """

    filename: str
    """Source filename."""

    line: int
    """Line number."""

    col: int
    """Column number."""


def get_operation_locations(operation: Operation) -> list[LocationInfo]:
    """
    Extract the source locations attributed to an operation.

    Args:
        operation: Operation to extract locations from.

    Returns:
        Unique locations, order preserved, innermost last.

    """
    file_line_cols = _mlir.get_file_line_col_locations(operation.location)  # type: ignore[attr-defined]

    locations = [
        LocationInfo(filename=loc.filename, line=loc.line, col=loc.col)
        for loc in file_line_cols
    ]

    # Dedupe while preserving order.
    return list(reversed(OrderedDict.fromkeys(locations)))


# ---------------------------------------------------------------------------
# Composite naming, shared by the histogram and the graph comparisons
# ---------------------------------------------------------------------------

_UUID_SUFFIX_RE = re.compile(r"_(?:[a-z]{8,}|[0-9a-f]{8})$")
"""Trailing random suffix on a generated composite symbol name.

Two generators produce one. `coreai.graph`'s deferred-construction path draws from
`string.ascii_lowercase` only, never digits; module externalization uses
`uuid4().hex[:8]`, so digits are usual there. Used only as a fallback when `template_op`
is unavailable -- which is always, on the externalization path.
"""


def _collect_entry_points(module: Any) -> dict[str, Any]:
    """Collect all coreai.graph ops from a module, keyed by sym_name."""
    entry_points: dict[str, Any] = {}
    for op in module.body.operations:
        if op.name != "coreai.graph":
            continue
        if not hasattr(op, "sym_name"):
            continue
        entry_points[op.sym_name.value] = op
    return entry_points


def _strip_uuid_suffix(name: str) -> str:
    """
    Strip trailing random suffix for display: 'sdpa_maskless_qxrbmzua' -> 'sdpa_maskless'.

    A best-effort fallback for when the generating `GraphOp` carries no `template_op`
    attribute (see `_composite_label`) -- pattern-matching a randomised suffix rather
    than reading what named it. Handles both generators `_UUID_SUFFIX_RE` documents. A
    composite created some other way, or a future generator using a different alphabet
    or length, may not match; the name is then shown as printed, suffix and all, rather
    than stripped incorrectly.
    """
    return _UUID_SUFFIX_RE.sub("", name)


_COMPOSITE_DECL_RE = re.compile(r'composite_declaration<"([^"]+)"')
"""The op name a `composite_decl` attribute declares.

Printed as ``#coreai.composite_declaration<"rms_norm" = {input_names = ...}>``. Set when
a module is externalized through an `ExternalizeSpec` carrying a ``composite_op_name``.
"""


def _composite_label(sym_name: str, entry_points: Mapping[str, Any]) -> str:
    """
    Human-readable name for a composite callee.

    Prefers `template_op`, the pre-randomisation name the generating `GraphOp`
    recorded verbatim alongside its randomised symbol name -- see
    `coreai._compiler.dialects.coreai.graph._generate_fn_with_body`, which sets both
    together. Exact and independent of the suffix's alphabet or length, unlike
    `_strip_uuid_suffix`, which has to guess the pattern and cannot: composite names
    are not a closed set to match against, since module externalization lets a caller
    register one under any name it chooses.

    Falls back to `composite_decl`'s declared op name next, which is the *operation* a
    composite stands for (``rms_norm``) rather than where it sits in the module tree
    (``layer0.attn_norm``). Two instances of one composite op then share a label, which
    is what makes them comparable; the symbol still separates them.

    Args:
        sym_name: The composite's (possibly suffixed) symbol name.
        entry_points: `sym_name -> GraphOp`, as `_collect_entry_points` returns.

    Returns:
        `template_op`, else the declared composite op name, else `sym_name` with a
        best-effort suffix strip.

    """
    op = entry_points.get(sym_name)
    if op is None:
        return _strip_uuid_suffix(sym_name)
    template_op = getattr(op, "template_op", None)
    if template_op:
        return template_op
    declared = _declared_op_name(op)
    return declared or _strip_uuid_suffix(sym_name)


def _declared_op_name(graph_op: Any) -> str | None:
    """
    The op name a graph's `composite_decl` declares, when it carries one.

    Args:
        graph_op: The `coreai.graph` to read.

    Returns:
        The declared name, or None when the attribute is absent or unparseable.

    """
    try:
        printed = str(graph_op.attributes["composite_decl"])
    except (KeyError, RuntimeError):
        return None
    match = _COMPOSITE_DECL_RE.search(printed)
    return match.group(1) if match is not None else None
