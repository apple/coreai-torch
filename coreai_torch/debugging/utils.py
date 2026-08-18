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

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
from coreai._compiler.ir import Operation
from coreai.authoring import AIProgram


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
