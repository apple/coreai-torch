# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Standalone tool to strip debug info from Core AI programs and ``.aimodel`` assets.

This file depends only on the ``coreai`` package, not on ``coreai_torch``.

Usage::

    python strip_debug_info.py model.aimodel                  # overwrite in place
    python strip_debug_info.py model.aimodel -o stripped.aimodel
    python strip_debug_info.py model.aimodel --no-op-ids      # all locations unknown
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from coreai._compiler._mlir_libs._coreaiIR._bindings.mlir import (
    set_block_arg_location,
    set_op_location,
)
from coreai._compiler.dialects.debuginfo.attrs import location_attr, metadata_attr
from coreai._compiler.ir import (
    Context,
    DictAttr,
    IntegerAttr,
    IntegerType,
    Location,
    Operation,
    StringAttr,
    UnitAttr,
)
from coreai.authoring import AIModelAsset, AIProgram

_OP_ID_LEVEL = "coreai"


def _get_nested_operations(operation: Operation) -> Iterator[Operation]:
    """Iteratively yield all operations nested in *operation*'s regions."""
    operation_stack: list[Operation] = [operation]
    while operation_stack:
        current_op = operation_stack.pop()
        for region in current_op.regions:
            for block in region:
                for nested_op in block:
                    yield nested_op
                    operation_stack.append(nested_op)


def _create_unknown_location(
    unknown_src: Location,
    context: Context,
    metadata_attrs: list[metadata_attr] | None = None,
) -> location_attr:
    """Create an unknown debuginfo location with optional metadata."""
    return location_attr(
        src=unknown_src,
        scope=UnitAttr.get(context=context),
        metadata=metadata_attrs,
        context=context,
    )


def _create_op_id_location(
    op_id: int, unknown_src: Location, context: Context
) -> location_attr:
    """Create an unknown debuginfo location carrying only a ``coreai`` op ID."""
    uint64 = IntegerType.get_unsigned(64, context=context)
    data = DictAttr.get(
        {
            "type": StringAttr.get(_OP_ID_LEVEL, context=context),
            "value": IntegerAttr.get(uint64, op_id),
        },
        context=context,
    )
    op_id_metadata = metadata_attr(name="op_id", data=data, context=context)
    return _create_unknown_location(unknown_src, context, [op_id_metadata])


def _assign_fresh_op_ids(module_op: Operation) -> None:
    """Replace all locations with unknown source locations carrying fresh op IDs.

    Args:
        module_op: The module operation to rewrite. Modified in place.
    """
    context = module_op.context

    # Create a shared unknown source location for reuse
    unknown_src = Location.file(filename="", line=0, col=0, context=context)

    # Set module operation location (no operation ID for the module itself)
    set_op_location(module_op, _create_unknown_location(unknown_src, context))

    # Walk all nested operations and assign fresh sequential IDs
    for operation_id, nested_op in enumerate(_get_nested_operations(module_op)):
        loc = _create_op_id_location(operation_id, unknown_src, context)
        set_op_location(nested_op, loc)

        # Update block argument locations
        for region in nested_op.regions:
            for block in region:
                for arg in block.arguments:
                    set_block_arg_location(arg, loc)


def strip_debug_info(program: AIProgram, include_op_ids: bool = True) -> None:
    """Strip debugging information from all operations in the program.

    This is useful for reducing asset size when full debug traces are
    no longer needed.

    Args:
        program: The AIProgram to strip debug info from. Modified in place.
        include_op_ids: If True (default), assign fresh sequential operation
            IDs to every operation. If False, every location (module,
            operations and block arguments) is replaced with a plain unknown
            location carrying no operation IDs.
    """
    module_op = program._module._mlir_module.operation

    if include_op_ids:
        _assign_fresh_op_ids(module_op)
        return

    unknown_loc = Location.unknown(context=module_op.context)
    set_op_location(module_op, unknown_loc)
    for nested_op in _get_nested_operations(module_op):
        set_op_location(nested_op, unknown_loc)
        for region in nested_op.regions:
            for block in region:
                for arg in block.arguments:
                    set_block_arg_location(arg, unknown_loc)


def strip_debug_info_from_aimodel(
    path: Path | str,
    output_path: Path | str | None = None,
    include_op_ids: bool = True,
) -> Path:
    """Load an ``.aimodel`` asset, strip its debug info and save it back.

    Args:
        path: Path to the ``.aimodel`` asset to read.
        output_path: Where to save the stripped asset. If None (default), the
            asset at *path* is overwritten.
        include_op_ids: Forwarded to :func:`strip_debug_info`.

    Returns:
        The path of the saved asset.
    """
    path = Path(path)
    output_path = path if output_path is None else Path(output_path)

    asset = AIModelAsset.load(path)
    program = asset.program
    strip_debug_info(program, include_op_ids=include_op_ids)
    program.save_asset(output_path, metadata=asset.metadata)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="Input .aimodel asset.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .aimodel path (default: overwrite the input).",
    )
    parser.add_argument(
        "--no-op-ids",
        action="store_true",
        help="Replace every location with a plain unknown location (no op IDs).",
    )
    args = parser.parse_args()
    saved = strip_debug_info_from_aimodel(
        args.path, output_path=args.output, include_op_ids=not args.no_op_ids
    )
    print(f"Saved stripped asset to {saved}")


if __name__ == "__main__":
    main()
