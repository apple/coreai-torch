# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Which Core AI operations hold a module's output.

The compiler fuses, folds and reshapes, so a module's output is not simply the last
operation carrying its name. This reads each operation's torch ids and module stacks,
and the node table `coreai_torch` records on the graph, and reports a module only when
nothing could have changed its value on the way out.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto

from coreai._compiler.ir import Attribute, DictAttr, IntegerAttr, Operation, StringAttr
from coreai.authoring import AIProgram

from .._debug_locations import NODE_TABLE_METADATA, NODE_TABLE_VERSION
from .debug_info import _build_coreai_op_map, get_operation_id
from .source_annotator import (
    ModulePath,
    _leaf_stack_traces,
    operation_stack_trace,
    operation_torch_ids,
)
from .utils import STRUCTURAL_OPS

_SHAPE_ONLY = frozenset(
    {
        "view",
        "reshape",
        "_unsafe_view",
        "squeeze",
        "unsqueeze",
        "flatten",
        "contiguous",
        "clone",
        "alias",
        "detach",
        "expand",
    }
)
"""ATen operations that return their input's values unchanged, in another shape or broadcast."""

_STATE_OPS = frozenset(
    {"coreai.create_token", "coreai.read_handle", "coreai.write_handle"}
)
"""Operations that move a value into or out of program state."""


@dataclass(frozen=True)
class _SourceNodes:
    """What a graph's location says about the torch nodes converted into it."""

    targets: dict[int, str]
    """ATen target of each ``call_function`` node, by torch operation ID."""

    output_id: int | None
    """ID of the node that returns the graph's values, None when not recorded."""

    def shape_only(self, node_id: int) -> bool:
        """Whether the node returns its input's values unchanged, in another shape."""
        parts = self.targets.get(node_id, "").split(".")
        return len(parts) > 1 and parts[1] in _SHAPE_ONLY


def _metadata_data(printed: str, name: str) -> str | None:
    """
    The printed data of the metadata called *name* in a printed location, if any.

    A location's metadata is read from its printed form because builds differ in
    what they expose of it: some give the list of metadata, others only the
    attribute. The data is a brace-delimited dictionary, which MLIR parses back.
    """
    marker = f'<"{name}": '
    start = printed.find(marker)
    if start < 0:
        return None
    start += len(marker)
    depth, quoted = 0, False
    for index in range(start, len(printed)):
        char = printed[index]
        if char == '"' and printed[index - 1] != "\\":
            quoted = not quoted
        elif not quoted and char == "{":
            depth += 1
        elif not quoted and char == "}":
            depth -= 1
            if depth == 0:
                return printed[start : index + 1]
    return None


def _read_source_nodes(program: AIProgram) -> dict[str, _SourceNodes]:
    """
    The node table recorded on each graph's location, by graph name.

    A graph with no table, or one this version cannot read, is absent: nothing is
    assumed about the nodes of a program that was not converted with one.
    """
    graphs: dict[str, _SourceNodes] = {}
    for operation in _build_coreai_op_map(program).values():
        if operation.name != "coreai.graph":
            continue
        printed = _metadata_data(str(operation.location), NODE_TABLE_METADATA)
        if printed is None:
            continue
        with operation.context:
            data = DictAttr(Attribute.parse(printed))
        if IntegerAttr(data["version"]).value != NODE_TABLE_VERSION:
            continue
        name = StringAttr(operation.attributes["sym_name"]).value
        graphs[name] = _SourceNodes(
            targets={
                int(entry.name): StringAttr(entry.attr).value
                for entry in DictAttr(data["nodes"])
            },
            output_id=(
                IntegerAttr(data["output_id"]).value if "output_id" in data else None
            ),
        )
    return graphs


class _Verdict(Enum):
    """What one reader of an operation's value says about its module."""

    STAYS = auto()
    """The reader is in the same module, so the value has not left it."""

    LEAVES = auto()
    """The reader is in another module, with every node on the way accounted for."""

    UNKNOWN = auto()
    """A node on the way could have changed the value, so the boundary is not certain."""


class _Boundaries:
    """
    The module each operation of a program belongs to, and what lies between them.

    An operation is owned by the torch node with the largest ID in its location that
    is not shape-only. The smaller IDs belong to producers it absorbed: they say where
    it came from, not what it computes.
    """

    def __init__(self, program: AIProgram) -> None:
        self._graphs = _read_source_nodes(program)
        targets = {
            i: t for graph in self._graphs.values() for i, t in graph.targets.items()
        }
        self._shape_only = _SourceNodes(targets, None).shape_only
        self._ops = _build_coreai_op_map(program)
        self._node: dict[int, int] = {}
        self._stack: dict[int, ModulePath] = {}
        self._seen: set[int] = set()
        for op_id, operation in self._ops.items():
            if operation.name not in STRUCTURAL_OPS:
                self._own(op_id, operation)

    def _own(self, op_id: int, operation: Operation) -> None:
        """Record which torch node and module an operation belongs to."""
        ids = operation_torch_ids(operation)
        self._seen |= ids
        if operation.name == "coreai.constant" or not ids:
            return
        node = max({i for i in ids if not self._shape_only(i)} or ids)
        stacks = {node_id: stack for stack, node_id in _leaf_stack_traces(operation)}
        self._node[op_id] = node
        self._stack[op_id] = stacks.get(node) or operation_stack_trace(operation)

    def modules(self) -> list[ModulePath]:
        """Every module that owns an operation, parents included."""
        return sorted(
            {
                stack[:depth]
                for stack in self._stack.values()
                for depth in range(1, len(stack) + 1)
            }
        )

    def members(self, module: ModulePath) -> list[int]:
        """The operations owned by *module* or by a module inside it."""
        return [
            op_id
            for op_id, stack in self._stack.items()
            if stack[: len(module)] == module
        ]

    def readers(self, op_id: int) -> Iterator[Operation]:
        """
        The operations that read any value of operation *op_id*.

        A state operation is not a reader: writing a value to state is not it leaving
        the module, and a module's output is what its caller receives, not what it
        stores.
        """
        for result in self._ops[op_id].results:
            for use in result.uses:
                if use.owner.name not in _STATE_OPS:
                    yield use.owner

    def verdict(self, op_id: int, reader: Operation, module: ModulePath) -> _Verdict:
        """What *reader* says about whether *op_id*'s value has left *module*."""
        if reader.name == "coreai.output":
            return self._verdict_for_output(op_id, reader, module)
        reader_id = get_operation_id(reader)
        if reader_id not in self._stack:
            return _Verdict.UNKNOWN
        if self._stack[reader_id][: len(module)] == module:
            return _Verdict.STAYS
        return self._verdict_between(self._node[op_id], self._node[reader_id])

    def _verdict_for_output(
        self, op_id: int, output: Operation, module: ModulePath
    ) -> _Verdict:
        """The program's own output: the root's, or a module's if nothing follows it."""
        if len(module) == 1:
            return _Verdict.LEAVES
        graph = StringAttr(output.parent.attributes["sym_name"]).value
        end = self._graphs[graph].output_id if graph in self._graphs else None
        if end is None:
            return _Verdict.UNKNOWN
        return self._verdict_between(self._node[op_id], end)

    def _verdict_between(self, low: int, high: int) -> _Verdict:
        """LEAVES when every torch node between the two is accounted for."""
        if high <= low:
            return _Verdict.UNKNOWN
        # An id on no operation was folded away without a trace, as batch norm is into a
        # conv. A reshape is harmless only if the node table says it was one.
        hidden = any(
            node_id not in self._seen and not self._shape_only(node_id)
            for node_id in range(low + 1, high)
        )
        return _Verdict.UNKNOWN if hidden else _Verdict.LEAVES


def module_output_op_ids(program: AIProgram) -> dict[ModulePath, list[int]]:
    """
    The Core AI operations that hold each module instance's output.

    A module's output is a value that leaves it: an operation it owns that an operation
    of another module reads, or that returns the program's own output. The module is
    left out, rather than guessed, whenever a node could have changed the value on its
    way out: an id between the producer's torch node and its reader's that shows up in
    no operation and is not known to be shape-only, a reader with no torch id, or one
    whose id is not larger. A value that is only written to state is not an output.

    Without the node table, written when `coreai_torch` converts the program, no id is
    known to be shape-only and the program's output is accepted only for the root
    module, so fewer modules are reported. A program with no debug info reports none.

    Args:
        program: Program to read, converted with debug info.

    Returns:
        Module path to the sorted ids of the operations holding its output, for every
        module whose output is certain.

    """
    boundaries = _Boundaries(program)
    outputs: dict[ModulePath, list[int]] = {}
    for module in boundaries.modules():
        verdicts = {
            op_id: [
                boundaries.verdict(op_id, reader, module)
                for reader in boundaries.readers(op_id)
            ]
            for op_id in boundaries.members(module)
        }
        if any(_Verdict.UNKNOWN in found for found in verdicts.values()):
            continue
        leaving = sorted(
            op_id for op_id, found in verdicts.items() if _Verdict.LEAVES in found
        )
        if leaving:
            outputs[module] = leaving
    return outputs
