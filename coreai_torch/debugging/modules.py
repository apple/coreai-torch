# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Grouping a per-operation finding by the module instance it came from.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from coreai.authoring import AIProgram
from typing_extensions import Self

from .._debug_locations import TorchNode
from .debug_info import _build_coreai_op_map, get_operation_id
from .source_annotator import (
    ModulePath,
    module_paths_by_op_id,
    operation_torch_ids,
    source_locations_by_op_id,
)
from .utils import STRUCTURAL_OPS, LocationInfo, split_module_frame

T = TypeVar("T")

UNATTRIBUTED = "<unknown>"
"""Bucket for a finding whose operation has no module path."""


@dataclass
class ModuleNode(Generic[T]):
    """One module instance, what is filed against it, and what is beneath it."""

    name: str
    """Frame as the stack trace gives it, instance number included: ``Linear$3``."""

    items: list[T] = field(default_factory=list)
    """Findings filed against this instance itself, not against its children."""

    children: list[ModuleNode[T]] = field(default_factory=list)
    """Module instances directly beneath this one."""

    @property
    def type_name(self: Self) -> str:
        """
        The module's type, without the instance number.

        Returns:
            ``"Linear"`` for ``Linear$3``, and the whole name when it carries no
            instance number.

        """
        return split_module_frame(self.name)[0]

    @property
    def instance(self: Self) -> int | None:
        """
        Which instance of :attr:`type_name` this module is.

        Returns:
            ``3`` for ``Linear$3``, and None when the name carries no number.

        """
        return split_module_frame(self.name)[1]

    def walk(self: Self) -> Iterator[ModuleNode[T]]:
        """
        This node and every descendant.

        Yields:
            Each node, parents before children.

        """
        yield self
        for child in self.children:
            yield from child.walk()

    def all_items(self: Self) -> list[T]:
        """
        Everything filed anywhere in this subtree.

        Returns:
            This node's own items followed by its descendants', in walk order.

        """
        return [item for node in self.walk() for item in node.items]

    def find(self: Self, name: str) -> ModuleNode[T] | None:
        """
        The node named *name* in this subtree, if there is one.

        Args:
            name: Module frame, e.g. ``"Linear$3"``, or a path such as
                ``"Block$2/Linear$1"``.

        Returns:
            The matching node, or None. A path matches against this subtree's own
            nesting, so ``"Block$2/Linear$1"`` finds the `Linear$1` inside `Block$2`
            rather than any `Linear$1`.

        """
        head, _, rest = name.partition("/")
        for node in self.walk():
            if node.name != head:
                continue
            if not rest:
                return node
            found = next(
                (child.find(rest) for child in node.children if child.find(rest)),
                None,
            )
            if found is not None:
                return found
        return None

    def to_dict(self: Self, item: Callable[[T], Any]) -> dict[str, Any]:
        """
        Return this subtree as plain values.

        Args:
            item: Renders one filed item as plain values. Taken as an argument rather
                than assumed to be ``item.to_dict`` so this works for a payload the
                caller does not own, an int or a str included.

        Returns:
            This module, its split into type and instance, how much is under it in
            total, its own items, and its children.

        """
        return {
            "name": self.name,
            "type_name": self.type_name,
            "instance": self.instance,
            "count": len(self.all_items()),
            "items": [item(entry) for entry in self.items],
            "children": [child.to_dict(item) for child in self.children],
        }


def build_module_tree(filed: Iterable[tuple[ModulePath, T]]) -> list[ModuleNode[T]]:
    """
    Group findings into the module tree they came from.

    Ancestors are created whether or not anything is filed against them, so the
    hierarchy is complete. That matters for the module whose work always fuses with a
    sibling's: the structure is there and nothing is attributed to it alone, which is
    the honest reading and the one `BenchmarkResult.get_module_timings` already takes.

    Args:
        filed: Each finding with the module path it belongs to, outermost frame first.
            An empty path files the finding under :data:`UNATTRIBUTED`.

    Returns:
        The root modules, each subtree ordered by descending item count so the branch
        that holds the most is read first. Ties break on name, so the order is stable
        between runs rather than dependent on which finding arrived first.

    """
    roots: dict[str, ModuleNode[T]] = {}

    def node_at(path: ModulePath) -> ModuleNode[T]:
        """Find or create the node at *path*, creating ancestors as needed."""
        level, node = roots, None
        for frame in path:
            if frame not in level:
                created: ModuleNode[T] = ModuleNode(name=frame)
                level[frame] = created
                if node is not None:
                    node.children.append(created)
            node = level[frame]
            level = {child.name: child for child in node.children}
        assert node is not None, "a path with no frames cannot reach here"
        return node

    for path, item in filed:
        node_at(tuple(path) or (UNATTRIBUTED,)).items.append(item)

    def sorted_by_weight(nodes: list[ModuleNode[T]]) -> list[ModuleNode[T]]:
        for node in nodes:
            node.children = sorted_by_weight(node.children)
        return sorted(nodes, key=lambda node: (-len(node.all_items()), node.name))

    return sorted_by_weight(list(roots.values()))


@dataclass(frozen=True)
class Attribution:
    """Where one Core AI operation came from."""

    op_id: int
    """Operation id, in its own program's numbering."""

    module: ModulePath
    """Module instance path, outermost first. Empty when none was recorded."""

    source: LocationInfo | None
    """The line in the author's own code, or None when every frame was library code."""

    @property
    def label(self: Self) -> str:
        """
        The module path as one string.

        Returns:
            ``"Block$2/Linear$1"``, or :data:`UNATTRIBUTED` when the path is empty.

        """
        return "/".join(self.module) if self.module else UNATTRIBUTED

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Return the attribution as plain values.

        Returns:
            The op id, the module as both a label and a path, and the source line.
            `module_path` is kept alongside `module` because a caller that wants to
            group by the outermost frame should not have to re-split the label.

        """
        return {
            "op_id": self.op_id,
            "module": self.label,
            "module_path": list(self.module),
            "source": (
                {"file": self.source.filename, "line": self.source.line}
                if self.source
                else None
            ),
        }


def attributions_by_op_id(program: AIProgram) -> dict[int, Attribution]:
    """
    Module and source line for every operation in a program, by op id.

    The one call a tool holding op ids makes to say *where*, so a finding can name
    ``self.fc2`` instead of leaving a reader to infer it from operation order.

    Composes the two lookups it is named after, which deliberately disagree on how to
    filter source locations: provenance must not trim frames or a model of stock `nn`
    modules collapses to one path, while a source line must trim them or it names a line
    inside torch. See `module_paths_by_op_id` for the full argument.

    Keyed by op id, and named for it, because an id is unique only within one program: a
    caller holding ids from two programs needs one of these per side, and a name that
    left the key implicit invited merging them.

    Args:
        program: Program to read locations from. Must have been converted with
            `include_stack_trace=True`, which `TorchConverter.Mode.DEBUG` -- the
            default -- sets. Without it no operation has a module path and every
            finding files under :data:`UNATTRIBUTED`.

    Returns:
        Op id to its attribution, holding every operation that has either a module path
        or a source line. Structural operations have neither and are absent; see
        `utils.STRUCTURAL_OPS`.

    """
    paths = module_paths_by_op_id(program)
    lines = source_locations_by_op_id(program)
    return {
        op_id: Attribution(
            op_id=op_id,
            module=paths.get(op_id, ()),
            source=lines.get(op_id),
        )
        for op_id in sorted({*paths, *lines})
    }


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
    }
)
"""ATen operations that return their input's values unchanged, in another shape."""


def _is_shape_only(node: TorchNode) -> bool:
    parts = node.target.split(".")
    return node.op == "call_function" and len(parts) > 1 and parts[1] in _SHAPE_ONLY


def module_output_op_ids(
    program: AIProgram, torch_graph: Mapping[int, TorchNode]
) -> dict[ModulePath, list[int]]:
    """
    The Core AI operations that hold each module instance's output.

    A module's output is a torch node inside it that a node outside it reads, or the
    graph returns. An operation realizes the torch node with the largest ID in its
    location; the smaller ones are producers it absorbed, and say where it came from,
    not what it computes. A node's value is therefore the operation realizing it whose
    value something realizing another node reads.

    The module is left out, rather than guessed, when that value is not certain: the
    node has no such operation, or several, or produces several values, or a node that
    reads it left no operation of its own. The last means the reader was folded into
    this operation, which then holds the reader's value instead, as batch norm folded
    into a conv does. A shape-only node (``view``) with no operation of its own is
    looked through to the node it reshapes, so the value returned has the producer's
    shape.

    The program's output maps are not used: they list every operation that carries a
    node's location and leaves its cluster, which holds for the downstream operation
    that absorbed the node as much as for the one that computes it.

    Args:
        program: Program to read.
        torch_graph: The FX nodes behind it, from `TorchConverter.torch_graph`.

    Returns:
        Module path to the sorted ids of the operations holding its output, for every
        module whose output is certain.

    """
    realizing: dict[int, list[int]] = {}
    ops = {}
    for op_id, operation in _build_coreai_op_map(program).items():
        if operation.name in STRUCTURAL_OPS or operation.name == "coreai.constant":
            continue
        ids = operation_torch_ids(operation)
        if ids:
            ops[op_id] = operation
            realizing.setdefault(max(ids), []).append(op_id)
    realized_by = {op_id: node for node, held in realizing.items() for op_id in held}

    def read_outside(op_id: int) -> bool:
        """Whether something realizing another node, or the program, reads the op."""
        return any(
            user.name == "coreai.output"
            or realized_by.get(get_operation_id(user)) != realized_by[op_id]
            for result in ops[op_id].results
            for user in (use.owner for use in result.uses)
        )

    def accounted(node_id: int) -> bool:
        """Whether a reader of a value has its own operation, so it did not fold in."""
        node = torch_graph[node_id]
        if node.op == "output" or realizing.get(node_id):
            return True
        return _is_shape_only(node) and all(accounted(user) for user in node.users)

    def value_of(node: TorchNode) -> int | None:
        """The Core AI operation holding *node*'s value, None when not certain."""
        while True:
            if node.multi_output:
                return None
            holders = [op for op in realizing.get(node.id, []) if read_outside(op)]
            if holders:
                if len(holders) != 1 or not all(map(accounted, node.users)):
                    return None
                return holders[0]
            # No operation of its own: only a shape-only node still has its input's values.
            if not _is_shape_only(node) or len(node.inputs) != 1:
                return None
            node = torch_graph[node.inputs[0]]

    def inside(node: TorchNode, module: ModulePath) -> bool:
        return node.op == "call_function" and node.stack[: len(module)] == module

    nodes = [node for node in torch_graph.values() if node.op == "call_function"]
    modules = {
        node.stack[:depth]
        for node in nodes
        for depth in range(1, len(node.stack) + 1)
    }
    outputs: dict[ModulePath, list[int]] = {}
    for module in sorted(modules):
        boundary = [
            node
            for node in nodes
            if inside(node, module)
            and any(not inside(torch_graph[user], module) for user in node.users)
        ]
        values = [value_of(node) for node in boundary]
        if boundary and None not in values:
            outputs[module] = sorted(set(values))  # type: ignore[arg-type]
    return outputs
