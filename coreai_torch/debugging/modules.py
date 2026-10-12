# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Grouping a per-operation finding by the module instance it came from.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from coreai.authoring import AIProgram
from typing_extensions import Self

from .source_annotator import (
    ModulePath,
    module_paths_by_op_id,
    source_locations_by_op_id,
)
from .utils import LocationInfo, split_module_frame

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
