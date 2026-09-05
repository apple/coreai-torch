# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Structural diff on NetworkX graphs for AIProgram and PyTorch operations.

This module provides functionality to build NetworkX graphs from both AIProgram
operations and PyTorch FX graphs, and compute structural diffs using graph
isomorphism, with enhanced textual output showing aligned matches and
IR-centric views.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field, fields
from enum import Enum
from io import StringIO
from typing import Any, TextIO

import networkx as nx  # type: ignore[import-untyped]
import torch
from coreai._compiler.ir import Block, Operation, Region, Value
from coreai.authoring import AIProgram

from .debug_info import get_operation_id
from .graph_match import (
    _OP_NODE,
    UNSTABLE_ATTRIBUTES,
    Alignment,
    Ambiguity,
    Label,
    WeightPolicy,
    _graph_blobs,
    align,
    node_labels,
    responsible_op,
)
from .table_writer import _Column, _Row, _TableSpec, _write_table
from .utils import _collect_entry_points, _composite_label, _plain

# ---------------------------------------------------------------------------
# Regex patterns for composite diffing
# ---------------------------------------------------------------------------

_CALLEE_SYMBOL_RE = re.compile(r"<@([^>]+)>")


class OpDiffType(Enum):
    """Type of operation difference in structural diff."""

    ALIGNED = "aligned"  # Structurally identical
    MODIFIED = "modified"  # Corresponds, but is not identical
    REMOVED = "removed"  # Only in source
    ADDED = "added"  # Only in target


class _AIProgramGraphBuilder:
    """Helper class to build NetworkX graph from program operations."""

    def __init__(self) -> None:
        """Initialize the graph builder."""
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.value_to_node: dict[Value, int] = {}

    def build(self, root_op: Operation) -> nx.DiGraph:
        """Build the graph from root operation."""
        self._process_operation(root_op)
        return self.graph

    def _get_next_id(self) -> int:
        """Generate a unique node ID."""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def _add_value_node(self, value: Value, value_type: str) -> int:
        """Add a value node to the graph if not already present."""
        if value not in self.value_to_node:
            node_id = self._get_next_id()
            self.value_to_node[value] = node_id
            self.graph.add_node(
                node_id,
                type="value",
                value_type=value_type,
                ir_type=str(value.type),
                ir_object=value,
            )
        return self.value_to_node[value]

    def _process_block(self, block: Block, block_node_id: int) -> None:
        """Process a block: add block arguments and operations."""
        # Add block arguments as value nodes
        for arg_idx, arg in enumerate(block.arguments):  # type: ignore[var-annotated, arg-type]
            arg_node_id = self._add_value_node(arg, "block_arg")
            self.graph.add_edge(
                block_node_id,
                arg_node_id,
                edge_type="block_arg",
                index=arg_idx,
            )

        # Process operations in the block
        for op in block.operations:
            self._process_operation(op)  # type: ignore[arg-type]

    def _process_region(
        self,
        region: Region,
        parent_op_id: int,
        region_idx: int,
    ) -> None:
        """Process a region: add region node, blocks, and recurse."""
        # Add region node
        region_node_id = self._get_next_id()
        self.graph.add_node(
            region_node_id,
            type="region",
            index=region_idx,
            ir_object=region,
        )

        # Add contains_region edge
        self.graph.add_edge(
            parent_op_id,
            region_node_id,
            edge_type="contains_region",
            index=region_idx,
        )

        # Process blocks in the region
        for block_idx, block in enumerate(region.blocks):
            self._process_block_in_region(region_node_id, block, block_idx)

    def _process_block_in_region(
        self,
        region_node_id: int,
        block: Block,
        block_idx: int,
    ) -> None:
        """Add block node and process it."""
        block_node_id = self._get_next_id()
        self.graph.add_node(
            block_node_id,
            type="block",
            index=block_idx,
            ir_object=block,
        )

        # Add contains_block edge
        self.graph.add_edge(
            region_node_id,
            block_node_id,
            edge_type="contains_block",
            index=block_idx,
        )

        # Process the block
        self._process_block(block, block_node_id)

    def _process_operation(self, operation: Operation) -> int:
        """Process an operation: add op node, results, operands, and recurse into regions."""
        # Add operation node
        op_node_id = self._get_next_id()
        self.graph.add_node(
            op_node_id,
            type="op",
            op_name=operation.name,
            ir_object=operation,
        )

        # Add operation results
        self._add_operation_results(operation, op_node_id)

        # Add operand edges
        self._add_operation_operands(operation, op_node_id)

        # Recurse into nested regions
        for region_idx, region in enumerate(operation.regions):
            self._process_region(region, op_node_id, region_idx)

        return op_node_id

    def _add_operation_results(self, operation: Operation, op_node_id: int) -> None:
        """Add operation results as value nodes with defines edges."""
        for result_idx, result in enumerate(operation.results):  # type: ignore[var-annotated, arg-type]
            result_node_id = self._add_value_node(result, "op_result")
            self.graph.add_edge(
                op_node_id,
                result_node_id,
                edge_type="defines",
                index=result_idx,
            )

    def _add_operation_operands(self, operation: Operation, op_node_id: int) -> None:
        """Add operand edges from source values to this operation."""
        for operand_idx, operand in enumerate(operation.operands):  # type: ignore[var-annotated, arg-type]
            operand_node_id = self._add_value_node(operand, "operand")
            self.graph.add_edge(
                operand_node_id,
                op_node_id,
                edge_type="operand",
                index=operand_idx,
            )


@dataclass
class GraphDiffSummary:
    """Summary statistics for graph comparison."""

    source_node_count: int
    target_node_count: int
    source_edge_count: int
    target_edge_count: int
    mapped_node_count: int = 0
    modified_node_count: int = 0
    unmapped_source_node_count: int = 0
    unmapped_target_node_count: int = 0
    unmapped_source_edge_count: int = 0
    unmapped_target_edge_count: int = 0
    ambiguous_node_count: int = 0
    """How many of the counts above a tie-break decided rather than the graphs; see
    `GraphDiff.ambiguity`. Not a subtotal of any single one -- it spans mapped,
    unmapped-source and unmapped-target alike."""

    def to_dict(self) -> dict[str, Any]:
        """
        Return the summary counts as plain values.

        Returns:
            Every count, unchanged -- they are already plain integers.

        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class GraphDiff:
    """
    Result of structural graph comparison.

    Attributes:
        is_isomorphic: Whether the graphs are provably the same graph. A complete
            verified correspondence accounting for every edge, not a search result:
            see `graph_match._is_isomorphism` for what the proof rests on.
        source_to_target_mapping: Every source node that has a counterpart, mapped
            to it -- identical and modified alike. `modified_node_pairs` is what
            separates the two.
        target_to_source_mapping: The same correspondence, inverted
        modified_node_pairs: Nodes that correspond but are not identical, as
            `(source, target)`. The same op, wired or configured differently -- a
            rewiring, a reshape, a changed attribute. Neither added nor removed,
            because it is still there and still in the same place in the source.
        unmapped_source_nodes: Node IDs with no counterpart in the target
        unmapped_target_nodes: Node IDs with no counterpart in the source
        unmapped_source_edges: Source edges with no corresponding target edge
        unmapped_target_edges: Target edges with no corresponding source edge
        summary: Summary statistics
        source_graph: Source NetworkX graph with IR references in nodes
        target_graph: Target NetworkX graph with IR references in nodes
        weights: The `WeightPolicy` this diff was computed under. `write_diff` reads
            it back to label a modified pair's *reason* under the same policy that
            decided the pair was not identical -- `node_labels` defaults to `IGNORE`,
            and a reason computed under a different policy than the one that rejected
            the pair can describe the wrong thing: an IGNORE-computed label elides a
            parameter's value, so a DIGEST-only difference is invisible to it and the
            true reason (`attributes: ...`) is never reached.
        ambiguity: Which of the node ids above rest on a tie-break among
            indistinguishable nodes rather than on anything in the graphs. Its
            `removed`/`added` are the ones worth reading first: those are differences
            reported only because an interchangeable group was larger on one side.
            See `graph_match.Ambiguity`.
        ignore_attributes: The attribute names left out of an op's identity when this
            diff was computed. Recorded for the same reason as `weights`, and it is
            the same defect: `write_diff` recomputes labels to name a modified pair's
            reason, and `node_labels` defaults to `UNSTABLE_ATTRIBUTES`. Computed
            under a wider set than `align` used, the recomputation finds a difference
            in an attribute `align` had agreed to overlook and reports *that* as the
            reason -- naming a field the caller deliberately excluded, for a pair
            rejected over something else entirely.

    Lifetime: the two graphs hold `ir_object` references owned by the programs they
    were built from, and comparison reads them lazily. A `GraphDiff` is therefore
    only usable while both programs are alive -- dropping an `AIProgram` and keeping
    its diff segfaults the interpreter, with no Python traceback to say why.

    """

    is_isomorphic: bool
    source_to_target_mapping: dict[int, int]
    target_to_source_mapping: dict[int, int]
    modified_node_pairs: list[tuple[int, int]]
    unmapped_source_nodes: list[int]
    unmapped_target_nodes: list[int]
    unmapped_source_edges: list[tuple[int, int]]
    unmapped_target_edges: list[tuple[int, int]]
    summary: GraphDiffSummary
    source_graph: nx.DiGraph
    target_graph: nx.DiGraph
    weights: WeightPolicy = WeightPolicy.IGNORE
    ambiguity: Ambiguity = field(default_factory=Ambiguity)
    ignore_attributes: frozenset[str] = frozenset(UNSTABLE_ATTRIBUTES)

    def to_dict(self) -> dict[str, Any]:
        """
        Return the diff as plain values.

        The two graphs are represented by :attr:`summary`'s node and edge counts
        rather than serialised. They hold live ``ir_object`` references owned by the
        programs, so they cannot be serialised and they tie the diff to those
        programs' lifetime -- ``dataclasses.asdict`` on a `GraphDiff` raises ``cannot
        pickle 'GraphOp' object`` for exactly this reason. Everything the diff
        *concluded* is here; only the inputs it concluded it from are not.

        Returns:
            The correspondence, what it leaves over, the counts, and the identity the
            comparison was made under -- policy, ignored attributes and ambiguity.

        """
        return {
            "is_isomorphic": self.is_isomorphic,
            "source_to_target_mapping": _plain(self.source_to_target_mapping),
            "target_to_source_mapping": _plain(self.target_to_source_mapping),
            "modified_node_pairs": [list(pair) for pair in self.modified_node_pairs],
            "unmapped_source_nodes": _plain(self.unmapped_source_nodes),
            "unmapped_target_nodes": _plain(self.unmapped_target_nodes),
            "unmapped_source_edges": [
                list(edge) for edge in self.unmapped_source_edges
            ],
            "unmapped_target_edges": [
                list(edge) for edge in self.unmapped_target_edges
            ],
            "summary": self.summary.to_dict(),
            "weights": _plain(self.weights),
            "ignore_attributes": _plain(self.ignore_attributes),
            "ambiguity": self.ambiguity.to_dict(),
        }


# ---------------------------------------------------------------------------
# Edge correspondence
# ---------------------------------------------------------------------------


def _slot(data: dict[str, Any]) -> tuple[str, int]:
    """
    Which operand position an edge occupies.

    Normalised the same way `graph_match` normalises it, so a graph flavour that
    omits either attribute compares equal to `("", 0)` rather than raising.
    """
    return (str(data.get("edge_type", "")), int(data.get("index", 0)))


def _unmapped_edges(
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    mapping: dict[int, int],
) -> list[tuple[int, int]]:
    """
    Source edges with no counterpart in the target under `mapping`.

    An edge corresponds only if both endpoints map, the mapped pair is an edge on
    the other side, *and* it occupies the same slot. Checking only that both
    endpoints were mapped would count an edge as common whenever its nodes were, so a
    pure rewiring would report every edge as common.

    Called with the full correspondence, modified pairs included: a modified op
    still exists on the other side, so its untouched operand edges genuinely
    correspond and only the moved ones do not.

    Args:
        source_graph: The graph whose edges are being classified.
        target_graph: The graph they are looked for in.
        mapping: Source node id -> target node id.

    Returns:
        The source edges with no corresponding target edge.

    """
    unmapped: list[tuple[int, int]] = []
    for producer, consumer, data in source_graph.edges(data=True):
        image_producer = mapping.get(producer)
        image_consumer = mapping.get(consumer)
        if image_producer is None or image_consumer is None:
            unmapped.append((producer, consumer))
            continue

        if not target_graph.has_edge(image_producer, image_consumer):
            unmapped.append((producer, consumer))
            continue

        image = target_graph.edges[image_producer, image_consumer]
        if _slot(image) != _slot(data):
            unmapped.append((producer, consumer))

    return unmapped


# ---------------------------------------------------------------------------
# Core diff computation
# ---------------------------------------------------------------------------


def compute_graph_diff(
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    *,
    weights: WeightPolicy = WeightPolicy.IGNORE,
    ignore_attributes: Collection[str] = UNSTABLE_ATTRIBUTES,
) -> GraphDiff:
    """
    Compute structural differences between two graphs.

    Which node became which comes from `graph_match.align`, which labels each node
    with what makes it that op, fingerprints bottom-up, anchors on those hashes,
    propagates along dataflow and verifies every pair. Linear, and no search.

    Args:
        source_graph: Source (reference/expected) graph
        target_graph: Target (actual/test) graph
        weights: Whether parameter *values* count towards a node's identity. Off by
            default: converting a model twice re-initialises its parameters, so
            comparing values reports every diff as a total rewrite. Shapes and
            dtypes are compared either way.
        ignore_attributes: Attribute names left out of a node's identity, defaulting
            to `UNSTABLE_ATTRIBUTES`. Recorded on the result so `write_diff` names a
            modified pair's reason under the same notion of identity that rejected
            the pair.

    Returns:
        GraphDiff object describing the correspondence and what it leaves over

    """
    alignment = align(
        source_graph,
        target_graph,
        weights=weights,
        ignore_attributes=ignore_attributes,
    )

    # Every source node with a counterpart, identical or not. Modified pairs belong
    # here: composite bodies are paired by walking this mapping, so leaving a
    # rewired `coreai.invoke` out of it would leave its callee never paired and
    # never reported. `modified_node_pairs` is what tells the two apart.
    source_to_target = {**alignment.mapping, **dict(alignment.modified)}
    target_to_source = {v: k for k, v in source_to_target.items()}

    unmapped_source_edges = _unmapped_edges(
        source_graph, target_graph, source_to_target
    )
    unmapped_target_edges = _unmapped_edges(
        target_graph, source_graph, target_to_source
    )

    summary = GraphDiffSummary(
        source_node_count=source_graph.number_of_nodes(),
        target_node_count=target_graph.number_of_nodes(),
        source_edge_count=source_graph.number_of_edges(),
        target_edge_count=target_graph.number_of_edges(),
        # Identically matched only, so that mapped + modified + unmapped adds up to
        # the source node count and "common subgraph" does not count a rewired op as
        # common. It is therefore smaller than `len(source_to_target_mapping)`
        # whenever anything is modified.
        mapped_node_count=len(alignment.mapping),
        modified_node_count=len(alignment.modified),
        unmapped_source_node_count=len(alignment.removed),
        unmapped_target_node_count=len(alignment.added),
        unmapped_source_edge_count=len(unmapped_source_edges),
        unmapped_target_edge_count=len(unmapped_target_edges),
        ambiguous_node_count=alignment.ambiguity.count,
    )

    return GraphDiff(
        is_isomorphic=alignment.identical,
        source_to_target_mapping=source_to_target,
        target_to_source_mapping=target_to_source,
        modified_node_pairs=alignment.modified,
        unmapped_source_nodes=alignment.removed,
        unmapped_target_nodes=alignment.added,
        unmapped_source_edges=unmapped_source_edges,
        unmapped_target_edges=unmapped_target_edges,
        summary=summary,
        source_graph=source_graph,
        target_graph=target_graph,
        weights=weights,
        ambiguity=alignment.ambiguity,
        ignore_attributes=frozenset(ignore_attributes),
    )


# ---------------------------------------------------------------------------
# Module-level graph builder
# ---------------------------------------------------------------------------


def _build_module_graph(module: Any, entry_point: str | None = None) -> nx.DiGraph:
    """Build a unified NetworkX graph from coreai.graph ops in a program module.

    When entry_point is None, all coreai.graph ops are included in a single graph.
    When entry_point is specified, only that graph is included.

    Args:
        module: Program module containing coreai.graph operations
        entry_point: Optional name of entry point to build graph for

    Returns:
        NetworkX directed graph representing the program structure

    Raises:
        ValueError: If entry_point is specified but not found in the module

    """
    builder = _AIProgramGraphBuilder()
    found_entry_point = False

    for op in module.body.operations:
        if op.name != "coreai.graph":
            continue
        if not hasattr(op, "sym_name"):
            continue
        if entry_point is not None and op.sym_name.value != entry_point:
            continue
        found_entry_point = True
        builder._process_operation(op)

    if entry_point is not None and not found_entry_point:
        msg = f"Entry point '{entry_point}' not found in program"
        raise ValueError(msg)

    return builder.graph


# ---------------------------------------------------------------------------
# Composite diffing helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpIdAlignment:
    """Which Core AI operation of one program became which of another."""

    mapping: dict[int, int] = field(default_factory=dict)
    """Before op id -> after op id, for operations that correspond."""

    modified: set[int] = field(default_factory=set)
    """Before ids in :attr:`mapping` whose counterpart is the same operation wired or
    configured differently, so a difference in it may be the rewiring itself."""

    removed: list[int] = field(default_factory=list)
    """Before ids with no counterpart."""

    added: list[int] = field(default_factory=list)
    """After ids with no counterpart."""

    identical: bool = False
    """Whether the two programs are provably the same graph. Any difference measured
    between runs of an identical program is noise, which is what makes such a pair
    worth running deliberately."""

    ambiguity: Ambiguity = field(default_factory=Ambiguity)
    """Which of the op ids above rest on a tie-break among operations nothing in
    either program tells apart, rather than on evidence -- in op ids, like the fields
    it qualifies. Read `ambiguity.removed` before treating :attr:`removed` as the list
    of operations an edit deleted. See `graph_match.Ambiguity`."""

    def to_dict(self) -> dict[str, Any]:
        """
        Return the correspondence as plain values.

        Returns:
            The mapping and what it leaves over, in op ids, with the ambiguity that
            qualifies each field.

        """
        return {
            "mapping": _plain(self.mapping),
            "modified": _plain(self.modified),
            "removed": _plain(self.removed),
            "added": _plain(self.added),
            "identical": self.identical,
            "ambiguity": self.ambiguity.to_dict(),
        }


def _coreai_ids(graph: nx.DiGraph) -> dict[int, int]:
    """
    Map each operation node of *graph* to the Core AI op id it carries.

    Node ids come from a counter over the whole build -- values, regions and blocks
    included -- so they are neither op ids nor comparable between programs. The
    operation each op node stores is what carries the id.

    Args:
        graph: Graph built by :func:`_build_module_graph`.

    Returns:
        Node id -> Core AI op id, for operation nodes carrying one.

    """
    ids: dict[int, int] = {}
    for node, data in graph.nodes(data=True):
        if data.get("type") != _OP_NODE:
            continue
        operation = data.get("ir_object")
        if operation is None:
            continue
        op_id = get_operation_id(operation)
        if op_id is not None:
            ids[node] = op_id
    return ids


def op_id_alignment(
    before: AIProgram,
    after: AIProgram,
    entry_point: str = "main",
    *,
    weights: WeightPolicy = WeightPolicy.IGNORE,
    ignore_attributes: Collection[str] = UNSTABLE_ATTRIBUTES,
) -> OpIdAlignment:
    """
    Which Core AI operation of *before* became which of *after*.

    Core AI op ids are positional, so inserting a layer renumbers everything after
    it. Comparing two programs by raw op id therefore compares unrelated operations,
    and does so quietly: most ids survive an edit unchanged.

    Args:
        before: The "before" program.
        after: The "after" program.
        entry_point: Function to compare.
        weights: Whether parameter *values* count towards an operation's identity.
            Ignored by default: a rebuild re-initialises them, which would report
            every edit as a total rewrite.
        ignore_attributes: Attribute names left out of an operation's identity,
            defaulting to `UNSTABLE_ATTRIBUTES`. Name one here to let operations that
            differ only in it pair and verify as identical.

    Returns:
        The correspondence, and what it leaves over.

    """
    before_graph = _build_module_graph(before._mlir_module, entry_point)
    after_graph = _build_module_graph(after._mlir_module, entry_point)
    alignment = align(
        before_graph,
        after_graph,
        weights=weights,
        ignore_attributes=ignore_attributes,
    )

    before_ids = _coreai_ids(before_graph)
    after_ids = _coreai_ids(after_graph)

    mapping: dict[int, int] = {}
    modified: set[int] = set()
    # Exact pairs first, then modified ones: a modified pair is the same operation
    # rewired, so it is a correspondence rather than a removal plus an addition --
    # but which pairs those are is worth reporting.
    for source, target in alignment.mapping.items():
        if source in before_ids and target in after_ids:
            mapping[before_ids[source]] = after_ids[target]
    for source, target in alignment.modified:
        if source in before_ids and target in after_ids:
            mapping[before_ids[source]] = after_ids[target]
            modified.add(before_ids[source])

    removed = sorted(
        {before_ids[n] for n in alignment.removed if n in before_ids} - set(mapping)
    )
    added = sorted(
        {after_ids[n] for n in alignment.added if n in after_ids}
        - set(mapping.values())
    )

    return OpIdAlignment(
        mapping=mapping,
        modified=modified,
        removed=removed,
        added=added,
        identical=alignment.identical,
        ambiguity=_ambiguity_in_op_ids(
            alignment, before_ids, after_ids, mapping, removed, added
        ),
    )


def _ambiguity_in_op_ids(
    alignment: Alignment,
    before_ids: dict[int, int],
    after_ids: dict[int, int],
    mapping: dict[int, int],
    removed: list[int],
    added: list[int],
) -> Ambiguity:
    """
    Restate a node-level `Ambiguity` in op ids, keeping it a subset of what it marks.

    Two things shrink it. Not every graph node is an operation -- values, regions and
    blocks are nodes too, and only operations carry an id -- and the op-id fields drop
    a node whose id is accounted for elsewhere, `removed` subtracting anything already
    in `mapping`. Intersecting is what keeps `ambiguity.removed <= removed` true, so a
    caller can subtract one from the other without getting a negative count.

    Args:
        alignment: The node-level alignment, carrying the ambiguity to restate.
        before_ids: Source node id -> op id.
        after_ids: Target node id -> op id.
        mapping: The finished op-id correspondence.
        removed: The finished removed op ids.
        added: The finished added op ids.

    Returns:
        The same ambiguity, in op ids.

    """
    ambiguity = alignment.ambiguity

    return Ambiguity(
        paired=frozenset(
            {before_ids[n] for n in ambiguity.paired if n in before_ids} & set(mapping)
        ),
        removed=frozenset(
            {before_ids[n] for n in ambiguity.removed if n in before_ids} & set(removed)
        ),
        added=frozenset(
            {after_ids[n] for n in ambiguity.added if n in after_ids} & set(added)
        ),
    )


def _extract_invoke_callee(graph: nx.DiGraph, node_id: int) -> str | None:
    """Extract callee symbol name from a coreai.invoke node."""
    node = graph.nodes[node_id]
    if node.get("op_name") != "coreai.invoke":
        return None
    ir_op = node.get("ir_object")
    if ir_op is None:
        return None
    try:
        callee_str = str(ir_op.attributes["callee"])
    except (KeyError, AttributeError):
        return None
    match = _CALLEE_SYMBOL_RE.search(callee_str)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_verdict(
    diff: GraphDiff,
    source_graph: nx.DiGraph,
) -> list[tuple[int, str]]:
    """Format the verdict line showing isomorphism status."""
    lines: list[tuple[int, str]] = []
    summary = diff.summary

    if diff.is_isomorphic:
        lines.append((0, "\u2713 Graphs are ISOMORPHIC."))
    else:
        lines.append((0, "\u2717 Graphs are NOT isomorphic."))
        lines.append(
            (
                1,
                f"Common subgraph: {summary.mapped_node_count}/{summary.source_node_count} nodes, "
                f"{summary.source_edge_count - summary.unmapped_source_edge_count}/{summary.source_edge_count} edges.",
            ),
        )

        # Find first differing op
        source_ops = [
            n
            for n in diff.unmapped_source_nodes
            if source_graph.nodes[n].get("type") == "op"
        ]
        if source_ops:
            first_diff = source_ops[0]
            op_name = source_graph.nodes[first_diff].get("op_name", "unknown")
            lines.append(
                (
                    1,
                    f"First differing op: source node {first_diff} ({op_name}) has no match.",
                ),
            )

    return lines


def _format_summary(diff: GraphDiff) -> list[tuple[int, str]]:
    """Format the detailed summary section."""
    lines: list[tuple[int, str]] = []
    summary = diff.summary

    lines.append((0, "Summary:"))
    lines.append(
        (
            1,
            f"Source graph: {summary.source_node_count} nodes, {summary.source_edge_count} edges",
        ),
    )
    lines.append(
        (
            1,
            f"Target graph: {summary.target_node_count} nodes, {summary.target_edge_count} edges",
        ),
    )

    if not diff.is_isomorphic:
        lines.append((1, f"Mapped nodes: {summary.mapped_node_count}"))
        lines.append(
            (1, f"Unmapped in source: {summary.unmapped_source_node_count} nodes"),
        )
        lines.append(
            (1, f"Unmapped in target: {summary.unmapped_target_node_count} nodes"),
        )
        lines.extend(_format_ambiguity(diff))

    return lines


def _format_ambiguity(diff: GraphDiff) -> list[tuple[int, str]]:
    """
    Say which of the counts above a tie-break produced, if any did.

    Printed with the summary rather than beside each op, because the caution is about
    the diff as a whole: the reader's question is whether "3 removed" names the layer
    that changed, and the answer is one line. Suppressed entirely when nothing was
    arbitrary, so its presence means something.
    """
    ambiguity = diff.ambiguity
    if not ambiguity:
        return []

    lines = [
        (
            1,
            f"Decided by tie-break: {ambiguity.count} nodes "
            f"({len(ambiguity.paired)} paired among equals, "
            f"{len(ambiguity.removed)} removed, {len(ambiguity.added)} added)",
        ),
    ]
    if ambiguity.removed or ambiguity.added:
        lines.append(
            (
                2,
                "Those removals and additions are not evidence of an edit, and are "
                "a lower bound: a node displaced by someone else's tie-break is not "
                "itself ambiguous and is not counted.",
            ),
        )
    if diff.weights is WeightPolicy.IGNORE:
        lines.append(
            (
                2,
                "Re-run with weights=WeightPolicy.DIGEST to compare parameter "
                "values, which tells apart what ignoring them made alike.",
            ),
        )

    return lines


def _incoming(graph: nx.DiGraph, node_id: int) -> set[tuple[str, int, int]]:
    """This node's incoming edges as `(edge_type, index, producer)`."""
    return {
        (*_slot(data), producer)
        for producer, _, data in graph.in_edges(node_id, data=True)
    }


def _describe_modification(
    src_id: int,
    tgt_id: int,
    labels: tuple[dict[int, Label], dict[int, Label]],
    graphs: tuple[nx.DiGraph, nx.DiGraph],
    mapping: dict[int, int],
) -> str:
    """
    Why a corresponding pair is not identical, in the terms the comparison used.

    Reports the first label field that differs, or the operand slots whose producer
    moved -- which is exactly what verification rejected the pair on. This replaces
    counting operand/result/region edges per graph flavour and reporting the deltas:
    those counts are equal for a rewiring, which is the case most worth naming.

    Args:
        src_id: Source node of the pair.
        tgt_id: Target node of the pair.
        labels: Node labels for `(source, target)`, computed once by the caller.
        graphs: The `(source, target)` graphs.
        mapping: The full source-to-target correspondence.

    Returns:
        A short description, empty if nothing can be pinned down.

    """
    source_labels, target_labels = labels
    source_graph, target_graph = graphs

    source_label = source_labels.get(src_id, Label())
    target_label = target_labels.get(tgt_id, Label())
    if source_label != target_label:
        for field in fields(Label):
            before = getattr(source_label, field.name)
            after = getattr(target_label, field.name)
            if before != after:
                return f"{field.name}: {before or '-'}\u2192{after or '-'}"

        return "label differs"

    target_incoming = _incoming(target_graph, tgt_id)
    moved = sorted(
        index
        for edge_type, index, producer in _incoming(source_graph, src_id)
        if (edge_type, index, mapping.get(producer, -1)) not in target_incoming
    )
    if moved:
        return "rewired: " + ", ".join(f"operand {index}" for index in moved)

    # Nothing else is left. Verification rejected this pair, and it rejects only on
    # a label mismatch or an operand mismatch. `labels` is computed under the diff's
    # own identity -- policy, ignored attributes and blob digests alike (see
    # `_format_unified_ops_table`) -- so a genuine value-only difference is already
    # caught above as an `attributes: ...` mismatch. Reaching here means this
    # after-the-fact recomputation disagrees with `align`'s own check, which should
    # not happen; the wording is kept because a value difference is the likeliest
    # cause if it ever does.
    return "parameter values differ"


def _is_one_op_changed(
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    src_id: int,
    tgt_id: int,
) -> bool:
    """
    Whether a corresponding pair is one op that changed, rather than two ops.

    `modified` means *the same operation*, wired or configured differently. A pair
    whose operations do not even share a name is not that: it is one op gone and
    another arrived, and reporting it as a modification hides what the first one was.

    The distinction matters because a difference often lands on a **value** node --
    `relu`'s result pairs with `sigmoid`'s, since they have the same type and the same
    consumer -- and resolving that pair to the operations behind it would otherwise
    collapse a removal and an addition into "sigmoid was modified", with no mention of
    the relu.
    """
    return source_graph.nodes[src_id].get("op_name") == target_graph.nodes[tgt_id].get(
        "op_name"
    )


def _format_unified_ops_table(
    diff: GraphDiff,
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    max_items: int | None = None,
) -> _TableSpec | None:
    """
    Build the unified operations table, styling removed/added ops.

    Args:
        diff: GraphDiff result to render.
        source_graph: Source (reference) graph.
        target_graph: Target (actual) graph.
        max_items: Maximum number of rows to include, or None for all.

    Returns:
        The table, or None when there are no differing operations to show.

    """
    rows: list[tuple[str, str, str, str, str, str]] = []

    # A modification is one fact about one pair of ops, so it claims both sides:
    # an op that still exists but computes something different is *modified*, and
    # listing it as removed or added as well would say two contradictory things
    # about one op.
    claimed_source: set[int] = set()
    claimed_target: set[int] = set()
    # Recomputed under exactly the identity `align` used -- policy, ignored
    # attributes and, for DIGEST_PORTABLE, the blob digests. Any of the three left at
    # its default names the wrong reason for a modified pair: `node_labels` defaults
    # to IGNORE, which elides a parameter's value, so a DIGEST-only difference would
    # never reach the "attributes: ..." branch; a default `ignore_attributes` finds a
    # difference in a field the caller asked to overlook and reports that instead of
    # the real one; and without blobs, DIGEST_PORTABLE compares resource handles,
    # which differ between two assets holding identical weights.
    blobs = (
        (_graph_blobs(source_graph), _graph_blobs(target_graph))
        if diff.weights is WeightPolicy.DIGEST_PORTABLE
        else ({}, {})
    )
    labels = (
        node_labels(source_graph, diff.weights, blobs[0], diff.ignore_attributes),
        node_labels(target_graph, diff.weights, blobs[1], diff.ignore_attributes),
    )

    for src_node, tgt_node in diff.modified_node_pairs:
        src_id = responsible_op(source_graph, src_node)
        tgt_id = responsible_op(target_graph, tgt_node)
        if src_id is None or tgt_id is None:
            continue
        if src_id in claimed_source or tgt_id in claimed_target:
            continue
        if not _is_one_op_changed(source_graph, target_graph, src_id, tgt_id):
            # Two different ops. Leave them to the removed and added passes, which
            # will name both.
            continue

        claimed_source.add(src_id)
        claimed_target.add(tgt_id)
        rows.append(
            (
                str(src_id),
                str(tgt_id),
                OpDiffType.MODIFIED.value,
                source_graph.nodes[src_id].get("op_name", "unknown"),
                target_graph.nodes[tgt_id].get("op_name", "unknown"),
                # Described against the *responsible ops*, not the raw pair. A
                # difference on a leaf op (a constant, no operands) often lands on
                # its result value instead: a value's Label never carries
                # `attributes`, so the op producing it is not what was compared and
                # a real value change is invisible, falling through to a vaguer
                # "rewired"/"parameter values differ" guess. The op just confirmed
                # `_is_one_op_changed` is the same op on both sides, whether or not
                # `align` ever paired it -- `_describe_modification`'s first check
                # only needs the two labels, not a correspondence entry for either
                # id, so describing the op directly is always at least as precise.
                _describe_modification(
                    src_id,
                    tgt_id,
                    labels,
                    (source_graph, target_graph),
                    diff.source_to_target_mapping,
                ),
            )
        )

    for nodes, graph, claimed, removed in (
        (diff.unmapped_source_nodes, source_graph, claimed_source, True),
        (diff.unmapped_target_nodes, target_graph, claimed_target, False),
    ):
        for node in nodes:
            op_id = responsible_op(graph, node)
            if op_id is None or op_id in claimed:
                continue

            claimed.add(op_id)
            op_name = graph.nodes[op_id].get("op_name", "unknown")
            rows.append(
                (
                    str(op_id),
                    "-",
                    OpDiffType.REMOVED.value,
                    op_name,
                    "",
                    "no match in target",
                )
                if removed
                else (
                    "-",
                    str(op_id),
                    OpDiffType.ADDED.value,
                    "",
                    op_name,
                    "no match in source",
                )
            )

    if not rows:
        return None

    items_to_show = rows if max_items is None else rows[:max_items]
    caption = None
    if max_items is not None and len(rows) > max_items:
        caption = f"... and {len(rows) - max_items} more operations"

    spec = _TableSpec(
        title="Operations Diff Table:",
        columns=(
            _Column("src_id", justify="right"),
            _Column("tgt_id", justify="right"),
            _Column("status"),
            _Column("src_op"),
            _Column("tgt_op"),
            _Column("details"),
        ),
        caption=caption,
    )
    for row in items_to_show:
        status = row[2]
        if status == OpDiffType.REMOVED.value:
            style = "white on rgb(50,10,10)"
        elif status == OpDiffType.ADDED.value:
            style = "white on rgb(10,40,10)"
        else:
            style = ""
        spec.add(_Row(cells=row, style=style))
    return spec


def _apply_indentation(lines: list[tuple[int, str]], indent_size: int = 2) -> list[str]:
    """Apply indentation to lines based on their indent level."""
    return [" " * (level * indent_size) + text for level, text in lines]


def write_diff(
    diff: GraphDiff,
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    *,
    output: TextIO | None = None,
    indent_size: int = 2,
    max_items: int | None = None,
) -> None:
    """
    Write isomorphism-based structural diff as human-readable text.

    Shows:
    - Verdict line with summary
    - Side-by-side aligned matches table
    - IR-centric diff views (ops, uses, structural changes)

    Args:
        diff: GraphDiff result from compute_graph_diff
        source_graph: Source (reference) graph with IR references in nodes
        target_graph: Target (actual) graph with IR references in nodes
        output: Text stream to write to (default: sys.stdout)
        indent_size: Number of spaces per indentation level (default: 2)
        max_items: Maximum number of items to show per section (default: None, shows all)

    """
    if output is None:
        output = sys.stdout

    # Collect all sections as (indent_level, text) tuples
    lines: list[tuple[int, str]] = []

    # Header
    lines.append((0, "=" * 80))
    lines.append((0, "GRAPH DIFF"))
    lines.append((0, "=" * 80))
    lines.append((0, ""))

    # Verdict
    lines.extend(_format_verdict(diff, source_graph))
    lines.append((0, ""))

    # Summary
    lines.extend(_format_summary(diff))
    lines.append((0, ""))

    # Write everything above the operations table, then the table itself.
    for line in _apply_indentation(lines, indent_size):
        output.write(line + "\n")

    # Unified operations table (shows aligned, removed, and added ops)
    table_limit = None if max_items is None else max_items * 2
    ops_table = _format_unified_ops_table(diff, source_graph, target_graph, table_limit)
    if ops_table is not None:
        _write_table(ops_table, output)
        output.write("\n")

    output.write("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# High-level public API
# ---------------------------------------------------------------------------


def compute_coreai_program_diff(
    source_program: AIProgram,
    target_program: AIProgram,
    *,
    entry_point: str | None = "main",
    weights: WeightPolicy = WeightPolicy.IGNORE,
    ignore_attributes: Collection[str] = UNSTABLE_ATTRIBUTES,
) -> GraphDiff:
    """
    Compute structural diff between two AIPrograms.

    By default (entry_point="main"), compares the main graph entry point.
    Set entry_point=None to compare all coreai.graph ops in the module.

    Args:
        source_program: Source (reference/expected) AIProgram
        target_program: Target (actual/test) AIProgram
        entry_point: Name of the entry point function to compare.
            Use None to compare all graphs in the module.
        weights: Whether parameter *values* count towards a node's identity, so
            that two builds differing only in their weights are reported as
            modified rather than as identical. Off by default: converting a model
            twice re-initialises its parameters, so comparing values would report
            every layer as changed. Shapes and dtypes are compared either way.

            Only meaningful when **this process** produced both programs. A
            resource-backed parameter is compared by its blob name, which is
            serialised into a `.aimodel` and not recomputed on load, so two assets
            written by two runs carry different names for identical weights and
            `DIGEST` reports them as modified. Nothing here can detect that; a
            caller that loads assets from disk must not offer this option. See
            `graph_match._weight_label`.
        ignore_attributes: Attribute names left out of an operation's identity,
            defaulting to `UNSTABLE_ATTRIBUTES`. Name one here to let operations that
            differ only in it pair *and* verify as identical -- and pass
            `frozenset()` to compare everything, which is how to find out why two
            operations that should match do not.

    Returns:
        GraphDiff object with source_graph and target_graph included

    Raises:
        ValueError: If entry point is not found in either program

    """
    source_graph = _build_module_graph(source_program._mlir_module, entry_point)
    target_graph = _build_module_graph(target_program._mlir_module, entry_point)
    return compute_graph_diff(
        source_graph,
        target_graph,
        weights=weights,
        ignore_attributes=ignore_attributes,
    )


# ---------------------------------------------------------------------------
# Per-graph composite-aware diffing
# ---------------------------------------------------------------------------


def _match_composites_from_main_diff(
    main_diff: GraphDiff,
    source_main_graph: nx.DiGraph,
    target_main_graph: nx.DiGraph,
) -> tuple[list[tuple[str, str]], set[str], set[str]]:
    """Match composite graphs by pairing coreai.invoke ops in the main diff.

    Returns:
        Tuple of (matched_pairs, matched_src_callees, matched_tgt_callees)

    """
    matched_composites: list[tuple[str, str]] = []
    matched_src_callees: set[str] = set()
    matched_tgt_callees: set[str] = set()

    for src_id, tgt_id in main_diff.source_to_target_mapping.items():
        src_callee = _extract_invoke_callee(source_main_graph, src_id)
        tgt_callee = _extract_invoke_callee(target_main_graph, tgt_id)
        if src_callee and tgt_callee:
            matched_composites.append((src_callee, tgt_callee))
            matched_src_callees.add(src_callee)
            matched_tgt_callees.add(tgt_callee)

    return matched_composites, matched_src_callees, matched_tgt_callees


def _diff_matched_composites(
    matched_composites: list[tuple[str, str]],
    source_eps: dict[str, Any],
    target_eps: dict[str, Any],
    source_module: Any,
    target_module: Any,
    weights: WeightPolicy = WeightPolicy.IGNORE,
    ignore_attributes: Collection[str] = UNSTABLE_ATTRIBUTES,
) -> list[tuple[str, GraphDiff | None]]:
    """Diff each matched composite pair and return labeled results."""
    results: list[tuple[str, GraphDiff | None]] = []
    for src_callee, tgt_callee in matched_composites:
        if src_callee not in source_eps or tgt_callee not in target_eps:
            continue
        src_graph = _build_module_graph(source_module, src_callee)
        tgt_graph = _build_module_graph(target_module, tgt_callee)
        diff = compute_graph_diff(
            src_graph,
            tgt_graph,
            weights=weights,
            ignore_attributes=ignore_attributes,
        )
        label = _composite_label(src_callee, source_eps)
        results.append(
            (f"{label} (source: @{src_callee}, target: @{tgt_callee})", diff)
        )
    return results


def _report_unmatched_composites(
    main_diff: GraphDiff,
    source_main_graph: nx.DiGraph,
    target_main_graph: nx.DiGraph,
    source_eps: dict[str, Any],
    target_eps: dict[str, Any],
    matched_src_callees: set[str],
    matched_tgt_callees: set[str],
) -> list[tuple[str, GraphDiff | None]]:
    """Report composites only present in source (removed) or target (added)."""
    results: list[tuple[str, GraphDiff | None]] = []

    for src_id in main_diff.unmapped_source_nodes:
        src_callee = _extract_invoke_callee(source_main_graph, src_id)
        if src_callee and src_callee not in matched_src_callees:
            label = _composite_label(src_callee, source_eps)
            results.append((f"REMOVED composite: {label} (@{src_callee})", None))
            matched_src_callees.add(src_callee)

    for tgt_id in main_diff.unmapped_target_nodes:
        tgt_callee = _extract_invoke_callee(target_main_graph, tgt_id)
        if tgt_callee and tgt_callee not in matched_tgt_callees:
            label = _composite_label(tgt_callee, target_eps)
            results.append((f"ADDED composite: {label} (@{tgt_callee})", None))
            matched_tgt_callees.add(tgt_callee)

    return results


def _report_orphan_composites(
    source_eps: dict[str, Any],
    target_eps: dict[str, Any],
    matched_src_callees: set[str],
    matched_tgt_callees: set[str],
    source_module: Any,
    target_module: Any,
    weights: WeightPolicy = WeightPolicy.IGNORE,
    ignore_attributes: Collection[str] = UNSTABLE_ATTRIBUTES,
) -> list[tuple[str, GraphDiff | None]]:
    """Report composites not referenced by any invoke in main."""
    results: list[tuple[str, GraphDiff | None]] = []

    for sym_name in source_eps:
        if sym_name != "main" and sym_name not in matched_src_callees:
            label = _composite_label(sym_name, source_eps)
            if sym_name in target_eps:
                src_graph = _build_module_graph(source_module, sym_name)
                tgt_graph = _build_module_graph(target_module, sym_name)
                results.append(
                    (
                        f"{label} (unreferenced, @{sym_name})",
                        compute_graph_diff(
                            src_graph,
                            tgt_graph,
                            weights=weights,
                            ignore_attributes=ignore_attributes,
                        ),
                    )
                )
            else:
                results.append((f"REMOVED composite: {label} (@{sym_name})", None))

    for sym_name in target_eps:
        if (
            sym_name != "main"
            and sym_name not in matched_tgt_callees
            and sym_name not in source_eps
        ):
            label = _composite_label(sym_name, target_eps)
            results.append((f"ADDED composite: {label} (@{sym_name})", None))

    return results


def compute_per_graph_diff(
    source_program: AIProgram,
    target_program: AIProgram,
    *,
    weights: WeightPolicy = WeightPolicy.IGNORE,
    ignore_attributes: Collection[str] = UNSTABLE_ATTRIBUTES,
) -> list[tuple[str, GraphDiff | None]]:
    """Compute per-graph diffs, matching composites via invoke call sites.

    Returns a list of (label, diff) tuples. The first entry is always "main".
    Composite graphs are matched by pairing coreai.invoke ops in the main diff.
    Unmatched composites produce entries with diff=None.

    Args:
        source_program: Source (reference/expected) AIProgram
        target_program: Target (actual/test) AIProgram
        weights: Whether parameter *values* count towards a node's identity; see
            `compute_coreai_program_diff`. Under `DIGEST` each graph reaches its own
            module's parameters, which costs a print of it per graph compared.
        ignore_attributes: Attribute names left out of an operation's identity; see
            `compute_coreai_program_diff`. Applied to every graph compared, main and
            composite alike, so one notion of identity governs the whole report.

    Returns:
        List of (label, GraphDiff | None) tuples for each graph in the programs

    """
    source_eps = _collect_entry_points(source_program._mlir_module)
    target_eps = _collect_entry_points(target_program._mlir_module)
    # Fallback when no main graph exists
    if "main" not in source_eps or "main" not in target_eps:
        source_graph = _build_module_graph(source_program._mlir_module)
        target_graph = _build_module_graph(target_program._mlir_module)
        return [
            (
                "all",
                compute_graph_diff(
                    source_graph,
                    target_graph,
                    weights=weights,
                    ignore_attributes=ignore_attributes,
                ),
            )
        ]

    # Diff main graphs
    source_main_graph = _build_module_graph(source_program._mlir_module, "main")
    target_main_graph = _build_module_graph(target_program._mlir_module, "main")
    main_diff = compute_graph_diff(
        source_main_graph,
        target_main_graph,
        weights=weights,
        ignore_attributes=ignore_attributes,
    )

    results: list[tuple[str, GraphDiff | None]] = [("main", main_diff)]

    # Match and diff composites via invoke call sites
    matched_composites, matched_src_callees, matched_tgt_callees = (
        _match_composites_from_main_diff(
            main_diff, source_main_graph, target_main_graph
        )
    )

    results.extend(
        _diff_matched_composites(
            matched_composites,
            source_eps,
            target_eps,
            source_program._mlir_module,
            target_program._mlir_module,
            weights,
            ignore_attributes,
        )
    )

    # Report removed/added composites from unmapped invoke ops
    results.extend(
        _report_unmatched_composites(
            main_diff,
            source_main_graph,
            target_main_graph,
            source_eps,
            target_eps,
            matched_src_callees,
            matched_tgt_callees,
        )
    )

    # Report orphan composites not referenced by invokes
    results.extend(
        _report_orphan_composites(
            source_eps,
            target_eps,
            matched_src_callees,
            matched_tgt_callees,
            source_program._mlir_module,
            target_program._mlir_module,
            weights,
            ignore_attributes,
        )
    )

    return results


def format_multi_graph_diff(
    results: list[tuple[str, GraphDiff | None]],
    *,
    indent_size: int = 2,
    max_items: int | None = None,
) -> str:
    """Format per-graph diff results as human-readable text.

    Args:
        results: List of (label, GraphDiff | None) tuples from compute_per_graph_diff
        indent_size: Number of spaces per indentation level (default: 2)
        max_items: Maximum number of items to show per section (default: None, shows all)

    Returns:
        Formatted multi-graph diff as a string

    """
    sections: list[str] = []

    all_isomorphic = True
    for label, diff in results:
        lines: list[tuple[int, str]] = []
        lines.append((0, "=" * 80))
        lines.append((0, f"GRAPH: {label}"))
        lines.append((0, "=" * 80))
        lines.append((0, ""))

        ops_table: _TableSpec | None = None
        if diff is None:
            lines.append((0, "(no counterpart in the other program)"))
            all_isomorphic = False
        else:
            if not diff.is_isomorphic:
                all_isomorphic = False
            lines.extend(_format_verdict(diff, diff.source_graph))
            lines.append((0, ""))
            lines.extend(_format_summary(diff))
            lines.append((0, ""))
            table_limit = None if max_items is None else max_items * 2
            ops_table = _format_unified_ops_table(
                diff, diff.source_graph, diff.target_graph, table_limit
            )

        formatted = _apply_indentation(lines, indent_size)
        section = "\n".join(formatted)
        if ops_table is not None:
            # Render the table into a buffer so it can be embedded in the
            # returned string alongside the prose sections.
            buffer = StringIO()
            _write_table(ops_table, buffer)
            section = f"{section}\n{buffer.getvalue()}"
        sections.append(section)

    # Overall verdict
    overall: list[tuple[int, str]] = []
    overall.append((0, "=" * 80))
    if all_isomorphic:
        overall.append((0, "\u2713 ALL GRAPHS ARE ISOMORPHIC."))
    else:
        overall.append((0, "\u2717 GRAPHS DIFFER. See per-graph details above."))
    overall.append((0, "=" * 80))
    sections.append("\n".join(_apply_indentation(overall, indent_size)))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# PyTorch FX graph support
# ---------------------------------------------------------------------------


def _fx_nodes_in(value: Any) -> Iterator[torch.fx.Node]:
    """
    Every FX node reachable inside an argument, in a stable order.

    An FX argument is not always a node: `aten.cat` takes its inputs as a *list*, and a
    dict or a nested tuple is equally legal. Only top-level `fx.Node` args used to be
    followed, so a concatenation, a stack or a `_native_batch_norm` tuple contributed no
    edges at all, and the nodes feeding them looked unused to the comparison.

    Args:
        value: An FX argument, of any shape.

    Yields:
        The nodes inside it, outer to inner, left to right.

    """
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _fx_nodes_in(item)
    elif isinstance(value, dict):
        for _, item in sorted(value.items(), key=lambda entry: str(entry[0])):
            yield from _fx_nodes_in(item)


def _fx_constant(value: Any) -> Any:
    """
    An FX argument with its node references replaced, for the attribute digest.

    Nodes are edges, not configuration, so they are collapsed to a placeholder: their
    identity is carried by the graph, and their *names* shift whenever anything upstream
    changes. A tensor is reduced to its shape and dtype for the reason
    `WeightPolicy.IGNORE` exists -- a rebuild re-initialises parameters, so comparing
    values reports every weight of an unchanged model as changed.

    Args:
        value: An FX argument, of any shape.

    Returns:
        The same argument as plain, comparable values.

    """
    if isinstance(value, torch.fx.Node):
        return "<node>"
    if isinstance(value, (list, tuple)):
        return [_fx_constant(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _fx_constant(item) for key, item in sorted(value.items())}
    if isinstance(value, torch.Tensor):
        return f"tensor<{tuple(value.shape)}x{value.dtype}>"
    return repr(value)


def _fx_attributes(fx_node: Any) -> str:
    """
    An FX node's configuration: the parts of its arguments that are not nodes.

    The counterpart of `_attr_digest` for a graph with no MLIR behind it. Without it
    `aten.mean(x, dim=-1)` and `aten.mean(x, dim=0)` share an identity, as do two
    `aten.to` calls to different dtypes.

    Args:
        fx_node: The FX node to describe.

    Returns:
        The node's constant arguments and keyword arguments, as a comparable string.

    """
    parts = [repr(_fx_constant(arg)) for arg in fx_node.args]
    parts.extend(
        f"{name}={_fx_constant(value)!r}"
        for name, value in sorted(fx_node.kwargs.items())
    )
    return "|".join(parts)


def _fx_result_type(fx_node: Any) -> str:
    """
    The shape and dtype an FX node produces, when export recorded them.

    Where a torch graph keeps what MLIR keeps in a result type. An FX graph is not
    bipartite -- a node *is* its result -- so this goes in the node's own `ir_type`,
    which `structural_labels` drops and `node_labels` keeps. That split is what lets a
    node whose shape changed still pair, and then read as modified.

    Args:
        fx_node: The FX node to describe.

    Returns:
        The type as text, or an empty string when export recorded none.

    """
    value = fx_node.meta.get("val")
    if isinstance(value, (list, tuple)):
        return ",".join(_fx_result_type_of(item) for item in value)
    return _fx_result_type_of(value)


def _fx_result_type_of(value: Any) -> str:
    """
    One tensor's shape and dtype.

    Args:
        value: A fake tensor from `node.meta["val"]`, or any other metadata value.

    Returns:
        The type as text, or an empty string when it is not a tensor.

    """
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return "" if value is None else type(value).__name__
    return f"tensor<{'x'.join(str(dimension) for dimension in shape)}x{dtype}>"


class _TorchFXGraphBuilder:
    """Helper class to build NetworkX graph from PyTorch FX graphs."""

    def __init__(self) -> None:
        """Initialize the graph builder."""
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.fx_node_to_id: dict[Any, int] = {}

    def build(self, fx_graph: Any) -> nx.DiGraph:
        """Build the graph from PyTorch FX graph."""
        # Process all nodes in the FX graph
        for node in fx_graph.nodes:
            self._process_fx_node(node)

        # Add edges based on node inputs
        for node in fx_graph.nodes:
            if node in self.fx_node_to_id:
                self._add_edges(node)

        return self.graph

    def _add_edges(self, fx_node: Any) -> None:
        """
        Add one edge per node this node consumes, positional args then keywords.

        The slot an edge occupies is `(edge_type, index)`, and operand order is
        semantic, so positional inputs are numbered across the *flattened* arguments:
        `cat([a, b])` gives `a` slot 0 and `b` slot 1, where before it gave neither an
        edge. Keyword inputs are keyed by name instead of position, since that is what
        identifies them -- reordering kwargs is not a change.

        Args:
            fx_node: The FX node whose inputs are being wired up.

        """
        node_id = self.fx_node_to_id[fx_node]

        for index, argument in enumerate(_fx_nodes_in(fx_node.args)):
            if argument in self.fx_node_to_id:
                self.graph.add_edge(
                    self.fx_node_to_id[argument],
                    node_id,
                    edge_type="data_flow",
                    index=index,
                )

        for name, value in sorted(fx_node.kwargs.items()):
            for index, argument in enumerate(_fx_nodes_in(value)):
                if argument in self.fx_node_to_id:
                    self.graph.add_edge(
                        self.fx_node_to_id[argument],
                        node_id,
                        edge_type=f"kwarg:{name}",
                        index=index,
                    )

    def _get_next_id(self) -> int:
        """Generate a unique node ID."""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def _process_fx_node(self, fx_node: Any) -> None:
        """Process a PyTorch FX node and add it to the graph."""
        node_id = self._get_next_id()
        self.fx_node_to_id[fx_node] = node_id

        # Determine node type and operation name
        op_type = str(fx_node.op)
        target = str(fx_node.target) if fx_node.target else "unknown"

        # Add node to graph. `attributes` and `ir_type` are precomputed here rather
        # than read from an `ir_object`, which an FX node does not have: see
        # `graph_match._attr_digest` for the fallback they feed.
        self.graph.add_node(
            node_id,
            type="op",
            op_name=f"{op_type}:{target}",
            op_type=op_type,
            target=target,
            attributes=_fx_attributes(fx_node),
            ir_type=_fx_result_type(fx_node),
            torch_object=fx_node,
        )


def _build_torch_fx_graph(exported_program: torch.export.ExportedProgram) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from a PyTorch ExportedProgram.

    Args:
        exported_program: PyTorch ExportedProgram to convert

    Returns:
        NetworkX directed graph representing the FX graph structure

    """
    builder = _TorchFXGraphBuilder()
    return builder.build(exported_program.graph_module.graph)


def compute_exported_program_diff(
    source_program: torch.export.ExportedProgram,
    target_program: torch.export.ExportedProgram,
    *,
    ignore_attributes: Collection[str] = UNSTABLE_ATTRIBUTES,
) -> GraphDiff:
    """
    Compute structural diff between two PyTorch ExportedPrograms.

    Extracts the FX graphs, builds NetworkX graphs, and compares them with the same
    `graph_match.align` the Core AI diff uses.

    A node's identity is its op and target, its constant arguments and keyword
    arguments, and the shape and dtype export recorded for it. Its edges are every FX
    node reachable inside its arguments, so an input passed in a list -- `aten.cat`,
    `aten.stack` -- is followed like any other.

    Two differences from :func:`compute_coreai_program_diff` are worth knowing:

    * **An FX graph is not bipartite.** A node is its own result, so there are no value
      nodes and `responsible_op` is the identity. Node counts are therefore much smaller
      than the Core AI ones for the same model, and not comparable with them.
    * **There is no `WeightPolicy`.** Parameters live in the module's state dict rather
      than in the graph, so the graph carries no values to compare; a parameter appears
      only as a `placeholder` named after it. A retrained model diffs as unchanged here.
      Use :func:`compute_coreai_program_diff` with `DIGEST` to compare weights.

    Args:
        source_program: Source (reference/expected) ExportedProgram
        target_program: Target (actual/test) ExportedProgram
        ignore_attributes: Attribute names left out of a node's identity. Accepted for
            symmetry with the Core AI entry points; an FX node carries no named MLIR
            attributes, so this reaches only the labelling of nodes that have them.

    Returns:
        GraphDiff object with source_graph and target_graph included

    """
    # Build graphs from FX graphs
    source_graph = _build_torch_fx_graph(source_program)
    target_graph = _build_torch_fx_graph(target_program)

    # Compute and return diff
    return compute_graph_diff(
        source_graph,
        target_graph,
        ignore_attributes=ignore_attributes,
    )
