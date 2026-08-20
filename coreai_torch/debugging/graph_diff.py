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
from dataclasses import dataclass, fields
from enum import Enum
from io import StringIO
from typing import Any, TextIO

import networkx as nx  # type: ignore[import-untyped]
import torch
from coreai._compiler.ir import Block, Operation, Region, Value
from coreai.authoring import AIProgram

from .graph_match import (
    Label,
    WeightPolicy,
    align,
    node_labels,
    responsible_op,
)
from .table_writer import _Column, _Row, _TableSpec, _write_table

# ---------------------------------------------------------------------------
# Regex patterns for composite diffing
# ---------------------------------------------------------------------------

_CALLEE_SYMBOL_RE = re.compile(r"<@([^>]+)>")
_UUID_SUFFIX_RE = re.compile(r"_[0-9a-f]{8,}$")


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
    endpoints were mapped -- which is what this replaced -- counted an edge as
    common whenever its nodes were, so a pure rewiring reported every edge as
    common and the "common subgraph" line read 100% for a graph that had been
    rewired.

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

    Returns:
        GraphDiff object describing the correspondence and what it leaves over

    """
    alignment = align(source_graph, target_graph, weights=weights)

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


def _strip_uuid_suffix(name: str) -> str:
    """Strip trailing UUID suffix for display: 'sdpa_abc123ef' -> 'sdpa'."""
    return _UUID_SUFFIX_RE.sub("", name)


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
    # a label mismatch or an operand mismatch; the labels here are the ones with
    # parameter values *left out*, and the operands correspond. So the values are
    # what differ -- which is only reachable when the diff was computed under
    # `WeightPolicy.DIGEST`, since that is the only policy that looks at them.
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
    another arrived, and reporting it as a modification hides what used to be there.

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
    labels = (node_labels(source_graph), node_labels(target_graph))

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
                _describe_modification(
                    src_node,
                    tgt_node,
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

    Returns:
        GraphDiff object with source_graph and target_graph included

    Raises:
        ValueError: If entry point is not found in either program

    """
    source_graph = _build_module_graph(source_program._mlir_module, entry_point)
    target_graph = _build_module_graph(target_program._mlir_module, entry_point)
    return compute_graph_diff(source_graph, target_graph, weights=weights)


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
) -> list[tuple[str, GraphDiff | None]]:
    """Diff each matched composite pair and return labeled results."""
    results: list[tuple[str, GraphDiff | None]] = []
    for src_callee, tgt_callee in matched_composites:
        if src_callee not in source_eps or tgt_callee not in target_eps:
            continue
        src_graph = _build_module_graph(source_module, src_callee)
        tgt_graph = _build_module_graph(target_module, tgt_callee)
        diff = compute_graph_diff(src_graph, tgt_graph, weights=weights)
        label = _strip_uuid_suffix(src_callee)
        results.append(
            (f"{label} (source: @{src_callee}, target: @{tgt_callee})", diff)
        )
    return results


def _report_unmatched_composites(
    main_diff: GraphDiff,
    source_main_graph: nx.DiGraph,
    target_main_graph: nx.DiGraph,
    matched_src_callees: set[str],
    matched_tgt_callees: set[str],
) -> list[tuple[str, GraphDiff | None]]:
    """Report composites only present in source (removed) or target (added)."""
    results: list[tuple[str, GraphDiff | None]] = []

    for src_id in main_diff.unmapped_source_nodes:
        src_callee = _extract_invoke_callee(source_main_graph, src_id)
        if src_callee and src_callee not in matched_src_callees:
            label = _strip_uuid_suffix(src_callee)
            results.append((f"REMOVED composite: {label} (@{src_callee})", None))
            matched_src_callees.add(src_callee)

    for tgt_id in main_diff.unmapped_target_nodes:
        tgt_callee = _extract_invoke_callee(target_main_graph, tgt_id)
        if tgt_callee and tgt_callee not in matched_tgt_callees:
            label = _strip_uuid_suffix(tgt_callee)
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
) -> list[tuple[str, GraphDiff | None]]:
    """Report composites not referenced by any invoke in main."""
    results: list[tuple[str, GraphDiff | None]] = []

    for sym_name in source_eps:
        if sym_name != "main" and sym_name not in matched_src_callees:
            label = _strip_uuid_suffix(sym_name)
            if sym_name in target_eps:
                src_graph = _build_module_graph(source_module, sym_name)
                tgt_graph = _build_module_graph(target_module, sym_name)
                results.append(
                    (
                        f"{label} (unreferenced, @{sym_name})",
                        compute_graph_diff(src_graph, tgt_graph, weights=weights),
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
            label = _strip_uuid_suffix(sym_name)
            results.append((f"ADDED composite: {label} (@{sym_name})", None))

    return results


def compute_per_graph_diff(
    source_program: AIProgram,
    target_program: AIProgram,
    *,
    weights: WeightPolicy = WeightPolicy.IGNORE,
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
            ("all", compute_graph_diff(source_graph, target_graph, weights=weights))
        ]

    # Diff main graphs
    source_main_graph = _build_module_graph(source_program._mlir_module, "main")
    target_main_graph = _build_module_graph(target_program._mlir_module, "main")
    main_diff = compute_graph_diff(
        source_main_graph, target_main_graph, weights=weights
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
        )
    )

    # Report removed/added composites from unmapped invoke ops
    results.extend(
        _report_unmatched_composites(
            main_diff,
            source_main_graph,
            target_main_graph,
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
                node_id = self.fx_node_to_id[node]
                for i, arg in enumerate(node.args):
                    # Check if arg is an FX node (has 'op' attribute and is in our mapping)
                    if hasattr(arg, "op") and arg in self.fx_node_to_id:
                        arg_id = self.fx_node_to_id[arg]
                        self.graph.add_edge(
                            arg_id,
                            node_id,
                            edge_type="data_flow",
                            index=i,
                        )

        return self.graph

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

        # Add node to graph
        self.graph.add_node(
            node_id,
            type="op",
            op_name=f"{op_type}:{target}",
            op_type=op_type,
            target=target,
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
) -> GraphDiff:
    """
    Compute structural diff between two PyTorch ExportedPrograms.

    Extracts the FX graphs, builds NetworkX graphs, and computes structural diff
    using graph isomorphism.

    Args:
        source_program: Source (reference/expected) ExportedProgram
        target_program: Target (actual/test) ExportedProgram

    Returns:
        GraphDiff object with source_graph and target_graph included

    """
    # Build graphs from FX graphs
    source_graph = _build_torch_fx_graph(source_program)
    target_graph = _build_torch_fx_graph(target_program)

    # Compute and return diff
    return compute_graph_diff(source_graph, target_graph)
