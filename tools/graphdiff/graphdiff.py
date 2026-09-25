#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""graphdiff — Structural graph diff between two Core AI programs.

Usage:
    python tools/graphdiff/graphdiff.py <source> <target>
    python tools/graphdiff/graphdiff.py --entry-point main <source> <target>
    python tools/graphdiff/graphdiff.py --max-items 50 <source> <target>

Loads AIModel asset (.aimodel) directories via coreai's AIProgram,
which automatically converts the serialized form to the coreai dialect.

This file is self-contained: all graph-diff logic (previously in the
upstream `_graph_diff` implementation) is inlined here so that the tool
does not depend on the upstream package for anything beyond loading
programs and IR types.
"""

from __future__ import annotations

import argparse
import html as html_mod
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx  # type: ignore[import-untyped]
from coreai.authoring import AIModelAsset, AIProgram
from networkx.algorithms import isomorphism  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import torch
    from coreai._compiler.ir import Block, Operation, Region, Value

# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------


class _OpDiffType(Enum):
    """Type of operation difference in structural diff."""

    ALIGNED = "aligned"  # Structurally identical
    MODIFIED = "modified"  # Same name, different structure
    REMOVED = "removed"  # Only in source
    ADDED = "added"  # Only in target
    POSITION_ONLY = "position_only"  # Same structure, different position (not shown)


@dataclass
class _GraphDiffSummary:
    """Summary statistics for graph comparison."""

    source_node_count: int
    target_node_count: int
    source_edge_count: int
    target_edge_count: int
    mapped_node_count: int = 0
    unmapped_source_node_count: int = 0
    unmapped_target_node_count: int = 0
    unmapped_source_edge_count: int = 0
    unmapped_target_edge_count: int = 0


@dataclass
class _GraphDiff:
    """Result of structural graph comparison using isomorphism."""

    is_isomorphic: bool
    source_to_target_mapping: dict[int, int]
    target_to_source_mapping: dict[int, int]
    unmapped_source_nodes: list[int]
    unmapped_target_nodes: list[int]
    unmapped_source_edges: list[tuple[int, int]]
    unmapped_target_edges: list[tuple[int, int]]
    summary: _GraphDiffSummary
    source_graph: nx.DiGraph
    target_graph: nx.DiGraph


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class _CoreAIGraphBuilder:
    """Helper class to build NetworkX graph from Core AI operations."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.value_to_node: dict[Value, int] = {}

    def build(self, root_op: Operation) -> nx.DiGraph:
        self._process_operation(root_op)
        return self.graph

    def _get_next_id(self) -> int:
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def _add_value_node(self, value: Value, value_type: str) -> int:
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
        for arg_idx, arg in enumerate(block.arguments):  # type: ignore[var-annotated, arg-type]
            arg_node_id = self._add_value_node(arg, "block_arg")
            self.graph.add_edge(
                block_node_id, arg_node_id, edge_type="block_arg", index=arg_idx
            )
        for op in block.operations:
            self._process_operation(op)  # type: ignore[arg-type]

    def _process_region(
        self, region: Region, parent_op_id: int, region_idx: int
    ) -> None:
        region_node_id = self._get_next_id()
        self.graph.add_node(
            region_node_id, type="region", index=region_idx, ir_object=region
        )
        self.graph.add_edge(
            parent_op_id,
            region_node_id,
            edge_type="contains_region",
            index=region_idx,
        )
        for block_idx, block in enumerate(region.blocks):
            self._process_block_in_region(region_node_id, block, block_idx)

    def _process_block_in_region(
        self, region_node_id: int, block: Block, block_idx: int
    ) -> None:
        block_node_id = self._get_next_id()
        self.graph.add_node(
            block_node_id, type="block", index=block_idx, ir_object=block
        )
        self.graph.add_edge(
            region_node_id,
            block_node_id,
            edge_type="contains_block",
            index=block_idx,
        )
        self._process_block(block, block_node_id)

    def _process_operation(self, operation: Operation) -> int:
        op_node_id = self._get_next_id()
        self.graph.add_node(
            op_node_id, type="op", op_name=operation.name, ir_object=operation
        )
        self._add_operation_results(operation, op_node_id)
        self._add_operation_operands(operation, op_node_id)
        for region_idx, region in enumerate(operation.regions):
            self._process_region(region, op_node_id, region_idx)
        return op_node_id

    def _add_operation_results(self, operation: Operation, op_node_id: int) -> None:
        for result_idx, result in enumerate(operation.results):  # type: ignore[var-annotated, arg-type]
            result_node_id = self._add_value_node(result, "op_result")
            self.graph.add_edge(
                op_node_id, result_node_id, edge_type="defines", index=result_idx
            )

    def _add_operation_operands(self, operation: Operation, op_node_id: int) -> None:
        for operand_idx, operand in enumerate(operation.operands):  # type: ignore[var-annotated, arg-type]
            operand_node_id = self._add_value_node(operand, "operand")
            self.graph.add_edge(
                operand_node_id, op_node_id, edge_type="operand", index=operand_idx
            )


# ---------------------------------------------------------------------------
# Torch FX graph builder
# ---------------------------------------------------------------------------


class _TorchFXGraphBuilder:
    """Helper class to build NetworkX graph from PyTorch FX graphs."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.fx_node_to_id: dict[Any, int] = {}

    def build(self, fx_graph: Any) -> nx.DiGraph:
        for node in fx_graph.nodes:
            self._process_fx_node(node)
        for node in fx_graph.nodes:
            if node in self.fx_node_to_id:
                node_id = self.fx_node_to_id[node]
                for i, arg in enumerate(node.args):
                    if hasattr(arg, "op") and arg in self.fx_node_to_id:
                        arg_id = self.fx_node_to_id[arg]
                        self.graph.add_edge(
                            arg_id, node_id, edge_type="data_flow", index=i
                        )
        return self.graph

    def _get_next_id(self) -> int:
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def _process_fx_node(self, fx_node: Any) -> None:
        node_id = self._get_next_id()
        self.fx_node_to_id[fx_node] = node_id
        op_type = str(fx_node.op)
        target = str(fx_node.target) if fx_node.target else "unknown"
        self.graph.add_node(
            node_id,
            type="op",
            op_name=f"{op_type}:{target}",
            op_type=op_type,
            target=target,
            torch_object=fx_node,
        )


# ---------------------------------------------------------------------------
# Location metadata extraction (regex-based)
# ---------------------------------------------------------------------------

_TORCH_IDENTIFIER_RE = re.compile(r'"torch\.identifiers\.\d+"\("([^"]+)":\d+:\d+\)')
_FILE_LOC_RE = re.compile(r'"([^"]+\.py)":(\d+):\d+')
_CALLEE_SYMBOL_RE = re.compile(r"<@([^>]+)>")
_UUID_SUFFIX_RE = re.compile(r"_[0-9a-f]{8,}$")


def _get_loc_str(graph: nx.DiGraph, node_id: int) -> str:
    ir_obj = graph.nodes[node_id].get("ir_object")
    if ir_obj is None:
        return ""
    try:
        return str(ir_obj.location)
    except Exception:
        return ""


def _extract_fx_node(graph: nx.DiGraph, node_id: int) -> str:
    """Extract the torch fx.Node name from a Core AI op's location metadata."""
    loc_str = _get_loc_str(graph, node_id)
    match = _TORCH_IDENTIFIER_RE.search(loc_str)
    return match.group(1) if match else ""


def _extract_file_location(graph: nx.DiGraph, node_id: int) -> str:
    """Extract the first source file:line from a Core AI op's location metadata."""
    loc_str = _get_loc_str(graph, node_id)
    match = _FILE_LOC_RE.search(loc_str)
    if not match:
        return ""
    filepath, line = match.group(1), match.group(2)
    basename = filepath.rsplit("/", 1)[-1]
    return f"{basename}:{line}"


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


# ---------------------------------------------------------------------------
# Graph matching
# ---------------------------------------------------------------------------


def _greedy_topological_match(
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    node_match: Any,
    edge_match: Any,
) -> dict[int, int]:
    """Find partial node mapping using greedy topological matching."""

    def get_op_nodes_by_name(graph: nx.DiGraph) -> dict[str, list[int]]:
        ops_by_name: dict[str, list[int]] = {}
        try:
            ordered_nodes = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            ordered_nodes = sorted(graph.nodes())
        for n in ordered_nodes:
            attrs = graph.nodes[n]
            if attrs.get("type") == "op":
                op_name = attrs.get("op_name", "unknown")
                ops_by_name.setdefault(op_name, []).append(n)
        return ops_by_name

    source_ops = get_op_nodes_by_name(source_graph)
    target_ops = get_op_nodes_by_name(target_graph)

    mapping: dict[int, int] = {}
    for op_name in source_ops:
        if op_name not in target_ops:
            continue
        src_ids = source_ops[op_name]
        tgt_ids = target_ops[op_name]
        for src_id, tgt_id in zip(src_ids, tgt_ids, strict=False):
            mapping[src_id] = tgt_id

    _extend_mapping_to_connected_nodes(mapping, source_graph, target_graph)
    return mapping


def _extend_mapping_to_connected_nodes(
    mapping: dict[int, int],
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
) -> None:
    """Extend an op-level mapping to include connected non-op nodes."""
    mapped_target: set[int] = set(mapping.values())
    for src_op, tgt_op in list(mapping.items()):
        _match_neighbors(
            src_op, tgt_op, source_graph, target_graph, mapping, mapped_target, "out"
        )
        _match_neighbors(
            src_op, tgt_op, source_graph, target_graph, mapping, mapped_target, "in"
        )


def _match_neighbors(
    src_node: int,
    tgt_node: int,
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    mapping: dict[int, int],
    mapped_target: set[int],
    direction: str,
) -> None:
    """Match neighboring non-op nodes by edge type and index."""
    if direction == "out":
        src_edges = list(source_graph.out_edges(src_node, data=True))
        tgt_edges = list(target_graph.out_edges(tgt_node, data=True))
    else:
        src_edges = list(source_graph.in_edges(src_node, data=True))
        tgt_edges = list(target_graph.in_edges(tgt_node, data=True))

    def group_edges(
        edges: list[tuple[int, int, dict[str, Any]]], direction: str
    ) -> dict[tuple[str, int], int]:
        groups: dict[tuple[str, int], int] = {}
        for u, v, data in edges:
            key = (data.get("edge_type", ""), data.get("index", 0))
            neighbor = v if direction == "out" else u
            groups[key] = neighbor
        return groups

    src_groups = group_edges(src_edges, direction)
    tgt_groups = group_edges(tgt_edges, direction)

    for key, src_neighbor in src_groups.items():
        if src_neighbor in mapping:
            continue
        tgt_neighbor = tgt_groups.get(key)
        if tgt_neighbor is None or tgt_neighbor in mapped_target:
            continue
        src_type = source_graph.nodes[src_neighbor].get("type")
        tgt_type = target_graph.nodes[tgt_neighbor].get("type")
        if src_type == tgt_type and src_type != "op":
            mapping[src_neighbor] = tgt_neighbor
            mapped_target.add(tgt_neighbor)


# ---------------------------------------------------------------------------
# Core diff computation
# ---------------------------------------------------------------------------


def _compute_graph_diff(
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
) -> _GraphDiff:
    """Compute structural differences using graph isomorphism."""

    def node_match(n1: dict[str, Any], n2: dict[str, Any]) -> bool:
        return n1.get("type") == n2.get("type") and n1.get("op_name") == n2.get(
            "op_name"
        )

    def edge_match(e1: dict[str, Any], e2: dict[str, Any]) -> bool:
        return e1.get("edge_type") == e2.get("edge_type")

    matcher = isomorphism.DiGraphMatcher(
        source_graph, target_graph, node_match=node_match, edge_match=edge_match
    )

    is_iso = matcher.is_isomorphic()

    if is_iso:
        source_to_target = matcher.mapping
        return _GraphDiff(
            is_isomorphic=True,
            source_to_target_mapping=source_to_target,
            target_to_source_mapping={v: k for k, v in source_to_target.items()},
            unmapped_source_nodes=[],
            unmapped_target_nodes=[],
            unmapped_source_edges=[],
            unmapped_target_edges=[],
            summary=_GraphDiffSummary(
                source_node_count=source_graph.number_of_nodes(),
                target_node_count=target_graph.number_of_nodes(),
                source_edge_count=source_graph.number_of_edges(),
                target_edge_count=target_graph.number_of_edges(),
                mapped_node_count=len(source_to_target),
            ),
            source_graph=source_graph,
            target_graph=target_graph,
        )

    # Greedy topological fallback for non-isomorphic graphs
    best_mapping = _greedy_topological_match(
        source_graph, target_graph, node_match, edge_match
    )

    mapped_source = set(best_mapping.keys())
    mapped_target = set(best_mapping.values())

    unmapped_source_nodes = [n for n in source_graph.nodes() if n not in mapped_source]
    unmapped_target_nodes = [n for n in target_graph.nodes() if n not in mapped_target]

    def edge_is_mapped(edge: tuple[int, int], mapping: dict[int, int]) -> bool:
        return edge[0] in mapping and edge[1] in mapping

    unmapped_source_edges = [
        e for e in source_graph.edges() if not edge_is_mapped(e, best_mapping)
    ]
    unmapped_target_edges = [
        e
        for e in target_graph.edges()
        if not edge_is_mapped(e, {v: k for k, v in best_mapping.items()})
    ]

    return _GraphDiff(
        is_isomorphic=False,
        source_to_target_mapping=best_mapping,
        target_to_source_mapping={v: k for k, v in best_mapping.items()},
        unmapped_source_nodes=unmapped_source_nodes,
        unmapped_target_nodes=unmapped_target_nodes,
        unmapped_source_edges=unmapped_source_edges,
        unmapped_target_edges=unmapped_target_edges,
        summary=_GraphDiffSummary(
            source_node_count=source_graph.number_of_nodes(),
            target_node_count=target_graph.number_of_nodes(),
            source_edge_count=source_graph.number_of_edges(),
            target_edge_count=target_graph.number_of_edges(),
            mapped_node_count=len(best_mapping),
            unmapped_source_node_count=len(unmapped_source_nodes),
            unmapped_target_node_count=len(unmapped_target_nodes),
            unmapped_source_edge_count=len(unmapped_source_edges),
            unmapped_target_edge_count=len(unmapped_target_edges),
        ),
        source_graph=source_graph,
        target_graph=target_graph,
    )


# ---------------------------------------------------------------------------
# Diff detail helpers
# ---------------------------------------------------------------------------


def _count_edges_by_type(
    graph: nx.DiGraph, node_id: int, edge_type: str, direction: str = "out"
) -> int:
    edges = graph.out_edges(node_id) if direction == "out" else graph.in_edges(node_id)
    return sum(1 for e in edges if graph[e[0]][e[1]].get("edge_type") == edge_type)


def _compute_coreai_op_diff_details(
    src_id: int, tgt_id: int, source_graph: nx.DiGraph, target_graph: nx.DiGraph
) -> tuple[_OpDiffType, str]:
    src_operands = _count_edges_by_type(source_graph, src_id, "operand", "in")
    tgt_operands = _count_edges_by_type(target_graph, tgt_id, "operand", "in")
    src_results = _count_edges_by_type(source_graph, src_id, "defines", "out")
    tgt_results = _count_edges_by_type(target_graph, tgt_id, "defines", "out")
    src_regions = _count_edges_by_type(source_graph, src_id, "contains_region", "out")
    tgt_regions = _count_edges_by_type(target_graph, tgt_id, "contains_region", "out")

    parts = []
    if src_operands != tgt_operands:
        parts.append(f"operands:{src_operands}\u2192{tgt_operands}")
    if src_results != tgt_results:
        parts.append(f"results:{src_results}\u2192{tgt_results}")
    if src_regions != tgt_regions:
        parts.append(f"regions:{src_regions}\u2192{tgt_regions}")

    if parts:
        return (_OpDiffType.MODIFIED, ", ".join(parts))
    return (_OpDiffType.POSITION_ONLY, "")


def _compute_torch_op_diff_details(
    src_id: int, tgt_id: int, source_graph: nx.DiGraph, target_graph: nx.DiGraph
) -> tuple[_OpDiffType, str]:
    src_inputs = _count_edges_by_type(source_graph, src_id, "data_flow", "in")
    tgt_inputs = _count_edges_by_type(target_graph, tgt_id, "data_flow", "in")
    src_outputs = _count_edges_by_type(source_graph, src_id, "data_flow", "out")
    tgt_outputs = _count_edges_by_type(target_graph, tgt_id, "data_flow", "out")

    parts = []
    if src_inputs != tgt_inputs:
        parts.append(f"inputs:{src_inputs}\u2192{tgt_inputs}")
    if src_outputs != tgt_outputs:
        parts.append(f"outputs:{src_outputs}\u2192{tgt_outputs}")

    if parts:
        return (_OpDiffType.MODIFIED, ", ".join(parts))
    return (_OpDiffType.POSITION_ONLY, "")


def _compute_op_diff_details(
    src_id: int, tgt_id: int, source_graph: nx.DiGraph, target_graph: nx.DiGraph
) -> tuple[_OpDiffType, str]:
    """Dispatch to Core AI or torch-specific diff computation."""
    is_torch_graph = "torch_object" in source_graph.nodes[src_id]
    if is_torch_graph:
        return _compute_torch_op_diff_details(
            src_id, tgt_id, source_graph, target_graph
        )
    return _compute_coreai_op_diff_details(src_id, tgt_id, source_graph, target_graph)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _group_ops_by_name(nodes: list[int], graph: nx.DiGraph) -> dict[str, list[int]]:
    ops_by_name: dict[str, list[int]] = {}
    for n in nodes:
        if graph.nodes[n].get("type") == "op":
            op_name = graph.nodes[n].get("op_name", "unknown")
            ops_by_name.setdefault(op_name, []).append(n)
    return ops_by_name


def _add_aligned_ops(
    diff: _GraphDiff,
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    rows: list[tuple[str, str, str, str, str, str]],
) -> None:
    for src_id, tgt_id in sorted(diff.source_to_target_mapping.items()):
        if source_graph.nodes[src_id].get("type") == "op":
            src_op = source_graph.nodes[src_id].get("op_name", "unknown")
            tgt_op = target_graph.nodes[tgt_id].get("op_name", "unknown")
            rows.append(
                (
                    str(src_id),
                    str(tgt_id),
                    _OpDiffType.ALIGNED.value,
                    src_op,
                    tgt_op,
                    "",
                )
            )


def _add_matched_by_name_ops(
    source_ids: list[int],
    target_ids: list[int],
    op_name: str,
    graphs: tuple[nx.DiGraph, nx.DiGraph],
    output: tuple[list[tuple[str, str, str, str, str, str]], set[int]],
) -> None:
    source_graph, target_graph = graphs
    rows, matched = output

    for src_id, tgt_id in zip(source_ids, target_ids, strict=False):
        diff_type, details = _compute_op_diff_details(
            src_id, tgt_id, source_graph, target_graph
        )
        if diff_type != _OpDiffType.POSITION_ONLY:
            fx = _extract_fx_node(source_graph, src_id)
            loc = _extract_file_location(source_graph, src_id)
            rows.append((str(src_id), str(tgt_id), diff_type.value, op_name, fx, loc))
        matched.update([src_id, tgt_id])

    count_diff = len(source_ids) - len(target_ids)
    if count_diff > 0:
        rows.extend(
            (
                str(src_id),
                "-",
                _OpDiffType.REMOVED.value,
                op_name,
                _extract_fx_node(source_graph, src_id),
                _extract_file_location(source_graph, src_id),
            )
            for src_id in source_ids[len(target_ids) :]
        )
        matched.update(source_ids[len(target_ids) :])
    elif count_diff < 0:
        rows.extend(
            (
                "-",
                str(tgt_id),
                _OpDiffType.ADDED.value,
                op_name,
                _extract_fx_node(target_graph, tgt_id),
                _extract_file_location(target_graph, tgt_id),
            )
            for tgt_id in target_ids[len(source_ids) :]
        )
        matched.update(target_ids[len(source_ids) :])


def _format_verdict(
    diff: _GraphDiff, source_graph: nx.DiGraph
) -> list[tuple[int, str]]:
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
            )
        )
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
                )
            )
    return lines


def _format_summary(diff: _GraphDiff) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    summary = diff.summary
    lines.append((0, "Summary:"))
    lines.append(
        (
            1,
            f"Source graph: {summary.source_node_count} nodes, {summary.source_edge_count} edges",
        )
    )
    lines.append(
        (
            1,
            f"Target graph: {summary.target_node_count} nodes, {summary.target_edge_count} edges",
        )
    )
    if not diff.is_isomorphic:
        lines.append((1, f"Mapped nodes: {summary.mapped_node_count}"))
        lines.append(
            (1, f"Unmapped in source: {summary.unmapped_source_node_count} nodes")
        )
        lines.append(
            (1, f"Unmapped in target: {summary.unmapped_target_node_count} nodes")
        )
    return lines


def _format_unified_ops_table(
    diff: _GraphDiff,
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    max_items: int | None = None,
) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    lines.append((0, "Operations Diff Table:"))

    rows: list[tuple[str, str, str, str, str, str]] = []

    source_ops_by_name = _group_ops_by_name(diff.unmapped_source_nodes, source_graph)
    target_ops_by_name = _group_ops_by_name(diff.unmapped_target_nodes, target_graph)

    matched_by_name: set[int] = set()
    common_names = set(source_ops_by_name.keys()) & set(target_ops_by_name.keys())

    for op_name in sorted(common_names):
        _add_matched_by_name_ops(
            source_ops_by_name[op_name],
            target_ops_by_name[op_name],
            op_name,
            (source_graph, target_graph),
            (rows, matched_by_name),
        )

    for op_name, src_ids in source_ops_by_name.items():
        if op_name not in common_names:
            rows.extend(
                (
                    str(src_id),
                    "-",
                    _OpDiffType.REMOVED.value,
                    op_name,
                    _extract_fx_node(source_graph, src_id),
                    _extract_file_location(source_graph, src_id),
                )
                for src_id in src_ids
            )

    for op_name, tgt_ids in target_ops_by_name.items():
        if op_name not in common_names:
            rows.extend(
                (
                    "-",
                    str(tgt_id),
                    _OpDiffType.ADDED.value,
                    op_name,
                    _extract_fx_node(target_graph, tgt_id),
                    _extract_file_location(target_graph, tgt_id),
                )
                for tgt_id in tgt_ids
            )

    if not rows:
        return lines

    headers = ("src_id", "tgt_id", "status", "op", "fx_node", "location")
    items_to_show = rows if max_items is None else rows[:max_items]
    col_widths = list(len(h) for h in headers)
    for row in items_to_show:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("\u2500" * w for w in col_widths)

    RED_BG = "\033[48;2;50;10;10m\033[97m"
    GREEN_BG = "\033[48;2;10;40;10m\033[97m"
    RESET = "\033[0m"

    lines.append((1, fmt.format(*headers)))
    lines.append((1, sep))

    for row in items_to_show:
        line = fmt.format(*row)
        status = row[2]
        if status == _OpDiffType.REMOVED.value:
            line = f"{RED_BG}{line}{RESET}"
        elif status == _OpDiffType.ADDED.value:
            line = f"{GREEN_BG}{line}{RESET}"
        lines.append((1, line))

    if max_items is not None and len(rows) > max_items:
        lines.append((1, f"... and {len(rows) - max_items} more operations"))

    return lines


def _apply_indentation(lines: list[tuple[int, str]], indent_size: int = 2) -> list[str]:
    return [" " * (level * indent_size) + text for level, text in lines]


def _format_diff_as_text(
    diff: _GraphDiff,
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    *,
    indent_size: int = 2,
    max_items: int | None = None,
) -> str:
    """Format isomorphism-based structural diff as human-readable text."""
    lines: list[tuple[int, str]] = []

    lines.append((0, "=" * 80))
    lines.append((0, "STRUCTURAL GRAPH DIFF (Isomorphism-Based)"))
    lines.append((0, "=" * 80))
    lines.append((0, ""))

    lines.extend(_format_verdict(diff, source_graph))
    lines.append((0, ""))

    lines.extend(_format_summary(diff))
    lines.append((0, ""))

    table_limit = None if max_items is None else max_items * 2
    ops_table = _format_unified_ops_table(diff, source_graph, target_graph, table_limit)
    if ops_table:
        lines.extend(ops_table)
        lines.append((0, ""))

    lines.append((0, "=" * 80))

    formatted_lines = _apply_indentation(lines, indent_size)
    return "\n".join(formatted_lines)


# ---------------------------------------------------------------------------
# High-level graph builders
# ---------------------------------------------------------------------------


def _build_graph(root_op: Operation) -> nx.DiGraph:
    """Build a NetworkX directed graph from a Core AI operation."""
    builder = _CoreAIGraphBuilder()
    return builder.build(root_op)


def _build_module_graph(module: Any, entry_point: str | None = None) -> nx.DiGraph:
    """Build a unified NetworkX graph from coreai.graph ops in a Core AI module."""
    builder = _CoreAIGraphBuilder()
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


def _build_torch_fx_graph(exported_program: torch.export.ExportedProgram) -> nx.DiGraph:
    """Build a NetworkX directed graph from a PyTorch ExportedProgram."""
    builder = _TorchFXGraphBuilder()
    return builder.build(exported_program.graph_module.graph)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_coreai_program_diff(
    source_program: AIProgram,
    target_program: AIProgram,
    *,
    entry_point: str | None = None,
) -> _GraphDiff:
    """Compute structural diff between two AIPrograms.

    By default (entry_point=None), compares all coreai.graph ops in the module,
    including sub-graphs for composites (layer_norm, sdpa, etc.). Set entry_point
    to compare a single graph only.
    """
    source_graph = _build_module_graph(source_program.module, entry_point)
    target_graph = _build_module_graph(target_program.module, entry_point)
    return _compute_graph_diff(source_graph, target_graph)


def compute_per_graph_diff(
    source_program: AIProgram,
    target_program: AIProgram,
) -> list[tuple[str, _GraphDiff | None]]:
    """Compute per-graph diffs, matching composites via invoke call sites.

    Returns a list of (label, diff) tuples. The first entry is always "main".
    Composite graphs are matched by pairing coreai.invoke ops in the main diff.
    Unmatched composites produce entries with diff=None.
    """
    source_eps = _collect_entry_points(source_program.module)
    target_eps = _collect_entry_points(target_program.module)

    results: list[tuple[str, _GraphDiff | None]] = []

    # 1. Diff main vs main
    if "main" not in source_eps or "main" not in target_eps:
        # Fallback: no main graph, diff everything flat
        source_graph = _build_module_graph(source_program.module)
        target_graph = _build_module_graph(target_program.module)
        results.append(("all", _compute_graph_diff(source_graph, target_graph)))
        return results

    source_main_graph = _build_module_graph(source_program.module, "main")
    target_main_graph = _build_module_graph(target_program.module, "main")
    main_diff = _compute_graph_diff(source_main_graph, target_main_graph)
    results.append(("main", main_diff))

    # 2. Match composite graphs via paired coreai.invoke ops in main
    matched_composites: list[tuple[str, str]] = []  # (src_callee, tgt_callee)
    matched_src_callees: set[str] = set()
    matched_tgt_callees: set[str] = set()

    # From aligned invoke pairs in the mapping
    for src_id, tgt_id in main_diff.source_to_target_mapping.items():
        src_callee = _extract_invoke_callee(source_main_graph, src_id)
        tgt_callee = _extract_invoke_callee(target_main_graph, tgt_id)
        if src_callee and tgt_callee:
            matched_composites.append((src_callee, tgt_callee))
            matched_src_callees.add(src_callee)
            matched_tgt_callees.add(tgt_callee)

    # 3. Diff each matched composite pair
    for src_callee, tgt_callee in matched_composites:
        if src_callee not in source_eps or tgt_callee not in target_eps:
            continue
        src_graph = _build_module_graph(source_program.module, src_callee)
        tgt_graph = _build_module_graph(target_program.module, tgt_callee)
        diff = _compute_graph_diff(src_graph, tgt_graph)
        label = _strip_uuid_suffix(src_callee)
        results.append(
            (f"{label} (source: @{src_callee}, target: @{tgt_callee})", diff)
        )

    # 4. Report unmatched composites (removed / added)
    # Invokes only in source (removed composites)
    for src_id in main_diff.unmapped_source_nodes:
        src_callee = _extract_invoke_callee(source_main_graph, src_id)
        if src_callee and src_callee not in matched_src_callees:
            label = _strip_uuid_suffix(src_callee)
            results.append((f"REMOVED composite: {label} (@{src_callee})", None))
            matched_src_callees.add(src_callee)

    # Invokes only in target (added composites)
    for tgt_id in main_diff.unmapped_target_nodes:
        tgt_callee = _extract_invoke_callee(target_main_graph, tgt_id)
        if tgt_callee and tgt_callee not in matched_tgt_callees:
            label = _strip_uuid_suffix(tgt_callee)
            results.append((f"ADDED composite: {label} (@{tgt_callee})", None))
            matched_tgt_callees.add(tgt_callee)

    # 5. Report composites not referenced by any invoke in main
    # (orphan graphs that exist in the module but aren't called)
    for sym_name in source_eps:
        if sym_name != "main" and sym_name not in matched_src_callees:
            label = _strip_uuid_suffix(sym_name)
            if sym_name in target_eps:
                # Both have it but it's not invoked from main — diff anyway
                src_graph = _build_module_graph(source_program.module, sym_name)
                tgt_graph = _build_module_graph(target_program.module, sym_name)
                results.append(
                    (
                        f"{label} (unreferenced, @{sym_name})",
                        _compute_graph_diff(src_graph, tgt_graph),
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


def _format_multi_graph_diff(
    results: list[tuple[str, _GraphDiff | None]],
    *,
    indent_size: int = 2,
    max_items: int | None = None,
) -> str:
    """Format per-graph diff results as human-readable text."""
    sections: list[str] = []

    all_isomorphic = True
    for label, diff in results:
        lines: list[tuple[int, str]] = []
        lines.append((0, "=" * 80))
        lines.append((0, f"GRAPH: {label}"))
        lines.append((0, "=" * 80))
        lines.append((0, ""))

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
            if ops_table:
                lines.extend(ops_table)
                lines.append((0, ""))

        formatted = _apply_indentation(lines, indent_size)
        sections.append("\n".join(formatted))

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


def compute_exported_program_diff(
    source_program: torch.export.ExportedProgram,
    target_program: torch.export.ExportedProgram,
) -> _GraphDiff:
    """Compute structural diff between two PyTorch ExportedPrograms."""
    source_graph = _build_torch_fx_graph(source_program)
    target_graph = _build_torch_fx_graph(target_program)
    return _compute_graph_diff(source_graph, target_graph)


# ---------------------------------------------------------------------------
# ANSI to HTML conversion
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m|\x1b\(B")


def _ansi_to_html(text: str) -> str:
    """Convert text with ANSI color codes to a self-contained HTML document."""
    parts: list[str] = []
    last_end = 0
    open_spans = 0

    for m in _ANSI_RE.finditer(text):
        # Escape and append the text before this ANSI code
        parts.append(html_mod.escape(text[last_end : m.start()]))
        last_end = m.end()

        codes_str = m.group(1)
        if codes_str is None:
            # \x1b(B — reset, treat as close
            if open_spans > 0:
                parts.append("</span>")
                open_spans -= 1
            continue

        codes = codes_str.split(";")
        if codes == ["0"] or codes == [""]:
            if open_spans > 0:
                parts.append("</span>")
                open_spans -= 1
            continue

        # Parse 24-bit color: 48;2;R;G;B = bg, 38;2;R;G;B = fg
        styles: list[str] = []
        i = 0
        while i < len(codes):
            c = codes[i]
            if c == "48" and i + 4 < len(codes) and codes[i + 1] == "2":
                r, g, b = codes[i + 2], codes[i + 3], codes[i + 4]
                styles.append(f"background-color:rgb({r},{g},{b})")
                i += 5
            elif c == "38" and i + 4 < len(codes) and codes[i + 1] == "2":
                r, g, b = codes[i + 2], codes[i + 3], codes[i + 4]
                styles.append(f"color:rgb({r},{g},{b})")
                i += 5
            elif c == "97":
                styles.append("color:#fff")
                i += 1
            else:
                i += 1
        if styles:
            parts.append(f'<span style="{";".join(styles)}">')
            open_spans += 1

    # Append remaining text
    parts.append(html_mod.escape(text[last_end:]))
    # Close any unclosed spans
    parts.extend("</span>" for _ in range(open_spans))

    body = "".join(parts)
    return (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'>\n"
        "<style>body{background:#1e1e1e;color:#d4d4d4;font-family:monospace;"
        "white-space:pre;padding:16px;line-height:1.4;font-size:13px;}</style>\n"
        "</head><body>\n"
        f"{body}\n"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_program(path: Path):
    """Load an AIModel asset (.aimodel) and return an AIProgram."""

    return AIModelAsset.load(path).program


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="graphdiff",
        description=(
            "Structural graph diff between two Core AI programs.\n\n"
            "Compares the graph structure of two programs and reports\n"
            "added, removed, and reordered operations. AIModel assets\n"
            "(.aimodel) are loaded via AIProgram which automatically\n"
            "converts the serialized form to the coreai dialect."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s model_a.aimodel model_b.aimodel\n"
            "  %(prog)s --entry-point main model_a.aimodel model_b.aimodel\n"
            "  %(prog)s --max-items 50 model_a.aimodel model_b.aimodel"
        ),
    )
    parser.add_argument(
        "source",
        type=Path,
        metavar="SOURCE",
        help="source AIModel asset (.aimodel)",
    )
    parser.add_argument(
        "target",
        type=Path,
        metavar="TARGET",
        help="target AIModel asset to compare against (.aimodel)",
    )
    parser.add_argument(
        "--entry-point",
        default=None,
        metavar="NAME",
        help="coreai.graph entry point to compare (default: all graphs)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        metavar="N",
        help="limit the number of items shown in the diff table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="write output to FILE (.html for styled HTML, otherwise raw ANSI text)",
    )
    args = parser.parse_args()

    for path in (args.source, args.target):
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            sys.exit(2)

    source = load_program(args.source)
    target = load_program(args.target)

    format_kwargs = {}
    if args.max_items is not None:
        format_kwargs["max_items"] = args.max_items

    if args.entry_point is not None:
        # Single entry-point mode (original behavior)
        diff = compute_coreai_program_diff(source, target, entry_point=args.entry_point)
        text = _format_diff_as_text(
            diff, diff.source_graph, diff.target_graph, **format_kwargs
        )
        is_iso = diff.is_isomorphic
    else:
        # Per-graph composite-aware mode
        results = compute_per_graph_diff(source, target)
        text = _format_multi_graph_diff(results, **format_kwargs)
        is_iso = all(diff is not None and diff.is_isomorphic for _, diff in results)

    print(text)
    if args.output is not None:
        if args.output.suffix.lower() in (".html", ".htm"):
            args.output.write_text(_ansi_to_html(text), encoding="utf-8")
        else:
            # Strip ANSI escape codes for plain text output
            plain = _ANSI_RE.sub("", text)
            args.output.write_text(plain + "\n", encoding="utf-8")
        print(f"\nOutput written to {args.output}", file=sys.stderr)

    sys.exit(0 if is_iso else 1)


if __name__ == "__main__":
    main()
