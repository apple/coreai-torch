# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Generic computation graph representation for search strategies.

This module provides a graph abstraction that can be created from various
computation graph sources, enabling graph-based search strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
from coreai._compiler.ir import Module, Operation, Region
from torch.export import ExportedProgram

# Type variables for generic graph
TNode = TypeVar("TNode")
TGraph = TypeVar("TGraph")


class ComputationGraph(Generic[TNode, TGraph]):
    """
    Private generic representation of a computation graph.

    Provides a minimal interface for graph-based search strategies while
    maintaining type-safe references to the original graph and nodes.

    Type Parameters:
        TNode: Type of the original node objects
        TGraph: Type of the original graph object
    """

    OpID = int | str

    class Scope:
        """
        Represents a hierarchical scope in a computation graph.

        A scope represents nested structures like IR regions, control flow,
        or function bodies. Scope ID uniquely identifies it.

        Attributes:
            scope_id: Unique ID as (parent_node_id, scope_index)
                     parent_node_id=None for top-level scopes
            nesting_depth: How deeply nested (0 = top-level, 1+ = nested)

        """

        def __init__(
            self,
            scope_id: tuple[int | None, int],
            nesting_depth: int = 0,
        ):
            """
            Initialize a scope.

            Args:
                scope_id: Tuple of (parent_node_id, scope_index)
                nesting_depth: Nesting depth (default: 0 for top-level)

            """
            self.scope_id = scope_id
            self.nesting_depth = nesting_depth

        def __repr__(self) -> str:
            """Return string representation."""
            parent, idx = self.scope_id
            if parent is None:
                return f"Scope(top[{idx}], depth={self.nesting_depth})"
            return f"Scope(node={parent}[{idx}], depth={self.nesting_depth})"

    class Node:
        """
        Generic node in a computation graph.

        Represents a single operation with its ID, predecessors, successors,
        and a type-safe reference to the original node object.

        Note: This class uses the type variable from the parent ComputationGraph class.
        """

        def __init__(  # noqa: PLR0913
            self,
            op_id: ComputationGraph.OpID,
            original_node: Any = None,
            predecessors: list[ComputationGraph.OpID,] | None = None,
            successors: list[ComputationGraph.OpID,] | None = None,
            depth: int = 0,
            scope: ComputationGraph.Scope | None = None,
            sequence_index: int = 0,
        ):
            """
            Initialize a graph node.

            Args:
                op_id: Unique identifier for this operation
                original_node: Reference to the original node object
                predecessors: List of operation IDs that this node depends on
                successors: List of operation IDs that depend on this node
                depth: Depth/level of this node in the graph (for level-order strategies)
                scope: Scope this node belongs to (None = top-level/no scopes)
                sequence_index: Position within scope's execution sequences

            """
            self.op_id = op_id
            self.original_node = original_node
            self.predecessors = predecessors or []
            self.successors = successors or []
            self.depth = depth
            self.scope = scope
            self.sequence_index = sequence_index

        @property
        def scope_id(self) -> tuple[int | None, int] | None:
            """Get the scope ID this node belongs to."""
            return self.scope.scope_id if self.scope else None

        @property
        def nesting_depth(self) -> int:
            """Get the nesting depth of this node's scope."""
            return self.scope.nesting_depth if self.scope else 0

    def __init__(
        self,
        nodes: list[Node],
        original_graph: Any = None,
        calculate_depths: bool = True,
    ):
        """
        Initialize the computation graph.

        Args:
            nodes: List of nodes in topological order
            original_graph: Reference to the original graph object
            calculate_depths: Whether to calculate depths based on predecessors (default: True)

        """
        self._nodes = nodes
        self._node_map = {node.op_id: node for node in nodes}
        self.original_graph = original_graph

        # Build scope index for efficient scope-based queries
        self._scope_map: dict[
            tuple[int | None, int],
            list[ComputationGraph.Node],
        ] = {}
        for node in nodes:
            if node.scope_id:
                self._scope_map.setdefault(node.scope_id, []).append(node)

        if calculate_depths:
            self._calculate_depths()

    def _calculate_depths(self) -> None:
        """Calculate depth for each node based on predecessor dependencies."""
        # Iterative Kahn-style topological pass to avoid recursion limits
        # Since self._nodes is already in topological order, we can calculate
        # depths in a single forward pass: depth = 1 + max(predecessor depths)

        depth_cache: dict[ComputationGraph.OpID, int] = {}

        for node in self._nodes:
            max_pred_depth = -1
            for pred_id in node.predecessors:
                if pred_id in self._node_map:
                    # Predecessor depths are already calculated since we're in topological order
                    pred_depth = depth_cache.get(pred_id, 0)
                    max_pred_depth = max(max_pred_depth, pred_depth)

            depth = max_pred_depth + 1
            depth_cache[node.op_id] = depth
            node.depth = depth

    def get_nodes(self) -> list[Node]:
        """
        Get all nodes in the graph in topological order.

        Returns:
            List of Node objects in execution order

        """
        return self._nodes

    def get_node_by_id(self, op_id: ComputationGraph.OpID) -> Node:
        """
        Get a specific node by its ID.

        Args:
            op_id: Operation identifier

        Returns:
            The Node with the specified ID

        """
        return self._node_map[op_id]

    def get_op_ids(self) -> list[ComputationGraph.OpID,]:
        """
        Get ordered list of all operation IDs.

        Returns:
            List of operation IDs in topological order

        """
        return [node.op_id for node in self._nodes]

    def get_nodes_by_depth(self) -> dict[int, list[Node]]:
        """
        Group nodes by their dependency depth/level in the graph.

        Dependency depth is calculated based on predecessors, where nodes with
        no dependencies have depth 0, nodes depending only on depth-0 nodes have
        depth 1, etc. This is different from nesting_depth which represents
        region hierarchy.

        Returns:
            Dictionary mapping dependency depth to list of nodes at that depth

        """
        depth_map: dict[int, list[ComputationGraph.Node]] = {}
        for node in self._nodes:
            depth_map.setdefault(node.depth, []).append(node)
        return depth_map

    def get_max_depth(self) -> int:
        """
        Get the maximum dependency depth in the graph.

        Returns:
            Maximum dependency depth, or -1 if graph is empty

        """
        if not self._nodes:
            return -1
        return max(node.depth for node in self._nodes)

    def get_nodes_at_depth(self, depth: int) -> list[Node]:
        """
        Get all nodes at a specific dependency depth level.

        Args:
            depth: Dependency depth level to query

        Returns:
            List of nodes at the specified dependency depth, in topological order

        """
        return [node for node in self._nodes if node.depth == depth]

    def get_subgraph(self, start_idx: int, end_idx: int) -> list[Node]:
        """
        Get a subgraph as a slice of nodes.

        Args:
            start_idx: Starting index (inclusive)
            end_idx: Ending index (exclusive)

        Returns:
            List of Node objects in the specified range

        """
        return self._nodes[start_idx:end_idx]

    # Scope-aware query methods

    def get_nodes_in_scope(
        self,
        scope_id: tuple[int | None, int],
    ) -> list[Node]:
        """
        Get all nodes within a specific scope.

        Args:
            scope_id: Tuple of (parent_node_id, scope_index)

        Returns:
            List of nodes in the specified scope

        """
        return self._scope_map.get(scope_id, [])

    def get_scopes(self) -> list[tuple[int | None, int]]:
        """
        Get list of all unique scope IDs in the graph.

        Returns:
            List of scope IDs as (parent_node_id, scope_index) tuples

        """
        return list(self._scope_map.keys())

    def get_scope_hierarchy(self) -> dict[int, list[tuple[int | None, int]]]:
        """
        Get scopes grouped by nesting depth.

        Returns:
            Dictionary mapping depth to list of scope IDs at that depth

        """
        hierarchy: dict[int, list[tuple[int | None, int]]] = {}
        for scope_id, nodes in self._scope_map.items():
            if nodes:
                depth = nodes[0].nesting_depth
                hierarchy.setdefault(depth, []).append(scope_id)
        return hierarchy

    def get_nested_nodes(self, parent_node: Node) -> list[Node]:
        """
        Get all nodes from a parent node's nested scopes.

        Uses scope information to find all nodes in scopes where this node is the parent.

        Args:
            parent_node: Node to get nested nodes from

        Returns:
            List of nodes in all nested scopes of the parent node

        """
        nested_nodes = []

        # Find all scopes where this node is the parent
        for scope_id in self.get_scopes():
            parent_id, _ = scope_id
            if parent_id == parent_node.op_id:
                nodes_in_scope = self.get_nodes_in_scope(scope_id)
                nested_nodes.extend(nodes_in_scope)

        return nested_nodes


@dataclass
class _OpWithScope:
    """Private helper to store operation with its scope information."""

    op_id: int
    operation: Operation
    scope: ComputationGraph.Scope
    sequence_index: int


def _collect_operations_hierarchical(
    graph_op: Operation,
) -> list[_OpWithScope]:
    """
    Collect operations from Core AI graph, preserving region hierarchy.

    Args:
        graph_op: The coreai.graph operation to walk

    Returns:
        List of operation data with scope information

    """
    ops_data: list[_OpWithScope] = []

    def walk_region(
        region: Region,
        parent_op_id: int | None,
        region_index: int,
        nesting_depth: int,
    ) -> None:
        """Recursively walk a region preserving hierarchy."""
        scope = ComputationGraph.Scope(
            scope_id=(parent_op_id, region_index),
            nesting_depth=nesting_depth,
        )

        for block_idx, block in enumerate(region.blocks):
            for operation in block.operations:
                op_id_obj = _mlir.get_operation_id(operation.location, "coreai")  # type: ignore[attr-defined]
                if op_id_obj is not None:
                    op_id = op_id_obj.value
                    ops_data.append(
                        _OpWithScope(
                            op_id=op_id,
                            operation=operation.operation,
                            scope=scope,
                            sequence_index=block_idx,
                        ),
                    )

                    # Recursively process nested regions
                    for nested_idx, nested_region in enumerate(operation.regions):
                        walk_region(nested_region, op_id, nested_idx, nesting_depth + 1)

    # Walk all top-level regions of the graph operation
    for region_idx, region in enumerate(graph_op.regions):
        walk_region(region, None, region_idx, 0)

    return ops_data


def _extract_predecessors(
    ops_data: list[_OpWithScope],
) -> dict[int, list[int]]:
    """
    Extract predecessor operation IDs based on operands.

    Args:
        ops_data: List of operation data with scope information

    Returns:
        Dictionary mapping operation ID to list of predecessor IDs

    """
    op_to_id = {op_data.operation: op_data.op_id for op_data in ops_data}
    predecessors: dict[int, list[int]] = {}

    for op_data in ops_data:
        op_id = op_data.op_id
        operation = op_data.operation
        pred_ids = []
        # OpOperandList is iterable in the Python bindings
        for operand in list(operation.operands):  # type: ignore[call-overload]
            defining_op = operand.owner
            if defining_op and defining_op in op_to_id:
                pred_ids.append(op_to_id[defining_op])
        predecessors[op_id] = pred_ids

    return predecessors


def create_graph_from_coreai_program(
    module: Module,
    entry_point: str,
) -> ComputationGraph[Operation, Module]:
    """
    Create a computation graph from an AIProgram (Core AI module).

    Preserves Core AI region hierarchy while building the graph.

    Args:
        module: Core AI module containing coreai.graph operations
        entry_point: Name of the coreai.graph (sym_name attribute value)

    Returns:
        Computation graph with Core AI Operation nodes and Module reference

    Raises:
        ValueError: If the specified graph is not found

    """
    for op in module.body.operations:
        if op.name != "coreai.graph":
            continue
        if hasattr(op, "sym_name") and op.sym_name.value == entry_point:
            # Collect operations preserving region hierarchy
            ops_data = _collect_operations_hierarchical(op)  # type: ignore[arg-type]
            # Extract predecessor relationships
            predecessors = _extract_predecessors(ops_data)

            # Build nodes with scope and sequence information
            nodes: list[ComputationGraph.Node] = [
                ComputationGraph.Node(
                    op_id=op_data.op_id,
                    original_node=op_data.operation,
                    predecessors=list(predecessors.get(op_data.op_id, [])),
                    scope=op_data.scope,
                    sequence_index=op_data.sequence_index,
                )
                for op_data in ops_data
            ]

            # Graph will calculate depths automatically
            return ComputationGraph(
                nodes=nodes,
                original_graph=module,
            )

    msg = f"graph {entry_point!r} not found"
    raise ValueError(msg)


def create_graph_from_exported_program(
    program: ExportedProgram,
) -> ComputationGraph[Any, Any]:
    """
    Create a computation graph from a PyTorch ExportedProgram.

    Torch FX graphs are flat (no nested regions), so all nodes belong
    to a single top-level scope.

    Args:
        program: PyTorch ExportedProgram

    Returns:
        Computation graph with FX Node nodes and FX Graph reference

    """
    # Build node map and extract predecessors
    fx_nodes = list(program.graph.nodes)
    node_to_name = {fx_node: fx_node.name for fx_node in fx_nodes}

    # All torch.fx nodes are in a single top-level scope
    top_level_scope = ComputationGraph.Scope(
        scope_id=(None, 0),
        nesting_depth=0,
    )

    # Predecessors per node, and the inverse relation.
    predecessors_by_name: dict[str, list[str]] = {}
    successors_by_name: dict[str, list[str]] = {}
    for fx_node in fx_nodes:
        pred_names = [
            node_to_name[arg] for arg in fx_node.all_input_nodes if arg in node_to_name
        ]
        predecessors_by_name[fx_node.name] = pred_names
        for pred_name in pred_names:
            # An operation can appear twice in another's inputs (`x + x`); record it once.
            consumers = successors_by_name.setdefault(pred_name, [])
            if fx_node.name not in consumers:
                consumers.append(fx_node.name)

    nodes = []
    for idx, fx_node in enumerate(fx_nodes):
        node = ComputationGraph.Node(
            op_id=fx_node.name,
            original_node=fx_node,
            predecessors=predecessors_by_name[fx_node.name],
            successors=successors_by_name.get(fx_node.name, []),
            scope=top_level_scope,
            sequence_index=idx,
        )
        nodes.append(node)

    # Graph will calculate depths automatically
    return ComputationGraph(nodes=nodes, original_graph=program.graph)
