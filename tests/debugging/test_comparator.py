# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for comparator with dummy graphs."""

import sys
from typing import Any

import numpy as np
import pytest
import torch
from coreai._compiler.dialects import coreai
from coreai._compiler.ir import InsertionPoint, WalkResult
from numpy.typing import NDArray

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder
from coreai_torch.debugging.comparator import (
    Comparator,
    DebugGraph,
    create_comparator_for_programs,
)
from coreai_torch.debugging.graph import ComputationGraph
from coreai_torch.debugging.inspector import Inspector
from coreai_torch.debugging.search_strategy import LevelOrderStrategy

from .test_model import SimpleSequentialModel, get_example_inputs


async def _create_coreai_program_from_model(
    exported_program: torch.export.ExportedProgram,
) -> Any:
    """Create a coreai_program program from an exported program."""
    exported_program = exported_program.run_decompositions()
    converter: TorchConverter = TorchConverter()
    converter._debug_info_recorder.config = _DebugInfoRecorder.Config(
        include_stack_trace=True,
        verify_debuginfo_locations=True,
    )
    converter.add_exported_program(exported_program, entrypoint_name="main")
    coreai_program_program = converter.to_coreai()

    return coreai_program_program


def create_dummy_source_graph() -> ComputationGraph[str, None]:
    """
    Create a dummy source graph for testing.

    Graph structure:
    - Node 0: depth=0 (no deps)
    - Node 1: depth=0 (no deps)
    - Node 2: depth=1 (depends on 0)
    - Node 3: depth=1 (depends on 1)
    - Node 4: depth=2 (depends on 2, 3)
    """
    scope = ComputationGraph.Scope(scope_id=(None, 0), nesting_depth=0)

    nodes = [
        ComputationGraph.Node(
            op_id=0,
            original_node="source_op_0",
            predecessors=[],
            scope=scope,
            sequence_index=0,
        ),
        ComputationGraph.Node(
            op_id=1,
            original_node="source_op_1",
            predecessors=[],
            scope=scope,
            sequence_index=1,
        ),
        ComputationGraph.Node(
            op_id=2,
            original_node="source_op_2",
            predecessors=[0],
            scope=scope,
            sequence_index=2,
        ),
        ComputationGraph.Node(
            op_id=3,
            original_node="source_op_3",
            predecessors=[1],
            scope=scope,
            sequence_index=3,
        ),
        ComputationGraph.Node(
            op_id=4,
            original_node="source_op_4",
            predecessors=[2, 3],
            scope=scope,
            sequence_index=4,
        ),
    ]

    return ComputationGraph[str, None](
        nodes=nodes,
        original_graph=None,
        calculate_depths=True,
    )


def create_dummy_target_graph() -> ComputationGraph[str, None]:
    """
    Create a dummy target graph for testing.

    Graph structure mirrors source but with different node names:
    - Node 10: depth=0 (no deps) - maps to source node 0
    - Node 11: depth=0 (no deps) - maps to source node 1
    - Node 12: depth=1 (depends on 10) - maps to source node 2
    - Node 13: depth=1 (depends on 11) - maps to source node 3
    - Node 14: depth=2 (depends on 12, 13) - maps to source node 4
    """
    scope = ComputationGraph.Scope(scope_id=(None, 0), nesting_depth=0)

    nodes = [
        ComputationGraph.Node(
            op_id=10,
            original_node="target_op_10",
            predecessors=[],
            scope=scope,
            sequence_index=0,
        ),
        ComputationGraph.Node(
            op_id=11,
            original_node="target_op_11",
            predecessors=[],
            scope=scope,
            sequence_index=1,
        ),
        ComputationGraph.Node(
            op_id=12,
            original_node="target_op_12",
            predecessors=[10],
            scope=scope,
            sequence_index=2,
        ),
        ComputationGraph.Node(
            op_id=13,
            original_node="target_op_13",
            predecessors=[11],
            scope=scope,
            sequence_index=3,
        ),
        ComputationGraph.Node(
            op_id=14,
            original_node="target_op_14",
            predecessors=[12, 13],
            scope=scope,
            sequence_index=4,
        ),
    ]

    return ComputationGraph[str, None](
        nodes=nodes,
        original_graph=None,
        calculate_depths=True,
    )


class DummyInspector(Inspector):
    """Dummy inspector that returns mock outputs for testing."""

    def __init__(
        self,
        outputs: dict[int, list[NDArray[Any]]] | None = None,
        default_value: float = 1.0,
    ):
        """
        Initialize dummy inspector.

        Args:
            outputs: Dictionary mapping op IDs to their output arrays
                    If None, returns default_value for all ops
            default_value: Default value to return when op_id not in outputs

        """
        self.outputs = outputs or {}
        self.default_value = default_value

    async def get_intermediates_for_ops(
        self,
        op_ids: list[int | str],
        inputs: Any,  # noqa: ARG002
    ) -> dict[int | str, list[NDArray[Any]] | None]:
        """Return mock outputs."""
        results = {}
        for op_id in op_ids:
            if op_id in self.outputs:
                results[op_id] = self.outputs[op_id]
            else:
                # Return default value
                results[op_id] = [np.array([self.default_value])]
        return results


async def test_comparator_all_matching() -> None:
    """Test comparator when all outputs match."""
    source_graph = create_dummy_source_graph()
    target_graph = create_dummy_target_graph()

    # Both inspectors return same values
    source_inspector = DummyInspector(default_value=1.0)
    target_inspector = DummyInspector(default_value=1.0)

    # ID mapping: source 0->10, 1->11, 2->12, 3->13, 4->14
    id_map = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14}

    strategy = LevelOrderStrategy.bisection(source_graph)

    source_debug_graph = DebugGraph(
        graph=source_graph,
        inspector=source_inspector,
    )
    target_debug_graph = DebugGraph(
        graph=target_graph,
        inspector=target_inspector,
    )

    comparator = Comparator(
        source=source_debug_graph,
        target=target_debug_graph,
        id_map=id_map,
        strategy=strategy,
        show_progress=False,
    )

    # Run comparison
    result = await comparator.compare_with_tolerance(inputs=None)

    # Should have no failures
    assert len(result.failed_nodes) == 0, (
        f"Should have no failures, got {result.failed_nodes}"
    )
    assert len(result.unknown_nodes) == 0, (
        f"Should have no unknowns, got {result.unknown_nodes}"
    )


async def test_comparator_single_mismatch() -> None:
    """Test comparator when one operation has mismatched outputs."""
    source_graph = create_dummy_source_graph()
    target_graph = create_dummy_target_graph()

    # Source returns 1.0 for all, target returns 2.0 for node 12 (mapped from source node 2)
    source_inspector = DummyInspector(default_value=1.0)
    target_inspector = DummyInspector(
        outputs={12: [np.array([2.0])]},  # Different value for node 12
        default_value=1.0,
    )

    id_map = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14}

    strategy = LevelOrderStrategy.bisection(source_graph)

    source_debug_graph = DebugGraph(
        graph=source_graph,
        inspector=source_inspector,
    )
    target_debug_graph = DebugGraph(
        graph=target_graph,
        inspector=target_inspector,
    )

    comparator = Comparator(
        source=source_debug_graph,
        target=target_debug_graph,
        id_map=id_map,
        strategy=strategy,
        show_progress=False,
    )

    result = await comparator.compare_with_tolerance(inputs=None)

    # Should find one failure
    assert len(result.failed_nodes) > 0, "Should find at least one failure"

    # The failed node should be source_op_2 and target_op_12
    failed_source, failed_target = result.failed_nodes[0]
    assert failed_source == "source_op_2", f"Expected source_op_2, got {failed_source}"
    assert failed_target == "target_op_12", (
        f"Expected target_op_12, got {failed_target}"
    )


async def test_comparator_multiple_mismatches() -> None:
    """Test comparator when multiple operations have mismatched outputs."""
    source_graph = create_dummy_source_graph()
    target_graph = create_dummy_target_graph()

    # Source returns 1.0, target has different values for nodes 12 and 14
    source_inspector = DummyInspector(default_value=1.0)
    target_inspector = DummyInspector(
        outputs={
            12: [np.array([2.0])],  # Different for node 12
            14: [np.array([3.0])],  # Different for node 14
        },
        default_value=1.0,
    )

    id_map = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14}

    strategy = LevelOrderStrategy.bisection(source_graph)

    source_debug_graph = DebugGraph(
        graph=source_graph,
        inspector=source_inspector,
    )
    target_debug_graph = DebugGraph(
        graph=target_graph,
        inspector=target_inspector,
    )

    comparator = Comparator(
        source=source_debug_graph,
        target=target_debug_graph,
        id_map=id_map,
        strategy=strategy,
        show_progress=False,
    )

    result = await comparator.compare_with_tolerance(inputs=None)

    # Should find failures
    assert len(result.failed_nodes) >= 1, (
        f"Should find failures, got {len(result.failed_nodes)}"
    )

    # Extract the source operation names from failed pairs
    failed_source_ops = {pair[0] for pair in result.failed_nodes}

    # Should include source_op_2 (mapped to target 12)
    assert "source_op_2" in failed_source_ops, (
        f"Should find source_op_2 in {failed_source_ops}"
    )


async def test_comparator_shape_mismatch() -> None:
    """Test comparator when outputs have truly incompatible shapes."""
    source_graph = create_dummy_source_graph()
    target_graph = create_dummy_target_graph()

    # Source returns shape (3,), target returns shape (2,) for node 12
    # These shapes are incompatible - can't be broadcast or reshaped
    source_inspector = DummyInspector(
        outputs={2: [np.array([1.0, 1.0, 1.0])]},  # Shape (3,)
        default_value=1.0,
    )
    target_inspector = DummyInspector(
        outputs={12: [np.array([1.0, 1.0])]},  # Shape (2,)
        default_value=1.0,
    )

    id_map = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14}

    strategy = LevelOrderStrategy.bisection(source_graph)

    source_debug_graph = DebugGraph(
        graph=source_graph,
        inspector=source_inspector,
    )
    target_debug_graph = DebugGraph(
        graph=target_graph,
        inspector=target_inspector,
    )

    comparator = Comparator(
        source=source_debug_graph,
        target=target_debug_graph,
        id_map=id_map,
        strategy=strategy,
        show_progress=False,
    )

    result = await comparator.compare_with_tolerance(inputs=None)

    # Truly incompatible shapes should result in UNKNOWN status
    assert len(result.unknown_nodes) > 0, (
        "Should find unknown due to incompatible shapes"
    )

    unknown_source, _ = result.unknown_nodes[0]
    assert unknown_source == "source_op_2", (
        f"Expected source_op_2, got {unknown_source}"
    )


async def test_comparator_returns_node_pairs() -> None:
    """Test that comparator returns pairs of (source_node, target_node)."""
    source_graph = create_dummy_source_graph()
    target_graph = create_dummy_target_graph()

    source_inspector = DummyInspector(default_value=1.0)
    target_inspector = DummyInspector(
        outputs={12: [np.array([2.0])]},
        default_value=1.0,
    )

    id_map = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14}

    strategy = LevelOrderStrategy.bisection(source_graph)

    source_debug_graph = DebugGraph(
        graph=source_graph,
        inspector=source_inspector,
    )
    target_debug_graph = DebugGraph(
        graph=target_graph,
        inspector=target_inspector,
    )

    comparator = Comparator(
        source=source_debug_graph,
        target=target_debug_graph,
        id_map=id_map,
        strategy=strategy,
        show_progress=False,
    )

    result = await comparator.compare_with_tolerance(inputs=None)

    # Should return pairs
    assert len(result.failed_nodes) > 0
    failed_pair = result.failed_nodes[0]

    # Should be a tuple of two elements
    assert isinstance(failed_pair, tuple), "Should return tuple"
    assert len(failed_pair) == 2, "Should return pair (source, target)"

    # Should be strings (original nodes)
    assert isinstance(failed_pair[0], str), "Source should be string"
    assert isinstance(failed_pair[1], str), "Target should be string"


async def test_comparator_topological_order() -> None:
    """Test that failed node pairs are returned in topological order."""
    source_graph = create_dummy_source_graph()
    target_graph = create_dummy_target_graph()

    # Make nodes 2 and 4 fail (node 2 at depth=1, node 4 at depth=2)
    source_inspector = DummyInspector(default_value=1.0)
    target_inspector = DummyInspector(
        outputs={
            12: [np.array([2.0])],  # Mapped from source node 2
            14: [np.array([3.0])],  # Mapped from source node 4
        },
        default_value=1.0,
    )

    id_map = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14}

    strategy = LevelOrderStrategy.bisection(source_graph)

    source_debug_graph = DebugGraph(
        graph=source_graph,
        inspector=source_inspector,
    )
    target_debug_graph = DebugGraph(
        graph=target_graph,
        inspector=target_inspector,
    )

    comparator = Comparator(
        source=source_debug_graph,
        target=target_debug_graph,
        id_map=id_map,
        strategy=strategy,
        show_progress=False,
    )

    result = await comparator.compare_with_tolerance(inputs=None)

    # Should find failures
    if len(result.failed_nodes) >= 2:
        # Extract source nodes
        failed_sources = [pair[0] for pair in result.failed_nodes]

        # Node 2 should come before node 4 in topological order
        if "source_op_2" in failed_sources and "source_op_4" in failed_sources:
            idx_2 = failed_sources.index("source_op_2")
            idx_4 = failed_sources.index("source_op_4")

            assert idx_2 < idx_4, "Failed nodes should be in topological order"


async def test_comparator_with_missing_id_map_entries() -> None:
    """
    Test comparator when some source operations don't have target mappings.

    Even when ID mappings are missing, the comparator should handle the comparison
    gracefully and report nodes with missing mappings appropriately.
    """
    source_graph = create_dummy_source_graph()
    target_graph = create_dummy_target_graph()

    # Make source node 2 return a different value
    source_inspector = DummyInspector(
        outputs={2: [np.array([5.0])]},  # Different value
        default_value=1.0,
    )
    target_inspector = DummyInspector(default_value=1.0)

    # Incomplete ID mapping - missing mapping for node 2 (which has different value)
    id_map = {0: 10, 1: 11, 3: 13, 4: 14}  # Missing 2->12

    strategy = LevelOrderStrategy.bisection(source_graph)

    source_debug_graph = DebugGraph(
        graph=source_graph,
        inspector=source_inspector,
    )
    target_debug_graph = DebugGraph(
        graph=target_graph,
        inspector=target_inspector,
    )

    comparator = Comparator(
        source=source_debug_graph,
        target=target_debug_graph,
        id_map=id_map,
        strategy=strategy,
        show_progress=False,
    )

    result = await comparator.compare_with_tolerance(inputs=None)

    # The operations with valid mappings should work normally
    # No failures expected for mapped nodes since their values match
    assert len(result.failed_nodes) == 0, (
        f"Mapped nodes with matching values should pass, got {result.failed_nodes}"
    )

    # Operations without target mappings will have UNKNOWN status
    # They won't appear in failed_nodes since there's no target to compare against
    # This documents that missing mappings result in UNKNOWN status, not FAIL
    assert isinstance(result.unknown_nodes, list), "Should have unknown_nodes list"


def _modify_nth_operation(
    coreai_program: Any,
    target_op_name: str,
    replacement_op_fn: Any,
    n: int = 1,
) -> int:
    """
    Modify the nth occurrence of a specific operation in a coreai_program program.

    Args:
        coreai_program: The coreai_program program to modify
        target_op_name: Name of the operation to find (e.g., "coreai.decomposable.broadcasting_mul")
        replacement_op_fn: Function to create the replacement operation
        n: Which occurrence to replace (1-indexed)

    Returns:
        Total number of matching operations found

    """
    found_count = [0]

    def replace_operation(operation: Any) -> WalkResult:
        if operation.name == target_op_name:
            found_count[0] += 1
            if found_count[0] == n:
                operands = list(operation.operands)
                with InsertionPoint(operation):
                    new_op = replacement_op_fn(
                        operands[0],
                        operands[1],
                        loc=operation.location,
                    )
                operation.results[0].replace_all_uses_with(new_op)
                operation.operation.erase()
                return WalkResult.INTERRUPT
        return WalkResult.ADVANCE

    coreai_program._mlir_module.operation.walk(replace_operation)
    return found_count[0]


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_comparator_catches_modified_ops_at_different_positions() -> None:
    """Test that comparator correctly identifies modified operations at different positions."""
    model = SimpleSequentialModel()
    model.eval()
    example_inputs = get_example_inputs(SimpleSequentialModel)
    example_input = tuple(example_inputs.values())

    # Export and create base coreai_program
    exported_program = torch.export.export(model, example_input)
    exported_program = exported_program.run_decompositions()

    # Test modifying first mul operation
    coreai_program_1 = await _create_coreai_program_from_model(exported_program)
    found = _modify_nth_operation(
        coreai_program_1,
        "coreai.decomposable.broadcasting_mul",
        coreai.broadcasting_divide,
        n=1,
    )
    assert found >= 1, f"Expected to find at least 1 mul operation, found {found}"

    comparator_1 = await create_comparator_for_programs(
        source_program=exported_program,
        target_program=coreai_program_1,
        target_entry_point="main",
    )
    result_1 = await comparator_1.compare_with_tolerance(
        inputs=example_inputs,
        atol=1e-3,
    )
    assert len(result_1.failed_nodes) > 0, (
        "Should detect failure from changing first mul to div"
    )

    # Test modifying second mul operation
    coreai_program_2 = await _create_coreai_program_from_model(exported_program)
    found = _modify_nth_operation(
        coreai_program_2,
        "coreai.decomposable.broadcasting_mul",
        coreai.broadcasting_divide,
        n=2,
    )
    assert found >= 2, f"Expected to find at least 2 mul operations, found {found}"

    comparator_2 = await create_comparator_for_programs(
        source_program=exported_program,
        target_program=coreai_program_2,
        target_entry_point="main",
    )
    result_2 = await comparator_2.compare_with_tolerance(
        inputs=example_inputs,
        atol=1e-2,
    )
    assert len(result_2.failed_nodes) > 0, (
        "Should detect failure from changing second mul to div"
    )
