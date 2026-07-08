# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for validator with hierarchical graphs."""

import sys
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder
from coreai_torch.debugging.graph import (
    ComputationGraph,
    create_graph_from_exported_program,
)
from coreai_torch.debugging.inspector import Inspector
from coreai_torch.debugging.search_strategy import LevelOrderStrategy
from coreai_torch.debugging.validator import (
    Validator,
    create_validator_for_coreai_program,
    create_validator_for_exported_program,
)

from .test_model import ParallelBranchModel, get_example_inputs


def create_hierarchical_graph_with_dependencies() -> ComputationGraph[str, None]:
    """
    Create a dummy hierarchical graph with complex dependencies.

    Graph structure:
    - Top level (nesting_depth=0):
      - Node 0: depth=0 (no deps)
      - Node 1: depth=0 (no deps)
      - Node 2: depth=1 (depends on 0)
      - Node 3: depth=1 (depends on 1)
      - Node 4: depth=2 (depends on 2, 3) - has nested region
      - Node 5: depth=2 (depends on 2)
      - Node 6: depth=3 (depends on 4, 5)

    - Nested region in Node 4 (nesting_depth=1):
      - Node 7: depth=0 (no deps in nested scope)
      - Node 8: depth=1 (depends on 7)
      - Node 9: depth=2 (depends on 8)
    """
    # Create top-level scope
    top_scope = ComputationGraph.Scope(scope_id=(None, 0), nesting_depth=0)

    # Create nested scope (parent is node 4)
    nested_scope = ComputationGraph.Scope(scope_id=(4, 0), nesting_depth=1)

    # Create nodes
    nodes = [
        # Top-level nodes
        ComputationGraph.Node(
            op_id=0,
            original_node="op_0",
            predecessors=[],
            scope=top_scope,
            sequence_index=0,
        ),
        ComputationGraph.Node(
            op_id=1,
            original_node="op_1",
            predecessors=[],
            scope=top_scope,
            sequence_index=1,
        ),
        ComputationGraph.Node(
            op_id=2,
            original_node="op_2",
            predecessors=[0],
            scope=top_scope,
            sequence_index=2,
        ),
        ComputationGraph.Node(
            op_id=3,
            original_node="op_3",
            predecessors=[1],
            scope=top_scope,
            sequence_index=3,
        ),
        ComputationGraph.Node(
            op_id=4,
            original_node="op_4",
            predecessors=[2, 3],
            scope=top_scope,
            sequence_index=4,
        ),
        ComputationGraph.Node(
            op_id=5,
            original_node="op_5",
            predecessors=[2],
            scope=top_scope,
            sequence_index=5,
        ),
        ComputationGraph.Node(
            op_id=6,
            original_node="op_6",
            predecessors=[4, 5],
            scope=top_scope,
            sequence_index=6,
        ),
        # Nested region nodes (inside node 4)
        ComputationGraph.Node(
            op_id=7,
            original_node="op_7",
            predecessors=[],
            scope=nested_scope,
            sequence_index=0,
        ),
        ComputationGraph.Node(
            op_id=8,
            original_node="op_8",
            predecessors=[7],
            scope=nested_scope,
            sequence_index=1,
        ),
        ComputationGraph.Node(
            op_id=9,
            original_node="op_9",
            predecessors=[8],
            scope=nested_scope,
            sequence_index=2,
        ),
    ]

    return ComputationGraph[str, None](
        nodes=nodes,
        original_graph=None,
        calculate_depths=True,
    )


class DummyInspector(Inspector):
    """Dummy inspector that returns mock outputs for testing."""

    def __init__(self, failing_op_ids: set[int] | None = None):
        """
        Initialize dummy inspector.

        Args:
            failing_op_ids: Set of op IDs that should return NaN values (None = all pass)

        """
        self.failing_op_ids = failing_op_ids or set()

    async def get_intermediates_for_ops(
        self,
        op_ids: list[int | str],
        inputs: Any,  # noqa: ARG002
    ) -> dict[int | str, list[NDArray[Any]] | None]:
        """Return mock outputs, with NaN for failing ops."""
        results = {}
        for op_id in op_ids:
            if op_id in self.failing_op_ids:
                # Return NaN for failing operations
                results[op_id] = [np.array([np.nan])]
            else:
                # Return normal values for passing operations
                results[op_id] = [np.array([1.0])]
        return results


async def test_validator_finds_failing_top_level_node() -> None:
    """Test that validator finds a failing node at top level."""
    graph = create_hierarchical_graph_with_dependencies()

    # Node 5 will fail (depth=2, top-level)
    inspector = DummyInspector(failing_op_ids={5})
    strategy = LevelOrderStrategy.bisection(graph)
    validator = Validator(
        graph=graph,
        inspector=inspector,
        strategy=strategy,
        show_progress=False,
    )

    # Run validation
    result = await validator.check_for_nans(inputs=None)

    # Should find node 5 as failed
    assert len(result.failed_nodes) > 0, "Should find at least one failed node"
    assert result.failed_nodes[0] == "op_5", (
        f"First failed node should be op_5, got {result.failed_nodes[0]}"
    )


async def test_validator_finds_failing_nested_node() -> None:
    """Test that validator finds a failing node in nested scope."""
    graph = create_hierarchical_graph_with_dependencies()

    # Node 8 will fail (depth=1, nested in node 4)
    # Also make node 4 fail so the strategy descends into it
    inspector = DummyInspector(failing_op_ids={4, 8})
    strategy = LevelOrderStrategy.bisection(graph)
    validator = Validator(
        graph=graph,
        inspector=inspector,
        strategy=strategy,
        show_progress=False,
    )

    # Run validation
    result = await validator.check_for_nans(inputs=None)

    # Should find both node 4 and node 8 as failed
    assert len(result.failed_nodes) > 0, "Should find at least one failed node"
    failed_ids = list(result.failed_nodes)

    # Node 4 should be found (parent with nested scope)
    assert "op_4" in failed_ids, f"Should find op_4 as failed, got {failed_ids}"

    # Node 8 should also be found (nested node)
    assert "op_8" in failed_ids, (
        f"Should find op_8 as failed (nested), got {failed_ids}"
    )


async def test_validator_no_failures() -> None:
    """Test validator when all nodes pass."""
    graph = create_hierarchical_graph_with_dependencies()

    # No failing nodes
    inspector = DummyInspector(failing_op_ids=None)
    strategy = LevelOrderStrategy.bisection(graph)
    validator = Validator(
        graph=graph,
        inspector=inspector,
        strategy=strategy,
        show_progress=False,
    )

    # Run validation
    result = await validator.check_for_nans(inputs=None)

    # Should have no failed nodes
    assert len(result.failed_nodes) == 0, (
        f"Should have no failed nodes, got {result.failed_nodes}"
    )
    assert len(result.unknown_nodes) == 0, (
        f"Should have no unknown nodes, got {result.unknown_nodes}"
    )


async def test_validator_returns_original_nodes() -> None:
    """Test that validator returns original nodes, not ComputationGraph.Node."""
    graph = create_hierarchical_graph_with_dependencies()

    # Node 2 will fail
    inspector = DummyInspector(failing_op_ids={2})
    strategy = LevelOrderStrategy.bisection(graph)
    validator = Validator(
        graph=graph,
        inspector=inspector,
        strategy=strategy,
        show_progress=False,
    )

    # Run validation
    result = await validator.check_for_nans(inputs=None)

    # Should return original node strings, not ComputationGraph.Node objects
    assert len(result.failed_nodes) > 0
    assert isinstance(result.failed_nodes[0], str), (
        "Should return original node (string), not ComputationGraph.Node"
    )
    assert result.failed_nodes[0] == "op_2"


async def test_validator_topological_order() -> None:
    """Test that failed nodes are returned in topological order."""
    graph = create_hierarchical_graph_with_dependencies()

    # Make multiple nodes fail: node 2 (depth=1) and node 5 (depth=2)
    inspector = DummyInspector(failing_op_ids={2, 5})
    strategy = LevelOrderStrategy.bisection(graph)
    validator = Validator(
        graph=graph,
        inspector=inspector,
        strategy=strategy,
        show_progress=False,
    )

    # Run validation
    result = await validator.check_for_nans(inputs=None)

    # Should return failed nodes in topological order
    # Node 2 comes before node 5 in execution order
    if len(result.failed_nodes) >= 2:
        # In result, op_2 should come before op_5
        result_idx_2 = result.failed_nodes.index("op_2")
        result_idx_5 = result.failed_nodes.index("op_5")

        assert result_idx_2 < result_idx_5, (
            "Failed nodes should be in topological order"
        )


async def test_validator_multiple_failures_at_same_level() -> None:
    """Test that validator finds multiple failures at the same depth level."""
    graph = create_hierarchical_graph_with_dependencies()

    # Make nodes 2 and 3 fail (both at depth=1)
    inspector = DummyInspector(failing_op_ids={2, 3})
    strategy = LevelOrderStrategy.bisection(graph)
    validator = Validator(
        graph=graph,
        inspector=inspector,
        strategy=strategy,
        show_progress=False,
    )

    # Run validation
    result = await validator.check_for_nans(inputs=None)

    # Should find both failed nodes
    assert len(result.failed_nodes) >= 2, (
        f"Should find at least 2 failed nodes, got {len(result.failed_nodes)}"
    )

    # Both op_2 and op_3 should be in failed nodes
    failed_ids = set(result.failed_nodes)
    assert "op_2" in failed_ids, (
        f"Should find op_2 as failed, got {result.failed_nodes}"
    )
    assert "op_3" in failed_ids, (
        f"Should find op_3 as failed, got {result.failed_nodes}"
    )

    # Verify they are returned in topological order (based on sequence_index in graph)
    idx_2 = result.failed_nodes.index("op_2")
    idx_3 = result.failed_nodes.index("op_3")

    # In the graph, node 2 comes before node 3 in topological order
    assert idx_2 < idx_3, "Failed nodes at same depth should be in topological order"


# PyTorch integration tests


@pytest.mark.parametrize("nan_branch", ["fc1", "fc2", None])
async def test_torch_parallel_model_with_nan_injection(nan_branch: str | None) -> None:
    """
    Test validator with PyTorch parallel model and different NaN injection points.

    Parameterized test covering:
    - NaN injection in first branch (fc1)
    - NaN injection in middle branch (fc2)
    - Clean execution (no NaN)
    """
    model = ParallelBranchModel()
    model.nan_injection_branch = nan_branch

    example_inputs = get_example_inputs(ParallelBranchModel)
    example_input = tuple(example_inputs.values())
    exported_program = torch.export.export(model, example_input)

    # Use internal version with custom batch size for more granular testing
    graph = create_graph_from_exported_program(exported_program)
    strategy = LevelOrderStrategy.bisection(graph, batch_size=2)
    validator = create_validator_for_exported_program(
        exported_program,
        strategy=strategy,
    )

    result = await validator.check_for_nans(inputs=example_input)

    # Verify results based on whether NaN was injected
    if nan_branch is None:
        assert len(result.failed_nodes) == 0, (
            f"Should have no failures, got {len(result.failed_nodes)}"
        )
    else:
        assert len(result.failed_nodes) > 0, (
            f"Should find failed nodes with NaN in {nan_branch}"
        )
        # PyTorch - Failed nodes can be inspected for debugging
        _ = [str(n) for n in result.failed_nodes[:3]]


# AIProgram integration tests


async def _create_coreai_program_from_model(
    model: torch.nn.Module,
    model_cls: type[torch.nn.Module],
) -> Any:
    """Create a coreai_program program with debug info from a torch model."""
    model = model.eval()
    example_inputs = get_example_inputs(model_cls)
    exported_program = torch.export.export(model, args=tuple(example_inputs.values()))
    exported_program = exported_program.run_decompositions()
    converter: TorchConverter = TorchConverter()
    converter._debug_info_recorder.config = _DebugInfoRecorder.Config(
        include_stack_trace=True,
        verify_debuginfo_locations=True,
    )
    converter.add_exported_program(exported_program, entrypoint_name="main")
    coreai_program = converter.to_coreai()
    return coreai_program


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Requires loading a runtime asset (AIModel.load); only supported on macOS",
)
@pytest.mark.parametrize(
    "nan_branch",
    [
        pytest.param(
            "fc1",
        ),
        pytest.param(
            "fc2",
        ),
        None,
    ],
)
async def test_coreai_program_with_inspector(
    nan_branch: str | None,
) -> None:
    """
    Test validator with AIProgram across NaN injection points.

    This parameterized test covers NaN injection in the first branch, the
    middle branch, and a clean baseline (no NaN).
    """
    model = ParallelBranchModel()
    model.nan_injection_branch = nan_branch

    # Create AIProgram
    program = await _create_coreai_program_from_model(model, ParallelBranchModel)

    # Create validator with specified inspector type
    validator = await create_validator_for_coreai_program(
        program,
        entry_point="main",
    )

    # Run validation
    result = await validator.check_for_nans(
        inputs=get_example_inputs(ParallelBranchModel)
    )

    # Verify results based on whether NaN was injected
    if nan_branch is None:
        assert len(result.failed_nodes) == 0, (
            f"Should have no failures in clean model, got {len(result.failed_nodes)}"
        )
    else:
        assert len(result.failed_nodes) > 0, (
            f"Should find failed nodes with NaN in {nan_branch}"
        )
        # Failed nodes can be inspected for debugging
        _ = [str(n) for n in result.failed_nodes[:3]]
