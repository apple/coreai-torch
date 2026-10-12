# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for search strategy with hierarchical graphs."""

from coreai_torch.debugging.graph import ComputationGraph
from coreai_torch.debugging.search_strategy import LevelOrderStrategy, SearchStrategy


def create_hierarchical_graph_with_dependencies() -> ComputationGraph:
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

    This tests:
    1. Different dependency depths at same nesting level
    2. Nested regions with their own dependency chains
    3. Nodes with multiple dependencies
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

    return ComputationGraph(nodes=nodes, original_graph=None, calculate_depths=True)


def test_hierarchical_graph_structure() -> None:
    """Test that the hierarchical graph is constructed correctly."""
    graph = create_hierarchical_graph_with_dependencies()

    # Verify total node count
    assert len(graph.get_nodes()) == 10

    # Verify top-level scope (None, 0)
    top_level_scope_id = (None, 0)
    top_level = graph.get_nodes_in_scope(top_level_scope_id)
    assert len(top_level) == 7
    assert all(node.op_id in {0, 1, 2, 3, 4, 5, 6} for node in top_level)

    # Verify nested scope (4, 0) - parent is node 4, scope index 0
    nested_scope_id = (4, 0)
    nested = graph.get_nodes_in_scope(nested_scope_id)
    assert len(nested) == 3
    assert all(node.op_id in {7, 8, 9} for node in nested)

    # Verify nested nodes are children of node 4
    node_4_nested = graph.get_nested_nodes(graph.get_node_by_id(4))
    assert len(node_4_nested) == 3
    assert {n.op_id for n in node_4_nested} == {7, 8, 9}


def test_dependency_depths_calculated_correctly() -> None:
    """Test that dependency depths are calculated correctly."""
    graph = create_hierarchical_graph_with_dependencies()

    # Expected depths for top-level nodes
    expected_depths = {
        0: 0,  # no deps
        1: 0,  # no deps
        2: 1,  # depends on 0
        3: 1,  # depends on 1
        4: 2,  # depends on 2, 3
        5: 2,  # depends on 2
        6: 3,  # depends on 4, 5
    }

    for op_id, expected_depth in expected_depths.items():
        node = graph.get_node_by_id(op_id)
        assert node.depth == expected_depth, (
            f"Node {op_id} has depth {node.depth}, expected {expected_depth}"
        )

    # Expected depths for nested nodes (within their scope)
    nested_depths = {
        7: 0,  # no deps in nested scope
        8: 1,  # depends on 7
        9: 2,  # depends on 8
    }

    for op_id, expected_depth in nested_depths.items():
        node = graph.get_node_by_id(op_id)
        assert node.depth == expected_depth, (
            f"Nested node {op_id} has depth {node.depth}, expected {expected_depth}"
        )


async def test_search_cuts_by_depth_levels() -> None:
    """Test that search strategy cuts based on dependency depth levels."""
    graph = create_hierarchical_graph_with_dependencies()
    strategy = LevelOrderStrategy.bisection(graph)

    # Get first batch - should be nodes at lower depth levels
    batch = await strategy.__anext__()

    # Verify batch is not empty
    assert len(batch) > 0

    # All nodes in batch should have their dependencies satisfied
    # (i.e., all predecessors should also be in the batch or have no predecessors)
    batch_ids = {node.op_id for node in batch}

    for node in batch:
        for pred_id in node.predecessors:
            # Each predecessor must either:
            # 1. Be in the current batch (will be checked together)
            # 2. Have a depth < the minimum depth in the batch (already checked or will be)
            pred_node = graph.get_node_by_id(pred_id)
            min_batch_depth = min(n.depth for n in batch)

            assert pred_id in batch_ids or pred_node.depth < min_batch_depth, (
                f"Node {node.op_id} at depth {node.depth} depends on {pred_id} "
                f"at depth {pred_node.depth}, but {pred_id} is not in batch and "
                f"has depth >= min batch depth {min_batch_depth}"
            )


async def test_search_respects_depth_ordering() -> None:
    """Test that search strategy returns nodes in depth order batches."""
    graph = create_hierarchical_graph_with_dependencies()
    strategy = LevelOrderStrategy.bisection(graph)

    # Get first batch
    batch1 = await strategy.__anext__()

    # Should start with depth 0 and 1 nodes (or similar)
    depths_in_batch = [node.depth for node in batch1]
    min_depth = min(depths_in_batch)
    max_depth = max(depths_in_batch)

    # All depths between min and max should be contiguous
    # (i.e., if we have depth 0 and 2, we must have depth 1)
    for depth in range(min_depth, max_depth + 1):
        assert any(node.depth == depth for node in batch1), (
            f"Batch contains depths {min_depth} to {max_depth} but missing depth {depth}"
        )


async def test_search_narrows_on_failure() -> None:
    """Test that search strategy narrows depth range when failures are found."""
    graph = create_hierarchical_graph_with_dependencies()
    strategy = LevelOrderStrategy.bisection(graph)

    # Get first batch
    batch = await strategy.__anext__()

    # Only mark first half as passing, rest as unknown to leave work for next batch
    results = []
    failed_node = None
    cutoff = len(batch) // 3

    for i, node in enumerate(batch):
        if i < cutoff:
            results.append((node, SearchStrategy.ValidationResult.PASS))
        elif i == cutoff:  # Fail one node
            results.append((node, SearchStrategy.ValidationResult.FAIL))
            failed_node = node
        else:
            # Leave rest unknown so they can be re-checked
            results.append((node, SearchStrategy.ValidationResult.UNKNOWN))

    # Update with results
    await strategy.update(results)

    # Verify the depth range was narrowed to the unknown region
    # The strategy should focus on the area around the failure and unknowns
    assert failed_node is not None

    # The next batch has to arrive, and it has to be narrowed. This was wrapped in
    # `try: ... except StopAsyncIteration: pass`, which meant a strategy that gave up
    # after one batch -- the failure worth catching, since bisection that stops
    # searching finds nothing -- passed without the assertion ever running. Measured:
    # batch2 arrives with 2 nodes at max depth 1, against a failure at depth 2.
    batch2 = await strategy.__anext__()

    max_depth_in_batch2 = max(n.depth for n in batch2)
    assert max_depth_in_batch2 <= failed_node.depth, (
        f"After failure at depth {failed_node.depth}, batch2 has max depth "
        f"{max_depth_in_batch2} which is too deep"
    )


async def test_search_advances_on_pass() -> None:
    """Test that search strategy continues processing when all nodes pass."""
    graph = create_hierarchical_graph_with_dependencies()
    strategy = LevelOrderStrategy.bisection(graph)

    # Get first batch
    batch1 = await strategy.__anext__()

    # Mark all nodes as passing
    results = [(node, SearchStrategy.ValidationResult.PASS) for node in batch1]
    await strategy.update(results)

    # Track all checked nodes across batches
    checked_nodes = {node.op_id for node in batch1}

    # Get subsequent batches until search completes
    batches_processed = 1
    try:
        while batches_processed < 10:  # Prevent infinite loop
            batch = await strategy.__anext__()

            # Mark all as passing and track
            results = [(node, SearchStrategy.ValidationResult.PASS) for node in batch]
            await strategy.update(results)

            for node in batch:
                checked_nodes.add(node.op_id)

            batches_processed += 1
    except StopAsyncIteration:
        # Search completed - this is expected behavior
        pass

    # Verify that strategy eventually checked all or most nodes
    # (at least the top-level ones) when everything passes
    assert len(checked_nodes) >= 7, (  # At least all 7 top-level nodes
        f"Expected to check at least 7 nodes, but only checked {len(checked_nodes)}"
    )


async def test_get_problematic_operations() -> None:
    """Test that problematic operations are correctly identified."""
    graph = create_hierarchical_graph_with_dependencies()
    strategy = LevelOrderStrategy.bisection(graph)

    # Get first batch and mark some as failed
    batch = await strategy.__anext__()

    failed_nodes = [batch[0], batch[1]] if len(batch) > 1 else [batch[0]]
    results = [
        (
            node,
            SearchStrategy.ValidationResult.FAIL
            if node in failed_nodes
            else SearchStrategy.ValidationResult.PASS,
        )
        for node in batch
    ]

    await strategy.update(results)

    # Get problematic operations
    problematic = strategy.get_problematic_operations()

    # Should include the failed nodes
    problematic_ids = {node.op_id for node in problematic}
    for failed_node in failed_nodes:
        assert failed_node.op_id in problematic_ids, (
            f"Failed node {failed_node.op_id} not in problematic operations"
        )


async def _drain(
    strategy: LevelOrderStrategy,
    result: SearchStrategy.ValidationResult,
) -> list[ComputationGraph.Node]:
    """Run *strategy* to completion, reporting *result* for everything it yields."""
    seen: list[ComputationGraph.Node] = []
    while True:
        try:
            batch = await strategy.__anext__()
        except StopAsyncIteration:
            return seen
        seen.extend(batch)
        await strategy.update([(node, result) for node in batch])


async def test_narrowing_actually_stops_deeper_nodes_being_offered() -> None:
    """A narrowed range must prune, not merely be recorded.

    `_depth_range` was computed and logged but never consulted when choosing what to
    yield -- only for the `min_depth >= max_depth` exit -- so every level was offered
    whatever the results said. Bisection therefore visited the whole graph at one model
    execution per side per batch: the most expensive way to check everything, and not
    what it claims to do.

    The fault has to leave something *unverified* at its own depth for this to be
    visible. When a level comes back wholly clean the lower bound marches up to meet the
    narrowed upper bound and the search ends anyway, which is why a simpler version of
    this test passed against the unenforced range.
    """
    graph = create_hierarchical_graph_with_dependencies()
    strategy = LevelOrderStrategy.top_down(graph)

    # Depth 0 is clean.
    batch = await strategy.__anext__()
    assert {node.depth for node in batch} == {0}
    await strategy.update(
        [(node, SearchStrategy.ValidationResult.PASS) for node in batch]
    )

    # Depth 1 holds the fault, and its sibling could not be verified -- so the lower
    # bound cannot advance and only the upper bound moves.
    batch = await strategy.__anext__()
    assert {node.depth for node in batch} == {1}
    assert len(batch) == 2
    await strategy.update(
        [
            (batch[0], SearchStrategy.ValidationResult.FAIL),
            (batch[1], SearchStrategy.ValidationResult.UNKNOWN),
        ]
    )

    remaining = await _drain(strategy, SearchStrategy.ValidationResult.PASS)
    assert all(node.depth <= 1 for node in remaining), (
        "After a failure at depth 1 the search still offered depths "
        f"{sorted({node.depth for node in remaining})}"
    )
    assert {node.op_id for node in strategy.get_problematic_operations()} == {
        batch[0].op_id
    }


async def test_a_level_is_finished_before_the_lower_bound_advances() -> None:
    """A level larger than the batch must not be left half-checked.

    The bound says "everything up to and including this depth has been checked", but the
    guard tested only strictly shallower depths, so the first batch of a split level
    advanced it past its own unchecked siblings. Harmless while the range was never
    consulted; once it prunes, those siblings are dropped and the validator reports a
    clean result over a graph it did not finish.
    """
    graph = create_hierarchical_graph_with_dependencies()
    # Depth 0 holds two nodes, so a batch of one splits it.
    strategy = LevelOrderStrategy.top_down(graph, batch_size=1)

    seen = await _drain(strategy, SearchStrategy.ValidationResult.PASS)

    checked = {node.op_id for node in seen}
    top_level = {node.op_id for node in graph.get_nodes_in_scope((None, 0))}
    assert top_level <= checked, (
        f"Never checked {sorted(top_level - checked)}, which a clean run must cover"
    )


async def test_skipped_nodes_do_not_narrow_the_search() -> None:
    """A node that computes nothing is not evidence, and must not end the search.

    Placeholders sit at depth 0 of every FX graph and the validator reports them as
    SKIPPED. Reported as UNKNOWN they narrowed the upper bound to depth 0 on the very
    first batch, so a level-order search finished having verified nothing -- and said
    so only as an empty failure list, which reads exactly like a clean model.
    """
    graph = create_hierarchical_graph_with_dependencies()
    strategy = LevelOrderStrategy.top_down(graph)

    batch = await strategy.__anext__()
    assert {node.depth for node in batch} == {0}
    await strategy.update(
        [(node, SearchStrategy.ValidationResult.SKIPPED) for node in batch]
    )

    remaining = await _drain(strategy, SearchStrategy.ValidationResult.PASS)
    depths = {node.depth for node in remaining}
    assert depths == {1, 2, 3}, f"Search stopped early, only reaching depths {depths}"
    assert not strategy.get_problematic_operations()
    assert not strategy.get_unknown_operations(), (
        "A skipped node is not an unknown one: it was never a candidate"
    )
