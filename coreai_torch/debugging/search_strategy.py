# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Search strategies for isolating problematic operations in computation graphs.

This module provides pluggable search strategies (bisection, level-order) that can
be used with the Validator to efficiently locate failing operations.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Generic, TypeVar

from .graph import ComputationGraph

logger = logging.getLogger(__name__)

# Type variables for generic search strategy
TNode = TypeVar("TNode")
TGraph = TypeVar("TGraph")


class SearchStrategy(ABC, Generic[TNode, TGraph]):
    """
    Async strategy that yields batches (lists) of nodes to check.

    Can be updated with results.
    """

    class ValidationResult(Enum):
        """Result of checking a node."""

        PASS = auto()
        FAIL = auto()
        UNKNOWN = auto()
        SKIPPED = auto()
        """This node was never a candidate, so nothing was checked.

        A placeholder, a `get_attr`, the `output` node, or an operation an exclusion
        policy removed. Distinct from UNKNOWN, which means a value was expected and
        could not be obtained: a strategy narrows on UNKNOWN because unverified work
        might hide the fault, and doing that for a node that computes nothing collapsed
        the search on its first batch -- every FX graph starts with placeholders at
        depth 0, so `LevelOrderStrategy` narrowed to depth 0 before checking anything.

        Recorded as a result rather than withheld: a strategy tracks which nodes are
        still unchecked, so a node it is never told about is one it offers forever."""

    @dataclass
    class CategorizedResults:
        """Categorized node results from a batch check."""

        failed: list[ComputationGraph.Node]
        passed: list[ComputationGraph.Node]
        unknown: list[ComputationGraph.Node]
        skipped: list[ComputationGraph.Node] = field(default_factory=list)
        """Nodes that were never candidates; see `ValidationResult.SKIPPED`. Neither
        evidence of a fault nor evidence against one, so no pass counts them and no
        narrowing acts on them."""

    def __aiter__(self) -> AsyncIterator[list[ComputationGraph.Node]]:
        return self

    @abstractmethod
    async def __anext__(self) -> list[ComputationGraph.Node]:
        """Return the next batch of nodes to check, or raise StopAsyncIteration."""
        raise StopAsyncIteration

    @abstractmethod
    async def update(
        self,
        results: list[tuple[ComputationGraph.Node, ValidationResult]],
    ) -> None:
        """
        Update the strategy with results from checking the last batch.

        Args:
            results: List of tuples containing (node, result) pairs where result
                    indicates PASS, FAIL, or UNKNOWN for each node

        """

    @abstractmethod
    def get_problematic_operations(
        self,
    ) -> list[ComputationGraph.Node]:
        """
        Return operations that are potential problems.

        Returns:
            List of nodes that failed checks

        """

    def get_unknown_operations(self) -> list[ComputationGraph.Node]:
        """
        Return operations with unknown validation results.

        Unknown results occur when outputs couldn't be retrieved or
        there were errors during validation.

        Returns:
            List of nodes that had unknown validation results

        """
        return []


class ExhaustiveStrategy(SearchStrategy[TNode, TGraph]):
    """Check every operation, in a single batch.

    Reports *every* divergence rather than searching for the first one, and costs one
    model execution per side to do it.
    """

    def __init__(self, graph: ComputationGraph[TNode, TGraph]):
        """
        Args:
            graph: Graph whose operations should all be checked.

        """
        self._graph = graph
        self._yielded = False
        self._failed: list[ComputationGraph.Node] = []
        self._unknown: list[ComputationGraph.Node] = []

    async def __anext__(self) -> list[ComputationGraph.Node]:
        """Yield every node once, then stop."""
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        nodes = self._graph.get_nodes()
        logger.info(
            "exhaustive: %d operation(s) in one batch -- 1 source execution + "
            "1 target execution",
            len(nodes),
        )
        return nodes

    async def update(
        self,
        results: list[tuple[ComputationGraph.Node, SearchStrategy.ValidationResult]],
    ) -> None:
        """Record every result. There is nothing to narrow, so nothing is discarded."""
        for node, result in results:
            if result == SearchStrategy.ValidationResult.FAIL:
                self._failed.append(node)
            elif result == SearchStrategy.ValidationResult.UNKNOWN:
                self._unknown.append(node)
        logger.info(
            "exhaustive: %d failed, %d unknown of %d checked",
            len(self._failed),
            len(self._unknown),
            len(results),
        )

    def get_problematic_operations(self) -> list[ComputationGraph.Node]:
        """Every operation that failed, not just the first."""
        return self._failed

    def get_unknown_operations(self) -> list[ComputationGraph.Node]:
        return self._unknown


class LevelOrderStrategy(SearchStrategy[TNode, TGraph]):
    """
    Level-based search strategy with configurable level selection.

    This strategy organizes nodes into depth levels (based on dependency distance from inputs)
    and processes them in batches. A level selector function determines which level to check
    next, enabling different search strategies (bisection, top-down, bottom-up, etc.).

    Search Flow:
    1. Select a depth level using level_selector function
    2. Yield nodes from that level in batches (batch_size at a time)
    3. Receive PASS/FAIL/UNKNOWN results for checked nodes
    4. Narrow search range if failures found (bisection behavior)
    5. Repeat until all relevant nodes checked or issue isolated

    Static factory methods provide common strategies:
    - bisection(): Binary search through depth levels (fastest for root cause isolation)
    - top_down(): Process from inputs to outputs (systematic exploration)
    - bottom_up(): Process from outputs to inputs
    - auto(): Automatically select sparsest levels first
    """

    @staticmethod
    def top_down(
        graph: ComputationGraph[TNode, TGraph],
        batch_size: int = 10,
        initial_scope_id: tuple[int | None, int] | None = None,
    ) -> "LevelOrderStrategy[TNode, TGraph]":
        """
        Create a top-down search strategy (shallowest level first).

        Args:
            graph: The computation graph to search
            batch_size: Number of operations to yield at once for batch processing
            initial_scope_id: Scope to start search in. If None, uses first top-level scope.

        Returns:
            _LevelOrderStrategy configured for top-down search

        """
        return LevelOrderStrategy(
            graph=graph,
            level_selector=lambda _: 0,
            batch_size=batch_size,
            initial_scope_id=initial_scope_id,
        )

    @staticmethod
    def bottom_up(
        graph: ComputationGraph[TNode, TGraph],
        batch_size: int = 10,
        initial_scope_id: tuple[int | None, int] | None = None,
    ) -> "LevelOrderStrategy[TNode, TGraph]":
        """
        Create a bottom-up search strategy (deepest level first).

        Args:
            graph: The computation graph to search
            batch_size: Number of operations to yield at once for batch processing
            initial_scope_id: Scope to start search in. If None, uses first top-level scope.

        Returns:
            LevelOrderStrategy configured for bottom-up search

        """
        return LevelOrderStrategy(
            graph=graph,
            level_selector=lambda level_nodes: len(level_nodes) - 1,
            batch_size=batch_size,
            initial_scope_id=initial_scope_id,
        )

    @staticmethod
    def bisection(
        graph: ComputationGraph[TNode, TGraph],
        batch_size: int = 10,
        initial_scope_id: tuple[int | None, int] | None = None,
    ) -> "LevelOrderStrategy[TNode, TGraph]":
        """
        Create a bisection search strategy (middle level for binary search).

        Args:
            graph: The computation graph to search
            batch_size: Number of operations to yield at once for batch processing
            initial_scope_id: Scope to start search in. If None, uses first top-level scope.

        Returns:
            _LevelOrderStrategy configured for bisection search

        """
        return LevelOrderStrategy(
            graph=graph,
            level_selector=lambda level_nodes: len(level_nodes) // 2,
            batch_size=batch_size,
            initial_scope_id=initial_scope_id,
        )

    @staticmethod
    def auto(
        graph: ComputationGraph[TNode, TGraph],
        batch_size: int = 10,
        initial_scope_id: tuple[int | None, int] | None = None,
    ) -> "LevelOrderStrategy[TNode, TGraph]":
        """
        Automatically choose the level with minimum number of nodes.

        This strategy selects the sparsest level (fewest nodes) at each step,
        which can be more efficient for finding issues in graphs with uneven
        node distribution across depth levels.

        Args:
            graph: The computation graph to search
            batch_size: Number of operations to yield at once for batch processing
            initial_scope_id: Scope to start search in. If None, uses first top-level scope.

        Returns:
            _LevelOrderStrategy configured to select minimum-size levels

        """

        def _min_size_level_selector(
            level_nodes: list[list[ComputationGraph.Node]],
        ) -> int:
            """Select level with minimum number of nodes, or 0 if empty."""
            if not level_nodes:
                return 0
            return min(range(len(level_nodes)), key=lambda i: len(level_nodes[i]))

        return LevelOrderStrategy(
            graph=graph,
            level_selector=_min_size_level_selector,
            batch_size=batch_size,
            initial_scope_id=initial_scope_id,
        )

    def __init__(
        self,
        graph: ComputationGraph[TNode, TGraph],
        level_selector: Callable[[list[list[ComputationGraph.Node]]], int],
        batch_size: int = 10,
        initial_scope_id: tuple[int | None, int] | None = None,
    ):
        """
        Initialize the level-order strategy.

        Args:
            graph: The computation graph to search
            level_selector: Function that takes a list of node lists (one per depth level)
                          and returns the index of the level to process
            batch_size: Number of operations to yield at once for batch processing
            initial_scope_id: Scope to start search in. If None, uses first top-level scope.

        """
        self.graph = graph
        self.batch_size = batch_size
        self.level_selector = level_selector
        self._initial_scope_id = initial_scope_id

        # Track search state
        self._scope_nodes: list[ComputationGraph.Node] = []
        self._depth_range: tuple[int, int] | None = None
        self._node_results: dict[
            ComputationGraph.OpID, SearchStrategy.ValidationResult
        ] = {}
        self._parent_nodes_to_descend: list[ComputationGraph.Node] = []
        self._initialized = False
        self._current_level_nodes: list[ComputationGraph.Node] = []
        self._current_level_index: int = 0

    def _initialize_search_scope(self) -> None:
        """
        Initialize the search scope and depth range.

        Sets up _scope_nodes with the initial set of nodes to search,
        either from a specified scope or from the first top-level scope.
        Calculates the initial depth range for these nodes.
        """
        if self._initial_scope_id is not None:
            scope_nodes = self.graph.get_nodes_in_scope(self._initial_scope_id)
        else:
            # Get first top-level scope (nesting_depth=0)
            scopes = self.graph.get_scopes()
            top_level_scopes = [
                scope_id
                for scope_id in scopes
                if scope_id[0] is None  # parent_node_id is None for top-level
            ]

            if top_level_scopes:
                scope_nodes = self.graph.get_nodes_in_scope(top_level_scopes[0])
            else:
                scope_nodes = self.graph.get_nodes()

        self._scope_nodes = scope_nodes

        # Initialize depth range
        if self._scope_nodes:
            min_depth = min(node.depth for node in self._scope_nodes)
            max_depth = max(node.depth for node in self._scope_nodes)
            self._depth_range = (min_depth, max_depth + 1)
        else:
            self._depth_range = (0, 0)

        self._initialized = True

    def _in_depth_range(self, node: ComputationGraph.Node) -> bool:
        """
        Whether *node* still falls inside the depth range the search has narrowed to.

        Args:
            node: Node to test.

        Returns:
            True when the node is still a candidate.

        """
        if self._depth_range is None:
            return True
        min_depth, max_depth = self._depth_range
        return min_depth <= node.depth < max_depth

    def _group_unchecked_nodes_by_depth(
        self,
    ) -> dict[int, list[ComputationGraph.Node]]:
        """
        Group the still-eligible unchecked nodes by their depth level.

        Filtered by :attr:`_depth_range`, which is what makes narrowing mean anything.
        Without the filter the range was computed, logged and never consulted except
        for the `min_depth >= max_depth` exit, so every level was offered whatever the
        results said and bisection reordered the graph instead of pruning it -- at one
        model execution per side per batch, the most expensive way to check everything.

        Returns:
            Dictionary mapping depth -> list of unchecked nodes at that depth

        """
        level_dict: dict[int, list[ComputationGraph.Node]] = {}

        for node in self._scope_nodes:
            if node.op_id in self._node_results:
                continue
            if not self._in_depth_range(node):
                continue
            level_dict.setdefault(node.depth, []).append(node)

        return level_dict

    def _select_next_level_and_get_first_batch(
        self,
    ) -> list[ComputationGraph.Node]:
        """
        Select the next depth level to check and return the first batch of nodes.

        Uses the level_selector function to choose which depth level to process,
        then initializes batching state and returns the first batch_size nodes.

        Returns:
            First batch of nodes from the selected level, or empty list if no levels remain

        """
        if not self._depth_range or not self._scope_nodes:
            return []

        min_depth, max_depth = self._depth_range
        if min_depth >= max_depth:
            return []

        # Get unchecked nodes grouped by level
        level_dict = self._group_unchecked_nodes_by_depth()
        if not level_dict:
            return []

        # Convert dict to list of lists, ordered by depth
        sorted_depths = sorted(level_dict.keys())
        level_nodes = [level_dict[depth] for depth in sorted_depths]

        # Use level_selector to choose which level index to process
        selected_index = self.level_selector(level_nodes)

        # Store nodes at the selected level for batching
        self._current_level_nodes = level_nodes[selected_index]
        self._current_level_index = 0

        # Return first batch from the selected level
        return self._get_next_batch_from_current_level()

    def _get_next_batch_from_current_level(
        self,
    ) -> list[ComputationGraph.Node]:
        """
        Get the next batch of nodes from the currently selected level.

        Nodes narrowed out since the level was selected are stepped over rather than
        yielded: a failure narrows the range mid-level, and the remainder of that level
        was still being handed out because only level *selection* consulted the range.

        Returns:
            Next batch of eligible nodes from the current position, or an empty list
            when the level is exhausted.

        """
        batch: list[ComputationGraph.Node] = []
        while (
            self._current_level_index < len(self._current_level_nodes)
            and len(batch) < self.batch_size
        ):
            node = self._current_level_nodes[self._current_level_index]
            self._current_level_index += 1
            if self._in_depth_range(node):
                batch.append(node)

        return batch

    def _descend_into_nested_scope(
        self,
    ) -> list[ComputationGraph.Node]:
        """
        Descend into a nested scope (e.g., inside a control flow operation).

        Pops a parent node from _parent_nodes_to_descend and sets up a new search
        within that node's nested region. Resets batching state for the new scope.

        Returns:
            First batch from the nested scope, or empty list if no nested scopes remain

        """
        if not self._parent_nodes_to_descend:
            return []

        parent_node = self._parent_nodes_to_descend.pop(0)
        nested_nodes = self.graph.get_nested_nodes(parent_node)

        if not nested_nodes:
            return []

        # Set up new search in nested scope
        self._scope_nodes = nested_nodes
        if nested_nodes:
            min_depth = min(node.depth for node in nested_nodes)
            max_depth = max(node.depth for node in nested_nodes)
            self._depth_range = (min_depth, max_depth + 1)
        else:
            self._depth_range = (0, 0)

        # Reset batching state for new scope
        self._current_level_nodes = []
        self._current_level_index = 0

        return self._select_next_level_and_get_first_batch()

    async def __anext__(self) -> list[ComputationGraph.Node]:
        """
        Return the next batch of nodes to check.

        This implements the async iterator protocol. The strategy:
        1. If still processing current level, return next batch from it
        2. Otherwise, select a new level and return its first batch
        3. If no levels remain, try descending into nested scopes
        4. If nothing left, raise StopAsyncIteration to signal completion

        Returns:
            Next batch of nodes to validate

        Raises:
            StopAsyncIteration: When search is complete

        """
        if not self._initialized:
            self._initialize_search_scope()

        # First, try to get next batch from current level
        if self._current_level_nodes and self._current_level_index < len(
            self._current_level_nodes,
        ):
            batch = self._get_next_batch_from_current_level()
            if batch:
                logger.info(
                    "Checking %d op(s) %s at depth(s) %s",
                    len(batch),
                    [node.op_id for node in batch],
                    {node.depth for node in batch},
                )
                return batch

        # If no more batches in current level, select a new level
        batch = self._select_next_level_and_get_first_batch()

        if not batch:
            # Try descending into nested scopes (e.g., inside control flow operations)
            batch = self._descend_into_nested_scope()

            if not batch:
                logger.info("Search complete")
                raise StopAsyncIteration

        logger.info(
            "Checking %d op(s) %s at depth(s) %s",
            len(batch),
            [node.op_id for node in batch],
            {node.depth for node in batch},
        )
        return batch

    def _categorize_validation_results(
        self,
        results: list[tuple[ComputationGraph.Node, SearchStrategy.ValidationResult]],
    ) -> SearchStrategy.CategorizedResults:
        """
        Categorize validation results into PASS/FAIL/UNKNOWN groups.

        Args:
            results: List of (node, result) tuples from validation

        Returns:
            CategorizedResults with nodes grouped by validation status

        """
        failed_nodes = []
        passed_nodes = []
        unknown_nodes = []
        skipped_nodes = []

        for node, result in results:
            if result == SearchStrategy.ValidationResult.FAIL:
                failed_nodes.append(node)
            elif result == SearchStrategy.ValidationResult.PASS:
                passed_nodes.append(node)
            elif result == SearchStrategy.ValidationResult.SKIPPED:
                skipped_nodes.append(node)
            else:
                unknown_nodes.append(node)

        return SearchStrategy.CategorizedResults(
            failed=failed_nodes,
            passed=passed_nodes,
            unknown=unknown_nodes,
            skipped=skipped_nodes,
        )

    def _track_failed_parent_nodes_for_descent(
        self,
        failed_nodes: list[ComputationGraph.Node],
        unknown_nodes: list[ComputationGraph.Node],
    ) -> None:
        """
        Track parent nodes with FAIL/UNKNOWN results for later descent into nested scopes.

        When a node with nested regions (e.g., control flow ops) fails or has unknown
        results, we queue it for descent to search within its nested scope.

        Args:
            failed_nodes: Nodes that failed validation
            unknown_nodes: Nodes with unknown validation results

        """
        candidates = failed_nodes + unknown_nodes

        for node in candidates:
            if (
                self.graph.get_nested_nodes(node)
                and node not in self._parent_nodes_to_descend
            ):
                self._parent_nodes_to_descend.append(node)

    async def update(
        self,
        results: list[tuple[ComputationGraph.Node, SearchStrategy.ValidationResult]],
    ) -> None:
        """
        Update the strategy with results from checking the last batch.

        Args:
            results: List of tuples containing (node, result) pairs where result
                    indicates PASS, FAIL, or UNKNOWN for each node

        """
        if not results:
            return

        # Log results for each node
        for node, result in results:
            logger.debug("Op %s (depth %d): %s", node.op_id, node.depth, result.name)

        # Store all results
        for node, result in results:
            self._node_results[node.op_id] = result

        # Categorize results and track parent nodes for potential descent
        categorized = self._categorize_validation_results(results)

        # Log summary
        if categorized.failed:
            logger.warning(
                "Failed: %s at depth(s) %s",
                [n.op_id for n in categorized.failed],
                {n.depth for n in categorized.failed},
            )
        if categorized.unknown:
            logger.warning(
                "Unknown: %s at depth(s) %s",
                [n.op_id for n in categorized.unknown],
                {n.depth for n in categorized.unknown},
            )
        if categorized.passed:
            logger.info(
                "Passed: %s at depth(s) %s",
                [n.op_id for n in categorized.passed],
                {n.depth for n in categorized.passed},
            )

        self._track_failed_parent_nodes_for_descent(
            categorized.failed,
            categorized.unknown,
        )

        # Narrow search range when failures found (implements bisection behavior)
        old_range = self._depth_range
        self._narrow_search_range_on_failure(categorized)

        if old_range != self._depth_range:
            logger.info("Narrowed range %s → %s", old_range, self._depth_range)

    def _narrow_search_range_on_failure(
        self,
        categorized: SearchStrategy.CategorizedResults,
    ) -> None:
        """
        Narrow the search depth range based on validation results (bisection behavior).

        Bisection logic:
        - When nodes FAIL or UNKNOWN: the root cause is at or before that depth,
          so narrow the upper bound to focus on earlier depths
        - When all nodes PASS (no failures/unknowns): Only narrow if we're actively
          searching for failures. Don't narrow when everything passes, as we need
          to ensure all nodes eventually get checked.

        SKIPPED results are ignored on both counts. They are neither evidence of a
        fault nor evidence against one, so they must not pull the upper bound down --
        and they must not let the lower bound advance past depths that were never
        actually verified.

        The range is enforced by :meth:`_group_unchecked_nodes_by_depth` and
        :meth:`_get_next_batch_from_current_level`, so narrowing here genuinely stops
        deeper nodes being offered.

        Args:
            categorized: Categorized validation results from the last batch

        """
        if not self._depth_range:
            return

        min_depth, max_depth = self._depth_range

        # If we have failures or unknowns, narrow the upper bound
        # UNKNOWN is treated like FAIL because we cannot verify correctness
        if categorized.failed or categorized.unknown:
            # Find the minimum depth among failed/unknown nodes
            problematic_nodes = categorized.failed + categorized.unknown
            min_problematic_depth = min(node.depth for node in problematic_nodes)
            # Narrow the upper bound to just past the problematic depth
            # This focuses search on finding the root cause at or before the issue
            max_depth = min(max_depth, min_problematic_depth + 1)
            self._depth_range = (min_depth, max_depth)
        # If all nodes passed, check if there are unchecked nodes at this depth or
        # shallower. Only narrow the lower bound once they have all been checked.
        elif categorized.passed:
            # Find the maximum depth among passed nodes
            max_passed_depth = max(node.depth for node in categorized.passed)

            # `<=`, not `<`: the bound about to be set claims everything up to *and
            # including* `max_passed_depth` has been checked, so the level itself has to
            # be finished. A level larger than `batch_size` arrives in several batches,
            # and testing only strictly shallower depths let the first batch advance the
            # bound past its own unchecked siblings. Inert while the range was never
            # consulted; now that it prunes, those siblings would simply never be
            # checked, and a validator would report a clean result having skipped them.
            unchecked_at_or_above = any(
                node.depth <= max_passed_depth and node.op_id not in self._node_results
                for node in self._scope_nodes
            )

            if not unchecked_at_or_above:
                # Narrow the lower bound to just after the passed depth
                # We know we've checked everything up to and including this depth
                min_depth = max(min_depth, max_passed_depth + 1)
                self._depth_range = (min_depth, max_depth)

    def get_problematic_operations(
        self,
    ) -> list[ComputationGraph.Node]:
        """
        Return operations that are potential problems.

        Returns:
            List of nodes that failed checks

        """
        return [
            node
            for node in self.graph.get_nodes()
            if self._node_results.get(node.op_id)
            == SearchStrategy.ValidationResult.FAIL
        ]

    def get_unknown_operations(self) -> list[ComputationGraph.Node]:
        """
        Return operations with unknown validation results.

        Unknown results occur when outputs couldn't be retrieved or
        there were errors during validation.

        Returns:
            List of nodes that had unknown validation results

        """
        return [
            node
            for node in self.graph.get_nodes()
            if self._node_results.get(node.op_id)
            == SearchStrategy.ValidationResult.UNKNOWN
        ]
