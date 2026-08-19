# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Comparator utility for comparing outputs between two graphs using search strategies.

This module provides a framework for comparing ML model implementations by
identifying operations where outputs differ between a source and target graph.
It uses pluggable search strategies with generic graph representations.

Key components:
- Inspector: Interface for retrieving intermediate operation values
- ComputationGraph: Generic graph representation for different frameworks
- SearchStrategy: Pluggable search algorithms (e.g., bisection, level-order)
- Comparator: Base class coordinating comparison using two inspectors and a strategy
- ID mapping between source and target graph nodes for comparison
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from coreai._compiler.ir import Module, Operation
from coreai.authoring import AIProgram
from coreai.runtime import AIModel, SpecializationOptions
from numpy.typing import NDArray

from .._utils import _ProgressBar
from .graph import (
    ComputationGraph,
    create_graph_from_coreai_program,
    create_graph_from_exported_program,
)
from .inspector import (
    CachingInspector,
    CoreAIInspector,
    Inspector,
    TorchFXInspector,
)
from .search_strategy import LevelOrderStrategy, SearchStrategy
from .torch_utils import get_torch_to_coreai_output_mapping

logger = logging.getLogger(__name__)

# Default set of torch operations to exclude from comparison
# Using frozenset to make it immutable
_DEFAULT_EXCLUDED_OPS: frozenset[str] = frozenset(
    {
        "aten.view",
        "aten.reshape",
        "aten.transpose",
        "aten.permute",
    },
)

# Type variables for generic comparator
# Source graph types
TSourceNode = TypeVar("TSourceNode")
TSourceGraph = TypeVar("TSourceGraph")
# Target graph types
TTargetNode = TypeVar("TTargetNode")
TTargetGraph = TypeVar("TTargetGraph")


@dataclass
class DebugGraph(Generic[TSourceNode, TSourceGraph]):
    """
    A computation graph paired with its inspector for debugging.

    This dataclass groups together related graph and inspector components,
    providing both the graph structure and the ability to retrieve
    intermediate values during execution for debugging purposes.
    """

    graph: ComputationGraph[TSourceNode, TSourceGraph]
    """The computation graph."""

    inspector: Inspector
    """The inspector for retrieving intermediate values from this graph."""


class Comparator(
    Generic[TSourceNode, TSourceGraph, TTargetNode, TTargetGraph],
):
    """
    Generic comparator with pluggable search strategy.

    This class coordinates the comparison of outputs between two graphs by combining
    two ComputationGraph instances, two Inspector instances, and a SearchStrategy.
    The search strategy runs on the source graph, and outputs are compared using
    an ID mapping between source and target nodes.
    """

    class Status(Enum):
        """Result of checking a node."""

        PASS = auto()
        """Comparison passed - outputs match within tolerance."""

        FAIL = auto()
        """Comparison failed - outputs differ."""

        UNKNOWN = auto()
        """Comparison result unknown - couldn't retrieve outputs."""

    @dataclass
    class BatchResult:
        """Result from processing a batch of operations."""

        pass_count: int
        """Number of operations that passed in this batch."""

        fail_count: int
        """Number of operations that failed in this batch."""

        unknown_count: int
        """Number of operations with unknown status in this batch."""

        statuses: list[tuple[Any, Any, "Comparator.Status"]]
        """List of (source_node, target_node, status) tuples for this batch."""

    @dataclass
    class Result:
        """
        Result of comparison check.

        Contains lists of failed and unknown operations, sorted by topological order.
        The nodes are tuples of (source_node, target_node) pairs from the original
        framework-specific nodes.
        """

        failed_nodes: list[tuple[Any, Any]]
        """
        Operations that failed comparison, sorted by source topological order.
        Each element is a (source_node, target_node) tuple.
        """

        unknown_nodes: list[tuple[Any, Any]]
        """
        Operations with unknown results (couldn't retrieve outputs), sorted by source topological order.
        Each element is a (source_node, target_node) tuple.
        """

        op_statuses: list[tuple[Any, Any, "Comparator.Status"]]
        """
        Status of each operation checked, in the order they were checked.
        Each element is a (source_node, target_node, status) tuple.
        """

    def __init__(
        self,
        source: DebugGraph[TSourceNode, TSourceGraph],
        target: DebugGraph[TTargetNode, TTargetGraph],
        id_map: dict[ComputationGraph.OpID, ComputationGraph.OpID],
        strategy: SearchStrategy[TSourceNode, TSourceGraph] | None = None,
        show_progress: bool = True,
    ):
        """
        Initialize the comparator.

        Args:
            source: Source debug graph containing computation graph and inspector
            target: Target debug graph containing computation graph and inspector
            id_map: Mapping from source node IDs to target node IDs
            strategy: Search strategy to use on source graph. Defaults to bisection (batch_size=10)
            show_progress: Whether to show progress bar during comparison (default: True)

        """
        self.source = source
        self.target = target
        self.id_map = id_map
        self.strategy = strategy or LevelOrderStrategy.bisection(
            graph=source.graph,
            batch_size=10,
        )
        self.show_progress = show_progress
        self._progress_bar: Any = None
        self._op_statuses: list[
            tuple[Any, Any, Comparator.Status]
        ] = []  # Store (source_node, target_node, status) for each checked op
        self._failed_pairs: list[
            tuple[ComputationGraph.Node, ComputationGraph.Node]
        ] = []
        self._unknown_pairs: list[
            tuple[ComputationGraph.Node, ComputationGraph.Node]
        ] = []

    def _will_start_comparison(self, total_ops: int) -> None:
        """
        Call before comparison starts for custom progress tracking.

        Args:
            total_ops: Total number of operations to compare

        """
        if self.show_progress:
            self._progress_bar = _ProgressBar(
                total=total_ops,
                description="Comparing operations",
            )

    def _did_check_batch(
        self,
        batch_size: int,
        pass_count: int,
        fail_count: int,
        unknown_count: int,
        batch_statuses: list[tuple[Any, Any, Status]] | None = None,
    ) -> None:
        """
        Call after checking each batch for custom progress tracking.

        Args:
            batch_size: Number of operations checked in this batch
            pass_count: Total number of passed operations so far
            fail_count: Total number of failed operations so far
            unknown_count: Total number of unknown operations so far
            batch_statuses: List of (source_node, target_node, status) tuples for operations in this batch

        """
        if batch_statuses:
            self._op_statuses.extend(batch_statuses)

        if self._progress_bar:
            self._progress_bar.update(batch_size)
            # Build status message showing recent operations
            recent_ops = []
            for source_node, _, status in batch_statuses or []:
                _, status_symbol = self._get_status_display(status)
                recent_ops.append(f"{status_symbol}{source_node}")

            postfix_dict: dict[str, int | str] = {
                "pass": pass_count,
                "fail": fail_count,
                "unknown": unknown_count,
            }
            if recent_ops:
                postfix_dict["recent"] = " ".join(recent_ops)

            self._progress_bar.set_postfix(postfix_dict)

    @staticmethod
    def _get_status_display(status: Status) -> tuple[str, str]:
        """
        Get the style and symbol for a status.

        The style is a `rich` style name rather than an ANSI escape, so a caller
        passing it to
        :func:`~coreai_torch.debugging.annotations._write_line` gets colour on a
        terminal and plain text in a file.

        Returns:
            Tuple of (style_name, status_symbol)

        """
        if status == Comparator.Status.PASS:
            return "green", "✓"
        elif status == Comparator.Status.FAIL:
            return "red", "✗"
        else:
            return "yellow", "?"

    def _sort_statuses_by_topo_order(
        self,
        statuses: list[tuple[Any, Any, Status]],
    ) -> list[tuple[Any, Any, Status]]:
        """
        Sort operation statuses by source graph topological order.

        Args:
            statuses: List of (source_node, target_node, status) tuples

        Returns:
            Sorted list of (source_node, target_node, status) tuples

        """
        op_ids = self.source.graph.get_op_ids()

        # Build index map once for O(1) lookups instead of O(N) op_ids.index() calls
        index_map = {op_id: i for i, op_id in enumerate(op_ids)}

        # Create mapping from source node to its topological index
        node_to_index: dict[Any, int] = {}
        for node in self.source.graph.get_nodes():
            # Use O(1) lookup instead of O(N) op_ids.index()
            node_to_index[node.original_node] = index_map.get(node.op_id, len(op_ids))

        # Sort statuses by topological order
        return sorted(
            statuses,
            key=lambda x: node_to_index.get(x[0], len(op_ids)),
        )

    def _display_comparison_results(self) -> None:
        """Display detailed operation comparison results in topological order."""
        if not self._op_statuses:
            return

        # Log summary statistics
        passed = sum(1 for _, _, s in self._op_statuses if s == Comparator.Status.PASS)
        failed = sum(1 for _, _, s in self._op_statuses if s == Comparator.Status.FAIL)
        unknown = sum(
            1 for _, _, s in self._op_statuses if s == Comparator.Status.UNKNOWN
        )

        logger.info(
            "Comparison complete: %d passed, %d failed, %d unknown",
            passed,
            failed,
            unknown,
        )

        # Sort and display all results in topological order
        sorted_statuses = self._sort_statuses_by_topo_order(self._op_statuses)

        for source_node, target_node, status in sorted_statuses:
            target_str = str(target_node) if target_node is not None else "N/A"
            if status == Comparator.Status.PASS:
                logger.info("✓ %s → %s", source_node, target_str)
            elif status == Comparator.Status.FAIL:
                logger.warning("✗ %s → %s", source_node, target_str)
            elif status == Comparator.Status.UNKNOWN:
                logger.warning("? %s → %s", source_node, target_str)

    def _did_finish_comparison(self) -> None:
        """Call after comparison completes for custom cleanup."""
        if self._progress_bar:
            self._progress_bar.close()
            self._progress_bar = None

        # Display the comparison results
        self._display_comparison_results()

    def _get_target_node(
        self,
        source_node: ComputationGraph.Node,
    ) -> ComputationGraph.Node | None:
        """
        Get the corresponding target node for a source node using the ID map.

        Args:
            source_node: Source node to find target for

        Returns:
            Target node if mapping exists, None otherwise

        """
        target_id = self.id_map.get(source_node.op_id)
        if target_id is None:
            return None
        try:
            return self.target.graph.get_node_by_id(target_id)
        except KeyError:
            return None

    def _evaluate_node_pair(
        self,
        source_node: TSourceNode,
        target_node: TTargetNode | None,
        source_outputs: list[NDArray[Any] | None] | None,
        target_outputs: list[NDArray[Any] | None] | None,
        check_fn: Callable[
            [
                TSourceNode,
                TTargetNode,
                list[NDArray[Any] | None] | None,
                list[NDArray[Any] | None] | None,
            ],
            Status,
        ],
    ) -> Status:
        """
        Evaluate a pair of nodes using the check function.

        Args:
            source_node: Source node (original framework-specific node)
            target_node: Target node (original framework-specific node) or None
            source_outputs: Outputs from source node
            target_outputs: Outputs from target node
            check_fn: Function to check node pair

        Returns:
            Status of the comparison

        """
        # If we couldn't get target node or either output is None, return UNKNOWN
        if target_node is None or source_outputs is None or target_outputs is None:
            return Comparator.Status.UNKNOWN

        return check_fn(source_node, target_node, source_outputs, target_outputs)

    def _sort_node_pairs_by_topo_order(
        self,
        node_pairs: list[tuple[ComputationGraph.Node, ComputationGraph.Node]],
        op_ids: list[ComputationGraph.OpID],
    ) -> list[tuple[Any, Any]]:
        """
        Sort node pairs by source topological order and return original nodes.

        Args:
            node_pairs: List of (source_node, target_node) pairs to sort
            op_ids: List of source operation IDs in topological order

        Returns:
            Original node pairs sorted by source topological order

        """
        # Build index map once for O(1) lookups instead of O(N) op_ids.index() calls
        index_map = {op_id: i for i, op_id in enumerate(op_ids)}

        # Map each pair to its index in topological order (using source node)
        pairs_with_idx: list[
            tuple[
                int,
                tuple[ComputationGraph.Node, ComputationGraph.Node],
            ]
        ] = [
            (index_map[source.op_id], (source, target)) for source, target in node_pairs
        ]

        # Sort by index
        pairs_with_idx.sort(key=lambda x: x[0])

        # Return original node pairs
        return [
            (source.original_node, target.original_node)
            for _, (source, target) in pairs_with_idx
        ]

    async def _fetch_target_intermediates(
        self,
        batch: list[ComputationGraph.Node],
        inputs: Any,
    ) -> dict[ComputationGraph.OpID, list[NDArray[Any] | None] | None]:
        """Fetch intermediates for target nodes corresponding to source batch."""
        target_batch_op_ids: list[ComputationGraph.OpID] = []
        for node in batch:
            target_id = self.id_map.get(node.op_id)
            if target_id is not None:
                target_batch_op_ids.append(target_id)

        if not target_batch_op_ids:
            return {}

        return await self.target.inspector.get_intermediates_for_ops(
            target_batch_op_ids,
            inputs,
        )

    @staticmethod
    def _validation_result_to_status(vr: SearchStrategy.ValidationResult) -> Status:
        """Convert ValidationResult to Status."""
        if vr == SearchStrategy.ValidationResult.PASS:
            return Comparator.Status.PASS
        elif vr == SearchStrategy.ValidationResult.FAIL:
            return Comparator.Status.FAIL
        else:
            return Comparator.Status.UNKNOWN

    @staticmethod
    def _status_to_validation_result(
        status: Status,
    ) -> SearchStrategy.ValidationResult:
        """Convert Status to ValidationResult."""
        if status == Comparator.Status.PASS:
            return SearchStrategy.ValidationResult.PASS
        elif status == Comparator.Status.FAIL:
            return SearchStrategy.ValidationResult.FAIL
        else:
            return SearchStrategy.ValidationResult.UNKNOWN

    def _process_node_comparison(
        self,
        source_node: ComputationGraph.Node,
        source_results: dict[ComputationGraph.OpID, list[NDArray[Any] | None] | None],
        target_results: dict[ComputationGraph.OpID, list[NDArray[Any] | None] | None],
        check_fn: Callable[
            [
                TSourceNode,
                TTargetNode,
                list[NDArray[Any] | None] | None,
                list[NDArray[Any] | None] | None,
            ],
            Status,
        ],
    ) -> tuple[SearchStrategy.ValidationResult, ComputationGraph.Node | None]:
        """Process comparison for a single node and return validation result."""
        target_node = self._get_target_node(source_node)
        source_outputs = source_results.get(source_node.op_id)
        target_outputs = None
        if target_node is not None:
            target_outputs = target_results.get(target_node.op_id)

        # Evaluate the pair
        status = self._evaluate_node_pair(
            source_node.original_node,
            target_node.original_node if target_node else None,
            source_outputs,
            target_outputs,
            check_fn,
        )

        # Convert Status to ValidationResult
        return self._status_to_validation_result(status), target_node

    def _collect_batch_statuses(
        self,
        batch_results: list[
            tuple[ComputationGraph.Node, SearchStrategy.ValidationResult]
        ],
    ) -> list[tuple[Any, Any, Status]]:
        """Collect batch statuses for progress display with target nodes."""
        batch_statuses = []
        for source_node, vr in batch_results:
            target_node = self._get_target_node(source_node)
            status = self._validation_result_to_status(vr)
            batch_statuses.append(
                (
                    source_node.original_node,
                    target_node.original_node if target_node else None,
                    status,
                ),
            )
        return batch_statuses

    async def _process_batch(
        self,
        batch: list[ComputationGraph.Node],
        inputs: Any,
        check_fn: Callable[
            [
                TSourceNode,
                TTargetNode,
                list[NDArray[Any] | None] | None,
                list[NDArray[Any] | None] | None,
            ],
            Status,
        ],
    ) -> BatchResult:
        """
        Process a batch of nodes for comparison.

        Returns:
            BatchResult containing counts and statuses for this batch

        """
        # Fetch intermediates for source and target batches
        source_batch_op_ids = [node.op_id for node in batch]
        source_results = await self.source.inspector.get_intermediates_for_ops(
            source_batch_op_ids,
            inputs,
        )
        target_results = await self._fetch_target_intermediates(batch, inputs)

        # Check each node pair in the batch
        batch_results = []
        pass_count = 0
        fail_count = 0
        unknown_count = 0

        for source_node in batch:
            validation_result, target_node = self._process_node_comparison(
                source_node,
                source_results,
                target_results,
                check_fn,
            )

            # Track results
            if validation_result == SearchStrategy.ValidationResult.PASS:
                pass_count += 1
            elif validation_result == SearchStrategy.ValidationResult.FAIL:
                fail_count += 1
                if target_node is not None:
                    self._failed_pairs.append((source_node, target_node))
            else:
                unknown_count += 1
                if target_node is not None:
                    self._unknown_pairs.append((source_node, target_node))

            batch_results.append((source_node, validation_result))

        # Update strategy with results
        await self.strategy.update(batch_results)

        # Collect batch statuses for progress display
        batch_statuses = self._collect_batch_statuses(batch_results)

        return Comparator.BatchResult(
            pass_count=pass_count,
            fail_count=fail_count,
            unknown_count=unknown_count,
            statuses=batch_statuses,
        )

    async def compare(
        self,
        check_fn: Callable[
            [
                TSourceNode,
                TTargetNode,
                list[NDArray[Any] | None] | None,
                list[NDArray[Any] | None] | None,
            ],
            Status,
        ],
        inputs: Any,
    ) -> Result:
        """
        Compare operations between source and target graphs.

        Args:
            check_fn: Function that takes (source_node, target_node, source_outputs, target_outputs)
                     and returns Status
            inputs: Model inputs to use during execution

        Returns:
            Result containing failed and unknown comparisons sorted by source topological order.
            Each element is a (source_node, target_node) tuple of original framework-specific nodes.

        """
        op_ids = self.source.graph.get_op_ids()
        total_ops = len(op_ids)

        # Initialize progress tracking and result tracking
        pass_count = 0
        fail_count = 0
        unknown_count = 0
        self._failed_pairs = []
        self._unknown_pairs = []
        self._op_statuses = []

        # Notify start of comparison
        self._will_start_comparison(total_ops)

        try:
            async for batch in self.strategy:
                # Process the batch and update counts
                batch_result = await self._process_batch(
                    batch,
                    inputs,
                    check_fn,
                )

                pass_count += batch_result.pass_count
                fail_count += batch_result.fail_count
                unknown_count += batch_result.unknown_count

                # Notify batch completion
                self._did_check_batch(
                    len(batch),
                    pass_count,
                    fail_count,
                    unknown_count,
                    batch_result.statuses,
                )

            # Sort by topological order and get original nodes
            sorted_failed = (
                self._sort_node_pairs_by_topo_order(self._failed_pairs, op_ids)
                if self._failed_pairs
                else []
            )
            sorted_unknown = (
                self._sort_node_pairs_by_topo_order(self._unknown_pairs, op_ids)
                if self._unknown_pairs
                else []
            )

            return Comparator.Result(
                failed_nodes=sorted_failed,
                unknown_nodes=sorted_unknown,
                op_statuses=self._op_statuses.copy(),
            )
        finally:
            # Notify comparison completion
            self._did_finish_comparison()

    @staticmethod
    def _align_shapes(
        src: NDArray[Any],
        tgt: NDArray[Any],
    ) -> tuple[NDArray[Any], NDArray[Any]] | None:
        """
        Try to align shapes of two arrays for comparison.

        Attempts broadcasting first, then reshaping if arrays have same size.

        Args:
            src: Source array
            tgt: Target array

        Returns:
            Tuple of (aligned_src, aligned_tgt) if successful, None if incompatible

        """
        if src.shape == tgt.shape:
            return src, tgt

        # First try broadcasting (for compatible shapes like (2,8,1) and (2,8,256))
        try:
            result_shape = np.broadcast_shapes(src.shape, tgt.shape)
            return np.broadcast_to(src, result_shape), np.broadcast_to(
                tgt,
                result_shape,
            )
        except ValueError:
            pass

        # If broadcasting fails, check if they have the same number of elements
        if src.size == tgt.size:
            # Log a warning about this potentially dangerous fallback
            logger.warning(
                "Shape alignment fallback: reshaping tensors with same size but incompatible shapes. "
                "This may mask layout/permutation bugs. Shapes: %s (%d elements) vs %s (%d elements)",
                src.shape,
                src.size,
                tgt.shape,
                tgt.size,
            )
            # Treat as UNKNOWN instead of reshape-and-compare to avoid masking semantic differences
            return None

        # Shapes are incompatible
        return None

    async def compare_with_tolerance(
        self,
        inputs: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Result:
        """
        Compare outputs between graphs with numerical tolerance.

        Args:
            inputs: Model inputs to use during execution
            rtol: Relative tolerance for comparison (default: 1e-5)
            atol: Absolute tolerance for comparison (default: 1e-8)

        Returns:
            Result containing operations where outputs differ beyond tolerance

        """

        def check_fn(
            _source_node: TSourceNode,
            _target_node: TTargetNode,
            source_outputs: list[NDArray[Any] | None] | None,
            target_outputs: list[NDArray[Any] | None] | None,
        ) -> Comparator.Status:
            """Check if outputs are close within tolerance."""
            if source_outputs is None or target_outputs is None:
                return Comparator.Status.UNKNOWN

            if len(source_outputs) != len(target_outputs):
                return Comparator.Status.FAIL

            for src, tgt in zip(source_outputs, target_outputs, strict=True):
                if src is None or tgt is None:
                    return Comparator.Status.UNKNOWN

                # Try to align shapes if they don't match
                aligned = Comparator._align_shapes(src, tgt)
                if aligned is None:
                    # Shapes are incompatible
                    logger.warning(
                        "Shape mismatch for %s vs %s: shapes %s vs %s (not broadcastable, sizes %d vs %d)",
                        _source_node,
                        _target_node,
                        src.shape,
                        tgt.shape,
                        src.size,
                        tgt.size,
                    )
                    return Comparator.Status.UNKNOWN

                aligned_src, aligned_tgt = aligned

                # Compare values based on dtype
                are_equal = (
                    np.allclose(aligned_src, aligned_tgt, rtol=rtol, atol=atol)
                    if np.issubdtype(aligned_src.dtype, np.floating)
                    and np.issubdtype(aligned_tgt.dtype, np.floating)
                    else np.array_equal(aligned_src, aligned_tgt)
                )
                if not are_equal:
                    max_diff = (
                        np.max(np.abs(aligned_src - aligned_tgt))
                        if aligned_src.dtype == aligned_tgt.dtype
                        and np.issubdtype(aligned_src.dtype, np.number)
                        else None
                    )
                    if max_diff is not None:
                        logger.warning(
                            "Value mismatch for %s vs %s: shapes %s vs %s, dtypes %s vs %s, max_diff=%.6e (rtol=%.1e, atol=%.1e)",
                            _source_node,
                            _target_node,
                            aligned_src.shape,
                            aligned_tgt.shape,
                            aligned_src.dtype,
                            aligned_tgt.dtype,
                            max_diff,
                            rtol,
                            atol,
                        )
                    else:
                        logger.warning(
                            "Value mismatch for %s vs %s: shapes %s vs %s, dtypes %s vs %s",
                            _source_node,
                            _target_node,
                            aligned_src.shape,
                            aligned_tgt.shape,
                            aligned_src.dtype,
                            aligned_tgt.dtype,
                        )
                    return Comparator.Status.FAIL

            return Comparator.Status.PASS

        return await self.compare(check_fn, inputs)


def _create_id_map_from_coreai_program(
    coreai_program: AIProgram,
    source_program: torch.export.ExportedProgram,
    exclude_ops: frozenset[str] = _DEFAULT_EXCLUDED_OPS,
) -> dict[ComputationGraph.OpID, ComputationGraph.OpID]:
    """
    Create ID mapping from torch operations to coreai operations.

    Extracts the mapping from the AIProgram's debug information,
    creating a dictionary that maps source (torch) operation IDs to
    target (coreai) operation IDs. Operations matching the exclude_ops
    set will be filtered out.

    Args:
        coreai_program: AIProgram containing debug mappings
        source_program: PyTorch ExportedProgram to check operation types
        exclude_ops: Frozenset of torch operation names to exclude from the mapping.
                     Defaults to _DEFAULT_EXCLUDED_OPS. Pass frozenset() to disable exclusions.

    Returns:
        Dictionary mapping source operation IDs to target operation IDs,
        with excluded operations filtered out

    """
    # Extract torch to compiled mappings (torch -> coreai)
    torch_to_compiled = get_torch_to_coreai_output_mapping(
        coreai_program,
    )

    # Build set of identifiers to exclude
    excluded_identifiers: set[str] = set()
    if exclude_ops:
        for node in source_program.graph.nodes:
            target_str = str(node.target)
            if any(excluded_op in target_str for excluded_op in exclude_ops):
                excluded_identifiers.add(node.name)

    # Filter torch_to_compiled to remove excluded identifiers
    filtered_torch_to_compiled = {
        identifier: mapping
        for identifier, mapping in torch_to_compiled.items()
        if identifier not in excluded_identifiers
    }

    # Build id_map from filtered torch node identifiers to coreai operation IDs
    id_map: dict[ComputationGraph.OpID, ComputationGraph.OpID] = {}
    for torch_identifier, mapping in filtered_torch_to_compiled.items():
        if torch_identifier not in id_map:
            id_map[torch_identifier] = mapping.target_op_id

    return id_map


async def create_comparator_for_programs(
    source_program: torch.export.ExportedProgram,
    target_program: AIProgram,
    target_entry_point: str,
    strategy: SearchStrategy[torch.fx.Node, torch.fx.Graph] | None = None,
    use_caching: bool = True,
    exclude_ops: frozenset[str] = _DEFAULT_EXCLUDED_OPS,
    specialization_options: SpecializationOptions | None = None,
) -> Comparator[torch.fx.Node, torch.fx.Graph, Operation, Module]:
    """
    Create a comparator between PyTorch ExportedProgram and AIProgram.

    This function creates inspectors for both programs and sets up a comparator
    to compare their outputs operation by operation. The ID mapping between
    source and target operations is automatically extracted from the
    AIProgram's debug information.

    Args:
        source_program: PyTorch ExportedProgram (source model)
        target_program: AIProgram (target compiled model)
        target_entry_point: Name of the coreai.graph in target program
        inspector_type: Type of inspector for the target program
        strategy: Search strategy for source graph (defaults to bisection)
        use_caching: Whether to use caching inspectors (default: True)
        exclude_ops: Frozenset of torch operation names to exclude from comparison.
                     Defaults to _DEFAULT_EXCLUDED_OPS which includes view/reshape
                     operations. Pass frozenset() to disable exclusions.
        specialization_options: Options for configuring model specialization

    Returns:
        Comparator instance configured for comparing the two programs

    """
    # Create ID mapping from torch to coreai operations
    id_map = _create_id_map_from_coreai_program(
        target_program,
        source_program,
        exclude_ops,
    )

    # Create source (PyTorch) inspector
    source_inspector: Inspector = TorchFXInspector(exported_program=source_program)
    if use_caching:
        source_inspector = CachingInspector(source_inspector)

    # Create source graph
    source_graph = create_graph_from_exported_program(source_program)

    # Create target (AIProgram) inspector based on inspector type
    temp_dir = TemporaryDirectory()
    asset_path = Path(temp_dir.name) / "model.aimodel"

    # Create asset from AIProgram and load model from asset
    asset = target_program.save_asset(asset_path)
    specialization_options = (
        specialization_options.with_debug(enabled=True)
        if specialization_options is not None
        else None
    )
    model = await AIModel.load(asset.path, specialization_options)
    target_inspector = CoreAIInspector(
        model=model,
        function_name=target_entry_point,
        temp_dir=temp_dir,
    )

    if use_caching:
        target_inspector = CachingInspector(target_inspector)

    # Create target graph
    target_graph = create_graph_from_coreai_program(
        module=target_program._mlir_module,
        entry_point=target_entry_point,
    )

    # Create DebugGraph instances
    source_debug_graph = DebugGraph(
        graph=source_graph,
        inspector=source_inspector,
    )
    target_debug_graph = DebugGraph(
        graph=target_graph,
        inspector=target_inspector,
    )

    return Comparator(
        source=source_debug_graph,
        target=target_debug_graph,
        id_map=id_map,
        strategy=strategy,
    )
