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
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
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
from .search_strategy import (
    ExhaustiveStrategy,
    SearchStrategy,
)
from .table_writer import _Column, _Row, _TableSpec, _write_table
from .torch_utils import get_torch_to_coreai_output_mapping
from .utils import _plain, _with_debug

logger = logging.getLogger(__name__)

_DEFAULT_RTOL: float = 1e-5
_DEFAULT_ATOL: float = 1e-3

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

        NO_TARGET_VALUE = auto()
        """The source produced a value; the target did not.

        Usually expected rather than wrong: when a lowering fuses several operations into
        one kernel, only the operation producing the kernel's output has a materialised
        value; the operations it absorbed have none.

        Worth separating from UNKNOWN because it says which side is silent, and therefore
        who to ask. A source-side gap is a harness problem; a target-side one is usually
        fusion, and only occasionally a missing output mapping."""

        EXCLUDED = auto()
        """Not a comparison candidate. Two cases:

        * an operation the exclusion policy removes (`aten.view`, `permute`, `reshape`,
          `transpose`);
        * a graph node that is not an operation at all (a placeholder, `get_attr`, or the
          `output` node).

        Split from NOT_MAPPED because the two mean opposite things to a reader. This one is
        expected and uninteresting; on `attention` it accounts for 33 of the 37 nodes with
        no counterpart -- 27 excluded by policy, 6 not operations."""

        NOT_MAPPED = auto()
        """A real operation with no target counterpart. Nothing deliberate about it.

        Either the operation was folded away during lowering, or its mapping is missing --
        both worth knowing, and neither expected. On `attention` this is 4 nodes, all
        `aten.clone.default`, against 33 EXCLUDED and 5 actually compared.

        Distinct from UNKNOWN, where a counterpart existed and its values could not be
        obtained, and from EXCLUDED, which was never a candidate. Reporting all three as
        UNKNOWN turned 5 real comparisons into "44 unknown"."""

        SHAPE_AMBIGUOUS = auto()
        """Both outputs arrived with the same element count, but shapes that will not
        broadcast -- e.g. (2, 8, 128) against (16, 128) -- so whether they hold the same
        tensor is undecidable without assuming one interpretation.

        Pairs with UNKNOWN rather than duplicating it: UNKNOWN means no values were
        obtained, this means both were obtained but the comparison is not decidable.

        Deliberately not compared by reshaping. Row-major order usually does match for a
        pure regrouping of dimensions, but a constant or symmetric tensor compares equal
        under *any* permutation -- and saturating activations produce near-constant
        tensors -- so a reshape-and-compare would report a genuine layout bug as a match.
        Naming the case lets a caller decide, with the shapes in hand."""

    #: Returned by `_align_shapes` when both arrays arrived with the same element count
    #: but shapes that will not broadcast. A plain None cannot express this: it already
    #: means "these cannot be compared at all", and the two lead to different findings.
    _SHAPE_AMBIGUOUS_SENTINEL: tuple[None, None] = (None, None)

    #: Row cap for the comparisons table. A whole-model comparison can run to hundreds of
    #: operations; the caption records what was omitted so the table is never mistaken for
    #: the full set.
    _MAX_ROWS_DISPLAYED: int = 60

    @dataclass
    class BatchResult:
        """Result from processing a batch of operations."""

        pass_count: int
        """Number of operations that passed in this batch."""

        fail_count: int
        """Number of operations that failed in this batch."""

        unknown_count: int
        """Number of operations with unknown status in this batch."""

        statuses: list[tuple[Any, Any, "Comparator.Status", float | None]]
        """(source_node, target_node, status, max_diff) per operation in this batch."""

        shape_ambiguous_count: int = 0
        """How many of `unknown_count` were shape-ambiguous rather than unobtainable.

        A subset of `unknown_count`, not a sibling: the search treats both as unverified,
        but a report needs to say which, since one is missing data and the other is data
        that was deliberately not compared."""

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
        Every operation that did not pass or fail, sorted by source topological order.

        The name is kept for compatibility but is misleading: this is not only
        "couldn't retrieve outputs". Everything that is neither PASS nor FAIL lands here --
        EXCLUDED, NOT_MAPPED, SHAPE_AMBIGUOUS, NO_TARGET_VALUE and UNKNOWN -- because they
        all project to a single unverified result for the search. On `attention` that is 45
        entries of which 4 deserve attention.

        Prefer :meth:`nodes_with_status` or :attr:`reportable_nodes`, which do not conflate
        an operation that was never a candidate with one whose mapping is missing.
        """

        op_statuses: list[tuple[Any, Any, "Comparator.Status", float | None]]
        """
        Every operation checked, in check order, as
        (source_node, target_node, status, max_diff).

        `max_diff` is the largest absolute difference measured, or None when nothing could
        be measured -- which is not the same as 0.0 and must not be read as a match.
        """

        explanations: dict[Any, Any] = field(default_factory=dict)
        """source op id -> why the target had no value, for NO_TARGET_VALUE operations.

        Each value carries a machine-readable `reason` and a `describe()` line. Absent for
        operations that did have a value."""

        @property
        def compared(self) -> list[Any]:
            """
            The operations a verdict was actually reached on.

            `failed_nodes` being empty is not evidence that anything passed: an operation the
            search never reached, or whose target had no value, appears in neither
            `failed_nodes` nor `unknown_nodes`. Check this is non-empty -- and large enough --
            before reading an empty failure list as a clean result: on real models the
            comparable fraction runs well under half.

            Returns:
                Source nodes whose status is PASS or FAIL.

            """
            return [
                source
                for source, _target, status, _max_diff in self.op_statuses
                if status in (Comparator.Status.PASS, Comparator.Status.FAIL)
            ]

        @property
        def pass_count(self) -> int:
            """
            How many operations were compared and agreed.

            Returns:
                The number of PASS verdicts.

            """
            return len(self.nodes_with_status(Comparator.Status.PASS))

        @property
        def verified_fraction(self) -> float:
            """
            The share of examined operations that reached a verdict.

            The denominator is every operation a status was recorded for, so this reports
            coverage of what was looked at rather than of the model.

            Returns:
                A value in 0.0..1.0, or 0.0 when nothing was examined.

            """
            if not self.op_statuses:
                return 0.0
            return len(self.compared) / len(self.op_statuses)

        def nodes_with_status(
            self,
            *statuses: "Comparator.Status",
        ) -> list[tuple[Any, Any]]:
            """
            Operations whose verdict is one of *statuses*.

            Args:
                statuses: Statuses to select.

            Returns:
                (source_node, target_node) pairs, in check order.

            """
            wanted = set(statuses)
            return [
                (source, target)
                for source, target, status, _max_diff in self.op_statuses
                if status in wanted
            ]

        def to_dict(self) -> dict[str, Any]:
            """
            Return the comparison as plain values.

            The node objects are projected to their op ids and names. They are graph
            nodes holding a live `Operation`, so they cannot be serialised, and the
            identity a reader needs from them is the id.

            :attr:`verified_fraction`, :attr:`pass_count` and :attr:`compared` are
            included because they are the point: this class's failure mode is
            reporting ``failed_nodes == []`` for a run that compared nothing, and the
            fraction is what distinguishes the two. A caller reading only the fields
            would have to reimplement which statuses count as compared.

            Returns:
                Every operation's verdict, and the counts that qualify them.

            """

            def name_of(node: Any) -> Any:
                """A node's op id, which is the identity that survives."""
                return getattr(node, "op_id", None)

            return {
                "op_statuses": [
                    {
                        "source_op_id": name_of(source),
                        "target_op_id": name_of(target),
                        "status": _plain(status),
                        "max_diff": _plain(max_diff),
                    }
                    for source, target, status, max_diff in self.op_statuses
                ],
                "failed_op_ids": [name_of(source) for source, _ in self.failed_nodes],
                "unknown_op_ids": [name_of(source) for source, _ in self.unknown_nodes],
                "compared_count": len(self.compared),
                "pass_count": self.pass_count,
                "verified_fraction": _plain(self.verified_fraction),
                "explanations": {
                    str(key): value.to_dict()
                    if hasattr(value, "to_dict")
                    else str(value)
                    for key, value in self.explanations.items()
                },
            }

        @property
        def reportable_nodes(self) -> list[tuple[Any, Any, "Comparator.Status"]]:
            """
            Only the operations a person should look at, worst first.

            FAIL, then NOT_MAPPED and SHAPE_AMBIGUOUS. Deliberately excludes EXCLUDED and
            NO_TARGET_VALUE: an operation that was never a candidate, or whose value was
            absorbed into a fused kernel, is the expected outcome and drowns the rest.
            Sorted by `max_diff` descending so the largest disagreement is first, with
            unmeasured entries last.
            """
            reportable = {
                Comparator.Status.FAIL,
                Comparator.Status.NOT_MAPPED,
                Comparator.Status.SHAPE_AMBIGUOUS,
            }
            selected = [
                (source, target, status, max_diff)
                for source, target, status, max_diff in self.op_statuses
                if status in reportable
            ]
            selected.sort(key=lambda entry: (entry[3] is None, -(entry[3] or 0.0)))
            return [(source, target, status) for source, target, status, _ in selected]

        def max_diff_for(self, source_node: Any) -> float | None:
            """
            The largest absolute difference measured for *source_node*.

            Args:
                source_node: The source operation to look up.

            Returns:
                The difference, or None when it was not measured.

            """
            for source, _target, _status, max_diff in self.op_statuses:
                if source == source_node:
                    return max_diff
            return None

    def __init__(
        self,
        source: DebugGraph[TSourceNode, TSourceGraph],
        target: DebugGraph[TTargetNode, TTargetGraph],
        id_map: dict[ComputationGraph.OpID, ComputationGraph.OpID],
        strategy: SearchStrategy[TSourceNode, TSourceGraph] | None = None,
        show_progress: bool = True,
        exclude_ops: frozenset[str] = _DEFAULT_EXCLUDED_OPS,
    ):
        """
        Initialize the comparator.

        Args:
            source: Source debug graph containing computation graph and inspector
            target: Target debug graph containing computation graph and inspector
            id_map: Mapping from source node IDs to target node IDs
            strategy: Search strategy to use on source graph. Defaults to
                :class:`~coreai_torch.debugging.search_strategy.ExhaustiveStrategy`,
                which checks every operation in one batch; see the note below for why
                that rather than bisection.
            show_progress: Whether to show progress bar during comparison (default: True)
            exclude_ops: Torch operation names treated as deliberate exclusions when
                explaining what the id_map dropped.

        """
        self.source = source
        self.target = target
        self.id_map = id_map
        # Exhaustive by default. Every batch a strategy yields costs a full model
        # execution on *both* sides, so a search only pays when a capture is more expensive
        # than a run. Bisection over a chain degenerates to one node per batch, costing an
        # execution per node against this strategy's one. Choose LevelOrderStrategy.bisection
        # explicitly when intermediates are large or an early exit skips most of the graph.
        self.strategy = strategy or ExhaustiveStrategy(graph=source.graph)
        self.show_progress = show_progress
        # Needed to tell a deliberately-excluded operation from one that simply has no
        # counterpart; the id_map has already dropped both by the time comparison runs.
        self.exclude_ops = exclude_ops
        self._progress_bar: Any = None
        # Every batch costs one full execution per side. A search strategy trades captured
        # tensors against executions, so this is the number to watch when judging whether
        # the strategy is helping: bisection is only worth it when a capture costs more
        # than a run.
        #: source op_id -> why the target had no value, when it did not.
        self._explanations: dict[Any, Any] = {}
        #: op_id -> largest absolute difference measured, or None when unmeasurable.
        self._max_diffs: dict[Any, float | None] = {}
        self._source_executions = 0
        self._target_executions = 0
        self._op_statuses: list[
            tuple[Any, Any, Comparator.Status, float | None]
        ] = []  # (source_node, target_node, status, max_diff) per checked op
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
        batch_statuses: list[tuple[Any, Any, Status, float | None]] | None = None,
    ) -> None:
        """
        Call after checking each batch for custom progress tracking.

        Args:
            batch_size: Number of operations checked in this batch
            pass_count: Total number of passed operations so far
            fail_count: Total number of failed operations so far
            unknown_count: Total number of unknown operations so far
            batch_statuses: List of (source_node, target_node, status, max_diff) tuples
                for operations in this batch

        """
        if batch_statuses:
            self._op_statuses.extend(batch_statuses)

        if self._progress_bar:
            self._progress_bar.update(batch_size)
            # Build status message showing recent operations
            recent_ops = []
            for source_node, _target, status, _max_diff in batch_statuses or []:
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
        elif status == Comparator.Status.SHAPE_AMBIGUOUS:
            # Unverified like UNKNOWN, hence the same colour, but its own glyph: the
            # values were obtained and the shapes are the reason no verdict exists.
            return "yellow", "≠"
        elif status == Comparator.Status.EXCLUDED:
            # Dim: nothing went wrong and these dominate the rows on a real model.
            return "dim", "·"
        elif status == Comparator.Status.NOT_MAPPED:
            # Not dim: a real operation lost its counterpart, which is reportable.
            return "yellow", "!"
        elif status == Comparator.Status.NO_TARGET_VALUE:
            # Dim: on a fused lowering this is the normal outcome for absorbed operations.
            return "dim", "○"
        else:
            return "yellow", "?"

    def _sort_statuses_by_topo_order(
        self,
        statuses: list[tuple[Any, Any, Status, float | None]],
    ) -> list[tuple[Any, Any, Status, float | None]]:
        """
        Sort operation statuses by source graph topological order.

        Args:
            statuses: List of (source_node, target_node, status, max_diff) tuples

        Returns:
            Sorted list of (source_node, target_node, status, max_diff) tuples

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

        counts = Counter(entry[2] for entry in self._op_statuses)
        total = len(self._op_statuses)
        sorted_statuses = self._sort_statuses_by_topo_order(self._op_statuses)

        # One row per comparison, in topological order: which source operation was checked
        # against which target operation, and what came of it. That pairing is the thing
        # being debugged -- a status histogram says how many disagreed but never which, and
        # locating the disagreement is the whole task.
        shown = sorted_statuses[: self._MAX_ROWS_DISPLAYED]
        spec = _TableSpec(
            title=f"Comparisons ({total} operation(s))",
            columns=(
                _Column(header=""),
                _Column(header="source (torch)"),
                _Column(header="target (Core AI)"),
                _Column(header="result"),
                _Column(header="max diff", justify="right"),
            ),
            caption=(
                # Never truncate silently: a table that stops at N reads as complete.
                f"showing first {len(shown)} of {total}; "
                f"{total - len(shown)} row(s) omitted"
                if total > len(shown)
                else None
            ),
        )
        for source_node, target_node, status, max_diff in shown:
            style, symbol = self._get_status_display(status)
            spec.add(
                _Row(
                    cells=(
                        symbol,
                        str(source_node),
                        str(target_node) if target_node is not None else "-",
                        status.name,
                        # "not measured" rather than 0, which would read as a perfect match.
                        f"{max_diff:.3e}" if max_diff is not None else "-",
                    ),
                    style=style,
                ),
            )
        _write_table(spec)

        # Why each missing value is missing. Kept out of the table -- a sixth column would
        # not fit the reasons -- but printed so a reader is not left with a bare category.
        for source_node, _target, status, _diff in shown:
            if status is not Comparator.Status.NO_TARGET_VALUE:
                continue
            explanation = self._explanations.get(
                getattr(source_node, "op_id", source_node),
            )
            if explanation is None:
                # Fall back to matching on the original node, which is what `shown` holds.
                explanation = next(
                    (
                        value
                        for key, value in self._explanations.items()
                        if str(key) == str(source_node)
                    ),
                    None,
                )
            if explanation is not None:
                logger.info("  %s: %s", source_node, explanation.describe())

        # The counts stay, on one line, because a table of rows does not answer "how many
        # of the total" at a glance and a count without its denominator cannot be read.
        logger.info(
            "Comparison complete: %d passed, %d failed, %d shape-ambiguous, %d unknown, "
            "%d not-mapped, %d no-target-value, %d excluded (of %d)",
            counts.get(Comparator.Status.PASS, 0),
            counts.get(Comparator.Status.FAIL, 0),
            counts.get(Comparator.Status.SHAPE_AMBIGUOUS, 0),
            counts.get(Comparator.Status.UNKNOWN, 0),
            counts.get(Comparator.Status.NOT_MAPPED, 0),
            counts.get(Comparator.Status.NO_TARGET_VALUE, 0),
            counts.get(Comparator.Status.EXCLUDED, 0),
            total,
        )

    def _log_execution_cost(self, total_ops: int) -> None:
        """Report what the search actually cost, in executions rather than batches.

        Stated explicitly because the trade-off is easy to get backwards. Capturing every
        operation in a single execution is one run per side; bisection is one run per side
        *per batch*. It pays off only when a capture is expensive relative to a run --
        large intermediates, or an early exit that avoids most of the graph. For small
        tensors on a device, the executions dominate and a single full capture is cheaper.
        """
        logger.info(
            "comparison cost: %d source execution(s), %d target execution(s) for %d "
            "operation(s). A single full capture would be 1 and 1.",
            self._source_executions,
            self._target_executions,
            total_ops,
        )

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
        # No counterpart. Say *why*: a policy exclusion and a missing mapping are opposite
        # findings, and neither is a failed retrieval.
        if target_node is None:
            return self._classify_unmapped(source_node)
        # A silent target is called out because it is the common case and has a known
        # cause (fusion absorbing an operation). A silent source stays UNKNOWN: it has no
        # known cause, and a dedicated status would imply one.
        if target_outputs is None and source_outputs is not None:
            return Comparator.Status.NO_TARGET_VALUE
        if source_outputs is None or target_outputs is None:
            return Comparator.Status.UNKNOWN

        status = check_fn(source_node, target_node, source_outputs, target_outputs)

        return status

    def _classify_unmapped(self, source_node: Any) -> "Comparator.Status":
        """Say why a source operation has no counterpart.

        Args:
            source_node: The original framework node (an `fx.Node` for a torch source).

        Returns:
            EXCLUDED when it was never a candidate, NOT_MAPPED when it is a real operation
            whose counterpart is missing.
        """
        # Not an operation: a placeholder (input or weight), a `get_attr`, or the output
        # node. These have values but no computation, so there is nothing to compare.
        if getattr(source_node, "op", None) in ("placeholder", "output", "get_attr"):
            return Comparator.Status.EXCLUDED

        target = str(getattr(source_node, "target", source_node))
        if any(excluded in target for excluded in self.exclude_ops):
            return Comparator.Status.EXCLUDED

        return Comparator.Status.NOT_MAPPED

    @staticmethod
    def _measure_max_diff(
        source_outputs: list[NDArray[Any] | None],
        target_outputs: list[NDArray[Any] | None],
    ) -> float | None:
        """Largest absolute difference across the outputs the two sides have in common.

        Returns None when nothing could be compared -- a missing output, or shapes that
        will not align. None means "not measured", which is not the same as 0.0, and the
        two must not be conflated in a report.

        Never raises: this is diagnostic, and a report that fails to render because one
        pair could not be measured is worse than a gap in one cell.
        """
        worst: float | None = None
        for src, tgt in zip(
            source_outputs,
            target_outputs,
            strict=False,  # differing arities are legitimate across frameworks
        ):
            if src is None or tgt is None:
                continue
            aligned = Comparator._align_shapes(src, tgt)
            if aligned is None or aligned is Comparator._SHAPE_AMBIGUOUS_SENTINEL:
                continue
            aligned_src, aligned_tgt = aligned
            try:
                left = aligned_src.astype(np.float64)
                right = aligned_tgt.astype(np.float64)
                # Measure only where both sides are finite. `inf - inf` is NaN, and
                # `np.max` propagates it, so a single shared infinity would report the
                # whole comparison as `nan` -- hiding a real difference elsewhere in the
                # array and putting `nan` into the sort key that orders findings by
                # severity. NaN and inf positions carry no magnitude to report; whether
                # they *agree* is the comparison's job, not this measurement's.
                finite = np.isfinite(left) & np.isfinite(right)
                if not finite.any():
                    continue
                difference = float(np.max(np.abs(left[finite] - right[finite])))
            except (ValueError, TypeError):
                continue
            worst = difference if worst is None else max(worst, difference)
        return worst

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
        """Project a Status onto what a search strategy acts on.

        Lossy by design, but the loss has to preserve one distinction: whether the
        operation was ever a candidate. EXCLUDED was projected to UNKNOWN along with
        everything else, and a strategy narrows on UNKNOWN -- so a graph opening with
        placeholders and `aten.view`s, both EXCLUDED, narrowed a level-order search to
        depth 0 before a single value was compared.
        """
        if status == Comparator.Status.PASS:
            return SearchStrategy.ValidationResult.PASS
        elif status == Comparator.Status.FAIL:
            return SearchStrategy.ValidationResult.FAIL
        elif status == Comparator.Status.EXCLUDED:
            # Never a candidate: a policy exclusion, or a node that computes nothing.
            # Reporting it as unverified would have the search hunt for a fault in an
            # operation there was never anything to check.
            return SearchStrategy.ValidationResult.SKIPPED
        elif status == Comparator.Status.SHAPE_AMBIGUOUS:
            # Unverified, not clean: the values were never compared. Reporting it as PASS
            # would let a search prune a subgraph on the strength of a comparison that
            # never happened.
            return SearchStrategy.ValidationResult.UNKNOWN
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
        """Compare one node; return its validation result, target node, and status.

        The status is returned as well as the validation result because the latter is a
        deliberately lossy projection -- SHAPE_AMBIGUOUS collapses into UNKNOWN so a search
        treats it as unverified -- and a report needs the distinction the projection drops.
        """
        target_node = self._get_target_node(source_node)
        source_outputs = source_results.get(source_node.op_id)
        target_outputs = None
        if target_node is not None:
            target_outputs = target_results.get(target_node.op_id)

        # Evaluate the pair
        # Record the magnitude separately from the verdict. `check_fn` returns only a
        # Status, and widening that contract would break every caller and custom callback;
        # measuring here keeps the difference available whatever callback is in use, and
        # keys it by op_id, which is what the display looks it up by. "FAIL at 1e-3" and
        # "FAIL at 18.5" are very different findings that a bare status cannot separate.
        if source_outputs is not None and target_outputs is not None:
            self._max_diffs[source_node.op_id] = self._measure_max_diff(
                source_outputs,
                target_outputs,
            )

        status = self._evaluate_node_pair(
            source_node.original_node,
            target_node.original_node if target_node else None,
            source_outputs,
            target_outputs,
            check_fn,
        )

        # Convert Status to ValidationResult
        if status is Comparator.Status.NO_TARGET_VALUE:
            self._record_missing_value_explanation(source_node)

        return self._status_to_validation_result(status), target_node, status

    def _record_missing_value_explanation(
        self,
        source_node: ComputationGraph.Node,
    ) -> None:
        """Ask the target inspector why it had no value for this operation.

        A status says *what* happened; this says *why*, which is the difference between an
        unexplained gap and a covered one: an operation absorbed into a fused kernel whose
        output was compared and passed is accounted for, while one in no kernel at all is
        a genuinely different situation.

        Best-effort: an inspector that cannot explain is not an error, and a failure here
        must not fail the comparison, which is why it is guarded. Unwraps one layer of
        `CachingInspector`, since the capability lives on the concrete inspector.
        """
        inspector = self.target.inspector
        inspector = getattr(inspector, "_inspector", inspector)
        explain = getattr(inspector, "explain_missing", None)
        if explain is None:
            return

        target_id = self.id_map.get(source_node.op_id)
        if target_id is None:
            return
        try:
            self._explanations[source_node.op_id] = explain(target_id)
        except Exception:  # noqa: BLE001 - diagnostics must not break the comparison
            logger.debug("Could not explain missing value for %s", source_node.op_id)

    def _collect_batch_statuses(
        self,
        batch_results: list[
            tuple[
                ComputationGraph.Node,
                SearchStrategy.ValidationResult,
                "Comparator.Status",
            ]
        ],
    ) -> list[tuple[Any, Any, Status, float | None]]:
        """Collect batch statuses for progress display with target nodes.

        Uses the Status recorded during comparison rather than re-deriving it from the
        ValidationResult: that projection is lossy (SHAPE_AMBIGUOUS becomes UNKNOWN), so
        re-deriving would erase the distinction just before displaying it.
        """
        batch_statuses = []
        for source_node, _vr, status in batch_results:
            target_node = self._get_target_node(source_node)
            max_diff = self._max_diffs.get(source_node.op_id)
            batch_statuses.append(
                (
                    source_node.original_node,
                    target_node.original_node if target_node else None,
                    status,
                    max_diff,
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
        self._source_executions += 1
        self._target_executions += 1
        logger.info(
            "batch %d: %d op(s) -> one source execution + one target execution "
            "(running totals: source=%d, target=%d); ops=%s",
            self._source_executions,
            len(batch),
            self._source_executions,
            self._target_executions,
            [getattr(node, "op_id", node) for node in batch][:12],
        )
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
        shape_ambiguous_count = 0

        for source_node in batch:
            validation_result, target_node, status = self._process_node_comparison(
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
                if status == Comparator.Status.SHAPE_AMBIGUOUS:
                    shape_ambiguous_count += 1
                if target_node is not None:
                    self._unknown_pairs.append((source_node, target_node))

            batch_results.append((source_node, validation_result, status))

        # Update strategy with (node, result) pairs only: the status is for reporting, and
        # a strategy must act on the projection it was designed around.
        await self.strategy.update([(node, vr) for node, vr, _status in batch_results])

        # Collect batch statuses for progress display
        batch_statuses = self._collect_batch_statuses(batch_results)

        return Comparator.BatchResult(
            pass_count=pass_count,
            fail_count=fail_count,
            unknown_count=unknown_count,
            shape_ambiguous_count=shape_ambiguous_count,
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
                explanations=dict(self._explanations),
            )
        finally:
            # Both in `finally`: the `return` above is inside the `try`, so anything after the
            # block is unreachable -- which is how the entire report (the per-operation table,
            # the per-status histogram, the explanations, and closing the progress bar) came
            # to be dead code while the cost log kept firing.
            self._log_execution_cost(total_ops)
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
            # Not reshaped: see Status.SHAPE_AMBIGUOUS for why. Signalled distinctly from
            # the incompatible case below via _SHAPE_AMBIGUOUS_SENTINEL.
            return Comparator._SHAPE_AMBIGUOUS_SENTINEL

        # Shapes are incompatible
        return None

    @staticmethod
    def _compare_output(
        src: NDArray[Any] | None,
        tgt: NDArray[Any] | None,
        *,
        rtol: float,
        atol: float,
        source_node: Any,
        target_node: Any,
    ) -> "Comparator.Status":
        """Compare one output pair and say what happened.

        Assumptions worth knowing:

        * **Broadcasting implies equivalence.** `_align_shapes` will broadcast (2, 8, 1)
          against (2, 8, 256) and compare. If one side genuinely holds a per-row statistic
          and the other a full tensor, the two match only where the tensor is constant
          along that axis -- so this can pass spuriously on constant data. Note the
          inconsistency: a same-size reshape is *refused* for that reason
          (SHAPE_AMBIGUOUS) while a broadcast is accepted.
        * **NaN on both sides counts as agreement** (`equal_nan=True`). A masked softmax or
          an -inf logit produces NaN legitimately and identically on both sides. A NaN on
          only one side still fails, which is the informative case.
        * **`rtol` is not dtype-aware**, unlike `atol`. At fp16, 1e-5 is well below
          epsilon, so the relative term contributes nothing there.
        * **Non-float values must match exactly.** Correct for indices and shapes, where
          being off by one is not a rounding error.
        """
        if src is None or tgt is None:
            return Comparator.Status.UNKNOWN

        aligned = Comparator._align_shapes(src, tgt)
        if aligned is Comparator._SHAPE_AMBIGUOUS_SENTINEL:
            logger.warning(
                "Shape mismatch for %s vs %s: %s vs %s, same element count (%d). Not "
                "compared by reshaping; reported as SHAPE_AMBIGUOUS.",
                source_node,
                target_node,
                src.shape,
                tgt.shape,
                src.size,
            )
            return Comparator.Status.SHAPE_AMBIGUOUS
        if aligned is None:
            logger.warning(
                "Shape mismatch for %s vs %s: %s vs %s, not broadcastable "
                "(sizes %d vs %d)",
                source_node,
                target_node,
                src.shape,
                tgt.shape,
                src.size,
                tgt.size,
            )
            return Comparator.Status.UNKNOWN

        aligned_src, aligned_tgt = aligned
        both_float = np.issubdtype(aligned_src.dtype, np.floating) and np.issubdtype(
            aligned_tgt.dtype,
            np.floating,
        )
        if both_float:
            # equal_nan=True: two sides that both produce NaN at the same position agree.
            # A masked softmax, an -inf logit, a 0/0 in a normalisation all yield NaN
            # legitimately and identically on both sides, and numpy's default of
            # equal_nan=False calls that a difference -- a failure report for a correct
            # lowering. A NaN on only *one* side still fails, which is the case that
            # matters: it means one implementation produced a number and the other did not.
            #
            # np.allclose treats +/-inf as equal to itself already, but only with matching
            # sign, so no separate handling is needed there.
            are_equal = np.allclose(
                aligned_src,
                aligned_tgt,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        else:
            # Non-float values: exact equality, and equal_nan is meaningless for integers
            # so it is not passed.
            are_equal = np.array_equal(aligned_src, aligned_tgt)
        if are_equal:
            return Comparator.Status.PASS

        max_diff = Comparator._measure_max_diff([aligned_src], [aligned_tgt])
        logger.warning(
            "Value mismatch for %s vs %s: shapes %s vs %s, dtypes %s vs %s, "
            "max_diff=%s (rtol=%.1e, atol=%.1e)",
            source_node,
            target_node,
            aligned_src.shape,
            aligned_tgt.shape,
            aligned_src.dtype,
            aligned_tgt.dtype,
            f"{max_diff:.6e}" if max_diff is not None else "unmeasurable",
            rtol,
            atol,
        )
        return Comparator.Status.FAIL

    async def compare_with_tolerance(
        self,
        inputs: Any,
        rtol: float = _DEFAULT_RTOL,
        atol: float = _DEFAULT_ATOL,
        exclude_multi_output: bool = False,
    ) -> Result:
        """
        Compare outputs between graphs with numerical tolerance.

        Args:
            inputs: Model inputs to use during execution
            rtol: Relative tolerance, defaulting to `_DEFAULT_RTOL` (1e-5).
            atol: Absolute tolerance, defaulting to `_DEFAULT_ATOL` (1e-3), chosen from
                measured data rather than numpy's float64 1e-8. Pass a tighter value when
                looking for small systematic error; see that constant for what 1e-3 gives
                up.
            exclude_multi_output: Skip any operation whose source has several outputs,
                reporting it as EXCLUDED rather than comparing.

        Returns:
            Result containing operations where outputs differ beyond tolerance

        """

        def check_fn(
            _source_node: TSourceNode,
            _target_node: TTargetNode,
            source_outputs: list[NDArray[Any] | None] | None,
            target_outputs: list[NDArray[Any] | None] | None,
        ) -> Comparator.Status:
            """Compare an operation's outputs, returning the first non-PASS verdict.

            A thin loop over `_compare_output`, which holds the per-output logic.

            Assumptions, listed because several are not obviously safe:

            * **Outputs correspond positionally** -- source output *i* against target
              output *i*. The debug-info output mappings carry explicit source and target
              indices precisely because this is not guaranteed, and it has been observed
              false: a `split` returned index 1 where index 0 was expected. This is the
              weakest assumption here.
            * **The first bad output decides the verdict.** Later outputs of the same
              operation are not examined, so a multi-output operation reports one problem
              even when it has several.
            * **Differing arities are legitimate**, so only the outputs both sides have are
              compared. torch's `native_layer_norm` returns three values where Core AI's
              `layer_norm` returns one.
            """
            if source_outputs is None or target_outputs is None:
                return Comparator.Status.UNKNOWN

            # Only the outputs both sides have are compared. A source operation and its
            # target counterpart legitimately differ in arity: torch's `native_layer_norm`
            # returns (result, mean, rstd) where Core AI's `layer_norm` returns the result
            # alone, and comparing output 0 to output 0 there is correct.
            #
            # Not suppressed for multi-output operations, though it is tempting. Where a
            # split has several outputs and the runtime returns one, the comparison fails
            # -- and that is a true finding, not noise: the mapping says output 0
            # corresponds to output 0, and the runtime returns a different chunk.
            # Skipping
            # these would hide the only real defect found across thirteen models.
            if exclude_multi_output and len(source_outputs) > 1:
                logger.info(
                    "Skipping %s vs %s at the caller's request: %d source output(s), %d "
                    "target output(s).",
                    _source_node,
                    _target_node,
                    len(source_outputs),
                    len(target_outputs),
                )
                return Comparator.Status.EXCLUDED

            paired = min(len(source_outputs), len(target_outputs))
            if paired == 0:
                return Comparator.Status.UNKNOWN

            for src, tgt in zip(
                source_outputs[:paired],
                target_outputs[:paired],
                strict=True,
            ):
                status = self._compare_output(
                    src,
                    tgt,
                    rtol=rtol,
                    atol=atol,
                    source_node=_source_node,
                    target_node=_target_node,
                )
                if status is not Comparator.Status.PASS:
                    return status

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
        strategy: Search strategy for source graph. Defaults to `ExhaustiveStrategy`,
                  which checks every operation in one batch and reports every
                  divergence; see :class:`Comparator` for why that rather than bisection.
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
    specialization_options = _with_debug(specialization_options)
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
        # Pass the caller's own exclusion set, not the default: the Comparator uses it to
        # explain what the id_map dropped, so a custom set must reach it or an excluded
        # operation is reported as an unexplained missing mapping.
        exclude_ops=exclude_ops,
    )
