# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Validator utility for isolating NaN/inf issues using search strategies.

This module provides a framework for debugging numerical issues in ML models by
identifying the first operation that produces invalid values (NaN/inf). It uses
pluggable search strategies (bisection, level-order) with a generic graph representation.

Key components:
- Inspector: Interface for retrieving intermediate operation values
- ComputationGraph: Generic graph representation for different frameworks
- SearchStrategy: Pluggable search algorithms (e.g., bisection, level-order)
- Validator: Base class coordinating search using inspector and strategy
- Concrete implementations for Core AI AIProgram and PyTorch ExportedProgram
"""

from collections.abc import Callable
from dataclasses import dataclass, field
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
from .utils import _with_debug

TNode = TypeVar("TNode")
TGraph = TypeVar("TGraph")


class Validator(Generic[TNode, TGraph]):
    """
    Generic validator with pluggable search strategy.

    This class coordinates the search for problematic operations by combining
    a ComputationGraph, Inspector, and SearchStrategy. Works with any graph
    representation and inspector implementation.
    """

    @dataclass
    class Result:
        """
        Result of validation check.

        Contains lists of failed and unknown operations, sorted by topological order.
        The nodes are the original framework-specific nodes (e.g., Core AI operations,
        PyTorch FX nodes) from the computation graph.
        """

        failed_nodes: list[Any]
        """Operations that failed validation, sorted by topological (execution) order."""

        unknown_nodes: list[Any]
        """Operations whose values could not be retrieved, sorted by topological order.

        Only genuine retrieval failures. Graph nodes that are not operations are reported
        separately in :attr:`excluded_nodes`, because reporting them here says a value was
        expected and missing when none was ever computed."""

        excluded_nodes: list[Any] = field(default_factory=list)
        """Graph nodes that are not operations: placeholders (inputs and weights),
        `get_attr`, and the `output` node.

        Split from `unknown_nodes` for the same reason the comparator separates them: on a
        clean model these are the whole unknown count, so reporting them there reads as
        failed retrievals when nothing went wrong. The `output` node is also not a failure
        worth naming in a NaN check: it forwards a value rather than producing one."""

    def __init__(
        self,
        graph: ComputationGraph[TNode, TGraph],
        inspector: Inspector,
        strategy: SearchStrategy[TNode, TGraph] | None = None,
        show_progress: bool = True,
    ):
        """
        Initialize the validator.

        Args:
            graph: Computation graph to validate
            inspector: Inspector implementation for retrieving intermediate values
            strategy: Search strategy to use. Defaults to
                :class:`~coreai_torch.debugging.search_strategy.ExhaustiveStrategy`,
                which checks every operation in one batch.

                Each batch a strategy yields costs a full model execution on *both*
                sides, so narrowing only pays when a capture costs more than a run.
                Level-order batches within a depth level, so over a chain -- one node per
                level -- every batch comes back size 1 whatever the batch size, and
                bisection costs an execution per node while reporting only the first
                divergence.

                Bisection remains the right choice when a capture is expensive: very
                large intermediates, or an early exit that skips most of the graph.
                Pass it explicitly for those.
            show_progress: Whether to show progress bar during validation (default: True)

        """
        self.graph = graph
        self.inspector = inspector
        self.strategy = strategy or ExhaustiveStrategy(graph=graph)
        self.show_progress = show_progress
        self._progress_bar: Any = None

    def _will_start_validation(self, total_ops: int) -> None:
        """
        Call before validation starts for custom progress tracking.

        Args:
            total_ops: Total number of operations in the graph

        """
        if self.show_progress:
            self._progress_bar = _ProgressBar(
                total=total_ops,
                description="Validating operations",
            )

    def _did_check_batch(
        self,
        batch_size: int,
        pass_count: int,
        fail_count: int,
        unknown_count: int,
    ) -> None:
        """
        Call after checking each batch for custom progress tracking.

        Args:
            batch_size: Number of operations checked in this batch
            pass_count: Total number of passed operations so far
            fail_count: Total number of failed operations so far
            unknown_count: Total number of unknown operations so far

        """
        if self._progress_bar:
            self._progress_bar.update(batch_size)
            self._progress_bar.set_postfix(
                {
                    "pass": pass_count,
                    "fail": fail_count,
                    "unknown": unknown_count,
                },
            )

    def _did_finish_validation(self) -> None:
        """Call after validation completes for custom cleanup."""
        if self._progress_bar:
            self._progress_bar.close()
            self._progress_bar = None

    @staticmethod
    def _is_not_an_operation(node: ComputationGraph.Node) -> bool:
        """Whether this graph node computes nothing.

        A placeholder (a model input or weight), a `get_attr`, or the `output` node. They
        carry values but perform no arithmetic, so a NaN check has nothing to say about
        them -- and the `output` node in particular would otherwise be reported as
        *producing* a NaN it merely forwards.

        Args:
            node: Graph node to test.

        Returns:
            True when the node performs no computation.

        """
        original = getattr(node, "original_node", None)
        return getattr(original, "op", None) in ("placeholder", "output", "get_attr")

    def _evaluate_node(
        self,
        outputs: list[NDArray[Any] | None] | None,
        check_fn: Callable[[list[NDArray[Any] | None]], bool],
    ) -> SearchStrategy.ValidationResult:
        """Evaluate a single node's outputs."""
        if outputs is None:
            return SearchStrategy.ValidationResult.UNKNOWN

        if check_fn(outputs):
            return SearchStrategy.ValidationResult.FAIL

        return SearchStrategy.ValidationResult.PASS

    def _sort_nodes_by_topo_order(
        self,
        nodes: list[ComputationGraph.Node],
        op_ids: list[int | str],
    ) -> list[Any]:
        """
        Sort nodes by topological order and return original nodes.

        Topological order is determined by the position in the op_ids list,
        which represents the execution order of operations in the graph.
        Returns the original framework-specific nodes (e.g., Core AI operations,
        PyTorch FX nodes) rather than ComputationGraph.Node objects.

        Args:
            nodes: List of nodes to sort
            op_ids: List of operation IDs in topological (execution) order

        Returns:
            Original nodes sorted by their position in op_ids (topological order)

        Raises:
            ValueError: If a node's op_id is not found in op_ids

        """
        # Build index map once for O(1) lookups instead of O(N) op_ids.index() calls
        index_map = {op_id: i for i, op_id in enumerate(op_ids)}

        # Map each node to its index in topological order
        nodes_with_idx: list[tuple[int, ComputationGraph.Node]] = [
            (index_map[node.op_id], node) for node in nodes
        ]
        # Sort by index
        nodes_with_idx.sort(key=lambda x: x[0])
        # Return original nodes
        return [node.original_node for _, node in nodes_with_idx]

    async def check(
        self,
        check_fn: Callable[[list[NDArray[Any] | None]], bool],
        inputs: Any,
    ) -> Result:
        """
        Find operations where check_fn returns True.

        Args:
            check_fn: Function that returns True if outputs have issues
            inputs: Model inputs to use during execution

        Returns:
            ValidationResult containing failed and unknown operations sorted by topological order.
            The nodes are the original framework-specific nodes (not _ComputationGraph.Node).

        """
        op_ids = self.graph.get_op_ids()
        total_ops = len(op_ids)

        # Initialize progress tracking
        pass_count = 0
        excluded: list[ComputationGraph.Node] = []
        fail_count = 0
        unknown_count = 0

        # Notify start of validation
        self._will_start_validation(total_ops)

        try:
            async for batch in self.strategy:
                # Fetch intermediates for entire batch at once
                batch_op_ids = [node.op_id for node in batch]
                results = await self.inspector.get_intermediates_for_ops(
                    batch_op_ids,
                    inputs,
                )

                # Check each node in the batch and collect results
                batch_results = []
                for node in batch:
                    if self._is_not_an_operation(node):
                        # Not a computation, so there is no value to validate. Recorded
                        # apart from the unknowns rather than counted as a failed retrieval.
                        excluded.append(node)
                        batch_results.append(
                            (node, SearchStrategy.ValidationResult.SKIPPED),
                        )
                        continue

                    outputs = results.get(node.op_id)
                    result = self._evaluate_node(outputs, check_fn)
                    batch_results.append((node, result))

                    # Update counts
                    if result == SearchStrategy.ValidationResult.PASS:
                        pass_count += 1
                    elif result == SearchStrategy.ValidationResult.FAIL:
                        fail_count += 1
                    else:
                        unknown_count += 1

                # Update strategy with results
                await self.strategy.update(batch_results)

                # Notify batch completion
                self._did_check_batch(len(batch), pass_count, fail_count, unknown_count)

            # After search completes, get problematic operations
            failed_nodes = self.strategy.get_problematic_operations()
            unknown_nodes: list[ComputationGraph.Node] = []
            unknown_nodes = [
                node
                for node in self.strategy.get_unknown_operations()
                if not self._is_not_an_operation(node)
            ]

            # Sort both lists by topological (execution) order and get original nodes
            sorted_failed = (
                self._sort_nodes_by_topo_order(failed_nodes, op_ids)
                if failed_nodes
                else []
            )
            sorted_unknown = (
                self._sort_nodes_by_topo_order(unknown_nodes, op_ids)
                if unknown_nodes
                else []
            )

            return Validator.Result(
                failed_nodes=sorted_failed,
                unknown_nodes=sorted_unknown,
                excluded_nodes=(
                    self._sort_nodes_by_topo_order(excluded, op_ids) if excluded else []
                ),
            )
        finally:
            # Notify validation completion
            self._did_finish_validation()

    async def check_for_nans(
        self,
        inputs: Any,
    ) -> Result:
        """
        Find operations that produce NaN values.

        Args:
            inputs: Model inputs to use during execution

        Returns:
            ValidationResult containing operations that produced NaN and unknown operations,
            sorted by topological order. The nodes are the original framework-specific nodes.

        """
        return await self.check(
            lambda outputs: any(
                np.isnan(arr).any()
                if arr is not None and np.issubdtype(arr.dtype, np.floating)
                else False
                for arr in outputs
            ),
            inputs,
        )

    async def check_for_infs(
        self,
        inputs: Any,
    ) -> Result:
        """
        Find operations that produce infinite values.

        Args:
            inputs: Model inputs to use during execution

        Returns:
            ValidationResult containing operations that produced inf and unknown operations,
            sorted by topological order. The nodes are the original framework-specific nodes.

        """
        return await self.check(
            lambda outputs: any(
                np.isinf(arr).any()
                if arr is not None and np.issubdtype(arr.dtype, np.floating)
                else False
                for arr in outputs
            ),
            inputs,
        )


async def create_validator_for_coreai_program(
    program: AIProgram,
    entry_point: str,
    strategy: SearchStrategy[Operation, Module] | None = None,
    use_caching: bool = True,
    specialization_options: SpecializationOptions | None = None,
) -> Validator[Operation, Module]:
    """
    Create a validator for an AIProgram.

    This function creates an executable from the AIProgram and sets up
    a validator to debug it. The executable is created in a temporary directory.

    Args:
        program: AIProgram to validate
        entry_point: Name of the coreai.graph
        inspector_type: Type of inspector.
        strategy: Search strategy to use. Defaults to `ExhaustiveStrategy`, which
            checks every operation in one batch and reports every divergence; see
            :class:`Validator` for why that is the default rather than bisection.
        use_caching: Whether to wrap inspector with _CachingInspector (default: True)
        specialization_options: Options for configuring model specialization

    Returns:
        Validator instance configured for the AIProgram

    """
    # Create a temporary directory for the executable asset
    temp_dir = TemporaryDirectory()
    asset_path = Path(temp_dir.name) / "model.aimodel"

    # Create asset from AIProgram and load model from asset
    asset = program.save_asset(asset_path)
    specialization_options = _with_debug(specialization_options)
    model = await AIModel.load(asset.path, specialization_options)

    inspector = CoreAIInspector(
        model=model,
        function_name=entry_point,
        temp_dir=temp_dir,
    )

    if use_caching:
        inspector = CachingInspector(inspector)

    graph = create_graph_from_coreai_program(
        module=program._mlir_module,
        entry_point=entry_point,
    )

    return Validator(
        graph=graph,
        inspector=inspector,
        strategy=strategy,
    )


def _create_validator_for_exported_program(
    program: torch.export.ExportedProgram,
    strategy: SearchStrategy[torch.fx.Node, torch.fx.Graph] | None = None,
    use_caching: bool = True,
) -> Validator[torch.fx.Node, torch.fx.Graph]:
    """
    Create a validator for a PyTorch ExportedProgram (internal version with strategy).

    Args:
        program: PyTorch ExportedProgram to validate
        strategy: Search strategy to use. Defaults to `ExhaustiveStrategy`, which
            checks every operation in one batch and reports every divergence; see
            :class:`Validator` for why that is the default rather than bisection.
        use_caching: Whether to wrap inspector with _CachingInspector (default: True)

    Returns:
        Validator instance configured for the ExportedProgram

    """
    # Create inspector
    base_inspector: Inspector = TorchFXInspector(exported_program=program)

    # Wrap with caching if requested
    inspector: Inspector = base_inspector
    if use_caching:
        inspector = CachingInspector(base_inspector)

    # Create graph from exported program
    graph = create_graph_from_exported_program(program=program)

    return Validator(
        graph=graph,
        inspector=inspector,
        strategy=strategy,
    )


def create_validator_for_exported_program(
    program: torch.export.ExportedProgram,
    strategy: SearchStrategy[torch.fx.Node, torch.fx.Graph] | None = None,
    use_caching: bool = True,
) -> Validator[torch.fx.Node, torch.fx.Graph]:
    """
    Create a validator for a PyTorch ExportedProgram.

    This function sets up a validator to debug a PyTorch ExportedProgram using
    TorchFX inspector with bisection search and caching.

    Args:
        program: PyTorch ExportedProgram to validate
        strategy: Search strategy to use. Defaults to `ExhaustiveStrategy`, which
            checks every operation in one batch and reports every divergence; see
            :class:`Validator` for why that is the default rather than bisection.
        use_caching: Whether to wrap inspector with CachingInspector (default: True)

    Returns:
        Validator instance configured for the ExportedProgram with:
        - Bisection search strategy (or custom if provided)
        - Caching inspector for performance (optional)

    """
    return _create_validator_for_exported_program(
        program=program,
        strategy=strategy,
        use_caching=use_caching,
    )
