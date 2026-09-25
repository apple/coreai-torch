# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Union, cast

import coreai
import numpy as np
import torch
from numpy.typing import NDArray

from .debug_info import (
    DebugInfo,
    DebugInfoRecord,
    _build_compile_id_to_coreai_map,
    parse_debug_infos,
)
from .torch_utils import _TorchFXNodeValueInterpreter

logger = logging.getLogger(__name__)


def _uses_os_framework() -> bool:
    """
    Whether the OS Core AI framework is selected, rather than the bundled one.

    Read from the environment because the choice is made there, before anything in
    this package is imported. A default OS-framework install that does not set the
    variable reads as False here.
    """
    return "USE_OS_COREAI" in os.environ


async def _wait_for_async_callbacks() -> None:
    """
    Wait for asynchronously-invoked intermediate capture callbacks to complete.

    Only the OS framework invokes them asynchronously; against the bundled runtime
    they have run by the time inference returns, so the wait is pure latency there.

    TODO: this sleep is a workaround. There is no signal that the callbacks have
    finished, so the duration is a guess -- too short silently drops captures, and
    the only symptom is an operation reporting no value.

    Gating this on whether tests are running rather than on the framework would skip it
    in exactly the case that needs it, and the tests would then exercise something other
    than what runs in production.
    """
    if not _uses_os_framework():
        return
    await asyncio.sleep(5.0)


@dataclass(frozen=True)
class _MappingKey:
    """
    Key for mapping ODIX outputs to source outputs.
    """

    odix_id: int
    delegate_id: int | None
    output_idx: int


@dataclass(frozen=True)
class _CompiledIdMappings:
    """Result of building compiled ID mappings."""

    target_to_source_output_map: dict[_MappingKey, tuple[int, int]]
    all_compiled_ids: list[tuple[int, int | None]]


# Resolve a source op id to an odix id through `_build_delegate_to_odix_map` below, which
# identifies the delegate call positively. Selecting the highest matching odix id instead
# lands on an arbitrary sibling of the call, and asking the runtime for that returns
# nothing. `debug_info` exports a similarly-named helper that reads fallback records; it
# answers a different question and is not a substitute.


def _build_delegate_to_odix_map(
    debug_info_records: list[DebugInfoRecord],
    source_level: str,
) -> dict[tuple[int, str], int]:
    """
    Map each delegate op ID to the odix instruction that calls that delegate.

    This is what the runtime needs to be asked for. Every odix instruction belonging
    to one delegate region carries the *same* source op IDs -- for a small model that
    is a dozen entries with an identical list -- and only the call among them produces
    an output. Resolving a source ID to an odix ID by taking the highest matching
    ``odix_id`` therefore lands on an arbitrary sibling (a ``load_imm``, a
    ``set_context``), and asking the runtime for that instruction returns nothing at
    all: the callback is never invoked and every intermediate comes back ``None``. It
    only appears to work when the call happens to be the highest-numbered instruction
    of its region, which varies between compilations.

    The call is identified positively rather than by position: it carries a
    ``delegate`` symbol *and* an output mapping. Which delegate region a call belongs
    to is then decided by the source IDs it covers, so a model with several delegate
    regions maps each delegate to its own call.

    The key includes the record identifier because a delegate op ID is only unique
    within its own record -- two delegates can each number an operation ``2``.

    Args:
        debug_info_records: Parsed debug information containing operation mappings.
        source_level: Dialect level to extract source op IDs from (e.g. ``"coreai"``).

    Returns:
        Dictionary mapping ``(delegate_op_id, record_identifier)`` to the calling
        odix ID.

    """
    # Delegate calls, as (odix_id, the source ids that region covers).
    delegate_calls: list[tuple[int, frozenset[int]]] = []
    for record in debug_info_records:
        if not record.identifier.startswith("odix"):
            continue
        for op in record.operations:
            if op.get_symbol_name(DebugInfo.SymbolType.DELEGATE) is None:
                continue
            if not op.get_output_mappings(source_level=source_level):
                continue
            delegate_calls.append((op.odix_id, frozenset(op.get_op_ids(source_level))))

    delegate_to_odix: dict[tuple[int, str], int] = {}
    for record in debug_info_records:
        if record.identifier.startswith("odix"):
            continue
        for op in record.operations:
            operation_ids = frozenset(op.get_op_ids(source_level))
            for mapping in op.get_output_mappings(source_level=source_level):
                # Each mapping is matched on the operation's own ids plus its own source
                # id, and nothing carried over from the mappings before it. Accumulating
                # into one set makes the delegate a mapping resolves to depend on the
                # order they are iterated: a later mapping can match a region only
                # because an earlier one contributed the id that overlapped it.
                source_ids = operation_ids | {mapping.source_op_id}
                # The delegate's own op id is the mapping target for these records.
                for odix_id, covered in delegate_calls:
                    if source_ids & covered:
                        delegate_to_odix[(mapping.target_op_id, record.identifier)] = (
                            odix_id
                        )
                        break
    return delegate_to_odix


class MissingValue:
    """Why an operation has no value, and what a caller can conclude from it.

    A namespace rather than a bare pair of module-level types, matching `Comparator.Status`
    and `SearchStrategy.ValidationResult` elsewhere: `MissingValue.Reason` and
    `MissingValue.Explanation` read as one concept at the call site.
    """

    class Reason(Enum):
        """Why no value came back for an operation.

        A category on its own -- "no value" -- does not say whether that is expected. These
        do, and they are separate members because the two call for opposite responses.
        """

        MERGED_INTO_ANOTHER_OPERATION = auto()
        """This operation's work happens inside another operation, which does have a value.

        The value is not separately observable, but it is not unaccounted for: the
        containing operation is named in `observable_op_id`, so comparing *that* covers
        this one's arithmetic. Expected -- a covered gap rather than an open one.

        Named for what a caller can conclude rather than for the transform that caused it.
        On Core AI this is a fusion, but the same situation arises from any lowering that
        computes several source operations in one step."""

        NO_PRODUCER_FOUND = auto()
        """Nothing in the compiled program produces this value.

        Not merged into anything, so it was removed or lost: eliminated during lowering, or
        its output mapping is missing. Unlike the above, no other operation's comparison
        covers it, so this is an open gap and worth investigating."""

    @dataclass(frozen=True)
    class Explanation:
        """Why one operation has no value, in a form both code and people can read."""

        op_id: int
        """The operation whose value is missing."""

        reason: "MissingValue.Reason"
        """Machine-readable cause; switch on this rather than parsing `describe()`."""

        merged_with: tuple[int, ...] = ()
        """Every operation computed together with this one, including itself. Empty when
        nothing produces the value."""

        observable_op_id: int | None = None
        """The one of `merged_with` whose value *is* available -- the operation to compare
        in order to cover this one. None when nothing produces the value."""

        def to_dict(self) -> dict[str, Any]:
            """
            Return the explanation as plain values.

            `describe()` is included: it is the sentence a reader acts on, and
            rebuilding it from `reason` means reimplementing the rules here.

            Returns:
                The cause, machine-readable and as prose.

            """
            return {
                "op_id": self.op_id,
                "reason": self.reason.value,
                "merged_with": list(self.merged_with),
                "observable_op_id": self.observable_op_id,
                "describe": self.describe(),
            }

        def describe(self) -> str:
            """A one-line explanation for a human reader."""
            if self.reason is MissingValue.Reason.MERGED_INTO_ANOTHER_OPERATION:
                others = [i for i in self.merged_with if i != self.op_id]
                return (
                    f"coreai {self.op_id}: computed together with {others}; the observable "
                    f"value belongs to coreai {self.observable_op_id}, so comparing that "
                    "operation covers this one"
                )
            return (
                f"coreai {self.op_id}: nothing in the compiled program produces this "
                "value -- removed during lowering, or its output mapping is missing"
            )


def _build_compile_identifiers_map(
    debug_info_records: list[DebugInfoRecord],
    source_level: str,
) -> dict[tuple[int, int], _MappingKey]:
    """
    Build a mapping from source output to compiled identifiers.

    Maps each ``(source_op_id, source_output_idx)`` to a
    ``_MappingKey(odix_id, delegate_id, output_idx)`` by extracting
    output mappings from every debug info record at the given
    *source_level*.  When duplicates target the same source output, the
    highest ``target_op_id`` wins.  For odix records
    (``identifier.startswith("odix")``), ``target_op_id`` is the
    ``odix_id``; for all other records it is the ``delegate_id``, and the
    calling odix instruction is resolved via
    :func:`_build_delegate_to_odix_map`.

    Args:
        debug_info_records: Parsed debug information containing
            operation mappings.
        source_level: Dialect level to extract op IDs from
            (e.g., ``"coreai"``). Defaults to ``"coreai"``.

    Returns:
        Dictionary mapping ``(source_op_id, source_output_idx)`` to
        ``_MappingKey``.

    """
    delegate_to_odix_map = _build_delegate_to_odix_map(debug_info_records, source_level)
    result: dict[tuple[int, int], _MappingKey] = {}

    for record in debug_info_records:
        is_odix = record.identifier.startswith("odix")

        for op in record.operations:
            for mapping in op.get_output_mappings(source_level=source_level):
                # For delegate records, the odix id is the instruction that calls
                # this delegate -- not whichever instruction of the region happens
                # to carry the same source ids and the highest odix id, which is
                # usually not the one that produces an output.
                if not is_odix:
                    odix_id = delegate_to_odix_map.get(
                        (mapping.target_op_id, record.identifier)
                    )
                    if odix_id is None:
                        continue

                source_key = (mapping.source_op_id, mapping.source_output)
                existing = result.get(source_key)

                # Compare against the relevant ID from the existing entry:
                # odix_id for odix records, delegate_id for delegate records.
                existing_op_id = (
                    (existing.odix_id if is_odix else existing.delegate_id)
                    if existing is not None
                    else None
                )

                # Only update if new or has a higher target_op_id
                if existing_op_id is None or mapping.target_op_id > existing_op_id:
                    if is_odix:
                        new_entry = _MappingKey(
                            odix_id=op.odix_id,
                            delegate_id=None,
                            output_idx=mapping.target_output,
                        )
                    else:
                        new_entry = _MappingKey(
                            odix_id=odix_id,
                            delegate_id=mapping.target_op_id,
                            output_idx=mapping.target_output,
                        )
                    result[source_key] = new_entry

                    if existing is not None:
                        logger.debug(
                            "  %s.%d[%d] -> %s.%d[%d] (replaced %d)",
                            source_level,
                            mapping.source_op_id,
                            mapping.source_output,
                            record.identifier,
                            mapping.target_op_id,
                            mapping.target_output,
                            existing_op_id,
                        )
                    else:
                        logger.debug(
                            "  %s.%d[%d] -> %s.%d[%d]",
                            source_level,
                            mapping.source_op_id,
                            mapping.source_output,
                            record.identifier,
                            mapping.target_op_id,
                            mapping.target_output,
                        )

    return result


def _create_operation_mappings(
    op_ids: Sequence[int],
    compile_map: dict[tuple[int, int], _MappingKey],
) -> _CompiledIdMappings:
    """
    Create reverse mappings from compiled identifiers back to source outputs.

    Args:
        op_ids: Source operation IDs to include.
        compile_map: Pre-built map from
            ``_build_compile_identifiers_map``.

    Returns:
        ``_CompiledIdMappings`` with the reverse map and a list of
        ``(odix_id, delegate_id)`` pairs for all matched operations.

    """
    requested = set(op_ids)
    target_to_source_output_map: dict[_MappingKey, tuple[int, int]] = {}
    all_compiled_ids: list[tuple[int, int | None]] = []

    for (source_op_id, source_output_idx), mapping_key in compile_map.items():
        if source_op_id not in requested:
            continue
        all_compiled_ids.append((mapping_key.odix_id, mapping_key.delegate_id))
        # Keep the terminal. Assigning unconditionally let the last operation iterated
        # win, which is dictionary order and so neither stable nor meaningful: for one
        # model it named the bias add, for another the reshape beside it.
        existing = target_to_source_output_map.get(mapping_key)
        if existing is None or source_op_id > existing[0]:
            target_to_source_output_map[mapping_key] = (
                source_op_id,
                source_output_idx,
            )

    return _CompiledIdMappings(target_to_source_output_map, all_compiled_ids)


def _convert_to_dict(
    op_ids: Sequence[int],
    results: Mapping[int, dict[int, NDArray[Any]]],
) -> dict[int, list[NDArray[Any] | None] | None]:
    """
    Convert output dictionary structure to list format.

    Transforms the internal sparse dictionary representation (keyed by output index)
    into a dense list format expected by the public API. Missing indices are filled
    with None values.

    Args:
        op_ids: List of operation IDs to process
        results: Mapping of operation_id -> {output_index -> array}

    Returns:
        Dictionary mapping operation_id -> list of arrays (or None if operation not found).
        Each list contains arrays ordered by output index, with None for missing outputs.

    """
    final_results: dict[int, list[NDArray[Any] | None] | None] = {}
    for op_id in op_ids:
        output_dict = results.get(op_id)
        if output_dict:
            max_idx = max(output_dict.keys())
            output_list: list[NDArray[Any] | None] = [
                output_dict.get(i, None) for i in range(max_idx + 1)
            ]
            final_results[op_id] = output_list
        else:
            final_results[op_id] = None

    return final_results


class IntermediateKind(Enum):
    """
    Type of intermediate value captured during model execution.

    Attributes:
        INPUT: Represents an operation input value
        OUTPUT: Represents an operation output value
        UNKNOWN: Fallback for unrecognized intermediate types

    """

    INPUT = "input"
    OUTPUT = "output"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "IntermediateKind":
        """Return UNKNOWN for unrecognized values to avoid lookup errors."""
        return cls.UNKNOWN


class Inspector(ABC):
    """
    Abstract base class for capturing intermediate operation values during model execution.

    Inspectors provide a unified interface for executing models and capturing outputs
    from specific operations. This is essential for debugging workflows such as
    bisection search, where intermediate values are compared to identify numerical issues.

    Different implementations handle various model formats (PyTorch, compiled Core AI, etc.).
    """

    # Type alias for operation identifiers (string for PyTorch FX nodes, int for Core AI debug IDs)
    OpID = Union[str, int]

    @abstractmethod
    async def get_intermediates_for_ops(
        self,
        op_ids: list[OpID],
        inputs: Any,
    ) -> dict[OpID, list[NDArray[Any] | None] | None]:
        """
        Execute model and capture intermediate outputs for specified operations.

        Args:
            op_ids: List of operation identifiers to capture outputs for
            inputs: Model inputs (format varies by implementation)

        Returns:
            Dictionary mapping each operation ID to a list of output arrays.
            None indicates the operation wasn't executed or produced no outputs.

        """

    @classmethod
    def convert_to_numpy(cls, arr: Any) -> NDArray[Any]:
        """
        Convert an array to numpy.ndarray format.

        This method allows different inspector implementations to handle their
        framework-specific array types (e.g., Core AI NDArray).

        Args:
            arr: Array to convert (framework-specific or numpy array)

        Returns:
            NumPy array

        """
        # Default implementation: assume it's already numpy or numpy-compatible
        return np.asarray(arr)


class CachingInspector(Inspector):
    """
    Inspector decorator that caches intermediate values to avoid redundant execution.

    Maintains a cache of previously captured intermediate values, significantly
    improving performance when querying the same operations multiple times (common
    in bisection search). The cache is automatically invalidated when inputs change.

    This is a transparent wrapper - it preserves the same interface as the underlying
    inspector while adding caching behavior. When max_cache_size is set, uses LRU
    (Least Recently Used) eviction policy.
    """

    def __init__(self, inspector: Inspector, max_cache_size: int | None = None):
        """
        Initialize the caching inspector.

        Args:
            inspector: Underlying inspector instance to wrap with caching
            max_cache_size: Maximum number of entries to keep in cache. If None, cache
                          size is unlimited. When limit is reached, least recently used
                          entries are evicted (LRU policy).

        """
        self._inspector = inspector
        self._cache: OrderedDict[Inspector.OpID, list[NDArray[Any] | None] | None] = (
            OrderedDict()
        )
        self._current_inputs: Any = None
        self._max_cache_size = max_cache_size

    @staticmethod
    def _inputs_equal(inputs1: Any, inputs2: Any) -> bool:
        """
        Compare inputs for cache invalidation, handling containers properly.

        For tuple/list/dict structures, compares contents rather than object identity.
        For tensors and other objects, falls back to object identity comparison.

        Args:
            inputs1: First input to compare
            inputs2: Second input to compare

        Returns:
            True if inputs should be considered equal for caching purposes
        """
        if inputs1 is inputs2:
            return True

        if type(inputs1) is not type(inputs2):
            return False

        if isinstance(inputs1, dict):
            if inputs1.keys() != inputs2.keys():
                return False
            return all(
                CachingInspector._inputs_equal(inputs1[k], inputs2[k])
                for k in inputs1.keys()
            )
        elif isinstance(inputs1, (tuple, list)):
            if len(inputs1) != len(inputs2):
                return False
            return all(
                CachingInspector._inputs_equal(a, b) for a, b in zip(inputs1, inputs2)
            )
        else:
            # For tensors and other objects, use object identity
            return inputs1 is inputs2

    @staticmethod
    def _copy_inputs(inputs: Any) -> Any:
        """
        Create a copy of inputs for cache tracking.

        For tuple/list/dict structures, creates shallow copies.
        For other objects (like tensors), stores reference since deep copying
        tensors can be expensive and we only need to track identity changes.

        Args:
            inputs: Input to copy

        Returns:
            Copy of the input suitable for cache tracking
        """
        if isinstance(inputs, dict):
            return {k: CachingInspector._copy_inputs(v) for k, v in inputs.items()}
        elif isinstance(inputs, tuple):
            return tuple(CachingInspector._copy_inputs(item) for item in inputs)
        elif isinstance(inputs, list):
            return [CachingInspector._copy_inputs(item) for item in inputs]
        else:
            # For tensors and other objects, just return the reference
            # We rely on identity comparison for these
            return inputs

    async def get_intermediates_for_ops(
        self,
        op_ids: list[Inspector.OpID],
        inputs: Any,
    ) -> dict[Inspector.OpID, list[NDArray[Any] | None] | None]:
        """
        Retrieve intermediate outputs with automatic caching.

        Returns cached values when available, only executing the model for operations
        not yet in the cache. Automatically clears the cache when inputs change.
        Implements LRU eviction when max_cache_size is set.

        Args:
            op_ids: List of operation identifiers to capture outputs for
            inputs: Model inputs (cache is invalidated if these change)

        Returns:
            Dictionary mapping operation IDs to output arrays (cached or freshly captured)

        """
        if not self._inputs_equal(self._current_inputs, inputs):
            self._cache.clear()
            self._current_inputs = self._copy_inputs(inputs)

        uncached_ops = [op_id for op_id in op_ids if op_id not in self._cache]

        if uncached_ops:
            results = await self._inspector.get_intermediates_for_ops(
                uncached_ops,
                inputs,
            )
            if results is not None:
                for op_id, value in results.items():
                    if (
                        self._max_cache_size is not None
                        and len(self._cache) >= self._max_cache_size
                        and op_id not in self._cache
                    ):
                        self._cache.popitem(last=False)
                    self._cache[op_id] = value

        result = {}
        for op_id in op_ids:
            if op_id in self._cache:
                self._cache.move_to_end(op_id)
                result[op_id] = self._cache[op_id]
            else:
                result[op_id] = None
        return result

    def clear_cache(self) -> None:
        """Clear all cached intermediate values and reset input tracking."""
        self._cache.clear()
        self._current_inputs = None


class TorchFXInspector(Inspector):
    """
    Inspector for PyTorch ExportedProgram models.

    Executes a PyTorch ExportedProgram and captures intermediate values at specified
    FX graph nodes using a custom interpreter. This is used for debugging PyTorch
    models before compilation.

    The inspector works at the FX graph level, where operation IDs are node names.
    """

    def __init__(self, exported_program: torch.export.ExportedProgram):
        """
        Initialize the PyTorch FX inspector.

        Args:
            exported_program: PyTorch ExportedProgram to execute and inspect

        """
        self.exported_program = exported_program

    async def get_intermediates_for_ops(
        self,
        op_ids: list[Inspector.OpID],
        inputs: Any,
    ) -> dict[Inspector.OpID, list[NDArray[Any] | None] | None]:
        """
        Capture intermediate values at specified FX graph nodes.

        Executes the PyTorch model using a custom FX interpreter that captures
        intermediate values via callbacks. All outputs are converted to NumPy arrays.

        Args:
            op_ids: List of FX node names (operation IDs) to capture
            inputs: Tuple of input tensors matching the model's expected signature

        Returns:
            Dictionary mapping node names to lists of output arrays (as NumPy).
            None indicates a node wasn't executed or produced no outputs.

        """
        requested_nodes = set(op_ids)
        results: dict[Inspector.OpID, list[NDArray[Any] | None] | None] = {}

        def capture_callback(node: torch.fx.Node, result: Any) -> None:
            """Invoke callback for each node during interpretation."""
            if node.name in requested_nodes:
                if isinstance(result, (tuple, list)):
                    results[node.name] = [
                        self.__class__.convert_to_numpy(r) for r in result
                    ]
                else:
                    results[node.name] = [self.__class__.convert_to_numpy(result)]

        interpreter = _TorchFXNodeValueInterpreter(
            self.exported_program.module(),
            callback=capture_callback,
        )

        # Accept the same shape of inputs as CoreAIInspector, which requires a
        # name -> array dict. Without this the two inspectors in a Comparator need
        # different input objects for the same comparison -- and a dict handed to the FX
        # interpreter arrives as a single positional argument, failing deep inside the
        # traced graph with an error that names neither inputs nor this class.
        def _as_tensor(value: Any) -> Any:
            """Adapt one input to what an FX graph needs: a torch tensor."""
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            return value

        if isinstance(inputs, dict):
            positional = [_as_tensor(value) for value in inputs.values()]
        elif isinstance(inputs, (tuple, list)):
            positional = [_as_tensor(value) for value in inputs]
        else:
            positional = [_as_tensor(inputs)]
        interpreter.run(*positional)

        for node_name in op_ids:
            if node_name not in results:
                results[node_name] = None

        return results

    @classmethod
    def convert_to_numpy(cls, value: Any) -> NDArray[Any]:
        """
        Convert PyTorch tensors or other values to NumPy arrays.

        Args:
            value: Value to convert (torch.Tensor, np.ndarray, or scalar)

        Returns:
            NumPy array (detached from PyTorch computation graph if applicable)

        """
        # Import torch here to avoid requiring it when not using PyTorch models
        import torch  # noqa: PLC0415

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)


class CoreAIInspector(Inspector):
    """
    Inspector for Core AI Runtime deployed models.

    Captures intermediate operation outputs from models deployed via Core AI Runtime.
    Uses the IntermediateLogger to intercept values during execution and map them
    back to source-level operations using debug info.

    Operation IDs are integer debug IDs from the source dialect (e.g., PyTorch operation IDs).
    """

    IntermediateCallback = Callable[
        [
            list[coreai.runtime._NDArray | None],
            str,
            coreai.runtime.CompileIdentifiers,
        ],
        None,
    ]

    def __init__(
        self,
        model: coreai.runtime.AIModel,
        function_name: str = "main",
        temp_dir: Any = None,
    ):
        """
        Initialize the Core AI Runtime inspector.

        Args:
            model: Loaded AIModel instance to execute and inspect
            function_name: Inference function name to execute (default: "main")
            temp_dir: Optional TemporaryDirectory to keep alive for the model's lifetime

        """
        self._model = model
        self._function_name = function_name
        self._last_outputs: dict[str, NDArray[Any]] | None = None
        self._temp_dir = temp_dir  # Keep temp directory alive

        # Load debug info from model
        debug_infos_bytes = model._debug_infos
        self._debug_info_records = parse_debug_infos(debug_infos_bytes)

        self._source_level = "coreai"
        self._compile_map = _build_compile_identifiers_map(
            self._debug_info_records,
            self._source_level,
        )

    def explain_missing(self, op_id: int) -> MissingValue.Explanation:
        """Say why no value came back for *op_id*.

        Distinguishes an operation whose work happens inside another -- covered, because
        comparing that one covers this -- from one nothing produces, which is an open gap.

        Args:
            op_id: The source operation whose value is missing.

        Returns:
            The explanation, with a machine-readable `reason`.

        """
        for members in self._coreai_ids_by_compile_id().values():
            if op_id not in members:
                continue
            # A group of one is not a merge. Every operation belongs to some compiled
            # dispatch, so membership alone says nothing: treating it as evidence names
            # an operation as merged into itself, with an empty list of partners.
            if len(members) < 2:
                break
            # The terminal of a group is the operation whose value the kernel computes, so
            # it is not merged *into* anything -- the others are merged into it. If its
            # value is missing the cause is elsewhere, and claiming a merge would send a
            # reader looking for a containing operation that does not exist.
            observable = max(members)
            if op_id == observable:
                break
            return MissingValue.Explanation(
                op_id=op_id,
                reason=MissingValue.Reason.MERGED_INTO_ANOTHER_OPERATION,
                merged_with=tuple(members),
                # The highest id is the terminal because operation ids are assigned in
                # dataflow order, so a consumer always outranks the values it consumes.
                # That was an unverified assumption when this was written -- and false for
                # any FX node expanding into a chain, which was numbered result-first --
                # until the numbering was fixed to follow IR order.
                observable_op_id=observable,
            )
        return MissingValue.Explanation(
            op_id=op_id,
            reason=MissingValue.Reason.NO_PRODUCER_FOUND,
        )

    def _coreai_ids_by_compile_id(self) -> dict[tuple[int, int | None], list[int]]:
        """Which source operations share each compiled dispatch.

        Built from the debug info records rather than by inverting `_compile_map`. That map
        is keyed by source operations that *have* an output mapping, and an operation
        absorbed into a fusion has none -- which is exactly why its value is missing. So
        inverting it can never find the ops this needs to explain: it reported every
        absorbed operation as belonging to no kernel. The records carry the
        absorbed ids through each fused location's `fusedWith` list.
        """
        return _build_compile_id_to_coreai_map(self._debug_info_records, "coreai")

    def _build_mapping_and_compile_ids(
        self,
        op_ids: Sequence[int],
    ) -> tuple[
        dict[_MappingKey, tuple[int, int]],
        list[coreai.runtime.CompileIdentifiers],
    ]:
        """
        Build output mappings and compile identifiers for requested source operations.

        Creates the mapping infrastructure needed to translate captured compiled operation
        outputs back to their source operation equivalents.

        Args:
            op_ids: List of source operation IDs to create mappings for

        Returns:
            Tuple of (output_mapping, compile_identifiers) where:
            - output_mapping: Maps compiled outputs back to source outputs
            - compile_identifiers: List of unique compiled operation IDs to capture

        """
        mappings = _create_operation_mappings(op_ids, self._compile_map)

        # Get unique compiled IDs preserving insertion order (dict.fromkeys for stable deduplication)
        unique_compiled_ids = dict.fromkeys(mappings.all_compiled_ids)
        all_compile_ids = [
            coreai.runtime.CompileIdentifiers(odix_id, delegate_id)
            for odix_id, delegate_id in unique_compiled_ids
        ]

        return mappings.target_to_source_output_map, all_compile_ids

    def _create_capture_callback(
        self,
        results: defaultdict[int, dict[int, NDArray[Any]]],
        odix_output_to_source_map: dict[_MappingKey, tuple[int, int]],
    ) -> "CoreAIInspector.IntermediateCallback":
        """
        Create callback function for Core AI Runtime IntermediateLogger.

        The callback is invoked during model execution for each intermediate value.
        It processes and stores the values in the results dictionary. Core AI provides
        all outputs for an operation at once (as a list).

        Args:
            results: Storage dictionary for captured values (source_op_id -> {output_index -> array})
            odix_output_to_source_map: Mapping from compiled to source operation outputs

        Returns:
            Callback function compatible with Core AI Runtime IntermediateLogger interface

        """

        def capture_callback(
            intermediates: list[coreai.runtime._NDArray | None],
            kind: str,
            compile_ids: coreai.runtime.CompileIdentifiers,
        ) -> None:
            kind_type = IntermediateKind(kind)

            if kind_type == IntermediateKind.UNKNOWN:
                msg = f"Unknown intermediate kind: {kind}"
                raise ValueError(msg)

            if kind_type != IntermediateKind.OUTPUT:
                return

            logger.debug(
                "Capturing %s for odix.%d (delegate=%s)",
                kind,
                compile_ids.id,
                compile_ids.delegate_id,
            )

            for odix_output_idx, intermediate in enumerate(intermediates):
                mapping_key = _MappingKey(
                    compile_ids.id,
                    compile_ids.delegate_id,
                    odix_output_idx,
                )

                if mapping_key not in odix_output_to_source_map:
                    logger.warning(
                        "  No source mapping found for odix.%d[%d]",
                        compile_ids.id,
                        odix_output_idx,
                    )
                    continue

                source_op_id, source_output_idx = odix_output_to_source_map[mapping_key]

                if source_output_idx in results[source_op_id]:
                    msg = f"Multiple compile_ids map to the same source operation output: source_op_id={source_op_id}, source_output_idx={source_output_idx}"
                    raise ValueError(msg)

                if intermediate is None:
                    logger.warning(
                        "  Intermediate is None for odix.%d[%d] -> source.%d[%d]",
                        compile_ids.id,
                        odix_output_idx,
                        source_op_id,
                        source_output_idx,
                    )
                    continue
                # Convert _NDArray (internal Core AI runtime type) to NDArray wrapper then to numpy
                ndarray = coreai.runtime._ndarray.NDArray._wrap(intermediate)
                results[source_op_id][source_output_idx] = (
                    self.__class__.convert_to_numpy(ndarray)
                )
                logger.debug(
                    "  odix.%d[%d] -> source.%d[%d] shape=%s",
                    compile_ids.id,
                    odix_output_idx,
                    source_op_id,
                    source_output_idx,
                    ndarray.numpy().shape,
                )

        return capture_callback

    async def get_intermediates_for_ops(
        self,
        op_ids: list[Inspector.OpID],
        inputs: Any,
    ) -> dict[Inspector.OpID, list[NDArray[Any] | None] | None]:
        """
        Capture intermediate outputs from Core AI Runtime model execution.

        Executes the model with an IntermediateLogger that captures values at
        requested operations and maps them back to source-level operations.

        Args:
            op_ids: List of source operation debug IDs to capture
            inputs: Model inputs (dictionary mapping input names to NDArray or numpy arrays)

        Returns:
            Dictionary mapping operation IDs to lists of output arrays.
            None indicates an operation wasn't executed or produced no outputs.

        Raises:
            TypeError: If inputs is not a dictionary

        """
        if not isinstance(inputs, dict):
            msg = "inputs must be a dictionary mapping input names to NDArray"
            raise TypeError(msg)

        # Convert inputs to NDArray objects if needed
        ndarray_inputs = {}
        for name, value in inputs.items():
            if not isinstance(value, coreai.runtime.NDArray):
                ndarray_inputs[name] = coreai.runtime.NDArray(value)
            else:
                ndarray_inputs[name] = value

        int_op_ids = cast(list[int], op_ids)

        logger.debug("Requested source op_ids: %s", int_op_ids)

        odix_output_to_source_map, all_compile_ids = (
            self._build_mapping_and_compile_ids(int_op_ids)
        )

        logger.debug("Found %d compile_ids to capture", len(all_compile_ids))
        for compile_id in all_compile_ids:
            logger.debug(
                "  - odix_id=%s, delegate_id=%s",
                compile_id.id,
                compile_id.delegate_id,
            )

        # If no compile IDs found, return None for all requested ops
        if not all_compile_ids:
            logger.warning("No compile_ids found for requested operations")
            return dict.fromkeys(op_ids)

        results: defaultdict[int, dict[int, NDArray[Any]]] = defaultdict(dict)

        capture_callback = self._create_capture_callback(
            results,
            odix_output_to_source_map,
        )

        intermediate_logger = coreai.runtime.IntermediateLogger(
            requested_intermediates=all_compile_ids,
            callback=capture_callback,
            is_enabled=True,
        )

        inference_function = self._model.load_function(
            self._function_name,
            intermediate_logger=intermediate_logger,
        )

        outputs = await inference_function(inputs=ndarray_inputs)
        await _wait_for_async_callbacks()

        self._last_outputs = {name: array.numpy() for name, array in outputs.items()}

        logger.debug("Successfully captured %d source operations", len(results))

        # int is a subtype of Inspector.OpID (int | str), so this is safe
        return _convert_to_dict(int_op_ids, results)  # type: ignore[return-value]

    @classmethod
    def convert_to_numpy(cls, arr: Any) -> NDArray[Any]:
        """
        Convert Core AI NDArray to numpy array.

        Args:
            arr: Core AI NDArray object

        Returns:
            NumPy array

        """
        return np.asarray(arr.numpy())

    @property
    def last_outputs(self) -> dict[str, NDArray[Any]] | None:
        """
        Get final model outputs from the most recent execution.

        Returns:
            Dictionary mapping output names to NumPy arrays, or None if model
            hasn't been executed yet

        """
        return self._last_outputs

    @staticmethod
    def get_compile_identifiers_for_op(
        source_level: str,
        source_op_id: int,
        debug_info_records: list[DebugInfoRecord],
    ) -> dict[int, coreai.runtime.CompileIdentifiers]:
        """
        Get compiled operation identifiers for a source operation.

        Maps each output of a source operation to its CompileIdentifiers
        (used by the Core AI Runtime).

        Args:
            source_level: Source dialect level (e.g., ``"coreai"``)
            source_op_id: Source operation ID to look up
            debug_info_records: Debug information containing operation mappings

        Returns:
            Dictionary mapping source output index to CompileIdentifiers

        """
        compile_map = _build_compile_identifiers_map(
            debug_info_records,
            source_level,
        )
        return {
            source_output_idx: coreai.runtime.CompileIdentifiers(
                mk.odix_id,
                mk.delegate_id,
            )
            for (op_id, source_output_idx), mk in compile_map.items()
            if op_id == source_op_id
        }

    @staticmethod
    def get_all_compile_identifiers(
        debug_info_records: list[DebugInfoRecord],
    ) -> dict[int, coreai.runtime.CompileIdentifiers]:
        """
        Get compiled operation identifiers for all coreai operations.

        Builds a mapping from every coreai op ID to its
        ``CompileIdentifiers`` by processing all debug info records.

        Args:
            debug_info_records: Debug information containing operation
                mappings.

        Returns:
            Dictionary mapping ``coreai_op_id`` to
            ``CompileIdentifiers``.

        """
        compile_map = _build_compile_identifiers_map(debug_info_records, "coreai")
        return {
            source_op_id: coreai.runtime.CompileIdentifiers(
                mk.odix_id,
                mk.delegate_id,
            )
            for (source_op_id, _output_idx), mk in compile_map.items()
        }
