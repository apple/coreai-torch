# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Benchmarker utility for profiling operation timing in AIProgram.

This module provides a framework for benchmarking ML model implementations by
measuring the execution time of each operation using the Profiler API from
standalone_swift and coreai-runtime.

Key components:
- Benchmarker: Main class for collecting and reporting operation timing
- BenchmarkResult: Data class containing timing information for operations
- CoreAIBenchmarker: Benchmarker implementation using Core AI Runtime
"""

import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TextIO

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
import numpy as np
from coreai._compiler.ir import Operation, WalkResult
from coreai.authoring import AIProgram
from coreai.runtime import AIModel, NDArray, Profiler, SpecializationOptions
from typing_extensions import Self

from .annotations import _ANNOTATION_STYLE, _write_line
from .debug_info import DebugInfoRecord, parse_debug_infos
from .source_annotator import _read_source_file, _should_exclude_location
from .utils import LocationInfo, get_operation_locations

logger = logging.getLogger(__name__)

# Constants
_MAX_OP_NAME_LENGTH = 58
_TRUNCATED_OP_NAME_LENGTH = 55


def _get_default_excluded_operations() -> tuple[str, ...]:
    """
    Get default tuple of operation names to exclude from timing measurements.

    Returns:
        Tuple of operation names that should not be timed by default

    """
    return (
        "coreai.graph",
        "coreai.constant",
    )


class _LogEventPhase(Enum):
    """Phase of a profiling event."""

    LOAD = "load"
    """Model loading phase."""

    COMPILE = "compile"
    """Compilation phase."""

    INFERENCE = "inference"
    """Inference/execution phase."""

    UNKNOWN = "unknown"
    """Unknown or unrecognized phase."""

    @classmethod
    def _missing_(cls, value: object) -> "_LogEventPhase":
        """Return UNKNOWN for unrecognized phase values."""
        return cls.UNKNOWN


class _BenchmarkerState(Enum):
    """State of the benchmarker."""

    LOADING = "loading"
    """Loading and initializing model."""

    RUNNING = "running"
    """Actively running benchmark iterations."""

    COMPLETED = "completed"
    """Benchmark completed."""


@dataclass(frozen=True)
class Statistics:
    """Statistical summary of measurements."""

    minimum: float
    """Minimum value."""

    maximum: float
    """Maximum value."""

    average: float
    """Average (mean) value."""

    std_dev: float
    """Standard deviation."""

    median: float
    """Median value."""

    @staticmethod
    def from_values(values: list[float]) -> "Statistics | None":
        """
        Create Statistics from a list of values.

        Args:
            values: List of numeric values

        Returns:
            Statistics object or None if values is empty

        """
        if len(values) == 0:
            return None

        minimum = float(np.min(values))
        maximum = float(np.max(values))
        average = float(np.mean(values))
        std_dev = float(np.std(values))
        median = float(np.median(values))

        return Statistics(
            minimum=minimum,
            maximum=maximum,
            average=average,
            std_dev=std_dev,
            median=median,
        )


@dataclass(frozen=True)
class Measurement:
    """Measurement containing statistics and raw samples."""

    statistics: Statistics | None
    """Statistical summary of the samples."""

    samples: list[float]
    """Raw sample values."""

    @staticmethod
    def from_samples(samples: list[float]) -> "Measurement":
        """
        Create Measurement from a list of samples.

        Args:
            samples: List of sample values

        Returns:
            Measurement object with computed statistics

        """
        return Measurement(
            statistics=Statistics.from_values(values=samples),
            samples=samples,
        )

    @property
    def sort_key(self) -> tuple[bool, float | None]:
        """
        Get sort key for this measurement.

        Returns:
            Tuple of (has_statistics, median_value) for sorting

        """
        return (
            self.statistics is not None,
            self.statistics.median if self.statistics is not None else None,
        )


def _group_operations_by_line(
    module_timing: "ModuleTiming",
    file_path: Path,
) -> dict[int, list[tuple[Operation, "OperationTiming"]]]:
    """
    Group operations by line number for a specific file.

    Args:
        module_timing: ModuleTiming to get operations from
        file_path: Path to match operations against

    Returns:
        Dictionary mapping line numbers to lists of (operation, timing) tuples

    """
    line_timings: dict[int, list[tuple[Operation, OperationTiming]]] = defaultdict(
        list,
    )

    for operation, timing in module_timing.get_all_operations():
        locations = get_operation_locations(operation)

        for loc in locations:
            # Match by filename (handle both absolute and relative paths)
            if Path(loc.filename).name == file_path.name or loc.filename == str(
                file_path,
            ):
                line_timings[loc.line].append((operation, timing))

    return line_timings


def _annotate_source_file(
    module_timing: "ModuleTiming",
    file_path: Path | str,
    output: TextIO,
) -> None:
    """
    Annotate a source file with timing information from a module.

    Reads the source file, finds operations from that file in the module, and
    writes the annotated source with a timing comment before each annotated line.

    The comment is styled by name rather than by escape code, so colour is applied
    for a terminal and dropped for a file or in-memory stream. Writing the escapes
    unconditionally left them in the text of every saved listing, which a terminal
    renders and an editor shows as noise.

    Args:
        module_timing: ModuleTiming to get operation timings from
        file_path: Path to the source file to annotate
        output: Text stream to write annotated source to (file or stdout)

    """
    file_path = Path(file_path)

    # Read the source file
    source_lines = _read_source_file(file_path, output)
    if source_lines is None:
        return

    # Group operations by line number
    line_timings = _group_operations_by_line(module_timing, file_path)

    # Annotate and write source lines
    for line_num, line_content in enumerate(source_lines, start=1):
        # Write timing annotation BEFORE the source line if present
        if line_num in line_timings:
            # Collect timing info for this line
            timings_for_line = []
            for operation, timing in line_timings[line_num]:
                if timing.measurement.statistics:
                    stats = timing.measurement.statistics
                    timings_for_line.append(
                        f"{operation.name}: {stats.average:.3f}ms (med: {stats.median:.3f}ms)",
                    )

            if timings_for_line:
                # Indent the comment to match the line it describes, so the
                # annotated listing stays readable as Python.
                margin = line_content[: len(line_content) - len(line_content.lstrip())]
                annotation = margin + "# " + ", ".join(timings_for_line)
                _write_line(output, annotation, _ANNOTATION_STYLE)

        # Write the original source line
        output.write(line_content)


@dataclass
class OperationTiming:
    """Timing information for a single operation."""

    op_id: int
    """Operation ID from compile identifiers."""

    measurement: Measurement
    """Measurement containing statistics and timing samples in milliseconds."""

    def write_to(
        self,
        output: TextIO,
        operation: Operation | None = None,
        prefix: str = "",
    ) -> None:
        """
        Write operation timing to output.

        Args:
            output: Text stream to write to
            operation: Optional operation object for getting name and ID
            prefix: Prefix string for indentation (default: "")

        """
        # Get operation name and ID
        if operation:
            op_name = operation.name
            op_id_obj = _mlir.get_operation_id(operation.location, "coreai")  # type: ignore[attr-defined]
            op_id = getattr(op_id_obj, "value", "N/A") if op_id_obj else "N/A"
        else:
            op_name = "unknown"
            op_id = self.op_id

        # Truncate long operation names
        if len(op_name) > _MAX_OP_NAME_LENGTH:
            op_name = op_name[:_TRUNCATED_OP_NAME_LENGTH] + "..."

        # Format and write statistics
        if self.measurement.statistics:
            stats = self.measurement.statistics
            output.write(
                f"{prefix}{op_id!s:<10} {op_name:<60} "
                f"{stats.median:<12.6f} {stats.average:<12.6f} "
                f"{stats.minimum:<12.6f} {stats.maximum:<12.6f} {stats.std_dev:<12.6f}\n",
            )
        else:
            output.write(
                f"{prefix}{op_id!s:<10} {op_name:<60} "
                f"{'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n",
            )


@dataclass
class ModuleTiming:
    """
    Timing information for a module and its operations.

    Represents a hierarchical grouping of operations by their module path.
    """

    name: str
    """Module name."""

    operation_timings: list[tuple[Operation, OperationTiming]]
    """List of (operation, timing) tuples for operations in this module."""

    children: list["ModuleTiming"]
    """Child modules."""

    @property
    def aggregated_op_stats(self) -> Statistics | None:
        """
        Get aggregated operation statistics for this module (including children).

        Collects all samples from operations in this module and its children,
        then computes statistics on the combined samples. The resulting statistics
        represent the distribution of per-operation timings, not per-iteration totals.

        Returns:
            Statistics object with aggregated per-operation timing or None if no samples

        """
        # Collect all samples from this module and all children efficiently
        all_samples = []
        for _, timing in self.get_all_operations():
            all_samples.extend(timing.measurement.samples)

        # Return statistics from all collected samples
        return Statistics.from_values(all_samples)

    @property
    def total_time(self) -> Statistics | None:
        """
        Get total time statistics for this module.

        This is an alias for aggregated_op_stats for backward compatibility.

        Returns:
            Statistics object with aggregated per-operation timing or None if no samples

        """
        return self.aggregated_op_stats

    def get_all_operations(self: Self) -> list[tuple[Operation, OperationTiming]]:
        """
        Get all operations in this module and its children recursively.

        Returns:
            List of all (operation, timing) tuples

        """
        all_ops = list(self.operation_timings)
        for child in self.children:
            all_ops.extend(child.get_all_operations())
        return all_ops

    def get_all_modules(self: Self) -> list["ModuleTiming"]:
        """
        Get a flattened list of this module and all its children recursively.

        Returns:
            List of all ModuleTiming objects (this module and all descendants)

        """
        all_modules: list[ModuleTiming] = [self]
        for child in self.children:
            all_modules.extend(child.get_all_modules())
        return all_modules

    def get_operations_at_location(
        self: Self,
        location: LocationInfo,
    ) -> list[tuple[Operation, OperationTiming]]:
        """
        Find operations at a specific source location.

        Searches this module and all children recursively for operations
        that have the given file/line/col location.

        Args:
            location: LocationInfo to search for

        Returns:
            List of (operation, timing) tuples matching the location

        """
        matching_ops = []

        # Search all operations in this module and children
        for operation, timing in self.get_all_operations():
            op_locations = get_operation_locations(operation)

            # Check if any of the operation's locations match
            for op_loc in op_locations:
                if (
                    op_loc.filename == location.filename
                    and op_loc.line == location.line
                    and op_loc.col == location.col
                ):
                    matching_ops.append((operation, timing))
                    break  # Don't add the same operation twice

        return matching_ops

    def annotate_dominant_source(
        self: Self,
        output: TextIO,
        exclude: Callable[[LocationInfo], bool] | None = None,
    ) -> None:
        """
        Find the dominant source file and annotate it with timing information.

        Uses operations only from this module (not children). For each operation,
        gets file/line/col locations, filters them, and takes the last valid file.
        The dominant file is the one that appears most frequently.

        Args:
            output: Text stream to write annotated source to (file or stdout)
            exclude: Optional callable to filter out locations. If None, uses default
                    which excludes torch files, exported_program.py, and "-"

        """
        # Use default exclusion if not provided
        if exclude is None:
            exclude = _should_exclude_location

        # Count occurrences of each source file from locations
        file_counts: dict[str, int] = defaultdict(int)
        # Only use operations from this module, not children
        for operation, _ in self.operation_timings:
            # Get file/line/col locations
            locations = get_operation_locations(operation)

            if locations:
                # Take the last file (innermost)
                last_loc = locations[-1]

                # Check if it should be excluded
                if not exclude(last_loc):
                    last_file = last_loc.filename

                    # Only count if file exists
                    if Path(last_file).exists():
                        file_counts[last_file] += 1

        if not file_counts:
            output.write("# No valid locations found in operations\n")
            return

        # Find the most common file (dominant)
        dominant_file = max(file_counts.items(), key=lambda x: x[1])[0]

        # Annotate the dominant file
        _annotate_source_file(self, dominant_file, output)

    def write_to(
        self: Self,
        output: TextIO,
        indent: int = 0,
        show_operations: bool = False,
    ) -> None:
        """
        Write module timing to output.

        Args:
            output: Text stream to write to
            indent: Indentation level (default: 0)
            show_operations: Whether to show individual operations (default: False)

        """
        prefix = "  " * indent

        # Module header with statistics
        stats = self.aggregated_op_stats
        if stats:
            output.write(
                f"{prefix}- {self.name} "
                f"[Avg: {stats.average:.3f}ms, "
                f"Median: {stats.median:.3f}ms, "
                f"Min: {stats.minimum:.3f}ms, "
                f"Max: {stats.maximum:.3f}ms]\n",
            )
        else:
            output.write(f"{prefix}- {self.name} [No timing data]\n")

        # Show operations if requested
        if show_operations and self.operation_timings:
            for operation, timing in self.operation_timings:
                # Use the timing's write_to method with proper indentation
                timing.write_to(output, operation, prefix + "  - ")
        elif self.operation_timings:
            output.write(f"{prefix}  ({len(self.operation_timings)} operations)\n")

        # Recursively write children
        for child in self.children:
            child.write_to(output, indent + 1, show_operations)


@dataclass
class BenchmarkResult:
    """
    Result of benchmarking an AIProgram.

    Contains timing information for each operation, organized by operation ID.
    """

    operation_timings: list[tuple[Operation, OperationTiming]]
    """
    List of (operation, timing) tuples for each profiled operation.
    """

    total_duration_ns: int
    """Total execution time in nanoseconds."""

    def get_average_timing(self: Self, op_id: int) -> float | None:
        """
        Get average execution time for an operation in milliseconds.

        Args:
            op_id: Operation ID to get timing for

        Returns:
            Average duration in milliseconds, or None if no measurements exist

        """
        for _, timing in self.operation_timings:
            if timing.op_id == op_id:
                if timing.measurement.statistics:
                    return timing.measurement.statistics.average
                return None
        return None

    def get_measurement(self: Self, op_id: int) -> Measurement | None:
        """
        Get measurement for an operation.

        Args:
            op_id: Operation ID to get measurement for

        Returns:
            Measurement object or None if no measurements exist

        """
        for _, timing in self.operation_timings:
            if timing.op_id == op_id:
                return timing.measurement
        return None

    def get_operation_summary(self: Self) -> list[tuple[Operation, Measurement]]:
        """
        Get summary of operation timings sorted by median duration.

        Returns:
            List of (operation, measurement) tuples sorted by median duration (descending)

        """
        summary = [(op, timing.measurement) for op, timing in self.operation_timings]

        # Sort by median duration (descending)
        summary.sort(key=lambda x: x[1].sort_key, reverse=True)
        return summary

    def get_module_timings(self) -> dict[str, ModuleTiming]:
        """
        Group operation timings by modules based on stack traces.

        Creates a hierarchical tree structure where each level in the stack trace
        becomes a nested module.

        Returns:
            Dictionary mapping module names to ModuleTiming objects at the top level

        """
        # Build module tree structure
        root_modules: dict[str, ModuleTiming] = {}

        for operation, timing in self.operation_timings:
            # Get stack trace from operation location
            stack_trace = _mlir.get_stack_trace(operation.location)  # type: ignore[attr-defined]

            # Treat operations without stack traces as belonging to "<unknown>" module
            if not stack_trace:
                stack_trace = ["<unknown>"]

            # Build hierarchy from stack trace (outermost to innermost)
            # The stack trace is already reversed, so first entry is outermost
            current_level = root_modules
            parent_module = None

            for frame in stack_trace:
                # Find or create module at this level
                if frame not in current_level:
                    new_module = ModuleTiming(
                        name=frame,
                        operation_timings=[],
                        children=[],
                    )
                    current_level[frame] = new_module

                    # Add to parent's children list if this isn't a root module
                    if parent_module is not None:
                        parent_module.children.append(new_module)

                parent_module = current_level[frame]

                # Move to next level (children)
                # Build a dict from children for easy lookup
                children_dict = {child.name: child for child in parent_module.children}
                current_level = children_dict

            # Add operation to the deepest (innermost) module
            if parent_module is not None:
                parent_module.operation_timings.append((operation, timing))

        return root_modules

    def write_summary(
        self: Self,
        output: TextIO,
        top_n: int | None = None,
    ) -> None:
        """
        Write benchmark results summary to output.

        Args:
            output: Text stream to write to
            top_n: If specified, only show top N slowest operations

        """
        summary = self.get_operation_summary()
        if top_n is not None:
            summary = summary[:top_n]

        output.write("=" * 150 + "\n")
        output.write("Benchmark Results\n")
        output.write("=" * 150 + "\n")
        output.write(f"Total execution time: {self.total_duration_ns / 1e9:.3f} s\n")
        output.write(f"Total operations profiled: {len(self.operation_timings)}\n")
        output.write("\n")
        output.write("Per-operation timing (sorted by median duration):\n")
        output.write("-" * 150 + "\n")
        output.write(
            f"{'Op ID':<10} {'Operation':<60} {'Median (ms)':<12} "
            f"{'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'StdDev (ms)':<12}\n",
        )
        output.write("-" * 150 + "\n")

        for operation, measurement in summary:
            # Find the timing for this operation to use its write_to method
            timing = None
            for _, t in self.operation_timings:
                if t.measurement == measurement:
                    timing = t
                    break

            if timing:
                timing.write_to(output, operation)
            else:
                # Fallback if timing not found (shouldn't happen)
                op_name = operation.name if operation else "unknown"
                op_id = "N/A"
                output.write(
                    f"{op_id!s:<10} {op_name:<60} "
                    f"{'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n",
                )

        output.write("=" * 150 + "\n")


class Benchmarker(ABC):
    """
    Base benchmarker class with common profiling logic.

    Subclasses implement runtime-specific execution methods.
    """

    def __init__(
        self: Self,
        coreai_program: AIProgram,
        entry_point: str = "main",
        excluded_operations: tuple[str, ...] | None = None,
        specialization_options: SpecializationOptions | None = None,
    ) -> None:
        """
        Initialize the base benchmarker.

        Args:
            coreai_program: AIProgram to benchmark
            entry_point: Name of the function to profile (default: "main")
            excluded_operations: Tuple of operation names to exclude from timing (default: None)
            specialization_options: Options for configuring model specialization

        """
        self.coreai_program = coreai_program
        self.entry_point = entry_point
        self.excluded_operations = excluded_operations or ()
        self.specialization_options = specialization_options
        self._intervals: dict[int, tuple[int, Any]] = {}
        self._timings: dict[int, list[float]] = defaultdict(list)
        self._interval_counter = 0
        self._total_start: int | None = None
        self._total_end: int | None = None
        self._debug_info_records: list[DebugInfoRecord] = []
        self._odix_to_coreai_map: dict[int, list[int]] = {}
        self._coreai_operations: dict[int, Operation] = {}
        self._lock = threading.Lock()
        self._state = _BenchmarkerState.LOADING

    def _extract_coreai_operations(self: Self) -> None:
        """
        Extract Core AI operations from the coreai_program module.

        Walks the module and stores operations by their Core AI operation ID.
        """
        self._coreai_operations.clear()

        def walk_operations(op: Operation) -> WalkResult:
            op_id_obj = _mlir.get_operation_id(op.location, "coreai")  # type: ignore[attr-defined]
            if op_id_obj is not None:
                coreai_id = getattr(op_id_obj, "value", None)
                if coreai_id is not None:
                    self._coreai_operations[coreai_id] = op
            return WalkResult.ADVANCE

        self.coreai_program._mlir_module.operation.walk(walk_operations)

    def _build_odix_to_coreai_map(self: Self) -> None:
        """
        Build a mapping from ODIX IDs to Core AI operation IDs for fast lookup.

        A single ODIX ID can map to multiple Core AI IDs, so we store a list.
        This should be called once after loading debug_infos.
        """
        self._odix_to_coreai_map.clear()

        for record in self._debug_info_records:
            for debug_info in record.operations:
                odix_id = debug_info.odix_id
                coreai_id = debug_info.get_op_id("coreai")

                if coreai_id is not None and isinstance(coreai_id, int):
                    if odix_id not in self._odix_to_coreai_map:
                        self._odix_to_coreai_map[odix_id] = []
                    self._odix_to_coreai_map[odix_id].append(coreai_id)

    def _get_coreai_op_ids(self: Self, odix_id: int) -> list[int]:
        """
        Convert ODIX ID from compile_ids to list of Core AI operation IDs using debug_infos.

        Args:
            odix_id: ODIX ID from compile_ids.id

        Returns:
            List of Core AI operation IDs if found, otherwise empty list

        """
        return self._odix_to_coreai_map.get(odix_id, [])

    def _reset_state(self: Self) -> None:
        """Reset internal state before a new benchmark run."""
        with self._lock:
            self._intervals.clear()
            self._timings.clear()
            self._interval_counter = 0
            self._total_start = None
            self._total_end = None

    def _on_log_event_begin(self: Self, event: Any) -> int:
        """
        Handle profiler interval begin events.

        Only processes events during RUNNING state and INFERENCE phase.

        Args:
            event: LogEvent from the profiler

        Returns:
            Interval ID for tracking this event

        """
        # Only process events when actively running benchmark
        if self._state != _BenchmarkerState.RUNNING:
            return 0  # Return dummy interval_id

        # Only process inference phase events
        phase = _LogEventPhase(event.phase)
        if phase != _LogEventPhase.INFERENCE:
            return 0  # Return dummy interval_id for non-inference events

        with self._lock:
            interval_id = self._interval_counter
            self._interval_counter += 1

            # Store interval start information
            self._intervals[interval_id] = (
                event.timestamp,
                event.compile_ids,
            )

            # Track total execution time
            if self._total_start is None:
                self._total_start = event.timestamp

            return interval_id

    def _on_log_event_end(self: Self, event: Any, interval_id: int) -> None:
        """
        Handle profiler interval end events.

        Args:
            event: LogEvent from the profiler
            interval_id: Interval ID from the begin callback

        """
        # Only process events when actively running benchmark
        if self._state != _BenchmarkerState.RUNNING:
            return

        # Only process inference phase events
        phase = _LogEventPhase(event.phase)
        if phase != _LogEventPhase.INFERENCE:
            return

        with self._lock:
            # Retrieve start information (only INFERENCE events are stored)
            start_info = self._intervals.get(interval_id)
            if start_info is None:
                # Event was not stored or already processed
                return

            start_time, compile_ids = start_info

            # Calculate duration in milliseconds
            duration_ns = event.timestamp - start_time
            duration_ms = float(duration_ns) / 1e6

            # Convert ODIX ID to Core AI IDs (there may be multiple)
            odix_id = compile_ids.id
            coreai_ids = self._get_coreai_op_ids(odix_id)

            # If we have Core AI IDs, distribute timing equally (mean) to all of them
            if coreai_ids:
                # Divide timing equally among all Core AI operations
                mean_duration_ms = duration_ms / len(coreai_ids)
                for coreai_id in coreai_ids:
                    self._timings[coreai_id].append(mean_duration_ms)
            else:
                # No Core AI ID mapping found - log warning and skip storing
                logger.warning(
                    "No Core AI ID mapping found for ODIX ID %d, skipping timing sample",
                    odix_id,
                )

            # Track total execution time
            self._total_end = event.timestamp

            # Clean up interval
            self._intervals.pop(interval_id)

    def _create_result(self: Self) -> BenchmarkResult:
        """
        Create BenchmarkResult from collected timings.

        Returns:
            BenchmarkResult containing timing information

        """
        with self._lock:
            total_duration = (
                self._total_end - self._total_start
                if self._total_start and self._total_end
                else 0
            )

            # Convert raw timings to list of (operation, _OperationTiming) tuples
            # Filter out excluded operations
            operation_timings_list = []
            for op_id, samples in self._timings.items():
                operation = self._coreai_operations.get(op_id)
                if operation is not None:
                    # Skip excluded operations
                    if operation.name in self.excluded_operations:
                        continue

                    timing = OperationTiming(
                        op_id=op_id,
                        measurement=Measurement.from_samples(samples),
                    )
                    operation_timings_list.append((operation, timing))

            return BenchmarkResult(
                operation_timings=operation_timings_list,
                total_duration_ns=total_duration,
            )

    @abstractmethod
    async def benchmark(
        self: Self,
        inputs: dict[str, Any],
        num_runs: int = 1,
    ) -> BenchmarkResult:
        """
        Benchmark the coreai_program program with the given inputs.

        Args:
            inputs: Dictionary mapping input names to tensor values
            num_runs: Number of times to run the benchmark (default: 1)

        Returns:
            BenchmarkResult containing timing information for all operations

        """
        ...


class CoreAIBenchmarker(Benchmarker):
    """Benchmarker using Core AI Runtime."""

    async def benchmark(
        self: Self,
        inputs: dict[str, Any],
        num_runs: int = 1,
    ) -> BenchmarkResult:
        """
        Benchmark using Core AI Runtime.

        Args:
            inputs: Dictionary mapping input names to tensor values
            num_runs: Number of times to run the benchmark (default: 1)

        Returns:
            BenchmarkResult containing timing information

        """
        # Reset state
        self._reset_state()

        # Create target (AIProgram) inspector based on inspector type
        with TemporaryDirectory() as temp_dir_name:
            asset_path = Path(temp_dir_name) / "model.aimodel"

            # Create asset from AIProgram and load model from asset
            asset = self.coreai_program.save_asset(asset_path)
            specialization_options = (
                self.specialization_options.with_debug(enabled=True)
                if self.specialization_options is not None
                else None
            )
            model = await AIModel.load(asset.path, specialization_options)
            # Load and parse debug_infos
            debug_infos_bytes = model._debug_infos
            self._debug_info_records = parse_debug_infos(debug_infos_bytes)

            # Extract Core AI operations from module
            self._extract_coreai_operations()

            # Build ODIX to Core AI ID mapping for fast lookup
            self._build_odix_to_coreai_map()

            # Create profiler with callbacks
            profiler = Profiler(
                on_log_event_begin=self._on_log_event_begin,
                on_log_event_end=self._on_log_event_end,
            )

            # Load function with profiler
            function = model.load_function(self.entry_point, profiler=profiler)
            if function is None:
                msg = f"Function '{self.entry_point}' not found in model"
                raise ValueError(msg)

            # Convert inputs to NDArray format
            nd_inputs = {}
            for name, value in inputs.items():
                if not isinstance(value, NDArray):
                    nd_inputs[name] = NDArray(value)
                else:
                    nd_inputs[name] = value

            # Run benchmark
            logger.info(
                "Running Core AI Runtime benchmark with %d iteration(s)...",
                num_runs,
            )

            # Transition to RUNNING state to start collecting timing data
            self._state = _BenchmarkerState.RUNNING

            for i in range(num_runs):
                logger.debug("Benchmark run %d/%d", i + 1, num_runs)
                await function(nd_inputs)

            # Transition to COMPLETED state to stop collecting timing data
            self._state = _BenchmarkerState.COMPLETED

            # Ensure all profiler callbacks have completed before creating result
            # The runtime should ensure callback completion after function execution,
            # but we add explicit synchronization to be safe
            with self._lock:
                # At this point, all function executions are complete, and this lock
                # ensures any remaining callback processing is finished
                pass

            result = self._create_result()
            logger.info(
                "Benchmark complete: %d operations profiled",
                len(result.operation_timings),
            )

            return result


async def benchmark_coreai_program(  # noqa: PLR0913
    coreai_program: AIProgram,
    inputs: dict[str, Any],
    entry_point: str = "main",
    num_runs: int = 1,
    excluded_operations: tuple[str, ...] | None = None,
    specialization_options: SpecializationOptions | None = None,
) -> BenchmarkResult:
    """
    Benchmark an AIProgram with profiling.

    Args:
        coreai_program: AIProgram to benchmark
        inputs: Dictionary mapping input names to tensor values
        entry_point: Name of the function to profile (default: "main")
        num_runs: Number of times to run the benchmark (default: 1)
        excluded_operations: Tuple of operation names to exclude from timing
                           (default: ("coreai.graph", "coreai.constant"))
        specialization_options: Options for configuring model specialization

    Returns:
        BenchmarkResult containing timing information for all operations

    """
    # Use default excluded operations if not specified
    if excluded_operations is None:
        excluded_operations = _get_default_excluded_operations()

    # Create the appropriate benchmarker implementation
    benchmarker = CoreAIBenchmarker(
        coreai_program, entry_point, excluded_operations, specialization_options
    )
    return await benchmarker.benchmark(inputs, num_runs)
