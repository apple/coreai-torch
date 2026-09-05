# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Compute plan utilities for Core AI models.

This module provides a :class:`ComputePlan` that, for each Core AI operation,
reports the :class:`ComputeDevice` it is scheduled to run on.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TextIO

from coreai._compiler.ir import Operation
from coreai.authoring import AIProgram
from coreai.runtime import AIModel, SpecializationOptions
from typing_extensions import Self

from .annotations import (
    _ANNOTATION_STYLE,
    _DETAIL_STYLE,
    _Annotation,
    _AnnotationLine,
)
from .debug_info import (
    DebugInfo,
    DebugInfoRecord,
    _Delegate,
    get_operation_id,
    parse_debug_infos,
)
from .source_annotator import (
    ModulePath,
    _annotate_operations,
    _operation_in_module,
)
from .table_writer import _Column, _Row, _TableSpec, _write_table
from .utils import _plain, _walk_operations, _with_debug

# Operations left out of write_summary: constants never carry a residency and
# would dominate the table, and a graph is a container, not schedulable work.
_EXCLUDED_FROM_SUMMARY = ("coreai.constant", "coreai.graph")


class Residency(Enum):
    """
    Why an operation has the devices it has -- or has none.
    """

    PLACED = "PLACED"
    """The plan holds an entry and it names at least one real device."""

    NO_ENTRY = "NO ENTRY"
    """The plan holds nothing for this operation. Usual for constants, layout
    operations and anything fused into a neighbor that holds the residency."""

    NO_DEVICE = "NO DEVICE"
    """The plan holds an entry, and it resolves only to ``UNKNOWN``. Distinct from
    :attr:`NO_ENTRY`: the operation was a candidate and the planner had nothing to say
    about it, rather than never being considered."""


@dataclass(frozen=True)
class _ComputeDeviceAnnotation:
    """
    :class:`~coreai_torch.debugging.annotations._Annotation` for compute info.

    Renders the operation's compute device(s) on one colored comment line and
    each delegate validation message on its own colored comment line below it.

    """

    operation_name: str
    """Name of the annotated Core AI operation."""

    devices: tuple[str, ...]
    """Sorted compute device names the operation runs on."""

    validation_messages: tuple[str, ...]
    """Sorted delegate validation messages, rendered one per line."""

    residency: Residency = Residency.PLACED
    """Why those devices, or why none. See :class:`Residency`."""

    def lines(self: Self) -> tuple[_AnnotationLine, ...]:
        """
        Describe the operation, its device, and why the device is what it is.

        The validation messages sit on the device line rather than beneath it. They
        explain the placement, so splitting them off left a reader pairing a device
        with a reason on the next line -- and where a source line carries a dozen
        operations, the reasons and the operations they belong to interleave into an
        unreadable column. A second message is rare enough to be worth the length.

        Returns:
            One line, or one line per message when there are several.

        """
        if self.residency is Residency.PLACED:
            body = ", ".join(self.devices)
            style = _ANNOTATION_STYLE
        else:
            # Styled as a detail rather than as a placement: it is the absence of
            # one, and colouring it like a device reads as though a device was named.
            body = self.residency.value
            style = _DETAIL_STYLE

        head = f"{self.operation_name}: {body}"
        if not self.validation_messages:
            return (_AnnotationLine(head, style),)
        # A declined placement is styled as a caveat in full, not just its reason.
        # `_AnnotationLine` carries one style per line, so folding the reason onto the
        # device line made it inherit the ordinary colour -- the one line worth spotting
        # in an 800-line listing looked exactly like the 88 that placed as asked.
        style = _DETAIL_STYLE
        return tuple(
            _AnnotationLine(f"{head} -- {message}", style)
            if index == 0
            else _AnnotationLine(f"also: {message}", _DETAIL_STYLE, indent="  ")
            for index, message in enumerate(self.validation_messages)
        )

    def data(self: Self) -> dict[str, Any]:
        """
        Return the placement as plain values.

        Returns:
            The operation's name, the devices it runs on, why it has them, and any
            validation messages. ``devices`` is a list because an operation may be
            scheduled on more than one device.

        """
        return {
            "operation": self.operation_name,
            "devices": list(self.devices),
            "residency": self.residency.value,
            "validation_messages": list(self.validation_messages),
        }


class ComputeDevice(Enum):
    """Compute device an operation is scheduled to run on."""

    CPU = "CPU"
    """CPU compute device."""

    GPU = "GPU"
    """GPU compute device."""

    NEURAL_ENGINE = "NEURAL_ENGINE"
    """Neural Engine compute device."""

    UNKNOWN = "UNKNOWN"
    """Unknown compute device."""

    @classmethod
    def _missing_(cls, value: object) -> "ComputeDevice":
        """Return UNKNOWN for unrecognized device values."""
        return cls.UNKNOWN

    @classmethod
    def from_string(cls, value: str) -> "ComputeDevice":
        """
        Parse a device string into a :class:`ComputeDevice`.

        Performs case-insensitive matching against the enum values.

        Args:
            value: Raw device string from debug info metadata.

        Returns:
            The matching :class:`ComputeDevice`, or ``UNKNOWN`` if unrecognized.

        """
        device_str = value.strip().upper()
        device_str = (
            ComputeDevice.NEURAL_ENGINE.value if device_str == "ANE" else device_str
        )
        return cls(device_str)


_SUMMARY_DEVICE_ORDER = (
    ComputeDevice.NEURAL_ENGINE,
    ComputeDevice.GPU,
    ComputeDevice.CPU,
    ComputeDevice.UNKNOWN,
)
"""Device column order used by :meth:`ComputePlan.write_summary`."""


def _get_compute_device(residency: str | None, delegate: _Delegate) -> ComputeDevice:
    if delegate == _Delegate.FALLBACK or delegate == _Delegate.BNNS:
        # These delegates are CPU-only, so the device follows from the delegate
        # itself and no residency is recorded.
        return ComputeDevice.CPU
    elif delegate == _Delegate.MPS:
        # MPS dispatches to more than one device, so only the recorded residency
        # establishes placement.
        return (
            ComputeDevice.UNKNOWN
            if residency is None
            else ComputeDevice.from_string(residency)
        )
    else:
        return ComputeDevice.UNKNOWN


def _requires_residency(delegate: _Delegate) -> bool:
    """
    Whether *delegate* records a residency to establish an operation's device.

    MPS dispatches across compute devices and annotates each operation with the
    device it landed on. CPU-only delegates record no residency, so their
    operations must not be dropped for lacking one.

    Args:
        delegate: Delegate owning the record.

    Returns:
        ``True`` if placement is only meaningful when a residency is present.

    """
    return delegate == _Delegate.MPS


def _get_ane_validation_message(operation: DebugInfo) -> str | None:
    message = operation.get_metadata("ane_validation_message")
    if message is None or message.value_type != "string":
        return None

    return message.value


def _get_validation_message(operation: DebugInfo, delegate: _Delegate) -> str | None:
    if delegate == _Delegate.MPS:
        return _get_ane_validation_message(operation=operation)

    return None


@dataclass
class _ComputeInfo:
    """Internal record tracking compute devices and validation messages for an op."""

    devices: set[ComputeDevice] = field(default_factory=set)
    """Compute devices the operation is scheduled to run on."""

    validation_messages: set[str] = field(default_factory=set)
    """Delegate validation messages (e.g. ANE validation diagnostics)."""


def _build_coreai_id_to_compute_info_map(
    debug_info_records: list[DebugInfoRecord],
) -> dict[int, _ComputeInfo]:
    coreai_id_to_compute_info_map: dict[int, _ComputeInfo] = defaultdict(_ComputeInfo)
    odix_record = next(
        (record for record in debug_info_records if record.is_fallback),
        None,
    )
    if odix_record is None:
        raise ValueError("No debug info record found; cannot build compute plan.")

    for operation in odix_record.operations:
        # Skip operations that represent a function or delegate symbol; these
        # are structural markers rather than directly schedulable ops.
        if operation.is_symbol():
            continue

        op_ids = operation.get_op_ids(level="coreai")
        for op_id in op_ids:
            coreai_id_to_compute_info_map[op_id] = _ComputeInfo(
                devices={ComputeDevice.CPU}
            )

    delegate_records = (
        record for record in debug_info_records if not record.is_fallback
    )
    # Resolve the delegate-specific device (and any validation message) for each
    # operation from the residency of the kernel it is scheduled as.
    for record in delegate_records:
        delegate = _Delegate.from_string(identifier=record.identifier)
        for operation in record.operations:
            # Every coreai ID in the entry fused into the same scheduled
            # kernel, so they all share its residency.
            residency = operation.get_residency()
            if residency is None and _requires_residency(delegate):
                continue

            compute_device = _get_compute_device(residency=residency, delegate=delegate)
            validation_message = _get_validation_message(
                operation=operation, delegate=delegate
            )
            for op_id in operation.get_op_ids(level="coreai"):
                compute_info = coreai_id_to_compute_info_map[op_id]
                compute_info.devices = {compute_device}
                if validation_message is not None:
                    compute_info.validation_messages.add(validation_message)

    return coreai_id_to_compute_info_map


class ComputePlan:
    """
    Describes the set of :class:`ComputeDevice` each Core AI operation executes on.

    A single operation may be scheduled across more than one compute device, so
    the plan associates each Core AI operation with a set of devices.

    The plan is constructed from a :class:`~coreai.authoring.AIProgram` by
    loading it into the runtime (optionally with explicit
    :class:`~coreai.runtime.SpecializationOptions`) and reading the
    operation-level debug info metadata embedded in the resulting model.
    """

    def __init__(self: Self, debug_info_records: list[DebugInfoRecord]) -> None:
        """
        Initialize the compute plan.

        Args:
            debug_info_records: Parsed operation-level debug info records from a
                deployed Core AI model, used to derive the Core AI operation ID
                to compute device mapping.

        """
        self._coreai_id_to_compute_info_map = _build_coreai_id_to_compute_info_map(
            debug_info_records=debug_info_records
        )

    @classmethod
    async def from_program(
        cls,
        program: AIProgram,
        specialization_options: SpecializationOptions | None = None,
    ) -> "ComputePlan":
        """
        Build a compute plan from an AIProgram.

        Args:
            program: AIProgram to build the compute plan for.
            specialization_options: Options for configuring model
                specialization (e.g. preferred compute unit). When None, the
                runtime defaults are used.

        Returns:
            A :class:`ComputePlan` mapping Core AI op IDs to compute devices.

        """
        with TemporaryDirectory() as temp_dir_name:
            asset_path = Path(temp_dir_name) / "model.aimodel"
            specialization_options = _with_debug(specialization_options)
            asset = program.save_asset(asset_path)
            model = await AIModel.load(asset.path, specialization_options)

            debug_info_records = parse_debug_infos(model._debug_infos)
            return cls(debug_info_records=debug_info_records)

    def to_dict(
        self: Self,
        program: AIProgram | None = None,
        *,
        module: ModulePath | None = None,
    ) -> dict[str, Any]:
        """
        Return the plan as plain values.

        One entry per operation the plan holds, each with its residency, so a reader
        never has to infer from ``{UNKNOWN}`` which of the three situations applied.

        *program* is optional because the plan genuinely does not know operation
        names -- it is built from debug info records and keyed by id, and the names
        live on the program's operations. Passing one adds ``operation_names`` and
        the per-name device counts that :meth:`write_summary` renders. Without it the
        entries are still complete, just harder for a human to read.

        Args:
            program: The program the plan was built from, for names and the summary.
            module: Restrict the summary to one module subtree. Ignored when
                *program* is None, since the walk needs it.

        Returns:
            Every entry, a device histogram over them, and -- given a program -- the
            names and the per-operation-name breakdown.

        """
        entries = [
            {
                "operation_id": operation_id,
                "devices": _plain(compute_info.devices),
                "residency": self.residency_for_id(operation_id).value,
                "validation": _plain(compute_info.validation_messages),
            }
            for operation_id, compute_info in sorted(
                self._coreai_id_to_compute_info_map.items()
            )
        ]
        histogram: Counter[str] = Counter(
            device for entry in entries for device in entry["devices"]
        )

        data: dict[str, Any] = {
            "entries": entries,
            "entry_count": len(entries),
            "device_histogram": dict(sorted(histogram.items())),
        }
        if program is None:
            return data

        operations = _walk_operations(program)
        if module is not None:
            operations = [op for op in operations if _operation_in_module(op, module)]
        data["operation_names"] = {
            operation_id: operation.name
            for operation in operations
            if (operation_id := get_operation_id(operation)) is not None
        }
        # The denominator write_summary reports against: how many operations the
        # program holds that the plan says nothing about.
        data["operations_without_entry"] = sorted(
            operation_id
            for operation_id in data["operation_names"]
            if operation_id not in self._coreai_id_to_compute_info_map
        )
        return data

    def _get_devices_for_id(self: Self, operation_id: int) -> set[ComputeDevice]:
        """
        Get the compute devices for a Core AI operation ID.

        Args:
            operation_id: Core AI operation ID.

        Returns:
            The set of :class:`ComputeDevice` the operation runs on, or a set
            containing ``UNKNOWN`` if the operation is not present in the plan.

        """
        compute_info = self._coreai_id_to_compute_info_map.get(operation_id)
        if compute_info is None:
            return {ComputeDevice.UNKNOWN}
        return set(compute_info.devices)

    def residency_for_id(self: Self, operation_id: int) -> Residency:
        """
        Why an operation has the devices it has, or why it has none.

        Args:
            operation_id: Core AI operation ID.

        Returns:
            Which of the three situations applies.

        """
        compute_info = self._coreai_id_to_compute_info_map.get(operation_id)
        if compute_info is None:
            return Residency.NO_ENTRY
        if not compute_info.devices - {ComputeDevice.UNKNOWN}:
            return Residency.NO_DEVICE
        return Residency.PLACED

    def devices_for_id(self: Self, operation_id: int) -> set[ComputeDevice]:
        """
        Get the compute devices for a Core AI operation ID.

        The id-keyed counterpart of :meth:`get_devices`, for callers holding ids rather than
        operations -- a diff against another plan, for instance.

        ``{UNKNOWN}`` conflates three situations; :meth:`residency_for_id` tells them
        apart.

        Args:
            operation_id: Core AI operation ID.

        Returns:
            The devices, or ``{ComputeDevice.UNKNOWN}`` when the plan has no entry.

        """
        return self._get_devices_for_id(operation_id=operation_id)

    def get_devices(self: Self, operation: Operation) -> set[ComputeDevice]:
        """
        Get the compute devices for a Core AI operation.

        Args:
            operation: Core AI operation whose location carries the operation ID.

        Returns:
            The set of :class:`ComputeDevice` the operation runs on, or a set
            containing ``UNKNOWN`` if the operation is not present in the plan.

        """
        operation_id = get_operation_id(operation)
        if operation_id is None:
            return {ComputeDevice.UNKNOWN}
        return self._get_devices_for_id(operation_id=operation_id)

    def validation_messages_for_id(self: Self, operation_id: int) -> set[str]:
        """
        Get the delegate validation messages for a Core AI operation ID.

        The id-keyed counterpart of :meth:`get_validation_messages`. These say why a
        delegate declined an operation, so they are the reason behind a placement.

        Args:
            operation_id: Core AI operation ID.

        Returns:
            The messages, or an empty set when the plan has no entry or the operation has
            none.

        """
        compute_info = self._coreai_id_to_compute_info_map.get(operation_id)
        if compute_info is None:
            return set()
        return set(compute_info.validation_messages)

    def get_validation_messages(self: Self, operation: Operation) -> set[str]:
        """
        Get the delegate validation messages for a Core AI operation.

        Args:
            operation: Core AI operation whose location carries the operation ID.

        Returns:
            The set of validation messages associated with the operation, or an
            empty set if the operation is not present in the plan or has none.

        """
        operation_id = get_operation_id(operation)
        if operation_id is None:
            return set()
        return self.validation_messages_for_id(operation_id=operation_id)

    def _build_summary(
        self: Self,
        program: AIProgram,
        *,
        module: ModulePath | None = None,
    ) -> _TableSpec:
        """
        Build the device-placement table without rendering it.

        Walks ``program`` and counts, per Core AI operation name, the compute
        devices its instances were scheduled on. ``coreai.constant`` and
        ``coreai.graph`` are left out: the former never carries a residency and
        would dominate the table, and the latter is a container rather than
        schedulable work.

        An operation counted as ``UNKNOWN`` is not necessarily unscheduled -- an
        operation fused into a neighbouring kernel has no residency of its own,
        and the device is reported on the operation that absorbed it.

        Returning the spec lets a caller read the same headers and counts that
        :meth:`write_summary` would print, instead of parsing rendered text.

        Args:
            program: AIProgram to summarize. Should be the same program the plan
                was built from so operation IDs line up.
            module: Optional module instance path (outermost first) to restrict
                the summary to a single module subtree.

        Returns:
            A :class:`~coreai_torch.debugging.table_writer._TableSpec` with one
            row per operation name, ordered by count descending then by name.

        """
        operations = _walk_operations(program)
        if module is not None:
            operations = [op for op in operations if _operation_in_module(op, module)]

        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for operation in operations:
            if operation.name in _EXCLUDED_FROM_SUMMARY:
                continue
            if get_operation_id(operation) is None:
                continue
            for device in self.get_devices(operation):
                counts[operation.name][device.value] += 1

        spec = _TableSpec(
            title="Compute device placement per operation",
            columns=(
                _Column("operation"),
                _Column("count", justify="right"),
                *(
                    _Column(device.value, justify="right")
                    for device in _SUMMARY_DEVICE_ORDER
                ),
            ),
        )
        for name, device_counts in sorted(
            counts.items(), key=lambda item: (-sum(item[1].values()), item[0])
        ):
            spec.add(
                _Row(
                    cells=(
                        name,
                        str(sum(device_counts.values())),
                        *(
                            str(device_counts.get(device.value, 0))
                            for device in _SUMMARY_DEVICE_ORDER
                        ),
                    ),
                ),
            )
        return spec

    def write_summary(
        self: Self,
        program: AIProgram,
        output: TextIO | None = None,
        *,
        module: ModulePath | None = None,
    ) -> None:
        """
        Write a table of how many operations of each kind ran on each device.

        Renders what :meth:`_build_summary` describes; see it for which operations
        are counted and how ``UNKNOWN`` should be read.

        Args:
            program: AIProgram to summarize. Should be the same program the plan
                was built from so operation IDs line up.
            output: Text stream to write the table to. Defaults to
                ``sys.stdout`` when None.
            module: Optional module instance path (outermost first) to restrict
                the summary to a single module subtree.

        """
        _write_table(self._build_summary(program, module=module), output)

    def _annotation_for(
        self: Self,
        operation: Operation,
        *,
        include_unplaced: bool = False,
    ) -> _Annotation | None:
        """
        Build the annotation describing where *operation* runs.

        Args:
            operation: Core AI operation to describe.
            include_unplaced: Return an annotation naming the reason when the plan
                holds no usable entry, rather than None.

        Returns:
            The annotation, or None when the plan holds nothing for the operation and
            `include_unplaced` is False.

        """
        operation_id = get_operation_id(operation)
        if operation_id is None:
            return None

        residency = self.residency_for_id(operation_id)
        if residency is not Residency.PLACED and not include_unplaced:
            return None

        compute_info = self._coreai_id_to_compute_info_map.get(operation_id)
        return _ComputeDeviceAnnotation(
            operation_name=operation.name,
            devices=(
                tuple(sorted(device.value for device in compute_info.devices))
                if compute_info is not None
                else ()
            ),
            validation_messages=(
                tuple(sorted(compute_info.validation_messages))
                if compute_info is not None
                else ()
            ),
            residency=residency,
        )

    def annotate_source(
        self: Self,
        program: AIProgram,
        output: TextIO | None = None,
        *,
        module: ModulePath | None = None,
        annotate_all_files: bool = False,
    ) -> None:
        """
        Annotate the program's source with each operation's compute device(s).

        Args:
            program: AIProgram whose source should be annotated. Should be the
                same program the plan was built from so operation IDs line up.
            output: Text stream to write annotated source to. Defaults to
                ``sys.stdout`` when None.
            module: Optional module instance path (outermost first) to restrict
                annotation to a single module subtree. When None (default), all
                operations are annotated.
            annotate_all_files: When True, annotate every attributed source file
                (ordered by attribution count, descending). When False
                (default), only the single dominant file is annotated.

        """
        operations = _walk_operations(program)
        if module is not None:
            operations = [op for op in operations if _operation_in_module(op, module)]

        _annotate_operations(
            operations,
            self._annotation_for,
            output,
            annotate_all_files=annotate_all_files,
        )
