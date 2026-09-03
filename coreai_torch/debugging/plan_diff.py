# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Compare two compute plans and report what moved.

An absolute placement report is only actionable against a baseline, so this reports
differences: which operations the planner put on a different device.

Placement is deterministic, so a diff is signal rather than noise. Use it across builds to
catch regressions, or across configurations to see where two devices disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TextIO

from coreai.authoring import AIProgram
from coreai.runtime import SpecializationOptions
from typing_extensions import Self

from .compute_plan import ComputeDevice, ComputePlan
from .debug_info import _build_coreai_op_map
from .graph_diff import OpIdAlignment, op_id_alignment
from .graph_match import WeightPolicy
from .table_writer import _Column, _Row, _TableSpec, _write_table
from .utils import _plain


@dataclass(frozen=True)
class ComputePlacement:
    """Where one operation runs, in a single plan."""

    operation_id: int
    """Identifier of the operation, in its own program's numbering."""

    devices: tuple[ComputeDevice, ...]
    """The devices it runs on, sorted by name."""

    name: str = "?"
    """Operation name, when known."""

    validation: tuple[str, ...] = ()
    """Delegate validation messages, sorted: why a delegate declined the operation."""

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Return the placement as plain values.

        Returns:
            Where the operation runs and why a delegate declined it, if one did.

        """
        return {
            "operation_id": self.operation_id,
            "name": self.name,
            "devices": _plain(self.devices),
            "validation": _plain(self.validation),
        }


@dataclass(frozen=True)
class ComputePlacementChange:
    """How one operation's placement differs between two compute plans."""

    operation_id: int
    """Identifier of the operation, in the earlier program's numbering."""

    before: tuple[ComputeDevice, ...]
    """Devices in the earlier plan, sorted by name."""

    after: tuple[ComputeDevice, ...]
    """Devices in the later plan, sorted by name.

    A plan reports ``UNKNOWN`` for an operation it holds no entry for, so a move to
    ``UNKNOWN`` means the later plan stopped resolving a placement the earlier one had.
    """

    name: str = "?"
    """Operation name, when known."""

    modified: bool = False
    """Whether the operation itself was wired or configured differently --
    ``alignment.modified``. The move may be that change rather than a planner decision:
    widening a matmul can move it off a device, which is the edit doing what it asked for.
    An unmodified operation moving is the stronger signal."""

    validation_before: tuple[str, ...] = ()
    """Delegate validation messages in the earlier plan, sorted."""

    validation_after: tuple[str, ...] = ()
    """Delegate validation messages in the later plan, sorted.

    A move off a delegate is only actionable with the reason it was declined, and the
    message is the plan's own account of it. Both sides are kept: which messages are new
    is what explains the move.
    """

    @property
    def reason(self: Self) -> tuple[str, ...]:
        """
        Why the operation is where it now is, as the plans account for it.

        The messages the later plan gives and the earlier one did not, since those are what
        the move introduced. Falls back to the earlier plan's, which is the account of a
        placement the operation has now stopped being refused.

        Returns:
            The messages, or empty when neither plan gave one.

        """
        appeared = tuple(
            message
            for message in self.validation_after
            if message not in self.validation_before
        )
        return appeared or self.validation_before

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Return the move as plain values.

        :attr:`reason` is included: it is derived from the two message sets and is
        the only field that says *why* the operation moved, so a caller reading the
        fields alone would have to reimplement the "what the later plan newly said"
        rule to get it.

        Returns:
            Both placements, whether the operation itself changed, both plans'
            messages, and the reason drawn from them.

        """
        return {
            "operation_id": self.operation_id,
            "name": self.name,
            "before": _plain(self.before),
            "after": _plain(self.after),
            "modified": self.modified,
            "validation_before": _plain(self.validation_before),
            "validation_after": _plain(self.validation_after),
            "reason": _plain(self.reason),
        }


@dataclass
class ComputePlanDiff:
    """
    Every placement difference between two compute plans.

    The fields account for every operation of either program: each appears in
    :attr:`changes`, in :attr:`only_before` or :attr:`only_after`, or is counted in
    :attr:`unchanged`. Nothing is dropped, so a total taken from here is the whole program.
    """

    changes: list[ComputePlacementChange] = field(default_factory=list)
    """Operations placed differently either side, in operation order."""

    unchanged: int = 0
    """Operations placed identically in both, so a change count has a denominator."""

    unresolved: int = 0
    """Paired operations neither plan holds an entry for, so nothing was compared."""

    only_before: list[ComputePlacement] = field(default_factory=list)
    """Operations the edit removed, with where they used to run."""

    only_after: list[ComputePlacement] = field(default_factory=list)
    """Operations the edit added, with where they landed."""

    alignment: OpIdAlignment | None = None
    """The operation correspondence this comparison was built on."""

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Return the placement comparison as plain values.

        Returns:
            Every move, the counts that give them a denominator, the operations
            present on one side only, and the correspondence used.

        """
        return {
            "changes": [change.to_dict() for change in self.changes],
            "unchanged": self.unchanged,
            "unresolved": self.unresolved,
            "only_before": [entry.to_dict() for entry in self.only_before],
            "only_after": [entry.to_dict() for entry in self.only_after],
            "alignment": self.alignment.to_dict() if self.alignment else None,
        }

    def write_to(self: Self, output: TextIO, *, width: int | None = None) -> None:
        """
        Write the diff as a table.

        Args:
            output: Text stream to write to.
            width: Console width to render at. Defaults to the table writer's own.

        """
        compared = len(self.changes) + self.unchanged
        modified = sum(entry.modified for entry in self.changes)
        spec = _TableSpec(
            title="Placement comparison",
            columns=(
                _Column("Status"),
                _Column("Operation"),
                _Column("Before"),
                _Column("After"),
                _Column("Reason"),
            ),
            caption=(
                f"{len(self.changes)} change(s) of {compared} placed either side "
                f"({modified} on a modified operation), {self.unchanged} unchanged, "
                f"{self.unresolved} unresolved, {len(self.only_before)} only before, "
                f"{len(self.only_after)} only after. Unresolved means neither plan held "
                "an entry, so nothing was compared -- constants and structural "
                "operations are never placed. An operation present on one side only has "
                "no counterpart to compare, so its placement is shown on its own side."
            ),
            show_lines=True,
        )

        # Unmodified first: a move the edit did not ask for is what a reader is after.
        for change in sorted(self.changes, key=lambda entry: entry.modified):
            spec.add(
                _Row(
                    cells=(
                        "moved (modified)" if change.modified else "moved",
                        _operation_cell(change.operation_id, change.name),
                        _device_cell(change.before),
                        _device_cell(change.after),
                        _reason_cell(change),
                    ),
                ),
            )

        for placement, status in (
            *((entry, "only before") for entry in self.only_before),
            *((entry, "only after") for entry in self.only_after),
        ):
            devices = _device_cell(placement.devices)
            spec.add(
                _Row(
                    cells=(
                        status,
                        _operation_cell(placement.operation_id, placement.name),
                        devices if status == "only before" else "--",
                        devices if status == "only after" else "--",
                        "\n".join(placement.validation) or "--",
                    ),
                ),
            )

        _write_table(spec, output, width=width)


def _sorted_devices(devices: set[ComputeDevice]) -> tuple[ComputeDevice, ...]:
    """
    Order a device set by name, so set order cannot cause a false difference.

    Args:
        devices: Devices for one operation.

    Returns:
        The same devices, ordered.

    """
    return tuple(sorted(devices, key=lambda device: device.name))


def _sorted_messages(messages: set[str]) -> tuple[str, ...]:
    """
    Order validation messages, so set order cannot reorder a report.

    Args:
        messages: Messages for one operation.

    Returns:
        The same messages, ordered.

    """
    return tuple(sorted(messages))


def _reason_cell(change: ComputePlacementChange) -> str:
    """
    Render why an operation moved, saying which plan gave the reason.

    Args:
        change: The change to explain.

    Returns:
        The messages, one per line, or ``--`` when neither plan gave one.

    """
    if change.reason and change.reason == change.validation_before:
        # A message only the earlier plan gave explains the earlier placement, not the
        # move. Unlabelled it reads as the cause of a move it predates.
        return "\n".join(f"before only: {message}" for message in change.reason)
    return "\n".join(change.reason) or "--"


def _device_cell(devices: tuple[ComputeDevice, ...]) -> str:
    """
    Render devices for a report.

    Args:
        devices: Devices for one operation.

    Returns:
        Their names, one per line.

    """
    return "\n".join(device.name for device in devices)


def _operation_cell(operation_id: int, name: str) -> str:
    """
    Render an operation for a report.

    Args:
        operation_id: Identifier of the operation.
        name: Its name.

    Returns:
        The id with the name, the dialect prefix dropped.

    """
    return f"{operation_id} {name.removeprefix('coreai.')}"


async def compare_compute_plans(
    before_program: AIProgram,
    after_program: AIProgram,
    entry_point: str = "main",
    *,
    specialization_options: SpecializationOptions | None = None,
    weights: WeightPolicy = WeightPolicy.IGNORE,
) -> ComputePlanDiff:
    """
    Plan two programs and report how their placements differ.

    One call rather than two plans a caller pairs up, because the pairing needs the op id
    alignment: ids renumber when a program changes, so comparing them directly would report
    a renumbered operation as removed and its new id as added.

    Args:
        before_program: The earlier program.
        after_program: The later program.
        entry_point: Function to plan.
        specialization_options: Options for both plans. Pass the same ones to compare
            programs; vary them and pass the same program twice to compare configurations.
        weights: Whether parameter values count towards an operation's identity, for the
            alignment. `IGNORE` cannot tell two same-shaped layers apart, so on a model with
            repeated identical blocks it may pair one block's weight preparation with
            another's and drop the operations between them as removed-plus-added, hiding a
            placement change behind a spurious insert and delete. Prefer `DIGEST`
            when one process produced both programs -- it disambiguates even when the values
            differ, since what pairing needs is a unique label, not an equal one.

    Returns:
        The differences.

    """
    alignment = op_id_alignment(
        before_program,
        after_program,
        entry_point,
        weights=weights,
    )
    before_plan = await ComputePlan.from_program(
        before_program,
        specialization_options=specialization_options,
    )
    after_plan = await ComputePlan.from_program(
        after_program,
        specialization_options=specialization_options,
    )
    return diff_compute_plans(
        before_plan,
        after_plan,
        alignment,
        names_before=_operation_names(before_program),
        names_after=_operation_names(after_program),
    )


def _operation_names(program: AIProgram) -> dict[int, str]:
    """
    Operation names for the report, keyed by that program's own op ids.

    One program at a time: the two numberings overlap, so merging them would name an added
    operation after whatever held its id in the other program.

    Args:
        program: The program to read.

    Returns:
        Names by op id.

    """
    return {
        op_id: operation.name
        for op_id, operation in _build_coreai_op_map(program).items()
    }


def diff_compute_plans(
    before: ComputePlan,
    after: ComputePlan,
    alignment: OpIdAlignment,
    names_before: dict[int, str] | None = None,
    names_after: dict[int, str] | None = None,
) -> ComputePlanDiff:
    """
    Compare two compute plans through an op id alignment.

    Operations are paired by :attr:`OpIdAlignment.mapping`, not by equal ids: an id means
    different things in two programs once anything is inserted. An id the alignment calls
    ``added`` or ``removed`` has no counterpart to compare, so its placement is reported on
    its own side rather than dropped -- where removed work ran, and where added work landed,
    are the questions a comparison cannot answer. A pair the alignment calls ``modified`` is
    still compared, flagged so its move can be read as the edit rather than the planner.

    Args:
        before: Plan of the earlier program.
        after: Plan of the later program.
        alignment: Which operation of the earlier program became which of the later.
        names_before: Names by op id in the earlier program. Missing entries render as ``?``.
        names_after: Names by op id in the later program.

    Returns:
        The differences.

    """
    names_before = names_before or {}
    names_after = names_after or {}
    changes: list[ComputePlacementChange] = []
    unchanged = 0
    unresolved = 0

    for before_id, after_id in sorted(alignment.mapping.items()):
        placed_before = _sorted_devices(before.devices_for_id(before_id))
        placed_after = _sorted_devices(after.devices_for_id(after_id))

        if placed_before == placed_after:
            if placed_before == (ComputeDevice.UNKNOWN,):
                unresolved += 1
            else:
                unchanged += 1
            continue

        changes.append(
            ComputePlacementChange(
                operation_id=before_id,
                before=placed_before,
                after=placed_after,
                name=names_before.get(before_id, "?"),
                modified=before_id in alignment.modified,
                validation_before=_sorted_messages(
                    before.validation_messages_for_id(before_id),
                ),
                validation_after=_sorted_messages(
                    after.validation_messages_for_id(after_id),
                ),
            ),
        )

    return ComputePlanDiff(
        changes=changes,
        unchanged=unchanged,
        unresolved=unresolved,
        only_before=[
            ComputePlacement(
                operation_id=op_id,
                devices=_sorted_devices(before.devices_for_id(op_id)),
                name=names_before.get(op_id, "?"),
                validation=_sorted_messages(before.validation_messages_for_id(op_id)),
            )
            for op_id in sorted(alignment.removed)
        ],
        only_after=[
            ComputePlacement(
                operation_id=op_id,
                devices=_sorted_devices(after.devices_for_id(op_id)),
                name=names_after.get(op_id, "?"),
                validation=_sorted_messages(after.validation_messages_for_id(op_id)),
            )
            for op_id in sorted(alignment.added)
        ],
        alignment=alignment,
    )
