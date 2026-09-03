# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Integration tests for `plan_diff`, against real programs and real compute plans.
"""

import io
import sys

import pytest
import torch
from coreai.authoring import AIProgram

from coreai_torch.converter import TorchConverter
from coreai_torch.debugging.compute_plan import ComputeDevice, ComputePlan
from coreai_torch.debugging.graph_diff import OpIdAlignment
from coreai_torch.debugging.graph_match import WeightPolicy
from coreai_torch.debugging.plan_diff import (
    ComputePlacementChange,
    compare_compute_plans,
    diff_compute_plans,
)

from .test_model import (
    ExtraLayerModel,
    ModifiedActivationModel,
    ThreeLinearModel,
    get_example_inputs,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Compute plans are only available on macOS"
)


def _program(model_cls: type[torch.nn.Module]) -> AIProgram:
    """Convert *model_cls* to a debuggable program, at `ThreeLinearModel`'s input."""
    args = tuple(get_example_inputs(ThreeLinearModel).values())
    exported = torch.export.export(model_cls().eval(), args).run_decompositions()
    converter = TorchConverter(mode=TorchConverter.Mode.DEBUG)
    converter.add_exported_program(exported, entrypoint_name="main")
    return converter.to_coreai()


# `DIGEST` throughout: `IGNORE` cannot tell two same-shaped layers apart, and these
# models are three identical `Linear`s. See `compare_compute_plans`' own docstring.
async def _diff(before: AIProgram, after: AIProgram):  # noqa: ANN202
    """Compare two programs' plans under the planner a plain run gets."""
    return await compare_compute_plans(before, after, weights=WeightPolicy.DIGEST)


# ---------------------------------------------------------------------------
# Added and removed
# ---------------------------------------------------------------------------


async def test_added_operations_are_reported_with_where_they_landed() -> None:
    """
    An added operation has no counterpart, so it cannot be a `change`.

    Dropping it would hide a whole layer, and the question being asked of a diff is
    usually exactly where the new work went. `only_after` must therefore hold every
    operation the alignment calls added, and no others -- an operation reported as
    added that the alignment paired would be a placement compared against nothing.
    """
    before, after = _program(ThreeLinearModel), _program(ExtraLayerModel)

    diff = await _diff(before, after)
    assert diff.alignment is not None

    assert diff.only_after, "ExtraLayerModel adds a layer"
    assert {placement.operation_id for placement in diff.only_after} == set(
        diff.alignment.added
    ), "only_after is exactly the alignment's added operations"
    assert not diff.only_before, "Nothing was removed in this direction"


async def test_removed_operations_are_reported_with_where_they_used_to_run() -> None:
    """
    The mirror image, and it has to be one: comparing the same two programs the other
    way round must move the same operations from `only_after` to `only_before`.

    Their placement has no counterpart either, but it is what makes the removal
    readable -- an operation that was on the neural engine and is now gone is a
    different edit from one that was on the CPU.
    """
    three, extra = _program(ThreeLinearModel), _program(ExtraLayerModel)

    forwards = await _diff(three, extra)
    backwards = await _diff(extra, three)
    assert backwards.alignment is not None

    assert {placement.operation_id for placement in backwards.only_before} == set(
        backwards.alignment.removed
    )
    assert not backwards.only_after
    assert {p.operation_id for p in backwards.only_before} == {
        p.operation_id for p in forwards.only_after
    }, "The same operations, reported on whichever side holds them"


async def test_an_added_operation_is_named_from_its_own_program() -> None:
    """
    The two numberings overlap, so a merged name table names an added operation after
    whatever held its id in the *other* program.

    Not hypothetical: comparing `ThreeLinearModel` with `ExtraLayerModel`, operation 20
    is added and is `coreai.relu`, while id 20 in the earlier program is
    `coreai.graph`. Merging would report the added operation as a graph.
    """
    before, after = _program(ThreeLinearModel), _program(ExtraLayerModel)

    diff = await _diff(before, after)

    names = {placement.operation_id: placement.name for placement in diff.only_after}
    assert names, "There are added operations to name"
    assert "?" not in names.values(), "Every added operation is named"
    assert "coreai.graph" not in names.values(), (
        "A graph is not an operation an edit adds -- that name comes from the other "
        "program's numbering"
    )


async def test_an_operation_swapped_for_another_is_added_and_removed() -> None:
    """
    Replacing an activation is one operation gone and a different one in its place, at
    the same id. Both sides must be reported: the id alone says nothing changed.
    """
    before, after = _program(ThreeLinearModel), _program(ModifiedActivationModel)

    diff = await _diff(before, after)
    assert diff.alignment is not None

    assert diff.only_before and diff.only_after, "One out, one in"
    assert {p.operation_id for p in diff.only_before} == set(diff.alignment.removed)
    assert {p.operation_id for p in diff.only_after} == set(diff.alignment.added)


# ---------------------------------------------------------------------------
# Modified
# ---------------------------------------------------------------------------


async def _placed_and_unplaced(program: AIProgram) -> tuple[ComputePlan, int, int]:
    """A plan, an operation it placed, and an id it holds no entry for."""
    plan = await ComputePlan.from_program(program)
    placed = [
        op_id
        for op_id in sorted(plan._coreai_id_to_compute_info_map)
        if ComputeDevice.UNKNOWN not in plan.devices_for_id(op_id)
    ]
    assert placed, "The planner placed nothing, so there is no placement to compare"
    unplaced = next(
        op_id
        for op_id in range(max(plan._coreai_id_to_compute_info_map) + 2)
        if ComputeDevice.UNKNOWN in plan.devices_for_id(op_id)
    )
    return plan, placed[0], unplaced


async def test_a_move_onto_a_modified_operation_is_flagged_as_such() -> None:
    """
    A modified operation moving may be the edit doing what it asked for.

    Widening a matmul can move it off a device, and reporting that as a planner
    decision blames the planner for the edit. The flag is what lets a reader separate
    the two, so it has to follow `alignment.modified` rather than being assumed.

    Reached through `diff_compute_plans` because a single planner places both sides
    alike, so `compare_compute_plans` reports no change at all for any pair of these
    models. The plan and its placements are real; the alignment is the caller's input,
    and pairing a placed operation with one the plan holds no entry for is the
    documented way a placement stops resolving.
    """
    program = _program(ThreeLinearModel)
    plan, placed, unplaced = await _placed_and_unplaced(program)

    diff = diff_compute_plans(
        plan,
        plan,
        OpIdAlignment(mapping={placed: unplaced}, modified={placed}),
    )

    assert len(diff.changes) == 1, "A placed operation paired with an unplaced one"
    change = diff.changes[0]
    assert change.operation_id == placed
    assert ComputeDevice.UNKNOWN not in change.before
    assert change.after == (ComputeDevice.UNKNOWN,), (
        "The later plan stopped resolving a placement the earlier one had"
    )
    assert change.modified, "The alignment called this operation modified"


async def test_a_move_the_edit_did_not_ask_for_is_not_flagged() -> None:
    """
    The contrast that makes the flag mean anything: the same move, on an operation the
    alignment did not call modified, must come back unflagged.
    """
    program = _program(ThreeLinearModel)
    plan, placed, unplaced = await _placed_and_unplaced(program)

    diff = diff_compute_plans(plan, plan, OpIdAlignment(mapping={placed: unplaced}))

    assert len(diff.changes) == 1
    assert not diff.changes[0].modified


def test_a_change_reports_the_reason_the_later_plan_newly_gave() -> None:
    """
    A move off a delegate is only actionable with the reason it was declined, and what
    explains the move is what the later plan says and the earlier one did not.

    Falling back to the earlier plan's messages covers the opposite case: an operation
    that has *stopped* being refused, where the only account of the placement is the
    refusal it no longer gets.
    """
    declined = ComputePlacementChange(
        operation_id=1,
        before=(ComputeDevice.CPU,),
        after=(ComputeDevice.CPU,),
        validation_before=("already known",),
        validation_after=("already known", "Incompatible element type for ANE"),
    )
    assert declined.reason == ("Incompatible element type for ANE",), (
        "Only what the move introduced"
    )

    accepted = ComputePlacementChange(
        operation_id=2,
        before=(ComputeDevice.CPU,),
        after=(ComputeDevice.CPU,),
        validation_before=("Incompatible element type for ANE",),
    )
    assert accepted.reason == ("Incompatible element type for ANE",), (
        "The earlier plan's account, when the later one gives none"
    )

    assert not ComputePlacementChange(
        operation_id=3, before=(ComputeDevice.CPU,), after=(ComputeDevice.GPU,)
    ).reason


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


async def test_every_paired_operation_is_accounted_for_exactly_once() -> None:
    """
    The partition `ComputePlanDiff` documents. A reader who sums the fields has to get
    the whole program back, or a placement change goes missing rather than wrong.
    """
    before, after = _program(ThreeLinearModel), _program(ExtraLayerModel)

    diff = await _diff(before, after)
    assert diff.alignment is not None

    assert len(diff.changes) + diff.unchanged + diff.unresolved == len(
        diff.alignment.mapping
    )
    changed = {change.operation_id for change in diff.changes}
    assert len(changed) == len(diff.changes), "An operation moves once"
    assert not changed & {p.operation_id for p in diff.only_before}


async def test_unresolved_is_counted_apart_from_unchanged() -> None:
    """
    Both sides reading `UNKNOWN` is agreement in form only.

    Constants and structural operations are never placed, so folding them into
    `unchanged` claims coverage of operations no plan spoke about -- measured at 13 of
    22 paired operations here, which is most of what a reader would take as "checked
    and identical".
    """
    before, after = _program(ThreeLinearModel), _program(ExtraLayerModel)

    diff = await _diff(before, after)
    assert diff.alignment is not None

    assert diff.unresolved > 0, "These models have constants no plan places"
    assert diff.unchanged > 0, "And operations that are genuinely placed alike"


async def test_a_program_against_itself_reports_no_placement_change() -> None:
    """
    The control. Planning one program twice must agree with itself, or every
    difference this module reports is noise of unknown size.
    """
    program = _program(ThreeLinearModel)

    diff = await _diff(program, program)

    assert not diff.changes, "One program, planned twice, placed differently"
    assert not diff.only_before and not diff.only_after
    assert diff.unchanged > 0, "And something was actually placed"


async def test_the_written_report_accounts_for_what_it_compared() -> None:
    """
    The caption is the only place the counts appear together, and a reader deciding
    whether to trust "0 changes" needs the denominator beside it.
    """
    before, after = _program(ThreeLinearModel), _program(ExtraLayerModel)
    diff = await _diff(before, after)

    output = io.StringIO()
    diff.write_to(output, width=200)

    # Wrapped to the table width, so matched on its words rather than as printed.
    caption = " ".join(output.getvalue().split())
    assert f"{diff.unresolved} unresolved" in caption
    assert f"{len(diff.only_after)} only after" in caption
    assert f"{len(diff.changes)} change(s) of" in caption
