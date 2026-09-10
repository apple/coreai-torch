# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Every report type must survive `json.dumps` without losing what it concluded.
"""

import dataclasses
import json
import sys

import pytest
import torch
from coreai.authoring import AIProgram

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder
from coreai_torch.debugging.benchmarker import (
    BenchmarkResult,
    Measurement,
    OperationTiming,
)
from coreai_torch.debugging.compute_plan import ComputePlan
from coreai_torch.debugging.graph_diff import (
    compute_coreai_program_diff,
    op_id_alignment,
)
from coreai_torch.debugging.graph_match import WeightPolicy
from coreai_torch.debugging.histogram import operation_histogram
from coreai_torch.debugging.plan_diff import compare_compute_plans
from coreai_torch.debugging.timing_diff import compare_results

from .test_model import ExtraLayerModel, ThreeLinearModel, get_example_inputs
from .test_operation_histogram import _program as _histogram_program

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Compute plans need macOS"
)


def _program(model_cls: type[torch.nn.Module]) -> AIProgram:
    """Convert *model_cls* with the debug info the source view needs."""
    args = tuple(get_example_inputs(ThreeLinearModel).values())
    exported = torch.export.export(model_cls().eval(), args).run_decompositions()
    converter = TorchConverter(mode=TorchConverter.Mode.DEBUG)
    converter._debug_info_recorder.config = _DebugInfoRecorder.Config(
        include_stack_trace=True, verify_debuginfo_locations=True
    )
    converter.add_exported_program(exported, entrypoint_name="main")
    return converter.to_coreai()


def _roundtrip(report: object) -> dict:
    """Serialise and parse back, so nothing merely *looks* serialisable."""
    return json.loads(json.dumps(report.to_dict()))


def _roundtrip_with(plan: ComputePlan, program: AIProgram) -> dict:
    """The same, for the one report that needs the program to name anything."""
    return json.loads(json.dumps(plan.to_dict(program)))


def test_asdict_is_not_a_substitute() -> None:
    """
    The reason these methods are hand-written rather than `asdict`.

    Pinned as a test because the obvious review comment is "why not just use
    `dataclasses.asdict`", and the answer is that it raises on the two types that
    matter most.
    """
    before, after = _program(ThreeLinearModel), _program(ExtraLayerModel)
    diff = compute_coreai_program_diff(before, after)

    assert dataclasses.is_dataclass(diff), "It is a dataclass, and that is not enough"
    with pytest.raises(TypeError, match="pickle"):
        dataclasses.asdict(diff)

    # And where asdict does succeed, the result still is not JSON.
    alignment = op_id_alignment(before, after, weights=WeightPolicy.DIGEST)
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(dataclasses.asdict(alignment))

    json.dumps(alignment.to_dict())


def test_a_graph_diff_keeps_what_it_concluded() -> None:
    """
    The graphs are omitted -- they are live IR -- but every conclusion survives.

    Including the identity the comparison was made under. A diff whose `weights` and
    `ignore_attributes` are lost cannot be compared against another run, because the
    two may have been computed under different notions of "the same operation".
    """
    before, after = _program(ThreeLinearModel), _program(ExtraLayerModel)
    diff = compute_coreai_program_diff(before, after)

    data = _roundtrip(diff)

    assert data["is_isomorphic"] is False
    assert data["summary"]["source_node_count"] == diff.summary.source_node_count
    assert data["weights"] == WeightPolicy.IGNORE.value
    assert data["ignore_attributes"] == sorted(diff.ignore_attributes)
    assert data["ambiguity"]["count"] == diff.ambiguity.count
    assert "source_graph" not in data, "Live IR is projected to counts, not serialised"


def test_an_alignment_keeps_its_ambiguity() -> None:
    """`Ambiguity.count` is a property, and it is the figure a reader acts on."""
    before, after = _program(ThreeLinearModel), _program(ExtraLayerModel)
    alignment = op_id_alignment(before, after, weights=WeightPolicy.DIGEST)

    data = _roundtrip(alignment)

    assert data["mapping"] == {str(k): v for k, v in alignment.mapping.items()}
    assert sorted(data["modified"]) == sorted(alignment.modified)
    assert data["ambiguity"]["count"] == alignment.ambiguity.count


async def test_a_plan_diff_keeps_the_reason_a_placement_moved() -> None:
    """
    `ComputePlacementChange.reason` is derived from two message sets.

    Dropping it would leave a caller to reimplement "what the later plan newly said",
    which is the whole content of a placement change.
    """
    before, after = _program(ThreeLinearModel), _program(ExtraLayerModel)
    diff = await compare_compute_plans(before, after, weights=WeightPolicy.DIGEST)

    data = _roundtrip(diff)

    assert data["unresolved"] == diff.unresolved
    assert len(data["only_after"]) == len(diff.only_after)
    assert all("reason" in change for change in data["changes"])
    assert all("devices" in placement for placement in data["only_after"])


def test_a_dispatch_projects_its_live_operations_to_names() -> None:
    """
    `operations` holds live handles; `op_names` is a property over them.

    So the serialisable form is strictly more useful than the field: it carries the
    only part a report needs, and it outlives the program the handles belong to.
    """
    timing = OperationTiming(
        op_ids=[1, 2],
        operations=[],
        measurement=Measurement.from_samples([0.5, 0.25]),
        odix_id=7,
    )

    data = _roundtrip(timing)

    assert data["odix_id"] == 7
    assert data["op_ids"] == [1, 2]
    assert data["representative_op_id"] == 1
    assert data["measurement"]["samples"] == [0.5, 0.25]
    assert "operations" not in data, "Live handles are projected, not serialised"


def test_a_benchmark_result_keeps_the_coverage_denominator() -> None:
    """
    `operations_by_id` is which operations existed; the timings are which were timed.

    Coverage is the ratio, so dropping the denominator would leave the result unable
    to answer the question it is usually read for -- and its values are live handles,
    so only the names can survive.
    """
    result = BenchmarkResult(
        operation_timings=[
            OperationTiming(
                op_ids=[1],
                operations=[],
                measurement=Measurement.from_samples([0.5]),
                odix_id=3,
            )
        ],
        unattributed_samples=9,
        unattributed_compile_ids=3,
        symbol_samples=6,
        symbol_intervals=2,
    )

    data = _roundtrip(result)

    assert len(data["operation_timings"]) == 1
    assert "operation_names" in data
    assert data["unattributed_samples"] == 9
    assert data["symbol_samples"] == 6, (
        "The three ways to attribute nothing must all survive"
    )


def test_a_timing_diff_keeps_the_deltas() -> None:
    """
    `median_delta_ms` is a property on every matched pair, and it is the answer.

    A caller reading only the fields gets four lists of dispatches and no comparison
    at all.
    """
    before = BenchmarkResult(
        operation_timings=[
            OperationTiming(
                op_ids=[1],
                operations=[],
                measurement=Measurement.from_samples([0.50]),
                odix_id=1,
            )
        ]
    )
    after = BenchmarkResult(
        operation_timings=[
            OperationTiming(
                op_ids=[1],
                operations=[],
                measurement=Measurement.from_samples([0.75]),
                odix_id=1,
            )
        ]
    )
    from coreai_torch.debugging.graph_diff import OpIdAlignment

    diff = compare_results(OpIdAlignment(mapping={1: 1}), before, after)

    data = _roundtrip(diff)

    assert len(data["matched"]) == 1
    assert data["matched"][0]["median_delta_ms"] == pytest.approx(0.25)
    assert data["before_dispatch_count"] == 1
    assert data["after_dispatch_count"] == 1


def test_an_operation_histogram_keeps_the_size_asdict_would_drop() -> None:
    """
    The one thing this report's `to_dict` adds over `asdict`.

    `asdict` already yields JSON for this type, so the round-trip alone proves little.
    What it drops is `size`, a property -- and recomputing it means rediscovering that a
    nested body counts once per invocation of each ancestor, invokes included.
    """
    histogram = operation_histogram(_histogram_program())

    data = _roundtrip(histogram)

    assert data["size"] == histogram.size
    composite = data["composites"][0]
    assert composite["histogram"]["size"] == histogram.composites[0].size, (
        "A nested body's total must survive too, not just the program's"
    )
    assert "size" not in dataclasses.asdict(histogram), (
        "If asdict starts carrying size, this method has no reason to exist"
    )


def test_a_non_finite_number_becomes_null_rather_than_invalid_json() -> None:
    """
    `json.dumps` emits a bare `Infinity`, which is not valid JSON and which most
    parsers reject -- so an agent reading the output fails on the one result that
    was trying to tell it something went wrong.
    """
    measurement = Measurement.from_samples([float("inf"), 1.0])

    data = _roundtrip(measurement)

    assert data["statistics"]["maximum"] is None
    assert data["samples"] == [None, 1.0]


async def test_a_compute_plan_says_which_operations_it_holds_nothing_about() -> None:
    """
    Every entry carries its residency, so `{UNKNOWN}` never has to be interpreted.
    """
    program = _program(ThreeLinearModel)
    plan = await ComputePlan.from_program(program)

    data = _roundtrip_with(plan, program)

    assert data["entry_count"] == len(plan._coreai_id_to_compute_info_map)
    assert data["entries"], "This model places something"
    assert all(entry["residency"] == "PLACED" for entry in data["entries"]), (
        "An entry the planner made names a device; NO ENTRY is for ids it lacks"
    )
    assert sum(data["device_histogram"].values()) >= data["entry_count"]

    held = {entry["operation_id"] for entry in data["entries"]}
    assert held.isdisjoint(data["operations_without_entry"])
    assert held | set(data["operations_without_entry"]) == {
        int(op_id) for op_id in data["operation_names"]
    }, "Held plus not-held must be every operation the program has"


async def test_a_compute_plan_without_a_program_still_serialises() -> None:
    """
    The plan is built from debug info and keyed by id; names live on the program.

    So a program is optional rather than required, and its absence costs the names
    and the denominator -- not the entries.
    """
    plan = await ComputePlan.from_program(_program(ThreeLinearModel))

    data = _roundtrip(plan)

    assert set(data) == {"entries", "entry_count", "device_histogram"}
    assert data["entries"]
