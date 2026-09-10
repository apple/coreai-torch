# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Tests for comparing two benchmark runs.

Pairing is exercised without a device: it reads only ``op_ids`` and the alignment, so
a dispatch can be built with no operations behind it. That keeps the classification
-- the part that can be subtly wrong -- testable.
"""

from __future__ import annotations

from io import StringIO

import pytest

from coreai_torch.debugging.benchmarker import (
    BenchmarkResult,
    Measurement,
    OperationTiming,
)
from coreai_torch.debugging.graph_diff import OpIdAlignment
from coreai_torch.debugging.timing_diff import (
    Resize,
    _no_dispatches_message,
    compare_results,
)


def _dispatch(op_ids: list[int], median_ms: float) -> OperationTiming:
    """A dispatch covering *op_ids*, measured at *median_ms* every sample."""
    return OperationTiming(
        op_ids=op_ids,
        operations=[],
        measurement=Measurement.from_samples([median_ms] * 3),
    )


def _result(*timings: OperationTiming) -> BenchmarkResult:
    """A benchmark result holding *timings*."""
    return BenchmarkResult(operation_timings=list(timings))


def _identity(op_ids: range | list[int]) -> OpIdAlignment:
    """An alignment where every operation kept its id."""
    return OpIdAlignment(mapping={op_id: op_id for op_id in op_ids}, identical=True)


def test_unchanged_dispatches_pair() -> None:
    """Identical dispatches pair, and the delta is zero."""
    before = _result(_dispatch([1, 2], 0.5), _dispatch([3], 0.25))
    after = _result(_dispatch([1, 2], 0.5), _dispatch([3], 0.25))

    diff = compare_results(_identity(range(1, 4)), before, after)

    assert len(diff.matched) == 2
    assert not diff.only_before
    assert not diff.only_after
    assert all(pair.median_delta_ms == 0.0 for pair in diff.matched)


def test_delta_is_after_minus_before() -> None:
    """A dispatch that got slower reports a positive delta."""
    before = _result(_dispatch([1, 2], 0.5))
    after = _result(_dispatch([1, 2], 0.75))

    diff = compare_results(_identity(range(1, 3)), before, after)

    assert len(diff.matched) == 1
    assert diff.matched[0].median_delta_ms == 0.25


def test_split_fusion_does_not_pair() -> None:
    """A kernel that split covers different operations, so nothing is compared.

    Pairing (1,2,3) with either half would compare a three-operation dispatch
    against a two-operation one and report the change of denominator as a cost.
    """
    before = _result(_dispatch([1, 2, 3], 0.9))
    after = _result(_dispatch([1, 2], 0.5), _dispatch([3, 4], 0.5))

    diff = compare_results(
        OpIdAlignment(mapping={1: 1, 2: 2, 3: 3}, added=[4]), before, after
    )

    assert not diff.matched
    assert [t.op_ids for t in diff.only_before] == [[1, 2, 3]]
    assert [t.op_ids for t in diff.only_after] == [[1, 2], [3, 4]]


def test_removed_operations_leave_a_dispatch_on_one_side() -> None:
    """A dispatch whose operations are gone has no counterpart."""
    before = _result(_dispatch([1, 2], 0.5), _dispatch([9], 0.1))
    after = _result(_dispatch([1, 2], 0.5))

    diff = compare_results(
        OpIdAlignment(mapping={1: 1, 2: 2}, removed=[9]), before, after
    )

    assert len(diff.matched) == 1
    assert [t.op_ids for t in diff.only_before] == [[9]]
    assert not diff.only_after


def test_renumbered_operations_still_pair() -> None:
    """An edit that renumbers ids pairs on the correspondence, not on the ids.

    This is the case raw ids get wrong: comparing by id would find no dispatch at
    (11, 12) before and report the whole model as replaced.
    """
    before = _result(_dispatch([1, 2], 0.5))
    after = _result(_dispatch([11, 12], 0.5))

    diff = compare_results(OpIdAlignment(mapping={1: 11, 2: 12}), before, after)

    assert len(diff.matched) == 1
    assert diff.matched[0].before.op_ids == [1, 2]
    assert diff.matched[0].after.op_ids == [11, 12]


def test_a_dispatch_that_lost_an_operation_is_resized_not_matched() -> None:
    """A dispatch missing an operation is the same dispatch, resized.

    (1,2,9) with 9 removed leaves (1,2), which the after dispatch covers exactly. It
    is not a like-for-like pair -- the before run did more work -- so it is reported
    as a loss of operation 9 rather than as matched, and the delta reads as the cost
    of that operation.
    """
    before = _result(_dispatch([1, 2, 9], 0.9))
    after = _result(_dispatch([1, 2], 0.5))

    diff = compare_results(
        OpIdAlignment(mapping={1: 1, 2: 2}, removed=[9]), before, after
    )

    assert not diff.matched, "Not the same operations"
    assert len(diff.resized) == 1
    resized = diff.resized[0]
    assert resized.resize is Resize.LOST
    assert resized.extra_op_ids == frozenset({9})
    assert resized.median_delta_ms == -0.4
    assert not diff.only_before
    assert not diff.only_after


def test_a_dispatch_that_gained_an_operation_is_resized() -> None:
    """A dispatch with one more operation is the same dispatch, resized.

    The delta is the cost of the operation it gained -- which is the reading an
    author wants after adding one, and which reporting both sides as unpaired denies
    them.
    """
    before = _result(_dispatch([1, 2], 0.5))
    after = _result(_dispatch([1, 2, 3], 0.7))

    diff = compare_results(
        OpIdAlignment(mapping={1: 1, 2: 2}, added=[3]), before, after
    )

    assert not diff.matched
    assert len(diff.resized) == 1
    resized = diff.resized[0]
    assert resized.resize is Resize.GAINED
    assert resized.extra_op_ids == frozenset({3})
    assert resized.median_delta_ms == pytest.approx(0.2)


def test_operations_that_merely_moved_are_not_a_gain() -> None:
    """A dispatch absorbing an operation that already existed is not a gain.

    Operation 1 exists before, in its own dispatch. After the edit it sits alongside 2
    in one dispatch. Calling that "gained 1" blames the edit for work it did not add,
    and counts operation 1 twice -- once in its own pair, once as the gain.
    """
    before = _result(_dispatch([1], 0.2), _dispatch([2], 0.5))
    after = _result(_dispatch([1], 0.2), _dispatch([1, 2], 0.7))

    diff = compare_results(_identity([1, 2]), before, after)

    assert not diff.resized, "Operation 1 was not added by the edit"
    assert [t.op_ids for t in diff.only_before] == [[2]]
    assert [t.op_ids for t in diff.only_after] == [[1, 2]]
    # The dispatch that did not change still pairs.
    assert [p.before.op_ids for p in diff.matched] == [[1]]


def test_two_candidate_supersets_are_not_resized_arbitrarily() -> None:
    """Two dispatches could contain these operations, so neither is chosen."""
    before = _result(_dispatch([1, 2], 0.5))
    after = _result(_dispatch([1, 2, 3], 0.7), _dispatch([1, 2, 4], 0.7))

    diff = compare_results(
        OpIdAlignment(mapping={1: 1, 2: 2}, added=[3, 4]), before, after
    )

    assert not diff.matched
    assert not diff.resized, "Neither superset is more correct than the other"
    assert len(diff.only_before) == 1
    assert len(diff.only_after) == 2


def test_ambiguous_sets_are_not_paired_arbitrarily() -> None:
    """Two dispatches covering the same operations cannot be paired one to one."""
    before = _result(_dispatch([1], 0.5), _dispatch([1], 0.6))
    after = _result(_dispatch([1], 0.5), _dispatch([1], 0.6))

    diff = compare_results(_identity([1]), before, after)

    assert not diff.matched, "Neither pairing is more correct than the other"
    assert len(diff.only_before) == 2
    assert len(diff.only_after) == 2


def test_a_modified_operation_marks_the_pair() -> None:
    """A pair holding a rewired operation is flagged, not silently compared.

    `align` folds a modified operation into `mapping` so its dispatch still pairs --
    dropping it would discard most of the comparison, since any reshape produces one.
    But its delta may be the reconfiguration rather than a change in cost.
    """
    before = _result(_dispatch([1, 2], 0.5), _dispatch([3], 0.2))
    after = _result(_dispatch([1, 2], 0.6), _dispatch([3], 0.2))

    diff = compare_results(
        OpIdAlignment(mapping={1: 1, 2: 2, 3: 3}, modified={2}), before, after
    )

    by_ids = {tuple(pair.before.op_ids): pair for pair in diff.matched}
    assert by_ids[(1, 2)].modified, "Operation 2 was modified"
    assert not by_ids[(3,)].modified, "Operation 3 was not"


def test_dispatches_are_partitioned() -> None:
    """Every dispatch appears exactly once, so nothing is silently dropped."""
    before = _result(_dispatch([1, 2], 0.5), _dispatch([3], 0.2), _dispatch([9], 0.1))
    after = _result(_dispatch([1, 2], 0.4), _dispatch([3, 4], 0.3))

    diff = compare_results(
        OpIdAlignment(mapping={1: 1, 2: 2, 3: 3}, removed=[9], added=[4]),
        before,
        after,
    )

    assert len(diff.matched) + len(diff.resized) + len(diff.only_before) == len(
        before.operation_timings
    )
    assert len(diff.matched) + len(diff.resized) + len(diff.only_after) == len(
        after.operation_timings
    )


def test_side_accessors_return_the_whole_side() -> None:
    """Each side lists every dispatch that run measured, paired ones included.

    Reading ``only_before`` alone would undercount by whatever did not change, which
    is the flattering direction: an edit would look like it removed dispatches it
    left alone.
    """
    before = _result(_dispatch([1, 2], 0.5), _dispatch([3], 0.2), _dispatch([9], 0.1))
    after = _result(_dispatch([1, 2], 0.4), _dispatch([3], 0.2), _dispatch([4], 0.3))

    diff = compare_results(
        OpIdAlignment(mapping={1: 1, 2: 2, 3: 3}, removed=[9], added=[4]),
        before,
        after,
    )

    assert len(diff.matched) == 2, "The unchanged dispatches should pair"
    assert diff.before_dispatches == before.operation_timings
    assert diff.after_dispatches == after.operation_timings
    # And the accessors are the partition, reassembled.
    assert len(diff.before_dispatches) == len(diff.matched) + len(diff.resized) + len(
        diff.only_before
    )
    assert len(diff.after_dispatches) == len(diff.matched) + len(diff.resized) + len(
        diff.only_after
    )


def test_write_to_reports_every_dispatch() -> None:
    """The table names each dispatch, whichever side it is on."""
    before = _result(_dispatch([1, 2], 0.5), _dispatch([9], 0.1))
    after = _result(_dispatch([1, 2], 0.4), _dispatch([3], 0.2))

    diff = compare_results(
        OpIdAlignment(mapping={1: 1, 2: 2}, removed=[9], added=[3]), before, after
    )
    buffer = StringIO()
    diff.write_to(buffer, width=120)
    written = buffer.getvalue()

    assert "paired" in written
    assert "only before" in written
    assert "only after" in written
    assert "1 comparable" in written


def test_an_empty_run_is_explained_by_what_it_measured() -> None:
    """
    Three ways to attribute nothing, and the message says which without guessing why.

    The message this replaced named a cause -- "per-operation timing needs the GPU
    delegate" -- which is false, since the bundled runtime attributes dispatches with
    no specialization options at all, and which told a caller already on the GPU
    delegate to select it. Nothing at this point distinguishes the causes, so the
    message reports what was measured and stops.
    """
    named_nothing = _no_dispatches_message(
        "before",
        BenchmarkResult(
            operation_timings=[],
            unattributed_samples=9,
            unattributed_compile_ids=3,
        ),
    )
    assert "9 sample(s) were measured but named no operation" in named_nothing

    whole_regions = _no_dispatches_message(
        "after",
        BenchmarkResult(operation_timings=[], symbol_samples=9, symbol_intervals=3),
    )
    assert "measured whole regions rather than individual operations" in whole_regions

    nothing = _no_dispatches_message("before", BenchmarkResult(operation_timings=[]))
    assert "no timing was reported at all" in nothing

    for message in (named_nothing, whole_regions, nothing):
        assert "attributed no dispatches" in message
        assert "GPU delegate" not in message, "It must not name a cause it cannot know"
