# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Comparing the timing of two runs of a model that changed between them.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TextIO

from coreai.authoring import AIProgram
from coreai.runtime import SpecializationOptions
from typing_extensions import Self

from .benchmarker import (
    BenchmarkResult,
    CoreAIBenchmarker,
    OperationTiming,
    _get_default_excluded_operations,
)
from .graph_diff import OpIdAlignment, op_id_alignment
from .graph_match import WeightPolicy
from .table_writer import _Column, _Row, _TableSpec, _write_table
from .utils import _plain

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchedDispatch:
    """
    One dispatch covering the same operations either side of an edit.

    "Same" means the same set of Core AI operations, and nothing stronger: the two
    may be executed by different kernels, at different shapes, or on a different
    encoder, since kernel selection depends on shapes and on the surrounding graph.
    So a delta says the cost of these operations moved, not that the same code got
    faster.
    """

    before: OperationTiming
    after: OperationTiming

    modified: bool = False
    """Whether any of these operations is the same operation wired or configured
    differently -- ``alignment.modified``. Its delta may be that change rather than a
    change in cost: doubling a matmul's width shows up here, not as an added
    operation."""

    @property
    def median_delta_ms(self) -> float | None:
        """
        After minus before.

        Returns:
            The difference in milliseconds, or None if either side has no
            statistics.

        """
        first = self.before.measurement.statistics
        second = self.after.measurement.statistics
        if first is None or second is None:
            return None
        return second.median - first.median

    def to_dict(self) -> dict[str, Any]:
        """
        Return the pair as plain values.

        :attr:`median_delta_ms` is included because it is the point of the pair and
        is a property, so a caller reading only the fields would have to recompute
        the comparison this class exists to make.

        Returns:
            Both dispatches, the delta between them, and whether the operation itself
            changed.

        """
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "modified": self.modified,
            "median_delta_ms": _plain(self.median_delta_ms),
        }


class Resize(Enum):
    """Whether a dispatch gained or lost operations."""

    GAINED = "gained"
    """The after dispatch is the before dispatch plus the extra operations."""

    LOST = "lost"
    """The before dispatch was the after dispatch plus the extra operations."""


@dataclass(frozen=True)
class ResizedDispatch:
    """
    The same dispatch either side of an edit, with operations gained or lost.

    Its operations on one side exactly contain the other's, so the two durations are
    comparable and the difference is attributable to the extra operations -- with the
    caveat that it also carries any change in what both sides share, and the two
    cannot be separated.

    Reported apart from :class:`MatchedDispatch` because it is not like for like: the
    after run did more work, or less.
    """

    before: OperationTiming
    after: OperationTiming

    extra_op_ids: frozenset[int]
    """Operations on the larger side and absent from the smaller. In the after
    numbering when :attr:`resize` is ``GAINED``; in the before numbering when it is
    ``LOST``, since an operation that was lost has no after counterpart to name it
    by."""

    resize: Resize
    """Which side is larger."""

    modified: bool = False
    """Whether any shared operation is the same operation wired or configured
    differently. See :attr:`MatchedDispatch.modified`."""

    @property
    def median_delta_ms(self) -> float | None:
        """
        After minus before.

        Returns:
            The difference in milliseconds, or None if either side has no statistics.
            Read as the cost of :attr:`extra_op_ids`, in the direction
            :attr:`resize` gives.

        """
        first = self.before.measurement.statistics
        second = self.after.measurement.statistics
        if first is None or second is None:
            return None
        return second.median - first.median

    def to_dict(self) -> dict[str, Any]:
        """
        Return the pair as plain values.

        :attr:`median_delta_ms` is included because it is the point of the pair and
        is a property, so a caller reading only the fields would have to recompute
        the comparison this class exists to make.

        Returns:
            Both dispatches, the delta between them, and whether the operation itself
            changed.

        """
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "modified": self.modified,
            "median_delta_ms": _plain(self.median_delta_ms),
            "extra_op_ids": _plain(self.extra_op_ids),
            "resize": _plain(self.resize),
        }


@dataclass(frozen=True)
class TimingDiff:
    """
    What changed between two benchmark runs.

    The lists partition the dispatches: every dispatch of either run appears exactly
    once -- in :attr:`matched`, in :attr:`resized`, or in the side it is only present
    on. Nothing is dropped, so a total taken from here accounts for everything
    measured.
    """

    matched: list[MatchedDispatch] = field(default_factory=list)
    """Dispatches covering the same operations either side. Comparable."""

    resized: list[ResizedDispatch] = field(default_factory=list)
    """Dispatches whose operations one side exactly contains: the same dispatch,
    having gained or lost some. Comparable, but not like for like."""

    only_before: list[OperationTiming] = field(default_factory=list)
    """Dispatches present before and not after -- because their operations are gone,
    or because those operations are now fused differently. Either way this dispatch,
    the unit the hardware measured, has no counterpart to compare against."""

    only_after: list[OperationTiming] = field(default_factory=list)
    """Dispatches present after and not before."""

    alignment: OpIdAlignment | None = None
    """The operation correspondence this comparison was built on."""

    @property
    def before_dispatches(self: Self) -> list[OperationTiming]:
        """
        Every dispatch the before run measured.

        The paired dispatches together with the rest, since a side is split across
        two of the lists above. Reading :attr:`only_before` alone as "the before
        dispatches" undercounts by however much did not change, which is usually
        most of it.

        Returns:
            The before run's dispatches.

        """
        return (
            [pair.before for pair in self.matched]
            + [resized.before for resized in self.resized]
            + self.only_before
        )

    @property
    def after_dispatches(self: Self) -> list[OperationTiming]:
        """
        Every dispatch the after run measured.

        Returns:
            The after run's dispatches.

        """
        return (
            [pair.after for pair in self.matched]
            + [resized.after for resized in self.resized]
            + self.only_after
        )

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Return the comparison as plain values.

        The dispatch *counts* per side are included because they are the denominator:
        "10 only-before" means nothing without knowing whether the run had 12
        dispatches or 200, and both sides are split across three of the lists.

        Returns:
            Every dispatch of both runs, classified, and the correspondence the
            comparison was built on.

        """
        return {
            "matched": [pair.to_dict() for pair in self.matched],
            "resized": [entry.to_dict() for entry in self.resized],
            "only_before": [timing.to_dict() for timing in self.only_before],
            "only_after": [timing.to_dict() for timing in self.only_after],
            "before_dispatch_count": len(self.before_dispatches),
            "after_dispatch_count": len(self.after_dispatches),
            "alignment": self.alignment.to_dict() if self.alignment else None,
        }

    def write_to(self: Self, output: TextIO, *, width: int | None = None) -> None:
        """
        Write the comparison as a table.

        Args:
            output: Text stream to write to.
            width: Console width to render at. Defaults to the table writer's own.

        """
        spec = _TableSpec(
            title="Timing comparison",
            columns=(
                _Column("Status"),
                _Column("Operations"),
                _Column("Before (ms)", justify="right"),
                _Column("After (ms)", justify="right"),
                _Column("Delta (ms)", justify="right"),
            ),
            caption=(
                f"{len(self.matched)} comparable "
                f"({sum(p.modified for p in self.matched)} carrying a modified "
                f"operation), {len(self.resized)} resized, "
                f"{len(self.only_before)} only before, "
                f"{len(self.only_after)} only after. A dispatch present on one side "
                "only has no counterpart to subtract, and no delta here is calibrated "
                "against run-to-run variance."
            ),
            show_lines=True,
        )

        for pair in sorted(self.matched, key=lambda p: -abs(p.median_delta_ms or 0.0)):
            before = pair.before.measurement.statistics
            after = pair.after.measurement.statistics
            delta = pair.median_delta_ms
            spec.add(
                _Row(
                    cells=(
                        "paired (modified)" if pair.modified else "paired",
                        "\n".join(pair.before.op_names),
                        f"{before.median:.6f}" if before else "N/A",
                        f"{after.median:.6f}" if after else "N/A",
                        f"{delta:+.6f}" if delta is not None else "N/A",
                    ),
                ),
            )

        for resized in sorted(
            self.resized, key=lambda r: -abs(r.median_delta_ms or 0.0)
        ):
            before = resized.before.measurement.statistics
            after = resized.after.measurement.statistics
            delta = resized.median_delta_ms
            ids = ", ".join(str(op_id) for op_id in sorted(resized.extra_op_ids))
            sign = "+" if resized.resize is Resize.GAINED else "-"
            spec.add(
                _Row(
                    cells=(
                        f"{resized.resize.value} {len(resized.extra_op_ids)} op(s)"
                        + (" (modified)" if resized.modified else ""),
                        "\n".join(resized.before.op_names) + f"\n{sign} ids {ids}",
                        f"{before.median:.6f}" if before else "N/A",
                        f"{after.median:.6f}" if after else "N/A",
                        f"{delta:+.6f}" if delta is not None else "N/A",
                    ),
                ),
            )

        for timing, status in (
            *((t, "only before") for t in self.only_before),
            *((t, "only after") for t in self.only_after),
        ):
            statistics = timing.measurement.statistics
            median = f"{statistics.median:.6f}" if statistics else "N/A"
            spec.add(
                _Row(
                    cells=(
                        status,
                        "\n".join(timing.op_names),
                        median if status == "only before" else "--",
                        median if status == "only after" else "--",
                        "--",
                    ),
                ),
            )

        _write_table(spec, output, width=width)


@dataclass(frozen=True)
class _Projection:
    """A dispatch's operations in the after numbering, and what did not survive."""

    mapped: frozenset[int]
    """Counterparts in the after numbering."""

    unmapped: frozenset[int]
    """Before ids with no counterpart at all: operations the edit removed."""


def _project(timing: OperationTiming, mapping: dict[int, int]) -> _Projection:
    """
    Translate a before dispatch's operations into the after run's numbering.

    Core AI op ids are positional, so an edit renumbers them and the two runs have to
    be compared in one numbering. What failed to translate is kept rather than
    dropped: a dispatch whose operations all survive can pair exactly, and one that
    lost some can still be recognised as the same dispatch having lost them.

    Args:
        timing: A dispatch from the before run.
        mapping: Before op id -> after op id, from :func:`op_id_alignment`.

    Returns:
        The translated operations, and those with no counterpart.

    """
    mapped = {mapping[op_id] for op_id in timing.op_ids if op_id in mapping}
    unmapped = {op_id for op_id in timing.op_ids if op_id not in mapping}
    return _Projection(mapped=frozenset(mapped), unmapped=frozenset(unmapped))


def _unique_by_set(
    op_sets: "list[frozenset[int] | None]",
) -> dict[frozenset[int], int]:
    """
    Map each set of operations to the one dispatch covering it.

    Args:
        op_sets: One entry per dispatch, in dispatch order.

    Returns:
        Sets covered by exactly one dispatch, mapped to its index. A set covered by
        more than one is dropped, since two distinct dispatches can cover the same
        operations and pairing either with a counterpart would be an arbitrary
        choice. Absent and empty sets are dropped too -- otherwise every dispatch
        that mapped to nothing would match every other one.

        The refusal is silent, which `TimingDiff.only_before`'s docstring warns
        about: a dispatch dropped here is reported alongside genuinely absent ones.

    """
    by_set: dict[frozenset[int], list[int]] = defaultdict(list)
    for index, op_set in enumerate(op_sets):
        if op_set:
            by_set[op_set].append(index)
    return {
        op_set: indices[0] for op_set, indices in by_set.items() if len(indices) == 1
    }


@dataclass(frozen=True)
class _Correspondence:
    """Which after dispatch a before dispatch became, and how."""

    after_index: int
    """Index into the after run's dispatches."""

    resize: Resize | None = None
    """None when both cover the same operations; otherwise which way it changed."""

    extra_op_ids: frozenset[int] = frozenset()
    """Operations on the larger side, when :attr:`resize` is set."""


@dataclass(frozen=True)
class _CorrespondenceIndex:
    """
    What a before dispatch's counterpart is looked up in.

    Named for the lookup rather than for the after run, since :attr:`added_op_ids`
    comes from the alignment rather than from that run.
    """

    sets: list[frozenset[int]]
    """Each after dispatch's operations, by index."""

    unique: dict[frozenset[int], int]
    """Operations covered by exactly one after dispatch, to its index."""

    added_op_ids: frozenset[int]
    """Operations the edit introduced, so a gain can be told from a regrouping."""


def _correspond(
    projection: _Projection,
    correspondence_index: _CorrespondenceIndex,
    claimed: set[int],
) -> _Correspondence | None:
    """
    Find the after dispatch a before dispatch became.

    Args:
        projection: The before dispatch's operations in the after numbering, and those
            with no counterpart.
        correspondence_index: What the counterpart is looked up in.
        claimed: After dispatches already corresponded to something.

    Returns:
        The correspondence, or None when nothing corresponds -- which is the answer
        whenever taking one would mean comparing different amounts of work.

    """
    # The same operations, or the same minus the ones this dispatch lost.
    after_index = correspondence_index.unique.get(projection.mapped)
    if after_index is not None:
        if projection.unmapped:
            return _Correspondence(after_index, Resize.LOST, projection.unmapped)
        return _Correspondence(after_index)

    # Operations lost *and* nothing covering what remains: guessing which dispatch
    # absorbed it would compare different work.
    if projection.unmapped:
        return None

    # Exactly one dispatch strictly containing these operations. Two candidates would
    # make the choice between them arbitrary.
    containing = [
        index
        for index, op_set in enumerate(correspondence_index.sets)
        if index not in claimed and op_set > projection.mapped
    ]
    if len(containing) != 1:
        return None

    # And the extra operations have to be new. Ones that existed before and merely
    # moved into this dispatch are a regrouping, not a gain: calling them gained
    # blames the edit for work it did not add, and counts them twice when the dispatch
    # they came from corresponded elsewhere.
    extra_op_ids = correspondence_index.sets[containing[0]] - projection.mapped
    if not extra_op_ids <= correspondence_index.added_op_ids:
        return None
    return _Correspondence(containing[0], Resize.GAINED, extra_op_ids)


def compare_results(
    alignment: OpIdAlignment,
    before: BenchmarkResult,
    after: BenchmarkResult,
) -> TimingDiff:
    """
    Compare two runs, given the operation correspondence between them.

    Pure, and deliberately so: pairing is where this can be subtly wrong, and keeping
    it free of programs and devices is what makes it testable.

    A dispatch pairs only with one covering the same operations. Anything else is
    reported as present on one side only rather than compared, because a dispatch
    whose composition changed measures a different amount of work -- comparing a
    six-operation kernel against the two three-operation kernels it became would
    report a change of denominator as a change in cost.

    Args:
        alignment: Which before operation became which after operation, from
            :func:`op_id_alignment`.
        before: The "before" run.
        after: The "after" run.

    Returns:
        What changed, as a partition of both runs' dispatches.

    """
    projections = [
        _project(timing, alignment.mapping) for timing in before.operation_timings
    ]
    after_sets = [frozenset(timing.op_ids) for timing in after.operation_timings]

    # Indexed by operations covered, since a dispatch has no other stable identity: a
    # compile identifier is a per-dispatch counter, and op ids renumber across an edit.
    before_unique = _unique_by_set([projection.mapped for projection in projections])
    correspondence_index = _CorrespondenceIndex(
        sets=after_sets,
        unique=_unique_by_set(after_sets),
        added_op_ids=frozenset(alignment.added),
    )

    matched: list[MatchedDispatch] = []
    resized: list[ResizedDispatch] = []
    paired_before: set[int] = set()
    paired_after: set[int] = set()

    # Iterated in before-dispatch order, so the outcome does not depend on how a
    # frozenset happens to hash.
    for index in before_unique.values():
        found = _correspond(projections[index], correspondence_index, paired_after)
        if found is None:
            continue

        first = before.operation_timings[index]
        second = after.operation_timings[found.after_index]
        modified = bool(set(first.op_ids) & alignment.modified)

        if found.resize is None:
            matched.append(MatchedDispatch(first, second, modified))
        else:
            resized.append(
                ResizedDispatch(
                    first, second, found.extra_op_ids, found.resize, modified
                ),
            )
        paired_before.add(index)
        paired_after.add(found.after_index)

    return TimingDiff(
        matched=matched,
        resized=resized,
        only_before=[
            timing
            for index, timing in enumerate(before.operation_timings)
            if index not in paired_before
        ],
        only_after=[
            timing
            for index, timing in enumerate(after.operation_timings)
            if index not in paired_after
        ],
        alignment=alignment,
    )


def _no_dispatches_message(label: str, result: BenchmarkResult) -> str:
    """
    Say that a run attributed nothing, and what it measured instead.

    Reports counts, not a cause. The message this replaced named one -- "per-operation
    timing needs the GPU delegate" -- which was false and sent the caller to change a
    setting that was not the problem. Nothing available here distinguishes the possible
    causes, so nothing here claims to.

    Args:
        label: Which run it was, "before" or "after".
        result: The run that attributed nothing.

    Returns:
        The message for the :class:`ValueError`.

    """
    if result.unattributed_samples:
        measured = (
            f"{result.unattributed_samples} sample(s) were measured but named no "
            "operation in this program"
        )
    elif result.symbol_samples:
        measured = (
            f"{result.symbol_samples} sample(s) measured whole regions rather than "
            "individual operations"
        )
    else:
        measured = "no timing was reported at all"

    return (
        f"The {label} run attributed no dispatches, so there is nothing to compare: "
        f"{measured}."
    )


async def compare_runs(
    before_program: AIProgram,
    after_program: AIProgram,
    inputs: dict[str, Any],
    num_runs: int = 1,
    entry_point: str = "main",
    *,
    excluded_operations: tuple[str, ...] | None = None,
    specialization_options: SpecializationOptions | None = None,
    weights: WeightPolicy = WeightPolicy.IGNORE,
) -> TimingDiff:
    """
    Benchmark both programs under identical settings and compare them.

    One call rather than two benchmarks a caller pairs up, because a difference in
    iteration count, excluded operations or specialization options can dwarf the edit
    being measured. The runs are still sequential, so whatever drifts over the call --
    clocks, thermals, other load -- lands on the second one.

    Args:
        before_program: The "before" program.
        after_program: The "after" program.
        inputs: Inputs for both, so the edit is assumed not to have changed them.
        num_runs: Timed iterations per program, each preceded by an untimed warmup so
            the samples describe the steady state either side. Worth raising: every
            number reported here is a difference of two measurements, so it carries
            both their noise, and a single sample apiece leaves no spread to judge
            whether a delta means anything.
        entry_point: Function to benchmark and compare.
        excluded_operations: Operations not to time, defaulting as
            :func:`~coreai_torch.debugging.benchmarker.benchmark_coreai_program` does.
            Applied to both runs. Not merely cosmetic here: a dispatch is identified by
            the operations it covers, so an untimed operation left in that set makes
            pairing turn on it, and a constant the edit happened to add would strand an
            otherwise unchanged dispatch on one side.
        specialization_options: Specialization options. Applied to both runs.
        weights: Whether parameter values count towards an operation's identity.

    Returns:
        What changed between the two runs.

    Raises:
        ValueError: If either run attributed no dispatches. Comparing two empty runs
            reports that nothing changed, which reads exactly like a working
            comparison.

    """
    alignment = op_id_alignment(
        before_program, after_program, entry_point, weights=weights
    )

    if excluded_operations is None:
        excluded_operations = _get_default_excluded_operations()

    results: list[BenchmarkResult] = []
    for program in (before_program, after_program):
        benchmarker = CoreAIBenchmarker(
            program, entry_point, excluded_operations, specialization_options
        )
        results.append(await benchmarker.benchmark(inputs, num_runs))

    for label, result in zip(("before", "after"), results, strict=True):
        if not result.operation_timings:
            raise ValueError(_no_dispatches_message(label, result))

    if alignment.identical:
        logger.info(
            "The two programs are the same graph, so any difference measured here is "
            "run-to-run variance rather than the effect of an edit.",
        )

    return compare_results(alignment, *results)
