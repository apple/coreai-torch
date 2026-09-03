# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Compare every mapped operation's intermediates between a torch graph and a Core AI one.

This answers "where do the two disagree, and on what", over the whole output mapping at once,
rather than searching for a first divergence the way :mod:`comparator` does. It takes captured
intermediates rather than running anything, so a caller decides how the values were obtained
and this stays a pure function of them.

A raw pass/fail count is the wrong report. Most operations that cannot be compared are not
defects -- a fused lowering materialises no value for absorbed operations, and a multi-output
operation's outputs come back in an order that is not stable between compilations -- so each
one is classified and counted separately. Statuses are :class:`Comparator.Status`, reused
rather than re-invented so the two tools describe the same situations with the same words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
from coreai.runtime import AIModel, SpecializationOptions
from typing_extensions import Self

from .comparator import _DEFAULT_ATOL, _DEFAULT_RTOL, Comparator
from .inspector import CoreAIInspector, TorchFXInspector
from .table_writer import _Column, _Row, _TableSpec, _write_table
from .torch_utils import get_torch_to_coreai_output_mapping
from .utils import _plain, _with_debug

if TYPE_CHECKING:
    import torch
    from coreai.authoring import AIProgram
    from numpy.typing import NDArray
    from torch.export import ExportedProgram

    from .inspector import Inspector
    from .torch_utils import OutputMapping


@dataclass(frozen=True)
class IntermediateComparison:
    """One mapped operation, compared."""

    torch_node_name: str
    """Name of the torch operation."""

    coreai_op_id: int
    """Core AI operation its output was mapped to."""

    status: Comparator.Status
    """What came of the comparison."""

    max_diff: float | None = None
    """Largest finite absolute difference, when values were compared.

    None when any difference was non-finite -- an `inf` or `nan` against a finite value -- so
    a reader cannot mistake an unmeasurable difference for a small one.
    """

    shape: tuple[int, ...] | None = None
    """Shape the comparison was made at, after any squeeze."""

    detail: str = ""
    """Why this was not compared, when it was not."""

    def to_dict(self) -> dict[str, Any]:
        """
        Return the comparison as plain values.

        Returns:
            Which operations were compared, the verdict, and the difference when
            values were compared at all.

        """
        return {
            "torch_node_name": self.torch_node_name,
            "coreai_op_id": self.coreai_op_id,
            "status": _plain(self.status),
            "max_diff": _plain(self.max_diff),
            "shape": _plain(self.shape),
            "detail": self.detail,
        }


@dataclass
class IntermediatesReport:
    """Every mapped operation of one model, and what came of comparing it."""

    comparisons: list[IntermediateComparison] = field(default_factory=list)
    """One entry per mapped operation, in mapping order."""

    @property
    def compared(self: Self) -> list[IntermediateComparison]:
        """
        The operations a verdict was actually reached on.

        Returns:
            Those that passed or failed, excluding everything unverified.

        """
        return [
            entry
            for entry in self.comparisons
            if entry.status in (Comparator.Status.PASS, Comparator.Status.FAIL)
        ]

    @property
    def failures(self: Self) -> list[IntermediateComparison]:
        """
        The operations whose values disagree, worst first.

        Returns:
            The failing comparisons, largest difference first.

        """
        failing = [
            entry
            for entry in self.comparisons
            if entry.status is Comparator.Status.FAIL
        ]
        failing.sort(key=lambda entry: -(entry.max_diff or 0.0))
        return failing

    def summary(self: Self) -> dict[str, int]:
        """
        Count the comparisons by status name.

        Returns:
            Status name to count.

        """
        counts: dict[str, int] = {}
        for entry in self.comparisons:
            counts[entry.status.name] = counts.get(entry.status.name, 0) + 1
        return counts

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Return the report as plain values.

        :meth:`summary` and the two derived lists are included as counts. The summary
        is the denominator the report exists to supply -- "0 failing" over 5 compared
        of 78 mapped means something quite different from "0 failing" over 78 -- and
        recomputing it from `comparisons` means reimplementing which statuses count
        as compared.

        Returns:
            Every comparison, and the counts that say how much was actually checked.

        """
        return {
            "comparisons": [entry.to_dict() for entry in self.comparisons],
            "summary": dict(self.summary()),
            "compared_count": len(self.compared),
            "failure_count": len(self.failures),
        }

    def write_to(self: Self, output: TextIO, *, width: int | None = None) -> None:
        """
        Write the comparisons as a table.

        Args:
            output: Text stream to write to.
            width: Console width to render at. Defaults to the table writer's own.

        """
        spec = _TableSpec(
            title="Intermediate comparison",
            columns=(
                _Column("Status"),
                _Column("Torch op"),
                _Column("Core AI op", justify="right"),
                _Column("Max diff", justify="right"),
                _Column("Detail"),
            ),
            caption=(
                f"{len(self.compared)} compared, {len(self.failures)} failing. "
                f"By status: {self.summary()}. Only PASS and FAIL are verdicts; the rest "
                "were not compared and say so in Detail."
            ),
            show_lines=True,
        )
        ordered = sorted(
            self.comparisons,
            key=lambda entry: (
                entry.status is not Comparator.Status.FAIL,
                -(entry.max_diff or 0.0),
            ),
        )
        for entry in ordered:
            spec.add(
                _Row(
                    cells=(
                        entry.status.name,
                        entry.torch_node_name,
                        str(entry.coreai_op_id),
                        "--" if entry.max_diff is None else f"{entry.max_diff:.6g}",
                        entry.detail or "--",
                    ),
                ),
            )
        _write_table(spec, output, width=width)


def _squeezed_pair(
    source: NDArray[Any],
    target: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any]] | None:
    """
    Reconcile two shapes that describe the same tensor.

    Args:
        source: Torch's value.
        target: Core AI's value.

    Returns:
        The pair at a common shape, or None when no squeeze reconciles them.

    """
    if source.shape == target.shape:
        return source, target
    squeezed_source, squeezed_target = np.squeeze(source), np.squeeze(target)
    if squeezed_source.shape == squeezed_target.shape:
        return squeezed_source, squeezed_target
    return None


def compare_intermediates(
    torch_intermediates: dict[Inspector.OpID, list[NDArray[Any] | None] | None],
    coreai_intermediates: dict[Inspector.OpID, list[NDArray[Any] | None] | None],
    mappings: dict[str, OutputMapping],
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    exclude_multi_output: bool = True,
) -> IntermediatesReport:
    """
    Compare each mapped operation's captured intermediates.

    Args:
        torch_intermediates: Torch values by operation name.
        coreai_intermediates: Core AI values by operation id.
        mappings: Torch operation name to its Core AI counterpart, from
            :func:`get_torch_to_coreai_output_mapping`.
        rtol: Relative tolerance.
        atol: Absolute tolerance, defaulting to 1e-3 as the comparator's does.
        exclude_multi_output: Report a multi-output operation as EXCLUDED rather than
            comparing it by index. On by default because the runtime's output order is not
            stable between compilations -- `TinySelfAttention`'s 3-way `split` came back in
            four different orders across one afternoon, every tensor numerically exact -- so
            an indexed comparison of one tests the labelling rather than the values. Pass
            False to compare anyway and let it fail: the unstable order is a defect, and a
            report that excuses it as "only the labelling" hides the thing worth fixing.

    Returns:
        One entry per mapping, classified.

    """
    report = IntermediatesReport()

    def record(
        name: str,
        op_id: int,
        status: Comparator.Status,
        detail: str = "",
        **extra: Any,  # noqa: ANN401
    ) -> None:
        """Append one classified comparison.

        Args:
            name: Torch operation name.
            op_id: Core AI operation id.
            status: The verdict, or why there is none.
            detail: Why it was not compared, when it was not.
            extra: Remaining `IntermediateComparison` fields.
        """
        report.comparisons.append(
            IntermediateComparison(
                torch_node_name=name,
                coreai_op_id=op_id,
                status=status,
                detail=detail,
                **extra,
            ),
        )

    for torch_node_name, mapping in mappings.items():
        torch_values = torch_intermediates.get(torch_node_name)
        coreai_values = coreai_intermediates.get(mapping.target_op_id)

        if torch_values is None:
            record(
                torch_node_name,
                mapping.target_op_id,
                Comparator.Status.UNKNOWN,
                "no torch intermediate captured",
            )
            continue
        if coreai_values is None or not coreai_values:
            record(
                torch_node_name,
                mapping.target_op_id,
                Comparator.Status.NO_TARGET_VALUE,
                "runtime materialised no value",
            )
            continue

        if (len(torch_values) > 1 or len(coreai_values) > 1) and exclude_multi_output:
            record(
                torch_node_name,
                mapping.target_op_id,
                Comparator.Status.EXCLUDED,
                f"{len(torch_values)} torch and {len(coreai_values)} Core AI outputs, and "
                "the runtime's output order is not stable between compilations, so no "
                "output can be identified",
            )
            continue

        if mapping.source_output >= len(torch_values) or mapping.target_output >= len(
            coreai_values,
        ):
            record(
                torch_node_name,
                mapping.target_op_id,
                Comparator.Status.NOT_MAPPED,
                "mapped output index is out of range",
            )
            continue

        torch_output = torch_values[mapping.source_output]
        coreai_output = coreai_values[mapping.target_output]
        if torch_output is None or coreai_output is None:
            record(
                torch_node_name,
                mapping.target_op_id,
                Comparator.Status.NO_TARGET_VALUE,
                "mapped output is missing",
            )
            continue

        pair = _squeezed_pair(torch_output, coreai_output)
        if pair is None:
            record(
                torch_node_name,
                mapping.target_op_id,
                Comparator.Status.SHAPE_AMBIGUOUS,
                f"torch {torch_output.shape} vs Core AI {coreai_output.shape}",
            )
            continue

        source, target = pair
        difference = np.abs(source.astype(np.float64) - target.astype(np.float64))
        # None whenever any difference is non-finite, rather than the maximum of the finite
        # ones. `inf - inf` is nan and an `inf` against a finite value is `inf`, so filtering
        # those and taking the rest reports 0.0 for a pair that disagrees infinitely -- which
        # reads as a perfect match and sorts the row *last* in `failures`, burying the worst
        # case. Unmeasurable is not zero; the comparator returns None here too.
        max_diff = (
            float(difference.max()) if bool(np.isfinite(difference).all()) else None
        )
        matched = bool(
            np.allclose(source, target, rtol=rtol, atol=atol, equal_nan=True)
        )
        record(
            torch_node_name,
            mapping.target_op_id,
            Comparator.Status.PASS if matched else Comparator.Status.FAIL,
            "",
            max_diff=max_diff,
            shape=tuple(source.shape),
        )

    return report


async def compare_program_intermediates(
    exported_program: ExportedProgram,
    program: AIProgram,
    inputs: dict[str, torch.Tensor],
    *,
    function_name: str = "main",
    specialization_options: SpecializationOptions | None = None,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    exclude_multi_output: bool = True,
) -> IntermediatesReport:
    """
    Run both programs on the same inputs and compare every mapped operation.

    The whole flow in one call: capture torch's intermediates, deploy the Core AI program
    under `specialization_options`, capture its intermediates, and compare them through the
    output mapping. Use :func:`compare_intermediates` instead when the values are already in
    hand, or when they were captured some other way.

    Both sides see the same inputs, since comparing values obtained from different inputs
    reports differences that are only the inputs.

    Args:
        exported_program: The decomposed torch program, whose graph names the operations to
            capture and which produces the reference values.
        program: The Core AI program converted from it, carrying the debug info the output
            mapping is read from.
        inputs: Inputs by name, in the program's argument order. Typed `dict` rather than
            `Mapping` because the order is part of the contract and `Mapping` promises none:
            the arguments are ordered by the exported program's signature when the keys are
            its input names, and by insertion order when they are not.
        function_name: Core AI function to run.
        specialization_options: Options for the deployment, so a comparison can be made of
            the model as a particular device will run it. Debug is enabled on top of whatever
            is passed, since the mapping and the intermediates are read from debug info.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        exclude_multi_output: As :func:`compare_intermediates`.

    Returns:
        One entry per mapping, classified.

    """
    # Order the arguments by the program's own signature where the keys allow it: feeding
    # them in the wrong order runs the model on inputs it was not given, and every operation
    # then differs for a reason that has nothing to do with the lowering. Callers do not
    # always key their inputs by the program's placeholder names, so a partial match falls
    # back to the mapping's order rather than failing -- pass an ordered mapping in argument
    # order when the names differ.
    signature_names = list(exported_program.graph_signature.user_inputs)
    if len(signature_names) == len(inputs) and all(
        name in inputs for name in signature_names
    ):
        torch_arguments = tuple(inputs[name] for name in signature_names)
    else:
        torch_arguments = tuple(inputs.values())
    numpy_inputs = {name: value.numpy() for name, value in inputs.items()}

    torch_inspector = TorchFXInspector(exported_program)
    torch_intermediates = await torch_inspector.get_intermediates_for_ops(
        [
            node.name
            for node in exported_program.graph.nodes
            if node.op == "call_function"
        ],
        torch_arguments,
    )

    mappings = get_torch_to_coreai_output_mapping(program)

    with TemporaryDirectory() as temp_dir_name:
        asset = program.save_asset(Path(temp_dir_name) / "model.aimodel")
        model = await AIModel.load(asset.path, _with_debug(specialization_options))
        coreai_inspector = CoreAIInspector(model=model, function_name=function_name)
        coreai_intermediates = await coreai_inspector.get_intermediates_for_ops(
            [mapping.target_op_id for mapping in mappings.values()],
            numpy_inputs,
        )

    return compare_intermediates(
        torch_intermediates,
        coreai_intermediates,
        mappings,
        rtol=rtol,
        atol=atol,
        exclude_multi_output=exclude_multi_output,
    )
