# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
An operation histogram must describe composite bodies, however deeply they nest.

A composite body may itself invoke a composite, so the counts form a tree rather than
two levels. `TwoNormModel` produces one: externalization emits a `coreai.graph` per
marked submodule, so marking both the block and the norm inside it gives a body that
invokes a body -- and calling the block twice makes the norm's work happen twice.
"""

import io
from collections import Counter

import torch
from coreai.authoring import AIProgram

from coreai_torch import TorchConverter, get_decomp_table
from coreai_torch.composite_ops import RMSNorm
from coreai_torch.debugging.histogram import (
    OperationHistogram,
    operation_histogram,
)
from coreai_torch.debugging.utils import _strip_uuid_suffix
from coreai_torch.externalize import ExternalizeSpec

from .test_model import NormBlock, TwoNormModel, get_example_inputs


def _program() -> AIProgram:
    """Convert `TwoNormModel` with the block and its norm externalized."""
    torch.manual_seed(0)
    model = TwoNormModel().eval()
    sample = tuple(get_example_inputs(TwoNormModel).values())
    return (
        TorchConverter()
        .add_pytorch_module(
            model,
            export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
                get_decomp_table()
            ),
            externalize_modules=[
                ExternalizeSpec(RMSNorm, composite_op_name="rms_norm"),
                NormBlock,
            ],
        )
        .to_coreai()
    )


def _reachable(histogram: OperationHistogram) -> Counter[str]:
    """
    Every operation reachable from *histogram*, bodies counted per invocation.

    An independent walk, so a test can check the reported structure against a count the
    module does not compute itself.
    """
    total: Counter[str] = Counter(histogram.counts)
    for composite in histogram.composites:
        for name, count in _reachable(composite.histogram).items():
            total[name] += count * composite.invocations
    return total


def test_a_composite_body_reports_the_composite_it_invokes() -> None:
    """
    A nested composite is described, not flattened away or reported as a leaf.

    Flattening it would leave no way to tell which body an operation is in, which is
    the question the split exists to answer.
    """
    histogram = operation_histogram(_program())

    assert histogram.composites, "The entry point invokes a block"
    block = histogram.composites[0]
    assert block.histogram.composites, (
        f"{block.label}'s body invokes the norm, so its histogram must describe it; "
        f"body holds {block.histogram.counts}"
    )
    norm = block.histogram.composites[0]
    assert "coreai.rsqrt" in norm.histogram.counts, (
        f"The norm body should hold the rsqrt, got {norm.histogram.counts}"
    )


def test_a_nested_body_counts_once_per_invocation_of_its_caller() -> None:
    """
    A nested body is reached once per invocation of every ancestor, not just its own.

    The norm sits inside the block and the block is called twice, so the single rsqrt in
    the norm accounts for two of them. Counting the norm once -- which reading only its
    own `invocations` does -- understates the work performed.
    """
    histogram = operation_histogram(_program())

    assert _reachable(histogram)["coreai.rsqrt"] == 2, (
        "One rsqrt per norm, reached through two invocations of the block"
    )


def test_a_size_totals_a_body_and_everything_it_invokes() -> None:
    """
    `counts` describes the body itself, so code size stays readable next to work done.
    """
    histogram = operation_histogram(_program())
    block = histogram.composites[0]

    assert block.size == sum(block.histogram.counts.values()) + sum(
        nested.size * nested.invocations for nested in block.histogram.composites
    ), "A body's size is its own operations plus what it invokes"
    assert histogram.size == sum(_reachable(histogram).values()), (
        "The program's size is every operation reachable from it"
    )


def test_a_declared_composite_op_is_labelled_by_its_op_name() -> None:
    """
    A composite op is named by the operation it stands for, not its module path.

    ``rms_norm`` is what the composite computes; ``block.norm`` is where one instance
    happens to sit. Labelling by path makes two instances of one op look like two
    different things, and makes the label change when a model is restructured.
    """
    histogram = operation_histogram(_program())
    norms = [
        nested.label
        for composite in histogram.composites
        for nested in composite.histogram.composites
    ]

    assert norms and set(norms) == {"rms_norm"}, (
        f"Every norm should carry the declared op name, got {norms}"
    )


def test_a_randomised_symbol_suffix_is_stripped_from_the_label() -> None:
    """
    Labels must not carry a per-build random suffix.

    Two generators produce one -- lowercase letters from `coreai.graph`'s deferred
    construction, hex from module externalization's `uuid4().hex[:8]`. A label that
    keeps either differs between two builds of the same model, so nothing that compares
    two programs by label can match.
    """
    assert _strip_uuid_suffix("sdpa_maskless_qxrbmzua") == "sdpa_maskless"
    assert _strip_uuid_suffix("block.norm_b847a524") == "block.norm"
    assert _strip_uuid_suffix("layer_norm") == "layer_norm", "Nothing to strip"


def test_a_composite_label_is_stable_across_two_builds() -> None:
    """
    The same model converted twice must produce the same labels.
    """

    def labels(histogram: OperationHistogram) -> list[str]:
        found = []
        for composite in histogram.composites:
            found.append(composite.label)
            found.extend(labels(composite.histogram))
        return found

    assert labels(operation_histogram(_program())) == labels(
        operation_histogram(_program())
    ), "Labels must not carry build-specific noise"


def test_a_summary_names_every_composite_it_found() -> None:
    """
    The written report shows the nested body, so a reader sees where operations live.
    """
    histogram = operation_histogram(_program())
    output = io.StringIO()
    histogram.write_summary(output)
    written = output.getvalue()

    block = histogram.composites[0]
    norm = block.histogram.composites[0]
    for composite in (block, norm):
        assert composite.label in written, f"{composite.label} missing from the summary"
        assert composite.symbol in written, (
            f"{composite.symbol} missing -- the symbol is the handle for a dump or trace"
        )
