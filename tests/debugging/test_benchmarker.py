# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for benchmarker with ODIX to Core AI ID mapping."""

import sys
from io import StringIO

import pytest
import torch
from coreai.authoring import AIProgram

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder
from coreai_torch.debugging.benchmarker import (
    BenchmarkResult,
    Measurement,
    OperationTiming,
    _timing_annotation_callback,
    benchmark_coreai_program,
)
from coreai_torch.debugging.debug_info import _build_coreai_op_map

from .test_model import HierarchicalModel, get_example_inputs


@pytest.fixture
async def hierarchical_coreai_program() -> AIProgram:
    """Fixture that provides a AIProgram from a hierarchical model."""
    model = HierarchicalModel().eval()
    example_inputs = get_example_inputs(HierarchicalModel)
    exported_program = torch.export.export(model, args=tuple(example_inputs.values()))
    exported_program = exported_program.run_decompositions()
    converter: TorchConverter = TorchConverter()
    converter._debug_info_recorder.config = _DebugInfoRecorder.Config(
        include_stack_trace=True,
        verify_debuginfo_locations=True,
    )
    converter.add_exported_program(exported_program, entrypoint_name="main")
    coreai_program = converter.to_coreai()

    return coreai_program


@pytest.mark.skip(reason="debugger issue (will be solved later)")
@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_odix_to_coreai_id_conversion(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """Test that benchmarker converts ODIX IDs to Core AI IDs when storing timings."""
    example_inputs = get_example_inputs(HierarchicalModel)
    num_runs = 10

    # Run benchmark
    result = await benchmark_coreai_program(
        coreai_program=hierarchical_coreai_program,
        inputs=example_inputs,
        entry_point="main",
        num_runs=num_runs,
    )

    result.write_summary(sys.stdout)

    # Should have profiled some operations
    assert len(result.operation_timings) > 0, "Should have operation timings"

    # Check each operation has valid timing data
    for _, timing in result.operation_timings:
        # Operation ID should be integer (Core AI ID)
        assert isinstance(timing.op_id, int), (
            f"Operation ID should be int, got {type(timing.op_id)}"
        )

        # Should have statistics
        assert timing.measurement.statistics is not None, (
            f"Operation {timing.op_id} should have statistics"
        )

        # Should have positive timing
        assert timing.measurement.statistics.average > 0, (
            f"Operation {timing.op_id} should have positive average timing"
        )

        # Should have correct number of samples
        assert len(timing.measurement.samples) == num_runs, (
            f"Operation {timing.op_id} should have at-least {num_runs} samples, got {len(timing.measurement.samples)}"
        )
        # All samples should be positive
        for sample in timing.measurement.samples:
            assert sample > 0, f"Operation {timing.op_id} sample should be positive"


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_module_timings(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """Test module timing hierarchy from stack traces."""
    example_inputs = get_example_inputs(HierarchicalModel)

    # Run benchmark
    result = await benchmark_coreai_program(
        coreai_program=hierarchical_coreai_program,
        inputs=example_inputs,
        entry_point="main",
        num_runs=10,
    )

    # Get module timings
    module_timings = result.get_module_timings()

    # Should have at least one module
    assert len(module_timings) > 0, "Should have at least one module"

    # Test module structure
    for module_name, module in module_timings.items():
        # Module should have a name
        assert isinstance(module.name, str), "Module name should be a string"
        assert len(module.name) > 0, "Module name should not be empty"

        # Module should have operations or children
        has_content = len(module.operation_timings) > 0 or len(module.children) > 0
        assert has_content, f"Module {module_name} should have operations or children"

        # A module reports no total -- fusion crosses module boundaries, so one
        # would charge a module for a sibling's work. Its dispatches carry the
        # timing instead.
        if module.operation_timings:
            for timing in module.get_all_operations():
                assert timing.measurement.statistics is not None, (
                    f"Dispatch {timing.op_ids} in {module_name} should have statistics"
                )

        # type_name and instance split the frame the stack trace gave
        assert module.type_name, "Module should have a type name"
        assert module.instance is None or module.instance > 0, (
            f"Module {module_name} instance should be positive when known"
        )

        # Test write_to method - we can't easily test TextIO output directly,
        # but we can verify it doesn't throw an error

        buffer = StringIO()
        module.write_to(buffer, show_operations=True)
        formatted = buffer.getvalue()
        assert len(formatted) > 0, "Formatted output should not be empty"
        assert module.name in formatted, "Formatted output should contain module name"

        # Test write_to without showing operations
        buffer_compact = StringIO()
        module.write_to(buffer_compact, show_operations=False)
        formatted_compact = buffer_compact.getvalue()
        assert len(formatted_compact) > 0, (
            "Compact formatted output should not be empty"
        )

    # Print formatted output for visual inspection
    for module in module_timings.values():
        module.write_to(sys.stdout, show_operations=True)
        sys.stdout.write("\n")


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_annotate_dominant_source(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """Test annotating dominant source file with timing information."""
    example_inputs = get_example_inputs(HierarchicalModel)

    # Run benchmark
    result = await benchmark_coreai_program(
        coreai_program=hierarchical_coreai_program,
        inputs=example_inputs,
        entry_point="main",
        num_runs=10,
    )

    # Get module timings
    root_module_timings = result.get_module_timings()["HierarchicalModel$1"]

    # Iterate through all modules including children
    for module in root_module_timings.get_all_modules():
        sys.stdout.write(f"\n--- Module: {module.name} ---\n")
        # Annotate dominant source to stdout
        # This tests that the method works with terminal output and hierarchies
        module.annotate_dominant_source(sys.stdout)
        sys.stdout.write(2 * "\n")  # Add blank line between modules


def _timing(op_ids: list[int], odix_id: int, ms: float) -> OperationTiming:
    """A dispatch of *op_ids*, measured by ODIX op *odix_id* at *ms* every sample."""
    return OperationTiming(
        op_ids=op_ids,
        operations=[],
        measurement=Measurement.from_samples([ms] * 3),
        odix_id=odix_id,
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_every_dispatch_of_an_operation_reaches_its_annotation(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """
    Two dispatches of one operation must both be reported, not silently deduplicated.
    """
    operations = _build_coreai_op_map(hierarchical_coreai_program)
    twice, once = sorted(operations)[:2]

    callback = _timing_annotation_callback(
        [
            _timing([twice], odix_id=125, ms=0.5),
            _timing([twice], odix_id=126, ms=0.25),
            _timing([once], odix_id=128, ms=0.1),
        ]
    )

    annotation = callback(operations[twice])
    assert annotation is not None
    assert len(annotation.dispatches) == 2, "Neither dispatch may be dropped"
    assert {dispatch.odix_id for dispatch in annotation.dispatches} == {125, 126}
    assert annotation.total_average_ms == pytest.approx(0.75), (
        "Separate work, so the operation's cost is the sum of its dispatches"
    )

    single = callback(operations[once])
    assert single is not None and len(single.dispatches) == 1


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_an_annotation_names_each_dispatch_when_there_are_several(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """
    A total with no breakdown hides that it is a sum, and a bare line hides the rest.

    One dispatch renders exactly as before; several render a total and then one line
    each, so a reader can see where the number came from.
    """
    operations = _build_coreai_op_map(hierarchical_coreai_program)
    twice = sorted(operations)[0]

    callback = _timing_annotation_callback(
        [_timing([twice], odix_id=125, ms=0.5), _timing([twice], odix_id=126, ms=0.25)]
    )
    lines = callback(operations[twice]).lines()

    assert len(lines) == 3, "A total, then one line per dispatch"
    assert "2 dispatches" in lines[0].text
    assert "125" in lines[1].text and "126" in lines[2].text

    only = _timing_annotation_callback([_timing([twice], odix_id=125, ms=0.5)])
    assert len(only(operations[twice]).lines()) == 1


def test_a_dispatch_row_names_which_dispatch_it_is() -> None:
    """
    Otherwise two dispatches of one operation are identical rows but for the numbers,
    which reads as the same work measured twice rather than as two pieces of it.

    ``-1`` is included because the runtime really reports it, so it is not free to
    use as a "no dispatch" sentinel: on the delegate path the identifier is assigned
    at run time and ``compile_ids.id`` comes back ``-1`` for every GPU dispatch. A
    numeric sentinel would print those measurements as though they were hand-made,
    which is why the field is ``None`` when absent.
    """
    first = _timing([1], odix_id=125, ms=0.5).to_row()
    second = _timing([1], odix_id=126, ms=0.25).to_row()
    from_gpu = _timing([1], odix_id=-1, ms=0.1).to_row()
    hand_made = OperationTiming(
        op_ids=[1], operations=[], measurement=Measurement.from_samples([0.1])
    ).to_row()

    assert first.cells[0] == "125"
    assert second.cells[0] == "126"
    assert from_gpu.cells[0] == "-1", "A real dispatch, not a missing one"
    assert hand_made.cells[0] == "-", "Only an absent dispatch renders as absent"
    assert first.cells[1] == second.cells[1] == "1", "Same operation, either way"


def test_the_summary_says_when_an_operation_spans_several_rows() -> None:
    """
    A reader totalling the median column needs to know the rows are separate work.

    Suppressed when nothing is split, so its presence carries information.
    """
    split = BenchmarkResult(
        operation_timings=[
            _timing([1], odix_id=125, ms=0.5),
            _timing([1], odix_id=126, ms=0.25),
            _timing([2], odix_id=128, ms=0.1),
        ]
    )
    output = StringIO()
    split.write_summary(output)
    assert "1 of 2 operation sets were measured by more than one dispatch" in (
        output.getvalue().replace("\n", " ")
    )

    plain = BenchmarkResult(operation_timings=[_timing([2], odix_id=128, ms=0.1)])
    output = StringIO()
    plain.write_summary(output)
    assert "more than one dispatch" not in output.getvalue()


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_a_benchmark_that_times_nothing_says_what_it_measured(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """An empty result must account for itself, so it cannot read as "the model is free".

    Some runtimes report one interval per delegate region rather than one per kernel. The
    benchmarker then has nothing to attribute and returns no operation timings -- a result
    indistinguishable, to a caller that only looks at `operation_timings`, from a model
    that cost nothing. The sample counters are what tell the two apart, so an empty result
    with all of them at zero is the failure this pins: it would mean samples arrived and
    were dropped without record, or none arrived and nothing said so.
    """
    result = await benchmark_coreai_program(
        coreai_program=hierarchical_coreai_program,
        inputs=get_example_inputs(HierarchicalModel),
        entry_point="main",
        num_runs=5,
    )

    if result.operation_timings:
        # Per-operation attribution worked; there is nothing to account for.
        return

    accounted = result.symbol_samples + result.unattributed_samples
    assert accounted > 0, (
        "no operation was timed and no sample was accounted for either: "
        f"symbol_samples={result.symbol_samples}, "
        f"unattributed_samples={result.unattributed_samples}. An empty benchmark with "
        "no explanation cannot be distinguished from a model that cost nothing."
    )
