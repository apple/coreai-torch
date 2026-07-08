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
from coreai_torch.debugging.benchmarker import benchmark_coreai_program

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

        # If module has operations, check total_time
        if module.operation_timings:
            total_time_stats = module.total_time
            assert total_time_stats is not None, (
                f"Module {module_name} should have timing statistics"
            )
            assert total_time_stats.average > 0, (
                f"Module {module_name} should have positive average time"
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
