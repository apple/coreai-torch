# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the compute plan utilities."""

import sys
from io import StringIO

import pytest
import torch
from coreai._compiler.ir import Operation, WalkResult
from coreai.authoring import AIProgram
from coreai.runtime import SpecializationOptions

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder
from coreai_torch.debugging.compute_plan import ComputeDevice, ComputePlan
from coreai_torch.debugging.debug_info import get_operation_id

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


def _collect_operations(coreai_program: AIProgram) -> list[Operation]:
    """Walk the module and collect all operations."""
    operations: list[Operation] = []

    def walk(op: Operation) -> WalkResult:
        operations.append(op)
        return WalkResult.ADVANCE

    coreai_program._mlir_module.operation.walk(walk)
    return operations


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_compute_plan_from_program(
    hierarchical_coreai_program: AIProgram,
    specialization_options: SpecializationOptions | None,
) -> None:
    """ComputePlan.from_program builds a non-empty operation-to-device mapping."""
    plan = await ComputePlan.from_program(
        hierarchical_coreai_program, specialization_options=specialization_options
    )

    assert isinstance(plan, ComputePlan)
    assert len(plan._coreai_id_to_compute_info_map) > 0, (
        "Expected the compute plan to contain at least one operation"
    )

    # An entry the planner made must name a device it chose. `<= set(ComputeDevice)`
    # was the assertion here and cannot fail: `ComputeDevice.__missing__` maps any
    # string at all to `UNKNOWN`, and `UNKNOWN` is itself a member, so a plan that
    # resolved nothing passed exactly as one that resolved everything. Measured on
    # this model: 31 entries all CPU under the bundled runtime, 10 all GPU under the
    # OS framework, and **no** entry resolving to UNKNOWN in either.
    for op_id, compute_info in plan._coreai_id_to_compute_info_map.items():
        assert isinstance(op_id, int), "Core AI operation IDs should be integers"
        assert len(compute_info.devices) > 0, (
            f"Operation {op_id} should map to at least one device"
        )
        assert ComputeDevice.UNKNOWN not in compute_info.devices, (
            f"Operation {op_id} has a plan entry that names no device: "
            f"{compute_info.devices}"
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_compute_plan_get_devices_for_operation(
    hierarchical_coreai_program: AIProgram,
    specialization_options: SpecializationOptions | None,
) -> None:
    """
    get_devices resolves a device for exactly the operations the plan holds.

    The distinction this pins down is the one `UNKNOWN` blurs: it is returned both for
    an operation the planner placed nowhere and for one the plan simply has no entry
    for. Asserting that the two coincide is what makes an `UNKNOWN` here mean "not in
    the plan" rather than "the planner gave up". Measured: 31 of 47 walked operations
    resolve under the bundled runtime and 10 of 47 under the OS framework, and in
    neither case does an operation *with* an entry come back UNKNOWN.
    """
    plan = await ComputePlan.from_program(
        hierarchical_coreai_program, specialization_options=specialization_options
    )

    operations = _collect_operations(hierarchical_coreai_program)
    assert len(operations) > 0, "Expected operations in the module"

    resolved = 0
    for operation in operations:
        devices = plan.get_devices(operation)
        assert len(devices) > 0, (
            f"Operation {operation.name} should resolve to at least one device"
        )

        op_id = get_operation_id(operation)
        in_plan = op_id is not None and op_id in plan._coreai_id_to_compute_info_map
        if in_plan:
            resolved += 1
            assert ComputeDevice.UNKNOWN not in devices, (
                f"Operation {operation.name} is in the plan but resolves to UNKNOWN"
            )
        else:
            assert devices == {ComputeDevice.UNKNOWN}, (
                f"Operation {operation.name} has no plan entry, so the only honest "
                f"answer is UNKNOWN, not {devices}"
            )

    assert resolved > 0, (
        "No operation resolved to a device, so this test would pass against a plan "
        "that placed nothing at all"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_compute_plan_validation_messages_are_sets(
    hierarchical_coreai_program: AIProgram,
    specialization_options: SpecializationOptions | None,
) -> None:
    """get_validation_messages always returns a (possibly empty) set of strings."""
    plan = await ComputePlan.from_program(
        hierarchical_coreai_program, specialization_options=specialization_options
    )

    operations = _collect_operations(hierarchical_coreai_program)
    assert len(operations) > 0

    for operation in operations:
        messages = plan.get_validation_messages(operation)
        assert isinstance(messages, set)
        assert all(isinstance(message, str) for message in messages)


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
async def test_compute_plan_annotate_source(
    hierarchical_coreai_program: AIProgram,
    specialization_options: SpecializationOptions | None,
) -> None:
    """annotate_source writes device annotations for the dominant source file."""
    plan = await ComputePlan.from_program(
        hierarchical_coreai_program, specialization_options=specialization_options
    )

    buffer = StringIO()
    # Annotate every attributed source file so submodule sources (e.g.
    # SubModel in test_submodel.py) are rendered too, not just the dominant one.
    plan.annotate_source(
        hierarchical_coreai_program,
        buffer,
        annotate_all_files=True,
    )
    output = buffer.getvalue()

    # Also write to stdout for visual inspection (mirrors test_benchmarker).
    plan.annotate_source(
        hierarchical_coreai_program,
        sys.stdout,
        annotate_all_files=True,
    )

    assert len(output) > 0, "Expected annotated source output"
    # A source file header should be present.
    assert "# ===" in output, "Expected a source file header in the output"
    # At least one known compute device label should appear in an annotation.
    assert any(device.value in output for device in ComputeDevice), (
        "Expected at least one compute device label in the annotated output"
    )
