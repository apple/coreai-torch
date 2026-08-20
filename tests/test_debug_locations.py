# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test for debug location functionality."""

import torch
import torch.nn as nn
from coreai._compiler._mlir_libs._coreaiIR._bindings import mlir as _mlir
from coreai._compiler.ir import Location
from torch.export.exported_program import ExportedProgram

from coreai_torch import get_decomp_table
from coreai_torch._debug_locations import _DebugInfoRecorder, _get_nested_operations
from coreai_torch.converter import TorchConverter

from .debugging.test_model import HierarchicalModel


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.relu(x)
        return x


class SimpleAddReluModel(nn.Module):
    """Simple test model with only add and relu operations."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use direct operations: add + relu only
        x = torch.add(x, 1.0)  # Add constant 1.0
        x = torch.relu(x)
        return x


class ConvModel(nn.Module):
    """Convolutional test model with different architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def is_debuginfo_location(location: Location) -> bool:
    """Check if location is a DebugInfo LocationAttr by checking the string representation."""
    if location is None:
        return False

    # Check if the location string representation contains debuginfo
    location_str = str(location)
    return "debuginfo.location" in location_str


async def test_debug_locations() -> None:
    """Test that debug locations are properly set on operations."""
    # Create model and example input
    model: SimpleAddReluModel = SimpleAddReluModel()
    example_input: torch.Tensor = torch.randn(1, 10)

    # Export the model
    exported_program: ExportedProgram = torch.export.export(model, (example_input,))
    exported_program = exported_program.run_decompositions()

    # Convert to Core AI using TorchConverter
    converter: TorchConverter = TorchConverter()
    debug_config = _DebugInfoRecorder.Config(
        include_stack_trace=True,
        verify_debuginfo_locations=True,
    )
    converter._debug_info_recorder = _DebugInfoRecorder(config=debug_config)
    converter.add_exported_program(exported_program)
    # Verification happens automatically during conversion via _verify_debuginfo_locations
    _ = converter.to_coreai()


def test_debug_locations_multiple_programs() -> None:
    """Test that debug locations are properly set when adding multiple exported programs with different architectures."""
    # Create two different models and appropriate example inputs
    simple_model: SimpleAddReluModel = SimpleAddReluModel()
    conv_model: ConvModel = ConvModel()
    simple_input: torch.Tensor = torch.randn(1, 10)
    conv_input: torch.Tensor = torch.randn(
        1, 3, 32, 32
    )  # Batch, channels, height, width

    # Export both models
    exported_program1: ExportedProgram = torch.export.export(
        simple_model, (simple_input,)
    )
    exported_program1 = exported_program1.run_decompositions()

    exported_program2: ExportedProgram = torch.export.export(conv_model, (conv_input,))
    exported_program2 = exported_program2.run_decompositions()

    # Convert to Core AI using TorchConverter with both programs
    converter: TorchConverter = TorchConverter()
    debug_config = _DebugInfoRecorder.Config(
        include_stack_trace=True,
        verify_debuginfo_locations=True,
    )
    converter._debug_info_recorder = _DebugInfoRecorder(config=debug_config)
    converter.add_exported_program(exported_program1, entrypoint_name="model_1")
    converter.add_exported_program(exported_program2, entrypoint_name="model_2")
    # Verification happens automatically during conversion via _verify_debuginfo_locations
    _ = converter.to_coreai()


def test_intermediate_ops_of_a_lowering_keep_their_attribution() -> None:
    """Test that every op a multi-op lowering emits keeps its debug info.

    Lowering one FX node can emit a chain of operations, of which only the last
    produces a returned result. aten.addmm is such a case: it becomes a transpose
    feeding a batch matmul feeding an add. The ops that only feed another op are
    reachable from the returned result solely through operand edges, and when
    those were not followed they received no file, no line and no module
    hierarchy -- just an operation ID.

    Uses HierarchicalModel because its Linear layers sit at two different depths,
    so the recovered hierarchy has to be the right one rather than merely present.
    """
    model: HierarchicalModel = HierarchicalModel()
    example_input: torch.Tensor = torch.randn(2, 4)

    exported_program: ExportedProgram = torch.export.export(model, (example_input,))
    exported_program = exported_program.run_decompositions(get_decomp_table())

    converter: TorchConverter = TorchConverter()
    converter.add_exported_program(exported_program)
    program = converter.to_coreai()

    matmuls = [
        operation
        for operation in _get_nested_operations(program._mlir_module.operation)
        if "batch_matmul" in operation.name
    ]
    assert matmuls, "expected the linear layers to lower to batch matmuls"

    for operation in matmuls:
        stack_trace = _mlir.get_stack_trace(operation.location)  # type: ignore[attr-defined]
        assert stack_trace, f"{operation.name} has no module hierarchy"
        # The matmul belongs to the Linear that produced its weight, not to an
        # enclosing module and not to nothing at all.
        assert stack_trace[-1].startswith("Linear"), stack_trace

        locations = _mlir.get_file_line_col_locations(operation.location)  # type: ignore[attr-defined]
        assert locations, f"{operation.name} has no source location"
        assert any(
            location.filename.endswith(".py") and location.line >= 1
            for location in locations
        ), locations
