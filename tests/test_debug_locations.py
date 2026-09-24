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

from coreai_torch import _debug_locations, get_decomp_table
from coreai_torch._debug_locations import _DebugInfoRecorder, _get_nested_operations
from coreai_torch.converter import TorchConverter
from coreai_torch.debugging.debug_info import get_operation_id

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
        for operation in _get_nested_operations(program._module._mlir_module.operation)
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


class DeepChainModel(nn.Module):
    """A chain long enough that per-node whole-graph work is visible."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def _convert(model: nn.Module, depth_input: torch.Tensor) -> object:
    exported_program: ExportedProgram = torch.export.export(
        model.eval(), (depth_input,)
    )
    exported_program = exported_program.run_decompositions(get_decomp_table())
    converter: TorchConverter = TorchConverter()
    converter.add_exported_program(
        exported_program, entrypoint_name="f", input_names=["x"], output_names=["y"]
    )
    return converter.to_coreai()


def test_operation_ids_increase_in_ir_order() -> None:
    """Operation IDs follow IR order, with none missing and none repeated.

    This is a *stronger* guarantee than before, not a preserved one. IDs used to be
    assigned as each node was lowered, which is not the same as IR order: constants are
    inserted at the top of the block rather than appended, so a chain of linears produced
    ``[0, 3, 7, 10, ..., 1, 2, 4, 5, ...]`` when read in IR order. Assigning them in one
    pass over the finished graph makes the numbering match the IR, which is what
    per-node ordering was reaching for.
    """
    program = _convert(DeepChainModel(6), torch.randn(1, 8))

    body_ids = [
        get_operation_id(operation)
        for operation in _get_nested_operations(program._module._mlir_module.operation)
        if operation.name != "coreai.graph"
    ]

    assert body_ids, "expected a graph body"
    assert None not in body_ids, "every operation gets an ID"
    assert len(body_ids) == len(set(body_ids)), f"repeated IDs: {body_ids}"
    assert all(later > earlier for earlier, later in zip(body_ids, body_ids[1:])), (
        f"IDs are out of IR order: {body_ids}"
    )


def test_the_graph_is_not_rewalked_for_every_lowered_node(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The graph is walked a fixed number of times, not once per lowered node.

    This is the shape of the performance fix rather than a timing assertion: walking the
    graph per node made conversion quadratic in graph size (~70x on a 512-layer chain),
    and each walk ends in a binding-level C++ exception, so the constant factor is large
    too. Counting walks is stable in CI in a way that wall-clock is not.
    """
    walks: list[str] = []
    original = _debug_locations._get_nested_operations

    def counting_get_nested_operations(operation):  # type: ignore[no-untyped-def]
        if operation.name == "coreai.graph":
            walks.append(operation.name)
        return original(operation)

    monkeypatch.setattr(
        _debug_locations, "_get_nested_operations", counting_get_nested_operations
    )

    _convert(DeepChainModel(4), torch.randn(1, 8))
    shallow = len(walks)
    walks.clear()
    _convert(DeepChainModel(16), torch.randn(1, 8))
    deep = len(walks)

    assert shallow > 0, "the graph is still walked once per graph"
    assert deep == shallow, (
        f"graph walks scale with node count: {shallow} for 4 layers, {deep} for 16"
    )


def test_output_maps_survive_deferred_operation_ids() -> None:
    """Torch-to-Core AI output maps are still recorded.

    They are attached while a node is lowered, but keyed on the operation having debug
    info -- not on it having an ID, which is only assigned later. Keying on the ID drops
    every output map silently.
    """
    program = _convert(DeepChainModel(3), torch.randn(1, 8))

    asm = program._module._mlir_module.operation.get_asm(enable_debug_info=True)

    assert "output_maps" in asm, "no output maps were recorded"
