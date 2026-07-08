# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for location API bindings (get_source_info and get_operation_id)."""

from typing import Any

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
import pytest
import torch
from coreai._compiler.ir import Context, Location, Operation, WalkResult
from coreai.authoring import AIProgram

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder

from .test_model import LinearMulAddModel, TwoLinearSigmoidModel, get_example_inputs


@pytest.fixture
async def simple_coreai_program() -> AIProgram:
    """Fixture that provides an AIProgram with debug info enabled."""
    model = LinearMulAddModel().eval()
    example_inputs = get_example_inputs(LinearMulAddModel)
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


@pytest.fixture
async def complex_coreai_program() -> AIProgram:
    """Fixture that provides an AIProgram from a complex torch model."""
    model = TwoLinearSigmoidModel().eval()
    example_inputs = get_example_inputs(TwoLinearSigmoidModel)
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


def _validate_and_count_source_info(program: AIProgram) -> int:
    """
    Validate source info and count occurrences.

    Args:
        program: The AIProgram to validate

    Returns:
        The number of operations with source info found

    """
    found_count: list[int] = [0]

    def check_operation(operation: Operation) -> WalkResult:
        # Get source info from operation location
        source_infos = _mlir.get_source_info(operation.location)

        # If we found source info, validate its structure
        if source_infos:
            found_count[0] += 1
            for source_info in source_infos:
                # Check that Source has expected attributes
                assert hasattr(source_info, "id")
                assert hasattr(source_info, "name")
                assert hasattr(source_info, "identifiers")

                # Source name should be a known dialect (torch or coreai after compilation)
                assert source_info.name in ["torch", "coreai"], (
                    f"Unexpected source name: '{source_info.name}'"
                )

                # Identifiers should be a list of strings
                assert isinstance(source_info.identifiers, list)
                for identifier in source_info.identifiers:
                    assert isinstance(identifier, str)

        return WalkResult.ADVANCE

    # Walk all operations in the module
    program._mlir_module.operation.walk(check_operation)

    return found_count[0]


def _validate_output_map_structure(output_maps: list[Any]) -> None:
    """
    Validate the structure of output maps.

    Args:
        output_maps: List of output maps to validate

    """
    # Should be a list and should have at least one output map
    assert isinstance(output_maps, list)
    assert len(output_maps) > 0, "Expected to find at least one output map"

    # Validate structure of all output maps
    for output_map in output_maps:
        # Check all required attributes exist
        assert hasattr(output_map, "source_level")
        assert hasattr(output_map, "source_op_id")
        assert hasattr(output_map, "source_output")
        assert hasattr(output_map, "target_level")
        assert hasattr(output_map, "target_op_id")
        assert hasattr(output_map, "target_output")

        # Check types
        assert isinstance(output_map.source_level, str)
        assert isinstance(output_map.source_op_id, int)
        assert isinstance(output_map.source_output, int)
        assert isinstance(output_map.target_level, str)
        assert isinstance(output_map.target_op_id, int)
        assert isinstance(output_map.target_output, int)

        # Validate non-negative values
        assert output_map.source_op_id >= 0
        assert output_map.source_output >= 0
        assert output_map.target_op_id >= 0
        assert output_map.target_output >= 0

        # Validate level names are not empty
        assert len(output_map.source_level) > 0
        assert len(output_map.target_level) > 0


def test_create_and_extract_source_basic() -> None:
    """Test creating a location from Source and extracting it back (round-trip)."""
    # Create a context
    ctx = Context()

    # Create a simple Source
    source = _mlir.Source()
    source.name = "torch"
    source.id = 42
    source.identifiers = ["aten.linear", "aten.relu"]

    # Create location from source
    loc = _mlir.create_location_from_source_info(ctx, source)
    assert loc is not None, "Failed to create location from source"

    # Extract source info back
    extracted_sources = _mlir.get_source_info(loc)

    # Validate we got exactly one source back
    assert len(extracted_sources) == 1, (
        f"Expected 1 source, got {len(extracted_sources)}"
    )

    extracted = extracted_sources[0]

    # Validate all fields match
    assert extracted.name == source.name, (
        f"Name mismatch: {extracted.name} != {source.name}"
    )
    assert extracted.id == source.id, f"ID mismatch: {extracted.id} != {source.id}"
    assert len(extracted.identifiers) == len(source.identifiers), (
        f"Identifier count mismatch: {len(extracted.identifiers)} != {len(source.identifiers)}"
    )

    for i, (extracted_id, original_id) in enumerate(
        zip(extracted.identifiers, source.identifiers, strict=True),
    ):
        assert extracted_id == original_id, (
            f"Identifier {i} mismatch: {extracted_id} != {original_id}"
        )


def test_create_and_extract_source_with_many_identifiers() -> None:
    """Test Source with multiple identifiers (3+)."""
    ctx = Context()

    source = _mlir.Source()
    source.name = "coreai"
    source.id = 999
    source.identifiers = ["matmul", "transpose", "reshape", "add", "multiply"]

    loc = _mlir.create_location_from_source_info(ctx, source)
    extracted_sources = _mlir.get_source_info(loc)

    assert len(extracted_sources) == 1
    extracted = extracted_sources[0]

    assert extracted.name == "coreai"
    assert extracted.id == 999
    assert len(extracted.identifiers) == 5
    assert extracted.identifiers == [
        "matmul",
        "transpose",
        "reshape",
        "add",
        "multiply",
    ]


def test_create_and_extract_source_empty_identifiers() -> None:
    """Test Source with no identifiers."""
    ctx = Context()

    source = _mlir.Source()
    source.name = "torch"
    source.id = 1
    source.identifiers = []

    loc = _mlir.create_location_from_source_info(ctx, source)
    extracted_sources = _mlir.get_source_info(loc)

    assert len(extracted_sources) == 1
    extracted = extracted_sources[0]

    assert extracted.name == "torch"
    assert extracted.id == 1
    assert len(extracted.identifiers) == 0


def test_get_operation_id_basic() -> None:
    """Test getting operation ID from a location."""
    ctx = Context()

    # Create a Source with operation ID
    source = _mlir.Source()
    source.name = "torch"
    source.id = 123
    source.identifiers = ["test_op"]

    # Create location from source
    loc = _mlir.create_location_from_source_info(ctx, source)

    # Extract operation ID
    op_id = _mlir.get_operation_id(loc, "torch")

    assert op_id is not None, "Expected to find operation ID"
    assert op_id.type == "torch"
    assert op_id.value == 123


def test_get_operation_id_different_types() -> None:
    """Test getting operation IDs of different types (torch, coreai)."""
    ctx = Context()

    # Test for each common type
    types_and_ids: list[tuple[str, int]] = [
        ("torch", 42),
        ("coreai", 100),
    ]

    for dialect_type, expected_id in types_and_ids:
        source = _mlir.Source()
        source.name = dialect_type
        source.id = expected_id
        source.identifiers = ["op"]

        loc = _mlir.create_location_from_source_info(ctx, source)
        op_id = _mlir.get_operation_id(loc, dialect_type)

        assert op_id is not None, f"Expected to find {dialect_type} operation ID"
        assert op_id.type == dialect_type, f"Type mismatch for {dialect_type}"
        assert op_id.value == expected_id, f"ID mismatch for {dialect_type}"


def test_get_operation_id_not_found() -> None:
    """Test that get_operation_id returns None when ID is not present."""
    ctx = Context()

    # Create a Source with torch ID
    source = _mlir.Source()
    source.name = "torch"
    source.id = 42
    source.identifiers = ["op"]

    loc = _mlir.create_location_from_source_info(ctx, source)

    # Try to get coreai ID (doesn't exist)
    op_id = _mlir.get_operation_id(loc, "coreai")

    assert op_id is None, "Expected None for non-existent operation ID"


def test_get_source_info_with_unknown_location() -> None:
    """Test get_source_info on unknown location returns empty list."""
    ctx = Context()
    loc = Location.unknown(context=ctx)

    source_infos = _mlir.get_source_info(loc)

    assert isinstance(source_infos, list)
    assert len(source_infos) == 0, "Expected no source info from unknown location"


def test_get_operation_id_with_unknown_location() -> None:
    """Test get_operation_id on unknown location returns None."""
    ctx = Context()
    loc = Location.unknown(context=ctx)

    op_id = _mlir.get_operation_id(loc, "torch")

    assert op_id is None, "Expected None for unknown location"


async def test_get_source_info(simple_coreai_program: AIProgram) -> None:
    """Test get_source_info extracts source metadata from locations in a real model."""
    # Validate and count source info
    count = _validate_and_count_source_info(simple_coreai_program)

    # We should have found at least some source info
    assert count > 0, "Expected to find source info in at least one operation"

    # Additionally, verify identifiers are non-empty strings when present
    def check_identifiers(operation: Operation) -> WalkResult:
        source_infos = _mlir.get_source_info(operation.location)
        if source_infos:
            for source_info in source_infos:
                if source_info.identifiers:
                    for identifier in source_info.identifiers:
                        assert len(identifier) > 0, (
                            "Identifier should not be empty string"
                        )
        return WalkResult.ADVANCE

    simple_coreai_program._mlir_module.operation.walk(check_identifiers)


async def test_get_all_output_maps_from_module_with_debug(
    simple_coreai_program: AIProgram,
) -> None:
    """
    Test extracting all output maps from a module with debug info enabled.

    Output maps are created by InferOutputMappings pass during torch->coreai conversion.
    """
    # Get all output maps from the module
    output_maps = _mlir.get_all_output_maps_from_module(
        simple_coreai_program._mlir_module
    )

    # Validate structure using helper
    _validate_output_map_structure(output_maps)


def test_operation_id_extraction_from_nested_fused() -> None:
    """Test operation ID extraction from deeply nested fused locations."""
    ctx = Context()

    # Create multiple nested sources
    source1 = _mlir.Source()
    source1.name = "torch"
    source1.id = 111
    source1.identifiers = ["op1"]

    loc1 = _mlir.create_location_from_source_info(ctx, source1)
    file_loc = Location.file("test.py", line=1, col=1, context=ctx)

    # Create nested fused location: fused(file_loc, fused(loc1))
    inner_fused = Location.fused([loc1], context=ctx)
    outer_fused = Location.fused([file_loc, inner_fused], context=ctx)

    # Should still be able to find the operation ID
    op_id = _mlir.get_operation_id(outer_fused, "torch")

    assert op_id is not None
    assert op_id.type == "torch"
    assert op_id.value == 111


async def test_complex_model_source_info(
    complex_coreai_program: AIProgram,
) -> None:
    """Test source info extraction from a more complex model."""
    source_info_count: list[int] = [0]
    operations_with_identifiers: list[int] = [0]

    def check_operation(operation: Operation) -> WalkResult:
        source_infos = _mlir.get_source_info(operation.location)

        if source_infos:
            source_info_count[0] += 1

            for source_info in source_infos:
                assert source_info.name == "torch"

                # Count operations with non-empty identifiers
                if source_info.identifiers and len(source_info.identifiers) > 0:
                    operations_with_identifiers[0] += 1

                    # Validate identifier format (should look like operation names)
                    for identifier in source_info.identifiers:
                        assert isinstance(identifier, str)
                        assert len(identifier) > 0

        return WalkResult.ADVANCE

    complex_coreai_program._mlir_module.operation.walk(check_operation)

    # Should have found source info
    assert source_info_count[0] > 0, "Expected source info in complex model"

    # Should have operations with identifiers
    assert operations_with_identifiers[0] > 0, "Expected operations with identifiers"


async def test_complex_model_operation_ids(
    complex_coreai_program: AIProgram,
) -> None:
    """Test operation ID extraction from a more complex model."""
    op_id_count: list[int] = [0]
    unique_ids: set[tuple[str, int]] = set()

    def check_operation(operation: Operation) -> WalkResult:
        op_id = _mlir.get_operation_id(operation.location)

        if op_id is not None:
            op_id_count[0] += 1
            unique_ids.add((op_id.type, op_id.value))

            # Validate structure
            assert isinstance(op_id.type, str)
            assert isinstance(op_id.value, int)
            assert op_id.value >= 0

        return WalkResult.ADVANCE

    complex_coreai_program._mlir_module.operation.walk(check_operation)

    # Note: Operation IDs may not be present after full compilation pipeline
    # When present, IDs should be unique for different operations
    if op_id_count[0] > 0:
        assert len(unique_ids) > 0, "Expected unique operation IDs"


def test_operation_id_type_specificity() -> None:
    """Test that operation ID queries are type-specific."""
    ctx = Context()

    # Create a source with torch ID
    source = _mlir.Source()
    source.name = "torch"
    source.id = 123
    source.identifiers = ["op"]

    loc = _mlir.create_location_from_source_info(ctx, source)

    # Can find torch ID
    torch_id = _mlir.get_operation_id(loc, "torch")
    assert torch_id is not None
    assert torch_id.value == 123

    # Cannot find coreai ID (doesn't exist)
    coreai_id = _mlir.get_operation_id(loc, "coreai")
    assert coreai_id is None


async def test_operation_id_uniqueness(
    simple_coreai_program: AIProgram,
) -> None:
    """Test that operation IDs are unique within their dialect level."""
    coreai_ids: list[int] = []

    def collect_ids(operation: Operation) -> WalkResult:
        coreai_id = _mlir.get_operation_id(operation.location, "coreai")
        if coreai_id:
            coreai_ids.append(coreai_id.value)

        return WalkResult.ADVANCE

    simple_coreai_program._mlir_module.operation.walk(collect_ids)

    if len(coreai_ids) > 1:
        assert len(coreai_ids) == len(set(coreai_ids)), (
            "Core AI operation IDs should be unique"
        )


async def test_get_operation_id(simple_coreai_program: AIProgram) -> None:
    """Test get_operation_id extracts operation ID from locations in a real model."""
    found_count: list[int] = [0]

    # Walk through operations and test get_operation_id
    def check_operation(operation: Operation) -> WalkResult:
        # Get operation ID from location
        op_id = _mlir.get_operation_id(operation.location)

        # If we found an operation ID, validate its structure
        if op_id is not None:
            found_count[0] += 1
            # Check that OperationID has expected attributes
            assert hasattr(op_id, "type")
            assert hasattr(op_id, "value")
            assert isinstance(op_id.type, str)
            assert isinstance(op_id.value, int)
            # Value should be non-negative
            assert op_id.value >= 0
            # Type should be one of the known dialects
            assert op_id.type in ["torch", "coreai"], (
                f"Unknown operation ID type: {op_id.type}"
            )

        return WalkResult.ADVANCE

    # Walk all operations in the module
    simple_coreai_program._mlir_module.operation.walk(check_operation)

    # Note: Operation IDs may not be present after full compilation pipeline
    # This test validates the structure when they are present


async def test_enable_debug_info(simple_coreai_program: AIProgram) -> None:
    """Test that enable_debug_info preserves location information in ai program."""
    # Validate and count source info using the shared helper
    count = _validate_and_count_source_info(simple_coreai_program)

    # We should have found source info with debug info enabled
    assert count > 0, "Expected to find source info when debug info is enabled"


def test_multiple_sources_in_fused_location() -> None:
    """Test extracting multiple Source objects from a fused location."""
    ctx = Context()

    # Create multiple sources
    source1 = _mlir.Source()
    source1.name = "torch"
    source1.id = 1
    source1.identifiers = ["aten.add"]

    source2 = _mlir.Source()
    source2.name = "coreai"
    source2.id = 2
    source2.identifiers = ["conv"]

    # Create locations from each source
    loc1 = _mlir.create_location_from_source_info(ctx, source1)
    loc2 = _mlir.create_location_from_source_info(ctx, source2)

    # Fuse them together
    fused_loc = Location.fused([loc1, loc2], context=ctx)

    # Extract source info
    extracted = _mlir.get_source_info(fused_loc)

    # Should get both sources back
    assert len(extracted) == 2, f"Expected 2 sources, got {len(extracted)}"

    # Find each source by name (order may vary)
    sources_by_name: dict[str, Any] = {src.name: src for src in extracted}

    assert "torch" in sources_by_name, "Missing torch source"
    assert "coreai" in sources_by_name, "Missing coreai source"

    # Validate torch source
    torch_src = sources_by_name["torch"]
    assert torch_src.id == 1
    assert torch_src.identifiers == ["aten.add"]

    # Validate coreai source
    coreai_src = sources_by_name["coreai"]
    assert coreai_src.id == 2
    assert coreai_src.identifiers == ["conv"]


def test_source_with_zero_id() -> None:
    """Test that ID value of 0 is valid."""
    ctx = Context()

    source = _mlir.Source()
    source.name = "torch"
    source.id = 0  # Zero is a valid ID
    source.identifiers = ["op"]

    loc = _mlir.create_location_from_source_info(ctx, source)
    extracted = _mlir.get_source_info(loc)

    assert len(extracted) == 1
    assert extracted[0].id == 0


def test_source_with_special_characters_in_identifiers() -> None:
    """Test identifiers with special characters."""
    ctx = Context()

    source = _mlir.Source()
    source.name = "torch"
    source.id = 50
    source.identifiers = ["aten::add", "custom_ops.my_op", "namespace.class.method"]

    loc = _mlir.create_location_from_source_info(ctx, source)
    extracted = _mlir.get_source_info(loc)

    assert len(extracted) == 1
    assert extracted[0].identifiers == [
        "aten::add",
        "custom_ops.my_op",
        "namespace.class.method",
    ]


def test_empty_source_info_on_file_location() -> None:
    """Test that regular file locations return empty source info."""
    ctx = Context()

    # Create a simple file location (not a source location)
    file_loc = Location.file("test.py", line=10, col=5, context=ctx)

    # Should return empty list
    source_infos = _mlir.get_source_info(file_loc)
    assert len(source_infos) == 0


async def test_get_source_info_from_model(
    simple_coreai_program: AIProgram,
) -> None:
    """Test get_source_info extracts source metadata from locations in a real model."""
    # Validate and count source info
    count = _validate_and_count_source_info(simple_coreai_program)

    # We should have found at least some source info
    assert count > 0, "Expected to find source info in at least one operation"


def test_get_file_line_col_locations_fused() -> None:
    """Test extracting file/line/col from a fused location."""
    ctx = Context()

    # Create multiple file locations
    loc1 = Location.file("file1.py", line=10, col=5, context=ctx)
    loc2 = Location.file("file2.py", line=20, col=10, context=ctx)

    # Fuse them together
    fused_loc = Location.fused([loc1, loc2], context=ctx)

    # Extract file/line/col locations
    file_locs = _mlir.get_file_line_col_locations(fused_loc)

    # Should get two locations back
    assert len(file_locs) == 2, f"Expected 2 locations, got {len(file_locs)}"

    # Check both locations (order may vary, so check by filename)
    filenames = {loc.filename for loc in file_locs}
    assert "file1.py" in filenames, "Expected to find file1.py"
    assert "file2.py" in filenames, "Expected to find file2.py"

    # Verify each location
    for loc in file_locs:
        if loc.filename == "file1.py":
            assert loc.line == 10
            assert loc.col == 5
        elif loc.filename == "file2.py":
            assert loc.line == 20
            assert loc.col == 10


async def test_get_stack_trace_from_coreai_program(
    complex_coreai_program: AIProgram,
) -> None:
    """Test extracting stack trace from an ai program with debug info."""
    # Track if we found any stack traces
    found_stack_traces: list[int] = [0]

    def check_operation(operation: Operation) -> WalkResult:
        # Get stack trace from operation location
        stack_trace = _mlir.get_stack_trace(operation.location)

        # Validate structure if stack trace is present
        if stack_trace:
            found_stack_traces[0] += 1

            # Stack trace should be a list of strings
            assert isinstance(stack_trace, list), "Stack trace should be a list"

            for entry in stack_trace:
                assert isinstance(entry, str), "Stack trace entry should be a string"
                assert len(entry) > 0, "Stack trace entry should not be empty"

        return WalkResult.ADVANCE

    # Walk all operations in the module
    complex_coreai_program._mlir_module.operation.walk(check_operation)

    # We should find at least some operations with stack traces
    # Note: Not all operations may have stack traces, but with debug info enabled
    # we should find at least some
    assert found_stack_traces[0] > 0, (
        "Expected to find at least one operation with stack trace when debug info is enabled"
    )
