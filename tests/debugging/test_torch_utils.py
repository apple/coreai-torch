# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for torch_utils module."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.export import export
from torch.fx import Node

from coreai_torch._debug_locations import _DebugInfoRecorder
from coreai_torch.converter import TorchConverter
from coreai_torch.debugging.torch_utils import (
    fetch_intermediate_values,
    load_intermediates,
    save_intermediates,
)

from .test_model import SimpleLinearModel, TwoLayerMLPModel, get_example_inputs


@pytest.fixture
def simple_model_and_input() -> tuple[SimpleLinearModel, tuple[torch.Tensor, ...]]:
    """Create a simple model and example input."""
    model = SimpleLinearModel()
    example_input = tuple(get_example_inputs(SimpleLinearModel).values())
    return model, example_input


@pytest.fixture
def two_layer_model_and_input() -> tuple[TwoLayerMLPModel, tuple[torch.Tensor, ...]]:
    """Create a two-layer model and example input."""
    model = TwoLayerMLPModel()
    example_input = tuple(get_example_inputs(TwoLayerMLPModel).values())
    return model, example_input


def test_fetch_intermediate_values_stores_all_nodes(
    simple_model_and_input: tuple[SimpleLinearModel, tuple[torch.Tensor, ...]],
) -> None:
    """Test that fetch_intermediate_values captures all intermediate values."""
    model, example_input = simple_model_and_input
    exported = export(model, example_input)

    # Store all intermediates
    intermediates = {}

    def store_callback(node: Node, result: Any) -> None:
        intermediates[node.name] = result

    output = fetch_intermediate_values(exported, example_input, store_callback)

    # Should have captured some intermediate values
    assert len(intermediates) > 0
    # Output should be a tensor
    assert isinstance(output, torch.Tensor)


def test_fetch_intermediate_values_with_filter(
    two_layer_model_and_input: tuple[TwoLayerMLPModel, tuple[torch.Tensor, ...]],
) -> None:
    """Test fetch_intermediate_values with a filter."""
    model, example_input = two_layer_model_and_input
    exported = export(model, example_input)

    # Store only tensor results
    tensor_intermediates = {}

    def tensor_callback(node: Node, result: Any) -> None:
        if isinstance(result, torch.Tensor):
            tensor_intermediates[node.name] = result

    fetch_intermediate_values(exported, example_input, tensor_callback)

    # All stored values should be tensors
    for value in tensor_intermediates.values():
        assert isinstance(value, torch.Tensor)


def test_save_and_load_intermediates(
    simple_model_and_input: tuple[SimpleLinearModel, tuple[torch.Tensor, ...]],
) -> None:
    """Test dumping and loading intermediates."""
    model, example_input = simple_model_and_input
    exported = export(model, example_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Dump intermediates
        metadata_path = save_intermediates(exported, example_input, tmpdir)

        # Check that files were created
        assert Path(metadata_path).exists()
        assert Path(metadata_path).name == "metadata.json"

        # Load intermediates back
        loaded = load_intermediates(metadata_path)

        # Check structure - should be DebugTrace dataclass
        assert hasattr(loaded, "inputs")
        assert hasattr(loaded, "intermediates")
        assert hasattr(loaded, "outputs")

        # Should have loaded some intermediate values
        assert len(loaded.intermediates) > 0

        # Should have inputs
        assert len(loaded.inputs) > 0

        # Should have outputs
        assert len(loaded.outputs) > 0

        # All values should be tensors
        for name, tensor in loaded.intermediates.items():
            assert isinstance(tensor, torch.Tensor)
            assert isinstance(name, str)


def test_save_intermediates_with_filter(
    two_layer_model_and_input: tuple[TwoLayerMLPModel, tuple[torch.Tensor, ...]],
) -> None:
    """Test dumping intermediates with a node filter."""
    model, example_input = two_layer_model_and_input
    exported = export(model, example_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Only dump nodes containing 'linear' in the name
        metadata_path = save_intermediates(
            exported,
            example_input,
            tmpdir,
            node_filter=lambda node, result: (
                "linear" in node.name and isinstance(result, torch.Tensor)
            ),
        )

        # Load back
        loaded = load_intermediates(metadata_path)

        # Check that only linear nodes were saved in intermediates
        for name in loaded.intermediates.keys():
            assert "linear" in name.lower() or "linear" in name


def test_load_intermediates_with_device(
    simple_model_and_input: tuple[SimpleLinearModel, tuple[torch.Tensor, ...]],
) -> None:
    """Test loading intermediates to a specific device."""
    model, example_input = simple_model_and_input
    exported = export(model, example_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        metadata_path = save_intermediates(exported, example_input, tmpdir)

        # Load to CPU (should always work)
        loaded = load_intermediates(metadata_path, device="cpu")

        # Check all sections
        for tensor in loaded.inputs.values():
            assert tensor.device.type == "cpu"
        for tensor in loaded.intermediates.values():
            assert tensor.device.type == "cpu"
        for tensor in loaded.outputs.values():
            assert tensor.device.type == "cpu"


def test_load_intermediates_from_directory(
    simple_model_and_input: tuple[SimpleLinearModel, tuple[torch.Tensor, ...]],
) -> None:
    """Test loading intermediates by passing .aimodelintermediates directory path."""
    model, example_input = simple_model_and_input
    exported = export(model, example_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        save_intermediates(exported, example_input, tmpdir)

        # Load using .aimodelintermediates directory path
        aimodel_dir = Path(tmpdir) / "main.aimodelintermediates"
        loaded = load_intermediates(aimodel_dir)

        assert len(loaded.intermediates) > 0


def test_metadata_contains_expected_fields(
    simple_model_and_input: tuple[SimpleLinearModel, tuple[torch.Tensor, ...]],
) -> None:
    """Test that saved metadata contains expected fields."""
    model, example_input = simple_model_and_input
    exported = export(model, example_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = save_intermediates(exported, example_input, tmpdir)

        # Load and check metadata structure
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check top-level structure
        assert "inputs" in metadata
        assert "intermediates" in metadata
        assert "outputs" in metadata

        # Check intermediates section has entries
        assert len(metadata["intermediates"]) > 0

        # Check expected fields in first intermediate entry
        first_entry = next(iter(metadata["intermediates"].values()))
        assert "node_name" in first_entry
        assert "node_op" in first_entry
        assert "node_target" in first_entry
        assert "data_file" in first_entry
        assert "shape" in first_entry
        assert "torch_dtype" in first_entry
        assert "numel" in first_entry

        # Check inputs section
        assert len(metadata["inputs"]) > 0
        first_input = next(iter(metadata["inputs"].values()))
        assert "data_file" in first_input
        assert "shape" in first_input

        # Check outputs section
        assert len(metadata["outputs"]) > 0
        first_output = next(iter(metadata["outputs"].values()))
        assert "data_file" in first_output
        assert "shape" in first_output


def test_dtype_preservation(
    simple_model_and_input: tuple[SimpleLinearModel, tuple[torch.Tensor, ...]],
) -> None:
    """Test that tensor dtypes are preserved during dump/load."""
    model, example_input = simple_model_and_input
    exported = export(model, example_input)

    # Collect original dtypes
    original_dtypes = {}

    def collect_dtypes(node: Node, result: Any) -> None:
        if isinstance(result, torch.Tensor):
            original_dtypes[node.name] = result.dtype

    fetch_intermediate_values(exported, example_input, collect_dtypes)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Dump and load
        metadata_path = save_intermediates(exported, example_input, tmpdir)
        loaded = load_intermediates(metadata_path)

        # Check that dtypes match for intermediates
        for name, tensor in loaded.intermediates.items():
            if name in original_dtypes:
                assert tensor.dtype == original_dtypes[name], (
                    f"Dtype mismatch for {name}: "
                    f"expected {original_dtypes[name]}, got {tensor.dtype}"
                )


def test_save_intermediates_with_coreai_program_program(
    two_layer_model_and_input: tuple[TwoLayerMLPModel, tuple[torch.Tensor, ...]],
) -> None:
    """Test dumping intermediates with coreai_program program to extract mappings."""
    # Create a simple model
    model = two_layer_model_and_input[0]
    inputs = {"x": two_layer_model_and_input[1][0]}
    exported_program = torch.export.export(model, args=(inputs["x"],))
    exported_program = exported_program.run_decompositions()

    converter: TorchConverter = TorchConverter()
    converter._debug_info_recorder.config = _DebugInfoRecorder.Config(
        include_stack_trace=True,
        verify_debuginfo_locations=True,
    )
    converter.add_exported_program(exported_program, entrypoint_name="main")
    coreai_program = converter.to_coreai()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save intermediates with Core AI program
        metadata_path = save_intermediates(
            exported_program,
            (inputs["x"],),
            tmpdir,
            coreai_program=coreai_program,
        )

        # Load intermediates back
        loaded = load_intermediates(metadata_path)

        # Check that intermediates were saved and loaded successfully
        assert len(loaded.intermediates) > 0, "Expected to find intermediates"

        # Check that mappings were loaded as proper dataclasses
        assert loaded.mappings is not None, "Expected mappings to be present"
        assert hasattr(loaded.mappings, "sources")
        assert hasattr(loaded.mappings, "outputs")

        # Verify sources is a dict of SourceInfo objects
        assert isinstance(loaded.mappings.sources, dict)
        for dialect, ops in loaded.mappings.sources.items():
            assert isinstance(ops, dict)
            for source_info in ops.values():
                # Should be a SourceInfo dataclass instance
                assert hasattr(source_info, "dialect")
                assert hasattr(source_info, "id")
                assert hasattr(source_info, "identifiers")
                assert source_info.dialect == dialect
                assert isinstance(source_info.identifiers, (list, tuple))

        # Verify outputs is a list of OutputMapping objects
        assert isinstance(loaded.mappings.outputs, (list, tuple))
        for output_mapping in loaded.mappings.outputs:
            # Should be an OutputMapping dataclass instance
            assert hasattr(output_mapping, "source_level")
            assert hasattr(output_mapping, "source_op_id")
            assert hasattr(output_mapping, "source_output")
            assert hasattr(output_mapping, "target_level")
            assert hasattr(output_mapping, "target_op_id")
            assert hasattr(output_mapping, "target_output")

        # Load metadata JSON directly to check serialized format
        with open(metadata_path) as f:  # noqa: ASYNC230
            metadata = json.load(f)

        # Check that mappings section exists in JSON
        assert "mappings" in metadata, "Expected mappings in metadata"
        mappings = metadata["mappings"]

        # Check JSON sources structure
        if "sources" in mappings:
            sources = mappings["sources"]
            assert isinstance(sources, dict)
            # Should have dialect keys like "torch", "coreai", etc.
            for ops in sources.values():
                assert isinstance(ops, dict)
                # Each operation should have dialect, id, and identifiers
                for op_info in ops.values():
                    assert "dialect" in op_info
                    assert "id" in op_info
                    assert "identifiers" in op_info
                    assert isinstance(op_info["identifiers"], (list, tuple))

        # Check JSON outputs structure
        if "outputs" in mappings:
            outputs_mappings = mappings["outputs"]
            assert isinstance(outputs_mappings, (list, tuple))
            # Each mapping should have source and target info
            for mapping in outputs_mappings:
                assert "source_level" in mapping
                assert "source_op_id" in mapping
                assert "source_output" in mapping
                assert "target_level" in mapping
                assert "target_op_id" in mapping
                assert "target_output" in mapping
