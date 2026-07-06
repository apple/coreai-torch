# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test graph diff functionality."""

import io
import sys

import pytest
import torch
from coreai.authoring import AIProgram

from coreai_torch.converter import TorchConverter
from coreai_torch.debugging.graph_diff import (
    compute_coreai_program_diff,
    compute_exported_program_diff,
    compute_per_graph_diff,
    format_multi_graph_diff,
    write_diff,
)

from .test_model import (
    ExtraLayerModel,
    ModifiedActivationModel,
    ThreeLinearModel,
    TwoLinearSkipModel,
    get_example_inputs,
)


async def _create_coreai_program_from_model(
    exported_program: torch.export.ExportedProgram,
) -> AIProgram:
    """Create a coreai_program program from an exported program."""
    converter: TorchConverter = TorchConverter(mode=TorchConverter.Mode.DEBUG)
    converter.add_exported_program(exported_program, entrypoint_name="main")
    coreai_program = converter.to_coreai()
    return coreai_program


@pytest.mark.asyncio
async def test_identical_multilayer_models() -> None:
    """Test diff of two identical multi-layer models."""
    model = ThreeLinearModel().eval()
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    # Export same model twice
    exported_1 = torch.export.export(model, args)
    exported_1 = exported_1.run_decompositions()

    exported_2 = torch.export.export(model, args)
    exported_2 = exported_2.run_decompositions()

    # Create coreai_program programs
    source = await _create_coreai_program_from_model(exported_1)
    target = await _create_coreai_program_from_model(exported_2)

    # Compute diff
    diff = compute_coreai_program_diff(source, target)

    # Should be isomorphic
    assert diff.is_isomorphic
    assert diff.summary.mapped_node_count > 0


@pytest.mark.asyncio
async def test_modified_activation() -> None:
    """Test diff when one activation function is changed."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    # Export models with different activation
    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = ModifiedActivationModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    # Create coreai_program programs
    source = await _create_coreai_program_from_model(exported_1)
    target = await _create_coreai_program_from_model(exported_2)

    # Compute diff
    diff = compute_coreai_program_diff(source, target)

    # Should NOT be isomorphic (different activation function)
    assert not diff.is_isomorphic
    assert (
        diff.summary.unmapped_source_node_count > 0
        or diff.summary.unmapped_target_node_count > 0
    )


@pytest.mark.asyncio
async def test_missing_layer() -> None:
    """Test diff when target is missing a middle layer."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    # Export models
    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = TwoLinearSkipModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    # Create coreai_program programs
    source = await _create_coreai_program_from_model(exported_1)
    target = await _create_coreai_program_from_model(exported_2)

    # Compute diff
    diff = compute_coreai_program_diff(source, target)

    # Should NOT be isomorphic
    assert not diff.is_isomorphic

    # Source should have more nodes
    assert diff.summary.source_node_count > diff.summary.target_node_count


@pytest.mark.asyncio
async def test_extra_layer() -> None:
    """Test diff when target has an extra layer."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    # Export models
    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = ExtraLayerModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    # Create coreai_program programs
    source = await _create_coreai_program_from_model(exported_1)
    target = await _create_coreai_program_from_model(exported_2)

    # Compute diff
    diff = compute_coreai_program_diff(source, target)

    # Should NOT be isomorphic
    assert not diff.is_isomorphic

    # Target should have more nodes
    assert diff.summary.target_node_count > diff.summary.source_node_count


@pytest.mark.asyncio
async def test_diff_shows_common_subgraph() -> None:
    """Test that diff shows aligned operations for common parts."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    # Models share first two layers
    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = ModifiedActivationModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    source = await _create_coreai_program_from_model(exported_1)
    target = await _create_coreai_program_from_model(exported_2)

    diff = compute_coreai_program_diff(source, target)

    # Should show mappings
    assert diff.summary.mapped_node_count > 0 or not diff.is_isomorphic


@pytest.mark.asyncio
async def test_diff_output_structure() -> None:
    """Test that diff output has expected structure."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = TwoLinearSkipModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    source = await _create_coreai_program_from_model(exported_1)
    target = await _create_coreai_program_from_model(exported_2)

    diff = compute_coreai_program_diff(source, target)

    # Validate diff object
    assert diff.summary.source_node_count > 0
    assert diff.summary.target_node_count > 0
    assert not diff.is_isomorphic


@pytest.mark.asyncio
async def test_diff_invalid_entry_point() -> None:
    """Test that diff raises error for invalid entry point."""
    model = ThreeLinearModel().eval()
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    exported = torch.export.export(model, args)
    exported = exported.run_decompositions()

    source = await _create_coreai_program_from_model(exported)
    target = await _create_coreai_program_from_model(exported)

    # Should raise ValueError for non-existent entry point
    with pytest.raises(ValueError, match=r"Entry point .* not found"):
        compute_coreai_program_diff(
            source,
            target,
            entry_point="nonexistent_function",
        )


# Tests for compute_exported_program_diff
def test_exported_program_identical_models() -> None:
    """Test diff of two identical ExportedPrograms."""
    model = ThreeLinearModel().eval()
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    # Export same model twice
    exported_1 = torch.export.export(model, args)
    exported_1 = exported_1.run_decompositions()

    exported_2 = torch.export.export(model, args)
    exported_2 = exported_2.run_decompositions()

    # Compute diff
    diff = compute_exported_program_diff(exported_1, exported_2)

    # Should be isomorphic
    assert diff.is_isomorphic
    assert diff.summary.mapped_node_count > 0


def test_exported_program_modified_activation() -> None:
    """Test diff of ExportedPrograms with different activation."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = ModifiedActivationModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    # Compute diff
    diff = compute_exported_program_diff(exported_1, exported_2)

    # Should NOT be isomorphic (different activation function)
    assert not diff.is_isomorphic
    assert (
        diff.summary.unmapped_source_node_count > 0
        or diff.summary.unmapped_target_node_count > 0
    )


def test_exported_program_missing_layer() -> None:
    """Test diff of ExportedPrograms when target is missing a layer."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = TwoLinearSkipModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    # Compute diff
    diff = compute_exported_program_diff(exported_1, exported_2)

    # Should NOT be isomorphic
    assert not diff.is_isomorphic

    # Source should have more nodes
    assert diff.summary.source_node_count > diff.summary.target_node_count


def test_exported_program_extra_layer() -> None:
    """Test diff of ExportedPrograms when target has an extra layer."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = ExtraLayerModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    # Compute diff
    diff = compute_exported_program_diff(exported_1, exported_2)

    # Should NOT be isomorphic
    assert not diff.is_isomorphic

    # Target should have more nodes
    assert diff.summary.target_node_count > diff.summary.source_node_count


def test_exported_program_write_diff() -> None:
    """Test that write_diff works with ExportedProgram diffs."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = ModifiedActivationModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    # Compute diff
    diff = compute_exported_program_diff(exported_1, exported_2)

    # Write diff to a StringIO stream
    output = io.StringIO()
    write_diff(diff, diff.source_graph, diff.target_graph, output=output)
    diff_text = output.getvalue()

    # Verify the formatted text contains expected sections
    assert "GRAPH DIFF" in diff_text
    assert "Summary:" in diff_text
    assert "Operations Diff Table:" in diff_text


def test_exported_program_write_diff_to_stdout() -> None:
    """Test that write_diff can write to sys.stdout."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = ModifiedActivationModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    # Compute diff
    diff = compute_exported_program_diff(exported_1, exported_2)

    # Write diff to sys.stdout explicitly
    write_diff(diff, diff.source_graph, diff.target_graph, output=sys.stdout)


@pytest.mark.asyncio
async def test_compute_per_graph_diff() -> None:
    """Test composite-aware per-graph diffing."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = ModifiedActivationModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    source = await _create_coreai_program_from_model(exported_1)
    target = await _create_coreai_program_from_model(exported_2)

    # Compute per-graph diff
    results = compute_per_graph_diff(source, target)

    # Should have at least the main graph
    assert len(results) >= 1
    assert results[0][0] == "main"

    # Main diff should exist
    main_diff = results[0][1]
    assert main_diff is not None


@pytest.mark.asyncio
async def test_format_multi_graph_diff() -> None:
    """Test multi-graph diff formatting."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = ModifiedActivationModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    source = await _create_coreai_program_from_model(exported_1)
    target = await _create_coreai_program_from_model(exported_2)

    # Compute per-graph diff and format
    results = compute_per_graph_diff(source, target)
    text = format_multi_graph_diff(results)

    # Verify formatted output contains expected sections
    assert "GRAPH: main" in text
    assert "Summary:" in text


@pytest.mark.asyncio
async def test_compute_coreai_program_diff_all_graphs() -> None:
    """Test diffing all graphs in the module (entry_point=None)."""
    example_inputs = get_example_inputs(ThreeLinearModel)
    args = tuple(example_inputs.values())

    model1 = ThreeLinearModel().eval()
    exported_1 = torch.export.export(model1, args)
    exported_1 = exported_1.run_decompositions()

    model2 = ModifiedActivationModel().eval()
    exported_2 = torch.export.export(model2, args)
    exported_2 = exported_2.run_decompositions()

    source = await _create_coreai_program_from_model(exported_1)
    target = await _create_coreai_program_from_model(exported_2)

    # Compare all graphs (entry_point=None)
    diff = compute_coreai_program_diff(source, target, entry_point=None)

    # Should have computed a diff
    assert diff.summary.source_node_count > 0
    assert diff.summary.target_node_count > 0
