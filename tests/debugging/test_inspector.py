# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test inspector implementations."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import AIModel

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder
from coreai_torch.debugging.inspector import (
    CachingInspector,
    CoreAIInspector,
    TorchFXInspector,
)
from coreai_torch.debugging.torch_utils import get_torch_to_coreai_output_mapping

from .test_model import LinearMulAddModel, get_example_inputs


@pytest.fixture
async def simple_coreai_program() -> AIProgram:
    """Fixture that provides a AIProgram with debug info enabled."""
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


async def test_torch_fx_inspector() -> None:
    """Test _TorchFXInspector with a simple torch model."""
    model = LinearMulAddModel().eval()
    example_inputs = get_example_inputs(LinearMulAddModel)
    args = tuple(example_inputs.values())

    exported_program = torch.export.export(model, args)
    exported_program = exported_program.run_decompositions()

    inspector = TorchFXInspector(exported_program)

    # Get all node names from the graph
    node_names = [
        node.name for node in exported_program.graph.nodes if node.op == "call_function"
    ]

    # Request intermediates for all operations to check y and z values
    results = await inspector.get_intermediates_for_ops(node_names, args)

    # Check each operation's result
    for op_name in node_names:
        assert op_name in results
        assert results[op_name] is not None
        assert isinstance(results[op_name], list)
        assert len(results[op_name]) > 0
        assert isinstance(results[op_name][0], np.ndarray)


async def test_caching_inspector() -> None:
    """Test CachingInspector with LRU caching."""
    model = LinearMulAddModel().eval()
    example_inputs = get_example_inputs(LinearMulAddModel)
    args = tuple(example_inputs.values())

    exported_program = torch.export.export(model, args)
    base_inspector = TorchFXInspector(exported_program)

    # Test with LRU cache of size 2
    caching_inspector = CachingInspector(base_inspector, max_cache_size=2)

    node_names = [
        node.name for node in exported_program.graph.nodes if node.op == "call_function"
    ]

    # First call - should populate cache
    results1 = await caching_inspector.get_intermediates_for_ops([node_names[0]], args)

    # Second call - should use cache
    results2 = await caching_inspector.get_intermediates_for_ops([node_names[0]], args)

    assert results1.keys() == results2.keys()
    assert node_names[0] in results1

    # Verify cached results match
    np.testing.assert_array_equal(
        results1[node_names[0]][0],
        results2[node_names[0]][0],
    )


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Requires loading a runtime asset (AIModel.load); only supported on macOS",
)
async def test_coreai_inspector(simple_coreai_program: AIProgram) -> None:
    """Test _CoreAIInspector with a deployed model."""
    # Get torch -> coreai mappings
    mappings = get_torch_to_coreai_output_mapping(simple_coreai_program)

    # Get coreai operation IDs from mappings
    coreai_op_ids = set()
    for mapping in mappings.values():
        coreai_op_ids.add(mapping.target_op_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        asset_path = Path(tmpdir) / "test_model.aimodel"

        # Create asset from AIProgram and load model from asset
        asset = simple_coreai_program.save_asset(asset_path)
        ai_model = await AIModel.load(asset.path)

        # Create inspector
        inspector = CoreAIInspector(
            model=ai_model,
            function_name="main",
        )

        # Create inputs matching the export shape (inspector will convert to NDArray)
        example_inputs = get_example_inputs(LinearMulAddModel)
        inputs = {k: v.numpy() for k, v in example_inputs.items()}

        # Get intermediates for first coreai operation
        results = await inspector.get_intermediates_for_ops(
            coreai_op_ids,
            inputs,
        )

        # Check that results contain numpy arrays
        for result in results.values():
            if result is None:
                continue

            assert isinstance(result, list)
            assert len(result) > 0
            for item in result:
                assert isinstance(item, np.ndarray)


async def test_torch_to_coreai_mappings(
    simple_coreai_program: AIProgram,
) -> None:
    """Test get_torch_to_coreai_output_mapping for torch -> coreai mappings."""
    # Get torch to coreai mappings
    mappings = get_torch_to_coreai_output_mapping(simple_coreai_program)

    # Should have found some mappings
    assert len(mappings) > 0, "Expected to find torch -> coreai mappings"

    # Verify structure of mappings
    for identifier, mapping in mappings.items():
        assert isinstance(identifier, str)
        assert len(identifier) > 0
        assert mapping.source_level == "torch"
        assert mapping.target_level == "coreai"
        assert mapping.source_op_id >= 0
        assert mapping.target_op_id >= 0
        assert mapping.source_output >= 0
        assert mapping.target_output >= 0
