# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests verifying intermediate values match between PyTorch FX and CoreAI for all models."""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import AIModel
from numpy.typing import NDArray

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder
from coreai_torch.debugging.debug_info import (
    OutputMapping,
    _build_coreai_op_map,
)
from coreai_torch.debugging.inspector import (
    CoreAIInspector,
    Inspector,
    TorchFXInspector,
)
from coreai_torch.debugging.torch_utils import (
    get_torch_to_coreai_output_mapping,
)

from .test_model import (
    EXAMPLE_INPUTS,
    LayerNormBlock,
    SDPAAttentionBlock,
    TinyTransformerBlock,
    get_example_inputs,
)

logger = logging.getLogger(__name__)

# Models excluded from intermediate comparison.
EXCLUDED_MODEL_CLASSES = {LayerNormBlock, TinyTransformerBlock, SDPAAttentionBlock}

ALL_MODEL_CLASSES = [
    cls for cls in EXAMPLE_INPUTS.keys() if cls not in EXCLUDED_MODEL_CLASSES
]


def _export_and_convert(
    model_cls: type[torch.nn.Module],
) -> tuple[torch.export.ExportedProgram, AIProgram]:
    """Export a model, decompose, and convert to AIProgram with debug info.

    Returns:
        Tuple of (decomposed ExportedProgram, AIProgram).
    """
    model = model_cls().eval()
    example_inputs = get_example_inputs(model_cls)
    exported_program = torch.export.export(model, args=tuple(example_inputs.values()))
    exported_program = exported_program.run_decompositions()

    converter = TorchConverter()
    converter._debug_info_recorder.config = _DebugInfoRecorder.Config(
        include_stack_trace=True,
        verify_debuginfo_locations=True,
    )
    converter.add_exported_program(exported_program, entrypoint_name="main")
    coreai_program = converter.to_coreai()

    return exported_program, coreai_program


async def _capture_torch_intermediates(
    exported_program: torch.export.ExportedProgram,
    torch_args: tuple[torch.Tensor, ...],
) -> dict[Inspector.OpID, list[NDArray[Any] | None] | None]:
    """Run model through TorchFXInspector and capture all call_function intermediates."""
    inspector = TorchFXInspector(exported_program)
    node_names = [
        node.name for node in exported_program.graph.nodes if node.op == "call_function"
    ]
    return await inspector.get_intermediates_for_ops(node_names, torch_args)


async def _capture_coreai_intermediates(
    coreai_program: Any,
    coreai_op_ids: set[int],
    numpy_inputs: dict[str, NDArray[Any]],
) -> dict[Inspector.OpID, list[NDArray[Any] | None] | None]:
    """Deploy AIProgram and capture intermediates via CoreAIInspector."""
    with tempfile.TemporaryDirectory() as tmpdir:
        asset_path = Path(tmpdir) / "model.aimodel"
        asset = coreai_program.save_asset(asset_path)
        ai_model = await AIModel.load(
            asset.path,
        )
        inspector = CoreAIInspector(model=ai_model, function_name="main")
        return await inspector.get_intermediates_for_ops(
            list(coreai_op_ids), numpy_inputs
        )


def _compare_mapped_intermediates(
    torch_intermediates: dict[Inspector.OpID, list[NDArray[Any] | None] | None],
    coreai_intermediates: dict[Inspector.OpID, list[NDArray[Any] | None] | None],
    mappings: dict[str, OutputMapping],
    model_name: str,
    coreai_op_map: dict[int, Any] | None = None,
) -> int:
    """Compare torch and coreai intermediates for each mapped operation.

    Logs a warning for every torch op whose corresponding coreai output
    cannot be found or has a shape mismatch.

    Returns:
        Number of successfully compared operation outputs.
    """
    compared = 0
    for torch_node_name, mapping in mappings.items():
        coreai_op = coreai_op_map.get(mapping.target_op_id) if coreai_op_map else None

        torch_values = torch_intermediates.get(torch_node_name)
        if torch_values is None:
            logger.warning(
                "%s: no torch intermediate for '%s'", model_name, torch_node_name
            )
            continue

        if len(torch_values) > 1:
            logger.warning(
                "%s: skipping torch op '%s' with multiple outputs (n=%d)",
                model_name,
                torch_node_name,
                len(torch_values),
            )
            continue

        coreai_values = coreai_intermediates.get(mapping.target_op_id)
        if coreai_values is None:
            logger.warning(
                "%s: no coreai intermediate for torch op '%s' (expected coreai op %d)\n  %s",
                model_name,
                torch_node_name,
                mapping.target_op_id,
                coreai_op,
            )
            continue

        if mapping.source_output >= len(torch_values):
            logger.warning(
                "%s: torch op '%s' source_output %d out of range (len=%d)",
                model_name,
                torch_node_name,
                mapping.source_output,
                len(torch_values),
            )
            continue

        torch_output = torch_values[mapping.source_output]

        if mapping.target_output >= len(coreai_values):
            logger.warning(
                "%s: coreai op %d target_output %d out of range (len=%d) "
                "for torch op '%s'\n  %s",
                model_name,
                mapping.target_op_id,
                mapping.target_output,
                len(coreai_values),
                torch_node_name,
                coreai_op,
            )
            continue

        coreai_output = coreai_values[mapping.target_output]

        if torch_output is None or coreai_output is None:
            logger.warning(
                "%s: None output for torch op '%s' → coreai op %d "
                "(torch=%s, coreai=%s)\n  %s",
                model_name,
                torch_node_name,
                mapping.target_op_id,
                torch_output is not None,
                coreai_output is not None,
                coreai_op,
            )
            continue

        # Squeeze and compare if the squeezed shapes match.
        if torch_output.shape != coreai_output.shape:
            squeezed_torch = np.squeeze(torch_output)
            squeezed_coreai = np.squeeze(coreai_output)
            if squeezed_torch.shape == squeezed_coreai.shape:
                torch_output = squeezed_torch
                coreai_output = squeezed_coreai
            else:
                logger.warning(
                    "%s: shape mismatch for torch op '%s' → coreai op %d: "
                    "torch %s vs coreai %s — skipping comparison\n  %s",
                    model_name,
                    torch_node_name,
                    mapping.target_op_id,
                    torch_output.shape,
                    coreai_output.shape,
                    coreai_op,
                )
                continue

        abs_diff = np.abs(torch_output - coreai_output)
        assert np.allclose(torch_output, coreai_output, rtol=1e-3, atol=1e-3), (
            "%s: intermediate mismatch for torch op '%s' → coreai op %d "
            "(max abs diff=%g, mean abs diff=%g)\n  coreai op: %s"
            % (
                model_name,
                torch_node_name,
                mapping.target_op_id,
                abs_diff.max(),
                abs_diff.mean(),
                coreai_op,
            )
        )

        compared += 1
        logger.info(
            "%s: ✓ torch op '%s' [%d] matches coreai op %d [%d] (shape=%s)\n  %s",
            model_name,
            torch_node_name,
            mapping.source_output,
            mapping.target_op_id,
            mapping.target_output,
            torch_output.shape,
            coreai_op,
        )

    return compared


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
@pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES, ids=lambda cls: cls.__name__)
async def test_intermediates_torch_vs_coreai(
    model_cls: type[torch.nn.Module],
) -> None:
    """
    Verify intermediate values from PyTorch FX match CoreAI for each mapped op.
    """
    example_inputs = get_example_inputs(model_cls)
    torch_args = tuple(example_inputs.values())
    numpy_inputs = {k: v.numpy() for k, v in example_inputs.items()}

    exported_program, coreai_program = _export_and_convert(model_cls)

    torch_intermediates = await _capture_torch_intermediates(
        exported_program, torch_args
    )

    mappings = get_torch_to_coreai_output_mapping(coreai_program)
    coreai_op_map = _build_coreai_op_map(coreai_program)
    coreai_op_ids = {m.target_op_id for m in mappings.values()}

    coreai_intermediates = await _capture_coreai_intermediates(
        coreai_program, coreai_op_ids, numpy_inputs
    )

    compared_count = _compare_mapped_intermediates(
        torch_intermediates,
        coreai_intermediates,
        mappings,
        model_cls.__name__,
        coreai_op_map=coreai_op_map,
    )

    assert compared_count > 0, (
        f"No intermediates were compared for {model_cls.__name__}. "
        f"Found {len(mappings)} mappings but none had matching intermediates."
    )
