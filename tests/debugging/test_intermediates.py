# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests verifying intermediate values match between PyTorch FX and CoreAI for all models."""

import io
import logging
import sys

import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import SpecializationOptions

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder
from coreai_torch.debugging.debug_info import (
    _build_coreai_op_map,
)
from coreai_torch.debugging.intermediates import compare_program_intermediates

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


@pytest.mark.skipif(sys.platform != "darwin", reason="Test only runs on macOS")
@pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES, ids=lambda cls: cls.__name__)
async def test_intermediates_torch_vs_coreai(
    model_cls: type[torch.nn.Module],
    specialization_options: SpecializationOptions | None,
) -> None:
    """
    Verify intermediate values from PyTorch FX match CoreAI for each mapped op.
    """
    exported_program, coreai_program = _export_and_convert(model_cls)
    coreai_op_map = _build_coreai_op_map(coreai_program)

    report = await compare_program_intermediates(
        exported_program,
        coreai_program,
        get_example_inputs(model_cls),
        specialization_options=specialization_options,
    )

    # Log the whole table, not just the failures: which operations could not be compared and
    # why is the part that explains a low count, and it is invisible in an assertion message.
    rendered = io.StringIO()
    report.write_to(rendered)
    logger.info("%s intermediates:\n%s", model_cls.__name__, rendered.getvalue())

    assert not report.failures, "%s: %d intermediate(s) differ, worst first:\n%s" % (
        model_cls.__name__,
        len(report.failures),
        "\n".join(
            f"  torch '{entry.torch_node_name}' vs coreai op {entry.coreai_op_id}: "
            f"max abs diff={entry.max_diff:g} at shape {entry.shape}\n"
            f"    {coreai_op_map.get(entry.coreai_op_id)}"
            for entry in report.failures
        ),
    )

    assert report.compared, (
        f"No intermediates were compared for {model_cls.__name__}, so nothing was "
        f"verified. By status: {report.summary()}"
    )
