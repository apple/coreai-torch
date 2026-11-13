# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the generic source annotator."""

from io import StringIO

import pytest
import torch
from coreai._compiler.ir import Operation
from coreai.authoring import AIProgram

from coreai_torch.converter import TorchConverter, _DebugInfoRecorder
from coreai_torch.debugging.debug_info import _build_coreai_op_map
from coreai_torch.debugging.source_annotator import (
    _annotate_source,
    _Annotation,
    _SourceAnnotator,
    _TextAnnotation,
    _walk_operations,
    module_paths_by_op_id,
    operation_stack_traces,
)
from coreai_torch.debugging.utils import STRUCTURAL_OPS

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


def test_text_annotation_write_renders_comment() -> None:
    """_TextAnnotation renders a colored comment line, plus a plain variant."""
    buffer = StringIO()
    _TextAnnotation("hello", color="").write(buffer)
    assert buffer.getvalue() == "# hello\n"

    colored = StringIO()
    _TextAnnotation("hi").write(colored)
    out = colored.getvalue()
    assert "# hi" in out
    assert out.endswith("\n")


def test_text_annotation_satisfies_protocol() -> None:
    """_TextAnnotation is recognized as an _Annotation via the runtime protocol."""
    assert isinstance(_TextAnnotation("x"), _Annotation)


async def test_annotate_source_uses_callback_per_operation(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """_annotate_source invokes the callback for each operation and writes output."""
    seen_ops: list[Operation] = []

    def annotate(operation: Operation) -> _Annotation:
        seen_ops.append(operation)
        return _TextAnnotation(f"op={operation.name}")

    buffer = StringIO()
    _annotate_source(hierarchical_coreai_program, annotate, buffer)

    # Callback should have been invoked for at least one operation.
    assert len(seen_ops) > 0, "Expected the callback to be invoked for operations"

    output = buffer.getvalue()
    assert len(output) > 0, "Expected annotated source output"
    # The dominant source file header should be written.
    assert "# ===" in output, "Expected a source file header in the output"
    # Annotations produced by the callback should appear in the output.
    assert "op=" in output, "Expected callback annotations in the output"


async def test_annotate_source_skips_none_annotations(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """Operations for which the callback returns None are not annotated."""
    buffer = StringIO()
    _annotate_source(
        hierarchical_coreai_program,
        lambda _operation: None,
        buffer,
    )

    output = buffer.getvalue()
    # With no annotations attributed, the function reports no valid locations.
    assert "No valid locations found" in output


async def test_annotate_source_exclude_filters_everything(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """A custom exclude that drops all locations yields no annotated file."""
    buffer = StringIO()
    _annotate_source(
        hierarchical_coreai_program,
        lambda operation: _TextAnnotation(operation.name),
        buffer,
        exclude=lambda _location: True,
    )

    output = buffer.getvalue()
    assert "No valid locations found" in output


async def test_source_annotator_get_annotation_unknown_location(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """get_annotation returns an empty list for unattributed locations."""
    operations = _walk_operations(hierarchical_coreai_program)
    annotator = _SourceAnnotator(
        operations,
        lambda operation: _TextAnnotation(operation.name),
    )

    assert annotator.get_annotation("/does/not/exist.py", 1) == []


async def test_every_operation_has_a_stack_trace(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """No operation of a converted model is left without a module."""
    paths = module_paths_by_op_id(hierarchical_coreai_program)
    ops = _build_coreai_op_map(hierarchical_coreai_program)
    missing = [
        op_id
        for op_id, op in ops.items()
        if op.name not in STRUCTURAL_OPS and op.name != "coreai.constant"
        if op_id not in paths
    ]
    assert missing == []


async def test_fused_location_gives_every_stack(
    hierarchical_coreai_program: AIProgram,
) -> None:
    """A matmul rebuilt from the previous layer's output is in both layers' stacks."""
    ops = _build_coreai_op_map(hierarchical_coreai_program)
    stacks = operation_stack_traces(ops[13])

    assert ("HierarchicalModel$1", "SubModel$1", "Linear$2") in stacks
    assert ("HierarchicalModel$1", "Linear$1") in stacks
    assert len(stacks) == len(set(stacks))
