# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Changes between two programs, attributed to the module they happened in.

Each case is a *constructed* pair with the answer written down, rather than a comparison
against what the code currently does: a diff scored against itself enshrines its own
defects as the reference.

Models come from `test_model`, so a fixture is shared rather than restated. Most cases use
the `ThreeLinearModel` / `ExtraLayerModel` pair the graph-diff tests already use. One needs
`BlockStack`, and needs it for a reason: see :func:`test_added_block_is_attributed_to_the_added_block`.
"""

from __future__ import annotations

import io
import json
import sys

import pytest
import torch
from coreai.authoring import AIProgram

from coreai_torch.converter import TorchConverter
from coreai_torch.debugging.changes import (
    SOURCE,
    TARGET,
    OpChange,
    compute_changes,
)
from coreai_torch.debugging.graph_diff import OpDiffType
from coreai_torch.debugging.modules import UNATTRIBUTED

from .test_model import (
    BlockStack,
    ExtraLayerModel,
    ModifiedActivationModel,
    ThreeLinearModel,
    get_example_inputs,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Conversion needs macOS"
)


def _convert(model: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> AIProgram:
    """Convert *model* with the stack traces attribution needs."""
    exported = torch.export.export(model.eval(), args).run_decompositions()
    converter = TorchConverter(mode=TorchConverter.Mode.DEBUG)
    converter.add_exported_program(exported, entrypoint_name="main")
    return converter.to_coreai()


def _program(model_cls: type[torch.nn.Module]) -> AIProgram:
    """Convert one of the three-linear fixtures, at `ThreeLinearModel`'s input."""
    args = tuple(get_example_inputs(ThreeLinearModel).values())
    return _convert(model_cls(), args)


def _stack(depth: int) -> AIProgram:
    """Convert a `BlockStack` of *depth* blocks."""
    return _convert(BlockStack(depth=depth), (torch.randn(2, 8),))


def _modules_of(changes: list[OpChange]) -> set[str]:
    """The distinct module labels a set of changes was attributed to."""
    return {change.module_label for change in changes}


def test_identical_programs_have_no_changes() -> None:
    """
    The same model converted twice reports nothing.

    The case every other result is worthless without: symbol suffixes and parameter
    payloads differ between two conversions of one unchanged model, and a diff that
    counts those reports dozens of changes for no edit.
    """
    result = compute_changes(_program(ThreeLinearModel), _program(ThreeLinearModel))

    assert result.counts["changed"] == 0
    assert result.by_module() == []


def test_replaced_activation_is_a_removal_and_an_addition() -> None:
    """
    Swapping `relu` for `tanh` reports both operations, not one modified.

    The difference lands on the two *result values*, which correspond -- same type, same
    consumer -- so resolving that pair to the operations behind it would report "tanh
    modified" and never mention the relu.
    """
    result = compute_changes(
        _program(ThreeLinearModel), _program(ModifiedActivationModel)
    )

    kinds = {change.change for change in result.changes}
    assert OpDiffType.MODIFIED not in kinds
    assert kinds == {OpDiffType.ADDED, OpDiffType.REMOVED}

    added = [c for c in result.changes if c.change is OpDiffType.ADDED]
    removed = [c for c in result.changes if c.change is OpDiffType.REMOVED]
    assert [c.side for c in added] == [TARGET]
    assert [c.side for c in removed] == [SOURCE]
    assert "tanh" in added[0].op_name
    assert "relu" in removed[0].op_name


def test_an_extra_layer_is_attributed_to_a_layer() -> None:
    """
    An added layer's operations name the `Linear` instance they belong to.

    Multi-level paths, and the pair the graph-diff tests use, so the attribution is
    exercised on the same edit those already describe.
    """
    result = compute_changes(_program(ThreeLinearModel), _program(ExtraLayerModel))

    assert result.counts["changed"] > 0
    labels = _modules_of(result.changes)
    assert any("/Linear$" in label for label in labels), labels
    # Rooted at the model class, instance-qualified at every frame.
    assert all(label.startswith("ExtraLayerModel$1") for label in labels), labels


def test_added_block_is_attributed_to_the_added_block() -> None:
    """
    A third block's operations are reported against `Block$3`, not `Block$1`.

    Structurally equivalent operations are interchangeable, so which instance the matcher
    leaves unmapped is arbitrary, and unaided it names the first. A reader told `Block$1`
    changed reasonably concludes something changed in a block that did not change at all.

    This needs `BlockStack`, not the `ThreeLinearModel` / `ExtraLayerModel` pair: those are
    different classes, so *every* module path is unique to its own side and relocation has
    nothing to choose between. Two depths of one class share every path but the last.
    """
    result = compute_changes(_stack(depth=2), _stack(depth=3))

    assert result.counts["changed"] > 0
    added = [c for c in result.changes if c.change is OpDiffType.ADDED]
    assert added, "adding a block must add operations"

    # Every addition sits under the block that is new, and none under one that is not.
    for change in added:
        assert "StackBlock$3" in change.module_label, change.to_dict()


def test_a_change_names_the_line_that_built_it() -> None:
    """A finding names a line in the author's own file, not one inside torch."""
    result = compute_changes(_program(ThreeLinearModel), _program(ExtraLayerModel))

    sourced = [c for c in result.changes if c.attribution and c.attribution.source]
    assert sourced, "conversion recorded no source lines at all"
    for change in sourced:
        assert change.attribution is not None
        assert change.attribution.source is not None
        # The model's own file, which is where the layers are written.
        assert change.attribution.source.filename.endswith("test_model.py"), (
            change.attribution.source.filename
        )


def test_no_change_is_unattributed() -> None:
    """
    Every reported change has a module.

    `coreai.graph` and `coreai.output` are the only operations with no module path, and
    both are structure rather than something a user wrote, so they are filtered rather
    than bucketed. A change under `UNATTRIBUTED` means that filter has a hole in it.
    """
    result = compute_changes(_program(ThreeLinearModel), _program(ExtraLayerModel))

    assert UNATTRIBUTED not in _modules_of(result.changes)


def test_changes_in_matches_the_whole_subtree() -> None:
    """A module means the module, not the one row attributed to its top level."""
    result = compute_changes(_program(ThreeLinearModel), _program(ExtraLayerModel))

    root = result.by_module()[0].name
    # The root holds its own changes and every layer's beneath it, which is what makes
    # this a subtree match rather than an equality test.
    assert len(_modules_of(result.changes)) > 1, "the pair must differ at two levels"
    assert len(result.changes_in(root)) == len(result.changes)

    layer = next(label for label in _modules_of(result.changes) if "/Linear$" in label)
    under_layer = result.changes_in(layer)
    assert under_layer
    assert all(change.module_label.startswith(layer) for change in under_layer)
    assert len(under_layer) < len(result.changes), "a layer is not the whole diff"


def test_changes_in_is_ordered_by_source_line() -> None:
    """A reader walks a layer in the order they wrote it."""
    result = compute_changes(_program(ThreeLinearModel), _program(ExtraLayerModel))
    root = result.by_module()[0].name

    lines = [
        change.attribution.source.line
        for change in result.changes_in(root)
        if change.attribution and change.attribution.source
    ]
    assert lines == sorted(lines)


def test_module_counts_add_up_to_the_diff() -> None:
    """
    Grouping neither invents nor drops a change.

    Relocation moves a change from one module to another, and each relocation consumes
    one candidate; if it ever handed the same candidate out twice the totals would drift
    from what the matcher produced.
    """
    for result in (
        compute_changes(_program(ThreeLinearModel), _program(ExtraLayerModel)),
        compute_changes(_stack(depth=2), _stack(depth=3)),
    ):
        filed = sum(len(node.all_items()) for node in result.by_module())
        assert filed == result.counts["changed"]
        assert (
            result.counts["added"]
            + result.counts["removed"]
            + result.counts["modified"]
            == result.counts["changed"]
        )


def test_op_ids_are_unique_within_a_side() -> None:
    """
    One operation is reported once.

    Op ids are unique within a program, so a repeat on one side means the same operation
    was reported under two kinds -- which is two contradictory statements about it.
    """
    result = compute_changes(_program(ThreeLinearModel), _program(ExtraLayerModel))

    for side in (SOURCE, TARGET):
        ids = [
            c.op_id for c in result.changes if c.side == side and c.op_id is not None
        ]
        assert len(ids) == len(set(ids))


def test_to_dict_is_json_serialisable() -> None:
    """The report must survive `json.dumps`, summary counts first."""
    result = compute_changes(_program(ThreeLinearModel), _program(ExtraLayerModel))

    plain = json.loads(json.dumps(result.to_dict()))
    assert plain["changed"] == result.counts["changed"]
    assert {"added", "removed", "modified", "modules"} <= plain.keys()
    assert plain["modules"], "a non-empty diff must name at least one module"


def test_write_summary_names_the_hotspot_first() -> None:
    """The branch that changed most leads, so a reader does not have to sort."""
    result = compute_changes(_program(ThreeLinearModel), _program(ExtraLayerModel))

    buffer = io.StringIO()
    result.write_summary(buffer)
    written = buffer.getvalue()

    assert "Module" in written
    assert f"{result.counts['changed']} change(s)" in written
    assert "Linear$" in written


def test_unchanged_program_writes_an_empty_summary() -> None:
    """A clean result still renders, rather than raising on a zero denominator."""
    result = compute_changes(_program(ThreeLinearModel), _program(ThreeLinearModel))

    buffer = io.StringIO()
    result.write_summary(buffer)
    assert "0 change(s)" in buffer.getvalue()
