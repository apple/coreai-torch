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
    _build_torch_fx_graph,
    compute_coreai_program_diff,
    compute_exported_program_diff,
    compute_per_graph_diff,
    format_multi_graph_diff,
    op_id_alignment,
    write_diff,
)
from coreai_torch.debugging.graph_match import UNSTABLE_ATTRIBUTES, WeightPolicy

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
    """
    Two models differing by one activation share nearly all of their graph.

    `mapped_node_count > 0 or not is_isomorphic` was the assertion here and cannot
    fail: an isomorphic diff maps every node, so the first disjunct holds, and a
    non-isomorphic one satisfies the second whatever it mapped. It passed equally for
    a diff that recognised the shared layers and one that matched nothing at all --
    which is the failure worth catching, because a matcher that gives up reports the
    whole model as rewritten rather than reporting an error.

    Measured: 43 of 45 nodes map, one is modified, and one either side is left over --
    the swapped activation and nothing else.
    """
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
    summary = diff.summary

    assert not diff.is_isomorphic, (
        "The activation differs, so this is not the same graph"
    )
    assert summary.mapped_node_count >= 0.9 * summary.source_node_count, (
        f"Only {summary.mapped_node_count} of {summary.source_node_count} nodes "
        "mapped; an edit to one activation should leave the rest recognised"
    )
    # The summary's own accounting: mapped, modified and unmapped partition the source.
    assert (
        summary.mapped_node_count
        + summary.modified_node_count
        + summary.unmapped_source_node_count
        == summary.source_node_count
    )
    assert summary.unmapped_source_node_count == summary.unmapped_target_node_count, (
        "One operation swapped for another leaves the same count over on both sides"
    )


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


@pytest.mark.asyncio
async def test_op_id_alignment_identical_programs() -> None:
    """An unchanged model maps every operation to itself, and says so."""
    model = ThreeLinearModel().eval()
    args = tuple(get_example_inputs(ThreeLinearModel).values())

    exported = torch.export.export(model, args).run_decompositions()
    before = await _create_coreai_program_from_model(exported)
    after = await _create_coreai_program_from_model(
        torch.export.export(model, args).run_decompositions()
    )

    alignment = op_id_alignment(before, after)

    assert alignment.identical, "Rebuilding the same model should be the same graph"
    assert alignment.mapping, "Should map operations"
    assert not alignment.removed
    assert not alignment.added
    assert not alignment.modified
    # Nothing moved, so every operation keeps its id -- which is exactly why raw ids
    # look safe to compare until an edit renumbers them.
    assert all(source == target for source, target in alignment.mapping.items())


@pytest.mark.asyncio
async def test_op_id_alignment_survives_an_added_layer() -> None:
    """An added layer renumbers ids; the mapping still pairs what survived."""
    args = tuple(get_example_inputs(ThreeLinearModel).values())
    before = await _create_coreai_program_from_model(
        torch.export.export(ThreeLinearModel().eval(), args).run_decompositions()
    )
    after = await _create_coreai_program_from_model(
        torch.export.export(ExtraLayerModel().eval(), args).run_decompositions()
    )

    alignment = op_id_alignment(before, after)

    assert not alignment.identical, "An added layer is a different graph"
    assert alignment.mapping, "Operations either side of the addition should pair"
    assert alignment.added, "The added layer's operations have no counterpart"
    # A mapped pair is one operation, so no id may be claimed twice on either side.
    assert len(set(alignment.mapping.values())) == len(alignment.mapping)
    assert not set(alignment.mapping) & set(alignment.removed)
    assert not set(alignment.mapping.values()) & set(alignment.added)


@pytest.mark.asyncio
async def test_op_id_alignment_of_a_rebuild_reports_no_tie_break() -> None:
    """
    An unedited rebuild must carry no caution, or the caution means nothing.

    Under the default `IGNORE` every parameter constant of one shape is
    interchangeable, so this is the comparison with the most to choose between and
    the least worth warning about: `identical` proves each choice harmless.
    """
    model = ThreeLinearModel().eval()
    args = tuple(get_example_inputs(ThreeLinearModel).values())

    exported = torch.export.export(model, args).run_decompositions()
    before = await _create_coreai_program_from_model(exported)
    after = await _create_coreai_program_from_model(
        torch.export.export(model, args).run_decompositions()
    )

    alignment = op_id_alignment(before, after)

    assert alignment.identical
    assert not alignment.ambiguity, (
        "Nothing was reported changed, so nothing can rest on a tie-break"
    )


@pytest.mark.asyncio
async def test_op_id_alignment_names_the_removals_a_tie_break_produced() -> None:
    """
    Dropping a layer leaves interchangeable ops over, and the diff has to say which.

    Otherwise `removed` reads as the list of operations an edit deleted when part of
    it is only the leftover of a group the matcher took in topological order -- the
    failure that blames an edit on the wrong layer.

    The sets narrow the like-named fields rather than adding to them, so subtracting
    one from the other cannot go negative.
    """
    args = tuple(get_example_inputs(ThreeLinearModel).values())
    before = await _create_coreai_program_from_model(
        torch.export.export(ThreeLinearModel().eval(), args).run_decompositions()
    )
    after = await _create_coreai_program_from_model(
        torch.export.export(TwoLinearSkipModel().eval(), args).run_decompositions()
    )

    alignment = op_id_alignment(before, after)
    ambiguity = alignment.ambiguity

    assert ambiguity, "Two of three like layers survived; which two is a tie-break"
    assert ambiguity.paired <= set(alignment.mapping)
    assert ambiguity.removed <= set(alignment.removed)
    assert ambiguity.added <= set(alignment.added)


@pytest.mark.asyncio
async def test_written_diff_says_when_a_tie_break_decided_it() -> None:
    """
    The caution has to reach the report, not just the object nobody prints.

    Suppressed when nothing was arbitrary, so its presence carries information: an
    unedited rebuild renders no such line.
    """
    args = tuple(get_example_inputs(ThreeLinearModel).values())
    exported = torch.export.export(ThreeLinearModel().eval(), args).run_decompositions()
    before = await _create_coreai_program_from_model(exported)
    rebuilt = await _create_coreai_program_from_model(
        torch.export.export(ThreeLinearModel().eval(), args).run_decompositions()
    )
    after = await _create_coreai_program_from_model(
        torch.export.export(TwoLinearSkipModel().eval(), args).run_decompositions()
    )

    def rendered(source: AIProgram, target: AIProgram) -> str:
        diff = compute_coreai_program_diff(source, target)
        output = io.StringIO()
        write_diff(diff, diff.source_graph, diff.target_graph, output=output)
        return output.getvalue()

    dropped_layer = rendered(before, after)
    assert "Decided by tie-break" in dropped_layer
    assert "WeightPolicy.DIGEST" in dropped_layer, "Name the remedy, not just the risk"

    assert "Decided by tie-break" not in rendered(before, rebuilt)


# ---------------------------------------------------------------------------
# Weight policies and the identity a diff was computed under
# ---------------------------------------------------------------------------


async def _two_programs_with_different_weights() -> tuple[AIProgram, AIProgram]:
    """
    One model class exported twice, each instance independently initialised.

    Structurally identical, numerically not -- a retrained model, which is the case
    `IGNORE` cannot see and `DIGEST` exists for.
    """
    args = tuple(get_example_inputs(ThreeLinearModel).values())
    return (
        await _create_coreai_program_from_model(
            torch.export.export(ThreeLinearModel().eval(), args).run_decompositions()
        ),
        await _create_coreai_program_from_model(
            torch.export.export(ThreeLinearModel().eval(), args).run_decompositions()
        ),
    )


@pytest.mark.asyncio
async def test_digest_sees_a_retrained_model_that_ignore_calls_identical() -> None:
    """
    What the default costs, stated as a test: `IGNORE` cannot see a weight change.

    Two independently initialised copies of one architecture come back `identical`
    under the default -- a proof of sameness for two models that compute different
    things. That is correct for the question `IGNORE` answers ("did the *structure*
    change") and wrong for the one a reader is likely asking, which is why the
    ambiguity warning points at `DIGEST`.
    """
    before, after = await _two_programs_with_different_weights()

    assert op_id_alignment(before, after).identical, (
        "IGNORE elides parameter values by design"
    )

    digested = op_id_alignment(before, after, weights=WeightPolicy.DIGEST)
    assert not digested.identical
    assert digested.modified, "The constants carrying the weights are what changed"
    assert not digested.removed and not digested.added, (
        "Nothing was added or deleted -- every operation still corresponds"
    )


@pytest.mark.asyncio
async def test_portable_digest_agrees_with_digest_in_one_process() -> None:
    """
    `DIGEST_PORTABLE` costs a print of each module and must buy correctness with it.

    Within one process the two policies have the same information, so they must reach
    the same verdict; the difference only shows between assets loaded from disk, where
    a resource handle is seeded per execution. Disagreeing here would mean the extra
    work changed the answer rather than preserving it.
    """
    before, after = await _two_programs_with_different_weights()

    digest = op_id_alignment(before, after, weights=WeightPolicy.DIGEST)
    portable = op_id_alignment(before, after, weights=WeightPolicy.DIGEST_PORTABLE)

    assert portable.identical == digest.identical
    assert portable.modified == digest.modified
    assert portable.mapping == digest.mapping


@pytest.mark.asyncio
async def test_ignore_attributes_reaches_align_from_the_program_entry_point() -> None:
    """
    The knob has to survive the trip from the public function down to the matcher.

    Excluding the attribute that carries a constant's payload should make `DIGEST`
    behave like `IGNORE` again: same structure, values no longer part of identity.
    Landing anywhere short of `align` -- which is where it used to stop -- leaves the
    diff computed under an identity the caller did not ask for.
    """
    before, after = await _two_programs_with_different_weights()
    no_values = UNSTABLE_ATTRIBUTES | {"value"}

    seen = compute_coreai_program_diff(before, after, weights=WeightPolicy.DIGEST)
    assert not seen.is_isomorphic

    elided = compute_coreai_program_diff(
        before, after, weights=WeightPolicy.DIGEST, ignore_attributes=no_values
    )
    assert elided.is_isomorphic, "The attribute holding the values was excluded"
    assert elided.ignore_attributes == frozenset(no_values), (
        "Recorded on the result, so a reader can tell which identity produced it"
    )


@pytest.mark.asyncio
async def test_a_modified_pair_is_explained_under_the_diffs_own_identity() -> None:
    """
    A reason computed under a wider identity than `align` used names the wrong field.

    `write_diff` recomputes labels to say *why* a pair is not identical. Recomputed
    with the default ignored set while the diff excluded `value`, it finds the payload
    difference `align` had agreed to overlook and reports `attributes: ...` -- naming
    an attribute the caller deliberately excluded, for a pair rejected over its result
    types or its wiring.
    """
    args = tuple(get_example_inputs(ThreeLinearModel).values())
    before = await _create_coreai_program_from_model(
        torch.export.export(ThreeLinearModel().eval(), args).run_decompositions()
    )
    after = await _create_coreai_program_from_model(
        torch.export.export(ExtraLayerModel().eval(), args).run_decompositions()
    )

    def reasons(**kwargs: object) -> str:
        diff = compute_coreai_program_diff(
            before, after, weights=WeightPolicy.DIGEST, **kwargs
        )
        assert diff.modified_node_pairs, "This pair of models has modified operations"
        output = io.StringIO()
        write_diff(diff, diff.source_graph, diff.target_graph, output=output)
        return output.getvalue()

    assert "attributes:" in reasons(), (
        "Under DIGEST the payloads are part of identity, so they are a real reason"
    )
    assert "attributes:" not in reasons(
        ignore_attributes=UNSTABLE_ATTRIBUTES | {"value"}
    ), "No reason may name an attribute the diff was told to ignore"


@pytest.mark.asyncio
async def test_per_graph_diff_applies_one_identity_to_every_graph() -> None:
    """
    Main and composites have to be compared the same way, or the report contradicts
    itself: one graph's operations judged on their payloads and another's not.
    """
    before, after = await _two_programs_with_different_weights()
    no_values = UNSTABLE_ATTRIBUTES | {"value"}

    results = compute_per_graph_diff(
        before, after, weights=WeightPolicy.DIGEST, ignore_attributes=no_values
    )

    assert results, "There is always at least a main graph"
    for _, diff in results:
        if diff is not None:
            assert diff.ignore_attributes == frozenset(no_values)
            assert diff.weights is WeightPolicy.DIGEST


class _ConcatModel(torch.nn.Module):
    """A model whose inputs reach an operation inside a *list* argument."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concatenate along the configured dimension."""
        return torch.cat([x, y], dim=self.dim).sum()


def _concat_program(dim: int) -> torch.export.ExportedProgram:
    """Export `_ConcatModel` at *dim*."""
    args = (torch.randn(2, 4), torch.randn(2, 4))
    return torch.export.export(_ConcatModel(dim), args).run_decompositions()


def test_exported_program_follows_nodes_inside_list_arguments() -> None:
    """An input passed in a list is still an operand.

    Only top-level `fx.Node` args were followed, so `aten.cat` -- which takes its inputs
    as a list -- got no incoming edges at all, and the tensors feeding it looked unused.
    Measured on this two-input model: 1 edge across 5 nodes, against 4 now.
    """
    graph = _build_torch_fx_graph(_concat_program(0))

    concat = [
        node
        for node, data in graph.nodes(data=True)
        if "cat" in data.get("op_name", "")
    ]
    assert len(concat) == 1, "Expected exactly one concatenation in the graph"

    operands = sorted(
        (data["index"], producer)
        for producer, _, data in graph.in_edges(concat[0], data=True)
    )
    assert [index for index, _ in operands] == [0, 1], (
        f"Both list members should be wired as operands, got {operands}"
    )


def test_exported_program_diff_detects_a_changed_constant_argument() -> None:
    """Two graphs differing only in a keyword argument are not the same graph.

    An FX node's label was its op and target alone -- `_attr_digest` reads an
    `ir_object`, which an FX node does not have -- so its whole configuration was
    outside its identity. `cat(dim=0)` and `cat(dim=1)` hashed identically and the diff
    reported them isomorphic: a silent false negative, the worst answer a diff can give.
    """
    diff = compute_exported_program_diff(_concat_program(0), _concat_program(1))

    assert not diff.is_isomorphic, "A changed `dim` must not read as the same graph"
    assert diff.summary.modified_node_count > 0, (
        "The concatenation is still the same operation, so it should be reported as "
        "modified rather than removed and added"
    )


def test_exported_program_diff_still_matches_identical_programs() -> None:
    """The stronger identity must not make two exports of one model differ."""
    diff = compute_exported_program_diff(_concat_program(0), _concat_program(0))

    assert diff.is_isomorphic
    assert diff.summary.modified_node_count == 0
