# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Grouping findings by module instance.

`build_module_tree` is a pure function of its input, so these need nothing converted
and nothing compiled -- which is why the tree is tested here rather than only through
the reports that use it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import SpecializationOptions

from coreai_torch._debug_locations import _DebugInfoRecorder
from coreai_torch.converter import TorchConverter
from coreai_torch.debugging.debug_info import _build_coreai_op_map
from coreai_torch.debugging.inspector import (
    CoreAIProbedInspector,
    ModuleOutputUnknownError,
    get_module_outputs,
)
from coreai_torch.debugging.modules import (
    UNATTRIBUTED,
    ModuleNode,
    build_module_tree,
    _read_node_tables,
    module_output_op_ids,
)
from coreai_torch.debugging.source_annotator import (
    module_paths_by_op_id,
    operation_stack_traces,
)
from coreai_torch.debugging.utils import STRUCTURAL_OPS

from .test_model import HierarchicalModel, get_example_inputs


def test_nests_by_path() -> None:
    """A path becomes a chain of nodes, one per frame."""
    roots = build_module_tree(
        [
            (("Model$1", "Block$1", "Linear$1"), "a"),
            (("Model$1", "Block$1"), "b"),
        ]
    )

    assert [node.name for node in roots] == ["Model$1"]
    model = roots[0]
    assert [child.name for child in model.children] == ["Block$1"]
    block = model.children[0]
    assert block.items == ["b"]
    assert [child.name for child in block.children] == ["Linear$1"]
    assert block.children[0].items == ["a"]


def test_items_are_own_not_inherited() -> None:
    """A finding is filed against its own module, and counted by its ancestors."""
    roots = build_module_tree(
        [
            (("Model$1", "Block$1"), "a"),
            (("Model$1", "Block$2"), "b"),
            (("Model$1", "Block$2"), "c"),
        ]
    )

    model = roots[0]
    assert model.items == []
    assert len(model.all_items()) == 3
    assert sorted(child.name for child in model.children) == ["Block$1", "Block$2"]


def test_ancestors_exist_without_items() -> None:
    """
    A module nothing is filed against still appears.

    The honest reading of a module whose work always fuses with a sibling's: the
    structure is there and nothing is attributed to it alone.
    """
    roots = build_module_tree([(("Model$1", "Block$1", "Linear$1"), "a")])

    model = roots[0]
    assert model.items == []
    assert model.children[0].items == []
    assert model.children[0].children[0].items == ["a"]


def test_ordered_by_weight_then_name() -> None:
    """The branch holding the most is first, and ties break on name so runs agree."""
    roots = build_module_tree(
        [
            (("Small$1",), "a"),
            (("Big$1",), "b"),
            (("Big$1",), "c"),
            (("Also$1",), "d"),
        ]
    )

    assert [node.name for node in roots] == ["Big$1", "Also$1", "Small$1"]


def test_empty_path_is_bucketed() -> None:
    """A finding with no module path lands in one named bucket, not at the root."""
    roots = build_module_tree([((), "a"), (("Model$1",), "b")])

    assert {node.name for node in roots} == {UNATTRIBUTED, "Model$1"}


def test_type_and_instance_split() -> None:
    """A frame names an instance; the type is what groups instances together."""
    node: ModuleNode[str] = ModuleNode(name="Linear$3")
    assert (node.type_name, node.instance) == ("Linear", 3)

    unnumbered: ModuleNode[str] = ModuleNode(name=UNATTRIBUTED)
    assert unnumbered.instance is None


def test_find_by_name_and_by_path() -> None:
    """A bare name finds any instance; a path finds the one under that parent."""
    roots = build_module_tree(
        [
            (("Model$1", "Block$1", "Linear$1"), "a"),
            (("Model$1", "Block$2", "Linear$1"), "b"),
        ]
    )
    model = roots[0]

    assert model.find("Block$2") is not None
    assert model.find("Block$2/Linear$1") is not None
    assert model.find("Block$2/Linear$1").items == ["b"]  # type: ignore[union-attr]
    assert model.find("Block$9") is None


def test_to_dict_is_json_serialisable() -> None:
    """A tree must survive `json.dumps`, which is how a caller not in Python reads it."""
    roots = build_module_tree(
        [
            (("Model$1", "Block$1"), 1),
            (("Model$1", "Block$1"), 2),
        ]
    )

    plain = json.loads(json.dumps(roots[0].to_dict(lambda item: item)))
    assert plain["name"] == "Model$1"
    assert plain["type_name"] == "Model"
    assert plain["instance"] == 1
    # The count is the subtree's, so a reader sees the branch total without descending.
    assert plain["count"] == 2
    assert plain["items"] == []
    assert plain["children"][0]["items"] == [1, 2]


def _convert(model: torch.nn.Module, *inputs: torch.Tensor) -> AIProgram:
    exported = torch.export.export(model.eval(), inputs).run_decompositions()
    converter = TorchConverter()
    converter.add_exported_program(exported, entrypoint_name="main")
    return converter.to_coreai()


@pytest.fixture
def hierarchical_program() -> AIProgram:
    """The Core AI program of the hierarchical model, with stack traces."""
    return _convert(
        HierarchicalModel(), *get_example_inputs(HierarchicalModel).values()
    )


def test_module_output_ops_are_where_values_leave(
    hierarchical_program: AIProgram,
) -> None:
    """A module's output is the operation realizing a node a node outside reads."""
    root = ("HierarchicalModel$1",)
    assert module_output_op_ids(hierarchical_program) == {
        root: [24],
        (*root, "Linear$1"): [11],
        (*root, "SubModel$1"): [15],
        (*root, "SubModel$1", "Linear$2"): [14],
        (*root, "SubModel$2"): [20],
        (*root, "SubModel$2", "Linear$3"): [19],
        (*root, "Linear$4"): [23],
    }


def test_output_is_the_operation_realizing_the_node(
    hierarchical_program: AIProgram,
) -> None:
    """An operation that absorbed a producer does not hold that producer's value."""
    outputs = module_output_op_ids(hierarchical_program)

    # Op 13 is Linear$2's matmul and also carries Linear$1's location, and the last
    # sigmoid carries Linear$4's. Neither holds the value of the layer it absorbed.
    assert outputs[("HierarchicalModel$1", "Linear$1")] == [11]
    assert outputs[("HierarchicalModel$1", "Linear$4")] == [23]


def test_without_the_node_table_fewer_modules_are_certain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No id is known to be a reshape without the table, so only some modules remain."""
    monkeypatch.setattr(
        _DebugInfoRecorder, "_node_table_metadata", lambda *_args, **_kw: []
    )
    program = _convert(
        HierarchicalModel(), *get_example_inputs(HierarchicalModel).values()
    )

    root = ("HierarchicalModel$1",)
    assert module_output_op_ids(program) == {
        root: [24],
        (*root, "Linear$4"): [23],
        (*root, "SubModel$1"): [15],
        (*root, "SubModel$2"): [20],
    }


def test_node_table_is_read_back_from_the_program(
    hierarchical_program: AIProgram,
) -> None:
    """The graph's location carries what each torch node was, and where the graph ends."""
    (table,) = _read_node_tables(hierarchical_program).values()

    assert table.targets[11] == "aten.addmm.default"
    assert table.shape_only(9)
    assert not table.shape_only(11)
    assert table.output_id == 29


def test_every_operation_has_a_stack_trace(hierarchical_program: AIProgram) -> None:
    """No operation of a converted model is left without a module."""
    paths = module_paths_by_op_id(hierarchical_program)
    ops = _build_coreai_op_map(hierarchical_program)
    missing = [
        op_id
        for op_id, op in ops.items()
        if op.name not in STRUCTURAL_OPS and op.name != "coreai.constant"
        if op_id not in paths
    ]
    assert missing == []


def test_fused_location_gives_every_stack(hierarchical_program: AIProgram) -> None:
    """A matmul rebuilt from the previous layer's output is in both layers' stacks."""
    ops = _build_coreai_op_map(hierarchical_program)
    stacks = operation_stack_traces(ops[13])

    assert ("HierarchicalModel$1", "SubModel$1", "Linear$2") in stacks
    assert ("HierarchicalModel$1", "Linear$1") in stacks
    assert len(stacks) == len(set(stacks))


class _ConvBatchNorm(torch.nn.Module):
    """A conv whose batch norm the compiler folds into its weights."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.bn = torch.nn.BatchNorm2d(4)
        self.head = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.bn(self.conv(x))))


def test_module_whose_reader_was_folded_in_is_left_out() -> None:
    """A conv's op holds the batch norm's value once folded, not the conv's."""
    outputs = module_output_op_ids(
        _convert(_ConvBatchNorm(), torch.randn(1, 3, 8, 8))
    )

    root = ("_ConvBatchNorm$1",)
    assert (*root, "Conv2d$1") not in outputs
    assert (*root, "BatchNorm2d$1") not in outputs
    assert (*root, "Conv2d$2") in outputs


async def test_get_module_outputs_matches_torch(
    specialization_options: SpecializationOptions | None,
) -> None:
    """Probing a module's output gives the value torch's own module returns."""
    torch.manual_seed(0)
    model = HierarchicalModel().eval()
    x = get_example_inputs(HierarchicalModel)["x"]
    program = _convert(model, x)

    seen: list[torch.Tensor] = []
    handle = model.sub_model1.register_forward_hook(
        lambda _m, _i, out: seen.append(out.detach())
    )
    model(x)
    handle.remove()

    inspector = CoreAIProbedInspector(
        program, specialization_options=specialization_options
    )
    (value,) = await get_module_outputs(
        inspector,
        program,
        ("HierarchicalModel$1", "SubModel$1"),
        {"x": x.numpy()},
    )

    np.testing.assert_allclose(
        value.reshape(-1), seen[0].numpy().reshape(-1), rtol=1e-3, atol=1e-3
    )


async def test_get_module_outputs_refuses_what_it_cannot_place(
    hierarchical_program: AIProgram,
    specialization_options: SpecializationOptions | None,
) -> None:
    """A module that is absent or folded away is an error, never a guess."""
    inspector = CoreAIProbedInspector(
        hierarchical_program, specialization_options=specialization_options
    )
    x = get_example_inputs(HierarchicalModel)["x"].numpy()

    with pytest.raises(ModuleOutputUnknownError, match="no operation belongs"):
        await get_module_outputs(
            inspector,
            hierarchical_program,
            ("HierarchicalModel$1", "Nope$1"),
            {"x": x},
        )

    program = _convert(_ConvBatchNorm(), torch.randn(1, 3, 8, 8))
    with pytest.raises(ModuleOutputUnknownError, match="not certain"):
        await get_module_outputs(
            CoreAIProbedInspector(
                program, specialization_options=specialization_options
            ),
            program,
            ("_ConvBatchNorm$1", "Conv2d$1"),
            {"x": np.zeros((1, 3, 8, 8), dtype=np.float32)},
        )
