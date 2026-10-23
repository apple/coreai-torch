# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Which Core AI operations hold a module's output.

Checked on a model whose layers fuse (each `nn.Linear` is a matmul that absorbs the
previous layer's reshape), and on one whose batch norm the compiler folds into a conv.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import SpecializationOptions

from coreai_torch._debug_locations import _DebugInfoRecorder
from coreai_torch.converter import TorchConverter
from coreai_torch.debugging.inspector import (
    CoreAIProbedInspector,
    ModuleOutputUnknownError,
    get_module_outputs,
)
from coreai_torch.debugging.module_outputs import (
    _read_source_nodes,
    module_output_op_ids,
)

from .test_model import HierarchicalModel, get_example_inputs


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
    (table,) = _read_source_nodes(hierarchical_program).values()

    assert table.targets[11] == "aten.addmm.default"
    assert table.shape_only(9)
    assert not table.shape_only(11)
    assert table.output_id == 29


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
    outputs = module_output_op_ids(_convert(_ConvBatchNorm(), torch.randn(1, 3, 8, 8)))

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


class _StatefulHeads(torch.nn.Module):
    """Two heads over a shared body, and a buffer that is updated in place."""

    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Linear(6, 6)
        self.a = torch.nn.Linear(6, 3)
        self.b = torch.nn.Linear(6, 2)
        self.register_buffer("acc", torch.zeros(1, 6))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.body(x))
        self.acc.copy_(self.acc + h.mean(0, keepdim=True))
        return self.a(h), self.b(h)


async def test_a_value_written_to_state_is_not_a_module_output(
    specialization_options: SpecializationOptions | None,
) -> None:
    """The root returns its two heads; the buffer it updates is not one of them."""
    torch.manual_seed(0)
    model = _StatefulHeads().eval()
    x = torch.randn(2, 6)
    program = _convert(model, x)
    with torch.no_grad():
        expected = [out.numpy() for out in model(x)]

    inspector = CoreAIProbedInspector(
        program, specialization_options=specialization_options
    )
    arrays = await get_module_outputs(
        inspector,
        program,
        ("_StatefulHeads$1",),
        {"x": x.numpy()},
        state={"b_acc": np.zeros((1, 6), dtype=np.float32)},
    )

    assert len(arrays) == len(expected)
    for array, want in zip(arrays, expected, strict=True):
        np.testing.assert_allclose(array, want, rtol=1e-3, atol=1e-3)
