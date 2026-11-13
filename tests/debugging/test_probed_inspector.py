# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the inspector that reads intermediates as extra model outputs."""

import numpy as np
import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import SpecializationOptions

from coreai_torch.converter import TorchConverter
from coreai_torch.debugging import inspector as inspector_module
from coreai_torch.debugging.debug_info import _build_coreai_op_map
from coreai_torch.debugging.inspector import (
    CoreAIProbedInspector,
    PerturbationError,
    PerturbationPolicy,
)
from coreai_torch.debugging.probes import (
    Perturbation,
    add_probe_outputs,
    compare_to_reference,
)


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))


class _Accumulator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("acc", torch.zeros(1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.acc + x
        self.acc.copy_(y * 2)
        return torch.relu(y)


def _convert(model: torch.nn.Module, x: torch.Tensor) -> AIProgram:
    exported = torch.export.export(model.eval(), (x,)).run_decompositions()
    converter = TorchConverter()
    converter.add_exported_program(exported, entrypoint_name="main")
    return converter.to_coreai()


@pytest.fixture
def program_and_input() -> tuple[AIProgram, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8, 8).half()
    return _convert(_Net().half(), x), x


def _ids(program: AIProgram, name: str) -> list[int]:
    return [i for i, op in _build_coreai_op_map(program).items() if op.name == name]


def test_probe_outputs_are_appended_without_touching_the_program(
    program_and_input: tuple[AIProgram, torch.Tensor],
) -> None:
    program, _ = program_and_input
    before = str(program)
    (relu_id,) = _ids(program, "coreai.relu")
    (output_id,) = _ids(program, "coreai.output")

    probed = add_probe_outputs(program, [relu_id, output_id])

    assert str(program) == before
    assert probed.outputs == {relu_id: [f"__probe_{relu_id}_0"]}
    assert probed.unprobed == [output_id]
    names = [
        r["coreai.name"].value for r in probed.program._get_graph("main").res_attrs
    ]
    assert names == ["relu", f"__probe_{relu_id}_0"]
    own, by_op = probed.split_outputs({"relu": 1, f"__probe_{relu_id}_0": 2})
    assert (own, by_op) == ({"relu": 1}, {relu_id: [2]})


async def test_intermediates_come_back_by_op_id(
    program_and_input: tuple[AIProgram, torch.Tensor],
    specialization_options: SpecializationOptions | None,
) -> None:
    program, x = program_and_input
    (conv_id,) = _ids(program, "coreai.conv2d")
    (relu_id,) = _ids(program, "coreai.relu")
    inspector = CoreAIProbedInspector(
        program, specialization_options=specialization_options
    )

    result = await inspector.get_intermediates_for_ops(
        [conv_id, relu_id, 10_000], {"x": x.numpy()}
    )

    assert result[10_000] is None
    ((conv,), (relu,)) = result[conv_id], result[relu_id]
    assert conv.shape == relu.shape == (1, 4, 8, 8)
    assert np.all(relu >= 0)
    # last_outputs is the unprobed run's; the probed copy is within an fp16 step.
    np.testing.assert_allclose(
        relu, inspector.last_outputs["relu"], rtol=1e-2, atol=2e-3
    )
    assert all(p.within_tolerance for p in inspector.perturbations)


async def test_state_is_read_and_the_callers_copy_is_left_alone(
    specialization_options: SpecializationOptions | None,
) -> None:
    x = torch.randn(1, 4)
    program = _convert(_Accumulator(), x)
    (read_id,) = _ids(program, "coreai.read_handle")
    (written_id,) = _ids(program, "coreai.decomposable.broadcasting_mul")
    inspector = CoreAIProbedInspector(
        program, specialization_options=specialization_options
    )
    start = np.ones((1, 4), dtype=np.float32)

    result = await inspector.get_intermediates_for_ops(
        [read_id, written_id], {"x": x.numpy()}, state={"b_acc": start}
    )

    np.testing.assert_array_equal(result[read_id][0], start)
    np.testing.assert_allclose(
        result[written_id][0], (start + x.numpy()) * 2, rtol=1e-5
    )
    np.testing.assert_array_equal(inspector.last_state["b_acc"], result[written_id][0])
    np.testing.assert_array_equal(start, np.ones((1, 4)))


async def test_compiled_copies_are_reused_and_bounded(
    program_and_input: tuple[AIProgram, torch.Tensor],
    specialization_options: SpecializationOptions | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program, x = program_and_input
    (conv_id,) = _ids(program, "coreai.conv2d")
    (relu_id,) = _ids(program, "coreai.relu")
    loads = []
    real = CoreAIProbedInspector._compile

    async def counting(self, op_ids):  # noqa: ANN001, ANN202
        loads.append(tuple(op_ids))
        return await real(self, op_ids)

    monkeypatch.setattr(CoreAIProbedInspector, "_compile", counting)
    inspector = CoreAIProbedInspector(
        program, specialization_options=specialization_options, max_cached_models=1
    )
    inputs = {"x": x.numpy()}

    await inspector.get_intermediates_for_ops([relu_id], inputs)
    await inspector.get_intermediates_for_ops([relu_id], {"x": x.numpy() * 2})
    assert loads == [(), (relu_id,)]  # the reference once, the probed copy once

    await inspector.get_intermediates_for_ops([conv_id], inputs)  # evicts relu
    await inspector.get_intermediates_for_ops([relu_id], inputs)
    assert loads[2:] == [(conv_id,), (relu_id,)]
    assert inspector.cached_models == 1


def test_comparison_follows_the_allclose_rule() -> None:
    reference = {"y": np.array([1.0, 100.0], dtype=np.float32)}

    near = compare_to_reference(reference, {"y": [1.0005, 100.05]}, [7], 1e-3, 1e-3)
    far = compare_to_reference(reference, {"y": [1.0, 100.5]}, [7], 1e-3, 1e-3)
    nan = compare_to_reference(reference, {"y": [1.0, np.nan]}, [7], 1.0, 1.0)

    assert near.within_tolerance
    assert not far.within_tolerance
    assert not nan.within_tolerance


@pytest.mark.parametrize("failure", ["moves_outputs", "fails_to_compile"])
async def test_a_bad_probe_set_is_reduced(
    program_and_input: tuple[AIProgram, torch.Tensor],
    specialization_options: SpecializationOptions | None,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    program, x = program_and_input
    (conv_id,) = _ids(program, "coreai.conv2d")
    (relu_id,) = _ids(program, "coreai.relu")
    (add_id,) = _ids(program, "coreai.decomposable.broadcasting_add")

    if failure == "moves_outputs":

        def compare(reference, probed, op_ids, rtol, atol):  # noqa: ANN001, ANN202, ARG001
            ids = tuple(op_ids)
            return Perturbation(ids, 0.0, 0.0, relu_id not in ids)

        monkeypatch.setattr(inspector_module, "compare_to_reference", compare)
    else:
        real = CoreAIProbedInspector._compile

        async def compile_(self, op_ids):  # noqa: ANN001, ANN202
            if relu_id in op_ids:
                msg = "cannot load"
                raise RuntimeError(msg)
            return await real(self, op_ids)

        monkeypatch.setattr(CoreAIProbedInspector, "_compile", compile_)

    inspector = CoreAIProbedInspector(
        program, specialization_options=specialization_options
    )
    ids = [conv_id, add_id, relu_id]
    result = await inspector.get_intermediates_for_ops(ids, {"x": x.numpy()})

    assert result[relu_id] is None
    assert result[conv_id] is not None
    assert result[add_id] is not None
    culprits = (
        inspector.perturbing_ops if failure == "moves_outputs" else inspector.failed_ops
    )
    assert culprits == {relu_id}

    strict = CoreAIProbedInspector(
        program,
        specialization_options=specialization_options,
        on_perturbation=PerturbationPolicy.RAISE,
    )
    with pytest.raises(PerturbationError):
        await strict.get_intermediates_for_ops(ids, {"x": x.numpy()})
