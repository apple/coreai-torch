# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the inspector that reads intermediates as extra model outputs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import SpecializationOptions

from coreai_torch.converter import TorchConverter
from coreai_torch.debugging.debug_info import _build_coreai_op_map
from coreai_torch.debugging.inspector import CoreAIProbedInspector
from coreai_torch.debugging.probes import add_probe_outputs


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))


@pytest.fixture
def program_and_input() -> tuple[AIProgram, torch.Tensor]:
    torch.manual_seed(0)
    model = _Net().eval().half()
    x = torch.randn(1, 4, 8, 8).half()
    exported = torch.export.export(model, (x,)).run_decompositions()
    converter = TorchConverter()
    converter.add_exported_program(exported, entrypoint_name="main")
    return converter.to_coreai(), x


def _ids_named(program: AIProgram, name: str) -> list[int]:
    return [i for i, op in _build_coreai_op_map(program).items() if op.name == name]


def test_probing_leaves_the_original_program_and_outputs_alone(
    program_and_input: tuple[AIProgram, torch.Tensor],
) -> None:
    program, _ = program_and_input
    before = str(program)
    (relu_id,) = _ids_named(program, "coreai.relu")

    probed = add_probe_outputs(program, [relu_id])

    assert str(program) == before
    assert probed.outputs == {relu_id: [f"__probe_{relu_id}_0"]}
    graph = probed.program._get_graph("main")
    assert [r["coreai.name"].value for r in graph.res_attrs] == [
        "relu",
        f"__probe_{relu_id}_0",
    ]


def test_ids_that_cannot_be_probed_are_reported(
    program_and_input: tuple[AIProgram, torch.Tensor],
) -> None:
    program, _ = program_and_input
    (output_id,) = _ids_named(program, "coreai.output")

    probed = add_probe_outputs(program, [output_id, 10_000])

    assert probed.outputs == {}
    assert probed.unprobed == [output_id, 10_000]


async def test_probed_inspector_returns_intermediates(
    program_and_input: tuple[AIProgram, torch.Tensor],
    specialization_options: SpecializationOptions | None,
) -> None:
    program, x = program_and_input
    (conv_id,) = _ids_named(program, "coreai.conv2d")
    (relu_id,) = _ids_named(program, "coreai.relu")
    inspector = CoreAIProbedInspector(
        program, specialization_options=specialization_options
    )

    result = await inspector.get_intermediates_for_ops(
        [conv_id, relu_id, 10_000], {"x": x.numpy()}
    )

    assert result[10_000] is None
    (relu,) = result[relu_id]
    (conv,) = result[conv_id]
    assert conv.shape == relu.shape == (1, 4, 8, 8)
    np.testing.assert_array_equal(relu, inspector.last_outputs["relu"])
    assert np.all(relu >= 0)
    # The value is the convolution before the bias add, so relu(conv) is not it.
    assert not np.array_equal(conv, relu)


class _Accumulator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("acc", torch.zeros(1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.acc + x
        self.acc.copy_(y * 2)
        return torch.relu(y)


async def test_probed_inspector_reads_stateful_models_without_advancing_state(
    specialization_options: SpecializationOptions | None,
) -> None:
    x = torch.randn(1, 4)
    exported = torch.export.export(_Accumulator().eval(), (x,)).run_decompositions()
    converter = TorchConverter()
    converter.add_exported_program(exported, entrypoint_name="main")
    program = converter.to_coreai()
    (read_id,) = _ids_named(program, "coreai.read_handle")
    (written_id,) = _ids_named(program, "coreai.decomposable.broadcasting_mul")
    inspector = CoreAIProbedInspector(
        program, specialization_options=specialization_options
    )
    start = np.ones((1, 4), dtype=np.float32)
    caller_state = {"b_acc": start}

    result = await inspector.get_intermediates_for_ops(
        [read_id, written_id], {"x": x.numpy()}, state=caller_state
    )

    ((read,),) = (result[read_id],)
    ((written,),) = (result[written_id],)
    np.testing.assert_array_equal(read, start)
    np.testing.assert_allclose(written, (start + x.numpy()) * 2, rtol=1e-5)
    # What was written is what the model left in its state, and the caller's own
    # array is untouched so a second call starts from the same place.
    np.testing.assert_array_equal(inspector.last_state["b_acc"], written)
    np.testing.assert_array_equal(caller_state["b_acc"], np.ones((1, 4)))
    assert set(inspector.last_outputs) == {"relu"}


def test_probed_program_describes_itself_without_a_runtime(
    program_and_input: tuple[AIProgram, torch.Tensor],
    tmp_path: Path,
) -> None:
    program, _ = program_and_input
    (relu_id,) = _ids_named(program, "coreai.relu")
    probed = add_probe_outputs(program, [relu_id, 10_000])

    # A run's outputs split by name alone, so a caller using another runtime can do it.
    own, by_op = probed.split_outputs({"relu": 1, f"__probe_{relu_id}_0": 2})
    assert own == {"relu": 1}
    assert by_op == {relu_id: [2]}

    manifest = probed.manifest()
    assert manifest == {
        "function": "main",
        "probes": {str(relu_id): [f"__probe_{relu_id}_0"]},
        "unprobed": [10_000],
    }

    asset = probed.save(tmp_path / "out")
    assert asset.exists()
    assert json.loads((tmp_path / "out" / "probes.json").read_text()) == manifest


class _CountLoads:
    """Count compiles by wrapping AIModel.load."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import coreai.runtime

        self.count = 0
        real = coreai.runtime.AIModel.load

        async def counting(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            self.count += 1
            return await real(*args, **kwargs)

        monkeypatch.setattr(coreai.runtime.AIModel, "load", counting)


async def test_compiled_copy_is_reused_for_the_same_operations(
    program_and_input: tuple[AIProgram, torch.Tensor],
    specialization_options: SpecializationOptions | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program, x = program_and_input
    (conv_id,) = _ids_named(program, "coreai.conv2d")
    (relu_id,) = _ids_named(program, "coreai.relu")
    loads = _CountLoads(monkeypatch)
    inspector = CoreAIProbedInspector(
        program, specialization_options=specialization_options
    )

    first = await inspector.get_intermediates_for_ops([relu_id], {"x": x.numpy()})
    # Same set in another order, other inputs: no compile, and the answer follows the input.
    again = await inspector.get_intermediates_for_ops([relu_id], {"x": x.numpy() * 2})
    assert loads.count == 1
    assert not np.array_equal(first[relu_id][0], again[relu_id][0])

    await inspector.get_intermediates_for_ops([conv_id, relu_id], {"x": x.numpy()})
    assert loads.count == 2
    assert inspector.cached_models == 2


async def test_cache_is_bounded_and_can_be_switched_off(
    program_and_input: tuple[AIProgram, torch.Tensor],
    specialization_options: SpecializationOptions | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program, x = program_and_input
    (conv_id,) = _ids_named(program, "coreai.conv2d")
    (relu_id,) = _ids_named(program, "coreai.relu")
    loads = _CountLoads(monkeypatch)
    inputs = {"x": x.numpy()}

    bounded = CoreAIProbedInspector(
        program, specialization_options=specialization_options, max_cached_models=1
    )
    await bounded.get_intermediates_for_ops([conv_id], inputs)
    await bounded.get_intermediates_for_ops([relu_id], inputs)  # evicts conv
    assert bounded.cached_models == 1
    await bounded.get_intermediates_for_ops([conv_id], inputs)  # compiles again
    assert loads.count == 3
    bounded.clear_cache()
    assert bounded.cached_models == 0

    loads.count = 0
    uncached = CoreAIProbedInspector(
        program, specialization_options=specialization_options, max_cached_models=0
    )
    await uncached.get_intermediates_for_ops([relu_id], inputs)
    await uncached.get_intermediates_for_ops([relu_id], inputs)
    assert loads.count == 2
    assert uncached.cached_models == 0

    with pytest.raises(ValueError, match="must not be negative"):
        CoreAIProbedInspector(program, max_cached_models=-1)
