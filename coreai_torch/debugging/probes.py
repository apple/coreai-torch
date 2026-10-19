# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Read intermediate values by making them outputs of the program."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coreai._compiler.dialects import coreai as coreai_dialect
from coreai._compiler.ir import (
    ArrayAttr,
    DictAttr,
    FunctionType,
    InsertionPoint,
    RankedTensorType,
    StringAttr,
    TypeAttr,
)
from coreai.authoring import AIProgram

from .debug_info import _build_coreai_op_map

_PROBE_NAME_PREFIX = "__probe"


@dataclass(frozen=True)
class ProbedProgram:
    """
    A copy of a program with extra outputs that carry intermediate values.

    Nothing here needs the Python runtime. The rewritten program can be saved and run by
    any runtime, and :meth:`manifest` says which of its outputs are probes, so a caller
    that gets the arrays back some other way can still tell them apart.

    Attributes:
        program: The rewritten copy. The original is not modified.
        function_name: The graph the outputs were added to.
        outputs: Output names, one per result, by the operation ID probed.
        unprobed: Requested operation IDs that could not be probed.

    """

    program: AIProgram
    function_name: str
    outputs: dict[int, list[str]]
    unprobed: list[int]

    @property
    def probe_names(self) -> set[str]:
        """Every output name that carries a probe rather than a model output."""
        return {name for names in self.outputs.values() for name in names}

    def split_outputs(
        self,
        arrays: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[int, list[Any]]]:
        """
        Separate a run's outputs into the model's own and the probed values.

        Args:
            arrays: Every output of the run, by output name.

        Returns:
            The model's own outputs by name, and the probed arrays by operation ID, one
            per result.

        """
        probes = self.probe_names
        model_outputs = {k: v for k, v in arrays.items() if k not in probes}
        probed = {
            op_id: [arrays[name] for name in names]
            for op_id, names in self.outputs.items()
        }
        return model_outputs, probed

    def manifest(self) -> dict[str, Any]:
        """
        Describe the probes as plain values.

        Returns:
            The function name, the output names by operation ID (keys are strings, as
            JSON requires), and the operation IDs that could not be probed.

        """
        return {
            "function": self.function_name,
            "probes": {str(i): names for i, names in self.outputs.items()},
            "unprobed": list(self.unprobed),
        }

    def save(self, directory: Path) -> Path:
        """
        Save the rewritten program and its manifest for another runtime.

        Args:
            directory: Where to write ``probed.aimodel`` and ``probes.json``.

        Returns:
            The path of the saved asset.

        """
        directory.mkdir(parents=True, exist_ok=True)
        asset = self.program.save_asset(directory / "probed.aimodel")
        (directory / "probes.json").write_text(json.dumps(self.manifest(), indent=2))
        return Path(asset.path)


def _probe_name(op_id: int, result_idx: int) -> str:
    return f"{_PROBE_NAME_PREFIX}_{op_id}_{result_idx}"


def add_probe_outputs(
    program: AIProgram,
    op_ids: Iterable[int],
    function_name: str = "main",
) -> ProbedProgram:
    """
    Copy *program* and return the values of the given operations as extra outputs.

    This reads an intermediate without the runtime's intermediate logger, which does
    not fire on the neural engine: the value leaves the graph like any other output.
    The extra outputs are appended after the existing ones, so the original outputs
    keep their names and positions.

    Only tensor results of operations in the entry block of *function_name* can be
    probed. An operation elsewhere (a nested graph, say), one with no results, or one
    with no operation ID is reported in :attr:`ProbedProgram.unprobed`.

    Probing changes what the compiler sees: a value that is an output cannot be
    fused away, so the compiled program can place work differently from the
    original. Compare placement before trusting a probed value as a stand-in.

    Args:
        program: The program to read values from. Not modified.
        op_ids: Core AI operation IDs to read.
        function_name: The graph to add outputs to.

    Returns:
        The rewritten copy, the output names by operation ID, and the IDs that could
        not be probed.

    Raises:
        ValueError: If *program* has no graph named *function_name* or the graph has
            no ``coreai.output``.

    """
    requested = list(dict.fromkeys(op_ids))
    probed = copy.deepcopy(program)
    graph = probed._get_graph(function_name)  # noqa: SLF001
    if graph is None:
        msg = f"program has no graph named {function_name!r}"
        raise ValueError(msg)

    block = graph.entry_block
    terminator = next(
        (op for op in block.operations if op.name == "coreai.output"), None
    )
    if terminator is None:
        msg = f"graph {function_name!r} has no coreai.output"
        raise ValueError(msg)

    all_ops = _build_coreai_op_map(probed)

    values = []
    names: list[str] = []
    outputs: dict[int, list[str]] = {}
    unprobed: list[int] = []
    for op_id in requested:
        op = all_ops.get(op_id)
        results = (
            [r for r in op.results if isinstance(r.type, RankedTensorType)]
            if op is not None and op.operation.parent == graph.operation
            else []
        )
        if not results:
            unprobed.append(op_id)
            continue
        outputs[op_id] = []
        for result_idx, result in enumerate(results):
            name = _probe_name(op_id, result_idx)
            values.append(result)
            names.append(name)
            outputs[op_id].append(name)

    if values:
        with probed._module._mlir_module.context, terminator.location:  # noqa: SLF001
            with InsertionPoint(terminator):
                coreai_dialect.OutputOp([*terminator.operands, *values])
            terminator.erase()

            signature = graph.function_type.value
            attributes = graph.operation.attributes
            attributes["function_type"] = TypeAttr.get(
                FunctionType.get(
                    list(signature.inputs),
                    [*signature.results, *(v.type for v in values)],
                )
            )
            attributes["res_attrs"] = ArrayAttr.get(
                [
                    *attributes["res_attrs"],
                    *(
                        DictAttr.get({"coreai.name": StringAttr.get(name)})
                        for name in names
                    ),
                ]
            )

    return ProbedProgram(
        program=probed,
        function_name=function_name,
        outputs=outputs,
        unprobed=unprobed,
    )
