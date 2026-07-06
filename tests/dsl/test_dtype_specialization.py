# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for ``template_dtypes`` / type specialization in custom Metal kernels.

A ``CustomMetalKernel`` accepts a ``template_dtypes`` dict that maps
input-name → placeholder-string. At ``_construct_kernel_op`` time the
placeholder is substituted with the *actual* metal dtype string of that
input (``half``, ``float``, ``bfloat``, …). This produces a per-shape kernel
*variant* — the same Python kernel emitted twice with different dtypes
generates two distinct PSOs because the templated MSL source differs.

These tests pin:

* Single-template substitution lowers to the right metal type.
* Multiple template params on different inputs each get substituted
  independently.
* The same Python kernel called twice in one model with different input
  dtypes emits two distinct kernel sources (and randomized names).
* A template parameter on an input that is *also* fed into another op as
  a regular data input continues to lower without issue.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch

from coreai_torch import (
    MetalParameter,
    TorchConverter,
    TorchMetalKernel,
    get_decomp_table,
)


def _convert_model(
    model: torch.nn.Module,
    args: tuple,
    kernels: list[TorchMetalKernel],
    output_names: list[str] | None = None,
) -> Any:
    exported = torch.export.export(model, args=args)
    ep = exported.run_decompositions(get_decomp_table())
    converter = TorchConverter()
    converter.register_custom_kernels(kernels)
    converter.add_exported_program(ep, output_names=output_names or [])
    return converter.to_coreai()


def _kernel_source_strings(ir: str) -> list[str]:
    """Extract the ``kernel_source = "..."`` string-attr value(s) from IR.

    Each ``coreai.metal4_kernel`` op carries the templated MSL source as a
    string attribute. The randomization suffix on ``kernel_name`` makes
    diffing kernels by name brittle, so tests inspect the source body.
    """
    needle = 'kernel_source = "'
    out = []
    pos = 0
    while True:
        i = ir.find(needle, pos)
        if i < 0:
            return out
        i += len(needle)
        # Find the closing unescaped quote.
        end = i
        while end < len(ir):
            if ir[end] == "\\":
                end += 2
                continue
            if ir[end] == '"':
                break
            end += 1
        out.append(ir[i:end])
        pos = end + 1


# ---------------------------------------------------------------------------


class TestSingleTemplateSubstitution:
    """A single ``template_dtypes`` entry substitutes into the MSL source."""

    @staticmethod
    @pytest.mark.parametrize(
        ("dtype", "expected_metal_type"),
        [
            (torch.float16, "half"),
            (torch.float32, "float"),
            (torch.bfloat16, "bfloat"),
        ],
    )
    def test_template_substitution_picks_metal_type(
        dtype: torch.dtype,
        expected_metal_type: str,
    ) -> None:
        """``TYPE`` in the body is replaced with the metal type of input ``x``."""

        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        kernel = TorchMetalKernel(
            "single_template",
            input_names=["x"],
            result_names=["out"],
            src="out[id] = x[id] + TYPE(1.0);",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
            template_dtypes={"x": "TYPE"},
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(x.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=dtype),),
            kernels=[kernel],
            output_names=["out"],
        )

        sources = _kernel_source_strings(str(coreai_program))
        assert len(sources) == 1, (
            f"Expected exactly one metal4_kernel source, got {len(sources)}"
        )
        # Placeholder replaced.
        assert "TYPE" not in sources[0]
        assert f"{expected_metal_type}(1.0)" in sources[0]


class TestMultipleTemplateParams:
    """Multiple ``template_dtypes`` entries substitute independently."""

    @staticmethod
    def test_two_templates_substitute_independently() -> None:
        """Inputs ``x`` and ``y`` map to ``T_X`` / ``T_Y`` with distinct dtypes."""

        def torch_defn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x.to(torch.float32) + y.to(torch.float32)

        kernel = TorchMetalKernel(
            "two_templates",
            input_names=["x", "y"],
            result_names=["out"],
            src=("T_X xv = x[id]; T_Y yv = y[id]; out[id] = float(xv) + float(yv);"),
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
            template_dtypes={"x": "T_X", "y": "T_Y"},
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    y,
                    threads_per_grid=(x.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        coreai_program = _convert_model(
            Model().eval(),
            args=(
                torch.zeros(4, dtype=torch.float16),  # x → half
                torch.zeros(4, dtype=torch.float32),  # y → float
            ),
            kernels=[kernel],
            output_names=["out"],
        )

        sources = _kernel_source_strings(str(coreai_program))
        assert len(sources) == 1
        src = sources[0]
        assert "T_X" not in src and "T_Y" not in src
        # x's template substituted to `half`, y's to `float`.
        assert "half xv" in src
        assert "float yv" in src


class TestSameKernelTwoDtypeCombinations:
    """Same Python kernel used with two dtype combinations → two distinct PSOs."""

    @staticmethod
    def test_two_invocations_with_different_dtypes_emit_two_sources() -> None:
        """Each ``(rank, metal_dtype)`` combo bypasses the kernel cache."""

        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            return x

        kernel = TorchMetalKernel(
            "two_dtype_combos",
            input_names=["x"],
            result_names=["out"],
            src="out[id] = x[id];",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
            template_dtypes={"x": "TYPE"},
        )

        class Model(torch.nn.Module):
            def forward(
                self, a: torch.Tensor, b: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # `a` is float16 → kernel templated with `half`.
                ra = kernel(
                    a,
                    threads_per_grid=(a.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(a.shape)],
                )
                # `b` is float32 → kernel templated with `float`.
                rb = kernel(
                    b,
                    threads_per_grid=(b.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(b.shape)],
                )
                return ra, rb

        coreai_program = _convert_model(
            Model().eval(),
            args=(
                torch.zeros(4, dtype=torch.float16),
                torch.zeros(4, dtype=torch.float32),
            ),
            kernels=[kernel],
            output_names=["ra", "rb"],
        )

        ir = str(coreai_program)
        sources = _kernel_source_strings(ir)
        assert len(sources) == 2, (
            f"Expected two distinct metal4_kernel sources (one per dtype), "
            f"got {len(sources)}"
        )
        # The two sources differ — one mentions `half`, the other `float`.
        type_a, type_b = sources
        assert type_a != type_b
        assert "device half" in (type_a + type_b)
        assert "device float" in (type_a + type_b)
        # Two distinct randomized names — the kernel cache key includes dtype.
        assert ir.count("coreai.metal4_kernel") == 2


class TestTemplateOnPassthroughInput:
    """A template-bound input that is also forwarded to other ops."""

    @staticmethod
    def test_template_input_is_also_a_data_input() -> None:
        """Same tensor flows into the kernel and into a stock op — both work."""

        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

        kernel = TorchMetalKernel(
            "template_passthrough",
            input_names=["x"],
            result_names=["out"],
            src="out[id] = x[id] + TYPE(1.0);",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
            template_dtypes={"x": "TYPE"},
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                kernel_out = kernel(
                    x,
                    threads_per_grid=(x.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )
                # x is also consumed by a stock aten op in the same graph.
                return kernel_out + torch.relu(x)

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float16),),
            kernels=[kernel],
            output_names=["out"],
        )
        ir = str(coreai_program)
        # Kernel is emitted exactly once (same shape, same dtype → cache hit).
        assert ir.count("coreai.metal4_kernel") == 1
        # The MSL source has had `TYPE` replaced with `half` (input dtype).
        sources = _kernel_source_strings(ir)
        assert "TYPE" not in sources[0]
        assert "half(1.0)" in sources[0]


# ---------------------------------------------------------------------------
# Numerical: end-to-end on macOS
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests run only on Mac")
class TestTemplateNumerical:
    """Same kernel emitted with two dtypes produces correct results in both."""

    @staticmethod
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    async def test_template_specialized_kernel_numerical(
        dtype: torch.dtype,
    ) -> None:
        from ..utils import validate_numerical_output

        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

        kernel = TorchMetalKernel(
            f"template_num_{str(dtype).split('.')[-1]}",
            input_names=["x"],
            result_names=["out"],
            src="out[id] = x[id] * TYPE(2.0);",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
            template_dtypes={"x": "TYPE"},
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(x.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        await validate_numerical_output(
            model=Model().eval(),
            custom_kernels=[kernel],
            metal_inputs=True,
            input_names=["x"],
            output_names=["result"],
            x=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype),
        )
