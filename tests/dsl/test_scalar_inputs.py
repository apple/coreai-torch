# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for scalar inputs to custom Metal kernels (``n_scalar_inputs`` path).

A scalar passed to a ``TorchMetalKernel`` (``int``, ``float``, ``bool``) is
captured by ``get_operand`` as a ``coreai.constant`` rank-0 tensor and bound
to the kernel as a ``constant T& name [[buffer(N)]]`` parameter — a different
runtime path from the regular tensor (``MTLTensor``) bindings.

Both data inputs and scalar inputs share the 31-buffer limit imposed by
``CustomMetalKernel.PARAMETER_LIMIT``. The validation tests here pin that
contract at the converter layer so the failure surfaces with a clear
``ValueError`` rather than as a runtime crash inside MPS.
"""

from __future__ import annotations

import re
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
    """Export, register, and convert a model with custom kernels."""
    exported = torch.export.export(model, args=args)
    ep = exported.run_decompositions(get_decomp_table())
    converter = TorchConverter()
    converter.register_custom_kernels(kernels)
    converter.add_exported_program(ep, output_names=output_names or [])
    return converter.to_coreai()


# ---------------------------------------------------------------------------
# IR-level: scalar inputs lower through the converter
# ---------------------------------------------------------------------------


class TestScalarInputLowering:
    """A scalar input is captured as a rank-0 constant and bound as buffer."""

    @staticmethod
    @pytest.mark.parametrize(
        ("annotation", "scalar_value", "metal_dtype"),
        [
            (float, 2.5, "float"),
            (int, 7, "int"),
            (bool, True, "bool"),
        ],
    )
    def test_single_scalar_input_lowers(
        annotation: type,
        scalar_value: Any,
        metal_dtype: str,
    ) -> None:
        """One tensor input + one scalar input — verify IR signature has a rank-0 operand."""

        if annotation is float:

            def torch_defn(x: torch.Tensor, c: float) -> torch.Tensor:  # type: ignore[misc]
                return x + c
        elif annotation is int:

            def torch_defn(x: torch.Tensor, c: int) -> torch.Tensor:  # type: ignore[misc, no-redef]
                return x + c
        else:

            def torch_defn(x: torch.Tensor, c: bool) -> torch.Tensor:  # type: ignore[misc, no-redef]
                return x + int(c)

        kernel = TorchMetalKernel(
            f"scalar_{metal_dtype}_kernel",
            input_names=["x", "c"],
            result_names=["out"],
            src="out[id] = x[id];",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    scalar_value,
                    threads_per_grid=(x.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float16),),
            kernels=[kernel],
            output_names=["out"],
        )
        ir = str(coreai_program)

        # Kernel name appears in IR.
        assert f"scalar_{metal_dtype}_kernel_" in ir
        # MSL signature emits scalar as `constant T&` rather than `tensor<...>`.
        # The kernel_source attribute embeds the Metal source directly.
        assert f"constant {metal_dtype}& c " in ir, (
            f"Expected `constant {metal_dtype}& c` in the emitted MSL "
            f"source but got: {ir!s}"
        )

    @staticmethod
    def test_max_scalar_inputs_at_buffer_limit() -> None:
        """Scalars-only kernel close to the 31-buffer limit lowers cleanly.

        29 scalar inputs + 1 result = 30 buffers, well under the 31 cap.
        """
        n_scalars = 29

        # `def *args` is rejected by the constructor (variadic). Build a real
        # signature with N scalar parameters via exec.
        scalar_args = ", ".join(f"s{i}: float" for i in range(n_scalars))
        ns: dict[str, Any] = {"torch": torch}
        exec(  # noqa: S102
            f"def torch_defn({scalar_args}) -> torch.Tensor:\n"
            "    return torch.zeros(4, dtype=torch.float16)\n",
            ns,
        )
        torch_defn = ns["torch_defn"]

        kernel = TorchMetalKernel(
            "max_scalar_inputs",
            input_names=[f"s{i}" for i in range(n_scalars)],
            result_names=["out"],
            src="out[id] = 0.0;",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        scalar_values = [float(i) for i in range(n_scalars)]

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (
                    kernel(
                        *scalar_values,
                        threads_per_grid=(4, 1, 1),
                        threads_per_thread_group=(1, 1, 1),
                        result_shapes=[[4]],
                    )
                    + x
                )

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float16),),
            kernels=[kernel],
            output_names=["out"],
        )
        ir = str(coreai_program)
        assert "max_scalar_inputs_" in ir
        # All 29 scalars should appear as `constant float&` parameters.
        assert ir.count("constant float&") == n_scalars

    @staticmethod
    def test_scalars_plus_data_inputs_exceeding_limit_rejected() -> None:
        """Total of (data inputs + scalar inputs + results) > 31 must error."""
        # 25 data inputs + 6 scalar inputs + 1 result = 32 > 31.
        n_data = 25
        n_scalars = 6
        names_data = [f"t{i}" for i in range(n_data)]
        names_scalars = [f"s{i}" for i in range(n_scalars)]

        params = ", ".join(
            [f"{name}: torch.Tensor" for name in names_data]
            + [f"{name}: float" for name in names_scalars]
        )
        body = " + ".join(names_data) + (
            "" if not names_scalars else " + " + " + ".join(names_scalars)
        )
        ns: dict[str, Any] = {"torch": torch}
        exec(  # noqa: S102
            f"def torch_defn({params}) -> torch.Tensor:\n    return {body}\n",
            ns,
        )
        torch_defn = ns["torch_defn"]

        kernel = TorchMetalKernel(
            "over_limit",
            input_names=[*names_data, *names_scalars],
            result_names=["out"],
            src="out[id] = 0.0;",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                tensor_args = [x] * n_data
                scalar_args = [float(i) for i in range(n_scalars)]
                return kernel(
                    *tensor_args,
                    *scalar_args,
                    threads_per_grid=(4, 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        with pytest.raises(ValueError, match=r"metal kernels support 31 inputs"):
            _convert_model(
                Model().eval(),
                args=(torch.zeros(4, dtype=torch.float16),),
                kernels=[kernel],
            )


class TestScalarCaching:
    """Scalar-value-aware kernel caching.

    Scalar-bearing kernels bake the literal into the MSL body, so the base
    class's cache (keyed only on ``(rank, dtype)``) can't be shared blindly
    across call sites. ``TorchMetalKernel`` keeps one sub-cache per distinct set
    of scalar values: identical ``(scalar_values, rank, dtype)`` call sites reuse
    a single templated kernel (one PSO, one randomized name), while differing
    scalar values stay isolated.
    """

    @staticmethod
    def _kernel_names(ir: str) -> list[str]:
        """Every emitted ``metal4_kernel`` name, one per call site."""
        return re.findall(r'kernel_name = "([^"]+)"', ir)

    @staticmethod
    def _scalar_add_kernel(name: str) -> TorchMetalKernel:
        def torch_defn(x: torch.Tensor, c: float) -> torch.Tensor:
            return x + c

        return TorchMetalKernel(
            name,
            input_names=["x", "c"],
            result_names=["out"],
            src="out[id] = x[id] + c;",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

    @staticmethod
    def _two_call_model(
        kernel: TorchMetalKernel,
        scalar_a: float,
        scalar_b: float,
    ) -> torch.nn.Module:
        class Model(torch.nn.Module):
            def forward(
                self, a: torch.Tensor, b: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                ra = kernel(
                    a,
                    scalar_a,
                    threads_per_grid=(a.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(a.shape)],
                )
                rb = kernel(
                    b,
                    scalar_b,
                    threads_per_grid=(b.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(b.shape)],
                )
                return ra, rb

        return Model().eval()

    @staticmethod
    def test_same_scalar_value_reuses_one_kernel() -> None:
        """Two call sites with the same scalar value + shape share one PSO."""
        kernel = TestScalarCaching._scalar_add_kernel("cached_scalar")
        coreai_program = _convert_model(
            TestScalarCaching._two_call_model(kernel, 5.0, 5.0),
            args=(
                torch.zeros(4, dtype=torch.float16),
                torch.ones(4, dtype=torch.float16),
            ),
            kernels=[kernel],
            output_names=["ra", "rb"],
        )
        names = TestScalarCaching._kernel_names(str(coreai_program))
        # One op per call site, but the shared scalar/shape collapses to one PSO.
        assert len(names) == 2
        assert len(set(names)) == 1, (
            f"Expected both call sites to share one kernel name, got {names}"
        )

    @staticmethod
    def test_different_scalar_values_emit_distinct_kernels() -> None:
        """Two call sites with different scalar values stay isolated."""
        kernel = TestScalarCaching._scalar_add_kernel("scalar_per_value")
        coreai_program = _convert_model(
            TestScalarCaching._two_call_model(kernel, 5.0, 9.0),
            args=(
                torch.zeros(4, dtype=torch.float16),
                torch.ones(4, dtype=torch.float16),
            ),
            kernels=[kernel],
            output_names=["ra", "rb"],
        )
        ir = str(coreai_program)
        names = TestScalarCaching._kernel_names(ir)
        assert len(names) == 2
        assert len(set(names)) == 2, (
            f"Expected two distinct kernels for differing scalar values, got {names}"
        )
        # Each call site baked its own literal.
        assert "c = 5.0" in ir
        assert "c = 9.0" in ir


# ---------------------------------------------------------------------------
# Numerical: scalar input behavior end-to-end (macOS only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests run only on Mac")
class TestScalarInputNumerical:
    """End-to-end numerical correctness when scalars are baked into the graph."""

    @staticmethod
    async def test_float_scalar_added_elementwise() -> None:
        """A float scalar passed alongside a tensor produces correct output."""
        from ..utils import validate_numerical_output

        def torch_defn(x: torch.Tensor, c: float) -> torch.Tensor:
            return x + c

        kernel = TorchMetalKernel(
            "scalar_add_float",
            input_names=["x", "c"],
            result_names=["out"],
            src="out[id] = x[id] + (TYPE)c;",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
            template_dtypes={"x": "TYPE"},
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    3.5,
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
            x=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        )

    @staticmethod
    async def test_int_scalar_used_as_index_offset() -> None:
        """An int scalar is bound and read in the kernel body."""
        from ..utils import validate_numerical_output

        def torch_defn(x: torch.Tensor, n: int) -> torch.Tensor:
            return x + float(n)

        kernel = TorchMetalKernel(
            "scalar_add_int",
            input_names=["x", "n"],
            result_names=["out"],
            src="out[id] = x[id] + float(n);",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    7,
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
            x=torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32),
        )

    @staticmethod
    async def test_bool_scalar_branches_kernel_path() -> None:
        """A bool scalar selects between two kernel branches at runtime."""
        from ..utils import validate_numerical_output

        def torch_defn(x: torch.Tensor, flag: bool) -> torch.Tensor:
            return x * 2.0 if flag else x

        kernel = TorchMetalKernel(
            "scalar_bool_branch",
            input_names=["x", "flag"],
            result_names=["out"],
            src="out[id] = flag ? (x[id] * 2.0f) : x[id];",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    True,
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
            x=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        )
