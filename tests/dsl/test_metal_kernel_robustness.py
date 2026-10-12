# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Robustness / hardening tests for :class:`TorchMetalKernel`.

Two categories of tests live here:

1. **Stress tests** for shapes the feature must support without falling over
   (no kernel inputs, empty kernel body, kernels chained with normal torch
   ops, kernels inside control flow).
2. **Adversarial tests** that pin down validation behavior at the Python
   layer so misuse fails fast with a clear message instead of surfacing as
   a runtime crash from Metal / the Swift runtime.

The tests are conversion / IR-only (no GPU dispatch) so they are stable on
any macOS host. Each kernel name is unique per test to avoid collisions
in ``torch.library``'s global custom-op registry across the same session.
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _convert_model(
    model: torch.nn.Module,
    args: tuple,
    kernels: list[TorchMetalKernel],
    output_names: list[str] | None = None,
) -> Any:
    """Export a model and convert it to a CoreAI program."""
    exported = torch.export.export(model, args=args)
    ep = exported.run_decompositions(get_decomp_table())
    converter = TorchConverter()
    converter.register_custom_kernels(kernels)
    converter.add_exported_program(ep, output_names=output_names or [])
    return converter.to_coreai()


def _identity_kernel(
    name: str,
    *,
    src: str = "output[id] = x[id];",
) -> TorchMetalKernel:
    """Convenience: identity kernel with one tensor input and one tensor output."""

    def torch_defn(x: torch.Tensor) -> torch.Tensor:
        return x

    return TorchMetalKernel(
        name,
        input_names=["x"],
        result_names=["output"],
        src=src,
        torch_defn=torch_defn,
        metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
    )


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests run only on Mac")
class TestNoKernelInputs:
    """Kernels with zero kernel inputs (only outputs + thread params)."""

    @staticmethod
    def test_constructor_accepts_empty_input_names() -> None:
        """A kernel with no kernel inputs is still a valid construction."""

        def torch_defn() -> torch.Tensor:
            return torch.zeros(4, dtype=torch.float16)

        kernel = TorchMetalKernel(
            "robustness_no_inputs_ctor",
            input_names=[],
            result_names=["output"],
            src="output[id] = 0;",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        assert kernel.input_names == []
        assert kernel.result_names == ["output"]

    @staticmethod
    def test_no_input_kernel_lowers_through_converter() -> None:
        """A no-input kernel embedded in a model lowers to MLIR cleanly."""

        def torch_defn() -> torch.Tensor:
            return torch.zeros(4, dtype=torch.float16)

        kernel = TorchMetalKernel(
            "robustness_no_inputs_lower",
            input_names=[],
            result_names=["output"],
            src="output[id] = 0;",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                k = kernel(
                    threads_per_grid=(4, 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[[4]],
                )
                return x + k

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float16),),
            kernels=[kernel],
        )
        # Sanity: emitted IR mentions our kernel name.
        assert "robustness_no_inputs_lower_" in str(coreai_program)


class TestNoOutputs:
    """Kernels with no outputs are not meaningful and must be rejected."""

    @staticmethod
    def test_empty_result_names_rejected() -> None:
        """A 0-output kernel is rejected with a clear message at construction."""

        def torch_defn(x: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            return torch.zeros(())

        with pytest.raises(
            ValueError,
            match="result_names must contain at least one entry",
        ):
            TorchMetalKernel(
                "robustness_no_outputs",
                input_names=["x"],
                result_names=[],
                src="",
                torch_defn=torch_defn,
                metal_params=[
                    MetalParameter("id", "uint", "thread_position_in_grid"),
                ],
            )


class TestEmptyKernelBody:
    """Kernels whose Metal body is empty are well-formed (just side-effect-free)."""

    @staticmethod
    def test_empty_body_construction_succeeds() -> None:
        """An empty body ``src`` is acceptable — Python should not over-validate."""
        kernel = _identity_kernel("robustness_empty_body_ctor", src="")
        assert kernel.src == ""

    @staticmethod
    def test_empty_body_lowers_through_converter() -> None:
        """An empty body still produces valid Metal source and lowers cleanly."""
        kernel = _identity_kernel("robustness_empty_body_lower", src="")

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(4, 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float16),),
            kernels=[kernel],
            output_names=["out"],
        )
        # Even though the body is empty, the kernel signature still appears in IR.
        assert "robustness_empty_body_lower_" in str(coreai_program)


class TestKernelInteractionWithOtherOps:
    """Custom kernels should compose freely with the rest of the converter."""

    @staticmethod
    def test_kernel_chained_with_aten_ops() -> None:
        """Custom kernel result feeds into / is fed by stock aten ops."""
        kernel = _identity_kernel(
            "robustness_with_other_ops",
            src="output[id] = x[id] * x[id];",  # square
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                squared = kernel(
                    torch.relu(x),
                    threads_per_grid=(x.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )
                return squared + 1.0

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float16),),
            kernels=[kernel],
            output_names=["out"],
        )
        ir = str(coreai_program)
        assert "robustness_with_other_ops_" in ir
        # We also see at least one normal coreai op (relu / add) in IR.
        assert "coreai." in ir

    @staticmethod
    def test_kernel_invoked_twice_in_one_graph() -> None:
        """Same kernel called twice — kernel cache keeps a single signature."""
        kernel = _identity_kernel("robustness_called_twice")

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a = kernel(
                    x,
                    threads_per_grid=(x.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )
                b = kernel(
                    a,
                    threads_per_grid=(x.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )
                return b

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float16),),
            kernels=[kernel],
            output_names=["out"],
        )
        ir = str(coreai_program)
        # Two metal4_kernel ops emitted, but kernel cache means a single
        # randomized name (the same suffix appears twice).
        assert ir.count("coreai.metal4_kernel") == 2


class TestKernelInControlFlow:
    """Custom kernels inside ``torch.cond`` branches.

    The converter now threads user-defined torch lowerings (and the
    ``coreai``/``coreaix`` resolver) through ``replace_cond`` /
    ``replace_while_loop`` into :func:`coreai_torch._utils.convert_branch_subgraph`,
    so a custom kernel inside a branch resolves instead of raising
    ``unsupported op in branch``. That dispatch fix is unit-tested in
    ``tests/test_branch_dispatch.py``.

    Exercising it end to end through ``torch.export`` is still blocked upstream:
    a ``torch.library.custom_op`` whose fake implementation returns a fresh
    tensor cannot be traced inside a ``torch.cond`` branch through
    ``run_decompositions`` on the pinned torch (it raises "Attempting to use
    FunctionalTensor on its own"), and an identity fake instead trips
    ``torch.cond``'s no-aliasing rule at export. The test below pins that
    remaining upstream gap; once torch can trace it, this ``xfail`` flips to a
    failure and should become a passing assertion (the converter half already
    works).
    """

    @staticmethod
    @pytest.mark.xfail(
        reason=(
            "Custom kernel inside a torch.cond branch cannot be traced through "
            "torch.export.run_decompositions on the pinned torch (FunctionalTensor "
            "error re-functionalizing the custom-op fake inside cond). The converter "
            "dispatch fix itself is covered by tests/test_branch_dispatch.py."
        ),
        strict=True,
    )
    def test_kernel_inside_cond_branch() -> None:
        """Conversion should succeed once torch can trace the kernel in a branch."""

        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            # Non-aliasing (mirrors the kernel body `y = 2*x`); an identity
            # torch_defn would instead fail torch.cond's aliasing check.
            return x * 2

        kernel = TorchMetalKernel(
            "robustness_cond_branch",
            input_names=["x"],
            result_names=["output"],
            src="output[id] = x[id] + x[id];",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                def true_branch(t: torch.Tensor) -> torch.Tensor:
                    return kernel(
                        t,
                        threads_per_grid=(4, 1, 1),
                        threads_per_thread_group=(1, 1, 1),
                        result_shapes=[list(t.shape)],
                    )

                def false_branch(t: torch.Tensor) -> torch.Tensor:
                    return t + 1.0

                return torch.cond(x.sum() > 0, true_branch, false_branch, [x])

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float16),),
            kernels=[kernel],
            output_names=["out"],
        )
        assert "robustness_cond_branch_" in str(coreai_program)


# ---------------------------------------------------------------------------
# Adversarial / validation tests
# ---------------------------------------------------------------------------


class TestNameValidation:
    """The Swift runtime requires non-empty kernel names."""

    @staticmethod
    @pytest.mark.parametrize("bad_name", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_name_rejected(bad_name: str) -> None:
        """Empty / whitespace names fail at construction."""

        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            return x

        with pytest.raises(
            ValueError,
            match="Kernel name must be a non-empty string",
        ):
            TorchMetalKernel(
                bad_name,
                input_names=["x"],
                result_names=["out"],
                src="out[id] = x[id];",
                torch_defn=torch_defn,
                metal_params=[
                    MetalParameter("id", "uint", "thread_position_in_grid"),
                ],
            )


class TestIONameValidation:
    """Duplicate / overlapping input/result names produce broken Metal source."""

    @staticmethod
    def test_duplicate_input_names_rejected() -> None:
        def torch_defn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            return x

        with pytest.raises(
            ValueError,
            match=re.escape("Duplicate input names: ['x']"),
        ):
            TorchMetalKernel(
                "robustness_duplicate_input",
                input_names=["x", "x"],
                result_names=["out"],
                src="out[id] = x[id];",
                torch_defn=torch_defn,
            )

    @staticmethod
    def test_duplicate_result_names_rejected() -> None:
        def torch_defn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x, x

        with pytest.raises(
            ValueError,
            match=re.escape("Duplicate result names: ['out']"),
        ):
            TorchMetalKernel(
                "robustness_duplicate_result",
                input_names=["x"],
                result_names=["out", "out"],
                src="out[id] = x[id];",
                torch_defn=torch_defn,
            )

    @staticmethod
    def test_input_result_name_overlap_rejected() -> None:
        """Same identifier in input and result lists would shadow itself in Metal."""

        def torch_defn(shared: torch.Tensor) -> torch.Tensor:
            return shared

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Names appear in both input_names and result_names: ['shared']"
            ),
        ):
            TorchMetalKernel(
                "robustness_io_overlap",
                input_names=["shared"],
                result_names=["shared"],
                src="shared[id] = shared[id];",
                torch_defn=torch_defn,
            )


class TestTorchDefnSignatureValidation:
    """Adversarial torch_defn signatures."""

    @staticmethod
    def test_var_positional_rejected() -> None:
        def torch_defn(*tensors: torch.Tensor) -> torch.Tensor:
            return tensors[0]

        with pytest.raises(
            TypeError,
            match="custom kernels do not support variadic parameters",
        ):
            TorchMetalKernel(
                "robustness_varargs",
                input_names=["x"],
                result_names=["out"],
                src="out[id] = x[id];",
                torch_defn=torch_defn,
            )

    @staticmethod
    def test_var_keyword_rejected() -> None:
        def torch_defn(x: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # noqa: ARG001
            return x

        with pytest.raises(
            TypeError,
            match="custom kernels do not support variadic parameters",
        ):
            TorchMetalKernel(
                "robustness_varkwargs",
                input_names=["x"],
                result_names=["out"],
                src="out[id] = x[id];",
                torch_defn=torch_defn,
            )


class TestReturnCountValidation:
    """Return-count must match ``len(result_names)`` for concrete annotations."""

    @staticmethod
    def test_single_tensor_return_with_multiple_result_names_rejected() -> None:
        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            return x

        err = "torch_defn returns a single torch.Tensor, but result_names has 2 entries"
        with pytest.raises(ValueError, match=re.escape(err)):
            TorchMetalKernel(
                "robustness_single_return_multi_names",
                input_names=["x"],
                result_names=["out_a", "out_b"],
                src="out_a[id] = x[id]; out_b[id] = x[id];",
                torch_defn=torch_defn,
            )

    @staticmethod
    def test_tuple_return_count_mismatch_rejected() -> None:
        def torch_defn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x, x

        err = "torch_defn returns tuple of 2 tensors, but result_names has 3 entries"
        with pytest.raises(ValueError, match=re.escape(err)):
            TorchMetalKernel(
                "robustness_tuple_count_mismatch",
                input_names=["x"],
                result_names=["a", "b", "c"],
                src="a[id]=x[id]; b[id]=x[id]; c[id]=x[id];",
                torch_defn=torch_defn,
            )

    @staticmethod
    def test_list_return_count_validated_at_call_time_only() -> None:
        """``list[Tensor]`` is variable-length; accept at construction."""

        def torch_defn(x: torch.Tensor) -> list[torch.Tensor]:
            return [x, x]

        kernel = TorchMetalKernel(
            "robustness_list_return",
            input_names=["x"],
            result_names=["a", "b"],
            src="a[id]=x[id]; b[id]=x[id];",
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )
        assert kernel.result_names == ["a", "b"]


class TestCallTimeValidation:
    """``__call__`` must validate dispatch parameters before reaching torch.library."""

    @staticmethod
    @pytest.fixture
    def kernel() -> TorchMetalKernel:
        return _identity_kernel("robustness_call_args")

    @staticmethod
    @pytest.mark.parametrize("bad_grid", [(1, 1), (1, 1, 1, 1), ()])
    def test_threads_per_grid_must_be_3_tuple(
        kernel: TorchMetalKernel,
        bad_grid: tuple[int, ...],
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"threads_per_grid must be a 3-tuple",
        ):
            kernel(
                torch.zeros(4, dtype=torch.float16),
                threads_per_grid=bad_grid,
                threads_per_thread_group=(1, 1, 1),
                result_shapes=[[4]],
            )

    @staticmethod
    @pytest.mark.parametrize("bad_group", [(1,), (1, 1, 1, 1)])
    def test_threads_per_thread_group_must_be_3_tuple(
        kernel: TorchMetalKernel,
        bad_group: tuple[int, ...],
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"threads_per_thread_group must be a 3-tuple",
        ):
            kernel(
                torch.zeros(4, dtype=torch.float16),
                threads_per_grid=(1, 1, 1),
                threads_per_thread_group=bad_group,
                result_shapes=[[4]],
            )

    @staticmethod
    def test_result_shapes_count_must_match_result_names(
        kernel: TorchMetalKernel,
    ) -> None:
        """Single-output kernel must receive exactly one result shape."""
        with pytest.raises(
            ValueError,
            match=r"result_shapes must contain one shape per result name",
        ):
            kernel(
                torch.zeros(4, dtype=torch.float16),
                threads_per_grid=(1, 1, 1),
                threads_per_thread_group=(1, 1, 1),
                result_shapes=[[4], [4]],  # two shapes, but only one result_name
            )


class TestRegistrationCollisions:
    """``register_custom_kernels`` must not silently overwrite a lowering."""

    @staticmethod
    def test_register_custom_kernels_twice_fails() -> None:
        """``register_custom_kernels`` refuses to override an existing lowering."""
        kernel = _identity_kernel("robustness_register_twice")

        converter = TorchConverter()
        converter.register_custom_kernels([kernel])
        with pytest.raises(ValueError, match="already registered"):
            converter.register_custom_kernels([kernel])


# ---------------------------------------------------------------------------
# Sanity: existing happy-path still works after the validation additions.
# ---------------------------------------------------------------------------


class TestPostFixSanity:
    """Cheap smoke test: a typical kernel still constructs and lowers."""

    @staticmethod
    def test_typical_kernel_still_lowers() -> None:
        def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        kernel = TorchMetalKernel(
            "robustness_postfix_sanity",
            input_names=["x", "y"],
            result_names=["sum"],
            src="sum[id] = x[id] + y[id];",
            torch_defn=torch_add,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
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
                torch.zeros(4, dtype=torch.float16),
                torch.zeros(4, dtype=torch.float16),
            ),
            kernels=[kernel],
            output_names=["sum"],
        )
        assert "robustness_postfix_sanity_" in str(coreai_program)
