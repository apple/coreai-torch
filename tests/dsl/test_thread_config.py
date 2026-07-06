# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Thread-configuration edge cases for ``TorchMetalKernel`` dispatch.

The runtime forwards ``threads_per_grid`` and ``threads_per_threadgroup``
verbatim to Metal's ``dispatchThreads``. Threadgroup sizes exceeding
``maxTotalThreadsPerThreadgroup`` (1024 on current Apple Silicon) are
rejected by Metal at PSO time; valid sizes near or above the visible
grid are simply rounded up — the kernel is responsible for guarding
out-of-bounds reads/writes when ``threads_per_grid`` exceeds the
visible tensor extent.

Note: ``MTLTensor`` extents are stored in *reverse* of the torch shape
(see ``NDArray+Metal.swift``: ``shapeSpan.reversed()``). For a torch
tensor of shape ``(D0, D1, D2)`` the kernel sees extents
``(D2, D1, D0)``; ``get_extent(0)`` is the innermost (fastest-varying)
torch dim. Multi-dim dispatch tuples must match this convention.

Only IR-level checks here can be fully cross-platform; the numerical tests
below need a Metal-backed runtime to actually execute. They are gated on
macOS via the dsl conftest's collection hook.
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


def _make_identity_kernel(name: str) -> TorchMetalKernel:
    """Identity kernel with explicit bounds-check on ``id``."""

    def torch_defn(x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    return TorchMetalKernel(
        name,
        input_names=["x"],
        result_names=["out"],
        src=("if (id >= x.get_extent(0)) return; out[id] = x[id];"),
        torch_defn=torch_defn,
        metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
    )


# ---------------------------------------------------------------------------
# IR-level: dispatch values land in the IR unchanged
# ---------------------------------------------------------------------------


class TestDispatchValueLowering:
    """The 3-tuple values land in the IR; clamping is a runtime concern."""

    @staticmethod
    def test_unit_grid_dispatch_lowers() -> None:
        """``threads_per_grid=(1,1,1)`` is valid and lowers cleanly.

        The kernel's bounds-check guards against the over-large tensor.
        """
        kernel = _make_identity_kernel("thread_unit_grid")

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(1, 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float16),),
            kernels=[kernel],
            output_names=["out"],
        )
        assert "thread_unit_grid_" in str(coreai_program)

    @staticmethod
    def test_full_3d_dispatch_lowers() -> None:
        """Non-trivial values for x, y and z dimensions all lower."""
        kernel = _make_identity_kernel("thread_full_3d")

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(8, 4, 2),
                    threads_per_thread_group=(4, 2, 2),
                    result_shapes=[list(x.shape)],
                )

        # 64-element flat tensor. Kernel's bounds-check ensures correctness
        # regardless of the 3D dispatch decomposition.
        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(64, dtype=torch.float16),),
            kernels=[kernel],
            output_names=["out"],
        )
        assert "thread_full_3d_" in str(coreai_program)

    @staticmethod
    def test_threadgroup_larger_than_typical_pso_max_lowers() -> None:
        """An over-large ``threads_per_thread_group`` is accepted at the IR layer.

        The runtime clamps this to ``pso.maxTotalThreadsPerThreadgroup``
        (1024 on most Apple Silicon GPUs); the converter should not pre-validate
        that — it's a hardware property only known at PSO compilation time.
        """
        kernel = _make_identity_kernel("thread_clamping_test")

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(2048, 1, 1),
                    # Way above any real GPU's max threadgroup size; runtime
                    # clamping must handle this gracefully.
                    threads_per_thread_group=(2048, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        coreai_program = _convert_model(
            Model().eval(),
            args=(torch.zeros(2048, dtype=torch.float16),),
            kernels=[kernel],
            output_names=["out"],
        )
        assert "thread_clamping_test_" in str(coreai_program)


# ---------------------------------------------------------------------------
# Numerical: behavior under unusual dispatch configurations
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests run only on Mac")
class TestThreadDispatchNumerical:
    """Behavior under boundary dispatch configurations."""

    @staticmethod
    async def test_threadgroup_larger_than_grid_does_not_corrupt_output() -> None:
        """Threadgroup size larger than the grid still dispatches correctly.

        ``dispatchThreads`` rounds the grid up to the next multiple of the
        threadgroup, so a 1024-wide threadgroup over a 64-element grid still
        launches one full threadgroup. The kernel's bounds-check filters out
        the over-dispatched threads.
        """
        from ..utils import validate_numerical_output

        kernel = _make_identity_kernel("thread_clamp_numerical")

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(x.shape[0], 1, 1),
                    # 1024 = maxTotalThreadsPerThreadgroup on current Apple
                    # Silicon — the largest threadgroup Metal will accept.
                    threads_per_thread_group=(1024, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        await validate_numerical_output(
            model=Model().eval(),
            custom_kernels=[kernel],
            metal_inputs=True,
            input_names=["x"],
            output_names=["result"],
            x=torch.arange(64, dtype=torch.float32),
        )

    @staticmethod
    async def test_full_3d_dispatch_numerical() -> None:
        """A full 3D dispatch produces the same identity result as a 1D one."""
        from ..utils import validate_numerical_output

        # Index a 3D tensor by (gid.x, gid.y, gid.z).
        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        kernel = TorchMetalKernel(
            "thread_3d_identity",
            input_names=["x"],
            result_names=["out"],
            src=(
                "if (gid.x >= x.get_extent(0) || "
                "    gid.y >= x.get_extent(1) || "
                "    gid.z >= x.get_extent(2)) return; "
                "out[gid.x, gid.y, gid.z] = x[gid.x, gid.y, gid.z];"
            ),
            torch_defn=torch_defn,
            metal_params=[
                MetalParameter("gid", "uint3", "thread_position_in_grid"),
            ],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    # MTLTensor extents are reversed from the torch shape, so
                    # ``get_extent(0)`` is the innermost torch dim. Dispatch
                    # in the same reversed order so each ``gid`` axis lines
                    # up with the matching ``get_extent``.
                    threads_per_grid=(x.shape[2], x.shape[1], x.shape[0]),
                    threads_per_thread_group=(2, 2, 2),
                    result_shapes=[list(x.shape)],
                )

        await validate_numerical_output(
            model=Model().eval(),
            custom_kernels=[kernel],
            metal_inputs=True,
            input_names=["x"],
            output_names=["result"],
            x=torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6),
        )

    @staticmethod
    async def test_unit_grid_writes_only_first_element() -> None:
        """``threads_per_grid=(1,1,1)`` writes exactly one element.

        The output buffer for the un-touched elements remains at whatever the
        runtime initialized it to. The kernel here writes a sentinel into the
        first position so we can assert it landed.
        """
        from ..utils import validate_numerical_output

        # The kernel zeros the output and only writes index 0. We then add the
        # input tensor downstream so the model's torch reference is the same.
        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            out[0] = x[0]
            return out

        kernel = TorchMetalKernel(
            "thread_unit_grid_numerical",
            input_names=["x"],
            result_names=["out"],
            src=(
                # Initialize all elements to 0, then thread 0 writes x[0].
                "for (uint i = 0; i < x.get_extent(0); ++i) out[i] = 0; "
                "if (id == 0) out[0] = x[0];"
            ),
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(1, 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        await validate_numerical_output(
            model=Model().eval(),
            custom_kernels=[kernel],
            metal_inputs=True,
            input_names=["x"],
            output_names=["result"],
            x=torch.tensor([7.0, 1.0, 2.0, 3.0], dtype=torch.float32),
        )

    @staticmethod
    async def test_grid_smaller_than_tensor_size_with_bounds_check() -> None:
        """Under-dispatch leaves untouched tail elements; the kernel must not OOB.

        Sentinel-output kernel that initializes the entire output to zero
        across all dispatched threads, then only writes ``id < grid_size``.
        Reference torch_defn matches.
        """
        from ..utils import validate_numerical_output

        # Tensor size = 16, grid = 8 → only first 8 outputs are written from x.
        # Reference torch_defn replicates: out[0..8] = x[0..8] * 2, rest = 0.
        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            out[:8] = x[:8] * 2
            return out

        kernel = TorchMetalKernel(
            "thread_under_dispatch",
            input_names=["x"],
            result_names=["out"],
            src=(
                # Each dispatched thread is responsible for its own tail
                # elements as well, zeroing them.
                "for (uint i = id; i < x.get_extent(0); i += 8) out[i] = 0; "
                "if (id < 8 && id < x.get_extent(0)) out[id] = x[id] * 2;"
            ),
            torch_defn=torch_defn,
            metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
        )

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(8, 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        await validate_numerical_output(
            model=Model().eval(),
            custom_kernels=[kernel],
            metal_inputs=True,
            input_names=["x"],
            output_names=["result"],
            x=torch.arange(16, dtype=torch.float32),
        )
