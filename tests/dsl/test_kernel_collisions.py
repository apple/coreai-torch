# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for kernel-name collision behavior.

Two ``CustomMetalKernel`` instances each pick an 8-character random suffix per
``(rank, dtype)`` combination at lowering time, so the *emitted* function
names cannot collide across distinct kernels — even when the user-visible
``name`` field is identical. The MPS runtime's ``MPSRuntime.mm`` dedupe
(by ``function_name``) is therefore not directly reachable from
coreai-torch unless the same kernel is reused.

These tests pin:

* Two kernels with the same ``name`` are rejected at
  ``register_custom_kernels`` time, with a clear "already registered" error.
* A kernel that is registered, used, then re-registered (or registered twice
  in the same call) fails the same way.
* A single kernel reused with the same ``(rank, dtype)`` reuses the cached
  randomized name (the cache short-circuits MSL re-templating).
* A single kernel reused with *different* ``(rank, dtype)`` produces two
  distinct randomized names — exercising the cache-miss path.
"""

from __future__ import annotations

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


def _identity_kernel(name: str, *, src: str = "out[id] = x[id];") -> TorchMetalKernel:
    def torch_defn(x: torch.Tensor) -> torch.Tensor:
        return x

    return TorchMetalKernel(
        name,
        input_names=["x"],
        result_names=["out"],
        src=src,
        torch_defn=torch_defn,
        metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
    )


def _helper_guard_names(kernel: TorchMetalKernel) -> set[str]:
    return {
        line.removeprefix("#ifndef ")
        for _, source in kernel.kernel_cache.values()
        for line in source.splitlines()
        if line.startswith("#ifndef COREAI_TORCH_HELPER_")
    }


# ---------------------------------------------------------------------------
# Collision at register time
# ---------------------------------------------------------------------------


class TestRegisterTimeCollision:
    """Two distinct kernel objects with the same ``name`` field."""

    @staticmethod
    def test_same_name_identical_source_rejected_at_register() -> None:
        """Even with the *same* MSL source, the second registration fails fast.

        A coreai-torch ``register_custom_kernels`` call cannot tell that two
        kernels are equivalent; its only option is to fail and let the user
        register a single instance.
        """
        kernel_a = _identity_kernel("name_collision_same_src")
        kernel_b = _identity_kernel("name_collision_same_src")

        converter = TorchConverter()
        with pytest.raises(ValueError, match="already registered"):
            converter.register_custom_kernels([kernel_a, kernel_b])

    @staticmethod
    def test_same_name_different_source_rejected_at_register() -> None:
        """Distinct MSL bodies under the same ``name`` would silently shadow.

        The converter must not allow this — the second register call should
        raise.
        """
        kernel_a = _identity_kernel("name_collision_diff_src", src="out[id] = x[id];")
        kernel_b = _identity_kernel(
            "name_collision_diff_src", src="out[id] = x[id] * x[id];"
        )

        converter = TorchConverter()
        with pytest.raises(ValueError, match="already registered"):
            converter.register_custom_kernels([kernel_a, kernel_b])

    @staticmethod
    def test_same_name_split_across_two_register_calls_rejected() -> None:
        """Splitting the two registrations across calls still collides."""
        kernel_a = _identity_kernel("name_collision_two_calls")
        kernel_b = _identity_kernel("name_collision_two_calls")

        converter = TorchConverter()
        converter.register_custom_kernels([kernel_a])
        with pytest.raises(ValueError, match="already registered"):
            converter.register_custom_kernels([kernel_b])

    @staticmethod
    def test_distinct_names_register_cleanly() -> None:
        """The collision check is keyed on ``name``, not on object identity."""
        kernel_a = _identity_kernel("name_distinct_a")
        kernel_b = _identity_kernel("name_distinct_b")

        converter = TorchConverter()
        converter.register_custom_kernels([kernel_a, kernel_b])
        # No error.


# ---------------------------------------------------------------------------
# Shared helper source guards
# ---------------------------------------------------------------------------


class TestSharedHelperGuards:
    """Rendered helper text is emitted once per combined Metal library."""

    @staticmethod
    def test_identical_helpers_share_one_content_guard() -> None:
        helper = (
            "static inline float shared_increment(float value) { return value + 1.0f; }"
        )

        def make_kernel(name: str) -> TorchMetalKernel:
            def torch_defn(x: torch.Tensor) -> torch.Tensor:
                return x + 1

            return TorchMetalKernel(
                name,
                input_names=["x"],
                result_names=["out"],
                src="out[id] = shared_increment(x[id]);",
                torch_defn=torch_defn,
                helper_src=helper,
                metal_params=[
                    MetalParameter("id", "uint", "thread_position_in_grid"),
                ],
            )

        kernel_a = make_kernel("shared_helper_a")
        kernel_b = make_kernel("shared_helper_b")

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                outputs = []
                for kernel in (kernel_a, kernel_b):
                    outputs.append(
                        kernel(
                            x,
                            threads_per_grid=(x.shape[0], 1, 1),
                            threads_per_thread_group=(1, 1, 1),
                            result_shapes=[list(x.shape)],
                        ),
                    )
                return outputs[0], outputs[1]

        _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float32),),
            kernels=[kernel_a, kernel_b],
            output_names=["a", "b"],
        )

        guards_a = _helper_guard_names(kernel_a)
        guards_b = _helper_guard_names(kernel_b)
        assert len(guards_a) == 1
        assert guards_a == guards_b
        guard = next(iter(guards_a))
        for kernel in (kernel_a, kernel_b):
            source = next(iter(kernel.kernel_cache.values()))[1]
            assert source.count(f"#ifndef {guard}") == 1
            assert source.count(f"#define {guard}") == 1
            assert source.count(f"#endif  // {guard}") == 1

    @staticmethod
    def test_rendered_dtype_variants_get_distinct_guards() -> None:
        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        kernel = TorchMetalKernel(
            "templated_shared_helper",
            input_names=["x"],
            result_names=["out"],
            src="out[id] = typed_increment(x[id]);",
            torch_defn=torch_defn,
            helper_src=(
                "static inline TYPE typed_increment(TYPE value) "
                "{ return value + TYPE(1); }"
            ),
            metal_params=[
                MetalParameter("id", "uint", "thread_position_in_grid"),
            ],
            template_dtypes={"x": "TYPE"},
        )

        class Model(torch.nn.Module):
            def forward(
                self,
                half_input: torch.Tensor,
                float_input: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                half_output = kernel(
                    half_input,
                    threads_per_grid=(half_input.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(half_input.shape)],
                )
                float_output = kernel(
                    float_input,
                    threads_per_grid=(float_input.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(float_input.shape)],
                )
                return half_output, float_output

        _convert_model(
            Model().eval(),
            args=(
                torch.zeros(4, dtype=torch.float16),
                torch.zeros(4, dtype=torch.float32),
            ),
            kernels=[kernel],
            output_names=["half_output", "float_output"],
        )

        guards = _helper_guard_names(kernel)
        assert len(kernel.kernel_cache) == 2
        assert len(guards) == 2
        assert all("TYPE" not in source for _, source in kernel.kernel_cache.values())

    @staticmethod
    def test_kernel_without_helpers_is_unchanged() -> None:
        kernel = _identity_kernel("no_helper_guard")

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return kernel(
                    x,
                    threads_per_grid=(x.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(x.shape)],
                )

        _convert_model(
            Model().eval(),
            args=(torch.zeros(4, dtype=torch.float32),),
            kernels=[kernel],
            output_names=["out"],
        )

        source = next(iter(kernel.kernel_cache.values()))[1]
        assert "COREAI_TORCH_HELPER_" not in source


# ---------------------------------------------------------------------------
# Per-instance kernel cache: same-instance reuse
# ---------------------------------------------------------------------------


class TestPerInstanceCaching:
    """The ``kernel_cache`` keys randomized names by ``(rank, dtype)``."""

    @staticmethod
    def test_same_kernel_same_shape_dtype_reuses_cached_name() -> None:
        """Calling the same kernel twice with the same shape reuses the source."""
        kernel = _identity_kernel("cache_hit")

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
        assert ir.count("coreai.metal4_kernel") == 2
        # The kernel cache is keyed on (rank, metal_dtype). Both calls have
        # rank-1 + half tensors → same cached randomized name.
        cached_entries = list(kernel.kernel_cache.values())
        assert len(cached_entries) == 1, (
            f"Expected exactly one cached randomized name, got {len(cached_entries)}"
        )

    @staticmethod
    def test_same_kernel_different_dtypes_emits_two_cache_entries() -> None:
        """Distinct ``(rank, dtype)`` combinations produce distinct PSO sources."""

        def torch_defn(x: torch.Tensor) -> torch.Tensor:
            return x

        kernel = TorchMetalKernel(
            "cache_miss",
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
                ra = kernel(
                    a,
                    threads_per_grid=(a.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(a.shape)],
                )
                rb = kernel(
                    b,
                    threads_per_grid=(b.shape[0], 1, 1),
                    threads_per_thread_group=(1, 1, 1),
                    result_shapes=[list(b.shape)],
                )
                return ra, rb

        _convert_model(
            Model().eval(),
            args=(
                torch.zeros(4, dtype=torch.float16),
                torch.zeros(4, dtype=torch.float32),
            ),
            kernels=[kernel],
            output_names=["ra", "rb"],
        )
        # Two distinct (rank, dtype) keys → two cache entries.
        cached_entries = list(kernel.kernel_cache.values())
        assert len(cached_entries) == 2
        # Each entry has a unique randomized name.
        names = {entry[0] for entry in cached_entries}
        assert len(names) == 2

    @staticmethod
    def test_two_instances_same_name_get_distinct_randomized_names() -> None:
        """Two kernel objects with the same ``name`` randomize independently.

        Even though ``register_custom_kernels`` rejects this, the underlying
        randomized-name machinery must produce distinct suffixes per instance
        so that any future cross-converter use cannot silently collide.
        """
        kernel_a = _identity_kernel("instance_a")
        kernel_b = _identity_kernel("instance_a")

        # Drive each through its own converter to bypass the register-time
        # collision check; the kernel_cache is per-instance.
        for k in (kernel_a, kernel_b):

            class Model(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self._k = k

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self._k(
                        x,
                        threads_per_grid=(x.shape[0], 1, 1),
                        threads_per_thread_group=(1, 1, 1),
                        result_shapes=[list(x.shape)],
                    )

            converter = TorchConverter()
            converter.register_custom_kernels([k])
            ep = torch.export.export(
                Model().eval(), args=(torch.zeros(4, dtype=torch.float16),)
            ).run_decompositions(get_decomp_table())
            converter.add_exported_program(ep, output_names=["out"])
            converter.to_coreai()

        # Each instance has its own cache; the randomized names cannot overlap.
        names_a = {entry[0] for entry in kernel_a.kernel_cache.values()}
        names_b = {entry[0] for entry in kernel_b.kernel_cache.values()}
        assert names_a and names_b
        assert names_a.isdisjoint(names_b), (
            f"Per-instance randomized names must be disjoint; got "
            f"{names_a} and {names_b}"
        )
