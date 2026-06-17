# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pytest
import torch
import torch.nn as nn
from coreai.runtime import NDArray
from coremltools.optimize.torch.quantization import (  # type: ignore [import-untyped]
    PostTrainingQuantizer,
    PostTrainingQuantizerConfig,
)
from torch.export import Dim

from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
from coreai_torch.composite_ops import SDPA, GatherMM, RMSNorm, RMSNormImpl
from coreai_torch.externalize import (
    _derive_composite_io_names,
    _prepare_module,
)

from .utils import (
    compare_outputs,
    filecheck_pattern,
)

DIM = 16


async def _validate_numerics(
    coreai_program,
    model: torch.nn.Module,
    sample: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...] = ("x",),
) -> None:
    """Compile, run in Core AI runtime, and compare against PyTorch output."""
    with TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = coreai_program.save_asset(Path(tmp))
        async with asset.executable() as ai_model:
            rt_func = ai_model.load_function("main")
            inputs = {
                name: NDArray(tensor) for name, tensor in zip(input_names, sample)
            }
            coreai_out = await rt_func(inputs=inputs)
            coreai_numpy = {k: v.numpy() for k, v in coreai_out.items()}
            torch_output = model(*sample)
            output_key = list(coreai_out.keys())[0]
            assert compare_outputs({output_key: torch_output}, coreai_numpy)


# ---------------------------------------------------------------------------
# add_pytorch_module tests — TorchConverter().add_pytorch_module(model, export_fn=..., externalize_modules=...)
# ---------------------------------------------------------------------------


@pytest.mark.ir
def test_externalize_static_shapes_ir() -> None:
    """IR check: add_pytorch_module with static shapes produces noinline graph + invoke."""

    class InnerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(4, 4)
            self.inner = InnerModule()
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pre(x)
            x = self.inner(x)
            x = self.post(x)
            return x

    torch.manual_seed(42)
    model = OuterModel().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[InnerModule],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @inner_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @inner_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_static_shapes() -> None:
    """add_pytorch_module with static shapes produces noinline graph + invoke."""

    class InnerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(4, 4)
            self.inner = InnerModule()
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pre(x)
            x = self.inner(x)
            x = self.post(x)
            return x

    torch.manual_seed(42)
    model = OuterModel().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[InnerModule],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_dynamic_shapes_ir() -> None:
    """IR check: add_pytorch_module with dynamic batch dim produces ? in tensor types."""

    class InnerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(4, 4)
            self.inner = InnerModule()
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pre(x)
            x = self.inner(x)
            x = self.post(x)
            return x

    torch.manual_seed(42)
    model = OuterModel().eval()
    sample = (torch.randn(2, 4),)
    batch = Dim("batch", min=1, max=10)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(
            m, args=sample, dynamic_shapes={"x": {0: batch}}
        ).run_decompositions(get_decomp_table()),
        externalize_modules=[InnerModule],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @inner_{{[0-9a-f]+}}(%{{.*}}: tensor<?x4xf32>
        // CHECK:     coreai.output %{{.*}} : tensor<?x4xf32>
        // CHECK:   }
        // CHECK:   coreai.graph @main(%{{.*}}: tensor<?x4xf32>
        // CHECK:     coreai.invoke @inner_{{[0-9a-f]+}}(
        // CHECK:     coreai.output %{{.*}} : tensor<?x4xf32>
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_dynamic_shapes() -> None:
    """add_pytorch_module with dynamic batch dim produces ? in tensor types."""

    class InnerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(4, 4)
            self.inner = InnerModule()
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pre(x)
            x = self.inner(x)
            x = self.post(x)
            return x

    torch.manual_seed(42)
    model = OuterModel().eval()
    sample = (torch.randn(2, 4),)
    batch = Dim("batch", min=1, max=10)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(
            m, args=sample, dynamic_shapes={"x": {0: batch}}
        ).run_decompositions(get_decomp_table()),
        externalize_modules=[InnerModule],
    )
    coreai_program = converter.to_coreai()

    for batch_size in [1, 2, 5, 10]:
        await _validate_numerics(coreai_program, model, (torch.randn(batch_size, 4),))


@pytest.mark.ir
def test_externalize_multiple_submodules_ir() -> None:
    """IR check: Externalize two submodule classes; both get noinline graphs with invoke."""

    class InnerA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class InnerB(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.fc(x))

    class TwoInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = InnerA()
            self.b = InnerB()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x))

    torch.manual_seed(42)
    model = TwoInnerModel().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[InnerA, InnerB],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @a_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph noinline @b_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @a_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @b_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_multiple_submodules() -> None:
    """Externalize two submodule classes; both get noinline graphs with invoke."""

    class InnerA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class InnerB(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.fc(x))

    class TwoInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = InnerA()
            self.b = InnerB()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x))

    torch.manual_seed(42)
    model = TwoInnerModel().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[InnerA, InnerB],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_no_submodules_ir() -> None:
    """IR check: No externalize_modules produces the same result as the plain add_exported_program path."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(4, 4)
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.post(torch.relu(self.pre(x)))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    program = torch.export.export(model, args=sample)
    program = program.run_decompositions()

    converter = TorchConverter().add_exported_program(program)
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK-NEXT:   coreai.graph @main(
        // CHECK-NOT:    coreai.invoke
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_no_submodules() -> None:
    """No externalize_modules produces the same result as the plain add_exported_program path."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(4, 4)
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.post(torch.relu(self.pre(x)))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    program = torch.export.export(model, args=sample)
    program = program.run_decompositions()

    converter = TorchConverter().add_exported_program(program)
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


def test_wrap_module_invalid_submodule() -> None:
    """ValueError raised when submodule is not found in model."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = Model().eval()

    with pytest.raises(ValueError, match="submodule not found in model"):
        _prepare_module(model, nn.Linear(4, 4))


@pytest.mark.ir
def test_externalize_deep_model_with_rmsnorm_ir() -> None:
    """IR check: Externalize RMSNorm (depth-2 child) from a deeper model hierarchy."""

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class FeedForward(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.up = nn.Linear(dim, hidden)
            self.down = nn.Linear(hidden, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(torch.relu(self.up(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, ff_hidden: int):
            super().__init__()
            self.norm = RMSNorm(dim)
            self.attn_proj = nn.Linear(dim, dim)
            self.ff = FeedForward(dim, ff_hidden)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            h = self.attn_proj(h)
            return x + self.ff(h)

    class MiniTransformer(nn.Module):
        def __init__(self, dim: int = DIM, ff_hidden: int = DIM * 4):
            super().__init__()
            self.embed = nn.Linear(dim, dim)
            self.block = TransformerBlock(dim, ff_hidden)
            self.final_norm = RMSNorm(dim)
            self.head = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            x = self.block(x)
            x = self.final_norm(x)
            return self.head(x)

    torch.manual_seed(42)
    model = MiniTransformer().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[RMSNorm],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @block.norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph noinline @final_norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @block.norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @final_norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_deep_model_with_rmsnorm() -> None:
    """Externalize RMSNorm (depth-2 child) from a deeper model hierarchy."""

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class FeedForward(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.up = nn.Linear(dim, hidden)
            self.down = nn.Linear(hidden, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(torch.relu(self.up(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, ff_hidden: int):
            super().__init__()
            self.norm = RMSNorm(dim)
            self.attn_proj = nn.Linear(dim, dim)
            self.ff = FeedForward(dim, ff_hidden)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            h = self.attn_proj(h)
            return x + self.ff(h)

    class MiniTransformer(nn.Module):
        def __init__(self, dim: int = DIM, ff_hidden: int = DIM * 4):
            super().__init__()
            self.embed = nn.Linear(dim, dim)
            self.block = TransformerBlock(dim, ff_hidden)
            self.final_norm = RMSNorm(dim)
            self.head = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            x = self.block(x)
            x = self.final_norm(x)
            return self.head(x)

    torch.manual_seed(42)
    model = MiniTransformer().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[RMSNorm],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_composite_op_config_ir() -> None:
    """IR check: ExternalizeSpec with composite_op_name + composite_attrs produces composite_decl in IR."""

    class RMSNorm(nn.Module):
        def __init__(
            self,
            axes: int | list[int] = -1,
            eps: float = 1e-5,
            version: int = 1,
        ):
            super().__init__()
            self.axes = axes
            self.eps = eps
            self.version = version

        def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x_f32 = x.to(torch.float32)
            inv_rms = torch.rsqrt(
                (x_f32 * x_f32).mean(self.axes, keepdim=True) + self.eps
            )
            return (x * inv_rms).to(x.dtype) * scale

    class FeedForward(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.up = nn.Linear(dim, hidden)
            self.down = nn.Linear(hidden, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(torch.relu(self.up(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, ff_hidden: int):
            super().__init__()
            self.norm = RMSNorm()
            self.scale = nn.Parameter(torch.ones(dim))
            self.attn_proj = nn.Linear(dim, dim)
            self.ff = FeedForward(dim, ff_hidden)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x, self.scale)
            h = self.attn_proj(h)
            return x + self.ff(h)

    class MiniTransformer(nn.Module):
        def __init__(self, dim: int = DIM, ff_hidden: int = DIM * 4):
            super().__init__()
            self.embed = nn.Linear(dim, dim)
            self.block = TransformerBlock(dim, ff_hidden)
            self.final_norm = RMSNorm()
            self.final_scale = nn.Parameter(torch.ones(dim))
            self.head = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            x = self.block(x)
            x = self.final_norm(x, self.final_scale)
            return self.head(x)

    torch.manual_seed(42)
    model = MiniTransformer().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=RMSNorm,
                composite_op_name="rms_norm",
                composite_attrs=["axes", "eps", "version"],
            )
        ],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph private noinline @block.norm_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm"
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph private noinline @final_norm_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm"
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @block.norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @final_norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_composite_op_config() -> None:
    """ExternalizeSpec with composite_op_name + composite_attrs produces composite_decl in IR."""

    class RMSNorm(nn.Module):
        def __init__(
            self,
            axes: int | list[int] = -1,
            eps: float = 1e-5,
            version: int = 1,
        ):
            super().__init__()
            self.axes = axes
            self.eps = eps
            self.version = version

        def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x_f32 = x.to(torch.float32)
            inv_rms = torch.rsqrt(
                (x_f32 * x_f32).mean(self.axes, keepdim=True) + self.eps
            )
            return (x * inv_rms).to(x.dtype) * scale

    class FeedForward(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.up = nn.Linear(dim, hidden)
            self.down = nn.Linear(hidden, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(torch.relu(self.up(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, ff_hidden: int):
            super().__init__()
            self.norm = RMSNorm()
            self.scale = nn.Parameter(torch.ones(dim))
            self.attn_proj = nn.Linear(dim, dim)
            self.ff = FeedForward(dim, ff_hidden)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x, self.scale)
            h = self.attn_proj(h)
            return x + self.ff(h)

    class MiniTransformer(nn.Module):
        def __init__(self, dim: int = DIM, ff_hidden: int = DIM * 4):
            super().__init__()
            self.embed = nn.Linear(dim, dim)
            self.block = TransformerBlock(dim, ff_hidden)
            self.final_norm = RMSNorm()
            self.final_scale = nn.Parameter(torch.ones(dim))
            self.head = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            x = self.block(x)
            x = self.final_norm(x, self.final_scale)
            return self.head(x)

    torch.manual_seed(42)
    model = MiniTransformer().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=RMSNorm,
                composite_op_name="rms_norm",
                composite_attrs=["axes", "eps", "version"],
            )
        ],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_nested_ir() -> None:
    """IR check: Externalize with multiple types handles nested externalization (parent invokes child)."""

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class FeedForward(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.up = nn.Linear(dim, hidden)
            self.down = nn.Linear(hidden, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(torch.relu(self.up(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, ff_hidden: int):
            super().__init__()
            self.norm = RMSNorm(dim)
            self.attn_proj = nn.Linear(dim, dim)
            self.ff = FeedForward(dim, ff_hidden)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            h = self.attn_proj(h)
            return x + self.ff(h)

    class MiniTransformer(nn.Module):
        def __init__(self, dim: int = DIM, ff_hidden: int = DIM * 4):
            super().__init__()
            self.embed = nn.Linear(dim, dim)
            self.block = TransformerBlock(dim, ff_hidden)
            self.head = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            x = self.block(x)
            return self.head(x)

    torch.manual_seed(42)
    model = MiniTransformer().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[RMSNorm, TransformerBlock],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @block.norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph noinline @block_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @block.norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @block_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_nested() -> None:
    """Externalize with multiple types handles nested externalization (parent invokes child)."""

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class FeedForward(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.up = nn.Linear(dim, hidden)
            self.down = nn.Linear(hidden, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(torch.relu(self.up(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, ff_hidden: int):
            super().__init__()
            self.norm = RMSNorm(dim)
            self.attn_proj = nn.Linear(dim, dim)
            self.ff = FeedForward(dim, ff_hidden)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            h = self.attn_proj(h)
            return x + self.ff(h)

    class MiniTransformer(nn.Module):
        def __init__(self, dim: int = DIM, ff_hidden: int = DIM * 4):
            super().__init__()
            self.embed = nn.Linear(dim, dim)
            self.block = TransformerBlock(dim, ff_hidden)
            self.head = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            x = self.block(x)
            return self.head(x)

    torch.manual_seed(42)
    model = MiniTransformer().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[RMSNorm, TransformerBlock],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_composite_op_different_attrs_per_instance_ir() -> None:
    """IR check: composite_attrs reads per-instance values, so two instances with different eps/axes get distinct composite_decls."""

    class RMSNorm(nn.Module):
        def __init__(
            self,
            axes: int | list[int] = -1,
            eps: float = 1e-5,
            version: int = 1,
        ):
            super().__init__()
            self.axes = axes
            self.eps = eps
            self.version = version

        def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x_f32 = x.to(torch.float32)
            inv_rms = torch.rsqrt(
                (x_f32 * x_f32).mean(self.axes, keepdim=True) + self.eps
            )
            return (x * inv_rms).to(x.dtype) * scale

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(DIM, DIM)
            self.norm1 = RMSNorm(axes=-1, eps=1e-5)
            self.scale1 = nn.Parameter(torch.ones(DIM))
            self.norm2 = RMSNorm(axes=[-2, -1], eps=1e-6)
            self.scale2 = nn.Parameter(torch.ones(8, DIM))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.norm1(self.proj(x), self.scale1)
            return self.norm2(x, self.scale2)

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=RMSNorm,
                composite_op_name="rms_norm",
                composite_attrs=["axes", "eps", "version"],
            )
        ],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph private noinline @norm1_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm"
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph private noinline @norm2_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm"
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @norm1_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @norm2_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_composite_op_different_attrs_per_instance() -> None:
    """composite_attrs reads per-instance values, so two instances with different eps/axes get distinct composite_decls."""

    class RMSNorm(nn.Module):
        def __init__(
            self,
            axes: int | list[int] = -1,
            eps: float = 1e-5,
            version: int = 1,
        ):
            super().__init__()
            self.axes = axes
            self.eps = eps
            self.version = version

        def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x_f32 = x.to(torch.float32)
            inv_rms = torch.rsqrt(
                (x_f32 * x_f32).mean(self.axes, keepdim=True) + self.eps
            )
            return (x * inv_rms).to(x.dtype) * scale

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(DIM, DIM)
            self.norm1 = RMSNorm(axes=-1, eps=1e-5)
            self.scale1 = nn.Parameter(torch.ones(DIM))
            self.norm2 = RMSNorm(axes=[-2, -1], eps=1e-6)
            self.scale2 = nn.Parameter(torch.ones(8, DIM))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.norm1(self.proj(x), self.scale1)
            return self.norm2(x, self.scale2)

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=RMSNorm,
                composite_op_name="rms_norm",
                composite_attrs=["axes", "eps", "version"],
            )
        ],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_sdpa_composite_op_ir() -> None:
    """IR check: SDPA composite op names are auto-derived from the graph signature."""

    from coreai_torch.composite_ops._sdpa import SDPA

    class SDPAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(DIM, DIM * 3)
            self.sdpa = SDPA(scale=None, is_causal=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qkv = self.proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            # Add head dim: (batch, seq, dim) -> (batch, 1, seq, dim)
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            return self.sdpa(q, k, v).squeeze(1)

    torch.manual_seed(42)
    model = SDPAModel().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=SDPA,
                composite_op_name="scaled_dot_product_attention",
                composite_attrs=["scale", "is_causal", "window_size"],
            )
        ],
    )
    coreai_program = converter.to_coreai()

    # Without attn_mask, input_names should be ["query", "key", "value"]
    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph private noinline @sdpa_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention"
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @sdpa_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_sdpa_composite_op() -> None:
    """SDPA composite op names are auto-derived from the graph signature."""

    from coreai_torch.composite_ops._sdpa import SDPA

    class SDPAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(DIM, DIM * 3)
            self.sdpa = SDPA(scale=None, is_causal=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qkv = self.proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            # Add head dim: (batch, seq, dim) -> (batch, 1, seq, dim)
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            return self.sdpa(q, k, v).squeeze(1)

    torch.manual_seed(42)
    model = SDPAModel().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=SDPA,
                composite_op_name="scaled_dot_product_attention",
                composite_attrs=["scale", "is_causal", "window_size"],
            )
        ],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_sdpa_dynamic_input_names_ir() -> None:
    """IR check: Two SDPA instances in one model — one with attn_mask, one without — produce different input_names."""

    from coreai_torch.composite_ops._sdpa import SDPA

    class TwoSDPAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(DIM, DIM * 3)
            self.sdpa_no_mask = SDPA(scale=None, is_causal=True)
            self.sdpa_with_mask = SDPA(scale=None, is_causal=False)

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            qkv = self.proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            # First SDPA: no attn_mask
            out1 = self.sdpa_no_mask(q, k, v)
            # Second SDPA: with attn_mask
            out2 = self.sdpa_with_mask(q, k, v, attn_mask=attn_mask)
            return (out1 + out2).squeeze(1)

    torch.manual_seed(42)
    model = TwoSDPAModel().eval()
    seq_len = 8
    sample = (
        torch.randn(2, seq_len, DIM),
        torch.ones(seq_len, seq_len, dtype=torch.bool).tril(),
    )

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=SDPA,
                composite_op_name="scaled_dot_product_attention",
                composite_attrs=["scale", "is_causal", "window_size"],
            )
        ],
    )
    coreai_program = converter.to_coreai()

    # sdpa_no_mask gets input_names = ["query", "key", "value"]
    # sdpa_with_mask gets input_names = ["query", "key", "value", "attn_mask"]
    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph private noinline @sdpa_no_mask_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention"
        // CHECK-SAME: input_names = ["query", "key", "value"]
        // CHECK-SAME: output_names = ["output"]
        // CHECK:   }
        // CHECK:   coreai.graph private noinline @sdpa_with_mask_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention"
        // CHECK-SAME: input_names = ["query", "key", "value", "attn_mask"]
        // CHECK-SAME: output_names = ["output"]
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @sdpa_no_mask_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @sdpa_with_mask_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_sdpa_dynamic_input_names() -> None:
    """Two SDPA instances in one model — one with attn_mask, one without — produce different input_names."""

    from coreai_torch.composite_ops._sdpa import SDPA

    class TwoSDPAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(DIM, DIM * 3)
            self.sdpa_no_mask = SDPA(scale=None, is_causal=True)
            self.sdpa_with_mask = SDPA(scale=None, is_causal=False)

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            qkv = self.proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            # First SDPA: no attn_mask
            out1 = self.sdpa_no_mask(q, k, v)
            # Second SDPA: with attn_mask
            out2 = self.sdpa_with_mask(q, k, v, attn_mask=attn_mask)
            return (out1 + out2).squeeze(1)

    torch.manual_seed(42)
    model = TwoSDPAModel().eval()
    seq_len = 8
    sample = (
        torch.randn(2, seq_len, DIM),
        torch.ones(seq_len, seq_len, dtype=torch.bool).tril(),
    )

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=SDPA,
                composite_op_name="scaled_dot_product_attention",
                composite_attrs=["scale", "is_causal", "window_size"],
            )
        ],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(
        coreai_program, model, sample, input_names=("x", "attn_mask")
    )


@pytest.mark.ir
def test_externalize_single_sdpa_multi_call_site_ir() -> None:
    """IR check: Single SDPA instance called twice — once without mask, once with — produces two separate graphs."""

    from coreai_torch.composite_ops._sdpa import SDPA

    class SingleSDPAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(DIM, DIM * 3)
            self.sdpa = SDPA(scale=None, is_causal=False)

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            qkv = self.proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            out1 = self.sdpa(q, k, v)
            out2 = self.sdpa(q, k, v, attn_mask=attn_mask)
            return (out1 + out2).squeeze(1)

    torch.manual_seed(42)
    model = SingleSDPAModel().eval()
    seq_len = 8
    sample = (
        torch.randn(2, seq_len, DIM),
        torch.ones(seq_len, seq_len, dtype=torch.bool).tril(),
    )

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=SDPA,
                composite_op_name="scaled_dot_product_attention",
                composite_attrs=["scale", "is_causal", "window_size"],
            )
        ],
    )
    coreai_program = converter.to_coreai()

    # Both graphs get a UUID suffix: sdpa_<uuid> (3-arg) and sdpa_<uuid> (4-arg)
    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph private noinline @sdpa_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention"
        // CHECK-SAME: input_names = ["query", "key", "value"]
        // CHECK-SAME: output_names = ["output"]
        // CHECK:   }
        // CHECK:   coreai.graph private noinline @sdpa_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention"
        // CHECK-SAME: input_names = ["query", "key", "value", "attn_mask"]
        // CHECK-SAME: output_names = ["output"]
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @sdpa_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @sdpa_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_single_sdpa_multi_call_site() -> None:
    """Single SDPA instance called twice — once without mask, once with — produces two separate graphs."""

    from coreai_torch.composite_ops._sdpa import SDPA

    class SingleSDPAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(DIM, DIM * 3)
            self.sdpa = SDPA(scale=None, is_causal=False)

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            qkv = self.proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            out1 = self.sdpa(q, k, v)
            out2 = self.sdpa(q, k, v, attn_mask=attn_mask)
            return (out1 + out2).squeeze(1)

    torch.manual_seed(42)
    model = SingleSDPAModel().eval()
    seq_len = 8
    sample = (
        torch.randn(2, seq_len, DIM),
        torch.ones(seq_len, seq_len, dtype=torch.bool).tril(),
    )

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=SDPA,
                composite_op_name="scaled_dot_product_attention",
                composite_attrs=["scale", "is_causal", "window_size"],
            )
        ],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(
        coreai_program, model, sample, input_names=("x", "attn_mask")
    )


@pytest.mark.ir
def test_externalize_mixed_bare_and_spec_ir() -> None:
    """IR check: Accepts a mix of bare types and ExternalizeSpec in one call."""

    class InnerA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class RMSNorm(nn.Module):
        def __init__(self, eps: float = 1e-5):
            super().__init__()
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = InnerA()
            self.norm = RMSNorm(eps=1e-6)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.a(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            InnerA,
            ExternalizeSpec(
                target_class=RMSNorm,
                composite_op_name="rms_norm",
                composite_attrs=["eps"],
            ),
        ],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @a_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph private noinline @norm_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm"
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @a_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


@pytest.mark.asyncio
async def test_externalize_mixed_bare_and_spec() -> None:
    """Accepts a mix of bare types and ExternalizeSpec in one call.

    InnerA is passed as a bare type (simple externalization).
    RMSNorm is passed as an ExternalizeSpec with composite_op_name set.
    """

    class InnerA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class RMSNorm(nn.Module):
        def __init__(self, eps: float = 1e-5):
            super().__init__()
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = InnerA()
            self.norm = RMSNorm(eps=1e-6)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.a(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            InnerA,
            ExternalizeSpec(
                target_class=RMSNorm,
                composite_op_name="rms_norm",
                composite_attrs=["eps"],
            ),
        ],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


# ---------------------------------------------------------------------------
# add_pytorch_module — Model not mutated
# ---------------------------------------------------------------------------


def test_model_not_mutated_after_convert() -> None:
    """Restore ensures user's model is not mutated after add_pytorch_module usage."""

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.post(self.inner(x))

    torch.manual_seed(42)
    model = Outer().eval()
    sample = (torch.randn(2, 4),)

    # Capture original forward's underlying function
    original_inner_forward_func = model.inner.forward.__func__

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Inner],
    )
    converter.to_coreai()

    # The user's model should not have been patched
    assert model.inner.forward.__func__ is original_inner_forward_func
    assert not hasattr(model.inner, "_externalize_name")
    assert not hasattr(model.inner, "_externalize_op_name")
    assert not hasattr(model.inner, "_original_forward")


# ---------------------------------------------------------------------------
# add_exported_program — simple path still works
# ---------------------------------------------------------------------------


@pytest.mark.ir
def test_add_exported_program_simple_path_ir() -> None:
    """IR check: add_exported_program (ExportedProgram) works identically to before."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    ep = torch.export.export(model, args=sample)
    ep = ep.run_decompositions()

    converter = TorchConverter().add_exported_program(ep)
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK-NEXT:   coreai.graph @main(
        // CHECK-NOT:    coreai.invoke
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_add_exported_program_simple_path() -> None:
    """add_exported_program (ExportedProgram) works identically to before."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    ep = torch.export.export(model, args=sample)
    ep = ep.run_decompositions()

    converter = TorchConverter().add_exported_program(ep)
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


# ---------------------------------------------------------------------------
# add_pytorch_module — error attribution
# ---------------------------------------------------------------------------


def test_add_pytorch_module_export_fn_failure_is_user_error() -> None:
    """If export_fn fails in add_pytorch_module, it's reported as a user error."""

    class BadModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # data-dependent control flow — not exportable
            if x.sum() > 0:
                return x
            return -x

    model = BadModel()

    with pytest.raises(RuntimeError, match="Your model failed to export"):
        TorchConverter().add_pytorch_module(
            model,
            export_fn=lambda m: torch.export.export(
                m, args=(torch.randn(2, 4),)
            ).run_decompositions(get_decomp_table()),
        )


# ---------------------------------------------------------------------------
# add_pytorch_module — without externalize_modules
# ---------------------------------------------------------------------------


@pytest.mark.ir
def test_add_pytorch_module_without_externalize_modules_ir() -> None:
    """IR check: add_pytorch_module without externalize_modules just exports and decomposes."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK-NEXT:   coreai.graph @main(
        // CHECK-NOT:    coreai.invoke
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_add_pytorch_module_without_externalize_modules() -> None:
    """add_pytorch_module without externalize_modules just exports and decomposes."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


# ---------------------------------------------------------------------------
# ExternalizeSpec validation tests
# ---------------------------------------------------------------------------


def test_externalize_spec_composite_only_fields_require_composite_op_name() -> None:
    """Setting composite_attrs without composite_op_name raises ValueError."""

    class Dummy(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    with pytest.raises(ValueError, match="composite_attrs"):
        ExternalizeSpec(target_class=Dummy, composite_attrs=["eps"])


def test_externalize_spec_all_composite_fields_without_op_name_raises() -> None:
    """Setting composite_attrs without composite_op_name raises ValueError listing it."""

    class Dummy(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    with pytest.raises(ValueError) as exc_info:
        ExternalizeSpec(
            target_class=Dummy,
            composite_attrs=["eps"],
        )
    msg = str(exc_info.value)
    assert "composite_attrs" in msg


def test_externalize_spec_valid_without_composite_op_name() -> None:
    """ExternalizeSpec with only target_class is valid."""

    class Dummy(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    spec = ExternalizeSpec(target_class=Dummy)
    assert spec.composite_op_name is None
    assert spec.composite_attrs is None


def test_externalize_spec_valid_full_composite() -> None:
    """ExternalizeSpec with composite_op_name and all composite fields is valid."""

    class Dummy(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    spec = ExternalizeSpec(
        target_class=Dummy,
        composite_op_name="my_op",
        composite_attrs=["eps"],
    )
    assert spec.composite_op_name == "my_op"


# ---------------------------------------------------------------------------
# Restore tests — model state after to_coreai
# ---------------------------------------------------------------------------


DIM_RESTORE = 16


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class _MiniTransformer(nn.Module):
    def __init__(self, dim: int = DIM_RESTORE):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.norm = _RMSNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.embed(x)))


def test_model_restored_after_convert() -> None:
    """After to_coreai, all submodule forwards should be restored."""
    torch.manual_seed(42)
    model = _MiniTransformer().eval()
    sample = (torch.randn(2, 8, DIM_RESTORE),)

    # Save the underlying __func__ of the norm's forward (the externalized module)
    original_norm_forward_func = model.norm.forward.__func__

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[_RMSNorm],
    )
    converter.to_coreai()

    # Verify the externalized module's forward is restored
    assert model.norm.forward.__func__ is original_norm_forward_func

    # Verify no externalization markers remain on any module
    for name, mod in model.named_modules():
        assert not hasattr(mod, "_original_forward")
        assert not hasattr(mod, "_externalize_name")
        assert not hasattr(mod, "_externalize_op_name")
        assert not hasattr(mod, "_externalize_config")

    # Verify model still produces correct output (not broken)
    with torch.no_grad():
        out = model(*sample)
    assert out.shape == (2, 8, DIM_RESTORE)


def test_model_mutated_during_pipeline() -> None:
    """Verify the model IS actually patched during the externalize pipeline."""
    torch.manual_seed(42)
    model = _MiniTransformer().eval()
    sample = (torch.randn(2, 8, DIM_RESTORE),)

    original_norm_forward_func = model.norm.forward.__func__
    was_patched = False

    # Wrap export_fn to check mutation at the point of re-export
    def checking_export_fn(m):
        nonlocal was_patched
        # During re-export (2nd call), the norm should have _externalize_name marker
        if hasattr(m, "norm") and hasattr(m.norm, "_externalize_name"):
            was_patched = True
        return torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        )

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=checking_export_fn,
        externalize_modules=[_RMSNorm],
    )
    converter.to_coreai()

    # The pipeline should have patched the forward during processing
    assert was_patched, "Model was never mutated during externalize pipeline"
    # But it should be restored now
    assert model.norm.forward.__func__ is original_norm_forward_func
    assert not hasattr(model.norm, "_externalize_name")


def test_model_restored_on_error() -> None:
    """If to_coreai fails, the model should still be restored."""
    torch.manual_seed(42)
    model = _MiniTransformer().eval()
    sample = (torch.randn(2, 8, DIM_RESTORE),)

    original_norm_forward_func = model.norm.forward.__func__

    # Use a bad export_fn that fails on re-export (after marking)
    call_count = 0

    def flaky_export_fn(m):
        nonlocal call_count
        call_count += 1
        if call_count > 1:  # first call (eager validation) succeeds, re-export fails
            raise RuntimeError("simulated export failure")
        return torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        )

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=flaky_export_fn,
        externalize_modules=[_RMSNorm],
    )

    with pytest.raises(RuntimeError, match="simulated export failure"):
        converter.to_coreai()

    # Model should still be restored despite the error
    assert model.norm.forward.__func__ is original_norm_forward_func
    for name, mod in model.named_modules():
        assert not hasattr(mod, "_original_forward")
        assert not hasattr(mod, "_externalize_name")


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


@pytest.mark.ir
def test_externalize_shared_instance_same_signature_ir() -> None:
    """IR check: Single submodule instance called twice with the same signature in one forward."""

    class Norm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.norm = Norm(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Same instance, same arg count, called twice
            x = self.norm(self.proj(x))
            x = self.norm(x)
            return x

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Norm],
    )
    coreai_program = converter.to_coreai()

    # Two separate noinline graphs (one per call site), each with a unique UUID
    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph noinline @norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_shared_instance_same_signature() -> None:
    """Single submodule instance called twice with the same signature in one forward.

    Each call site gets its own noinline graph (distinct UUID) so the runtime
    does not deduplicate invocations.
    """

    class Norm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.norm = Norm(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Same instance, same arg count, called twice
            x = self.norm(self.proj(x))
            x = self.norm(x)
            return x

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Norm],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


async def test_externalize_no_matching_submodules_warns() -> None:
    """externalize_modules lists a class that the model does not contain.

    This should emit a UserWarning naming the unmatched class and still
    convert successfully — supports superset spec lists across model variants.
    """

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class Unrelated(nn.Module):
        """This class is *not* in the model."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.post(self.inner(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    with pytest.warns(UserWarning, match="Unrelated"):
        converter = TorchConverter().add_pytorch_module(
            model,
            export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
                get_decomp_table()
            ),
            externalize_modules=[Unrelated],
        )
        coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


async def test_externalize_partial_match_warns() -> None:
    """Superset spec list: one class matches, another does not.

    The matched spec is still externalized; the unmatched spec produces a
    UserWarning but does not fail conversion.
    """

    class Matched(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class Missing(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Matched()
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.post(self.block(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    with pytest.warns(UserWarning, match="Missing"):
        converter = TorchConverter().add_pytorch_module(
            model,
            export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
                get_decomp_table()
            ),
            externalize_modules=[Matched, Missing],
        )
        coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_three_level_nesting_ir() -> None:
    """IR check: Externalize at three levels of depth: grandchild, child, and each gets its own graph."""

    class Norm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class FeedForward(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm = Norm(dim)
            self.up = nn.Linear(dim, dim * 2)
            self.down = nn.Linear(dim * 2, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(torch.relu(self.up(self.norm(x))))

    class Block(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.ff = FeedForward(dim)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.proj(self.ff(x))

    class Model(nn.Module):
        def __init__(self, dim: int = DIM):
            super().__init__()
            self.embed = nn.Linear(dim, dim)
            self.block = Block(dim)
            self.head = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.block(self.embed(x)))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Norm, FeedForward],
    )
    coreai_program = converter.to_coreai()

    # Norm is deepest (noinline first), then FeedForward (invokes Norm), then main (invokes FF)
    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @block.ff.norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph noinline @block.ff_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @block.ff.norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @block.ff_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_three_level_nesting() -> None:
    """Externalize at three levels of depth: grandchild, child, and each gets its own graph.

    L3 (Norm) sits inside L2 (FeedForward) inside L1 (Block) inside root.
    Externalize both Norm and FeedForward. FeedForward's graph should invoke
    Norm's graph, and main should invoke FeedForward's graph.
    """

    class Norm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class FeedForward(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm = Norm(dim)
            self.up = nn.Linear(dim, dim * 2)
            self.down = nn.Linear(dim * 2, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down(torch.relu(self.up(self.norm(x))))

    class Block(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.ff = FeedForward(dim)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.proj(self.ff(x))

    class Model(nn.Module):
        def __init__(self, dim: int = DIM):
            super().__init__()
            self.embed = nn.Linear(dim, dim)
            self.block = Block(dim)
            self.head = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.block(self.embed(x)))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 8, DIM),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Norm, FeedForward],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_identity_passthrough_ir() -> None:
    """IR check: Externalize a submodule that is an identity (returns input unchanged)."""

    class Identity(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.identity = Identity()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.identity(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Identity],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @identity_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @identity_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_identity_passthrough() -> None:
    """Externalize a submodule that is an identity (returns input unchanged).

    The noinline graph should still be emitted and numerics should match.
    """

    class Identity(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.identity = Identity()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.identity(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Identity],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_composite_op_empty_attrs_ir() -> None:
    """IR check: ExternalizeSpec with composite_op_name but empty composite_attrs list."""

    class Scale(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.weight

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.scale = Scale(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.scale(self.proj(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=Scale,
                composite_op_name="element_scale",
                composite_attrs=[],
            ),
        ],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph private noinline @scale_{{[0-9a-f]+}}(
        // CHECK-SAME: composite_decl = #coreai.composite_declaration<"element_scale"
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @scale_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_composite_op_empty_attrs() -> None:
    """ExternalizeSpec with composite_op_name but empty composite_attrs list.

    Should produce a composite_decl with the op name but no attribute payload.
    """

    class Scale(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.weight

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.scale = Scale(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.scale(self.proj(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=Scale,
                composite_op_name="element_scale",
                composite_attrs=[],
            ),
        ],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_multiple_inputs_ir() -> None:
    """IR check: Externalize a module whose forward takes multiple tensor inputs."""

    class Bilinear(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(dim, dim))

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b @ self.weight

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bilinear = Bilinear(4)
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.proj(self.bilinear(x, y))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4), torch.randn(2, 4))

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Bilinear],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @bilinear_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @bilinear_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_multiple_inputs() -> None:
    """Externalize a module whose forward takes multiple tensor inputs."""

    class Bilinear(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(dim, dim))

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b @ self.weight

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bilinear = Bilinear(4)
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.proj(self.bilinear(x, y))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4), torch.randn(2, 4))

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Bilinear],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample, input_names=("x", "y"))


@pytest.mark.ir
def test_externalize_dynamic_shapes_multiple_inputs_ir() -> None:
    """IR check: Dynamic batch dim with multiple model inputs, both sharing the same dynamic dim."""

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.inner(x) + y

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4), torch.randn(2, 4))

    batch = Dim("batch", min=1, max=10)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(
            m, args=sample, dynamic_shapes={"x": {0: batch}, "y": {0: batch}}
        ).run_decompositions(get_decomp_table()),
        externalize_modules=[Inner],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @inner_{{[0-9a-f]+}}(%{{.*}}: tensor<?x4xf32>
        // CHECK:     coreai.output %{{.*}} : tensor<?x4xf32>
        // CHECK:   }
        // CHECK:   coreai.graph @main(%{{.*}}: tensor<?x4xf32>
        // CHECK:     coreai.invoke @inner_{{[0-9a-f]+}}(
        // CHECK:     coreai.output %{{.*}} : tensor<?x4xf32>
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_dynamic_shapes_multiple_inputs() -> None:
    """Dynamic batch dim with multiple model inputs, both sharing the same dynamic dim."""

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.inner(x) + y

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4), torch.randn(2, 4))

    batch = Dim("batch", min=1, max=10)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(
            m, args=sample, dynamic_shapes={"x": {0: batch}, "y": {0: batch}}
        ).run_decompositions(get_decomp_table()),
        externalize_modules=[Inner],
    )
    coreai_program = converter.to_coreai()

    for batch_size in [1, 3, 7, 10]:
        await _validate_numerics(
            coreai_program,
            model,
            (torch.randn(batch_size, 4), torch.randn(batch_size, 4)),
            input_names=("x", "y"),
        )


@pytest.mark.ir
def test_externalize_all_submodules_ir() -> None:
    """IR check: Externalize every submodule class so main graph is only invokes."""

    class LayerA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class LayerB(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.fc(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = LayerA()
            self.b = LayerB()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[LayerA, LayerB],
    )
    coreai_program = converter.to_coreai()

    # Main graph body should only contain invokes — no aten ops
    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @a_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph noinline @b_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @a_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @b_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_all_submodules() -> None:
    """Externalize every submodule class so main graph is only invokes."""

    class LayerA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class LayerB(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.fc(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = LayerA()
            self.b = LayerB()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[LayerA, LayerB],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_many_instances_of_same_class_ir() -> None:
    """IR check: Four instances of the same class — all get separate noinline graphs."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.b0 = Block()
            self.b1 = Block()
            self.b2 = Block()
            self.b3 = Block()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b3(self.b2(self.b1(self.b0(x))))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Block],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @b0_{{[0-9a-f]+}}(
        // CHECK:   coreai.graph noinline @b1_{{[0-9a-f]+}}(
        // CHECK:   coreai.graph noinline @b2_{{[0-9a-f]+}}(
        // CHECK:   coreai.graph noinline @b3_{{[0-9a-f]+}}(
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @b0_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @b1_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @b2_{{[0-9a-f]+}}(
        // CHECK:     coreai.invoke @b3_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_many_instances_of_same_class() -> None:
    """Four instances of the same class — all get separate noinline graphs."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.b0 = Block()
            self.b1 = Block()
            self.b2 = Block()
            self.b3 = Block()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b3(self.b2(self.b1(self.b0(x))))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[Block],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
def test_externalize_subclass_matches_parent_spec_ir() -> None:
    """IR check: ExternalizeSpec with a base class matches subclass instances via isinstance."""

    class BaseNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class CustomNorm(BaseNorm):
        """Subclass that inherits forward from BaseNorm."""

        pass

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.norm = CustomNorm(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.proj(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    # Target BaseNorm — should match CustomNorm via isinstance
    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[BaseNorm],
    )
    coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK:   coreai.graph noinline @norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK:   coreai.graph @main(
        // CHECK:     coreai.invoke @norm_{{[0-9a-f]+}}(
        // CHECK:     coreai.output
        // CHECK:   }
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_subclass_matches_parent_spec() -> None:
    """ExternalizeSpec with a base class matches subclass instances via isinstance."""

    class BaseNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class CustomNorm(BaseNorm):
        """Subclass that inherits forward from BaseNorm."""

        pass

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.norm = CustomNorm(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.proj(x))

    torch.manual_seed(42)
    model = Model().eval()
    sample = (torch.randn(2, 4),)

    # Target BaseNorm — should match CustomNorm via isinstance
    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[BaseNorm],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


# ---------------------------------------------------------------------------
# _derive_composite_io_names tests
# ---------------------------------------------------------------------------


def test_derive_composite_io_names_basic() -> None:
    """Basic user inputs produce correct input names; single output gives 'output'."""

    class M(nn.Module):
        def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
            return query + key

    ep = torch.export.export(M(), args=(torch.randn(2, 4), torch.randn(2, 4)))
    input_names, output_names = _derive_composite_io_names(ep)
    assert input_names == ["query", "key"]
    assert output_names == ["output"]


def test_derive_composite_io_names_with_params() -> None:
    """Parameters appear as input names using their attribute target."""

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input * self.weight

    ep = torch.export.export(M(), args=(torch.randn(2, 4),))
    input_names, output_names = _derive_composite_io_names(ep)
    assert input_names == ["weight", "input"]
    assert output_names == ["output"]


def test_derive_composite_io_names_multi_output() -> None:
    """Multiple outputs produce 'output_0', 'output_1', etc."""

    class M(nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x, x * 2

    ep = torch.export.export(M(), args=(torch.randn(2, 4),))
    input_names, output_names = _derive_composite_io_names(ep)
    assert input_names == ["x"]
    assert output_names == ["output_0", "output_1"]


def test_derive_composite_io_names_optional_excluded() -> None:
    """Optional params that are None at export time don't appear in input names."""

    class M(nn.Module):
        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if mask is not None:
                return query + key + mask
            return query + key

    # Export without mask
    ep = torch.export.export(M(), args=(torch.randn(2, 4), torch.randn(2, 4)))
    input_names, output_names = _derive_composite_io_names(ep)
    assert input_names == ["query", "key"]
    assert "mask" not in input_names

    # Export with mask
    ep_with_mask = torch.export.export(
        M(), args=(torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4))
    )
    input_names2, _ = _derive_composite_io_names(ep_with_mask)
    assert input_names2 == ["query", "key", "mask"]


def test_derive_composite_io_names_middle_optional_skipped() -> None:
    """Skipping a middle optional while passing a later one produces correct ordered names."""

    class M(nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            a: torch.Tensor | None = None,
            b: torch.Tensor | None = None,
            c: torch.Tensor | None = None,
        ) -> torch.Tensor:
            out = x
            if a is not None:
                out = out + a
            if b is not None:
                out = out + b
            if c is not None:
                out = out + c
            return out

    # Skip a and b, pass only x and c
    ep = torch.export.export(
        M(), args=(torch.randn(2, 4),), kwargs={"c": torch.randn(2, 4)}
    )
    input_names, output_names = _derive_composite_io_names(ep)
    assert input_names == ["x", "c"]
    assert output_names == ["output"]

    # Skip b only, pass x, a, and c
    ep2 = torch.export.export(
        M(),
        args=(torch.randn(2, 4), torch.randn(2, 4)),
        kwargs={"c": torch.randn(2, 4)},
    )
    input_names2, _ = _derive_composite_io_names(ep2)
    assert input_names2 == ["x", "a", "c"]

    # Pass all
    ep3 = torch.export.export(
        M(),
        args=(
            torch.randn(2, 4),
            torch.randn(2, 4),
            torch.randn(2, 4),
            torch.randn(2, 4),
        ),
    )
    input_names3, _ = _derive_composite_io_names(ep3)
    assert input_names3 == ["x", "a", "b", "c"]


# ---------------------------------------------------------------------------
# Externalize + compression tests
# ---------------------------------------------------------------------------


@pytest.mark.ir
@pytest.mark.flaky(reruns=3)
@pytest.mark.skip(
    reason=(
        "transform_with_custom_compression_ops has been deprecated. Consider removing "
        "these tests or use an alternative way to generate quantized weights"
    )
)
def test_externalize_rms_norm_with_quantized_linears_ir() -> None:
    """IR check: Quantized (int4) weights retain si4 dtype when externalization re-exports the model."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(24, 32)
            self.norm = RMSNorm(32)
            self.head = nn.Linear(32, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.norm(self.proj(x)))

    torch.manual_seed(42)
    model = Model().eval()

    # Quantize linear weights to int4 per-block
    quantization_config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": "symmetric",
                "granularity": "per_block",
                "weight_dtype": "int4",
                "block_size": 4,
            },
        }
    )
    quantizer = PostTrainingQuantizer(model, quantization_config)
    model = cast("nn.Module", quantizer.compress())
    transform_with_custom_compression_ops(model)  # noqa: F821

    sample = (torch.randn(2, 24),)

    externalize_spec = ExternalizeSpec(
        target_class=RMSNormImpl,
        composite_op_name="rms_norm",
        composite_attrs=["axes", "eps"],
    )
    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[externalize_spec],
    )
    coreai_program = converter.to_coreai()

    # Verify si4 quantized weights survived the re-export
    ir = str(coreai_program)
    assert "si4" in ir, (
        "Expected si4 quantized weight constants in the IR but found none. "
        "The externalize re-export likely discarded the sub-byte injection."
    )
    assert "si8" not in ir, (
        "Found si8 constants in the IR — quantized weights were not packed to si4."
    )

    # Verify RMSNorm is externalized as a composite op with correct declaration
    pattern = """
    // CHECK: coreai.graph private noinline @norm.rmsnorm_impl_[[S:[a-f0-9]+]](
    // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm" =
    // CHECK-SAME: input_names = ["input", "scale"]
    // CHECK-SAME: op_attrs =
    // CHECK-SAME: axes = -1 : si64
    // CHECK-SAME: eps = 9.99999974E-6 : f32
    // CHECK-SAME: version = 1 : si64
    // CHECK-SAME: output_names = ["output"]
    // CHECK: coreai.invoke @norm.rmsnorm_impl_[[S]](%{{[0-9]+}}, %{{[0-9]+}})
    """
    filecheck_pattern(ir, check_file=pattern)


@pytest.mark.flaky(reruns=3)
@pytest.mark.skip(
    reason=(
        "transform_with_custom_compression_ops has been deprecated. Consider removing "
        "these tests or use an alternative way to generate quantized weights"
    )
)
async def test_externalize_rms_norm_with_quantized_linears() -> None:
    """Quantized (int4) weights retain si4 dtype when externalization re-exports the model.

    Regression test: the externalize pipeline re-exports the model, which used to
    discard the sub-byte injection (int8 -> int4 packing) applied by
    inject_subbyte_tensors.  Without the fix, weight constants appear as si8
    instead of si4 in the emitted MLIR.

    We externalize RMSNorm (which has no quantized weights) while the Linear
    layers carry int4 quantized weights.  This mirrors real LLM export where
    composite ops (RMSNorm, RoPE, SDPA) are externalized alongside quantized
    linear projections.
    """

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(24, 32)
            self.norm = RMSNorm(32)
            self.head = nn.Linear(32, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.norm(self.proj(x)))

    torch.manual_seed(42)
    model = Model().eval()

    # Quantize linear weights to int4 per-block
    quantization_config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": "symmetric",
                "granularity": "per_block",
                "weight_dtype": "int4",
                "block_size": 4,
            },
        }
    )
    quantizer = PostTrainingQuantizer(model, quantization_config)
    model = cast("nn.Module", quantizer.compress())
    transform_with_custom_compression_ops(model)  # noqa: F821

    sample = (torch.randn(2, 24),)

    externalize_spec = ExternalizeSpec(
        target_class=RMSNormImpl,
        composite_op_name="rms_norm",
        composite_attrs=["axes", "eps"],
    )
    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[externalize_spec],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
@pytest.mark.flaky(reruns=3)
@pytest.mark.skip(
    reason=(
        "transform_with_custom_compression_ops has been deprecated. Consider removing "
        "these tests or use an alternative way to generate quantized weights"
    )
)
def test_externalize_gather_mm_with_quantized_rhs_ir() -> None:
    """IR check: Quantized expert weight flows as rhs into an externalized GatherMM composite."""
    num_experts = 4
    in_dim = 24
    out_dim = 32

    class MoEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_dim, in_dim)
            self.experts = nn.Linear(in_dim, num_experts * out_dim, bias=False)
            self.gather_mm = GatherMM(num_batch_axes=0)
            self.head = nn.Linear(out_dim, out_dim)

        def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            x = self.proj(x)
            rhs = self.experts.weight.view(num_experts, out_dim, in_dim)
            rhs = rhs.transpose(-1, -2)
            out = self.gather_mm(x, rhs, rhs_indices=indices)
            out = out.squeeze(-2)
            return self.head(out)

    torch.manual_seed(42)
    model = MoEBlock().eval()

    quantization_config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": "symmetric",
                "granularity": "per_block",
                "weight_dtype": "int4",
                "block_size": 4,
            },
        }
    )
    quantizer = PostTrainingQuantizer(model, quantization_config)
    model = cast("nn.Module", quantizer.compress())
    transform_with_custom_compression_ops(model)  # noqa: F821

    x = torch.randn(2, 1, 1, in_dim)
    indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int16)
    sample = (x, indices)

    externalize_spec = ExternalizeSpec(
        target_class=GatherMM,
        composite_op_name="gather_mm",
        composite_attrs=["num_batch_axes"],
    )
    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[externalize_spec],
    )
    coreai_program = converter.to_coreai()

    ir = str(coreai_program)
    assert "si4" in ir, (
        "Expected si4 quantized weight constants in the IR but found none."
    )
    assert "si8" not in ir, (
        "Found si8 constants in the IR — quantized weights were not packed to si4."
    )

    # TODO: Remove the reshape once we can simply use torch.nn.Parameter
    #       for expert weight, see TODO in module definition
    pattern = f"""
    // CHECK: coreai.graph private noinline @gather_mm_[[S:[a-f0-9]+]](
    // CHECK-SAME: composite_decl = #coreai.composite_declaration<"gather_mm" =
    // CHECK-SAME: num_batch_axes = 0 : si64
    // CHECK-SAME: version = 1 : si64
    // CHECK-SAME: output_names = ["output"]
    // CHECK: coreai.graph @main
    // CHECK: [[EW:%[0-9]+]] = coreai.blockwise_shift_scale {{{{.*}}}}si4>{{{{.*}}}} -> tensor<{num_experts * out_dim}x{in_dim}xf32>
    // CHECK: [[ER:%[0-9]+]] = coreai.reshape [[EW]]{{{{.*}}}} -> tensor<{num_experts}x{out_dim}x{in_dim}xf32>
    // CHECK: [[ET:%[0-9]+]] = coreai.transpose [[ER]]{{{{.*}}}} -> tensor<{num_experts}x{in_dim}x{out_dim}xf32>
    // CHECK: coreai.invoke @gather_mm_[[S]]({{{{.*}}}}, [[ET]],
    """
    filecheck_pattern(ir, check_file=pattern)


@pytest.mark.flaky(reruns=3)
@pytest.mark.skip(
    reason=(
        "transform_with_custom_compression_ops has been deprecated. Consider removing "
        "these tests or use an alternative way to generate quantized weights"
    )
)
async def test_externalize_gather_mm_with_quantized_rhs() -> None:
    """Quantized expert weight flows as rhs into an externalized GatherMM composite.

    In MoE models (SwitchLinear), expert weights get quantized to int4.
    The transposed weight is passed as ``rhs`` to GatherMM.  This test
    verifies the full data-flow chain in the emitted IR:
        si4 constant → blockwise_shift_scale → transpose → invoke @gather_mm
    """
    num_experts = 4
    in_dim = 24
    out_dim = 32

    class MoEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_dim, in_dim)
            # Expert weights stored via nn.Linear so PostTrainingQuantizer
            # quantizes them.  Weight shape: (num_experts * out_dim, in_dim).
            # TODO: Simply use torch.nn.Parameter once we have a way to
            #       quantize torch.nn.Parameter directly
            self.experts = nn.Linear(in_dim, num_experts * out_dim, bias=False)
            self.gather_mm = GatherMM(num_batch_axes=0)
            self.head = nn.Linear(out_dim, out_dim)

        def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            x = self.proj(x)
            # Reshape Linear weight → (num_experts, out_dim, in_dim)
            # then transpose → (num_experts, in_dim, out_dim) for GatherMM rhs.
            rhs = self.experts.weight.view(num_experts, out_dim, in_dim)
            rhs = rhs.transpose(-1, -2)
            # x: (batch, 1, 1, in_dim), indices: (batch, k)
            out = self.gather_mm(x, rhs, rhs_indices=indices)
            # (batch, k, 1, out_dim) → (batch, k, out_dim)
            out = out.squeeze(-2)
            return self.head(out)

    torch.manual_seed(42)
    model = MoEBlock().eval()

    quantization_config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": "symmetric",
                "granularity": "per_block",
                "weight_dtype": "int4",
                "block_size": 4,
            },
        }
    )
    quantizer = PostTrainingQuantizer(model, quantization_config)
    model = cast("nn.Module", quantizer.compress())
    transform_with_custom_compression_ops(model)  # noqa: F821

    x = torch.randn(2, 1, 1, in_dim)
    indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int16)
    sample = (x, indices)

    externalize_spec = ExternalizeSpec(
        target_class=GatherMM,
        composite_op_name="gather_mm",
        composite_attrs=["num_batch_axes"],
    )
    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[externalize_spec],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(
        coreai_program, model, sample, input_names=("x", "indices")
    )


@pytest.mark.ir
@pytest.mark.flaky(reruns=3)
@pytest.mark.skip(
    reason=(
        "transform_with_custom_compression_ops has been deprecated. Consider removing "
        "these tests or use an alternative way to generate quantized weights"
    )
)
def test_externalize_multiple_composites_with_quantized_weights_ir() -> None:
    """IR check: Multiple composite ops (RMSNorm + SDPA) externalized with quantized linears."""
    head_dim = 16
    n_heads = 2
    embed_dim = n_heads * head_dim

    class MiniAttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RMSNorm(embed_dim)
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.sdpa = SDPA(scale=1.0 / math.sqrt(head_dim), is_causal=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, _ = x.shape
            h = self.norm(x)
            q = self.q_proj(h).view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(h).view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(h).view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            attn = self.sdpa(q, k, v)
            attn = attn.transpose(1, 2).reshape(batch, seq_len, embed_dim)
            return x + self.o_proj(attn)

    torch.manual_seed(42)
    model = MiniAttentionBlock().eval()

    quantization_config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": "symmetric",
                "granularity": "per_block",
                "weight_dtype": "int4",
                "block_size": 4,
            },
        }
    )
    quantizer = PostTrainingQuantizer(model, quantization_config)
    model = cast("nn.Module", quantizer.compress())
    transform_with_custom_compression_ops(model)  # noqa: F821

    sample = (torch.randn(1, 4, embed_dim),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=RMSNormImpl,
                composite_op_name="rms_norm",
                composite_attrs=["axes", "eps"],
            ),
            ExternalizeSpec(
                target_class=SDPA,
                composite_op_name="scaled_dot_product_attention",
                composite_attrs=["scale", "is_causal", "window_size"],
            ),
        ],
    )
    coreai_program = converter.to_coreai()

    ir = str(coreai_program)
    assert "si4" in ir, "Expected si4 quantized weight constants."
    assert "si8" not in ir, "Quantized weights were not packed to si4."

    # Both composite declarations must be present
    pattern = """
    // CHECK: coreai.graph private noinline @norm.rmsnorm_impl_[[RMS:[a-f0-9]+]](
    // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm" =
    // CHECK-SAME: input_names = ["input", "scale"]
    // CHECK: coreai.graph private noinline @sdpa_[[SDPA:[a-f0-9]+]](
    // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention" =
    // CHECK-SAME: input_names = ["query", "key", "value"]
    // CHECK: coreai.invoke @norm.rmsnorm_impl_[[RMS]](
    // CHECK: coreai.invoke @sdpa_[[SDPA]](
    """
    filecheck_pattern(ir, check_file=pattern)


@pytest.mark.flaky(reruns=3)
@pytest.mark.skip(
    reason=(
        "transform_with_custom_compression_ops has been deprecated. Consider removing "
        "these tests or use an alternative way to generate quantized weights"
    )
)
async def test_externalize_multiple_composites_with_quantized_weights() -> None:
    """Multiple composite ops (RMSNorm + SDPA) externalized with quantized linears.

    Real LLM export externalizes several composite ops simultaneously.
    This test verifies that multiple ExternalizeSpec entries produce
    independent composite declarations, and that quantized weights in
    surrounding Linear layers survive the shared re-export.
    """
    head_dim = 16
    n_heads = 2
    embed_dim = n_heads * head_dim

    class MiniAttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RMSNorm(embed_dim)
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.sdpa = SDPA(scale=1.0 / math.sqrt(head_dim), is_causal=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, _ = x.shape
            h = self.norm(x)
            q = self.q_proj(h).view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            k = self.k_proj(h).view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            v = self.v_proj(h).view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            attn = self.sdpa(q, k, v)
            attn = attn.transpose(1, 2).reshape(batch, seq_len, embed_dim)
            return x + self.o_proj(attn)

    torch.manual_seed(42)
    model = MiniAttentionBlock().eval()

    quantization_config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": "symmetric",
                "granularity": "per_block",
                "weight_dtype": "int4",
                "block_size": 4,
            },
        }
    )
    quantizer = PostTrainingQuantizer(model, quantization_config)
    model = cast("nn.Module", quantizer.compress())
    transform_with_custom_compression_ops(model)  # noqa: F821

    sample = (torch.randn(1, 4, embed_dim),)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=RMSNormImpl,
                composite_op_name="rms_norm",
                composite_attrs=["axes", "eps"],
            ),
            ExternalizeSpec(
                target_class=SDPA,
                composite_op_name="scaled_dot_product_attention",
                composite_attrs=["scale", "is_causal", "window_size"],
            ),
        ],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)


@pytest.mark.ir
@pytest.mark.skip(
    reason=(
        "transform_with_custom_compression_ops has been deprecated. Consider removing "
        "these tests or use an alternative way to generate quantized weights"
    )
)
def test_externalize_gather_mm_combined_with_rms_norm_ir() -> None:
    """IR check: GatherMM + RMSNorm both externalized alongside quantized weights."""
    num_experts = 4
    in_dim = 24
    out_dim = 32

    class MoEWithNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RMSNorm(in_dim)
            self.expert_weight = nn.Parameter(torch.randn(num_experts, out_dim, in_dim))
            self.gather_mm = GatherMM(num_batch_axes=0)
            self.head = nn.Linear(out_dim, out_dim)

        def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            h = h.unsqueeze(-2).unsqueeze(-2)  # (..., 1, 1, in_dim)
            rhs = self.expert_weight.transpose(-1, -2)
            out = self.gather_mm(h, rhs, rhs_indices=indices)
            out = out.squeeze(-2)  # (..., k, out_dim)
            return self.head(out)

    torch.manual_seed(42)
    model = MoEWithNorm().eval()

    quantization_config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": "symmetric",
                "granularity": "per_block",
                "weight_dtype": "int4",
                "block_size": 4,
            },
        }
    )
    quantizer = PostTrainingQuantizer(model, quantization_config)
    model = cast("nn.Module", quantizer.compress())
    transform_with_custom_compression_ops(model)  # noqa: F821

    x = torch.randn(2, in_dim)
    indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int16)
    sample = (x, indices)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=RMSNormImpl,
                composite_op_name="rms_norm",
                composite_attrs=["axes", "eps"],
            ),
            ExternalizeSpec(
                target_class=GatherMM,
                composite_op_name="gather_mm",
                composite_attrs=["num_batch_axes"],
            ),
        ],
    )
    coreai_program = converter.to_coreai()

    ir = str(coreai_program)
    assert "si4" in ir, "Expected si4 quantized weight constants."
    assert "si8" not in ir, "Quantized weights were not packed to si4."

    # Both rms_norm and gather_mm composite declarations must be present
    pattern = """
    // CHECK-DAG: composite_decl = #coreai.composite_declaration<"rms_norm"
    // CHECK-DAG: composite_decl = #coreai.composite_declaration<"gather_mm"
    """
    filecheck_pattern(ir, check_file=pattern)


@pytest.mark.skip(
    reason=(
        "transform_with_custom_compression_ops has been deprecated. Consider removing "
        "these tests or use an alternative way to generate quantized weights"
    )
)
async def test_externalize_gather_mm_combined_with_rms_norm() -> None:
    """GatherMM + RMSNorm both externalized alongside quantized weights.

    Mirrors real MoE transformer blocks (Mixtral, Qwen3-MoE) where
    RMSNorm normalizes the input, then GatherMM dispatches to quantized
    expert weights.  Tests mixed quantized/non-quantized parameters
    across multiple composite ops.
    """
    num_experts = 4
    in_dim = 24
    out_dim = 32

    class MoEWithNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RMSNorm(in_dim)
            self.expert_weight = nn.Parameter(torch.randn(num_experts, out_dim, in_dim))
            self.gather_mm = GatherMM(num_batch_axes=0)
            self.head = nn.Linear(out_dim, out_dim)

        def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            h = h.unsqueeze(-2).unsqueeze(-2)  # (..., 1, 1, in_dim)
            rhs = self.expert_weight.transpose(-1, -2)
            out = self.gather_mm(h, rhs, rhs_indices=indices)
            out = out.squeeze(-2)  # (..., k, out_dim)
            return self.head(out)

    torch.manual_seed(42)
    model = MoEWithNorm().eval()

    quantization_config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": "symmetric",
                "granularity": "per_block",
                "weight_dtype": "int4",
                "block_size": 4,
            },
        }
    )
    quantizer = PostTrainingQuantizer(model, quantization_config)
    model = cast("nn.Module", quantizer.compress())
    transform_with_custom_compression_ops(model)  # noqa: F821

    x = torch.randn(2, in_dim)
    indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int16)
    sample = (x, indices)

    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=RMSNormImpl,
                composite_op_name="rms_norm",
                composite_attrs=["axes", "eps"],
            ),
            ExternalizeSpec(
                target_class=GatherMM,
                composite_op_name="gather_mm",
                composite_attrs=["num_batch_axes"],
            ),
        ],
    )
    coreai_program = converter.to_coreai()

    await _validate_numerics(
        coreai_program, model, sample, input_names=("x", "indices")
    )


@pytest.mark.ir
def test_externalize_unused_submodule_ir() -> None:
    """An externalizable submodule that the model's forward never calls is skipped.

    Previously, a registered submodule that did not appear in the exported graph
    caused the externalize pipeline to raise ``ValueError: Custom op for ...
    not found in any ancestor program``. The pipeline should instead warn and
    proceed, lowering the rest of the model normally.
    """

    class InnerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(4, 4)
            # Registered as a submodule but intentionally never invoked.
            self.unused = InnerModule()
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.post(self.pre(x))

    torch.manual_seed(42)
    model = OuterModel().eval()
    sample = (torch.randn(2, 4),)

    with pytest.warns(UserWarning, match="skipping unused submodule"):
        converter = TorchConverter().add_pytorch_module(
            model,
            export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
                get_decomp_table()
            ),
            externalize_modules=[InnerModule],
        )
        coreai_program = converter.to_coreai()

    check_file = """
        // CHECK-LABEL: module {
        // CHECK-NOT: coreai.graph noinline
        // CHECK: coreai.graph @main(
        // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x4xf32> {coreai.name = "x"}
        // CHECK-SAME: ) -> (tensor<2x4xf32>
        // CHECK: %[[MM0:[0-9a-z_]+]] = coreai.decomposable.broadcasting_batch_matmul %[[ARG0]],
        // CHECK-SAME: : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
        // CHECK: %[[ADD0:[0-9a-z_]+]] = coreai.decomposable.broadcasting_add %[[MM0]],
        // CHECK-SAME: : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
        // CHECK: %[[MM1:[0-9a-z_]+]] = coreai.decomposable.broadcasting_batch_matmul %[[ADD0]],
        // CHECK-SAME: : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
        // CHECK: %[[ADD1:[0-9a-z_]+]] = coreai.decomposable.broadcasting_add %[[MM1]],
        // CHECK-SAME: : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
        // CHECK-NOT: coreai.invoke
        // CHECK-NOT: coreai.relu
        // CHECK: coreai.output %[[ADD1]] : tensor<2x4xf32>
        // CHECK: }
        // CHECK-NOT: coreai.graph
        // CHECK: }
    """
    filecheck_pattern(str(coreai_program), check_file=check_file)


async def test_externalize_unused_submodule_numerics() -> None:
    """Numerics: unused externalizable submodule does not affect output."""

    class InnerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc(x))

    class OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(4, 4)
            self.unused = InnerModule()
            self.post = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.post(self.pre(x))

    torch.manual_seed(42)
    model = OuterModel().eval()
    sample = (torch.randn(2, 4),)

    with pytest.warns(UserWarning, match="skipping unused submodule"):
        converter = TorchConverter().add_pytorch_module(
            model,
            export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
                get_decomp_table()
            ),
            externalize_modules=[InnerModule],
        )
        coreai_program = converter.to_coreai()

    await _validate_numerics(coreai_program, model, sample)
