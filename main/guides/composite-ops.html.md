# Composite Ops Guide

Composite ops preserve an operation’s boundary — such as RMSNorm, RoPE, or attention — keeping it intact as a recognizable unit during conversion. This lets the compiler recognize the operation and apply an implementation optimized for it.

coreai-torch produces composite ops in two ways:

- **Module-class composite ops** — `nn.Module` subclasses exposed in `coreai_torch.composite_ops` (such as `RMSNormImpl`, `RoPE`, and `SDPA`). You build these into your model and externalize them with an `ExternalizeSpec`, as shown in this guide.
- **ATen-derived composite ops** — recognized automatically from the ATen nodes (`fx.Node`s) in your `ExportedProgram` during conversion (such as `instance_norm`, `layer_norm`, and `pixel_shuffle`). These have no `nn.Module` wrapper: use the standard PyTorch APIs and the converter preserves them as composite ops, as long as `get_decomp_table()` keeps them from being decomposed.

This guide covers the module-class workflow. For the full list of both kinds, see the [Composite ops API reference](../api/composite-ops.md) reference.

## The General Pattern

Using any composite op involves three steps:

1. **Use the provided class** as a named submodule in your model — not as the root module.
2. **Convert via `add_pytorch_module`** — this is the required entrypoint for composite op externalization.
3. **Pass an `ExternalizeSpec`** with `composite_op_name` and `composite_attrs` identifying the op and the instance attributes that define it.

Generic skeleton:

```ipython3
import torch
import torch.nn as nn

import coreai_torch
from coreai_torch import ExternalizeSpec, TorchConverter
from coreai_torch.composite_ops import RMSNormImpl
```

```python
import torch

import coreai_torch
from coreai_torch import ExternalizeSpec, TorchConverter

coreai_program = (
    TorchConverter()
    .add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            coreai_torch.get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=SomeOp,
                composite_op_name="some_op",
                composite_attrs=["attr1", "attr2"],
            )
        ],
    )
    .to_coreai()
)
coreai_program.optimize()
```

The converter preserves each matching submodule instance as a named composite op carrying its attributes, so the compiler can recognize and optimize it.

---

## Worked Example: RMSNorm

This section walks through the full cycle — model definition, conversion, and numerical verification.

### Define a model

```ipython3
class RMSNorm(nn.Module):
    """Convenience wrapper that owns the learnable scale parameter."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.norm = RMSNormImpl(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x, self.weight)


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, ff_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(dim=dim, eps=eps)
        self.fc1 = nn.Linear(dim, ff_dim, bias=False)
        self.fc2 = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(self.norm(x))))


model = FeedForwardBlock(dim=64, ff_dim=256).eval()
sample = (torch.randn(2, 16, 64),)
```

### Convert

Externalize `RMSNormImpl` as the `rms_norm` composite op. Because `RMSNormImpl.forward(input, scale)` takes the scale as an explicit argument, the learnable `weight` parameter on the wrapper appears as a graph input on the composite op boundary rather than being baked in as a constant.

#### NOTE
`coreai_torch.composite_ops` also ships an `RMSNorm` convenience wrapper that owns the learnable scale for you, so you can import that instead of defining one yourself — but `target_class` in the `ExternalizeSpec` must still be `RMSNormImpl`.

```ipython3
import torch

coreai_program = (
    TorchConverter()
    .add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            coreai_torch.get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=RMSNormImpl,
                composite_op_name="rms_norm",
                composite_attrs=["axes", "eps"],
            )
        ],
    )
    .to_coreai()
)
coreai_program.optimize()
```

### Verify numerical correctness

```ipython3
import tempfile
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import NDArray


async def run_and_compare(coreai_program, model, sample):
    with tempfile.TemporaryDirectory() as tmpdir:
        asset = coreai_program.save_asset(Path(tmpdir) / "rms_norm_example.aimodel")
        async with asset.executable() as ai_model:
            function = ai_model.load_function("main")

            x = sample[0]
            coreai_out = await function({"x": NDArray(x)})
            coreai_np = list(coreai_out.values())[0].numpy()

            with torch.no_grad():
                torch_np = model(x).numpy()

            print(f"Max abs error: {np.abs(torch_np - coreai_np).max():.2e}")
            print(f"Outputs match: {np.allclose(torch_np, coreai_np, atol=1e-4)}")


await run_and_compare(coreai_program, model, sample)
```

---

## Other Composite Ops

The same three-step pattern (use the class as a submodule → convert via `add_pytorch_module` → pass an `ExternalizeSpec`) works for every built-in composite op. See the [Composite ops API reference](../api/composite-ops.md) reference for the full list, including constructor signatures, forward arguments, and attributes.

---

#### TIP
- Use `add_pytorch_module` — it is the required entrypoint for externalization.
- `composite_attrs` must match actual instance attribute names on the target class (e.g., `self.eps`, `self.axes`).

## Next Steps

- [Externalization](externalization.md) — extract composite op submodules as separate named functions in the converted model.
- [Custom Op Lowering](custom-op-lowering.md) — implement a custom composite op lowering with `register_torch_lowering` and `generate_composite_decl`.
- [Composite ops API reference](../api/composite-ops.md) — full API reference for all built-in composite op modules.

## Notices

PyTorch is a trademark of Meta Platforms, Inc. Hugging Face is a trademark of Hugging Face, Inc.
