# Conversion Workflows

`TorchConverter` accepts models in two forms. Pick based on what you have and whether you need [Externalization](externalization.md) to optimize submodules independently or hand off to specialized backends.

---

## From an ExportedProgram

You export and decompose the model yourself, then pass the `ExportedProgram` directly.

```ipython3
import torch
import torch.nn as nn

from coreai_torch import TorchConverter, get_decomp_table


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


model = MyModel().eval()

ep = torch.export.export(model, args=(torch.randn(1, 10),))
ep = ep.run_decompositions(get_decomp_table())

converter = TorchConverter().add_exported_program(ep)
coreai_program = converter.to_coreai()
coreai_program.optimize()
```

**When to use:** You need full control over the export pipeline, or you already have an exported program from another tool.

#### WARNING
You **must** call `run_decompositions()` before passing the program. Use `get_decomp_table()` to preserve the operations that `TorchConverter` lowers as composite ops.

---

## From an nn.Module

Pass your model and an `export_fn` that returns a decomposed `ExportedProgram`. This is equivalent to calling `add_exported_program()` with the result of `export_fn`.

```ipython3
import torch

import coreai_torch

model = MyModel().eval()
sample = (torch.randn(1, 10),)

converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)
coreai_program = converter.to_coreai()
coreai_program.optimize()
```

**When to use:** You have an `nn.Module` and want to pass it directly. Required if you need externalization; otherwise equivalent to `add_exported_program()`.

---

## Externalizing submodules

Externalizing a submodule preserves its operation boundary during conversion, so the operation stays intact as a recognizable unit. When you mark a well-known building block — such as attention, RoPE, or RMSNorm — as a **composite op**, the compiler recognizes that operation and can apply an optimized implementation tailored to it, producing a faster model.

Externalization uses `add_pytorch_module` with the `externalize_modules` argument.

### Composite op externalization

Tag a submodule with an `ExternalizeSpec`, giving it a composite op name and the attributes that define its behavior. The name and attributes are carried through conversion so the compiler can recognize the operation and optimize it:

```ipython3
import torch
import torch.nn as nn

from coreai_torch import ExternalizeSpec, TorchConverter
from coreai_torch.composite_ops import RMSNormImpl


class RMSNorm(nn.Module):
    """Convenience wrapper that owns the learnable scale parameter."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.norm = RMSNormImpl(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x, self.weight)


class ModelWithNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(dim=10)
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(self.norm(x))


model = ModelWithNorm().eval()
sample = (torch.randn(1, 10),)

converter = TorchConverter().add_pytorch_module(
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
coreai_program = converter.to_coreai()
coreai_program.optimize()
```

**When to use:** Mark performance-critical building blocks (RMSNorm, RoPE, SDPA, and similar) so the compiler can optimize them as recognized operations.

#### NOTE
`coreai_torch.composite_ops` ships convenience wrappers like `RMSNorm` that own the learnable scale for you, so you can use those instead of defining one yourself — but `target_class` in the `ExternalizeSpec` must still be `RMSNormImpl` (the inner module the converter recognizes as the `rms_norm` composite op).

Passing a bare module class to `externalize_modules` instead of an `ExternalizeSpec` performs *simple externalization*: the submodule is extracted into its own standalone graph with no composite-op metadata and no optimization benefit. This is **experimental** — prefer composite-op externalization above.

See [Composite Ops Guide](composite-ops.md) for built-in composite ops and [Externalization](externalization.md) for advanced externalization patterns.

---

## Comparison

|                     | From ExportedProgram          | From nn.Module                           | With externalization                     |
|---------------------|-------------------------------|------------------------------------------|------------------------------------------|
| **Entry point**     | `add_exported_program()`      | `add_pytorch_module()`                   | `add_pytorch_module()`                   |
| **You manage**      | Export + decomposition        | Export + decomposition (via `export_fn`) | Export + decomposition (via `export_fn`) |
| **Decomposition**   | Manual (`run_decompositions`) | Manual (inside `export_fn`)              | Manual (inside `export_fn`)              |
| **Externalization** | Not available                 | Not available                            | `externalize_modules`                    |
| **Best for**        | Full pipeline control         | nn.Module input                          | Composite ops                            |

## Next Steps

- [Composite Ops Guide](composite-ops.md) — preserve attention, normalization, and MoE ops as recognizable units for hardware-optimized dispatch.
- [Custom Op Lowering](custom-op-lowering.md) — register lowerings for custom or unsupported PyTorch ops.
- [Supported ATen ops](../api/supported-aten-ops.md) — full list of ATen ops with built-in lowering rules.

## Notices

PyTorch is a trademark of Meta Platforms, Inc.
