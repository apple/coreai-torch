# Externalization

Externalization preserves a submodule’s operation boundary during conversion, so the operation stays intact as a recognizable unit in the converted model. When you mark a well-known building block — such as attention, RoPE, or RMSNorm — as a **composite op**, the compiler recognizes that operation and can apply an implementation optimized for it, producing a faster model.

Use `ExternalizeSpec` to tag the extracted submodule with a composite op name and the attributes that define it, so the compiler can recognize and optimize the operation.

---

## What externalization produces

Both modes extract the matching submodule into its own `coreai.graph` that `@main` calls with `coreai.invoke`. The difference is what that graph carries. Take a model whose `forward` is `linear(norm(x))`, with the `norm` submodule externalized:

**Composite op externalization** marks the extracted graph as a named composite op — note the `private` graph and the `composite_decl` attribute recording the op name and its attributes, which the compiler recognizes and optimizes:

```llvm
module {
  coreai.graph private noinline @norm.rms_norm(
      %arg0: tensor<1x10xf32> {coreai.name = "input"},
      %arg1: tensor<10xf32> {coreai.name = "scale"}
  ) -> tensor<1x10xf32> attributes {
      composite_decl = #coreai.composite_declaration<"rms_norm" = {
          input_names = ["input", "scale"],
          op_attrs = {axes = -1 : si64, eps = 9.99999974E-6 : f32, version = 1 : si64},
          output_names = ["output"]}>
  } {
    // ... rms-norm body ...
    coreai.output %15 : tensor<1x10xf32>
  }
  coreai.graph @main(%arg0: tensor<1x10xf32>) -> tensor<1x5xf32> {
    %3 = coreai.invoke @norm.rms_norm(%arg0, %0)
        : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    // ... linear ...
    coreai.output %7 : tensor<1x5xf32>
  }
}
```

**Simple externalization** extracts the same submodule as a plain graph boundary — no `composite_decl`, so the compiler sees an opaque subgraph rather than a named op:

```llvm
module {
  coreai.graph noinline @norm(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
    // ... rms-norm body ...
    coreai.output %16 : tensor<1x10xf32>
  }
  coreai.graph @main(%arg0: tensor<1x10xf32>) -> tensor<1x5xf32> {
    %3 = coreai.invoke @norm(%arg0)
        : (tensor<1x10xf32>) -> tensor<1x10xf32>
    // ... linear ...
    coreai.output %7 : tensor<1x5xf32>
  }
}
```

Symbol names and constants above are illustrative (the converter appends a hash suffix to each externalized graph name).

---

## Composite Op Externalization

Use `ExternalizeSpec` to mark a submodule as a named composite op:

```ipython3
import torch
import torch.nn as nn


class RMSNormComposite(nn.Module):
    def __init__(self, axes=-1, eps=1e-5, version=1):
        super().__init__()
        self.axes = axes
        self.eps = eps
        self.version = version

    def forward(self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x_f32 = input.to(torch.float32)
        inv_rms = torch.rsqrt((x_f32 * x_f32).mean(self.axes, keepdim=True) + self.eps)
        return (input * inv_rms).to(input.dtype) * scale


model = RMSNormComposite().eval()
sample = (torch.randn(10), torch.randn(10))
```

```python
import torch

import coreai_torch
from coreai_torch import ExternalizeSpec, TorchConverter

converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
    externalize_modules=[
        ExternalizeSpec(
            target_class=RMSNormComposite,
            composite_op_name="rms_norm",
            composite_attrs=["axes", "eps", "version"],
        )
    ],
)
coreai_program = converter.to_coreai()
coreai_program.optimize()
```

Each instance is externalized using its own attribute values, so two instances configured differently are preserved as distinct composite ops.

### Input and output names

`input_names` and `output_names` for each composite op are derived automatically from the module’s parameters, buffers, and `forward()` signature — no manual bookkeeping is needed. Optional arguments passed as `None` at a call site are excluded automatically.

### Requirements for composite op modules

Composite op `forward` methods must follow two rules:

1. **Forward arguments must be tensors** — all `forward` parameters that become inputs must be `torch.Tensor`. Scalar configuration (e.g., `eps`, `is_causal`) should be stored as instance attributes and serialized via `composite_attrs` in `ExternalizeSpec`.
2. **Optional arguments must use `torch.Tensor | None = None`** — when an optional is not provided (left as `None`), it is excluded entirely and does not appear in `input_names`. There is no support for default tensor values.

---

## Simple externalization (experimental)

Passing a bare module class to `externalize_modules` — instead of an `ExternalizeSpec` — extracts the submodule into its own standalone graph with no composite-op metadata. It offers no optimization benefit and simply defines a boundary around the submodule.

#### WARNING
Simple externalization is **experimental**. Prefer composite-op externalization above.

---

## API Quick Reference

### ExternalizeSpec

| Field               | Type               | Description                                                                                         |
|---------------------|--------------------|-----------------------------------------------------------------------------------------------------|
| `target_class`      | `type`             | The `nn.Module` subclass to match. Every instance found in the model will be externalized.          |
| `composite_op_name` | `str | None`       | If set, the submodule is preserved as a named composite op the compiler can recognize and optimize. |
| `composite_attrs`   | `list[str] | None` | Instance attribute names to record as attributes of the composite op.                               |

Set `composite_op_name`, and optionally `composite_attrs`, to preserve the submodule as a composite op the compiler can optimize.

### Relevant `add_pytorch_module` parameters

| Parameter             | Type                           | Description                                                                                       |
|-----------------------|--------------------------------|---------------------------------------------------------------------------------------------------|
| `externalize_modules` | `list[ExternalizeSpec] | None` | The `ExternalizeSpec` objects describing which submodule classes to externalize as composite ops. |

See [TorchConverter API reference](../api/TorchConverter.md) for the full method signature.

## Next Steps

- [Composite Ops Guide](composite-ops.md) — the built-in composite op modules you can pass to `externalize_modules`.
- [Conversion Workflows](conversion-workflows.md) — `add_pytorch_module` workflow, which is required for externalization.
- [ExternalizeSpec](../api/ExternalizeSpec.md) — full API reference for `ExternalizeSpec`.

## Notices

PyTorch is a trademark of Meta Platforms, Inc.
