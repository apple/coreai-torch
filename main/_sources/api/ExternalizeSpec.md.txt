# ExternalizeSpec

Specifies which submodule class to externalize during conversion as a named composite op the compiler can recognize and optimize.

## Overview

**Public import:**

```python
from coreai_torch import ExternalizeSpec
```

`ExternalizeSpec` is passed to the `externalize_modules` parameter of `TorchConverter.add_pytorch_module()`. Every `nn.Module` instance whose class matches `target_class` (via `isinstance`) is preserved as a named composite op.

For an end-to-end walkthrough, see {doc}`../guides/externalization`.

```{note}
Passing a bare class (or an `ExternalizeSpec` with only `target_class`) performs *simple externalization* — the submodule is extracted into its own standalone graph with no composite-op metadata. This is **experimental**; prefer setting `composite_op_name`.
```

## Declaration

```python
@dataclass
class ExternalizeSpec:
    target_class: type
    composite_op_name: str | None = None
    composite_attrs: list[str] | None = None
```

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `target_class` | `type` | — | The `nn.Module` subclass to match. Every instance found in the model is externalized as a composite op. |
| `composite_op_name` | `str \| None` | `None` | The name of the composite op the submodule is preserved as, so the compiler can recognize and optimize it. |
| `composite_attrs` | `list[str] \| None` | `None` | Names of instance attributes (e.g. `["eps", "axes"]`) whose values are recorded as attributes of the composite op. Only valid when `composite_op_name` is set. |

## Validation

- `composite_attrs` may only be set when `composite_op_name` is also set. Setting `composite_attrs` without `composite_op_name` raises `ValueError`.
- If a `target_class` does not match any submodule in the model, the converter emits a `UserWarning` (not an error) — this allows passing a superset of specs across model variants.

## Usage

Set `composite_op_name` and (optionally) `composite_attrs`. The submodule is preserved as a named composite op carrying its attributes, so the compiler can recognize and optimize it.

```python
ExternalizeSpec(
    target_class=RMSNormImpl,
    composite_op_name="rms_norm",
    composite_attrs=["axes", "eps", "version"],
)
```

## Examples

**Composite-op externalization** — preserve the op as a named composite op:

```python
from coreai_torch import TorchConverter, ExternalizeSpec
from coreai_torch.composite_ops import SDPA

converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
        coreai_torch.get_decomp_table()
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
coreai_program.optimize()
```

## Notices

PyTorch is a trademark of Meta Platforms, Inc.
