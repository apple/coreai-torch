# Composite ops API reference

Use these module subclasses (and ATen-derived ops) to preserve an operation's boundary through Core AI conversion as named composite ops the compiler can recognize and optimize.

## Overview

**Public import:**

```python
from coreai_torch.composite_ops import GatherMM, GatedDeltaUpdate, RMSNormImpl, RoPE, SDPA
```

coreai-torch provides composite ops in two categories:

- **{doc}`Module-class composite ops <composite-ops/module-class>`** — `nn.Module` subclasses you build into your model and externalize with an `ExternalizeSpec`. Pass them (or `ExternalizeSpec` objects) to the `externalize_modules` parameter of `add_pytorch_module()` to trigger externalization. For a tutorial walkthrough, see {doc}`../guides/composite-ops`.
- **{doc}`ATen-derived composite ops <composite-ops/aten-derived>`** — recognized automatically from the ATen nodes in your `ExportedProgram` during conversion. These have no corresponding `nn.Module` wrapper; use the standard PyTorch APIs (e.g., `torch.nn.BatchNorm2d`, `torch.nn.functional.pixel_shuffle`) and Core AI preserves them as composite ops.

```{toctree}
:hidden:

composite-ops/module-class
composite-ops/aten-derived
```

## Notices

PyTorch is a trademark of Meta Platforms, Inc. Hugging Face is a trademark of Hugging Face, Inc.
