# Core AI PyTorch Extensions (coreai-torch)

Bring PyTorch models to Core AI for on-device execution.

## Overview

Core AI PyTorch Extensions (`coreai-torch`) is a Python package that bridges PyTorch and Core AI. You can use it to bring up an existing PyTorch model — exported as a `torch.export.ExportedProgram` — into a Core AI `AIProgram` ready to run on Apple hardware, traversing the FX graph node-by-node and mapping ATen operators to Core AI operations. You can equally use it to author Core AI models directly from PyTorch by composing the library of composite ops in `coreai_torch.composite_ops`, authoring new ops via `register_torch_lowering`, and authoring inline Metal GPU kernels through `TorchMetalKernel` and `register_custom_kernels` — all expressed as PyTorch `nn.Module`s and lowered to Core AI IR that the compiler recognizes and optimizes natively.

The bring-up pipeline has three steps. First, export your PyTorch model with `torch.export.export` to capture the computation graph. Second, decompose the exported program with `get_decomp_table()`, which lowers composite ATen ops to the primitive set that `TorchConverter` can map while preserving the operations that `TorchConverter` lowers as composite ops. Third, call `TorchConverter().add_exported_program(ep).to_coreai()` to produce the `AIProgram`.

For authoring, `coreai_torch.composite_ops` exposes well-known building blocks — such as attention, RoPE embeddings, RMSNorm, and gather-matmul (the MoE primitive) — as PyTorch modules. Passing these modules to `externalize_modules` preserves each one's operation boundary as a named composite op that the compiler can recognize and optimize. When a PyTorch op has no built-in lowering rule, register a custom lowering function with `register_torch_lowering`. For compute-intensive custom operations, `register_custom_kernels` lets you author Metal kernel source and wire it into the conversion pipeline.

## Quick example

```python
import torch
from coreai_torch import TorchConverter, get_decomp_table

model = MyModel().eval()
ep = torch.export.export(model, args=(torch.randn(1, 10),))
ep = ep.run_decompositions(get_decomp_table())
coreai_program = TorchConverter().add_exported_program(ep).to_coreai()
coreai_program.optimize()
```

## Choosing your workflow

| Starting point | Recommended approach |
|---|---|
| Already have a decomposed `ExportedProgram` | `TorchConverter().add_exported_program(ep).to_coreai()` |
| Have an `nn.Module`, no externalization | Either `add_exported_program` or `add_pytorch_module` |
| Have an `nn.Module`, need externalization | `add_pytorch_module(model, ..., externalize_modules=[...])` |

{doc}`guides/externalization` lets the Core AI compiler optimize submodules independently or hand them off to specialized backends. See {doc}`guides/conversion-workflows` for detailed code and a decision guide.

## Next steps

- **New users:** {doc}`getting-started/installation` and {doc}`getting-started/quickstart` walk you through setup and your first end-to-end bring-up.
- **Authoring Core AI models from PyTorch:** {doc}`guides/composite-ops` covers the built-in composite op library, {doc}`guides/custom-op-lowering` shows how to author Core AI IR for new torch ops, and {doc}`guides/custom-metal-kernels` walks through authoring inline Metal GPU kernels.
- **Customizing bring-up:** {doc}`guides/conversion-workflows` covers each bring-up workflow. {doc}`guides/externalization` covers preserving submodule boundaries as composite ops.
- **API reference:** {doc}`api/TorchConverter` documents every method and parameter. {doc}`api/composite-ops` lists all built-in composite ops. {doc}`api/TorchMetalKernel` covers the Metal-kernel authoring API.

## Links

- [Repository](https://github.com/apple/coreai-torch)
- [Issue tracker](https://github.com/apple/coreai-torch/issues)

## Notices

PyTorch is a trademark of Meta Platforms, Inc.

```{toctree}
:caption: Getting Started
:hidden:

getting-started/installation
getting-started/quickstart
whats-new
```

```{toctree}
:caption: Guides
:hidden:

guides/conversion-workflows
guides/custom-op-lowering
guides/custom-metal-kernels
guides/composite-ops
guides/externalization
```

```{toctree}
:caption: API Reference
:hidden:

api/TorchConverter
api/composite-ops
api/generate-composite-decl
api/ExternalizeSpec
api/TorchMetalKernel
api/supported-aten-ops
api/debugging
```

```{toctree}
:caption: Core AI (coreai-core)
:hidden:

coreai-core/index
```

```{toctree}
:caption: More
:hidden:

faq
contributing
resources
release-notes
```
