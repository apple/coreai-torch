# Core AI PyTorch Extensions (coreai-torch)

Bring PyTorch models to Core AI for on-device execution.

## Overview

`coreai-torch` is a Python package that bridges PyTorch and Core AI. It converts PyTorch models — exported as `torch.export.ExportedProgram` — into a Core AI `AIProgram` by traversing the FX graph and mapping ATen operators to Core AI operations. For an overview of the Core AI ecosystem and how coreai-torch fits in, see [What is Core AI?](#what-is-core-ai).

`coreai-torch` is built around the following capabilities:

- **Bring up existing models.** Export your model with `torch.export.export`, decompose with `get_decomp_table()`, then call `TorchConverter().add_exported_program(ep).to_coreai()` to produce an `AIProgram` ready to run on Apple hardware.

- **Author Core AI models from PyTorch.** `coreai_torch.composite_ops` provides building blocks — attention, RoPE embeddings, RMSNorm, and gather-matmul — as PyTorch modules. Use `externalize_modules` to preserve operation boundaries as named composite ops the compiler can recognize and optimize.

- **Extend with custom ops and Metal kernels.** Register custom lowering functions with `register_torch_lowering` for ops with no built-in rule, or author inline Metal GPU kernels with `TorchMetalKernel` and `register_custom_kernels`.

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

## What is Core AI?

Core AI is a set of technologies for deploying machine learning models on Apple hardware, covering the full model deployment lifecycle: from model conversion, debugging, optimization, and integration into your app. Models run entirely on device on Apple silicon, with no server required.

```{image} _images/core-ai-ecosystem.png
:alt: Diagram of the Core AI ecosystem. At the top, Core AI Models provides ready-to-use models and examples. Core AI Optimization and Core AI PyTorch Extensions prepare models for deployment, producing a .aimodel file. Core AI Debugger and Xcode support integration and debugging. Core AI Framework runs models on device.
:align: center
```

The Core AI ecosystem consists of the following components:

- Convert PyTorch models to the Core AI model format (`.aimodel`) using [Core AI PyTorch Package](https://github.com/apple/coreai-torch)
- Compress models with quantization, palettization, and pruning using [Core AI Optimization](https://github.com/apple/coreai-optimization)
- Load and run models in an app with the [Core AI Framework](https://developer.apple.com/documentation/coreai)
- Inspect, debug, and profile models using [Core AI Debugger](https://developer.apple.com/documentation/coreai/inspecting-debugging-and-profiling-core-ai-models)
- Get popular open-source covered models with optimization and Swift app integration code using [Core AI Models](https://github.com/apple/coreai-models)

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
