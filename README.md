# Core AI PyTorch Extensions (coreai-torch)

Core AI PyTorch Extensions (`coreai-torch`) is a Python package that bridges PyTorch and Core AI. Use it to bring up an existing PyTorch model into Core AI IR, or to author Core AI models directly from PyTorch by composing the built-in composite op library (`coreai_torch.composite_ops`), authoring new ops via `register_torch_lowering`, and authoring inline Metal GPU kernels via `TorchMetalKernel`. The resulting IR can be compiled and executed efficiently by the Core AI inference stack.

🔗 Jump to: [Getting started](#getting-started) · [Documentation](#documentation) · [Contributing](#contributing) · [Support](#support) · [License](#license)

## Overview

`coreai-torch` traverses a `torch.export.ExportedProgram` and produces Core AI
IR — the same IR consumed by the Core AI compiler and runtime. The public entry
point is `TorchConverter`, which lowers PyTorch operators to Core AI dialect
operations, preserves location and module-stack information for debugging, and
provides extension points for custom Metal kernels and submodule
externalization.

## Getting started

### Installation

```bash
pip install coreai-torch
```

Or from source with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### Usage

`TorchConverter` accepts models in two forms. Pick based on what you have and whether you need externalization:

| You have | Use |
|---|---|
| A decomposed `ExportedProgram` | `add_exported_program()` |
| An `nn.Module` + need externalization | `add_pytorch_module()` with `externalize_modules` |
| An `nn.Module`, no externalization | Either method — `add_exported_program()` is more direct |

#### From an ExportedProgram

You export and decompose the model yourself, then pass the `ExportedProgram` directly.
Use `get_decomp_table()` so that composite ops (`instance_norm`, `pixel_shuffle`, `scaled_dot_product_attention`) are preserved for optimal runtime performance.

```python
import torch
from coreai_torch import TorchConverter, get_decomp_table

model = ...  # your nn.Module
model.eval()

# Export and decompose — this is your responsibility
ep = torch.export.export(model, args=(torch.randn(1, 3, 224, 224),))
ep = ep.run_decompositions(get_decomp_table())

# Convert to Core AI IR
converter = TorchConverter().add_exported_program(ep)
coreai_program = converter.to_coreai()
coreai_program.optimize()
```

#### From an nn.Module

Pass your model and an `export_fn` that returns a decomposed `ExportedProgram`. This is equivalent to calling `add_exported_program()` with the result of `export_fn`.

```python
import coreai_torch
from coreai_torch import TorchConverter

model = ...  # your nn.Module
model.eval()
sample = (torch.randn(1, 3, 224, 224),)

converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)
coreai_program = converter.to_coreai()
coreai_program.optimize()
```

## Documentation

- API reference and conversion guides — see the [`docs/`](docs/) directory
- Supported model types and ops — see the operator coverage notebooks under [`docs/`](docs/)

### Building docs locally

```bash
uv sync --extra docs
uv run jupyter-book build docs/
open docs/_build/html/index.html
```

## Development

### Setting up pre-commit hooks

This repo uses [pre-commit](https://pre-commit.com) to run linting and
formatting checks automatically. Install the hooks once after cloning:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

### Running tests

```bash
uv sync --extra test
uv run pytest tests/ -n auto
```

### Testing notebooks

```bash
uv run pytest docs/ --nbmake -v
```

## Contributing

We welcome contributions within a defined scope. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) carefully before opening a pull request or
issue — particularly the section on contribution scope.

## Support

- [GitHub Issues](../../issues) — Bug reports, feature requests, and questions

## Security and code of conduct

Security vulnerability reporting and the Code of Conduct for this project are
governed at the org level via the
[Apple Open Source `.github` repository](https://github.com/apple/.github).

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).

## Related projects

- [Core AI](https://developer.apple.com/documentation/coreai) — Apple's on-device AI inference stack
- [Core AI Optimization](https://github.com/apple/coreai-optimization) — model compression for deployment on Apple Silicon
- [Core AI Models](https://github.com/apple/coreai-models) — ready-to-run optimized models, Python reproduction scripts, and Swift utilities for on-device integration
