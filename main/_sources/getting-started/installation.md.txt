# Installing coreai-torch

Set up a Python environment and install coreai-torch.

## Overview

This article covers two installation paths for coreai-torch: `uv` and `conda`. Use `uv` when possible — it creates an isolated virtual environment, pins the correct Python version, and resolves all dependencies in one command.

## Prerequisites

- Python 3.11 or later
- PyTorch 2.8.0 or later

## Install with uv

[uv](https://docs.astral.sh/uv/) is the recommended package manager. For a development environment, run `uv sync` from the repository root:

```bash
uv sync
```

This installs standard dependencies in editable mode. To also include test dependencies, run:

```bash
uv sync --extra test
```

## Install with conda

If you prefer `conda`, create and activate a new environment first:

```bash
conda create -n coreai-torch python=3.11 -y
conda activate coreai-torch
```

Then install **from source** — clone the repository, navigate to its root, and run:

```bash
pip install -e .
```

## Verify installation

Run the following to confirm coreai-torch is installed correctly — a version string confirms success:

```python
import coreai_torch
print(coreai_torch.__version__)
```

## Next steps

Head to the {doc}`quickstart` tutorial to convert your first PyTorch model and explore more.

## Notices

PyTorch is a trademark of Meta Platforms, Inc. conda is a product of Anaconda, Inc.
