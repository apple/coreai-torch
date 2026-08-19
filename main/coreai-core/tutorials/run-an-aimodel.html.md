# Running an `.aimodel` with `coreai.runtime`

This notebook loads an `.aimodel` asset and runs inference on it. We will use
four public types:

- **`AIModelAsset`** — the on-disk representation of a saved program
  (from `coreai.authoring`).
- **`InferenceFunction`** — a callable function inside the model (here, `main`).
- **`NDArray`** — the runtime’s multi-dimensional array type, used to pass
  inputs in and read outputs back.
- **`SpecializationOptions`** — for advanced device/configuration tuning
  (covered briefly at the end).

Inside the `async with` block you will also see an `AIModel` — the runnable
handle yielded by the asset’s `executable()` context manager. It is the type
you call `load_function` on, but you do not import or construct it directly.

We assume you have already produced a `hello.aimodel` from the previous
tutorial (*Constructing a CoreAI Graph*). The setup cell below recreates it
if it is missing, so this notebook is also safe to run standalone.

## Setup

Import the asset and runtime types. Inference is asynchronous, so we will
use `await` and `async with` directly inside cells (Jupyter runs each cell
in an event loop).

```ipython3
from pathlib import Path

import numpy as np
from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

asset_path = Path("./hello-run.aimodel")
```

### Ensure `hello.aimodel` exists

If the asset from the previous tutorial is not on disk, build a fresh copy
now. The build path mirrors the construction shown in *Constructing a CoreAI
Graph*. We’re going to go ahead and recreate the asset path.

```ipython3
from shutil import rmtree
from typing import Annotated

# Pending re-export from coreai.authoring; see the previous tutorial.
from coreai._compiler.dialects import coreai as ops
from coreai._compiler.ir import Value
from coreai.authoring import AIProgram, Module, TensorSpec

# Reconstruct asset.
if asset_path.exists():
    rmtree(asset_path)

module = Module.create()
with module:

    @ops.graph
    def main(
        x: Annotated[Value, TensorSpec(shape=[2, 3], dtype=np.float32)],
    ) -> Annotated[Value, TensorSpec(shape=[2, 3], dtype=np.float32, name="y")]:
        return ops.add(x, x)


AIProgram(module).save_asset(asset_path)
print(f"created {asset_path}")
```

## Open the asset

`AIModelAsset.load` reads the `.aimodel` directory header from disk so you
can inspect it; it does not yet compile the program for inference. That work
happens lazily inside the `executable()` async context manager used in the
next section.

#### NOTE
**Two ways to load a model.** This tutorial uses `AIModelAsset.load(path)`
followed by `async with asset.executable() as model:` — the resource-managed
form, which gives you explicit control over when the compiled model is
released. There is also a one-shot `await AIModel.load(path)` that returns
a runnable `AIModel` directly; reach for it when you want a long-lived
model handle without the `async with` block (e.g. inside an application
object that owns the model for its full lifetime).

```ipython3
asset = AIModelAsset.load(asset_path)
```

## Open the model and run inference

`asset.executable()` returns an async context manager that yields an
`AIModel` — a runnable handle to the compiled program. Resources are released
when the `async with` block exits, so all model usage happens inside it.

Inside the block we do five things in order:

1. list the functions exposed by the model,
2. look up `main` and inspect its `desc` (name, input names, output names),
3. build the input dict — each name from the signature maps to an `NDArray`
   (which accepts a NumPy array, a PyTorch tensor, or a Python list, wrapping
   the data without a copy where possible),
4. `await` the call to get back a `dict` keyed by output name, and
5. copy the result out as NumPy so we can inspect it after exiting the
   context.

```ipython3
async with asset.executable() as model:
    print(f"functions: {model.function_names}")

    function: InferenceFunction = model.load_function("main")
    desc = function.desc
    print(f"name:    {desc.name}")
    print(f"inputs:  {desc.input_names}")
    print(f"outputs: {desc.output_names}")

    x = np.full((2, 3), 1.5, dtype=np.float32)
    inputs = {"x": NDArray(x)}
    print(f"input x:\n{x}")

    outputs = await function(inputs)
    print(f"output keys: {list(outputs.keys())}")

    # Materialize the result inside the block — the model's backing buffers
    # are only guaranteed valid until the context exits.
    result = outputs["y"].numpy()
```

## Inspect the output

`.numpy()` was called inside the `async with` block to ensure the result was
materialized before the model’s buffers could be released. We can now inspect
the array freely. Our `main` function computes `x + x`, so for an input of
all `1.5`s we expect all `3.0`s back.

```ipython3
print(f"shape: {result.shape}")
print(f"dtype: {result.dtype}")
print(f"value:\n{result}")

assert result.shape == (2, 3)
assert result.dtype == np.float32
print("OK — inference produced expected output shape and dtype")
```

## What’s next

You now have the core loop: **load** an asset, **enter** its executable,
**build** inputs, **await** the call, and **read** the outputs back as NumPy.

For more advanced configuration, two types in `coreai.runtime` are worth
knowing about:

- **`SpecializationOptions`** — pass to `asset.executable(options)` to pin the
  preferred compute unit (CPU / GPU / Neural Engine) or enable debug mode.
   *(macOS only.)*
- **`StorageKind`** — passed to `NDArray(data, backing=...)` to choose
  byte-backed, IOSurface-backed, or Metal-backed storage. The default
  (`StorageKind.BYTES`) is what you want unless you are interoperating with
  graphics or camera buffers.

Both are importable from `coreai.runtime`, e.g.:

```python
from coreai.runtime import SpecializationOptions, StorageKind
```
