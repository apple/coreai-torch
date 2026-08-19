# Constructing a CoreAI Graph

This notebook walks through building a minimal CoreAI graph from scratch — without
going through a framework importer like PyTorch — and saving it as an `.aimodel`
asset on disk. By the end you will have a file you can load and execute in the
next tutorial.

Along the way we touch the four authoring concepts you will see throughout the
rest of these tutorials:

- **`TensorSpec`** — describes a tensor’s shape and dtype.
- **`Module`** — the container you append graphs into.
- **Graph entrypoint** — a named function (here, `main`) that defines the
  computation: its inputs, outputs, and the ops in between.
- **`AIProgram`** — a deployable program built from one or more modules. This is
  the unit you save to disk.

#### WARNING
**APIs in flux.** A few graph-building primitives (the `@graph` decorator and
elementwise op constructors) currently live under `coreai._compiler` while the
public authoring surface is finalized. They will be re-exported from
`coreai.authoring` in a follow-up release; the import paths in the setup cell
below are the only place this leaks through.

## Setup

Import the public authoring types from `coreai.authoring`, plus the (temporary)
graph-building primitives noted above.

```ipython3
import shutil
from pathlib import Path
from typing import Annotated

import numpy as np

# Graph-building primitives — pending re-export from coreai.authoring.
from coreai._compiler.dialects import coreai as ops
from coreai._compiler.ir import Value
from coreai.authoring import AIModelAsset, AIProgram, Module, TensorSpec
```

## Describe the inputs and outputs

A `TensorSpec` describes a tensor by **shape** and **dtype**. Inputs and outputs
of a graph are described with specs — the runtime uses them to validate the
tensors you pass in and the tensors it returns.

We will build a graph that takes one `2 × 3` `float32` tensor and produces one
`2 × 3` `float32` tensor.

```ipython3
input_spec = TensorSpec(shape=[2, 3], dtype=np.float32)
output_spec = TensorSpec(shape=[2, 3], dtype=np.float32, name="y")

print(f"input:  {input_spec}")
print(f"output: {output_spec}")
```

## Build the graph

Create a `Module` and, inside its `with` block, declare a graph entrypoint. The
`@ops.graph` decorator inspects a regular Python function and uses it to build
a CoreAI graph: the parameter and return annotations describe the graph’s
inputs and outputs, and the function body defines the computation. You do not
call `main` yourself — CoreAI builds the graph at decoration time.

Here the graph has one input `x` and returns `x + x` — a single elementwise add.

```ipython3
module = Module.create()
with module:

    @ops.graph
    def main(
        x: Annotated[Value, input_spec],
    ) -> Annotated[Value, output_spec]:
        return ops.add(x, x)


module.verify()
print("Module verified.")
```

## Wrap in an `AIProgram`

An `AIProgram` is the deployable unit. You construct one from a `Module` and use
it to run optimization passes or to serialize the program to disk.

```ipython3
program = AIProgram(module)
```

## Save as an `.aimodel`

`AIProgram.save_asset(path)` writes the program out as an `.aimodel` directory —
a small bundle containing the program bytecode plus a `metadata.json` file. The
saved asset is the artifact you ship; it can be loaded later for inference
without any of the authoring code you wrote above.

```ipython3
asset_path = Path("./hello-graph.aimodel")
if asset_path.exists():
    shutil.rmtree(asset_path)

asset = program.save_asset(asset_path)
print(f"saved to: {asset.path}")
```

## Inspect the saved asset

An `.aimodel` is a directory. List its contents and total size to confirm the
save succeeded — you should see the program bytecode plus its metadata.

```ipython3
files = sorted(p.name for p in asset_path.iterdir())
total_bytes = sum(p.stat().st_size for p in asset_path.rglob("*") if p.is_file())

print(f"contents:   {files}")
print(f"total size: {total_bytes} bytes")
```

## Validate that the asset is loadable

As a final check, load the asset back with `AIModelAsset.load`, open it as an
executable model, and confirm the `main` function is present. The next tutorial
in this series picks up from `hello.aimodel` and runs an actual inference.

```ipython3
reloaded = AIModelAsset.load(asset_path)
async with reloaded.executable() as model:
    # load_function raises KeyError if the name is missing.
    model.load_function("main")
print("OK")
```
