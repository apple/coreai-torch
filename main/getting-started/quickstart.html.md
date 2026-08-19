# Quickstart

Convert a PyTorch model to Core AI format, compile it, and run inference.

This tutorial walks through the full conversion pipeline for a simple PyTorch model — from export to on-device inference. By the end, you have a working end-to-end pipeline you can adapt for your own models. For a production-scale example, see [Real-World Model: MobileNetV2]() at the end of this tutorial.

## Define a Model

```ipython3
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


model = SimpleModel()
model.eval()
```

Always call `.eval()` before exporting. Layers such as `BatchNorm` and `Dropout` behave differently in training mode and produce a different graph.

## Export

Use `torch.export.export` to capture the computation graph as an `ExportedProgram`. The result is shape- and dtype-specialized to the provided example input.

```ipython3
example_input = (torch.randn(1, 10),)
exported = torch.export.export(model, args=example_input)
```

## Decompose

`get_decomp_table()` returns the default PyTorch ATen decomposition table minus the operations that `TorchConverter` lowers as composite ops, so those operations are preserved in the exported graph rather than being decomposed into lower-level primitives.

```ipython3
from coreai_torch import get_decomp_table

exported = exported.run_decompositions(get_decomp_table())
```

#### WARNING
This call is required when using `add_exported_program()`. Skipping it will leave ops in the graph that have no lowering rule.

## Convert

`TorchConverter` walks the FX graph node-by-node and emits CoreAI operations, returning a `AIProgram`.

```ipython3
from coreai_torch import TorchConverter

converter = TorchConverter()
coreai_program = converter.add_exported_program(
    exported,
    input_names=["x"],
    output_names=["out"],
).to_coreai()
coreai_program.optimize()
```

## Compile and Run

The `AIProgram` must be compiled to a native executable before it can run inference. The full async function below saves the program to an `.aimodel` directory, loads the executable, runs the `main` function on the example input, and compares the result to PyTorch.

```ipython3
import tempfile
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import NDArray


async def compile_and_run(coreai_program, example_input, model):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Compile: save the AIProgram to an .aimodel directory on disk.
        asset = coreai_program.save_asset(Path(tmpdir) / "quick_start_example.aimodel")

        # Load: open the executable and bind the `main` function.
        async with asset.executable() as ai_model:
            function = ai_model.load_function("main")

            # Run: invoke the function on the example input.
            coreai_outputs = await function({"x": NDArray(example_input[0])})

            # Compare with PyTorch: run the same input through the original model.
            with torch.no_grad():
                pytorch_output = model(example_input[0])

            coreai_output = coreai_outputs["out"].numpy()
            pytorch_numpy = pytorch_output.numpy()

            print(f"PyTorch output shape: {pytorch_numpy.shape}")
            print(f"Core AI output shape: {coreai_output.shape}")
            print(
                f"Outputs match: {np.allclose(pytorch_numpy, coreai_output, atol=1e-4)}"
            )


await compile_and_run(coreai_program, example_input, model)
```

---

## Real-World Model: MobileNetV2

#### NOTE
This example uses [torchvision](https://pytorch.org/vision/), which is not a dependency of coreai-torch. Install it into your environment first:

```bash
pip install torchvision
```

The same pattern works with production models. Here is the full pipeline for torchvision’s MobileNetV2:

```ipython3
import torch
import torchvision.models as tv_models

model = tv_models.mobilenet_v2(weights=None).eval()
example_input = (torch.randn(1, 3, 224, 224),)

exported = torch.export.export(model, args=example_input)
exported = exported.run_decompositions(get_decomp_table())

coreai_program = (
    TorchConverter()
    .add_exported_program(
        exported,
        input_names=["image"],
        output_names=["logits"],
    )
    .to_coreai()
)
coreai_program.optimize()
```

Compile and run the same way as above:

```ipython3
async def run():
    with tempfile.TemporaryDirectory() as tmpdir:
        asset = coreai_program.save_asset(Path(tmpdir) / "mobilenet_v2_example.aimodel")
        async with asset.executable() as ai_model:
            function = ai_model.load_function("main")
            outputs = await function({"image": NDArray(example_input[0])})
            logits = outputs["logits"].numpy()
            print("logits shape:", logits.shape)  # (1, 1000)


await run()
```

---

## Alternative: `add_pytorch_module`

`add_pytorch_module()` accepts an `nn.Module` directly. You still provide an `export_fn` that handles export and decomposition:

```ipython3
import torch

import coreai_torch

model = SimpleModel().eval()
sample = (torch.randn(1, 10),)

coreai_program = (
    TorchConverter()
    .add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            coreai_torch.get_decomp_table()
        ),
    )
    .to_coreai()
)
coreai_program.optimize()
```

This uses `add_pytorch_module` to wrap export and conversion in one call. See [Conversion Workflows](../guides/conversion-workflows.md) for all options.

## Next Steps

- [Conversion Workflows](../guides/conversion-workflows.md) — when to use `add_exported_program` vs `add_pytorch_module`, and how to handle dynamic shapes.
- [Composite Ops Guide](../guides/composite-ops.md) — preserve ops like attention and RMSNorm so the compiler can dispatch them to hardware-optimized kernels.
- [TorchConverter API reference](../api/TorchConverter.md) — full API reference for `TorchConverter`.

## Notices

PyTorch and torchvision are trademarks of Meta Platforms, Inc.
