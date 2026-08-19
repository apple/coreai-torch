# Custom Metal Kernels

This guide shows how to author inline Metal GPU kernels and convert them through `TorchConverter`. Custom Metal kernels let you write raw Metal shader code, wrap it as a PyTorch op, and compile it into CoreAI operations for on-device execution.

#### WARNING
Authoring Metal kernels uses APIs from `coreai-core` (such as `coreai.authoring`). These APIs are experimental and subject to change in future releases.

## When You Need This

- You need a GPU kernel that is not available as a standard PyTorch or CoreAI op.
- You want to fuse multiple operations into a single Metal dispatch for performance.
- You need fine-grained control over thread dispatch, shared memory, or Metal-specific features.

---

## Step 1 — Define the Metal Kernel

Use `TorchMetalKernel` from `coreai_torch` to specify the kernel. You provide:

- A **name** for the kernel.
- **Input and result names** that match the variables in your Metal source.
- The **Metal shader body** (the code inside the `[[kernel]]` function — the signature is generated automatically).
- A **torch reference implementation** used during `torch.export` for shape inference.
- **Metal parameters** like `thread_position_in_grid` for indexing.

```ipython3
import torch
from coreai.authoring import MetalParameter

from coreai_torch import TorchMetalKernel


def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference implementation for shape inference during export."""
    return x + y


custom_add = TorchMetalKernel(
    "vector_add",
    input_names=["x", "y"],
    result_names=["output"],
    src="output[id] = x[id] + y[id];",
    torch_defn=torch_add,
    metal_params=[
        MetalParameter("id", "uint", "thread_position_in_grid"),
    ],
)
```

The `src` string is the body of the Metal `[[kernel]]` function. The function signature — buffer bindings, thread attributes, and `#include <metal_stdlib>` — is generated automatically from `input_names`, `result_names`, and `metal_params`.

`TorchMetalKernel` wraps your kernel as a PyTorch op, so you can call it directly inside an `nn.Module`.

## Step 2 — Use the Kernel in a Model

Call the kernel inside an `nn.Module`. You must specify `threads_per_grid`, `threads_per_thread_group`, and `result_shapes` at each call site.

```ipython3
import torch
import torch.nn as nn


class AddModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return custom_add(
            x,
            y,
            threads_per_grid=(x.shape[0], 1, 1),
            threads_per_thread_group=(1, 1, 1),
            result_shapes=[list(x.shape)],
        )
```

## Step 3 — Export and Decompose

Export the model as usual. Custom kernel ops are preserved through `run_decompositions()` — they are not decomposed.

```ipython3
import torch

from coreai_torch import get_decomp_table

model = AddModel().eval()
example_inputs = (torch.randn(16), torch.randn(16))

exported = torch.export.export(model, args=example_inputs)
exported = exported.run_decompositions(get_decomp_table())
```

## Step 4 — Register Kernels and Convert

Use `register_custom_kernels()` to tell the converter how to lower the kernel ops. This must be called **before** `add_exported_program()`.

```ipython3
from coreai_torch import TorchConverter

converter = TorchConverter()
converter.register_custom_kernels([custom_add])
converter.add_exported_program(
    exported,
    input_names=["x", "y"],
    output_names=["result"],
)
coreai_program = converter.to_coreai()
coreai_program.optimize()
```

---

## Dtype Templating

Use `template_dtypes` to write a single kernel that works across multiple data types. Placeholder strings in the Metal source are replaced with the actual Metal type at compile time.

```ipython3
import torch


def torch_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, y)


custom_matmul = TorchMetalKernel(
    "matmul",
    input_names=["A", "B"],
    result_names=["C"],
    src="""
        const uint K = A.get_extent(0);
        const uint M = A.get_extent(1);
        const uint N = B.get_extent(0);
        if (gid.x >= N || gid.y >= M) return;
        TYPE sum = 0.0f;
        for (uint k = 0; k < K; ++k) {
            sum += A[k, gid.y] * B[gid.x, k];
        }
        C[gid.x, gid.y] = sum;
    """,
    torch_defn=torch_matmul,
    metal_params=[
        MetalParameter("gid", "uint2", "thread_position_in_grid"),
    ],
    # "A" is the input whose dtype determines the substitution;
    # every occurrence of "TYPE" in src is replaced with the
    # corresponding Metal type (e.g. "half", "float", "bfloat").
    template_dtypes={"A": "TYPE"},
)
```

```ipython3
import torch
import torch.nn as nn


def torch_sincos(x: torch.Tensor) -> list[torch.Tensor]:
    return [torch.sin(x), torch.cos(x)]


sincos_kernel = TorchMetalKernel(
    "sincos",
    input_names=["x"],
    result_names=["out_sin", "out_cos"],
    src="out_sin[id] = sin(x[id]); out_cos[id] = cos(x[id]);",
    torch_defn=torch_sincos,
    metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
)


class SinCosModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = sincos_kernel(
            x,
            threads_per_grid=(x.shape[0], 1, 1),
            threads_per_thread_group=(1, 1, 1),
            result_shapes=[list(x.shape), list(x.shape)],
        )
        return results[0] + results[1]  # sin(x) + cos(x)
```

## Next Steps

- [Custom Op Lowering](custom-op-lowering.md) — a simpler alternative for ops that can be expressed using standard Core AI operations.
- [TorchMetalKernel](../api/TorchMetalKernel.md) — full `TorchMetalKernel` API reference.
- [TorchConverter API reference](../api/TorchConverter.md) — `register_custom_kernels` API reference.

## Notices

PyTorch is a trademark of Meta Platforms, Inc.
