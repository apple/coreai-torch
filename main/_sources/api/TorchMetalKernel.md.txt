# TorchMetalKernel

Wraps a Metal GPU kernel as a PyTorch custom op so it traces through `torch.export` and is converted into a Core AI operation during conversion.

```{warning}
Authoring Metal kernels uses APIs from `coreai-core` (such as `coreai.authoring`). These APIs are experimental and subject to change in future releases.
```

## Public import

```python
from coreai_torch import TorchMetalKernel, MetalParameter
```

`MetalParameter` is re-exported from `coreai.authoring` for convenience and is used to declare Metal thread attributes (e.g. `thread_position_in_grid`).

For a tutorial walkthrough, see {doc}`../guides/custom-metal-kernels`.

## Constructor

```python
TorchMetalKernel(
    name: str,
    input_names: list[str],
    result_names: list[str],
    src: str,
    torch_defn: Callable[..., Any],
    metal_params: list[MetalParameter] | None = None,
    helper_src: str | None = None,
    template_dtypes: dict[str, str] | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Kernel identifier. Becomes part of the generated kernel's name in the converted model. |
| `input_names` | `list[str]` | — | Names matching the input variables in the Metal source. Must match the parameter count of `torch_defn`. |
| `result_names` | `list[str]` | — | Names matching the output variables in the Metal source. |
| `src` | `str` | — | Body of the Metal `[[kernel]]` function. The signature, buffer bindings, and `#include <metal_stdlib>` are generated automatically from `input_names`, `result_names`, and `metal_params`. |
| `torch_defn` | `Callable` | — | Reference PyTorch implementation used for shape inference during `torch.export`. See [Constraints](#constraints) below. |
| `metal_params` | `list[MetalParameter] \| None` | `None` | Metal thread attributes to bind in the generated kernel signature (e.g. `MetalParameter("id", "uint", "thread_position_in_grid")`). |
| `helper_src` | `str \| None` | `None` | Additional Metal source pasted before the kernel definition (helper functions, type aliases, etc.). |
| `template_dtypes` | `dict[str, str] \| None` | `None` | Map from input name to a placeholder string in `src`. Each placeholder is replaced with the corresponding Metal dtype at compile time, allowing one kernel to serve multiple dtypes. |

## Calling the kernel

```python
def __call__(
    self,
    *args,
    threads_per_grid: tuple[int, int, int],
    threads_per_thread_group: tuple[int, int, int],
    result_shapes: list[list[int]],
)
```

| Argument | Type | Description |
|---|---|---|
| `*args` | tensors / scalars | Positional arguments matching `input_names` and the signature of `torch_defn`. |
| `threads_per_grid` | `tuple[int, int, int]` | Total Metal grid dimensions. |
| `threads_per_thread_group` | `tuple[int, int, int]` | Threadgroup dimensions. |
| `result_shapes` | `list[list[int]]` | Shape of each result tensor, in the order of `result_names`. |

Returns a `torch.Tensor`, `list[torch.Tensor]`, or `tuple[torch.Tensor, ...]` matching the return annotation of `torch_defn`.

## Constraints

`torch_defn` must satisfy two rules:

1. **Inputs** — every parameter must be annotated as `torch.Tensor`, `int`, `float`, or `bool`. The parameter count must match `len(input_names)`.
2. **Return** — the return annotation must be `torch.Tensor`, `list[torch.Tensor]`, or `tuple[torch.Tensor, ...]` (with a concrete number of tuple members).

Violations raise `TypeError` (input/return annotations) or `ValueError` (parameter count mismatch) at construction time.

## Registering with the converter

`TorchMetalKernel` instances must be registered with the converter via `register_custom_kernels()` before `add_exported_program()`:

```python
converter = TorchConverter()
converter.register_custom_kernels([custom_add])
converter.add_exported_program(exported, ...)
```

See {doc}`TorchConverter` for `register_custom_kernels` details.

## Example

```python
import torch
from coreai_torch import TorchMetalKernel, MetalParameter


def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


custom_add = TorchMetalKernel(
    name="vector_add",
    input_names=["x", "y"],
    result_names=["output"],
    src="output[id] = x[id] + y[id];",
    torch_defn=torch_add,
    metal_params=[
        MetalParameter("id", "uint", "thread_position_in_grid"),
    ],
)
```

Use it inside an `nn.Module`:

```python
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

## Dtype templating

Use `template_dtypes` to write one kernel that compiles for multiple dtypes:

```python
custom_matmul = TorchMetalKernel(
    name="matmul",
    input_names=["A", "B"],
    result_names=["C"],
    src="""
        TYPE sum = 0.0f;
        ...
    """,
    torch_defn=torch_matmul,
    metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
    # The dtype of input "A" determines what "TYPE" is replaced with at compile time.
    template_dtypes={"A": "TYPE"},
)
```

Every occurrence of `"TYPE"` in `src` is replaced with the Metal type matching the dtype of input `A` (e.g. `half`, `float`, `bfloat`).

## Multiple outputs

`torch_defn` may return a `list[torch.Tensor]` or `tuple[torch.Tensor, ...]`; supply one entry in `result_shapes` per output:

```python
def torch_sincos(x: torch.Tensor) -> list[torch.Tensor]:
    return [torch.sin(x), torch.cos(x)]


sincos = TorchMetalKernel(
    name="sincos",
    input_names=["x"],
    result_names=["out_sin", "out_cos"],
    src="out_sin[id] = sin(x[id]); out_cos[id] = cos(x[id]);",
    torch_defn=torch_sincos,
    metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
)

# call site
results = sincos(
    x,
    threads_per_grid=(x.shape[0], 1, 1),
    threads_per_thread_group=(1, 1, 1),
    result_shapes=[list(x.shape), list(x.shape)],
)
```

## Notices

PyTorch is a trademark of Meta Platforms, Inc. Metal is a trademark of Apple Inc.
