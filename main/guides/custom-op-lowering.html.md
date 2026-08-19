# Custom Op Lowering

`TorchConverter` maps standard ATen ops to CoreAI graphs automatically. This guide shows how to register lowerings for custom or unsupported ops.

#### WARNING
Writing a lowering uses authoring APIs from `coreai-core` (such as `coreai._compiler.dialects`). The leading underscore on `_compiler` marks this as private upstream API — it may move or change without notice across `coreai-core` releases.

## When You Need This

- Your model calls a custom torch op (registered via `torch.library`) that the converter does not recognize.
- You want to replace the built-in lowering for a standard ATen op with a specialized implementation.

---

## Lowering a custom torch Op

### Step 1 — Define the custom torch Op

Use `@torch.library.custom_op` to register the op with the PyTorch dispatcher, and `register_fake` to provide the abstract implementation needed by `torch.export`.

```ipython3
import torch


@torch.library.custom_op("my_lib::scaled_add", mutates_args=())
def scaled_add(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    """Eager implementation: runs on CPU during normal PyTorch inference."""
    return x + scale * y


@scaled_add.register_fake
def _(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    """Abstract implementation: called by torch.export to infer output shapes."""
    return torch.empty_like(x)
```

The eager body runs during regular PyTorch execution. The `register_fake` body is called by the exporter’s shape-propagation machinery; it only needs to return a tensor with the correct shape and dtype.

### Step 2 — Use the Op in a model

```ipython3
import torch
import torch.nn as nn


class ScaledAddModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.my_lib.scaled_add(x, y, 0.5)
```

### Step 3 — Export the model

Export and run decompositions as usual. Custom ops are not touched by the decomposition pass, so `my_lib::scaled_add` remains as a single node in the graph.

```ipython3
import torch

from coreai_torch import get_decomp_table

model = ScaledAddModel().eval()
example_inputs = (torch.randn(4, 8), torch.randn(4, 8))

exported = torch.export.export(model, args=example_inputs)
exported = exported.run_decompositions(get_decomp_table())
```

### Step 4 — Register the CoreAI lowering

Create a `TorchConverter`, then use `register_torch_lowering` as a decorator. The lowering function receives:

| Argument     | Type               | Description                                                                                        |
|--------------|--------------------|----------------------------------------------------------------------------------------------------|
| `values_map` | `dict[str, Value]` | Maps FX node names to their CoreAI `Value`s. Use this to look up tensor operands.                  |
| `node`       | `torch.fx.Node`    | The FX node being lowered. Tensor args are `fx.Node` objects; scalar args are plain Python values. |
| `loc`        | `Location`         | CoreAI Location. Pass to CoreAI op constructors.                                                   |

The op’s qualified name in the FX graph always carries the overload suffix `.default`, so register it as `"my_lib::scaled_add.default"`.

```ipython3
from coreai._compiler.dialects import coreai

from coreai_torch import TorchConverter
from coreai_torch._utils import get_operands

converter = TorchConverter()


@converter.register_torch_lowering("my_lib::scaled_add.default")
def lower_scaled_add(values_map, node, loc):
    x, y = get_operands(values_map, node, [0, 1], loc)
    scale = node.args[2]  # plain Python float

    scale_val = coreai.constant(scale, dtype=x.type.element_type)
    scaled_y = coreai.broadcasting_mul(y, scale_val, loc=loc)
    return coreai.broadcasting_add(x, scaled_y, loc=loc)
```

### Step 5 — Convert

```ipython3
coreai_program = converter.add_exported_program(
    exported,
    input_names=["x", "y"],
    output_names=["result"],
).to_coreai()
coreai_program.optimize()
```

### Step 6 — Inspect the generated CoreAI graph

```ipython3
print(str(coreai_program))
```

---

## Overriding built-in lowerings

To replace the built-in lowering for a standard ATen op, pass `allow_override=True`. This is useful when you know your model’s runtime constraints allow a simpler implementation.

For example, the default lowering for `aten._adaptive_avg_pool2d` handles dynamic input shapes and non-divisible output sizes. If your model always runs with static shapes and an output size that evenly divides the input (e.g., ResNet’s final `adaptive_avg_pool2d(output_size=(1, 1))`), you can replace it with a simpler `sumpool2d` + divide:

```ipython3
import numpy as np
import torch
import torch.nn as nn

from coreai_torch._utils import get_operand


# A model that uses adaptive_avg_pool2d
class PoolModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))


pool_model = PoolModel().eval()
pool_input = (torch.randn(1, 3, 8, 8),)
pool_exported = torch.export.export(pool_model, args=pool_input)
pool_exported = pool_exported.run_decompositions(get_decomp_table())

converter = TorchConverter()


@converter.register_torch_lowering(
    "aten::_adaptive_avg_pool2d.default", allow_override=True
)
def lower_adaptive_avg_pool2d_static(values_map, node, loc):
    x = get_operand(values_map, node, 0, loc)
    output_h, output_w = node.args[1]
    input_h, input_w = x.type.shape[2], x.type.shape[3]
    stride_h, stride_w = input_h // output_h, input_w // output_w
    kernel_h = input_h - (output_h - 1) * stride_h
    kernel_w = input_w - (output_w - 1) * stride_w
    return coreai.broadcasting_divide(
        coreai.sumpool2d(
            x,
            kernel_size=np.array([kernel_h, kernel_w], dtype=np.uint32),
            strides=np.array([stride_h, stride_w], dtype=np.uint32),
            dilation=coreai.constant([1, 1], dtype=np.uint32),
        ),
        coreai.cast(float(kernel_h * kernel_w), x.type.element_type),
    )


coreai_program = converter.add_exported_program(pool_exported).to_coreai()
coreai_program.optimize()
```

---

## Notes

- **Op name format.** The qualified name must be `"namespace::op_name.overload"`. Custom ops defined with `@custom_op` always use the `.default` overload.
- **Reserved namespaces.** `aten`, `higher_order`, `coreai`, and `coreaix` are built-in. Overriding requires `allow_override=True`.
- **Per-instance registration.** Lowerings are stored on the `TorchConverter` instance and do not affect other converters or the global resolver tables.
- **Multiple return values.** If your op returns a tuple, return a Python list of `Value`s from the lowering function. The converter stores them as `"node_name#0"`, `"node_name#1"`, etc. in `values_map`.

## Next Steps

- [Supported ATen ops](../api/supported-aten-ops.md) — check the full built-in op coverage before writing a custom lowering.
- [Composite Ops Guide](composite-ops.md) — use `generate_composite_decl` to write composite op lowerings that dispatch to hardware-optimized kernels.
- [TorchConverter API reference](../api/TorchConverter.md) — `register_torch_lowering` API reference.

## Notices

PyTorch is a trademark of Meta Platforms, Inc.
