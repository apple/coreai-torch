# TorchConverter API reference

A converter that produces a Core AI `AIProgram` from a PyTorch exported program.

## Overview

```python
from coreai_torch import TorchConverter, get_decomp_table
```

`TorchConverter` traverses the exported FX graph of a PyTorch model, maps ATen operators to Core AI operations, and emits an `AIProgram` ready for on-device execution.

Load a model using `add_exported_program()` or `add_pytorch_module()`, then call `to_coreai()` to produce the artifact. Register custom lowerings with `register_torch_lowering()` for ops that have no built-in mapping, and supply Metal kernel source for compute-intensive operations with `register_custom_kernels()`.

---

## Constructor

```python
TorchConverter()
```

Creates a `TorchConverter` instance with no loaded programs and no custom lowerings.

**Example:**

```python
import torch
from coreai_torch import TorchConverter

converter = TorchConverter()
```

---

## Methods

### `add_exported_program`

```python
def add_exported_program(
    self,
    exported_program: ExportedProgram,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    state_names: Sequence[str] | None = None,
    entrypoint_name: str = "main",
) -> TorchConverter
```

Loads an exported PyTorch program into the converter. Returns `self` for method chaining.

```{warning}
The caller **must** call `run_decompositions()` on the program before passing it here — use `get_decomp_table()` to preserve known composite ops in the lowered IR.
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `exported_program` | `torch.export.ExportedProgram` | — | The exported PyTorch program to convert. |
| `input_names` | `Sequence[str] \| None` | `None` | Names for non-stateful user inputs only. Mutated inputs (buffers and user-input mutations) are renamed via `state_names`. **Breaking change**: previously this covered all graph inputs. |
| `output_names` | `Sequence[str] \| None` | `None` | Names for non-state outputs (return values) only. Mutation outputs are renamed via `state_names`. **Breaking change**: previously this covered all graph outputs. |
| `state_names` | `Sequence[str] \| None` | `None` | Names for stateful IO. Each name applies to both the state input and its corresponding mutation output — matching the Core AI runtime's single-name-per-state model. See [IO Naming](#io-naming) below. |
| `entrypoint_name` | `str` | `"main"` | Name of the entry-point graph in the generated Core AI module. Must be unique across all staged programs. |

**Returns:** `self` — the same `TorchConverter` instance, for chaining.

**Example:**

```python
import torch
from coreai_torch import TorchConverter, get_decomp_table

model = MyModel()
exported = torch.export.export(model, example_inputs)
exported = exported.run_decompositions(get_decomp_table())

converter = TorchConverter().add_exported_program(
    exported,
    input_names=["image"],
    output_names=["logits"],
)
coreai_program = converter.to_coreai()
coreai_program.optimize()
```

---

### `add_pytorch_module`

```python
def add_pytorch_module(
    self,
    module: nn.Module,
    export_fn: Callable[[nn.Module], ExportedProgram],
    externalize_modules: list | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    state_names: Sequence[str] | None = None,
    entrypoint_name: str = "main",
) -> TorchConverter
```

Stages an `nn.Module` for conversion. `export_fn` must return a **decomposed** `ExportedProgram` (i.e., with `run_decompositions()` already called). Returns `self` for method chaining.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `module` | `nn.Module` | — | The PyTorch module to convert. |
| `export_fn` | `Callable[[nn.Module], ExportedProgram]` | — | A callable that exports and decomposes the module. Must call `run_decompositions()` on the result (e.g., `lambda m: torch.export.export(m, args=sample).run_decompositions(coreai_torch.get_decomp_table())`). |
| `externalize_modules` | `list \| None` | `None` | Optional list of `ExternalizeSpec` objects describing submodule classes to preserve as composite ops. Bare class types are also accepted for *experimental* simple externalization. |
| `input_names` | `Sequence[str] \| None` | `None` | Names for non-stateful user inputs only. Mutated inputs (buffers and user-input mutations) are renamed via `state_names`. **Breaking change**: previously this covered all graph inputs. |
| `output_names` | `Sequence[str] \| None` | `None` | Names for non-state outputs (return values) only. Mutation outputs are renamed via `state_names`. **Breaking change**: previously this covered all graph outputs. |
| `state_names` | `Sequence[str] \| None` | `None` | Names for stateful IO. See [IO Naming](#io-naming) below. |
| `entrypoint_name` | `str` | `"main"` | Name of the entry-point graph in the generated Core AI module. Must be unique across all staged programs. |

**Returns:** `self` — the same `TorchConverter` instance, for chaining.

**Example:**

```python
model = MyModel().eval()
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

---

### `to_coreai`

```python
def to_coreai(
    self, *, entrypoints: Sequence[str] | None = None
) -> AIProgram
```

Converts the loaded program into a Core AI `AIProgram`. This is the main entry point for conversion. The model must have been loaded via `add_exported_program()` or `add_pytorch_module()` before calling this method.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `entrypoints` | `Sequence[str] \| None` | `None` | If provided, only convert programs with these entrypoint names. If `None`, convert all staged programs. |

**Returns:** `AIProgram` — a compiled Core AI model ready for deployment and execution.

**Example:**

```python
converter = TorchConverter().add_exported_program(
    exported,
    input_names=["image"],
    output_names=["logits"],
)
coreai_program = converter.to_coreai()
coreai_program.optimize()
```

---

### `clear`

```python
def clear(self, *, entrypoints: Sequence[str] | None = None) -> None
```

Remove staged programs. Custom lowerings registered via `register_torch_lowering()` are always preserved.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `entrypoints` | `Sequence[str] \| None` | `None` | If provided, remove only programs with these entrypoint names. If `None`, remove all staged programs. |

**Example:**

```python
converter = TorchConverter()
converter.add_exported_program(ep_a, entrypoint_name="encoder")
converter.add_exported_program(ep_b, entrypoint_name="decoder")

# Remove just the encoder
converter.clear(entrypoints=["encoder"])

# Remove everything
converter.clear()
```

---

### `register_torch_lowering`

```python
def register_torch_lowering(
    self,
    qualified_name: str,
    allow_override: bool = False,
) -> Callable
```

Decorator that registers a custom lowering function for an FX node. Use this to provide a Core AI implementation for a custom or unsupported torch operator.

```{warning}
Lowering functions are written against authoring APIs from `coreai-core` (such as `coreai._compiler.dialects`). The leading underscore on `_compiler` marks this as private upstream API — it may move or change without notice across `coreai-core` releases.
```


| Parameter | Type | Default | Description |
|---|---|---|---|
| `qualified_name` | `str` | — | Op name in `"namespace::op_name"` form, matching the torch op's qualified name (e.g. `"my_lib::my_op"`). |
| `allow_override` | `bool` | `False` | If `True`, silently replaces an existing lowering for the same op. If `False`, raises on conflict. |

**Returns:** A decorator function. The decorated function must have the signature:

```python
def lowering_func(
    values_map: dict[str, Value],
    node: torch.fx.Node,
    loc: Location,
) -> Value | list[Value]
```

| Callback Parameter | Type | Description |
|---|---|---|
| `values_map` | `dict[str, Value]` | Map of FX node names to their corresponding Core AI `Value`s. Use this to look up operands. |
| `node` | `torch.fx.Node` | The FX node being lowered. Access `node.args` for operands and `node.meta` for metadata. |
| `loc` | `Location` | The Core AI Location for the current operation. Pass this to Core AI op constructors. |

**Raises:**

- `ValueError` if `qualified_name` is not in `"namespace::op_name"` form.
- `ValueError` if the namespace is reserved (`aten`, `higher_order`, `coreai`, `coreaix`).
- `ValueError` if a lowering already exists for the op and `allow_override` is `False`.

**Example — register a lowering for a custom torch op:**

```python
import torch
from coreai._compiler.dialects import coreai
from coreai_torch._utils import get_operands

@torch.library.custom_op("my_lib::scaled_add", mutates_args=())
def scaled_add(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    return x + scale * y

@scaled_add.register_fake
def _(x, y, scale):
    return torch.empty_like(x)

converter = TorchConverter()

@converter.register_torch_lowering("my_lib::scaled_add.default")
def lower_scaled_add(values_map, node, loc):
    x, y = get_operands(values_map, node, [0, 1], loc)
    scale = node.args[2]  # plain Python float
    scale_val = coreai.constant(scale, dtype=x.type.element_type)
    scaled_y = coreai.broadcasting_mul(y, scale_val, loc=loc)
    return coreai.broadcasting_add(x, scaled_y, loc=loc)

coreai_program = converter.add_exported_program(exported).to_coreai()
coreai_program.optimize()
```

**Example — override a built-in ATen op lowering:**

```python
import numpy as np
from coreai._compiler.dialects import coreai
from coreai_torch._utils import get_operand

converter = TorchConverter()

@converter.register_torch_lowering("aten::_adaptive_avg_pool2d.default", allow_override=True)
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

coreai_program = converter.add_exported_program(exported).to_coreai()
coreai_program.optimize()
```

---

### `register_custom_kernels`

```python
def register_custom_kernels(
    self,
    kernels: Sequence[TorchMetalKernel],
) -> TorchConverter
```

Registers one or more `TorchMetalKernel` objects so the converter can convert them into Core AI operations during conversion. Call this **before** `add_exported_program()`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `kernels` | `Sequence[TorchMetalKernel]` | — | The custom Metal kernels used in the model. |

**Returns:** `self` — the same `TorchConverter` instance, for chaining.

**Example:**

```python
import torch
from coreai_torch import TorchConverter, TorchMetalKernel, MetalParameter, get_decomp_table


def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


kernel = TorchMetalKernel(
    "vector_add",
    input_names=["x", "y"],
    result_names=["output"],
    src="output[id] = x[id] + y[id];",
    torch_defn=torch_add,
    metal_params=[MetalParameter("id", "uint", "thread_position_in_grid")],
)

converter = TorchConverter()
converter.register_custom_kernels([kernel])

exported = torch.export.export(model, args=example_inputs)
exported = exported.run_decompositions(get_decomp_table())
coreai_program = converter.add_exported_program(exported).to_coreai()
coreai_program.optimize()
```

See the {doc}`../guides/custom-metal-kernels` guide for a full walkthrough.

---

## Standalone functions

### `get_decomp_table`

```python
from coreai_torch import get_decomp_table

def get_decomp_table() -> dict
```

Returns the default PyTorch ATen decomposition table minus the operations that `TorchConverter` lowers as composite ops, so those operations are preserved in the exported graph rather than being decomposed into lower-level primitives.

Each call returns a fresh copy of the table — mutating it does not affect other callers.

**Returns:** `dict` — a decomposition table suitable for `ExportedProgram.run_decompositions()`.

**With `add_exported_program`** (caller runs decompositions manually):

```python
import torch
from coreai_torch import TorchConverter, get_decomp_table

ep = torch.export.export(model, args=example_inputs)
ep = ep.run_decompositions(get_decomp_table())
coreai_program = TorchConverter().add_exported_program(ep).to_coreai()
coreai_program.optimize()
```

**With `add_pytorch_module`**:

```python
from coreai_torch import TorchConverter

coreai_program = (
    TorchConverter()
    .add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=example_inputs).run_decompositions(
            coreai_torch.get_decomp_table()
        ),
    )
    .to_coreai()
)
coreai_program.optimize()
```

---

## IO naming

The `input_names`, `output_names`, and `state_names` parameters control the names of the graph's inputs and outputs. These names propagate to the Core AI runtime descriptor (`desc.input_names`, `desc.output_names`, `desc.state_names`).

### Parameters

| Parameter | Controls | Scope |
|-----------|----------|-------|
| `input_names` | Non-stateful `forward()` arguments | Inputs that are read but not mutated |
| `output_names` | Return values from `forward()` | Outputs that are not mutation side-effects |
| `state_names` | Mutable state (buffers + mutated user inputs) | A single name per state, applied to both the input and its corresponding mutation output |

```{warning}
**What counts as state (no opt-out).** The converter treats two things as state:

1. **Mutable buffers** registered via `self.register_buffer(...)` and mutated in-place inside `forward()` (e.g., `self.buf.add_(x)`).
2. **User inputs mutated in-place** inside `forward()` (e.g., `x.mul_(2)` on a `forward()` arg).

Both are detected from the exported program's graph signature. There is **no flag** to opt a mutated user input out of state — it follows directly from how the model mutates its inputs and buffers.

If you don't want a `forward()` argument treated as state, eliminate the in-place mutation from your model — clone first (`x_local = x.clone(); x_local.mul_(2)`) or use the out-of-place form (`x_scaled = x * 2`). The mutation then disappears from the exported graph and the argument becomes a regular non-state input.
```

### Default names

When a naming parameter is not provided, the converter uses names derived from the FX graph:

| Category | FX graph source | Relates to | Example |
|----------|----------------|------------|---------|
| **Input** | Placeholder `node.name` | `forward()` arg name | `def forward(self, x, z)` &rarr; `"x"`, `"z"` |
| **Output** | Output node's input `node.name` | Internal op name | `return a + b, c * d` &rarr; `"add"`, `"mul"` |
| **State (buffer)** | Placeholder `node.name` | `"b_"` + `register_buffer` attr | `self.register_buffer("kv_cache", ...)` &rarr; `"b_kv_cache"` |
| **State (mutated user input)** | Placeholder `node.name` | `forward()` arg name | `def forward(self, y): y.mul_(2)` &rarr; `"y"` |

```{warning}
These naming conventions are observed behavior from the FX graph, not a stable contract from PyTorch. They may change across PyTorch versions. Always provide explicit names for production use.
```

### Ordering

| Parameter | Order follows |
|-----------|--------------|
| `input_names` | Non-mutated `forward()` args in signature order |
| `output_names` | `return` values in tuple order |
| `state_names` | Mutable buffers (`register_buffer` registration order), then mutated user inputs (`forward()` signature order) |

```{warning}
The ordering of `state_names` (buffers first, then mutated user inputs) is based on observed FX graph behavior, not a stable PyTorch contract. The converter asserts that the number of state inputs matches state outputs, but cannot detect silent reordering. Always verify state ordering when upgrading PyTorch versions.
```

### Examples

**Stateless model** — only `input_names` and `output_names` apply:

```python
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)

ep = torch.export.export(Linear().eval(), args=(torch.randn(1, 8),))
ep = ep.run_decompositions(get_decomp_table())

TorchConverter().add_exported_program(
    ep,
    input_names=["features"],
    output_names=["logits"],
).to_coreai().optimize()
```

**Stateful model** — all three parameters:

```python
class KVCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kv_cache", torch.zeros(1, 4))   # state[0]
        self.register_buffer("pos_idx", torch.zeros(1))       # state[1]

    def forward(self, x, y, z):
        self.kv_cache.add_(x)       # buffer mutation
        self.pos_idx.add_(1)        # buffer mutation
        y.mul_(2)                   # state[2]: mutated user input
        # non-mutated: x -> input[0], z -> input[1]
        return self.kv_cache + y, z * 3

ep = torch.export.export(
    KVCache().eval(),
    args=(torch.randn(1, 4), torch.randn(1, 4), torch.randn(1, 4)),
)
ep = ep.run_decompositions(get_decomp_table())

TorchConverter().add_exported_program(
    ep,
    state_names=["kv_cache", "pos_idx", "y_state"],
    input_names=["query", "context"],
    output_names=["attn_out", "scaled"],
).to_coreai().optimize()
```

The converter applies these names to the graph's inputs and outputs: mutated buffers and mutated user inputs become model state (in `state_names` order), followed by the non-mutated `input_names` and the returned `output_names`.

## Notices

PyTorch is a trademark of Meta Platforms, Inc.
