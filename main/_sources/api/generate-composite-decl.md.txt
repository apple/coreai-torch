# generate_composite_decl

Generates the composite declaration attribute required when implementing custom op lowerings.

```{warning}
Implementing custom op lowerings uses authoring APIs from `coreai-core` (such as `coreai._compiler.dialects`). The leading underscore on `_compiler` marks this as private upstream API — it may move or change without notice across `coreai-core` releases.
```

## Overview

When lowering a custom op to a `coreai.graph` composite, the graph must carry a `CompositeDeclaration` attribute that specifies the composite name, input and output names, and any op-specific attributes. `generate_composite_decl` builds this attribute from its arguments.

**Public import:**

```python
from coreai_torch import generate_composite_decl
```

## Declaration

```python
def generate_composite_decl(
    context,
    op_name: str,
    input_names: list[str],
    output_names: list[str],
    op_attributes: dict[str, Any],
) -> CompositeDeclaration
```

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `context` | `Context` | The Core AI Context. Pass `value.context` from any `Value` in scope. |
| `op_name` | `str` | Name of the composite op as it appears in the `composite_decl` attribute. |
| `input_names` | `list[str]` | Names for the composite inputs, in argument order. |
| `output_names` | `list[str]` | Names for the composite outputs, in result order. |
| `op_attributes` | `dict[str, Any]` | Op-specific attributes to embed in the declaration. The `"version"` key is required; pass `1` for the current API version. |

## Returns

A `CompositeDeclaration` Core AI attribute to pass as `composite_decl` when defining a `coreai.graph`.

## Discussion

Use this function inside a lowering function registered with `TorchConverter.register_torch_lowering()`.

The following example registers a lowering for a custom op `"mylib::my_custom_op"`:

```python
from coreai_torch import generate_composite_decl, TorchConverter


def my_custom_op_conversion(values_map, node, loc):
    arg0 = values_map[node.args[0].name]
    arg1 = values_map[node.args[1].name]
    op_attributes = {
        "some_attribute": 0.5,
        "version": 1,
    }
    composite_decl = generate_composite_decl(
        arg0.context,
        "my_custom_op",
        ["argument0", "argument1"],
        ["output"],
        op_attributes,
    )

    # The decorator transforms this function: calling it returns an OpResultList
    @coreai.graph(no_inline=True, composite_decl=composite_decl)
    def my_custom_op_impl(argument0: Value, argument1: Value) -> Value:
        ...
        return result

    return my_custom_op_impl(arg0, arg1)[0]


converter = TorchConverter()
converter.register_torch_lowering("mylib::my_custom_op")(my_custom_op_conversion)
```

The `coreai.graph` decorator always returns an `OpResultList`. Index it at `[0]` when the composite produces a single output.

For a full walkthrough of custom op lowering, see {doc}`../guides/custom-op-lowering`.
