# linalg_vector_norm

Computes the vector $p$-norm along one or more axes:

$$\lVert x \rVert_p = \left( \sum_i \lvert x_i \rvert^{p} \right)^{1/p}$$

For `ord = +inf` / `-inf`, the sum is replaced by max / min over the reduced axes.

**ATen source:** `aten.linalg_vector_norm`

## Inputs

| Name | Description |
|---|---|
| `input` | Input tensor |

## Attributes

| Name | Type | Description |
|---|---|---|
| `ord` | `int` / `float` | Order of the norm. `+inf` / `-inf` use max / min instead of the sum |
| `axes` | `list[int]` | Dims to reduce. `None` reduces over all dims |
| `keep_dim` | `bool` | If `True`, reduced dims remain with size 1; if `False`, they are removed |
| `version` | `int` | Composite op version |

## Output

| Name | Description |
|---|---|
| `output` | Input shape with reduced dims either set to `1` (`keep_dim=True`) or removed (`keep_dim=False`) |

## PyTorch example

```python
import torch

input = torch.randn(8, 128, 256, 1024)
# Note: the PyTorch arg is `dim`; Core AI's IR attribute is `axes`
output = torch.linalg.vector_norm(input, ord=-1.5, dim=[1, 3], keepdim=False)
# Output shape: (8, 256)
```

## Reference

[`torch.linalg.vector_norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html)
