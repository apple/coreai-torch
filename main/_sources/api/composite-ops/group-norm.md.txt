# group_norm

Splits the channel dimension into `num_groups` groups and normalizes each group independently across spatial dims. `num_channels` must be divisible by `num_groups`.

**ATen source:** `aten.native_group_norm`

## Inputs

| Name | Shape | Description |
|---|---|---|
| `input` | `(N, C, *spatial)` | Tensor to normalize |
| `weight` | `(C,)` | Per-channel scale applied after normalization |
| `bias` | `(C,)` | Per-channel shift applied after the scale |

## Attributes

| Name | Type | Description |
|---|---|---|
| `num_groups` | `int` | Number of groups to split channels into |
| `num_channels` | `int` | Number of channels in the input (must be divisible by `num_groups`) |
| `eps` | `float` | Numerical-stability epsilon |
| `version` | `int` | Composite op version |

## Output

| Name | Shape | Description |
|---|---|---|
| `output` | `(N, C, *spatial)` | Same shape as `input` |

## PyTorch example

```python
import torch

N, C, H, W = 2, 6, 10, 10
input = torch.randn(N, C, H, W)
weight = torch.randn(C)
bias = torch.randn(C)

output = torch.group_norm(input, num_groups=2, weight=weight, bias=bias, eps=1e-5)
```

## Reference

[`torch.nn.GroupNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)
