# layer_norm

Normalizes over the last `D` dimensions (specified via `axes`); mean and variance are computed across those axes for each remaining slice.

**ATen source:** `aten.native_layer_norm`

## Inputs

| Name | Shape | Description |
|---|---|---|
| `input` | `(*batch, *D)` | Tensor to normalize |
| `gamma` | matches normalized dims | Scale applied to the normalized tensor |
| `beta` | matches normalized dims | Shift added after the scale |

## Attributes

| Name | Type | Description |
|---|---|---|
| `axes` | `list[int]` | Dimensions over which mean/variance are computed (the trailing `D` dims) |
| `eps` | `float` | Numerical-stability epsilon |
| `version` | `int` | Composite op version |

## Output

| Name | Shape | Description |
|---|---|---|
| `output` | `(*batch, *D)` | Same shape as `input` |

## Data types

`fp16`, `fp32`, `bf16`.

## PyTorch example

```python
import torch
from torch.nn.functional import layer_norm

N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)

# Normalize over the last three dims (C, H, W)
output = layer_norm(input, normalized_shape=[C, H, W], weight=None, bias=None, eps=1e-5)
```

## Reference

[`torch.nn.LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
