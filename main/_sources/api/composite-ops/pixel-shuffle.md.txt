# pixel_shuffle

Rearranges elements in a tensor of shape $(*, C \cdot r^2, H, W)$ into shape $(*, C, H \cdot r, W \cdot r)$, where $r$ is the upscale factor. The input's channel dimension must be divisible by $r^2$.

**ATen source:** `aten.pixel_shuffle` (preserved as composite by `get_decomp_table()`)

## Inputs

| Name | Description |
|---|---|
| `input` | Tensor with at least 3 dims; size of dim `-3` must be divisible by `upscale_factor ** 2` |

## Attributes

| Name | Type | Description |
|---|---|---|
| `upscale_factor` | `int` | Factor by which to increase spatial resolution |
| `version` | `int` | Composite op version |

## Output

| Name | Description |
|---|---|
| `output` | Same leading dims; channel divided by $r^2$, spatial dims multiplied by $r$ |

## PyTorch example

```python
import torch

N, C, H, W = 8, 1000, 32, 32
input = torch.randn(N, C, H, W)
# Output shape: (8, 10, 320, 320)
output = torch.nn.functional.pixel_shuffle(input, upscale_factor=10)
```

## Reference

[`torch.nn.PixelShuffle`](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html)
