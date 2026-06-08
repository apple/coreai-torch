# instance_norm

Normalizes each `(sample, channel)` slice independently across its spatial dims — mean and variance are computed per-sample rather than across the batch.

**ATen source:** `aten.instance_norm` (preserved as composite by `get_decomp_table()`)

## Inputs

| Name | Shape | Description |
|---|---|---|
| `input` | `(N, C, *spatial)` | 1, 2, or 3 spatial dims (e.g., `(N, C, H, W)` for 2D) |
| `gamma` | `(C,)` | Per-channel scale applied after normalization |
| `beta` | `(C,)` | Per-channel shift applied after the scale |

## Attributes

| Name | Type | Description |
|---|---|---|
| `eps` | `float` | Numerical-stability epsilon |
| `version` | `int` | Composite op version |

## Output

| Name | Shape | Description |
|---|---|---|
| `output` | `(N, C, *spatial)` | Same shape as `input` |

## Data types

`fp16`, `fp32`.

## PyTorch example

```python
import torch

N, C, H, W = 2, 6, 10, 10
input = torch.randn(N, C, H, W)
gamma = torch.randn(C)
beta = torch.randn(C)

output = torch.ops.aten.instance_norm.default(
    input, gamma, beta,
    None, None,        # running_mean / running_var unused in inference
    True,              # use_input_stats
    0.1,               # momentum (ignored in inference)
    1e-5,              # eps
    True,              # cudnn_enabled (ignored)
)
```

## Reference

[`torch.nn.InstanceNorm2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html)
