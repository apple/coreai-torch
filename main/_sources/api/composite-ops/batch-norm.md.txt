# batch_norm

Inference-time batch normalization using running statistics:

$$y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta$$

The mean and variance are pre-computed running statistics passed in as inputs; `momentum` (a training-only construct) is dropped during conversion.

**ATen source:** `aten._native_batch_norm_legit_no_training`

## Inputs

| Name | Shape | Description |
|---|---|---|
| `input` | `(N, C, *spatial)` | Supported ranks: 2, 3, 4, 5 — `(N, C)`, `(N, C, L)`, `(N, C, H, W)`, `(N, C, D, H, W)` |
| `gamma` | `(C,)` | Per-channel scale, applied after normalization |
| `beta` | `(C,)` | Per-channel shift, added after the scale |
| `mean` | `(C,)` | Per-channel running mean |
| `variance` | `(C,)` | Per-channel running variance |

## Attributes

| Name | Type | Description |
|---|---|---|
| `eps` | `float` | Numerical-stability epsilon added to the variance |
| `version` | `int` | Composite op version |

## Output

| Name | Shape | Description |
|---|---|---|
| `output` | `(N, C, *spatial)` | Same shape as `input` |

## Data types

`fp16`, `fp32`, `bf16` for all tensor inputs and the output.

## PyTorch example

```python
import torch

N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)
running_mean = torch.zeros(C)
running_var = torch.ones(C)

output = torch.ops.aten._native_batch_norm_legit_no_training(
    input,
    weight=torch.ones(C),
    bias=torch.zeros(C),
    running_mean=running_mean,
    running_var=running_var,
    momentum=0.1,
    eps=1e-5,
)
```

## Reference

[`torch.nn.BatchNorm2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
