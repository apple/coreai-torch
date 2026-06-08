# log_softmax

Computes $\log(\text{softmax}(x))$ along a single dim.

**ATen source:** `aten._log_softmax`

## Inputs

| Name | Description |
|---|---|
| `input` | Tensor of any rank |

## Attributes

| Name | Type | Description |
|---|---|---|
| `axis` | `int` | Dimension along which `log_softmax` is computed |
| `version` | `int` | Composite op version |

## Output

| Name | Shape | Description |
|---|---|---|
| `output` | same as `input` | Same shape and dtype as `input` |

## Data types

`fp16`, `fp32`, `bf16`.

## PyTorch example

```python
import torch
from torch.nn.functional import log_softmax

input = torch.randn(2, 3)
output = log_softmax(input, dim=1)
```

## Reference

[`torch.nn.LogSoftmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html)
