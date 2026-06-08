# hard_sigmoid

Piecewise-linear approximation of sigmoid:

$$\text{HardSigmoid}(x) = \frac{\min(\max(x + 3,\ 0),\ 6)}{6}$$

**ATen source:** `aten.hardsigmoid`

## Inputs

| Name | Description |
|---|---|
| `input` | Tensor of any rank |

## Attributes

| Name | Type | Description |
|---|---|---|
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

input = torch.rand(1, 3, 64, 64)
output = torch.nn.functional.hardsigmoid(input)
```

## Reference

[`torch.nn.Hardsigmoid`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html)
