# RMSNormImpl

Root Mean Square Layer Normalization ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)).

$$\text{RMSNorm}(x, \gamma) = \frac{x}{\sqrt{E[x^2] + \varepsilon}} \cdot \gamma$$

`RMSNormImpl` is the true composite op — the class the converter externalizes as `rms_norm`. It takes both the input `x` and the scale `γ` as explicit forward arguments so that, when externalized, the scale appears as a graph input on the composite op boundary rather than being baked in as a constant from a sibling parameter. Hold the scale as an `nn.Parameter` on your enclosing module and pass it through.

## Constructor

```python
RMSNormImpl(eps=1e-5)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `eps` | `float` | `1e-5` | Epsilon for numerical stability. |

The reduction axis is fixed to the last dimension (`axes = -1`).

## Forward

```python
def forward(self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor
```

Normalizes `input` over its last dimension and multiplies by `scale`. The caller owns the `scale` tensor — typically an `nn.Parameter` on the enclosing module, with shape `(dim,)` for the standard case or `(n_heads, 1, dim)` for fused Q/K normalization.

## ExternalizeSpec

```python
from coreai_torch.composite_ops import RMSNormImpl

ExternalizeSpec(
    target_class=RMSNormImpl,
    composite_op_name="rms_norm",
    composite_attrs=["axes", "eps"],
)
```

## Data types

| Tensor | Allowed types |
|---|---|
| `input`, `scale`, `output` | `fp32`, `fp16`, `bf16` |

## `RMSNorm`: convenience wrapper

```python
from coreai_torch.composite_ops import RMSNorm

RMSNorm(dim, eps=1e-5, n_heads=None)
```

`RMSNorm` is a thin `nn.Module` wrapper around `RMSNormImpl` that owns the learnable scale parameter so callers don't have to wire one up themselves. Its `forward(x)` applies the normalization with the internally-held weight.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dim` | `int` | — | Size of the last dimension. Determines the shape of the learnable scale. |
| `eps` | `float` | `1e-5` | Epsilon for numerical stability. |
| `n_heads` | `int \| None` | `None` | If set, scale shape is `(n_heads, 1, dim)` for fused Q/K normalization. If `None`, scale shape is `(dim,)`. |

The wrapper itself is not the externalization target — it composes `RMSNormImpl` internally, so `target_class=RMSNormImpl` still produces the `rms_norm` composite op.
