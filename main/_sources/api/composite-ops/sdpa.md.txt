# SDPA

Scaled Dot-Product Attention with optional causal masking, sliding window, and attention sinks.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Use this class instead of `torch.nn.functional.scaled_dot_product_attention` when you need the full attention operation preserved as a single composite op in the lowered IR. The `is_causal`, `window_size`, and `sinks` options compose: you can enable causal masking, restrict the attended context with a sliding window, and designate a fixed number of global sink tokens — all in a single externalized op.

## Constructor

```python
SDPA(scale=None, is_causal=False, window_size=0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scale` | `float \| None` | `None` | Attention scale factor. If `None`, uses `head_dim ** -0.5`. |
| `is_causal` | `bool` | `False` | Apply lower-right causal mask. |
| `window_size` | `int` | `0` | Sliding window size. `0` means no window (full attention). |

## Forward

```python
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor
```

## Input names variants

| Arguments provided | `input_names` in IR |
|---|---|
| `query, key, value` | `["query", "key", "value"]` |
| `query, key, value, attn_mask` | `["query", "key", "value", "attn_mask"]` |
| `query, key, value, sinks` | `["query", "key", "value", "sinks"]` |
| `query, key, value, attn_mask, sinks` | `["query", "key", "value", "attn_mask", "sinks"]` |

## ExternalizeSpec

```python
ExternalizeSpec(
    target_class=SDPA,
    composite_op_name="scaled_dot_product_attention",
    composite_attrs=["scale", "is_causal", "window_size"],
)
```

## Supported attention schemas

`SDPA` covers Multi-Head, Grouped-Query, and Multi-Query Attention based on the relationship between `N_q` (query heads) and `N_kv` (key/value heads):

| Schema | Constraint | Example |
|---|---|---|
| Multi-Head Attention (MHA) | `N_q == N_kv` | 32 query heads, 32 kv heads |
| Grouped-Query Attention (GQA) | `N_q > N_kv`, `N_q % N_kv == 0` | 32 query heads, 8 kv heads |
| Multi-Query Attention (MQA) | `N_kv == 1` | 32 query heads, 1 kv head |

Tensor shapes: `query` is `[B, N_q, T_q, D]`, `key` is `[B, N_kv, T_kv, D]`, `value` is `[B, N_kv, T_kv, D_v]`. For GQA / MQA, do **not** pre-tile `key` / `value` to match `N_q` — pass them with their native `N_kv` and the broadcasting is recorded as part of the composite op.

## Reference

[`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
