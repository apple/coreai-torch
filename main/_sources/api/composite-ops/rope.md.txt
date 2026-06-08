# RoPE

Rotary Positional Embedding ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) — encodes both absolute position and relative distance between tokens by rotating pairs of elements in the last dimension. The rotation angle is derived from the token's absolute position; either the embedding is split in half or its alternate elements are paired (`interleaved=True`) before the rotation matrix is applied.

Three ways to drive the rotation, in priority order:

1. Pass precomputed `cos` and `sin` directly.
2. Pass `position_ids` (and optionally `freqs`) — the op constructs `cos`/`sin` internally.
3. Pass nothing extra — the op derives `position_ids` from `offset` + `scale` and `freqs` from `base`.

Use `position_ids` when your model computes position indices externally (custom sequence packing or variable-length inputs). Use `cos`/`sin` when you have precomputed frequency tensors. Use `offset` for KV-cache decoding steps where only a single new token position is needed.

## Constructor

```python
RoPE(scale=1.0, base=1e4, dims=None, interleaved=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scale` | `float` | `1.0` | Frequency scaling factor applied to positions. |
| `base` | `float` | `1e4` | Base for the geometric frequency sequence. |
| `dims` | `int \| None` | `None` | Number of dimensions to rotate. If `None`, rotates all dimensions. |
| `interleaved` | `bool` | `False` | If `True`, uses interleaved (Hugging Face-style) rotation; if `False`, uses split-half rotation. |

## Forward

```python
def forward(
    self,
    input: torch.Tensor,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    freqs: torch.Tensor | None = None,
    offset: torch.Tensor | None = None,
) -> torch.Tensor
```

| Argument | Description |
|---|---|
| `input` | Tensor of rank ≥ 3, shape `[batch, ..., seq_len, embed]`. |
| `cos`, `sin` | Precomputed cosines / sines, broadcastable to `[batch, ..., seq_len, embed/2]`. If both are provided, all other position-construction arguments are ignored. |
| `position_ids` | Position indices, broadcastable to `[batch, ..., seq_len]`. Ignored if `cos` and `sin` are provided. |
| `freqs` | Custom angular frequencies of shape `[embed/2]`. Useful for advanced variants like SuScaledRoPE. Ignored if `cos` and `sin` are provided. |
| `offset` | Starting position for the sequence. Tensor of shape `[]` / `[1]` / `[batch]` / `[batch, 1]`, or an `int` attribute. If a tensor is provided alongside the int attribute, the tensor wins. Default: `0`. |

When `cos`/`sin` are not provided, position ids are constructed as `position_ids = (offset + arange(seq_len)) * scale`, and frequencies are `freqs[i] = 1 / base ** (i / (embed/2))`.

## Optional input resolution order

1. If `cos` and `sin` are both provided, use them directly.
2. Else, build `cos`/`sin` from `position_ids` and `freqs`:
   - `position_ids`: use the argument if provided; otherwise construct from `offset` and `scale`.
   - `freqs`: use the argument if provided; otherwise construct from `base`.

## Input names variants

| Arguments provided | `input_names` in IR |
|---|---|
| `input` only | `["input"]` |
| `input`, `cos`, `sin` | `["input", "cos", "sin"]` |
| `input`, `freqs` | `["input", "freqs"]` |
| `input`, `offset` | `["input", "offset"]` |
| `input`, `position_ids` | `["input", "position_ids"]` |

## ExternalizeSpec

```python
ExternalizeSpec(
    target_class=RoPE,
    composite_op_name="rope",
    composite_attrs=["scale", "base", "dims", "interleaved"],
)
```

## Partial rotation (`dims`)

When `dims` is set to a positive even integer smaller than `embed`, only the first `dims` features are rotated; the rest pass through unchanged:

```python
y_partial = rope(input[..., :dims])
output = torch.cat([y_partial, input[..., dims:]], dim=-1)
```

When `dims` is `None` or `dims >= embed`, the full last dimension is rotated.

## Data types

| Tensor | Allowed dtypes |
|---|---|
| `input`, `cos`, `sin`, `freqs`, `output` | `fp32`, `fp16`, `bf16` |
| `position_ids`, `offset` | integer |

`input`, `cos`, and `sin` dtypes must be promotable; the output dtype is the promoted type.

## Reference

- [Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
