# GatedDeltaUpdate

Gated Delta Network recurrence — a linear-complexity alternative to softmax attention for sequence modeling. Used in modern efficient attention mechanisms like Delta Networks (Qwen3-Next) and other recurrent-style transformers. Use this op when your model implements a state-space or linear recurrence layer that you want preserved as a single composite op in the lowered IR. The state tensor `S` is a key-value memory matrix that accumulates over timesteps; `initial_state` lets you pass a cached state for autoregressive generation or chunked processing.

$$S_t = g_t \odot S_{t-1} + \beta_t \, k_t^\top \bigl(v_t - S_{t-1} k_t\bigr)$$

## Dimension variables

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `S` | Sequence length |
| `N_kq_heads` | Number of attention heads for query and key (note: Q and K have the same head count, unlike SDPA) |
| `N_v_heads` | Number of attention heads for value |
| `D_k` | Per-head dim for query / key |
| `D_v` | Per-head dim for value / output |

## Constructor

```python
GatedDeltaUpdate(use_qk_l2_norm=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_qk_l2_norm` | `bool` | `True` | Whether to apply L2 normalization to query and key before the delta update. |

## Forward

```python
def forward(
    self,
    query: torch.Tensor,          # [B, N_kq_heads, S, D_k]
    key: torch.Tensor,            # [B, N_kq_heads, S, D_k]
    value: torch.Tensor,          # [B, N_v_heads, S, D_v]
    g: torch.Tensor,              # [B, N_v_heads, S]
    beta: torch.Tensor,           # [B, N_v_heads, S]
    initial_state: torch.Tensor,  # [B, N_v_heads, D_k, D_v]
) -> tuple[torch.Tensor, torch.Tensor]
```

| Argument | Description |
|---|---|
| `query` | Queries. |
| `key` | Keys. Same head count as `query`. |
| `value` | Values. May have a different head count than `query` / `key` (`N_v_heads` is typically a multiple of `N_kq_heads`). |
| `g` | Gate / decay factors. Should be negative (the op applies `exp` internally; `exp(g)` ends up in `[0, 1]`). |
| `beta` | Update strength factors, typically in `[0, 1]` (often the output of a sigmoid). |
| `initial_state` | Recurrent state from prior sequence — pass zeros for a fresh sequence. |

Returns `(output, final_state)`:

| Return value | Shape | Description |
|---|---|---|
| `output` | `[B, S, N_v_heads, D_v]` | Per-timestep retrieval outputs. Same dtype as input. |
| `final_state` | `[B, N_v_heads, D_k, D_v]` | State matrix after processing all `S` timesteps. Same dtype as input. |

## ExternalizeSpec

```python
ExternalizeSpec(
    target_class=GatedDeltaUpdate,
    composite_op_name="gated_delta_update",
    composite_attrs=["use_qk_l2_norm"],
)
```

## Data types

`fp32`, `fp16`, `bf16` for all tensor inputs and outputs.

## Constraints

- All input tensors must have compatible (promotable) dtypes; the output dtype matches the promoted input dtype.
- `g` should be negative — the op applies `exp` internally and `exp(g)` must lie in `[0, 1]` for the decay to behave correctly.
- `beta` is typically in `[0, 1]` (often the output of a sigmoid).

## L2 normalization flag

When `use_qk_l2_norm=True` (default), the op applies L2 normalization to `query` and `key` before the recurrence. Set it to `False` if your model already L2-normalizes Q/K externally.
