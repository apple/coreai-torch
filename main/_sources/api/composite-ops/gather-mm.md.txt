# GatherMM

Gather matmul — optionally gathers rows from one or both operands before performing the matrix multiplication:

$$\text{GatherMM}(A, B) = \text{matmul}(\text{gather}(A,\, i_A),\, \text{gather}(B,\, i_B))$$

The primary use case is Mixture-of-Experts (MoE): each token selects a subset of expert weight matrices and the result is computed in a single fused operation. Without `GatherMM`, you would explicitly gather the relevant expert weights and then run a matmul; this op fuses both for better performance.

If neither `lhs_indices` nor `rhs_indices` is provided, the op is equivalent to a plain `matmul`.

## Constructor

```python
GatherMM(num_batch_axes=0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_batch_axes` | `int` | `0` | Number of leading batch axes shared by all operands. The gather is applied along the axis at position `num_batch_axes`. |

## Forward

```python
def forward(
    self,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_indices: torch.Tensor | None = None,
    rhs_indices: torch.Tensor | None = None,
) -> torch.Tensor
```

| Argument | Required | Description |
|---|---|---|
| `lhs` | Yes | Left-hand operand. Rank ≥ 2 (≥ 3 if `lhs_indices` is provided). MoE: the input hidden-state tensor. |
| `rhs` | Yes | Right-hand operand. Rank ≥ 2 (≥ 3 if `rhs_indices` is provided). MoE: the stacked expert weight matrices. |
| `lhs_indices` | No | Unsigned-int tensor of flat indices into the batch dims of `lhs` (range `[0, A1·A2·…·AS)` for an `lhs` shape `(A1, A2, …, AS, M, K)`). MoE: typically `None`. |
| `rhs_indices` | No | Unsigned-int tensor of flat indices into the batch dims of `rhs`. MoE: the active-experts indices. |

## Data types

| Tensor | Allowed dtypes |
|---|---|
| `lhs`, `rhs`, `output` | Any real or complex float type (`fp32`, `fp16`, `bf16`, complex) |
| `lhs_indices`, `rhs_indices` | Unsigned integer index types (e.g., `uint16`, `uint32`) |

## Input names variants

| Arguments provided | `input_names` in IR |
|---|---|
| `lhs`, `rhs` | `["lhs", "rhs"]` |
| `lhs`, `rhs`, `rhs_indices` | `["lhs", "rhs", "rhs_indices"]` |
| `lhs`, `rhs`, `lhs_indices` | `["lhs", "rhs", "lhs_indices"]` |
| `lhs`, `rhs`, `lhs_indices`, `rhs_indices` | `["lhs", "rhs", "lhs_indices", "rhs_indices"]` |

## ExternalizeSpec

```python
ExternalizeSpec(
    target_class=GatherMM,
    composite_op_name="gather_mm",
    composite_attrs=["num_batch_axes"],
)
```

## Examples

**MoE with `rhs_indices`:**

```python
import coreai_torch
from coreai_torch import TorchConverter, ExternalizeSpec
from coreai_torch.composite_ops import GatherMM
import torch
import torch.nn as nn


class MoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gather_mm = GatherMM(num_batch_axes=0)

    def forward(
        self,
        x: torch.Tensor,          # [B, T, 1, 1, D]
        experts: torch.Tensor,    # [E, D, H]
        indices: torch.Tensor,    # [B, T, K]
    ) -> torch.Tensor:            # [B, T, K, 1, H]
        return self.gather_mm(x, experts, rhs_indices=indices)


B, T, D, H, E, K = 1, 16, 64, 128, 8, 2
model = MoELayer().eval()
sample = (
    torch.randn(B, T, 1, 1, D),
    torch.randn(E, D, H),
    torch.zeros(B, T, K, dtype=torch.int32),
)

coreai_program = (
    TorchConverter()
    .add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
            coreai_torch.get_decomp_table()
        ),
        externalize_modules=[
            ExternalizeSpec(
                target_class=GatherMM,
                composite_op_name="gather_mm",
                composite_attrs=["num_batch_axes"],
            )
        ],
    )
    .to_coreai()
)
coreai_program.optimize()
```

**Fused projections (`num_batch_axes=1`):**

When gate and up projections are stacked along a leading fused axis, set `num_batch_axes=1` so the gather operates on the expert axis (dim 1) rather than dim 0:

```python
class FusedMoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gather_mm = GatherMM(num_batch_axes=1)

    def forward(
        self,
        x: torch.Tensor,            # [B, T, 1, 1, D]
        fused_experts: torch.Tensor, # [2, E, D, H]  (gate + up stacked)
        indices: torch.Tensor,      # [B, T, K]
    ) -> torch.Tensor:              # [2, B, T, K, 1, H]
        return self.gather_mm(x, fused_experts, rhs_indices=indices)
```

**`lhs_indices` only:**

```python
class LhsGatherLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gather_mm = GatherMM(num_batch_axes=0)

    def forward(self, x, weight, indices):
        return self.gather_mm(x, weight, lhs_indices=indices)
```

**Both `lhs_indices` and `rhs_indices`:**

```python
class BothGatherLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gather_mm = GatherMM(num_batch_axes=0)

    def forward(self, x, experts, lhs_idx, rhs_idx):
        return self.gather_mm(x, experts, lhs_indices=lhs_idx, rhs_indices=rhs_idx)
```

## Decomposition

`GatherMM` is semantically equivalent to a `gather` followed by a `matmul`:

```python
def _gather(x, indices, num_batch_axes=0):
    flat_indices = indices.to(torch.int32).flatten()
    flat_gather = torch.index_select(x, dim=num_batch_axes, index=flat_indices)
    result_shape = (
        x.shape[:num_batch_axes] + indices.shape + x.shape[num_batch_axes + 1:]
    )
    return flat_gather.view(result_shape)

def gather_mm(lhs, rhs, lhs_indices=None, rhs_indices=None, num_batch_axes=0):
    if lhs_indices is not None:
        lhs = _gather(lhs, lhs_indices, num_batch_axes=num_batch_axes)
    if rhs_indices is not None:
        rhs = _gather(rhs, rhs_indices, num_batch_axes=num_batch_axes)
    return torch.matmul(lhs, rhs)
```
