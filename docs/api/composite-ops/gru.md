# gru

Gated Recurrent Unit recurrent layer. Preserved as a named `gru` composite op so the compiler and delegates can recognize the whole recurrent layer instead of a decomposed, unrolled sequence of primitive ops.

Per timestep, for each layer and direction:

$$
\begin{aligned}
r_t &= \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) \\
z_t &= \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) \\
n_t &= \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn})) \\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1}
\end{aligned}
$$

**ATen source:** `aten.gru.input`

## `reset_after` and the MIL GRU

This composite models PyTorch's GRU, which applies the reset gate **after** the hidden matrix multiply, *including* its bias: `r_t ⊙ (W_hn h_{t-1} + b_hn)`. The CoreML MIL `gru` op instead applies the reset gate before the hidden bias (`r_t * W_ho h_{t-1} + b_ho`) — the `reset_after = false` convention. Because of this difference:

- coremltools does **not** lower a torch GRU to the native MIL `gru` op; it hand-builds the recurrence.
- the hidden bias `b_hn` **cannot** be folded into a single combined bias, so this composite takes the input and hidden biases separately (`bias_ih`, `bias_hh`).
- the composite carries a `reset_after` attribute (always `true` for `torch.nn.GRU`) so a delegate knows which convention to apply.

## Inputs

A single composite is emitted per layer; stacked layers chain one composite each, and bidirectional layers use one composite that also takes the backward weights (states packed on the hidden axis: `[:, :H]` forward, `[:, H:]` reverse).

> **Scope of the tables below.** The Inputs and Outputs tables describe the **composite-boundary** shapes. The graph-level `aten.gru.input` node and the final converted-program outputs are reassembled into the standard PyTorch `nn.GRU` layout (`batch_first` reapplied to `output`; `h_n` shaped `(num_layers * D, B, H)`).

| Name | Shape | Description |
|---|---|---|
| `x` | `(S, B, I)` | Input sequence (time-major; `batch_first` inputs are transposed first) |
| `initial_h` | `(B, H)` or `(B, 2H)` | Initial hidden state |
| `weight_ih` | `(3H, I)` | Input-hidden weights, PyTorch `[r, z, n]` gate layout |
| `weight_hh` | `(3H, H)` | Hidden-hidden weights, `[r, z, n]` layout |
| `bias_ih` | `(3H,)` | Input bias (zeros when the layer has no bias) |
| `bias_hh` | `(3H,)` | Hidden bias (zeros when the layer has no bias) |
| `weight_ih_back`, `weight_hh_back`, `bias_ih_back`, `bias_hh_back` | | Backward-direction inputs (bidirectional only) |

## Attributes

| Name | Type | Description |
|---|---|---|
| `direction` | `str` | `"forward"` or `"bidirectional"` |
| `output_sequence` | `bool` | Always `true` |
| `recurrent_activation` | `str` | Reset/update gate activation (`"sigmoid"`) |
| `activation` | `str` | New-gate activation (`"tanh"`) |
| `reset_after` | `bool` | Reset gate applied after the hidden bias (`true` for torch) |
| `version` | `int` | Composite op version |

## Outputs

| Name | Shape | Description |
|---|---|---|
| `output` | `(S, B, DH)` | Hidden state at every timestep (`D` = 2 if bidirectional) |
| `h_n` | `(B, DH)` | Final hidden state |

The converted graph reassembles the standard PyTorch layout (`batch_first` reapplied to `output`; `h_n` shaped `(num_layers * D, B, H)`).

## Data types

`fp16`, `fp32`, `bf16`.

## Limitations

- The sequence length must be static (known at export time).
- Packed / variable-length sequences (`pack_padded_sequence`) are not supported.
- Training-time dropout (`train=True`, `dropout>0`, `num_layers>1`) is not supported; dropout configured on an `nn.GRU` in inference mode is a no-op and ignored.

## PyTorch example

```python
import torch

gru = torch.nn.GRU(input_size=4, hidden_size=3, num_layers=2, batch_first=True).eval()
x = torch.randn(2, 5, 4)
h0 = torch.randn(2, 2, 3)
output, h_n = gru(x, h0)
```

## Reference

[`torch.nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
