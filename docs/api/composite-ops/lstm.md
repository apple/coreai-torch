# lstm

Long Short-Term Memory recurrent layer. Preserved as a named `lstm` composite op (mirroring the CoreML MIL `lstm` interface) so the compiler and delegates can recognize the whole recurrent layer instead of a decomposed, unrolled sequence of primitive ops.

Per timestep, for each layer and direction:

$$
\begin{aligned}
i_t &= \sigma(W_{ii} x_t + b_i + W_{hi} h_{t-1}) \\
f_t &= \sigma(W_{if} x_t + b_f + W_{hf} h_{t-1}) \\
g_t &= \tanh(W_{ig} x_t + b_g + W_{hg} h_{t-1}) \\
o_t &= \sigma(W_{io} x_t + b_o + W_{ho} h_{t-1}) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

**ATen source:** `aten.lstm.input`

A single composite is emitted per layer. Stacked (multi-layer) LSTMs chain one composite per layer; bidirectional layers use a single composite that also takes the backward weights. Weights are reordered from PyTorch's `[i, f, g, o]` gate layout to MIL's `[i, f, o, g]`, and the input/hidden biases are summed into one `bias`, so the composite inputs match the MIL `lstm` op directly.

> **Scope of the tables below.** The Inputs and Outputs tables describe the **composite-boundary** shapes — the arguments and results of the emitted `lstm` composite op itself. These differ from what a `torch.nn.LSTM` user sees: the graph-level `aten.lstm.input` node and the final converted-program outputs are reassembled into the standard PyTorch layout (see the note after the Outputs table).

## Inputs

| Name | Shape | Description |
|---|---|---|
| `x` | `(S, B, I)` | Input sequence (time-major; `batch_first` inputs are transposed before the composite) |
| `initial_h` | `(B, H)` or `(B, 2H)` | Initial hidden state (forward+reverse packed on the hidden axis when bidirectional) |
| `initial_c` | `(B, H)` or `(B, 2H)` | Initial cell state |
| `weight_ih` | `(4H, I)` | Input-hidden weights, `ifog` gate layout |
| `weight_hh` | `(4H, H)` | Hidden-hidden weights, `ifog` gate layout |
| `bias` | `(4H,)` | Combined input+hidden bias (zeros when the layer has no bias) |
| `weight_ih_back` | `(4H, I)` | Backward-direction input-hidden weights (bidirectional only) |
| `weight_hh_back` | `(4H, H)` | Backward-direction hidden-hidden weights (bidirectional only) |
| `bias_back` | `(4H,)` | Backward-direction combined bias (bidirectional only) |

## Attributes

| Name | Type | Description |
|---|---|---|
| `direction` | `str` | `"forward"` or `"bidirectional"` |
| `output_sequence` | `bool` | Always `true` — the full per-timestep hidden sequence is returned |
| `recurrent_activation` | `str` | Gate activation (`"sigmoid"`) |
| `cell_activation` | `str` | Cell activation (`"tanh"`) |
| `activation` | `str` | Output activation (`"tanh"`) |
| `version` | `int` | Composite op version |

## Outputs

| Name | Shape | Description |
|---|---|---|
| `output` | `(S, B, DH)` | Hidden state at every timestep (`D` = 2 if bidirectional) |
| `h_n` | `(B, DH)` | Final hidden state |
| `c_n` | `(B, DH)` | Final cell state |

The final `output`, `h_n`, and `c_n` returned by the converted graph follow the standard PyTorch `nn.LSTM` layout (`batch_first` reapplied to `output`; states shaped `(num_layers * D, B, H)`).

## Data types

`fp16`, `fp32`, `bf16`.

## Limitations

- The sequence length must be static (known at export time).
- Packed / variable-length sequences (`aten.lstm.data`, `pack_padded_sequence`) are not supported.
- Training-time dropout (`train=True`, `dropout>0`, `num_layers>1`) is not supported; a dropout rate configured on an `nn.LSTM` evaluated in inference mode is a no-op and safely ignored.
- For **batch size > 1**, the converted program's `h_n` / `c_n` carry an extra unit dim at axis 1 — shape `(num_layers * D, 1, B, H)` — because `torch.export`'s fake-tensor kernel reports that shape (eager `nn.LSTM` returns rank-3 `(num_layers * D, B, H)`). The converter matches the exported meta, so downstream consumers should expect the rank-4 states for batched inputs.

## PyTorch example

```python
import torch

lstm = torch.nn.LSTM(input_size=4, hidden_size=3, num_layers=2, batch_first=True).eval()
x = torch.randn(2, 5, 4)
h0 = torch.randn(2, 2, 3)
c0 = torch.randn(2, 2, 3)
output, (h_n, c_n) = lstm(x, (h0, c0))
```

## Reference

[`torch.nn.LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
