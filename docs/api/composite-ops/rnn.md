# rnn

Elman (simple) recurrent layer. Preserved as a named `rnn` composite op — mirroring the CoreML MIL `rnn` op — so the compiler and delegates can recognize the whole recurrent layer instead of a decomposed, unrolled sequence of primitive ops.

Per timestep, for each layer and direction:

$$h_t = \text{activation}(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$

where `activation` is `tanh` or `relu`.

**ATen source:** `aten.rnn_tanh.input` (tanh), `aten.rnn_relu.input` (relu)

Both biases are additive and are combined into a single `bias = b_ih + b_hh`, matching the MIL `rnn` op. A single composite is emitted per layer; stacked layers chain one composite each, and bidirectional layers use one composite that also takes the backward weights (states packed on the hidden axis).

> **Scope of the tables below.** The Inputs and Outputs tables describe the **composite-boundary** shapes. The graph-level `aten.rnn_{tanh,relu}.input` node and the final converted-program outputs are reassembled into the standard PyTorch `nn.RNN` layout (`batch_first` reapplied to `output`; `h_n` shaped `(num_layers * D, B, H)`).

## Inputs

| Name | Shape | Description |
|---|---|---|
| `x` | `(S, B, I)` | Input sequence (time-major; `batch_first` inputs are transposed first) |
| `initial_h` | `(B, H)` or `(B, 2H)` | Initial hidden state |
| `weight_ih` | `(H, I)` | Input-hidden weights |
| `weight_hh` | `(H, H)` | Hidden-hidden weights |
| `bias` | `(H,)` | Combined bias `b_ih + b_hh` (zeros when the layer has no bias) |
| `weight_ih_back`, `weight_hh_back`, `bias_back` | | Backward-direction inputs (bidirectional only) |

## Attributes

| Name | Type | Description |
|---|---|---|
| `direction` | `str` | `"forward"` or `"bidirectional"` |
| `output_sequence` | `bool` | Always `true` |
| `activation` | `str` | `"tanh"` or `"relu"` |
| `version` | `int` | Composite op version |

## Outputs

| Name | Shape | Description |
|---|---|---|
| `output` | `(S, B, DH)` | Hidden state at every timestep (`D` = 2 if bidirectional) |
| `h_n` | `(B, DH)` | Final hidden state |

The converted graph reassembles the standard PyTorch layout (`batch_first` reapplied to `output`; `h_n` shaped `(num_layers * D, B, H)`).

> **Note:** coremltools lowers torch RNN to the native MIL `rnn` op but supports **uni-directional** only. This composite additionally supports bidirectional RNN.

## Data types

`fp16`, `fp32`, `bf16`.

## Limitations

- The sequence length must be static (known at export time).
- Packed / variable-length sequences (`pack_padded_sequence`) are not supported.
- Training-time dropout (`train=True`, `dropout>0`, `num_layers>1`) is not supported; dropout configured on an `nn.RNN` in inference mode is a no-op and ignored.

## PyTorch example

```python
import torch

rnn = torch.nn.RNN(input_size=4, hidden_size=3, num_layers=2, nonlinearity="tanh", batch_first=True).eval()
x = torch.randn(2, 5, 4)
h0 = torch.randn(2, 2, 3)
output, h_n = rnn(x, h0)
```

## Reference

[`torch.nn.RNN`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
