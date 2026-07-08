# ATen-derived composite ops

Composite ops recognized automatically from the ATen nodes (`fx.Node`s) in your `ExportedProgram` during conversion. These have no corresponding `nn.Module` wrapper — use the standard PyTorch APIs (e.g., `torch.nn.BatchNorm2d`, `torch.nn.functional.pixel_shuffle`) and Core AI preserves them as composite ops, as long as `get_decomp_table()` keeps them from being decomposed.

```{toctree}
:maxdepth: 1

batch-norm
group-norm
gru
hard-sigmoid
instance-norm
layer-norm
linalg-vector-norm
log-softmax
lstm
pixel-shuffle
rnn
```

For the ATen ops these are derived from, see {doc}`../supported-aten-ops`.
