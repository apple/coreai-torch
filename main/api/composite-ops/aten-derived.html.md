# ATen-derived composite ops

Composite ops recognized automatically from the ATen nodes (`fx.Node`s) in your `ExportedProgram` during conversion. These have no corresponding `nn.Module` wrapper — use the standard PyTorch APIs (e.g., `torch.nn.BatchNorm2d`, `torch.nn.functional.pixel_shuffle`) and Core AI preserves them as composite ops, as long as `get_decomp_table()` keeps them from being decomposed.

* [batch_norm](batch-norm.md)
* [group_norm](group-norm.md)
* [hard_sigmoid](hard-sigmoid.md)
* [instance_norm](instance-norm.md)
* [layer_norm](layer-norm.md)
* [linalg_vector_norm](linalg-vector-norm.md)
* [log_softmax](log-softmax.md)
* [pixel_shuffle](pixel-shuffle.md)

For the ATen ops these are derived from, see [Supported ATen ops](../supported-aten-ops.md).
