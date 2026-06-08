# Supported ATen ops

This page lists every PyTorch ATen operator that `TorchConverter` lowers to Core AI operations out of the box.

## How to read this page

- Op names use the FX qualified-name form `op_name.overload` (e.g. `add.Tensor`, `mean.dim`). When PyTorch's decomposition pipeline produces a different overload than the one listed, that overload is not supported.
- A few names appear without an overload suffix (e.g. `add`, `mul`, `getitem`) — these match plain Python-operator FX nodes that have no `.default` overload.
- Three ops — `instance_norm.default`, `pixel_shuffle.default`, and `scaled_dot_product_attention.default` — are deliberately preserved by `get_decomp_table()` and emitted as composite ops in the lowered IR.
- All ops below are resolved through the registry in `coreai_torch._aten_to_core`. To override a built-in lowering with your own, pass `allow_override=True` to `register_torch_lowering()`.

## ATen ops

| ATen op | Notes |
|---|---|
| `_adaptive_avg_pool2d.default` | |
| `_local_scalar_dense.default` | Returns the 0-dim input as-is |
| `_log_softmax.default` | |
| `_native_batch_norm_legit_no_training.default` | Inference path only |
| `_softmax.default` | |
| `_to_copy.default` | Identity or `coreai.cast` |
| `_unsafe_view.default` | |
| `abs.default` | |
| `acos.default` | |
| `acosh.default` | |
| `add`, `add.Scalar`, `add.Tensor` | |
| `addmm.default` | `alpha`, `beta` honored |
| `alias.default` | Identity — no IR emitted |
| `amax.default` | |
| `amin.default` | |
| `any.default`, `any.dim`, `any.dims` | |
| `arange.start_step` | |
| `argmax.default` | |
| `asin.default` | |
| `asinh.default` | |
| `atan.default` | |
| `atanh.default` | |
| `avg_pool2d.default` | Lowered as a composite op |
| `avg_pool3d.default` | Lowered as a composite op |
| `bitwise_and.Tensor` | |
| `bitwise_not.default` | |
| `bitwise_or.Tensor` | |
| `bitwise_xor.Tensor` | |
| `bmm.default` | |
| `cat.default` | |
| `ceil`, `ceil.default` | |
| `clamp.default`, `clamp.Tensor` | |
| `clone.default` | Identity in the absence of memory-format changes |
| `complex.default` | |
| `constant_pad_nd.default` | |
| `convolution.default` | 1D / 2D / 3D, transposed, grouped |
| `copy.default` | |
| `cos.default` | |
| `cosh.default` | |
| `cumsum.default` | Lowered to `coreai.scan` with a sum combiner |
| `div.Scalar`, `div.Tensor` | |
| `div.Tensor_mode` | Honors `rounding_mode` (`None`, `"floor"`, `"trunc"`) |
| `embedding.default` | Lowered to `coreai.gather_nd` |
| `empty.default`, `empty.memory_format` | |
| `eq.Scalar`, `eq.Tensor` | |
| `erf.default` | |
| `exp.default` | |
| `exp2.default` | |
| `expand.default` | |
| `expm1.default` | |
| `flip.default` | |
| `floor.default` | |
| `floor_divide.default` | |
| `floordiv`, `floordiv.Scalar`, `floordiv.Tensor` | |
| `fmod.Scalar`, `fmod.Tensor` | |
| `full.default` | |
| `full_like.default` | |
| `gather.default` | Lowered to `coreai.gather_along_axis` |
| `ge.Scalar`, `ge.Tensor` | |
| `gelu.default` | |
| `getitem` | |
| `gt.Scalar`, `gt.Tensor` | |
| `hardsigmoid.default` | Lowered as a composite |
| `hardswish.default` | |
| `hardtanh.default` | |
| `index.Tensor` | Lowered to `coreai.gather_nd` |
| `index_put.default` | Lowered to `coreai.scatter_nd` |
| `index_select.default` | Lowered to `coreai.gather_along_axis` |
| `instance_norm.default` | Preserved as composite by `get_decomp_table()` |
| `isinf.default` | Lowered as `(x == +inf) \| (x == -inf)` |
| `le.Scalar`, `le.Tensor` | |
| `leaky_relu.default` | |
| `lift_fresh_copy.default` | |
| `linalg_vector_norm.default` | |
| `log.default` | |
| `log10.default` | |
| `log1p.default` | |
| `log2.default` | |
| `logical_and.default` | |
| `logical_not.default` | |
| `logical_or.default` | |
| `logical_xor.default` | |
| `lt.Scalar`, `lt.Tensor` | |
| `max.default`, `max.dim` | |
| `max_pool2d_with_indices.default` | |
| `maximum.default` | |
| `mean.default`, `mean.dim` | |
| `min.default`, `min.dim` | |
| `minimum.default` | |
| `mm.default` | |
| `mod`, `mod.Scalar`, `mod.Tensor` | |
| `mul`, `mul.Scalar`, `mul.Tensor` | |
| `native_group_norm.default` | |
| `native_layer_norm.default` | |
| `ne.Scalar`, `ne.Tensor` | |
| `neg`, `neg.default` | |
| `nonzero.default` | |
| `nonzero_numpy.default` | |
| `permute.default` | |
| `pixel_shuffle.default` | Preserved as composite by `get_decomp_table()` |
| `polar.default` | |
| `pow.Scalar`, `pow.Tensor_Scalar`, `pow.Tensor_Tensor` | |
| `prod.default`, `prod.dim_int` | |
| `reciprocal.default` | |
| `relu.default` | |
| `remainder.Tensor` | |
| `repeat.default` | |
| `round.default`, `round.decimals` | |
| `rsqrt.default` | |
| `scalar_tensor.default` | |
| `scaled_dot_product_attention.default` | Preserved as composite by `get_decomp_table()` |
| `scatter.reduce`, `scatter.src`, `scatter.value`, `scatter.value_reduce` | |
| `select.int` | Lowered to `coreai.slice_` plus a dim removal |
| `sigmoid.default` | |
| `sign.default` | |
| `silu.default` | |
| `sin.default` | |
| `sinh.default` | |
| `slice.Tensor` | Lowered to `coreai.slice_` |
| `slice_scatter.default` | Lowered to `coreai.slice_update` |
| `split_with_sizes.default` | Lowered to `coreai.split` |
| `sqrt.default` | |
| `squeeze.dims` | |
| `sub`, `sub.Scalar`, `sub.Tensor` | |
| `sum.dim_IntList` | |
| `sym_float` | Casts a `SymInt` scalar tensor to a `SymFloat` scalar tensor |
| `sym_min` | |
| `sym_size.int` | Returns the size of a tensor along a dim as a shape-`[1]` tensor |
| `tan.default` | |
| `tanh.default` | |
| `tile.default` | |
| `to.dtype` | Identity or `coreai.cast` |
| `topk.default` | |
| `true_divide.Tensor` | |
| `truediv` | |
| `trunc`, `trunc.default` | |
| `unsqueeze.default` | |
| `upsample_bilinear2d.vec` | Lowered to `coreai.interpolate` (linear mode) |
| `upsample_nearest2d.vec` | Lowered to `coreai.interpolate` (nearest-neighbor mode) |
| `view.default` | |
| `view_as_complex.default`, `view_as_complex_copy.default` | |
| `view_as_real.default`, `view_as_real_copy.default` | |
| `where.self` | |

## Higher-order ops

| Op | Notes |
|---|---|
| `cond` | `torch.cond` — emitted as a Core AI conditional with two branch subgraphs |
| `while_loop` | `torch._higher_order_ops.while_loop` |

## See also

- {doc}`TorchConverter` — `register_torch_lowering()` for adding or overriding op lowerings.
- {doc}`../guides/custom-op-lowering` — tutorial walkthrough.
- {doc}`composite-ops` — built-in composite op modules.

## Notices

PyTorch is a trademark of Meta Platforms, Inc.
