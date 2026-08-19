# Module-class composite ops

`nn.Module` subclasses exposed in `coreai_torch.composite_ops`. Build these into your model as named submodules and externalize them with an `ExternalizeSpec`, passing them to the `externalize_modules` parameter of `add_pytorch_module()`. For a tutorial walkthrough, see [Composite Ops Guide](../../guides/composite-ops.md).

* [GatherMM](gather-mm.md)
* [GatedDeltaUpdate](gated-delta-update.md)
* [RMSNormImpl](rms-norm.md)
* [RoPE](rope.md)
* [SDPA](sdpa.md)
