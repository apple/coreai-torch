# Module-class composite ops

`nn.Module` subclasses exposed in `coreai_torch.composite_ops`. Build these into your model as named submodules and externalize them with an `ExternalizeSpec`, passing them to the `externalize_modules` parameter of `add_pytorch_module()`. For a tutorial walkthrough, see {doc}`../../guides/composite-ops`.

```{toctree}
:maxdepth: 1

gather-mm
gated-delta-update
rms-norm
rope
sdpa
```
