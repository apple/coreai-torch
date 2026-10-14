# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Default decomposition table for coreai-torch."""

from __future__ import annotations

import torch

# These ATen ops must NOT be decomposed by run_decompositions().  Removing
# them from the default decomposition table preserves them in the exported
# graph so the converter can lower them to their optimized Core AI
# implementations (composite ops or direct lowerings).
_COMPOSITE_OPS: list = [
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.instance_norm.default,
    torch.ops.aten.logsumexp.default,
    torch.ops.aten.mish.default,
    torch.ops.aten.pixel_shuffle.default,
    torch.ops.aten.reflection_pad1d.default,
    torch.ops.aten.reflection_pad2d.default,
    torch.ops.aten.reflection_pad3d.default,
    torch.ops.aten.replication_pad1d.default,
    torch.ops.aten.replication_pad2d.default,
    torch.ops.aten.replication_pad3d.default,
    torch.ops.aten.scaled_dot_product_attention.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.softplus.default,
]


def get_decomp_table() -> dict:
    """Return the recommended decomposition table for ``run_decompositions()``.

    Starts from ``torch.export.default_decompositions()`` and removes the ops
    that coreai-torch lowers directly so they are preserved in the exported
    graph and converted to their optimized Core AI implementations:

    *Composite ops:*

    * ``torch.ops.aten.hardsigmoid.default``
    * ``torch.ops.aten.instance_norm.default``
    * ``torch.ops.aten.pixel_shuffle.default``
    * ``torch.ops.aten.scaled_dot_product_attention.default``

    *Direct lowerings:*

    * ``torch.ops.aten.hardswish.default``
    * ``torch.ops.aten.reflection_pad{1,2,3}d.default`` (to ``coreai.pad`` reflect)
    * ``torch.ops.aten.replication_pad{1,2,3}d.default`` (to ``coreai.pad`` replicate)
    * ``torch.ops.aten.silu.default``

    *Numerically stable lowerings (fp16 safety):*

    * ``torch.ops.aten.logsumexp.default``
    * ``torch.ops.aten.mish.default``
    * ``torch.ops.aten.softplus.default``

    **Usage with** ``add_exported_program`` (caller handles decomposition)::

        import torch
        import coreai_torch

        ep = torch.export.export(model, args=example_inputs)
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        result = coreai_torch.TorchConverter().add_exported_program(ep).to_coreai()

    **Usage with** ``add_pytorch_module`` (caller handles decomposition in export_fn)::

        import torch
        import coreai_torch

        result = (
            coreai_torch.TorchConverter()
            .add_pytorch_module(
                model,
                export_fn=lambda m: torch.export.export(
                    m, args=example_inputs
                ).run_decompositions(
                    coreai_torch.get_decomp_table()
                ),
            )
            .to_coreai()
        )

    Returns:
        A decomposition table dict suitable for
        ``ExportedProgram.run_decompositions()``.
    """
    table = torch.export.default_decompositions()
    for op in _COMPOSITE_OPS:
        table.pop(op, None)
    return table
