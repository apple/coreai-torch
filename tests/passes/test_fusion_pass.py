# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test for LayerNorm + GELU graph fusion pass."""

import torch
import torch.nn as nn

from coreai_torch.passes.fusion import fuse_layernorm_gelu


class TestFusionPass:
    """Validate LayerNorm + GELU graph pattern detection and fusion rewrite."""

    def test_layernorm_gelu_fusion_rewrite(self) -> None:
        """Verify that adjacent LayerNorm and GELU nodes are fused into a single kernel op."""
        dim = 32

        class TransformerBlockHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ln = nn.LayerNorm(dim)
                self.gelu = nn.GELU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.gelu(self.ln(x))

        mod = TransformerBlockHead().eval()
        x = torch.randn(2, 16, dim)

        exported = torch.export.export(mod, args=(x,))
        # Run decompositions if needed, or pass directly
        graph_module = exported.graph_module

        # Verify before pass: gelu and layernorm nodes exist
        nodes_before = [
            n.target for n in graph_module.graph.nodes if n.op == "call_function"
        ]
        assert any("gelu" in str(t) for t in nodes_before)

        # Run fusion pass
        fused_exported = fuse_layernorm_gelu(exported)
        fused_gm = fused_exported.graph_module

        # Verify after pass:
        # 1. gelu call should be gone
        nodes_after = [n for n in fused_gm.graph.nodes if n.op == "call_function"]
        target_names = [str(n.target) for n in nodes_after]

        assert not any("aten.gelu" in t for t in target_names), (
            "aten.gelu should be fused into kernel"
        )
        assert any("fused_layernorm_gelu" in t for t in target_names), (
            f"Expected fused_layernorm_gelu in graph targets, found: {target_names}"
        )

    def test_unfused_when_not_adjacent(self) -> None:
        """Verify that LayerNorm is not fused when another operation intervenes before GELU."""
        dim = 32

        class InterleavedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ln = nn.LayerNorm(dim)
                self.linear = nn.Linear(dim, dim)
                self.gelu = nn.GELU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.ln(x)
                z = self.linear(y)
                return self.gelu(z)

        mod = InterleavedModel().eval()
        x = torch.randn(2, 16, dim)
        exported = torch.export.export(mod, args=(x,))
        fused_exported = fuse_layernorm_gelu(exported)

        nodes_after = [
            n
            for n in fused_exported.graph_module.graph.nodes
            if n.op == "call_function"
        ]
        target_names = [str(n.target) for n in nodes_after]

        # No fused kernel should be created because linear is in between
        assert not any("fused_layernorm_gelu" in t for t in target_names)
