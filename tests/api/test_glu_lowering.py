# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test for direct ATen to Core AI IR lowering for composite ops."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_torch import get_decomp_table
from coreai_torch._aten_to_core import _aten_to_core_resolver
from coreai_torch._decomp import _COMPOSITE_OPS


class TestCompositeAtenLowering:
    """Validate that glu, softplus, mish, and elu are preserved in composite ops and registered in resolver."""

    def test_ops_registered_in_resolver(self) -> None:
        """Verify all new composite ops have registered lowering handlers in _aten_to_core_resolver."""
        expected_ops = [
            torch.ops.aten.glu.default,
            torch.ops.aten.softplus.default,
            torch.ops.aten.mish.default,
            torch.ops.aten.elu.default,
        ]
        for op in expected_ops:
            assert op in _aten_to_core_resolver, (
                f"Expected {op} to be registered in _aten_to_core_resolver"
            )
            assert callable(_aten_to_core_resolver[op]), (
                f"Handler for {op} must be callable"
            )

    def test_ops_in_composite_decomp_table(self) -> None:
        """Verify ops are included in _COMPOSITE_OPS and thus excluded from decomposition table."""
        expected_ops = [
            torch.ops.aten.glu.default,
            torch.ops.aten.softplus.default,
            torch.ops.aten.mish.default,
            torch.ops.aten.elu.default,
        ]
        for op in expected_ops:
            assert op in _COMPOSITE_OPS, (
                f"Expected {op} to be present in _COMPOSITE_OPS"
            )

        decomp_table = get_decomp_table()
        for op in expected_ops:
            assert op not in decomp_table, (
                f"Op {op} should NOT be decomposed so it can be lowered directly into Core AI IR"
            )

    @pytest.mark.parametrize(
        "op_name,fn,input_shape",
        [
            ("glu", lambda x: F.glu(x, dim=-1), (2, 8, 32)),
            ("softplus", lambda x: F.softplus(x, beta=1.0, threshold=20.0), (2, 8, 16)),
            ("mish", lambda x: F.mish(x), (2, 8, 16)),
            ("elu", lambda x: F.elu(x, alpha=1.0), (2, 8, 16)),
        ],
    )
    def test_fx_graph_preserves_target_op(self, op_name, fn, input_shape) -> None:
        """Verify that export with get_decomp_table() retains high-level ATen op nodes in the FX graph."""

        class TestModule(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return fn(x)

        mod = TestModule().eval()
        x = torch.randn(*input_shape)
        exported = torch.export.export(mod, args=(x,))
        decomposed = exported.run_decompositions(get_decomp_table())

        op_nodes = [
            node for node in decomposed.graph.nodes if node.op == "call_function"
        ]
        target_names = [str(node.target) for node in op_nodes]
        assert any(op_name in t for t in target_names), (
            f"Expected {op_name} target to be preserved in FX graph, got: {target_names}"
        )
