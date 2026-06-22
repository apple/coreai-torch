# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for dynamic-shape reconstruction in the externalize pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import sympy
import torch

from coreai_torch._utils import _dynamic_shapes_from_node


class TestDynamicShapesFromNode:
    """Cover ``_dynamic_shapes_from_node``'s reconstruction of a positional
    ``dynamic_shapes`` tuple from a custom op node's FakeTensors.
    """

    @staticmethod
    def test_skips_specialised_symints() -> None:
        """Specialised SymInts (``expr.is_number``) must not produce a Dim.

        When a model is externalised, the standalone re-export reconstructs
        a ``dynamic_shapes`` spec from the FakeTensors flowing into the
        custom op node. If a dim's symbol has been fully specialised to a
        literal int by the parent program (e.g. the key sequence length
        emerging from ``KVCache.update_and_fetch`` after the prefill chunk
        size is resolved), the SymInt's ``.node.expr`` is a sympy
        ``Integer``. Asking for a ``Dim`` for it would yield an unbounded
        ``Dim(min=1)`` and ``torch.export`` would reject the submodule
        with ``L['key'].size()[2] <= IntInfinity()``. The filter treats
        such dims as static (no Dim entry).
        """
        specialised = MagicMock(spec=torch.SymInt)
        specialised.node = SimpleNamespace(expr=sympy.Integer(20))

        class _FakeTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, shape: tuple[object, ...]) -> "_FakeTensor":
                t = torch.Tensor._make_subclass(cls, torch.empty(0))
                t._shape = shape
                return t

            @property  # type: ignore[override]
            def shape(self) -> tuple[object, ...]:  # noqa: D401
                return self._shape

        val = _FakeTensor((1, 8, specialised, 64))
        arg = SimpleNamespace(meta={"val": val})
        node = SimpleNamespace(args=(arg,), target="custom_op")

        # All non-symbolic / specialised dims → no Dim entries → None for the arg.
        assert _dynamic_shapes_from_node(node) == (None,)
