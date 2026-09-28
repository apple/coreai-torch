# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Dispatch tests for ``torch.cond`` / ``torch.while_loop`` branch conversion.

These cover the change that lets user-registered lowerings (custom Metal
kernels in the ``coreai_metal_kernels`` namespace) and ``coreai``/``coreaix``
ops appear inside a control-flow branch. Before the change,
:func:`coreai_torch._utils.convert_branch_subgraph` only knew how to dispatch
``aten``/``None`` ops and nested higher-order ops, so anything else hit the
catch-all ``ValueError: unsupported op in branch``.

Two layers are tested:

* :class:`TestConvertBranchSubgraphDispatch` drives ``convert_branch_subgraph``
  directly with hand-built FX branches and mock resolvers. This pins the
  dispatch priority order (user-defined -> aten -> coreai/coreaix ->
  higher-order), the variant-stripped key used for the custom resolver, the
  threading of ``user_defined_resolver`` into nested higher-order handlers, and
  the backward-compatible ``ValueError`` when the new resolvers are omitted.
* :class:`TestBranchResolverWiring` converts real (plain) ``cond`` /
  ``while_loop`` models while spying on ``convert_branch_subgraph`` to prove the
  higher-order handlers (``replace_cond`` / ``replace_while_loop``) and the
  converter thread the resolvers through as expected -- including the secondary
  fix where ``replace_cond`` now forwards the *full* higher-order resolver (so a
  ``while_loop`` can nest inside a ``cond`` branch) rather than ``{"cond": ...}``.

The dispatch tests are hermetic: they never run an op, only assert which
resolver a node is routed to, so they need no Metal device and no numerical
runtime.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
import torch.fx as fx
from coreai._compiler.ir import Context

import coreai_torch._aten_to_core as aten_to_core
from coreai_torch import TorchConverter, get_decomp_table
from coreai_torch._aten_to_core import (
    _higher_order_resolver,
    replace_cond,
    replace_while_loop,
    replace_yield,
)
from coreai_torch._custom_to_core import _custom_to_core_resolver
from coreai_torch._utils import convert_branch_subgraph

# Positional layout of a convert_branch_subgraph(...) call, so the wiring spy
# can name what it captured instead of using bare indices.
_BRANCH_MODULE = 0
_OPERAND_VALUES = 1
_GRAPH_MODULE = 2
_ATEN_RESOLVER = 3
_HIGHER_ORDER_HANDLERS = 4
_USER_DEFINED_RESOLVER = 5
_CUSTOM_RESOLVER = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_op(namespace: str | None, name: str) -> Any:
    """A callable FX target that mimics an op's ``namespace`` / ``__name__``.

    ``get_namespace`` reads ``target.namespace`` only when the attribute is
    present, so passing ``namespace=None`` (the "no namespace" case, e.g.
    ``operator.getitem``) leaves the attribute unset on purpose.
    """

    def op(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"fake op {namespace}::{name} must not be executed")

    op.__name__ = name
    if namespace is not None:
        op.namespace = namespace  # type: ignore[attr-defined]
    return op


def _single_call_branch(op: Any) -> fx.GraphModule:
    """Build a branch GraphModule: ``placeholder -> op(placeholder) -> output``."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.output((graph.call_function(op, (x,)),))
    return fx.GraphModule(torch.nn.Module(), graph)


def _recording_resolver(calls: list[str], tag: str, result: Any) -> Any:
    """A resolver handler that records it was invoked and returns ``result``."""

    def handler(branch_values: dict, node: fx.Node, loc: Any) -> Any:
        calls.append(tag)
        return result

    return handler


@pytest.fixture
def mlir_context() -> Any:
    """``convert_branch_subgraph`` calls ``Location.unknown()``, which needs a
    default MLIR context established for the duration of the call."""
    with Context() as ctx:
        yield ctx


# ---------------------------------------------------------------------------
# Direct dispatch tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mlir_context")
class TestConvertBranchSubgraphDispatch:
    """``convert_branch_subgraph`` routes each branch node to the right resolver."""

    @staticmethod
    def test_user_defined_resolver_preferred_over_aten() -> None:
        """A registered user lowering wins even if the target collides with aten.

        The key is the ``"namespace::target"`` qualified name, and the lookup
        happens before the ``aten`` fallback.
        """
        op = _make_fake_op("coreai_metal_kernels", "double_1.default")
        branch = _single_call_branch(op)
        calls: list[str] = []
        sentinel = object()

        out = convert_branch_subgraph(
            branch,
            [object()],
            branch,
            {"double_1.default": _recording_resolver(calls, "aten", object())},
            {},
            {
                "coreai_metal_kernels::double_1.default": _recording_resolver(
                    calls, "user", sentinel
                )
            },
            None,
        )

        assert calls == ["user"]
        assert out == [sentinel]

    @staticmethod
    @pytest.mark.parametrize("namespace", ["coreai", "coreaix"])
    def test_custom_resolver_handles_coreai_namespaces(namespace: str) -> None:
        """``coreai`` / ``coreaix`` ops route to ``custom_resolver``."""
        op = _make_fake_op(namespace, "quantize.default")
        branch = _single_call_branch(op)
        calls: list[str] = []
        sentinel = object()

        out = convert_branch_subgraph(
            branch,
            [object()],
            branch,
            {},
            {},
            None,
            {"quantize": _recording_resolver(calls, "custom", sentinel)},
        )

        assert calls == ["custom"]
        assert out == [sentinel]

    @staticmethod
    def test_custom_resolver_key_strips_variant_suffix() -> None:
        """The custom resolver is keyed by the variant-stripped target.

        ``dequantize.default`` must look up ``dequantize``; a resolver keyed by
        the full ``dequantize.default`` would (correctly) miss and raise.
        """
        op = _make_fake_op("coreai", "dequantize.default")
        branch = _single_call_branch(op)

        with pytest.raises(KeyError):
            convert_branch_subgraph(
                branch,
                [object()],
                branch,
                {},
                {},
                None,
                {"dequantize.default": _recording_resolver([], "custom", object())},
            )

        calls: list[str] = []
        sentinel = object()
        out = convert_branch_subgraph(
            branch,
            [object()],
            branch,
            {},
            {},
            None,
            {"dequantize": _recording_resolver(calls, "custom", sentinel)},
        )
        assert calls == ["custom"]
        assert out == [sentinel]

    @staticmethod
    @pytest.mark.parametrize("namespace", ["aten", None])
    def test_aten_and_none_namespaces_use_aten_resolver(namespace: str | None) -> None:
        """``aten`` ops and namespace-less ops (e.g. ``operator.getitem``) route
        to ``aten_resolver`` -- unchanged by the new resolver params."""
        op = _make_fake_op(namespace, "add.Tensor")
        branch = _single_call_branch(op)
        calls: list[str] = []
        sentinel = object()

        out = convert_branch_subgraph(
            branch,
            [object()],
            branch,
            {"add.Tensor": _recording_resolver(calls, "aten", sentinel)},
            {},
        )

        assert calls == ["aten"]
        assert out == [sentinel]

    @staticmethod
    def test_user_op_without_resolver_raises() -> None:
        """Backward compat: a custom-namespace op with no ``user_defined_resolver``
        still raises the original ``unsupported op in branch`` error."""
        op = _make_fake_op("coreai_metal_kernels", "double_1.default")
        branch = _single_call_branch(op)

        with pytest.raises(ValueError, match="unsupported op in branch"):
            convert_branch_subgraph(branch, [object()], branch, {}, {})

    @staticmethod
    def test_coreai_op_without_custom_resolver_raises() -> None:
        """Backward compat: a ``coreai`` op with no ``custom_resolver`` raises."""
        op = _make_fake_op("coreai", "quantize.default")
        branch = _single_call_branch(op)

        with pytest.raises(ValueError, match="unsupported op in branch"):
            convert_branch_subgraph(branch, [object()], branch, {}, {})

    @staticmethod
    def test_user_defined_resolver_forwarded_to_nested_handler() -> None:
        """A nested higher-order handler receives ``user_defined_resolver`` and
        ``graph_module`` so custom lowerings resolve inside nested control flow."""
        op = _make_fake_op("higher_order", "cond")
        branch = _single_call_branch(op)
        received: dict[str, Any] = {}
        sentinel = object()
        user_resolver = {"some::op": _recording_resolver([], "user", object())}

        def handler(
            branch_values: dict,
            node: fx.Node,
            *,
            graph_module: fx.GraphModule,
            user_defined_resolver: Any,
        ) -> list[Any]:
            received["graph_module"] = graph_module
            received["user_defined_resolver"] = user_defined_resolver
            return [sentinel]

        out = convert_branch_subgraph(
            branch,
            [object()],
            branch,
            {},
            {"cond": handler},
            user_resolver,
            None,
        )

        assert received["graph_module"] is branch
        assert received["user_defined_resolver"] is user_resolver
        assert out == [sentinel]

    @staticmethod
    def test_unknown_higher_order_op_raises() -> None:
        """A higher-order op with no matching handler raises the branch error."""
        op = _make_fake_op("higher_order", "map_impl")
        branch = _single_call_branch(op)

        with pytest.raises(ValueError, match="unsupported op in branch"):
            convert_branch_subgraph(
                branch,
                [object()],
                branch,
                {},
                {"cond": _recording_resolver([], "cond", object())},
            )


# ---------------------------------------------------------------------------
# Wiring tests (real conversion, spying on the resolver call)
# ---------------------------------------------------------------------------


def _convert_with_branch_spy(
    monkeypatch: pytest.MonkeyPatch, model: torch.nn.Module, args: tuple
) -> tuple[TorchConverter, list[tuple]]:
    """Convert ``model`` while capturing every ``convert_branch_subgraph`` call."""
    captured: list[tuple] = []
    real = aten_to_core.convert_branch_subgraph

    def spy(*call_args: Any, **call_kwargs: Any) -> Any:
        captured.append(call_args)
        return real(*call_args, **call_kwargs)

    monkeypatch.setattr(aten_to_core, "convert_branch_subgraph", spy)

    exported = torch.export.export(model, args)
    ep = exported.run_decompositions(get_decomp_table())
    converter = TorchConverter()
    converter.add_exported_program(ep, output_names=["out"])
    converter.to_coreai()
    assert captured, "convert_branch_subgraph was never called"
    return converter, captured


@pytest.mark.control_flow
class TestBranchResolverWiring:
    """The higher-order handlers thread the resolvers into the branch walker."""

    @staticmethod
    def test_replace_cond_forwards_full_higher_order_resolver(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Secondary fix: ``replace_cond`` passes the full ``_higher_order_resolver``
        (so ``while_loop`` can nest in a ``cond`` branch), not ``{"cond": ...}``."""

        class CondModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                def true_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return a + b

                def false_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return a - b

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x, y])

        _, captured = _convert_with_branch_spy(
            monkeypatch,
            CondModel().eval(),
            (torch.ones(4), torch.full((4,), 0.5)),
        )

        handlers = captured[0][_HIGHER_ORDER_HANDLERS]
        assert handlers is _higher_order_resolver
        # The whole point of the secondary fix: while_loop is reachable from cond.
        assert {"cond", "while_loop", "_yield"} <= set(handlers)

    @staticmethod
    def test_replace_cond_threads_user_defined_and_custom_resolvers(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``replace_cond`` forwards the converter's user lowerings and the
        module-level ``_custom_to_core_resolver``."""

        class CondModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                def true_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return a + b

                def false_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return a - b

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x, y])

        converter, captured = _convert_with_branch_spy(
            monkeypatch,
            CondModel().eval(),
            (torch.ones(4), torch.full((4,), 0.5)),
        )

        call = captured[0]
        assert call[_USER_DEFINED_RESOLVER] is converter._user_defined_torch_lowering
        assert call[_CUSTOM_RESOLVER] is _custom_to_core_resolver

    @staticmethod
    def test_replace_while_loop_threads_resolvers(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``replace_while_loop`` threads the same resolvers into both its
        condition and body branch conversions."""

        class WhileModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                def cond_fn(i: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    return i < 3

                def body_fn(
                    i: torch.Tensor, v: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    return i + 1, v + 1.0

                _, out = torch.ops.higher_order.while_loop(
                    cond_fn, body_fn, (torch.tensor(0), x), tuple()
                )
                return out

        converter, captured = _convert_with_branch_spy(
            monkeypatch, WhileModel().eval(), (torch.ones(4),)
        )

        # cond subgraph + body subgraph -> two branch conversions, both wired.
        assert len(captured) >= 2
        for call in captured:
            assert call[_HIGHER_ORDER_HANDLERS] is _higher_order_resolver
            assert (
                call[_USER_DEFINED_RESOLVER] is converter._user_defined_torch_lowering
            )
            assert call[_CUSTOM_RESOLVER] is _custom_to_core_resolver


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------


class TestResolverSignatures:
    """The new optional parameters exist with backward-compatible defaults."""

    @staticmethod
    def test_convert_branch_subgraph_optional_resolvers_default_none() -> None:
        params = inspect.signature(convert_branch_subgraph).parameters
        assert params["user_defined_resolver"].default is None
        assert params["custom_resolver"].default is None

    @staticmethod
    @pytest.mark.parametrize(
        "handler", [replace_cond, replace_while_loop, replace_yield]
    )
    def test_higher_order_handlers_accept_user_defined_resolver(handler: Any) -> None:
        param = inspect.signature(handler).parameters.get("user_defined_resolver")
        assert param is not None
        assert param.default is None
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
