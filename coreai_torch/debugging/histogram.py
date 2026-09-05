# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
What a program is made of, counted by operation kind.

The quickest read on an unfamiliar program, and the one that needs nothing compiled or
loaded. Composite bodies are described in their own right rather than folded into the
total, since a body invoked twice is still one body.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TextIO

from coreai._compiler.ir import Operation
from coreai.authoring import AIProgram
from typing_extensions import Self

from .annotations import _DETAIL_STYLE, _HEADING_STYLE
from .table_writer import _TreeNode, _write_tree
from .utils import _collect_entry_points, _composite_label

_INVOKE_PREFIX = "coreai.invoke -> "
"""Prefix of the count key standing for an invoke of a named composite."""


@dataclass(frozen=True)
class CompositeHistogram:
    """One composite a program invokes, and what its body contains."""

    label: str
    """Display name."""

    symbol: str
    """The symbol as it appears in the IR, randomised suffix and all."""

    invocations: int
    """How many `coreai.invoke` operations in the *enclosing* body name this composite."""

    histogram: OperationHistogram
    """What the body contains -- its own operations, and the composites it invokes."""

    @property
    def size(self) -> int:
        """Operations in the body, its own composites' bodies included."""
        return self.histogram.size

    def to_dict(self: Self) -> dict[str, Any]:
        """
        This composite as JSON-safe data.

        The size is not repeated here -- it is :attr:`histogram`'s, one level down.

        Returns:
            The label, symbol, invocation count and the body's own report.

        """
        return {
            "label": self.label,
            "symbol": self.symbol,
            "invocations": self.invocations,
            "histogram": self.histogram.to_dict(),
        }


@dataclass(frozen=True)
class OperationHistogram:
    """What a program is made of, with composite bodies kept separate."""

    counts: dict[str, int]
    """This body's own operations, commonest first. An invoke appears here as
    ``coreai.invoke -> <label>``, so which composite is called is visible without
    cross-referencing :attr:`composites`."""

    composites: tuple[CompositeHistogram, ...]
    """The composites this body invokes, largest first. A composite invoked from two
    bodies appears under each, since the question is what each body contains."""

    @property
    def size(self) -> int:
        """Operations in this body and in every body it invokes.

        Counts the invokes themselves too -- a call is an operation. A body reached
        through two levels of invocation is counted once per invocation of each ancestor,
        which is the work performed.
        """
        return sum(self.counts.values()) + sum(
            composite.size * composite.invocations for composite in self.composites
        )

    def to_dict(self: Self) -> dict[str, Any]:
        """
        This histogram as JSON-safe data.

        Hand-written for one reason only, unlike the module's other reports: `asdict`
        already yields JSON here, but drops :attr:`size`, which is a property. That is the
        number worth carrying -- a reader recomputing it has to rediscover that a nested
        body counts once per invocation of each ancestor, and that the invokes count too.

        Returns:
            The total, this body's counts, and one entry per composite.

        """
        return {
            "size": self.size,
            "counts": dict(self.counts),
            "composites": [composite.to_dict() for composite in self.composites],
        }

    def write_summary(
        self: Self,
        output: TextIO | None = None,
        *,
        folded: bool = False,
    ) -> None:
        """
        Write what the program is made of, commonest operation first.

        Args:
            output: Text stream to write to. Defaults to ``sys.stdout`` when None.
            folded: Write only this body's own operations, with each composite standing
                as its one ``coreai.invoke`` row, instead of a tree. For the top-level
                operation mix, where what a composite expands to is noise.

        """
        if folded:
            root = _TreeNode(
                label=f"{sum(self.counts.values())} operations at the top level",
                style=_HEADING_STYLE,
            )
            _add_counts(root, self.counts)
        else:
            root = _TreeNode(
                label=f"{self.size} operations, "
                f"{sum(self.counts.values())} in the entry point",
                style=_HEADING_STYLE,
            )
            _add_histogram(root, self)
        _write_tree(root, output)


def _add_counts(
    node: _TreeNode,
    counts: Mapping[str, int],
    bodies_per_label: Mapping[str, int] | None = None,
) -> None:
    """
    Add one operation-count line per entry to *node*.

    Args:
        node: Node to add beneath.
        counts: Operation name to count, in display order.
        bodies_per_label: Composite label to how many distinct bodies carry it. Used to
            mark the invoke lines where a count aggregates several bodies rather than
            counting one body several times.

    """
    for name, count in counts.items():
        node.add(f"{count:5}  {name}{_body_note(name, bodies_per_label)}")


def _body_note(name: str, bodies_per_label: Mapping[str, int] | None) -> str:
    """
    Note that an invoke count spans several bodies, when it does.

    Externalization emits one graph per call site, so a module called twice yields two
    bodies with the same label and different symbols. Their counts aggregate under one
    line, which otherwise reads as one body invoked twice -- a different thing, and the
    one that matters for code size.

    Args:
        name: The count line's operation name.
        bodies_per_label: Composite label to how many distinct bodies carry it.

    Returns:
        A parenthesised note, or the empty string when there is nothing to disambiguate.

    """
    if bodies_per_label is None or not name.startswith(_INVOKE_PREFIX):
        return ""
    bodies = bodies_per_label.get(name[len(_INVOKE_PREFIX) :], 1)
    return f"  ({bodies} distinct bodies)" if bodies > 1 else ""


def _add_histogram(node: _TreeNode, histogram: OperationHistogram) -> None:
    """
    Add *histogram*'s own operations to *node*, then a subtree per composite.

    Args:
        node: Node to add beneath.
        histogram: The body to describe.

    """
    bodies_per_label: Counter[str] = Counter(
        composite.label for composite in histogram.composites
    )
    _add_counts(node, histogram.counts, bodies_per_label)
    for composite in histogram.composites:
        invoked = (
            "" if composite.invocations == 1 else f" invoked {composite.invocations}x,"
        )
        child = node.add(
            _TreeNode(
                label=f"{composite.label}  ({composite.symbol}){invoked}"
                f"  {composite.size} operations",
                style=_DETAIL_STYLE,
            )
        )
        _add_histogram(child, composite.histogram)


def operation_histogram(program: AIProgram) -> OperationHistogram:
    """
    How many of each kind of operation a program contains, commonest first.

    Composite bodies are reported separately rather than folded in. Flattening them loses
    which operations belong to which composite, and hides that one body invoked twice is
    one body: a caller asking about code size and a caller asking about executed work
    need different answers, and only the split can give both. Each body's own operations
    stay in its :attr:`OperationHistogram.counts`, with an invoke standing for the call;
    :attr:`OperationHistogram.size` totals a body and everything it reaches.

    Args:
        program: Program to describe.

    Returns:
        The entry point's operations and each composite's, as an
        :class:`OperationHistogram`.

    """
    module = program._mlir_module  # noqa: SLF001
    entry_points = _collect_entry_points(module)

    invoked: set[str] = set()
    bodies: dict[str, tuple[Counter[str], Counter[str]]] = {}
    for symbol, graph_op in entry_points.items():
        counts, calls = _body_counts(graph_op, entry_points)
        bodies[symbol] = (counts, calls)
        invoked.update(calls)

    # A graph nothing invokes is an entry point; the rest are composite bodies. Deciding
    # this from the invokes rather than from the name means a composite registered under
    # any name is still recognised.
    roots = [symbol for symbol in bodies if symbol not in invoked]

    own: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    for symbol in roots:
        root_counts, root_calls = bodies[symbol]
        own.update(root_counts)
        calls.update(root_calls)
    return _histogram(own, calls, bodies, entry_points)


def _body_counts(
    graph_op: Any, entry_points: Mapping[str, Any]
) -> tuple[Counter[str], Counter[str]]:
    """
    One graph's operations, and the composites it invokes.

    Args:
        graph_op: The `coreai.graph` to read.
        entry_points: `sym_name -> GraphOp`, as `_collect_entry_points` returns.

    Returns:
        Operation counts (an invoke counted as ``coreai.invoke -> <label>``), and how
        many invokes name each callee symbol.

    """
    counts: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    for region in graph_op.regions:
        for block in region:
            for operation in block.operations:
                if operation.name != "coreai.invoke":
                    counts[operation.name] += 1
                    continue
                callee = _invoke_callee(operation)
                # A callee this module defines no `coreai.graph` for has no body to
                # describe, so it is counted as a plain invoke. Counting it as a call
                # meant `_histogram` looked the symbol up in `bodies` and raised
                # KeyError -- an invoke naming a `func.func`, or an external symbol,
                # crashed the whole histogram rather than being reported as itself.
                if callee is None or callee not in entry_points:
                    counts[operation.name] += 1
                    continue
                calls[callee] += 1
                counts[f"{_INVOKE_PREFIX}{_composite_label(callee, entry_points)}"] += 1
    return counts, calls


def _histogram(
    counts: Counter[str],
    calls: Counter[str],
    bodies: Mapping[str, tuple[Counter[str], Counter[str]]],
    entry_points: Mapping[str, Any],
) -> OperationHistogram:
    """
    Assemble one body's histogram, recursing into the composites it invokes.

    Recurses without a depth or cycle guard: a `coreai.invoke` names a symbol the
    module defines, and the graphs form a DAG, so the walk terminates.

    Every symbol in *calls* has an entry in *bodies*: `_body_counts` records a call
    only for a callee `entry_points` holds, and *bodies* is keyed by exactly those.

    Args:
        counts: The body's own operation counts.
        calls: How many invokes in this body name each callee.
        bodies: Every graph's counts and calls, keyed by symbol.
        entry_points: `sym_name -> GraphOp`, for labelling.

    Returns:
        The body's :class:`OperationHistogram`.

    """
    composites: list[CompositeHistogram] = []
    for symbol, count in calls.items():
        body_counts, body_calls = bodies[symbol]
        composites.append(
            CompositeHistogram(
                label=_composite_label(symbol, entry_points),
                symbol=symbol,
                invocations=count,
                histogram=_histogram(body_counts, body_calls, bodies, entry_points),
            )
        )
    composites.sort(key=lambda use: (-use.size, use.label))
    return OperationHistogram(
        counts=dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        composites=tuple(composites),
    )


def _invoke_callee(operation: Operation) -> str | None:
    """The symbol name a `coreai.invoke` calls, or None when it cannot be read."""
    try:
        callee = str(operation.attributes["callee"])
    except (KeyError, RuntimeError):
        return None
    match = re.search(r"<@([^>]+)>", callee)
    return match.group(1) if match is not None else None
