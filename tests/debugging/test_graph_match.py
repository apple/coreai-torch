# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Tests for the label side of `graph_match`, and for the ambiguity it reports.

`node_labels`, `structural_labels` and `align` are pure functions of a graph, so they
are tested against graphs built by hand: an operation's attributes are read through the
bindings, but only through the three members stubbed here, which keeps these tests
independent of a compiled program.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import pytest

from coreai_torch.debugging.graph_match import (
    UNSTABLE_ATTRIBUTES,
    Ambiguity,
    WeightPolicy,
    _graph_blobs,
    _module_of,
    align,
    node_labels,
    resource_digests,
    structural_labels,
)

DIGEST = WeightPolicy.DIGEST
DIGEST_PORTABLE = WeightPolicy.DIGEST_PORTABLE


@dataclass
class _Attribute:
    """One attribute, as much of it as `_attr_digest` reads."""

    name: str
    attr: Any


@dataclass
class _Operation:
    """An operation's attribute list, as much as `_attr_digest` reads."""

    attributes: list[_Attribute]


def _graph(**attributes: Any) -> nx.DiGraph:
    """
    A one-operation graph whose op carries `attributes`.

    Args:
        attributes: Attribute names and values for the operation.

    Returns:
        The graph.

    """
    graph = nx.DiGraph()
    graph.add_node(
        0,
        type="op",
        op_name="coreai.add",
        index="0",
        ir_object=_Operation(
            attributes=[_Attribute(name, value) for name, value in attributes.items()],
        ),
    )
    return graph


def test_unstable_attribute_is_excluded_by_default() -> None:
    """`sym_name` is regenerated per conversion, so it cannot count towards identity."""
    first = node_labels(_graph(sym_name="main_1", axis="0"))
    second = node_labels(_graph(sym_name="main_2", axis="0"))

    assert first[0] == second[0]


def test_ignore_attributes_can_be_widened() -> None:
    """A named attribute stops being part of identity, in both labels."""
    first = _graph(axis="0")
    second = _graph(axis="1")

    assert node_labels(first)[0] != node_labels(second)[0]
    assert structural_labels(first)[0] != structural_labels(second)[0]

    ignore = UNSTABLE_ATTRIBUTES | {"axis"}
    assert (
        node_labels(first, ignore_attributes=ignore)[0]
        == (node_labels(second, ignore_attributes=ignore)[0])
    )
    assert (
        structural_labels(first, ignore_attributes=ignore)[0]
        == (structural_labels(second, ignore_attributes=ignore)[0])
    )


def test_ignore_attributes_can_be_emptied() -> None:
    """
    An empty set compares everything, which is how to find why two ops fail to match.

    The default hides `sym_name`; asking for nothing to be hidden must reveal it, or the
    knob cannot answer that question.
    """
    first = _graph(sym_name="main_1")
    second = _graph(sym_name="main_2")

    assert node_labels(first)[0] == node_labels(second)[0]
    assert (
        node_labels(first, ignore_attributes=frozenset())[0]
        != (node_labels(second, ignore_attributes=frozenset())[0])
    )


# ---------------------------------------------------------------------------
# Ambiguity
# ---------------------------------------------------------------------------


def _branches(count: int, changed: frozenset[int] = frozenset()) -> nx.DiGraph:
    """
    `count` parallel branches over one shared argument, indistinguishable by default.

    Each branch is `const -> value -> mul(value, arg) -> value`, the shape of a
    weighted layer: under `WeightPolicy.IGNORE` the constants carry no payload to tell
    them apart, so every branch fingerprints alike and the matcher has to choose which
    became which. A branch named in `changed` uses f16 instead of f32, which is a
    difference the labels *can* see.

    Args:
        count: How many branches to build.
        changed: Indices of the branches to build at a different element type.

    Returns:
        The graph.

    """
    graph = nx.DiGraph()
    graph.add_node(
        0, type="value", value_type="arg", ir_type="tensor<4xf32>", index="0"
    )
    for branch in range(count):
        ir_type = "tensor<4xf16>" if branch in changed else "tensor<4xf32>"
        const, product = 100 + branch * 4, 102 + branch * 4
        for node, kind, name in (
            (const, "op", "coreai.const"),
            (product, "op", "coreai.mul"),
        ):
            graph.add_node(node, type=kind, op_name=name, index="0")
        for node in (const + 1, product + 1):
            graph.add_node(
                node, type="value", value_type="result", ir_type=ir_type, index="0"
            )
        graph.add_edge(const, const + 1, edge_type="defines", index=0)
        graph.add_edge(const + 1, product, edge_type="operand", index=0)
        graph.add_edge(0, product, edge_type="operand", index=1)
        graph.add_edge(product, product + 1, edge_type="defines", index=0)

    return graph


def test_no_ambiguity_when_nothing_is_interchangeable() -> None:
    """
    One branch leaves nothing to choose between, so nothing may be reported.

    The signal is only worth reading if its absence means something, and a caution
    attached to every diff is one nobody reads.
    """
    assert not align(_branches(1), _branches(1)).ambiguity
    assert not align(_branches(1), _branches(1, frozenset({0}))).ambiguity


def test_a_tie_break_that_became_a_removal_is_reported() -> None:
    """
    Deleting one of three identical branches: which two survived is the tie-break.

    The case the audit named -- a diff that reports removals with no way to tell that
    it picked *which* by topological order rather than by evidence.
    """
    alignment = align(_branches(3), _branches(2))

    assert alignment.removed, "A deleted branch leaves its nodes unpaired"
    assert alignment.ambiguity.removed, (
        "Every candidate for removal was interchangeable with two others, so the "
        "removal rests on a tie-break and has to say so"
    )


def test_ambiguity_is_a_subset_of_what_it_qualifies() -> None:
    """
    Each set narrows the like-named field, so a caller can subtract one from the other.

    Nothing may be reported ambiguous that the alignment does not report at all: the
    pass that records a tie-break does not get the last word on what became of the
    node, and reporting the outcome it *expected* would put a node in `removed` that
    a later pass went on to pair.
    """
    alignment = align(_branches(4), _branches(2, frozenset({1})))
    ambiguity = alignment.ambiguity

    paired = set(alignment.mapping) | {source for source, _ in alignment.modified}
    assert ambiguity.paired <= paired
    assert ambiguity.removed <= set(alignment.removed)
    assert ambiguity.added <= set(alignment.added)
    assert not ambiguity.paired & ambiguity.removed, "A node has one outcome, not two"
    assert ambiguity.count == len(ambiguity.paired) + len(ambiguity.removed) + len(
        ambiguity.added
    )


def test_empty_ambiguity_is_falsey() -> None:
    """`if diff.ambiguity:` is the check callers will write, so it has to work."""
    assert not Ambiguity()
    assert Ambiguity().count == 0
    assert Ambiguity(removed=frozenset({1}))
    assert Ambiguity(removed=frozenset({1})).count == 1


def test_a_proven_isomorphism_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Two conversions of an unchanged model must stay quiet, however much they chose.

    Every constant of one shape is interchangeable under `IGNORE`, so an unedited
    rebuild makes a great many arbitrary choices -- and `identical` proves each of
    them harmless, the mapping being a complete bijection accounting for every edge.
    Warning anyway would put a caution on the one comparison that needs none.
    """
    with caplog.at_level(logging.WARNING, logger="coreai_torch.debugging.graph_match"):
        alignment = align(_branches(3), _branches(3))

    assert alignment.identical
    assert alignment.ambiguity, "The choices were still made, and are still recorded"
    assert not caplog.records, "A proof of sameness is not a reason to caution"


def test_a_tie_break_behind_a_reported_change_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Otherwise the caution has to reach the reader, and only a log does that.

    Nothing in a diff's own output distinguishes "three ops were deleted" from "three
    of six indistinguishable ops were left over", which is the failure this exists to
    stop.
    """
    with caplog.at_level(logging.WARNING, logger="coreai_torch.debugging.graph_match"):
        align(_branches(3), _branches(2))

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "tie-break" in message
    assert "DIGEST" in message, "A caution with no remedy is just noise"


# ---------------------------------------------------------------------------
# Weight policies
# ---------------------------------------------------------------------------


@dataclass
class _Shaped:
    """A payload's shaped type, as much of it as `_is_parameter` reads."""

    shape: list[int]
    element_type: str = "f32"

    def __str__(self) -> str:
        return f"tensor<{'x'.join(map(str, self.shape))}x{self.element_type}>"


class _Payload(bytearray):
    """
    An inline parameter: bytes that also carry a shaped type.

    A `bytearray` subclass because `_payload_digest` hashes through the buffer
    protocol, which is what distinguishes an inline payload from a resource-backed
    one -- and which no pure-Python stub can offer.
    """

    type: _Shaped
    is_splat = False


def _payload(data: bytes, shape: list[int] | None = None) -> _Payload:
    """An inline float parameter holding `data`."""
    value = _Payload(data)
    value.type = _Shaped(shape or [4])
    return value


@dataclass
class _Resource:
    """A resource-backed parameter: no buffer, only a handle in its printed form."""

    handle: str
    type: _Shaped
    is_splat = False

    def __str__(self) -> str:
        return f"dense_resource<{self.handle}>"


def _weights_of(label_map: dict[int, Any]) -> str:
    """The attribute digest of the one-op graph these tests build."""
    return label_map[0].attributes


def test_ignore_elides_a_payload_but_keeps_its_type() -> None:
    """
    The default's whole bargain: a rebuild's re-initialised weights must not register,
    while a resized layer still must.
    """
    small = _weights_of(node_labels(_graph(value=_payload(b"aaaaaaaa"))))
    other = _weights_of(node_labels(_graph(value=_payload(b"bbbbbbbb"))))
    wider = _weights_of(node_labels(_graph(value=_payload(b"aaaaaaaa", [8]))))

    assert small == other, "Different values, same type: IGNORE must not see it"
    assert "<elided>" in small
    assert small != wider, "A shape change is visible under every policy"


def test_digest_sees_the_values_ignore_elides() -> None:
    """
    The reason to reach for `DIGEST` at all, and what the ambiguity warning offers.

    Under `IGNORE` two models whose weights differ are indistinguishable; the payload
    hash is what tells them apart, and it is computed from the buffer rather than from
    printed IR.
    """
    first = _weights_of(node_labels(_graph(value=_payload(b"aaaaaaaa")), DIGEST))
    second = _weights_of(node_labels(_graph(value=_payload(b"bbbbbbbb")), DIGEST))
    again = _weights_of(node_labels(_graph(value=_payload(b"aaaaaaaa")), DIGEST))

    assert "<sha:" in first, "An inline payload hashes through the buffer protocol"
    assert first != second, "Different bytes must not digest alike"
    assert first == again, "Equal bytes must, or a rebuild reports a false change"


def test_a_resource_payload_falls_back_to_its_handle() -> None:
    """
    A resource-backed payload offers no buffer, so `DIGEST` compares the handle.

    That handle is the compiler's own content hash -- exact within one process, which
    is all `DIGEST` claims. `DIGEST_PORTABLE` is what replaces it with a digest read
    from the module.
    """
    resource = _Resource("resource_123", _Shaped([4]))
    digest = _weights_of(node_labels(_graph(value=resource), DIGEST))
    portable = _weights_of(
        node_labels(_graph(value=resource), DIGEST_PORTABLE, {"resource_123": "beef"})
    )

    assert "resource_123" in digest, "Without blobs, the handle stands in"
    assert "<sha:beef>" in portable, "With blobs, the real digest replaces it"
    assert digest != portable


def test_portable_digest_survives_a_renamed_handle() -> None:
    """
    The point of `DIGEST_PORTABLE`: a handle is seeded per execution.

    Two assets holding byte-identical weights name them differently, so comparing
    handles reports a difference that does not exist. Comparing the blob digests the
    module actually carries does not.
    """
    first = _graph(value=_Resource("resource_111", _Shaped([4])))
    second = _graph(value=_Resource("resource_222", _Shaped([4])))
    blobs = ({"resource_111": "same"}, {"resource_222": "same"})

    assert _weights_of(node_labels(first, DIGEST)) != _weights_of(
        node_labels(second, DIGEST)
    ), "DIGEST compares handles, which is exactly what differs between two assets"
    assert _weights_of(node_labels(first, DIGEST_PORTABLE, blobs[0])) == _weights_of(
        node_labels(second, DIGEST_PORTABLE, blobs[1])
    )


def test_structural_labels_never_carry_a_payload() -> None:
    """
    Pairs are *proposed* on the structural label, so a payload there stops a reshaped
    or retrained op pairing at all -- reported as one op removed and another added,
    with nothing to say which changed.
    """
    first = structural_labels(_graph(value=_payload(b"aaaaaaaa")))[0]
    second = structural_labels(_graph(value=_payload(b"bbbbbbbb", [8])))[0]

    assert first == second
    assert "<elided>" in first.attributes and "tensor<" not in first.attributes


def test_resource_digests_reads_a_module_trailer() -> None:
    """`resource_digests` hashes the blobs a printed module carries, by name."""

    class _Module:
        class operation:  # noqa: N801 - mirrors the bindings' attribute name
            @staticmethod
            def get_asm(large_elements_limit: int | None = None) -> str:
                return (
                    'module { "op"() : () -> () }\n'
                    "{-#\n  dialect_resources: {\n"
                    '    builtin: { resource_1: "0x0100", resource_2: "0x0200" }\n'
                    "  }\n#-}\n"
                )

    digests = resource_digests(_Module())

    assert set(digests) == {"resource_1", "resource_2"}
    assert digests["resource_1"] != digests["resource_2"]
    assert all(len(digest) == 32 for digest in digests.values())


def test_resource_digests_of_an_unprintable_module_is_empty() -> None:
    """
    A module that will not print yields no digests rather than raising.

    `DIGEST_PORTABLE` then degrades to comparing handles, which is `DIGEST`'s
    behaviour -- worse, but not a crash in the middle of a diff.
    """

    class _Unprintable:
        class operation:  # noqa: N801 - mirrors the bindings' attribute name
            @staticmethod
            def get_asm(large_elements_limit: int | None = None) -> str:
                raise RuntimeError("cannot print")

    assert resource_digests(_Unprintable()) == {}


def test_graph_blobs_climbs_to_the_module() -> None:
    """
    `DIGEST_PORTABLE` needs nothing threaded in: the module is a parent hop away.

    A graph with no IR behind it -- built by hand, or from torch FX -- has no module
    to print and must yield no blobs rather than failing.
    """

    @dataclass
    class _Node:
        parent: Any = None

    module = _Node()
    graph = nx.DiGraph()
    graph.add_node(0, type="op", ir_object=_Node(parent=_Node(parent=module)))

    assert _module_of(graph) is module
    # `_branches` carries no `ir_object` at all -- a graph built by hand, or from
    # torch FX, where there is no module to print and nothing to digest.
    assert _module_of(_branches(1)) is None
    assert _graph_blobs(_branches(1)) == {}
