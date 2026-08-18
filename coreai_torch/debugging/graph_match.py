# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Aligning two program graphs, to report what changed.

Answers "which node became which", not "are these isomorphic" -- the diff is built
from the mapping, and a yes/no verdict says nothing about what differs. Linear time,
no search.

Five steps:

1. **Label** each node by what makes it that node, each edge by `(edge_type, index)`.
   The index matters because operand order is semantic: without it `sub(a, b)` and
   `sub(b, a)` look alike. Parameter payloads and symbol uniquifiers are excluded --
   they differ between two conversions of one unchanged model.
2. **Fingerprint** bottom-up over a node's label plus its operands' hashes, so equal
   hashes mean "computes the same thing from the same inputs". Labels alone cannot:
   every `add` in a model shares a label.
3. **Anchor** equal fingerprints, then **propagate** along dataflow slot by slot, so
   an edit costs what the edit is worth rather than the whole cone below it.
4. **Assign** the leftovers by shared neighbours, ignoring position -- the only pass
   that survives a move. See `_assign_residual`.
5. **Verify** every pair against the full label. Steps 2-4 are heuristics; this is
   what makes `identical` a proof. A failed pair is `modified`, not removed-plus-added,
   which would discard the identity and source line of an op that is still there.

Pairs are proposed on `structural_labels` (no result types, no payloads) and verified
against `node_labels`, so an op whose shape changed still pairs and reads as modified.

**Lifetime.** Labels read `ir_object` lazily at `align` time, so a graph is only usable
while its program is alive; dropping the `AIProgram` segfaults the interpreter.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

try:  # `scipy` is a declared dependency; the fallback is for a trimmed install.
    import numpy as _numpy
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except ImportError:  # pragma: no cover
    _numpy = None  # type: ignore[assignment]
    _linear_sum_assignment = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class WeightPolicy(str, Enum):
    """
    Whether a parameter's *values* are part of an op's identity.

    Shapes and dtypes are compared under every policy, so a resized layer is always
    visible; this is only about the numbers.

    `IGNORE` is the default because a rebuild re-initialises parameters, so comparing
    values would report every weight of an unchanged model as changed.

    `DIGEST` compares them, near-free: an inline payload is hashed from its buffer, a
    resource-backed one by its blob name, which is a content hash the compiler already
    computed. Valid only when one process produced both programs -- that name is
    seeded per execution and is serialised into a `.aimodel` (see `_weight_label`).

    `DIGEST_PORTABLE` hashes the blob bytes here instead, which is what makes two
    saved assets comparable, and pays a full print of each module for it (see
    `resource_digests`).
    """

    IGNORE = "ignore"  # Compare shape and dtype; ignore the values.
    DIGEST = "digest"  # Hash the values; trust the compiler's name for blobs.
    DIGEST_PORTABLE = "digest_portable"  # Hash blob bytes too; works across runs.


# `f32`, `f16`, `bf16`. Matched on the element type's name rather than through
# `ir.FloatType`, to keep this module free of a dependency on the bindings: it is a
# pure function of two `nx.DiGraph`s, and its tests build graphs by hand.
_FLOAT_ELEMENT_TYPE = re.compile(r"^(bf|f)\d+$")

# A symbol's *trailing* uniquifier, regenerated on every conversion:
# `@layer_norm_tzcnrgwt` and `@layer_norm_zovsaktp` name the same composite. Left in,
# every `coreai.invoke` reads as changed on every rebuild. The alphabet is `a-z0-9`, not
# the hex `_UUID_SUFFIX_RE` assumes when stripping these for display.
#
# The name part is greedy and the suffix must end the symbol: non-greedy, it ate the
# first long `_word` instead of the last, stripping part of the name and leaving the
# actual hash in place.
_SYMBOL_SUFFIX = re.compile(r"@([A-Za-z_][\w$.]*)_[a-z0-9]{8,}(?![\w$.])")
_SYMBOL_KEPT = r"@\1_"

# `dense_resource<resource_15196384634295780515>`. The name is the compiler's own
# hash of the payload's bytes, which is what lets it stand in for one -- see
# `_weight_label`.
_RESOURCE_REF = re.compile(r"dense_resource<([^>]+)>")

# A blob in a module's printed trailer: `resource_155104…: "0x1000…"`.
_RESOURCE_BLOB = re.compile(r'([A-Za-z_][\w.]*)\s*:\s*"0x([0-9A-Fa-f]+)"')

# A value's *display* name, which is not part of what an op computes.
#
# `coreai.graph` carries `coreai.name = "add_1"` from the traced node, and that name
# shifts whenever anything upstream changes. Left in, the entry-point op fails to pair
# and takes its region, block and block arguments with it, each identified by its parent.
_DISPLAY_NAME = re.compile(r'coreai\.name = "[^"]*"')
_DISPLAY_NAME_KEPT = "coreai.name"

# A graph's own symbol name is left out of its identity: it names a *place*, and which
# graph is compared with which is already decided before a pair reaches here, composites
# pairing through their invoke call sites. Nothing is lost -- the declared name is in
# `composite_decl` and an invoke's callee is a reference.
_UNSTABLE_ATTRS = frozenset({"sym_name"})

# A node that has no label: it belongs to neither graph, or was never labelled.
# Distinct from every real label, since every real node has at least a kind.
_MISSING = "?"
_UNMAPPED = None


@dataclass(frozen=True)
class Alignment:
    """
    Which node became which, and what that leaves over.

    Attributes:
        mapping: Verified source id -> target id, for nodes that correspond exactly.
        removed: Source ids with no counterpart.
        added: Target ids with no counterpart.
        modified: Source ids whose counterpart is the same op wired or configured
            differently, as `(source id, target id)`.
        identical: Whether the two graphs are provably the same graph.

    """

    mapping: dict[int, int] = field(default_factory=dict)
    removed: list[int] = field(default_factory=list)
    added: list[int] = field(default_factory=list)
    modified: list[tuple[int, int]] = field(default_factory=list)
    identical: bool = False

    @property
    def changed(self) -> bool:
        """Whether anything at all differs."""
        return bool(self.removed or self.added or self.modified)


def _is_parameter(value: Any) -> Any | None:
    """
    The shaped type of a payload that is a parameter, or `None` if it is not one.

    A parameter is a non-splat, floating-point tensor of more than one element -- what a
    rebuild re-initialises, so comparing such payloads by value reports every weight of
    an untouched model as changed.

    Everything else is compared by value, integers included: shapes, permutations, axes
    and gather indices are not re-initialised, and size is no safe proxy, since eliding
    a large integer payload would hide a real change to a gather index. A **quantised**
    weight is therefore compared by value too -- nothing distinguishes an `si4` weight
    from a large gather index, and a retrained 4-bit model genuinely has changed.
    Comparing two saved assets of one then needs `DIGEST_PORTABLE`, a resource handle
    being seeded per process. See `_resource_label`.
    """
    shaped = getattr(value, "type", None)
    shape = getattr(shaped, "shape", None)
    if shape is None or getattr(value, "is_splat", False):
        return None

    # More than one element. A dynamic dimension counts as many, since a shape that
    # is not known here is not a scalar.
    if all(0 <= dimension <= 1 for dimension in shape):
        return None

    element_type = str(getattr(shaped, "element_type", ""))
    return shaped if _FLOAT_ELEMENT_TYPE.match(element_type) else None


def _payload_digest(value: Any) -> str | None:
    """
    A hash of a parameter's bytes computed here, or `None` if they are out of reach.

    Inline payloads support the buffer protocol, so they hash directly without being
    printed. Resource-backed ones expose nothing -- no buffer, no accessor, no blob
    manager, and printing the op yields only the handle -- so those return `None` and
    `_weight_label` falls back to the compiler's hash.
    """
    try:
        return hashlib.sha256(memoryview(value)).hexdigest()[:32]
    except TypeError:
        return None


def _weight_label(
    value: Any,
    shaped: Any,
    policy: WeightPolicy,
    blobs: Mapping[str, str],
    types: bool = True,
) -> str:
    """
    What stands in for a parameter's payload.

    Under `IGNORE`, its type alone, so shape and dtype are still compared while the
    values are not. Under `DIGEST`, an inline payload is hashed here from its buffer and
    a resource-backed one is compared by its **blob name** -- the compiler's own hash of
    exactly those bytes (`getBlobNameForData`), which it already trusts to dedup blobs
    without comparing them.

    Two consequences of borrowing that hash, neither avoidable:

    * **Handles are comparable only within one process.** `llvm::hash_value` is seeded
      per execution in assertions builds, by design, and a handle is serialised into a
      `.aimodel` rather than recomputed on load -- so two saved assets name identical
      weights differently. Nothing in the handle reveals this, so the caller must know:
      see `compute_coreai_program_diff`, and `DIGEST_PORTABLE` for the fix.
    * **Inline and resource-backed payloads are digested in different spaces.** Harmless:
      identical bytes imply an identical size and so the same storage form.

    A payload offering neither route warns rather than passing quietly.
    """
    if policy is WeightPolicy.IGNORE:
        # Keep the shape out of the structural label, which a reshaped op has to pair
        # on. Carrying it there leaves every parameter of a widened model unable to
        # pair, reported as added and removed rather than modified.
        return f"<elided> : {shaped}" if types else "<elided>"

    digest = _payload_digest(value)
    if digest is not None:
        return f"<sha:{digest}> : {shaped}"

    text = str(value)
    handle = _RESOURCE_REF.search(text)
    if handle is not None:
        # A digest read out of the module beats the compiler's name, because it does
        # not depend on which process computed it.
        resolved = blobs.get(handle.group(1))
        return f"<sha:{resolved}> : {shaped}" if resolved else f"{text} : {shaped}"

    logger.warning(
        "Cannot compare the values of a %s parameter: it offers no buffer to hash "
        "and no resource handle to compare. Its type is compared instead.",
        shaped,
    )
    return f"{text} : {shaped}"


def _resource_label(value: Any, policy: WeightPolicy, blobs: Mapping[str, str]) -> str:
    """
    A payload compared by value, with any resource handle taken out of the label.

    A handle is a content hash with a **per-execution seed**, serialised into a
    `.aimodel` and never recomputed on load, so two saved assets name byte-identical
    weights differently and a label carrying one reports a difference that does not
    exist. What replaces it depends on what the policy can honestly claim:

    * `DIGEST_PORTABLE` -- the blob's real digest, read from the module. Exact.
    * `DIGEST` -- the handle itself, valid only between two programs one process
      produced, which is what the policy promises and no more.
    * `IGNORE` -- nothing, keeping the type. This path matters because a *quantised*
      weight is an `si4` payload, which `_is_parameter` does not claim, so a 4-bit
      model's weights reach here with their handles intact.

    Args:
        value: The attribute to render.
        policy: What the caller is willing to claim about parameter values.
        blobs: Digests by blob name, empty unless `DIGEST_PORTABLE` is in force.

    Returns:
        The attribute as text, with any handle replaced by something stable.

    """
    text = str(value)
    if policy is WeightPolicy.DIGEST or "dense_resource<" not in text:
        return text

    def substitute(match: re.Match[str]) -> str:
        digest = blobs.get(match.group(1))
        return f"dense_resource<sha:{digest}>" if digest else "dense_resource<elided>"

    return _RESOURCE_REF.sub(substitute, text)


def _attr_digest(
    attrs: dict[str, Any],
    policy: WeightPolicy,
    blobs: Mapping[str, str],
    types: bool = True,
) -> str:
    """
    A digest of an op's own attributes.

    Parameter payloads are left out by default and per-conversion symbol suffixes
    normalised away: both differ between two conversions of one unchanged model, so
    including either reports a change that did not happen. Read through the bindings,
    never by parsing printed IR.
    """
    ir_object = attrs.get("ir_object")
    operation = getattr(ir_object, "operation", ir_object)
    attributes = getattr(operation, "attributes", None)
    if attributes is None:
        return ""

    parts: list[str] = []
    for attribute in attributes:
        name = str(attribute.name)
        if name in _UNSTABLE_ATTRS:
            continue

        shaped = _is_parameter(attribute.attr)
        text = (
            _weight_label(attribute.attr, shaped, policy, blobs, types)
            if shaped is not None
            else _resource_label(attribute.attr, policy, blobs)
        )
        text = _SYMBOL_SUFFIX.sub(_SYMBOL_KEPT, text)
        parts.append(f"{name}={_DISPLAY_NAME.sub(_DISPLAY_NAME_KEPT, text)}")

    return "|".join(sorted(parts))


def resource_digests(module: Any) -> dict[str, str]:
    """
    Hash every resource blob in a module, by the name that module knows it under.

    Makes two `.aimodel` assets comparable: a blob name is seeded per execution, so two
    assets hold different names for identical weights.

    TODO: expose a blob reader in `MLIRModule.cpp` (the inverse of
    `create_elements_attr`) to make `DIGEST` portable and retire this function.

    Args:
        module: The MLIR module to read, as `program._mlir_module`.

    Returns:
        A digest per resource blob name, empty if the module cannot be printed.

    """
    try:
        text = module.operation.get_asm(large_elements_limit=None)
    except Exception:  # noqa: BLE001 - a module we cannot print yields no digests
        logger.warning("Could not print the module; resource digests unavailable")
        return {}

    _, _, trailer = text.rpartition("dialect_resources:")

    return {
        name: hashlib.sha256(payload.encode()).hexdigest()[:32]
        for name, payload in _RESOURCE_BLOB.findall(trailer)
    }


def _module_of(graph: nx.DiGraph) -> Any | None:
    """
    The module a graph's nodes live in, by climbing one node's parents.

    `None` for a graph with no IR behind it -- built by hand, or a torch FX graph --
    where there is nothing to print and nothing to digest.
    """
    for _, attrs in graph.nodes(data=True):
        ir_object = attrs.get("ir_object")
        node = getattr(ir_object, "operation", ir_object)
        if node is None:
            continue

        while (parent := getattr(node, "parent", None)) is not None:
            node = parent

        return node

    return None


def _graph_blobs(graph: nx.DiGraph) -> dict[str, str]:
    """
    Resource digests for whatever module a graph came from.

    So `DIGEST_PORTABLE` needs nothing threaded in: the module is one parent hop from
    any op the graph holds. A program with composites therefore pays the print once
    per graph rather than once per program -- accepted, because the alternative is two
    more parameters on every function between here and the caller.
    """
    module = _module_of(graph)
    return resource_digests(module) if module is not None else {}


def _result_types(attrs: dict[str, Any]) -> str:
    """
    The types an op produces, which is where shape and dtype live.

    What keeps a weight *shape* change visible while its values are ignored: a
    constant's result type carries both.
    """
    ir_object = attrs.get("ir_object")
    operation = getattr(ir_object, "operation", ir_object)
    results = getattr(operation, "results", None)
    if results is None:
        return ""

    try:
        return ",".join(str(result.type) for result in results)
    except Exception:  # noqa: BLE001 - a node without usable results is unlabelled
        return ""


@dataclass(frozen=True)
class Label:
    """
    What makes a node the node it is, field by field.

    Frozen, so it hashes and compares as a unit exactly as the string it replaces
    did -- while a consumer that needs to say *which* part differs can read the field
    by name. It was a `\x1f`-joined string, split back apart by `graph_diff` against
    a tuple of field names kept in that other module: two places to hold in step, and
    a `\x1f` inside any attribute value would have silently shifted every field after
    it.

    Attributes:
        kind: `op`, `value`, `region`, `block` -- what sort of node this is.
        op_name: The operation's name, empty for a node that is not one.
        index: Position among its siblings, which is semantic for a block argument.
        value_type: How a value came about: an op result, a block argument.
        ir_type: A value's own type, where its shape and dtype live.
        attributes: A digest of the op's attributes; see `_attr_digest`.
        results: The types an operation produces.

    """

    kind: str = ""
    op_name: str = ""
    index: str = ""
    value_type: str = ""
    ir_type: str = ""
    attributes: str = ""
    results: str = ""


# A node with no label at all -- absent from the mapping, or never labelled. Every
# real node has at least a kind, so this collides with none of them.
_NO_LABEL = Label()


def _labels(
    graph: nx.DiGraph,
    policy: WeightPolicy,
    blobs: Mapping[str, str],
    types: bool,
) -> dict[int, Label]:
    """The shared body of `node_labels` and `structural_labels`."""
    return {
        node: Label(
            kind=str(attrs.get("type", "")),
            op_name=str(attrs.get("op_name", "")),
            index=str(attrs.get("index", "")),
            value_type=str(attrs.get("value_type", "")),
            ir_type=str(attrs.get("ir_type", "")) if types else "",
            attributes=_attr_digest(attrs, policy, blobs, types),
            results=_result_types(attrs) if types else "",
        )
        for node, attrs in graph.nodes(data=True)
    }


def node_labels(
    graph: nx.DiGraph,
    policy: WeightPolicy = WeightPolicy.IGNORE,
    blobs: Mapping[str, str] | None = None,
) -> dict[int, Label]:
    """
    Everything that makes each node the node it is, as a comparable string.

    The label a pair is *verified* against; `structural_labels` is the weaker key
    pairs are proposed on. Result types are in it because a program graph is
    bipartite and a value's shape and dtype live in its own `ir_type` -- left out,
    every value node in a graph shared one label and could pair with a value of any
    type.

    Reads each node's `ir_object`, so the graph's program must still be alive; see
    the note on lifetime in this module's docstring.

    Args:
        graph: The graph to label.
        policy: Whether parameter values count towards identity.
        blobs: Resource digests for this graph's module, from `resource_digests`.
            Read only under `WeightPolicy.DIGEST_PORTABLE`.

    Returns:
        A label per node id.

    """
    return _labels(graph, policy, blobs or {}, types=True)


def structural_labels(graph: nx.DiGraph) -> dict[int, Label]:
    """
    What makes each node that node *apart from the data flowing through it*.

    The same label without result types, `ir_type` or parameter payloads -- an op's
    kind, name, position and configuration. Those three are precisely what an edit
    changes about an op that is *still the same op*, and a node whose label differs
    cannot pair at all, so including them made every such edit read as one op removed
    and an unrelated one added.

    Pairs proposed on this key are verified against `node_labels`, so over-matching
    costs precision, never correctness.

    Args:
        graph: The graph to label.

    Returns:
        A label per node id.

    """
    return _labels(graph, WeightPolicy.IGNORE, {}, types=False)


# The edge from an operation to a value it produces, which is how a changed value
# node is traced back to the operation responsible for it.
_DEFINES_EDGE = "defines"

# The node kind that stands for an operation. The graph is bipartite -- everything
# else is a value, a region or a block -- and the distinction matters wherever a
# difference has to be attributed to something a reader would recognise.
_OP_NODE = "op"


def defining_op(graph: nx.DiGraph, node_id: int) -> int | None:
    """
    The operation whose result a value node is, if it is one.

    Args:
        graph: The graph the node belongs to.
        node_id: A value node.

    Returns:
        The producing operation, or `None` for a block argument -- a graph input is
        not produced by an operation, so there is nothing to attribute it to.

    """
    for producer, _, data in graph.in_edges(node_id, data=True):
        if (
            data.get("edge_type") == _DEFINES_EDGE
            and graph.nodes[producer].get("type") == _OP_NODE
        ):
            return producer

    return None


def responsible_op(graph: nx.DiGraph, node_id: int) -> int | None:
    """
    The operation a changed node is about: itself, or whatever produced it.

    A program graph is bipartite, so a difference lands on a value node as readily as
    on an op node, and reading only op nodes loses the rest outright -- a rewiring
    inside a callee body put both its modifications on value nodes, and the diff
    reported nothing modified at all.

    Args:
        graph: The graph the node belongs to.
        node_id: Any node.

    Returns:
        The operation to report against, or `None` when nothing can be held
        responsible -- a block argument, a region or a block, none of which an
        operation produced. On a torch FX graph every node is an op, so this is the
        identity.

    """
    if graph.nodes[node_id].get("type") == _OP_NODE:
        return node_id

    return defining_op(graph, node_id)


def _ordered(graph: nx.DiGraph) -> list[int]:
    """Producers before consumers, falling back to id order on a cycle."""
    try:
        return list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        logger.warning("Graph has a cycle; falling back to id order")
        return sorted(graph.nodes())


def _operands(graph: nx.DiGraph, node: int) -> list[tuple[str, int, int]]:
    """Incoming edges as `(edge_type, index, producer)`, in a stable order."""
    return sorted(
        (
            str(data.get("edge_type", "")),
            int(data.get("index", 0)),
            producer,
        )
        for producer, _, data in graph.in_edges(node, data=True)
    )


def _hash(tag: str, label: Label, edges: list[tuple[str, int, str]]) -> str:
    """
    One node's fingerprint: a domain tag, its label, and its neighbours' hashes.

    The tag separates the two hash spaces. `fingerprints` and `co_fingerprints` are
    both compared against candidates from the other side, and a leaf has no operands
    while a terminator has no consumers -- so without a tag the two would agree on
    nodes that are not alike, having hashed the same label over an empty edge list.

    Edges are sorted by their *hashes*, never by node id: numbering is an artefact of
    traversal order and must not reach the hash.
    """
    return hashlib.sha256(repr((tag, label, sorted(edges))).encode()).hexdigest()


def fingerprints(graph: nx.DiGraph, labels: dict[int, Label]) -> dict[int, str]:
    """
    A hash per node covering its label and, recursively, its operands'.

    Equal fingerprints mean two nodes compute the same thing from the same inputs --
    far stronger than sharing an op name, which every `add` in a model does.

    Args:
        graph: The graph to hash.
        labels: Node labels, from `node_labels` or `structural_labels`.

    Returns:
        A hash per node id.

    """
    hashes: dict[int, str] = {}
    for node in _ordered(graph):
        hashes[node] = _hash(
            "operands",
            labels.get(node, _NO_LABEL),
            [
                (edge_type, index, hashes.get(producer, _MISSING))
                for edge_type, index, producer in _operands(graph, node)
            ],
        )

    return hashes


def _consumers(graph: nx.DiGraph, node: int) -> list[tuple[str, int, int]]:
    """Outgoing edges as `(edge_type, index, consumer)`, in a stable order."""
    return sorted(
        (
            str(data.get("edge_type", "")),
            int(data.get("index", 0)),
            consumer,
        )
        for _, consumer, data in graph.out_edges(node, data=True)
    )


@dataclass(frozen=True)
class _Adjacency:
    """
    Every node's neighbours, derived once per graph.

    `nx` rebuilds an edge view on each access, and every pass here wants the same
    two lists per node -- profiled on an 851-node graph, `_operands` was called
    14,580 times and `_consumers` 11,194, all recomputing the same answers inside
    `_propagate`'s loop.

    Slots hold a *list* per `(edge_type, index)`, not one neighbour: a value feeding
    several ops puts them all in `(operand, 0)`, and keeping only one silently
    dropped the rest -- on a 71-node graph four values lost consumers that way, one
    of them three of four.

    Attributes:
        operands: Incoming edges per node, as `(edge_type, index, producer)`.
        consumers: Outgoing edges per node, as `(edge_type, index, consumer)`.
        operand_slots: The same operands, grouped by `(edge_type, index)`.
        consumer_slots: The same consumers, grouped by `(edge_type, index)`.
        context: Each node's neighbours two hops out; see `_context`.

    """

    operands: dict[int, list[tuple[str, int, int]]]
    consumers: dict[int, list[tuple[str, int, int]]]
    operand_slots: dict[int, dict[tuple[str, int], list[int]]]
    consumer_slots: dict[int, dict[tuple[str, int], list[int]]]
    context: dict[int, set[int]]


def _slots(
    edges: list[tuple[str, int, int]],
) -> dict[tuple[str, int], list[int]]:
    """Neighbours grouped by the slot they occupy."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for edge_type, index, neighbor in edges:
        groups[(edge_type, index)].append(neighbor)

    return groups


def _context(
    operands: dict[int, list[tuple[str, int, int]]],
    consumers: dict[int, list[tuple[str, int, int]]],
) -> dict[int, set[int]]:
    """
    Each node's neighbours **two** hops out, ignoring direction and slot.

    Two, not one, because a program graph is bipartite: one hop from an op reaches only
    the values it reads and writes, which are themselves unmatched when a region is
    rearranged. Direction and slot are dropped deliberately -- they are what every other
    pass keys on, and exactly what a relocation changes.
    """
    adjacent = {
        node: {producer for _, _, producer in operands[node]}
        | {consumer for _, _, consumer in consumers[node]}
        for node in operands
    }

    return {
        node: set().union(*(adjacent[near] for near in near_nodes)) - {node}
        if near_nodes
        else set()
        for node, near_nodes in adjacent.items()
    }


def _adjacency(graph: nx.DiGraph) -> _Adjacency:
    """Read every edge once, in the order the comparison relies on."""
    operands = {node: _operands(graph, node) for node in graph.nodes}
    consumers = {node: _consumers(graph, node) for node in graph.nodes}

    return _Adjacency(
        operands=operands,
        consumers=consumers,
        operand_slots={node: _slots(edges) for node, edges in operands.items()},
        consumer_slots={node: _slots(edges) for node, edges in consumers.items()},
        context=_context(operands, consumers),
    )


def co_fingerprints(graph: nx.DiGraph, labels: dict[int, Label]) -> dict[int, str]:
    """
    A hash per node covering its label and, recursively, its *consumers'*.

    The complement of `fingerprints`, and necessary because a producer-side hash
    cannot recognise a node whose inputs changed but whose role did not -- the output
    of a chain that grew a stage still ends the chain. Hashing from the uses inward
    recognises it, so an insertion leaves one node over instead of orphaning
    everything after it.

    Args:
        graph: The graph to hash.
        labels: Node labels, from `node_labels` or `structural_labels`.

    Returns:
        A hash per node id.

    """
    hashes: dict[int, str] = {}
    for node in reversed(_ordered(graph)):
        hashes[node] = _hash(
            "consumers",
            labels.get(node, _NO_LABEL),
            [
                (edge_type, index, hashes.get(consumer, _MISSING))
                for edge_type, index, consumer in _consumers(graph, node)
            ],
        )

    return hashes


@dataclass(frozen=True)
class _Side:
    """
    One graph, and everything the comparison derives from it.

    `align` needs six things per graph, and every one of them was a `source_`/
    `target_` pair of locals threaded separately through the passes -- ten in all,
    where passing the wrong one of a pair is a silent, plausible-looking bug. As one
    value, a pass takes two arguments instead of six and cannot mix them up.

    Attributes:
        graph: The graph itself.
        order: Its nodes, producers before consumers.
        adjacency: Every node's neighbours, read once.
        identity: The full label per node, which pairs are *verified* against.
        structure: The weaker label, which pairs are *proposed* on.
        hashes: Bottom-up fingerprints of the identity labels.

    """

    graph: nx.DiGraph
    order: list[int]
    adjacency: _Adjacency
    identity: dict[int, Label]
    structure: dict[int, Label]
    hashes: dict[int, str]


def _side(graph: nx.DiGraph, weights: WeightPolicy) -> _Side:
    """Everything `align` reads from one graph, derived once."""
    blobs = _graph_blobs(graph) if weights is WeightPolicy.DIGEST_PORTABLE else None
    identity = node_labels(graph, weights, blobs)

    return _Side(
        graph=graph,
        order=_ordered(graph),
        adjacency=_adjacency(graph),
        identity=identity,
        structure=structural_labels(graph),
        hashes=fingerprints(graph, identity),
    )


def _extend_anchor(
    mapping: dict[int, int],
    source_hashes: dict[int, str],
    target_hashes: dict[int, str],
    source_order: list[int],
    target_order: list[int],
    unambiguous: bool = False,
) -> None:
    """
    Anchor whatever is still unmatched, on a second set of hashes.

    `unambiguous` requires the hash to name exactly one node on each side, and is for
    a *weaker* key: there a collision means "indistinguishable under a key that
    deliberately ignores things", not "interchangeable", so choosing arbitrarily among
    them pushes correct pairs out. Pairing ambiguously on the structural key turned a
    clean 16-node block addition into 20 added and 4 removed.
    """
    taken = set(mapping.values())
    by_hash_target: dict[str, list[int]] = defaultdict(list)
    for node in target_order:
        if node not in taken:
            by_hash_target[target_hashes[node]].append(node)

    if unambiguous:
        by_hash_source: dict[str, list[int]] = defaultdict(list)
        for node in source_order:
            if node not in mapping:
                by_hash_source[source_hashes[node]].append(node)

    for node in source_order:
        if node in mapping:
            continue

        digest = source_hashes[node]
        candidates = by_hash_target.get(digest)
        if not candidates:
            continue
        if unambiguous and (len(candidates) > 1 or len(by_hash_source[digest]) > 1):
            continue

        mapping[node] = candidates.pop(0)


def _anchor(
    source_hashes: dict[int, str],
    target_hashes: dict[int, str],
    source_order: list[int],
    target_order: list[int],
) -> dict[int, int]:
    """
    Pair nodes that hash alike.

    Several nodes sharing a fingerprint are interchangeable to anything that only
    reads the graph -- three identical blocks -- so they are paired in topological
    order: deterministic, and for equivalent nodes arbitrary by definition.

    That arbitrary choice is wrong whenever one side has one fewer;
    `_release_interchangeable` corrects it afterwards, from the consumers. Avoiding the
    choice here is worse either way -- deferring ambiguous groups to `_propagate`
    starves it of seeds, and keying leaves on their consumers makes a leaf move whenever
    anything downstream does. Both try to recover an identity the label threw away; the
    ambiguity is a consequence of eliding parameter values, which
    `WeightPolicy.DIGEST` does not do.
    """
    by_hash_source: dict[str, list[int]] = defaultdict(list)
    by_hash_target: dict[str, list[int]] = defaultdict(list)
    for node in source_order:
        by_hash_source[source_hashes[node]].append(node)
    for node in target_order:
        by_hash_target[target_hashes[node]].append(node)

    mapping: dict[int, int] = {}
    for digest, sources in by_hash_source.items():
        targets = by_hash_target.get(digest)
        if not targets:
            continue

        for source, target in zip(sources, targets):
            mapping[source] = target

    return mapping


def _propagate(
    mapping: dict[int, int],
    source: _Side,
    target: _Side,
    source_labels: dict[int, Label],
    target_labels: dict[int, Label],
) -> None:
    """
    Grow the mapping outward from what is already matched, along dataflow.

    The step that makes an edit cost what the edit is worth. A bottom-up fingerprint
    changes for *every* node downstream of a change, so anchoring alone reports the
    whole cone below it -- an edit early in a model costing far more than the same
    edit at the end, for no reason but how much graph sits after it.

    What rescues those nodes is that they are the same ops in the same slots, reached
    from a neighbour already matched. So each matched pair hands its neighbours to
    `try_pair`, keyed on `(edge_type, index)` -- both operands and consumers, so a
    chain matches from either end -- and only the node whose operands genuinely
    differ is left over.

    `try_pair` never steals: a node already mapped stays mapped, which is what lets
    this run repeatedly, on either label, with the earlier and more exact passes
    always winning.

    Driven by a worklist rather than by rescanning the mapping to a fixpoint. Only a
    *newly* paired node can place a neighbour that could not be placed before, so
    rescanning pairs that have already been examined finds nothing -- it just costs a
    pass over the whole mapping for every round.
    """
    taken = set(mapping.values())
    pending = deque(mapping.items())

    def try_pair(source: int, target: int) -> None:
        if source in mapping or target in taken:
            return
        if source_labels.get(source) != target_labels.get(target):
            return

        mapping[source] = target
        taken.add(target)
        pending.append((source, target))

    while pending:
        source_node, target_node = pending.popleft()
        # Operands, then consumers keyed the same way, so a chain matches from
        # either end.
        for source_slots, target_slots in (
            (source.adjacency.operand_slots, target.adjacency.operand_slots),
            (source.adjacency.consumer_slots, target.adjacency.consumer_slots),
        ):
            neighbours = target_slots.get(target_node, {})
            for slot, sources in source_slots.get(source_node, {}).items():
                for producer, image in zip(sources, neighbours.get(slot, ())):
                    try_pair(producer, image)


def _interchangeable(side: _Side) -> set[int]:
    """
    Nodes that nothing in the graph tells apart -- only the pairing does.

    A node qualifies when its label is shared *and* it has no operands, or all of its
    operands qualify. A leaf with a shared label is genuinely indistinguishable:
    verification reads its label, which is equal, and its operands, of which it has
    none. So one may stand in for another, and the cascade carries that through the
    value a leaf defines.

    Under `IGNORE` this is exactly the parameter constants of one shape -- eliding the
    values is what makes them interchangeable. Requiring a shared label keeps the
    cascade to nodes there is really a choice about.

    Args:
        side: The graph to scan, with its labels and adjacency.

    Returns:
        The ids of the nodes that are interchangeable with some other node.

    """
    counts = Counter(side.identity.values())
    released: set[int] = set()
    for node in side.order:
        if counts[side.identity.get(node, _NO_LABEL)] < 2:
            continue

        operands = side.adjacency.operands[node]
        if not operands or all(producer in released for _, _, producer in operands):
            released.add(node)

    return released


def _release_interchangeable(
    mapping: dict[int, int],
    source: _Side,
    target: _Side,
) -> list[tuple[int, int]]:
    """
    Unpair the interchangeable nodes, so their consumers can place them instead.

    Anchoring pairs a group of interchangeable nodes in topological order, which is
    arbitrary -- and wrong for every node after the missing one as soon as a side has one
    fewer, so an edit to one op reads as several modified.

    The choice is only arbitrary *in isolation*. By now the ops that use these nodes are
    matched, and an operand slot on a matched pair says which node belongs to which, so
    the pairing is dropped and `_propagate` redoes it from that context. Doing this
    *after* propagation is what makes it work: deferring the same group beforehand leaves
    propagation without seeds to grow from.

    Returns what it released so `_restore_released` can put back whatever nothing
    claimed. `_propagate` re-pairs only where labels are *equal*, so a released pair
    whose labels differ -- `tensor<24xf32>` against `tensor<32xf32>` in a widened model --
    can never be re-made by it, and without the restore is lost outright.

    Args:
        mapping: The candidate mapping, modified in place.
        source: The "before" graph.
        target: The "after" graph.

    Returns:
        The pairs removed from `mapping`, in the order they were removed.

    """
    sources = _interchangeable(source)
    if not sources:
        return []

    targets = _interchangeable(target)
    released = [
        (source_node, target_node)
        for source_node, target_node in mapping.items()
        if source_node in sources and target_node in targets
    ]
    for source_node, _ in released:
        del mapping[source_node]

    return released


def _restore_released(mapping: dict[int, int], released: list[tuple[int, int]]) -> None:
    """
    Put back any released pair that nothing else claimed.

    So releasing can only ever improve the mapping: a pair the consumers re-made
    differently is left as they made it, and one nobody wanted returns to what
    anchoring had chosen. Both sides must still be free -- restoring over a pair
    made since would undo the very repair the release was for.
    """
    taken = set(mapping.values())
    for source, target in released:
        if source in mapping or target in taken:
            continue

        mapping[source] = target
        taken.add(target)


def _assign_residual(
    mapping: dict[int, int],
    source: _Side,
    target: _Side,
) -> None:
    """
    Pair what is left over by best overall agreement, ignoring position entirely.

    The general answer to a *relocation*, and the only pass that does not assume
    something stayed put: every earlier pass keys on the cone below, the cone above or
    the operand slot, and a move breaks all three at once, leaving an op that plainly
    still exists paired with nothing.

    What survives a move is *company* -- a relocated op still sits among the same ops.
    So a pair is scored on how much of its 2-hop neighbourhood is already matched to the
    other's (Jaccard), and the assignment maximising the total is taken: optimal, and
    with no threshold to tune.

    Only nodes with **equal identity labels** may pair, so this cannot invent a
    correspondence between different ops, and only leftovers are considered, so it cannot
    overrule a pass with better evidence. On a model with no relocation it pairs nothing.

    Falls back to `_greedy_assignment` if `scipy` is unavailable, costing optimality on
    the rare row with two plausible partners, not correctness.
    """
    unmatched_source = [node for node in source.order if node not in mapping]
    if not unmatched_source:
        return

    taken = set(mapping.values())
    unmatched_target = [node for node in target.order if node not in taken]
    if not unmatched_target:
        return

    # A node's company, expressed in the *other* graph's terms where it is known.
    company = {
        node: {
            mapping[near] for near in source.adjacency.context[node] if near in mapping
        }
        for node in unmatched_source
    }

    scores: dict[tuple[int, int], float] = {}
    for source_node in unmatched_source:
        label = source.identity.get(source_node)
        known = company[source_node]
        if not known:
            continue

        for target_node in unmatched_target:
            if target.identity.get(target_node) != label:
                continue

            context = target.adjacency.context[target_node]
            shared = known & context
            if shared:
                scores[(source_node, target_node)] = len(shared) / len(known | context)

    for source_node, target_node in _best_assignment(
        scores, unmatched_source, unmatched_target
    ):
        mapping[source_node] = target_node


def _best_assignment(
    scores: dict[tuple[int, int], float],
    sources: list[int],
    targets: list[int],
) -> list[tuple[int, int]]:
    """
    The set of pairs with the greatest total score, at most one per node.

    Optimal rather than greedy because the scores sit close together, so a greedy walk
    takes whichever plausible partner it meets first rather than the best one.
    `linear_sum_assignment` is the Hungarian algorithm (Kuhn, 1955), which maximises
    the total and has no threshold to tune.
    """
    if not scores:
        return []

    # Only nodes that scored at all can be assigned, and the scores are sparse: a
    # node is comparable with the few that share its label, not with every leftover.
    # Ordered by `sources`/`targets` so the matrix is deterministic.
    scored_sources = {source_node for source_node, _ in scores}
    scored_targets = {target_node for _, target_node in scores}
    rows = [node for node in sources if node in scored_sources]
    columns = [node for node in targets if node in scored_targets]

    if _linear_sum_assignment is None:  # pragma: no cover - scipy is a dependency
        return _greedy_assignment(scores)

    row_of = {node: index for index, node in enumerate(rows)}
    column_of = {node: index for index, node in enumerate(columns)}
    matrix = _numpy.zeros((len(rows), len(columns)))
    for (source_node, target_node), score in scores.items():
        matrix[row_of[source_node], column_of[target_node]] = score

    return [
        (rows[row], columns[column])
        for row, column in zip(*_linear_sum_assignment(-matrix))
        if matrix[row, column] > 0
    ]


def _greedy_assignment(
    scores: dict[tuple[int, int], float],
) -> list[tuple[int, int]]:
    """
    Best-first pairing, for when `scipy` is not installed.

    Costs optimality on a row with two plausible partners, not correctness: every
    pair it makes is one the optimal assignment could also have made.
    """
    chosen_sources: set[int] = set()
    chosen_targets: set[int] = set()
    pairs = []
    for source_node, target_node in sorted(
        scores, key=lambda pair: (-scores[pair], pair)
    ):
        if source_node in chosen_sources or target_node in chosen_targets:
            continue

        chosen_sources.add(source_node)
        chosen_targets.add(target_node)
        pairs.append((source_node, target_node))

    return pairs


def _verify(
    mapping: dict[int, int],
    source: _Side,
    target: _Side,
) -> tuple[dict[int, int], list[tuple[int, int]]]:
    """
    Split a candidate mapping into pairs that correspond exactly and pairs that do not.

    The step that makes the rest safe to be heuristics. Anchoring and propagation
    both *guess*; nothing before this point has checked a guess. A pair survives only
    if its full labels agree and its operands correspond one for one -- same slot, and
    the producer this mapping pairs it with -- and anything else is `modified` rather
    than quietly accepted, which is exactly the defect that let a rewiring report as
    no change at all.

    Operands are compared as multisets, not as a slot dict, so two operands sharing a
    slot are both checked instead of one shadowing the other.
    """
    verified: dict[int, int] = {}
    modified: list[tuple[int, int]] = []

    for source_node, target_node in mapping.items():
        if source.identity.get(source_node) != target.identity.get(target_node):
            modified.append((source_node, target_node))
            continue

        mapped = Counter(
            (edge_type, index, mapping.get(producer, _UNMAPPED))
            for edge_type, index, producer in source.adjacency.operands[source_node]
        )
        if mapped == Counter(target.adjacency.operands[target_node]):
            verified[source_node] = target_node
        else:
            modified.append((source_node, target_node))

    return verified, modified


def _is_isomorphism(
    verified: dict[int, int],
    modified: list[tuple[int, int]],
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
) -> bool:
    """
    Whether a verified mapping proves the two graphs are the same graph.

    The whole argument: every pair in `verified` maps each operand to the
    corresponding operand, so every source edge maps to a distinct target edge. If the
    mapping is also a node bijection and the edge counts agree, that injection is onto
    -- an isomorphism, under labels.

    Comparing the two fingerprint *multisets* instead is not sound: a bottom-up
    fingerprint says nothing about how often a value is consumed, so a duplicated
    constant with one copy dead hashes exactly like a shared one.
    """
    node_count = source_graph.number_of_nodes()

    return (
        not modified
        and len(verified) == node_count
        and target_graph.number_of_nodes() == node_count
        and len(set(verified.values())) == node_count
        and source_graph.number_of_edges() == target_graph.number_of_edges()
    )


def align(
    source_graph: nx.DiGraph,
    target_graph: nx.DiGraph,
    weights: WeightPolicy = WeightPolicy.IGNORE,
) -> Alignment:
    """
    Which node of `source_graph` became which node of `target_graph`.

    Args:
        source_graph: The "before" graph.
        target_graph: The "after" graph.
        weights: Whether a parameter's values are part of an op's identity.
            `IGNORE` by default: parameters are re-initialised on every rebuild, so
            comparing their values reports every diff as a total rewrite. Shapes and
            dtypes are compared either way. `DIGEST` compares the values too, at
            no meaningful cost, but only between programs this process produced;
            `DIGEST_PORTABLE` also works for programs loaded from disk, and pays a
            print of each graph's module for it -- see `resource_digests`.

    Returns:
        An `Alignment`. `identical` is a certificate rather than a guess: it holds
        only when the verified mapping is a complete bijection that accounts for
        every edge, which is a proof the graphs are the same graph.

    """
    # Two labels per node, and the difference between them is the point. Pairs are
    # proposed on the structural label, so an op that is still the same op is
    # recognised when its shape or its weights moved; they are verified against the
    # full label, so such a pair is reported as *modified* rather than accepted.
    # Proposing on the full label made a reshaped op read as an unrelated op removed
    # and another added, leaving nothing to say which one changed.
    source = _side(source_graph, weights)
    target = _side(target_graph, weights)

    # Pass 1, the exact key: anchor equal fingerprints, from the producers up and
    # then from the uses inward, and grow both outward along dataflow. Every pair
    # made here is exact, which is what makes it safe to grow from.
    candidate = _anchor(source.hashes, target.hashes, source.order, target.order)
    _extend_anchor(
        candidate,
        co_fingerprints(source.graph, source.identity),
        co_fingerprints(target.graph, target.identity),
        source.order,
        target.order,
    )

    # Propagation runs *before* the weaker key, not after: growing from exact pairs
    # places the neighbours of a change correctly and leaves the structural pass less
    # to guess at. Propagating on the weaker key here instead turned a clean 16-node
    # block addition into 20 added and 4 removed, because once types are ignored a
    # slot-mate of the wrong shape becomes a candidate.
    _propagate(candidate, source, target, source.identity, target.identity)

    # Pass 2, the weaker key: whatever the exact key could not place, but only where
    # it names exactly one node on each side. A collision here means "indistinguishable
    # once types are ignored", which is not the same as "interchangeable".
    for weak in (fingerprints, co_fingerprints):
        _extend_anchor(
            candidate,
            weak(source.graph, source.structure),
            weak(target.graph, target.structure),
            source.order,
            target.order,
            unambiguous=True,
        )

    _propagate(candidate, source, target, source.structure, target.structure)

    # Pass 3: redo the one choice nothing informed -- which of a group of genuinely
    # interchangeable nodes is which -- now that their consumers are matched and can
    # answer it. Then re-anchor on the identity fingerprints, so a released node no
    # consumer claimed (a dead constant) lands where it would have before.
    released = _release_interchangeable(candidate, source, target)
    _propagate(candidate, source, target, source.identity, target.identity)
    _extend_anchor(candidate, source.hashes, target.hashes, source.order, target.order)
    _restore_released(candidate, released)

    # Pass 4, the last resort: whatever is still unpaired, matched on the company it
    # keeps rather than on where it sits. This is what recognises a *move*.
    _assign_residual(candidate, source, target)

    verified, modified = _verify(candidate, source, target)

    matched_targets = set(verified.values()) | {target for _, target in modified}
    matched_sources = set(verified) | {source for source, _ in modified}

    return Alignment(
        mapping=verified,
        removed=[node for node in source.order if node not in matched_sources],
        added=[node for node in target.order if node not in matched_targets],
        modified=modified,
        identical=_is_isomorphism(verified, modified, source.graph, target.graph),
    )
