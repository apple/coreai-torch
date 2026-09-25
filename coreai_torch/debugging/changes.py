# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
What changed between two programs, attributed to the module and line it changed in.

`graph_diff` answers *which node became which*; this answers the question a reader
actually asked, which is *what did I change and where*. The two are not the same report:
a node id names nothing a user wrote, and a diff of four thousand nodes is not read at
all unless it is grouped.

Three things stand between the correspondence and that answer, and each is a measured
defect in the naive version rather than a refinement:

* **A difference lands on a value node as readily as on an operation.** Reported as-is,
  a rewiring inside a callee body says nothing was modified. Resolved through
  `responsible_op`.
* **Most differing pairs are consequences of one edit, not edits.** An operation's
  operands are other operations, so inserting one changes every neighbour's slots. See
  `_is_independent_change`.
* **Which instance of an interchangeable operation is left unmapped is arbitrary.**
  Adding a third block reports its extra constant against `Block$1` as readily as
  `Block$3`. See `_relocate_to_new_modules`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TextIO

import networkx as nx  # type: ignore[import-untyped]
from coreai.authoring import AIProgram
from typing_extensions import Self

from .debug_info import _build_coreai_op_map, get_operation_id
from .graph_diff import (
    GraphDiff,
    OpDiffType,
    _is_one_op_changed,
    compute_per_graph_diff,
)
from .graph_match import (
    _OP_NODE,
    WeightPolicy,
    node_labels,
    responsible_op,
)
from .modules import (
    UNATTRIBUTED,
    Attribution,
    ModuleNode,
    ModulePath,
    attributions_by_op_id,
    build_module_tree,
)
from .table_writer import _Column, _Row, _TableSpec, _write_table
from .utils import STRUCTURAL_OPS, _plain

SOURCE = "source"
"""The "before" program: the first argument to :func:`compute_changes`."""

TARGET = "target"
"""The "after" program. A modification is reported against this side, because that is
where the operation still exists."""

_UNMAPPED = -1
"""Stands for "this node has no counterpart" where a node id is expected. Node ids come
from a counter over the whole graph build, so a negative value is never one."""


def _short_op_name(op_name: str) -> str:
    """
    Drop the dialect prefix from an operation name.

    Args:
        op_name: Operation name as the IR gives it, e.g. ``"coreai.matmul"``.

    Returns:
        The name without its ``coreai.`` prefix, which every operation shares and which
        therefore distinguishes nothing.

    """
    return op_name.split(".", 1)[1] if op_name.startswith("coreai.") else op_name


@dataclass(frozen=True)
class OpChange:
    """One operation that differs between two programs."""

    change: OpDiffType
    """What differs about it. Never `OpDiffType.ALIGNED`: an aligned operation is not a
    change. `MODIFIED` is its own kind rather than a removal plus an addition, because a
    rewired operation is the same operation, in the same place in the source, wired
    differently -- collapsing it would discard both its identity and its location."""

    op_name: str
    """Operation name as the IR gives it, dialect prefix included."""

    side: str
    """:data:`SOURCE` or :data:`TARGET` -- which program it belongs to, so a caller
    knows which one to look it up in. Op ids are unique within a program and mean
    nothing across two."""

    op_id: int | None
    """Its op id on :attr:`side`, or None when its location carries none."""

    attribution: Attribution | None
    """Where it came from, or None when the program recorded nothing for it."""

    graph_label: str
    """Which graph it is in -- ``"main"``, or a composite's label."""

    @property
    def module(self: Self) -> ModulePath:
        """
        Module instance path this change belongs under.

        Returns:
            The path, outermost frame first, or empty when nothing was attributed.

        """
        return self.attribution.module if self.attribution else ()

    @property
    def module_label(self: Self) -> str:
        """
        The module path as one string.

        Returns:
            ``"Block$2/Linear$1"``, or :data:`UNATTRIBUTED`.

        """
        return self.attribution.label if self.attribution else UNATTRIBUTED

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Return the change as plain values.

        Returns:
            What changed, which operation, on which side and in which graph, and the
            attribution flattened in so a caller reading one row does not have to
            follow a nested object to learn where it was.

        """
        attributed = self.attribution.to_dict() if self.attribution else {}
        return {
            "change": _plain(self.change),
            "op": _short_op_name(self.op_name),
            "op_name": self.op_name,
            "side": self.side,
            "op_id": self.op_id,
            "graph": self.graph_label,
            "module": attributed.get("module", UNATTRIBUTED),
            "module_path": attributed.get("module_path", []),
            "source": attributed.get("source"),
        }


@dataclass(frozen=True)
class _OpSite:
    """One operation of one program, as `_relocate_to_new_modules` needs it."""

    op_id: int
    op_name: str
    attribution: Attribution


def _sites(program: AIProgram) -> dict[int, _OpSite]:
    """
    Every attributable operation of a program, by op id.

    Args:
        program: Program to index.

    Returns:
        Op id to its name and attribution. Structural operations are left out: they
        have no module and are not a change a user made -- see `utils.STRUCTURAL_OPS`.

    """
    attributions = attributions_by_op_id(program)
    sites = {}
    for op_id, operation in _build_coreai_op_map(program).items():
        if operation.name in STRUCTURAL_OPS:
            continue
        attribution = attributions.get(op_id)
        if attribution is None:
            continue
        sites[op_id] = _OpSite(
            op_id=op_id, op_name=operation.name, attribution=attribution
        )
    return sites


def _responsible_ops(graph: nx.DiGraph, node_ids: Iterable[int]) -> list[int]:
    """
    The operations a set of changed nodes is about, operation nodes first.

    A program graph is bipartite, so an operation's result is a node of its own and a
    difference can land on either. Reading only the operation nodes loses the rest
    outright: on an externalized model a rewiring inside a callee body puts both
    modifications on value nodes, and the report is then that nothing was modified while
    six unrelated constants were added and removed.

    A value stands for the operation that produced it, so that is where it is reported.
    Operation nodes come first because the caller deduplicates in order and an
    operation's own entry is the one worth keeping -- a value only ever restates what
    its producer did.

    Args:
        graph: The graph the nodes belong to.
        node_ids: Nodes the diff reported, of either kind.

    Returns:
        The operation node ids to report against, operations before values, with
        anything nothing can be held responsible for dropped.

    """
    ordered = sorted(
        node_ids, key=lambda node: graph.nodes[node].get("type") != _OP_NODE
    )
    return [
        operation
        for operation in (responsible_op(graph, node) for node in ordered)
        if operation is not None
    ]


def _changed_ops(
    graph: nx.DiGraph,
    node_ids: Iterable[int],
    change: OpDiffType,
    side: str,
    graph_label: str,
    sites: Mapping[int, _OpSite],
    claimed: set[int],
    matched: Mapping[int, int] | None = None,
) -> list[OpChange]:
    """
    Describe the operations *node_ids* implicates, at most once each.

    Args:
        graph: The graph the nodes belong to.
        node_ids: Nodes the diff reported.
        change: What to report them as.
        side: :data:`SOURCE` or :data:`TARGET`.
        graph_label: Which graph, for the change's `graph_label`.
        sites: That side's operations by op id, from :func:`_sites`.
        claimed: Node ids already reported on this graph, added to as they are. Carried
            across calls so an operation reported under one kind is not reported again
            under another: a value node resolves to its producer, so a modified
            operation whose result no longer corresponds would otherwise be listed as
            modified *and* added -- two contradictory statements about one operation.
            The caller passes the kinds in order of precedence.
        matched: That side's correspondence. An operation that has a counterpart is
            neither added nor removed whatever its *values* did: inserting an operation
            takes over the value its predecessor produced, so the value nodes pair
            while the operations behind them differ, and the unmatched value resolves
            back to a producer that is alive and matched. Measured on a five-block
            model, ``+ 5`` on one line reported four untouched relus added.

    Returns:
        One change per implicated operation, in the order the nodes resolved.

    """
    changes: list[OpChange] = []
    for node_id in _responsible_ops(graph, node_ids):
        if node_id in claimed:
            continue
        if matched is not None and node_id in matched:
            continue

        operation = graph.nodes[node_id].get("ir_object")
        if operation is None:
            continue

        # Structure rather than computation: a graph or its terminator is not something
        # a user changed, and it is the only kind of operation with no module at all.
        if operation.name in STRUCTURAL_OPS:
            continue

        claimed.add(node_id)

        op_id = get_operation_id(operation)
        site = sites.get(op_id) if op_id is not None else None
        changes.append(
            OpChange(
                change=change,
                op_name=operation.name,
                side=side,
                op_id=op_id,
                attribution=site.attribution if site else None,
                graph_label=graph_label,
            )
        )

    return changes


def _operand_slots(graph: nx.DiGraph, node: int) -> dict[tuple[str, int], list[int]]:
    """
    A node's operands, grouped by the ``(edge_type, index)`` slot they occupy.

    The same slot `graph_match` verifies a pair on, so a slot means the same thing here
    as it did when the pair was rejected.

    Args:
        graph: The graph the node belongs to.
        node: The node whose operands to read.

    Returns:
        Slot to the producers occupying it.

    """
    occupants: dict[tuple[str, int], list[int]] = {}
    for producer, _, data in graph.in_edges(node, data=True):
        slot = (str(data.get("edge_type", "")), int(data.get("index", 0)))
        occupants.setdefault(slot, []).append(producer)
    return occupants


def _is_independent_change(
    diff: GraphDiff,
    source_node: int,
    target_node: int,
    source_labels: Mapping[int, Any],
    target_labels: Mapping[int, Any],
) -> bool:
    """
    Whether a pair's difference is a change in its own right, or a consequence.

    An edit script should not contain an operation that another operation already
    accounts for -- the minimality property of Chawathe et al. (1996), and what keeps a
    diff of ten thousand operations readable. Here the redundancy is structural: an
    operation's operands *are* other operations, so inserting or deleting one changes
    the slots of every neighbour, and reporting those neighbours restates an edit the
    script already contains.

    So each disagreeing slot is asked what accounts for it. Four transitions occur, and
    only the last two are edits in their own right:

    ==================== ======================================= ========
    transition           what happened                           verdict
    ==================== ======================================= ========
    deleted -> survivor  a deletion vacated the slot             implied
    survivor -> new      an insertion took the slot over         implied
    deleted -> new       one operand substituted for another     report
    survivor -> survivor both endpoints outlived the edit        report
    ==================== ======================================= ========

    The last two are the interesting ones. A substitution -- ``y * 2.0`` becoming
    ``y * 3.0`` -- reads as a delete and an insert only because the matcher paired
    neither constant, and nothing else in the diff says the operand changed. A rewiring
    is likewise mentioned nowhere else. Exactly one of "vacated" and "filled" must hold
    for the slot to be someone else's edit, which is an exclusive or, and the reason
    every one-sided condition tried here failed on one case or another.

    A pair whose *labels* differ is always independent: the node itself changed,
    whatever its neighbours did.

    What is deliberately *not* filtered: a slot whose old and new occupants are
    indistinguishable from each other. Under `WeightPolicy.IGNORE` a same-shaped weight
    and everything derived from it share one label, so two saved assets of one quantised
    model reported 56 matmuls modified purely in which of two identical transposes fed
    them. Suppressing those also suppresses a genuine rewiring, which has the same shape
    -- swapping a `norm` and a `relu` moves an operand between two equally-labelled
    nodes. That noise is a reason to compare weights, not a reason to drop rows.

    Args:
        diff: The graph diff the pair came from.
        source_node: The node on the "before" side.
        target_node: Its counterpart.
        source_labels: Identity labels for the source graph.
        target_labels: Identity labels for the target graph.

    Returns:
        Whether to report this pair as modified.

    """
    if source_labels.get(source_node) != target_labels.get(target_node):
        return True

    forward = diff.source_to_target_mapping
    backward = diff.target_to_source_mapping

    source_slots = _operand_slots(diff.source_graph, source_node)
    target_slots = _operand_slots(diff.target_graph, target_node)

    for slot in set(source_slots) | set(target_slots):
        before = source_slots.get(slot, [])
        after = target_slots.get(slot, [])
        # `_UNMAPPED` rather than None for a producer with no counterpart: node ids are
        # non-negative, so it can never equal a real one and the slot correctly reads as
        # changed -- and mixing None into the sort would raise rather than compare.
        if sorted(forward.get(producer, _UNMAPPED) for producer in before) == sorted(
            after
        ):
            continue

        departed = [
            producer for producer in before if forward.get(producer) not in after
        ]
        arrived = [
            producer for producer in after if backward.get(producer) not in before
        ]

        vacated = departed and all(producer not in forward for producer in departed)
        filled = arrived and all(producer not in backward for producer in arrived)
        if bool(vacated) != bool(filled):
            continue

        return True

    return False


def _sites_by_module(sites: Mapping[int, _OpSite]) -> dict[ModulePath, list[_OpSite]]:
    """
    A program's operations grouped by the module instance path they came from.

    Args:
        sites: That program's operations, from :func:`_sites`.

    Returns:
        Module path to its operations. Operations with no path are left out: an
        unattributed operation cannot make a module unique to one side.

    """
    grouped: dict[ModulePath, list[_OpSite]] = {}
    for site in sites.values():
        if site.attribution.module:
            grouped.setdefault(site.attribution.module, []).append(site)
    return grouped


def _relocate_to_new_modules(
    changes: list[OpChange],
    source_sites: Mapping[int, _OpSite],
    target_sites: Mapping[int, _OpSite],
) -> list[OpChange]:
    """
    Report a change against an equivalent operation in a module that is new, if there
    is one.

    Structurally equivalent operations are interchangeable, so when one side has more of
    them than the other, *which* instance the matcher leaves unmapped is arbitrary.
    Adding a third block to a model reports its extra constant against `Block$1` as
    readily as `Block$3`, and a reader seeing `Block$1` reasonably concludes something
    changed inside a block that did not change at all.

    Module scope breaks the tie, and it is information the matcher never looked at.
    Where a change sits in a module both programs have, but an equivalent operation of
    the same kind sits in a module only this side has, the change is reported against
    that one instead. The claim is equally true either way -- the operations are
    interchangeable -- so this only picks the more informative representative.

    Operations are never invented or dropped: each relocation consumes one candidate, so
    the counts per module still add up to the diff the matcher produced.

    **Additions and removals only.** The argument turns on interchangeability, and a
    *modification* is not interchangeable with anything: it names a pair, this operation
    corresponding to that one and differing from it. Re-seating one end onto an
    equivalent operation in another instance makes a claim that is simply false.
    Measured, before this was restricted: inserting a third block moved two
    modifications out of `Block$1/MLP$1`, where they belonged, into `Block$3/Norm$3` --
    a different module, a different layer, and a different operation.

    Args:
        changes: The changes as the matcher attributed them.
        source_sites: The source program's operations, from :func:`_sites`.
        target_sites: The target program's, likewise.

    Returns:
        The same changes, some re-seated onto a module unique to their own side.

    """
    by_module = {
        SOURCE: _sites_by_module(source_sites),
        TARGET: _sites_by_module(target_sites),
    }
    new_modules = {
        SOURCE: by_module[SOURCE].keys() - by_module[TARGET].keys(),
        TARGET: by_module[TARGET].keys() - by_module[SOURCE].keys(),
    }
    if not (new_modules[SOURCE] or new_modules[TARGET]):
        return changes

    # Candidate sites in the new modules, keyed by which operation they are.
    pools: dict[str, dict[str, list[_OpSite]]] = {SOURCE: {}, TARGET: {}}
    for side in (SOURCE, TARGET):
        for module in new_modules[side]:
            for site in by_module[side][module]:
                pools[side].setdefault(site.op_name, []).append(site)

    # An operation the diff already reports must not also be handed out as a candidate.
    taken = {change.op_id for change in changes if change.op_id is not None}

    relocated: list[OpChange] = []
    for change in changes:
        if change.change is OpDiffType.MODIFIED:
            relocated.append(change)
            continue

        candidates = [
            site
            for site in pools[change.side].get(change.op_name, [])
            if site.op_id not in taken
        ]
        # Prefer a candidate written at the same place in the source. The module
        # instance is what was ambiguous; the line the matcher attributed is not, and a
        # `Linear`'s weight and a block's own literal are both constants in a new block
        # yet come from different lines.
        wanted = change.attribution.source if change.attribution else None
        candidate = next(
            (site for site in candidates if site.attribution.source == wanted),
            # Failing that, the deepest module -- the most specific thing that can
            # still be said.
            max(
                candidates,
                key=lambda site: len(site.attribution.module),
                default=None,
            ),
        )

        # Already in a module unique to this side, or nowhere better to put it.
        if change.module in new_modules[change.side] or candidate is None:
            relocated.append(change)
            continue

        taken.add(candidate.op_id)
        pools[change.side][change.op_name].remove(candidate)
        relocated.append(
            replace(
                change,
                op_id=candidate.op_id,
                attribution=candidate.attribution,
            )
        )

    return relocated


@dataclass
class ProgramChanges:
    """What differs between two programs, and where."""

    changes: list[OpChange] = field(default_factory=list)
    """Every reported change, in no particular order. Group with :meth:`by_module`."""

    @property
    def counts(self: Self) -> dict[str, int]:
        """
        How many changes of each kind, and in total.

        Returns:
            ``changed``, ``added``, ``removed`` and ``modified``.

        """
        kinds = Counter(change.change for change in self.changes)
        return {
            "changed": len(self.changes),
            "added": kinds[OpDiffType.ADDED],
            "removed": kinds[OpDiffType.REMOVED],
            "modified": kinds[OpDiffType.MODIFIED],
        }

    def by_module(self: Self) -> list[ModuleNode[OpChange]]:
        """
        The changes as the module tree they happened in.

        Returns:
            Root modules, each subtree ordered by how much changed beneath it, so the
            branch that changed most is read first.

        """
        return build_module_tree((change.module, change) for change in self.changes)

    def changes_in(self: Self, module: str) -> list[OpChange]:
        """
        The changes inside one module.

        Matches a module and everything under it, so ``"Block$2"`` includes
        ``"Block$2/Linear$1"``: a caller asking about a layer means the layer, not the
        one row that happens to be attributed to its top level.

        Args:
            module: Module instance path, e.g. ``"Block$2"`` or ``"Block$2/Linear$1"``.

        Returns:
            The changes under it, ordered by source line then operation name, so a
            reader walks the layer in the order they wrote it.

        """
        return sorted(
            (
                change
                for change in self.changes
                if change.module_label == module
                or change.module_label.startswith(f"{module}/")
            ),
            key=lambda change: (
                change.attribution.source.line
                if change.attribution and change.attribution.source
                else 0,
                change.op_name,
            ),
        )

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Return the changes as plain values, summary first.

        Summary first because a caller's context is the scarce resource: a
        2,500-operation model must not arrive as 2,500 rows. The totals and the
        per-module counts say where to look, and :meth:`changes_in` returns one
        module's operations once a caller knows which to ask for.

        Returns:
            The counts, then the module tree with each change nested under the module
            it happened in.

        """
        return {
            **self.counts,
            "modules": [
                node.to_dict(lambda change: change.to_dict())
                for node in self.by_module()
            ],
        }

    def write_summary(
        self: Self,
        output: TextIO | None = None,
        *,
        max_modules: int | None = 20,
    ) -> None:
        """
        Write the per-module counts as a table, hotspot first.

        Flat paths rather than the nesting :meth:`by_module` gives, because a table row
        has to carry its own identity: a reader scanning for the biggest number needs
        the whole path on that line, not indentation relative to a row above it.

        Args:
            output: Destination stream. Defaults to ``sys.stdout``.
            max_modules: How many rows to write, or None for all. The tail of a long
                diff is modules with one change each, which a caption says better than
                four hundred rows do.

        """
        counts = self.counts
        per_module: dict[str, Counter[OpDiffType]] = {}
        for change in self.changes:
            per_module.setdefault(change.module_label, Counter())[change.change] += 1

        ordered = sorted(
            per_module.items(), key=lambda entry: (-sum(entry[1].values()), entry[0])
        )
        shown = ordered if max_modules is None else ordered[:max_modules]

        rows = [
            _Row(
                cells=(
                    module,
                    str(kinds[OpDiffType.ADDED]),
                    str(kinds[OpDiffType.REMOVED]),
                    str(kinds[OpDiffType.MODIFIED]),
                    str(sum(kinds.values())),
                    f"{100.0 * sum(kinds.values()) / counts['changed']:.0f}%"
                    if counts["changed"]
                    else "0%",
                )
            )
            for module, kinds in shown
        ]

        _write_table(
            _TableSpec(
                title=(
                    f"{counts['changed']} change(s): "
                    f"{counts['added']} added, "
                    f"{counts['removed']} removed, "
                    f"{counts['modified']} modified"
                ),
                columns=(
                    _Column(header="Module"),
                    _Column(header="Added", justify="right"),
                    _Column(header="Removed", justify="right"),
                    _Column(header="Modified", justify="right"),
                    _Column(header="Total", justify="right"),
                    _Column(header="Share", justify="right"),
                ),
                rows=rows,
                caption=(
                    None
                    if len(shown) == len(ordered)
                    else f"{len(ordered) - len(shown)} further module(s) not shown"
                ),
            ),
            output,
        )


def compute_changes(
    source_program: AIProgram,
    target_program: AIProgram,
    *,
    weights: WeightPolicy = WeightPolicy.IGNORE,
) -> ProgramChanges:
    """
    The operations that differ between two programs, attributed to source.

    Covers every graph, not just the entry point: composites are matched through their
    invoke call sites, and a composite present on only one side yields no diff at all --
    which is itself worth reporting rather than dropping.

    Which graphs pair up comes from `compute_per_graph_diff`, and which node became
    which from `graph_match.align`. Alignment yields a third kind the unmapped sets
    cannot express: an operation that is neither added nor removed but *modified*, wired
    or configured differently. It is reported against the target program, because that
    is where it still exists.

    Changes are then re-seated onto modules that exist on only one side where that is
    possible; see :func:`_relocate_to_new_modules`.

    Args:
        source_program: The "before" program. Must have been converted with
            `include_stack_trace=True` -- `TorchConverter.Mode.DEBUG`, the default --
            or nothing can be attributed and every change files under
            `modules.UNATTRIBUTED`.
        target_program: The "after" program, likewise.
        weights: Whether a parameter's *values* count towards an operation's identity.
            Under `WeightPolicy.DIGEST` a layer whose weights changed is reported as a
            modified constant, attributed to the module and line that produced it,
            which is the only way to answer *where* the weights differ. Off by default
            because a rebuild re-initialises parameters, so every layer would read as
            changed on every edit.

    Returns:
        The changes, ready to group by module.

    """
    source_sites = _sites(source_program)
    target_sites = _sites(target_program)
    changes: list[OpChange] = []

    for label, diff in compute_per_graph_diff(
        source_program, target_program, weights=weights
    ):
        if diff is None:
            # A composite with no counterpart. Its own operations are not reported
            # here: the graph is absent as a whole, which the label says, and listing
            # every operation in it as removed would bury that under its contents.
            continue

        changes.extend(
            _changes_in_graph(diff, label, weights, source_sites, target_sites)
        )

    return ProgramChanges(
        changes=_relocate_to_new_modules(changes, source_sites, target_sites)
    )


def _changes_in_graph(
    diff: GraphDiff,
    label: str,
    weights: WeightPolicy,
    source_sites: Mapping[int, _OpSite],
    target_sites: Mapping[int, _OpSite],
) -> list[OpChange]:
    """
    The changes in one pair of matched graphs.

    Modifications are resolved first and consume both sides, for two reasons:

    * on the target side a modification outranks an addition. An operation whose result
      no longer corresponds is modified, and listing it as added as well would say two
      contradictory things about one operation.
    * on the source side it cancels the matching removal. The source twin of a modified
      operation is exactly the fact the modification already states; reported as well,
      one changed constant read as a removal *and* a modification, doubling the diff.

    Args:
        diff: The graph pair's diff.
        label: Which graph, for each change's `graph_label`.
        weights: The policy the diff was computed under, so labels are recomputed under
            the same notion of identity that rejected a pair.
        source_sites: The source program's operations, from :func:`_sites`.
        target_sites: The target program's, likewise.

    Returns:
        The changes in this graph.

    """
    target_claims: set[int] = set()
    removed_claims: set[int] = set()
    modified_ops: list[int] = []
    # Source operations from a pair that was not a modification after all. They are not
    # in `unmapped_source_nodes` -- the aligner did find them a counterpart -- so unless
    # they are carried over they vanish, and the diff shows an addition with no matching
    # removal. `_changed_ops` drops the ones that turn out to be matched anyway.
    declined_sources: list[int] = []

    source_labels = node_labels(diff.source_graph, weights)
    target_labels = node_labels(diff.target_graph, weights)

    for source_node, target_node in diff.modified_node_pairs:
        target_op = responsible_op(diff.target_graph, target_node)
        if target_op is None:
            continue

        if not _is_independent_change(
            diff, source_node, target_node, source_labels, target_labels
        ):
            continue

        source_op = responsible_op(diff.source_graph, source_node)
        if source_op is not None and not _is_one_op_changed(
            diff.source_graph, diff.target_graph, source_op, target_op
        ):
            declined_sources.append(source_op)
            continue

        modified_ops.append(target_op)
        if source_op is not None:
            removed_claims.add(source_op)

    changes: list[OpChange] = []
    # An addition or a removal claims an operation exists on one side only, so both
    # passes are given the correspondence and drop anything it accounts for. Only those
    # two: a modified operation is matched by definition.
    for graph, node_ids, change, side, sites, claimed, matched in (
        (
            diff.target_graph,
            modified_ops,
            OpDiffType.MODIFIED,
            TARGET,
            target_sites,
            target_claims,
            None,
        ),
        (
            diff.target_graph,
            diff.unmapped_target_nodes,
            OpDiffType.ADDED,
            TARGET,
            target_sites,
            target_claims,
            diff.target_to_source_mapping,
        ),
        (
            diff.source_graph,
            [*declined_sources, *diff.unmapped_source_nodes],
            OpDiffType.REMOVED,
            SOURCE,
            source_sites,
            removed_claims,
            diff.source_to_target_mapping,
        ),
    ):
        changes.extend(
            _changed_ops(
                graph=graph,
                node_ids=node_ids,
                change=change,
                side=side,
                graph_label=label,
                sites=sites,
                claimed=claimed,
                matched=matched,
            )
        )

    return changes
