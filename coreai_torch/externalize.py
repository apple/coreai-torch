# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Internals for module externalization and composite op support.

Implementation Details
======================

The externalize pipeline works in four phases:

Phase 1: Mark (``_patch_model_for_externalization`` / ``_mark_externalize``)
--------------------------------------------------------------------
Called immediately when the user calls ``_patch_model_for_externalization(model, targets)``.
Walks ``model.named_modules()`` and for each matching instance: resolves the
module path, sanitizes the op name, saves the original forward as
``_original_forward``, registers a ``torch.library.custom_op`` from the
submodule's ``forward``, registers the original forward as the fake
implementation via ``register_fake``, patches ``submodule.forward`` to call
the custom op, and stamps ``_externalize_config = ExternalizeSpec(...)`` on
the module.  The model is left patched until ``_subexport_and_restore(model, ep)``
is called.

Phase 2: Prepare (``_PreparedModules`` / ``_prepare_module_export``)
----------------------------------------------------------------------
``_PreparedModules.__iter__`` yields ``_PreparedModule`` objects in
shallowest-first order. For each marked module, ``_prepare_module_export``
finds all FX nodes for this custom op, restores the original forward, reads
composite config, and creates one ``_PreparedModule`` per call-site node with
a UUID-suffixed graph name, fake inputs, dynamic shapes, and source nodes.

Phase 3: Export Submodules (``_torch_export_module`` / ``_finalize_module_export``)
-----------------------------------------------------------------------------------
For each ``_PreparedModule``: call ``torch.export.export`` with the prepared
fake inputs and dynamic shapes, then derive composite I/O names automatically
from the ``ExportedProgram``'s graph signature (parameters/buffers use their
target attribute name, user inputs use their forward parameter name). The
exported program is then decomposed via ``run_decompositions()``.

Phases 2 and 3 run inside ``_subexport_and_restore(model, ep)``, a standalone
function that rediscovers marked submodules by walking ``model`` for the
stamps left by ``_patch_model_for_externalization`` (no separate handle object is
kept). It returns the resulting ``list[_ExternalizedExportedProgram]`` and restores the
model in a ``finally`` block. Pass that list to
``TorchConverter.add_exported_program(ep, _externalized_exported_programs=...)``.

Phase 4: Emit Core AI IR (``_perform_externalization``)
------------------------------------------------------
Process ``_ExternalizedExportedProgram`` objects deepest-first so inner lowerings exist
when parent graphs are built. For each module, build a ``coreai.GraphOp``
(``noinline`` for all, ``private`` + ``composite_decl`` for composite ops)
and register per-node lowerings keyed by FX node name.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import torch
import torch.fx as fx
from torch.export import Dim
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind

from ._utils import (
    _EXTERNALIZE_NAMESPACE,
    _ancestor_paths,
    _dynamic_shapes_from_node,
    _fake_inputs_from_node,
    _find_all_custom_op_nodes,
    _find_program_for,
    _ProgressBar,
    _resolve_name,
    _sanitize_op_name,
)


@dataclass
class ExternalizeSpec:
    """User-facing config describing which submodule class to externalize.

    Used by ``_mark_externalize`` (step 1) to identify matching submodules.

    Args:
        target_class: ``nn.Module`` subclass to match (via ``isinstance``).
        composite_op_name: If set, the emitted graph gets a ``composite_decl``
            and is marked ``private``.
        composite_attrs: Module attribute names (e.g. ``["eps"]``) whose values
            are included in ``composite_decl``. Requires ``composite_op_name``.
    """

    target_class: type
    composite_op_name: str | None = None
    composite_attrs: list[str] | None = None

    def __post_init__(self) -> None:
        if self.composite_op_name is None:
            composite_only = {
                "composite_attrs": self.composite_attrs,
            }
            set_fields = [k for k, v in composite_only.items() if v is not None]
            if set_fields:
                raise ValueError(
                    f"ExternalizeSpec: {set_fields} can only be set when "
                    f"composite_op_name is provided."
                )


@dataclass
class _ExternalizedExportedProgram:
    """Final export result for one call site, ready for lowering (step 5).

    Produced by ``_finalize_module_export``. Each instance becomes one
    ``coreai.graph noinline`` and its corresponding ``coreai.invoke`` lowering
    in ``_perform_externalization``.
    """

    name: str  # module path (e.g. "block.norm"), used as the coreai.graph name
    op_name: str  # sanitized op name (e.g. "block_norm"), used for torch custom op
    exported_program: ExportedProgram  # decomposed standalone export of the submodule
    composite_op_name: str | None = None  # e.g. "rms_norm" for composite ops
    composite_decl_attrs: dict[str, Any] = field(default_factory=dict)
    composite_input_names: list[str] = field(default_factory=list)
    composite_output_names: list[str] = field(default_factory=list)
    source_nodes: list[str] = field(
        default_factory=list
    )  # FX node names for per-node dispatch


@dataclass
class _PreparedModule:
    """Intermediate state for one call site before ``torch.export`` (step 3).

    Produced by ``_prepare_module_export``. Holds the fake inputs, dynamic
    shapes, and composite metadata needed to run ``torch.export.export`` on
    the submodule. One ``_PreparedModule`` per FX call-site node.

    ``_program_registry`` is a back-reference to the shared
    ``_PreparedModules`` that owns the exported-program lookup table.
    ``_finalize_module_export`` uses it to register each exported program
    so nested children can find their parent's custom op nodes.
    """

    name: str
    op_name: str
    module_path: (
        str  # original dotted module path (e.g. "block"), for nested program lookup
    )
    module: torch.nn.Module
    fake_inputs: tuple[torch.Tensor, ...]
    dynamic_shapes: tuple[dict[int, Dim] | None, ...]
    composite_op_name: str | None = None
    composite_decl_attrs: dict[str, Any] = field(default_factory=dict)
    composite_input_names: list[str] = field(default_factory=list)
    composite_output_names: list[str] = field(default_factory=list)
    source_nodes: list[fx.Node] = field(
        default_factory=list
    )  # FX nodes this _PreparedModule covers
    _program_registry: _PreparedModules | None = field(default=None, repr=False)


def _derive_composite_io_names(
    ep: ExportedProgram,
) -> tuple[list[str], list[str]]:
    """Derive composite input/output names from an ``ExportedProgram``'s graph signature.

    Parameters and buffers use their ``target`` (attribute name), user inputs use
    their ``arg.name`` (forward parameter name).  Output names follow the
    convention ``"output"`` for a single return or ``"output_0"``, ``"output_1"``,
    ... for tuple returns.
    """
    input_names: list[str] = []
    for spec in ep.graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            input_names.append(spec.arg.name)
        elif spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            input_names.append(spec.target)

    user_outputs = [
        s for s in ep.graph_signature.output_specs if s.kind == OutputKind.USER_OUTPUT
    ]
    output_names = (
        ["output"]
        if len(user_outputs) == 1
        else [f"output_{i}" for i in range(len(user_outputs))]
    )
    return input_names, output_names


def _torch_export_module(
    prep: _PreparedModule,
) -> ExportedProgram:
    """Step 4: Run ``torch.export.export`` on a single submodule call site.

    Uses the ``_PreparedModule``'s fake inputs and dynamic shapes to export.
    After export, composite I/O names are derived from the graph signature.

    Returns the (not yet decomposed) ``ExportedProgram``.
    """
    # Determine if any optional args were skipped (None in the middle of node.args).
    # If so, use kwargs so the fake inputs land on the correct parameters.
    source_node = prep.source_nodes[0]
    non_none_positions = [i for i, a in enumerate(source_node.args) if a is not None]
    needs_kwargs = non_none_positions != list(range(len(prep.fake_inputs)))

    if needs_kwargs:
        param_names = list(inspect.signature(prep.module.forward).parameters.keys())
        export_args = (prep.fake_inputs[0],)
        export_kwargs = {
            param_names[pos]: fake
            for pos, fake in zip(non_none_positions[1:], prep.fake_inputs[1:])
        }
        shapes_tuple = prep.dynamic_shapes
        export_dynamic_shapes: dict | tuple = {
            param_names[pos]: shape
            for pos, shape in zip(non_none_positions, shapes_tuple)
        }
    else:
        export_args = prep.fake_inputs
        export_kwargs = {}
        export_dynamic_shapes = prep.dynamic_shapes

    ep = torch.export.export(
        prep.module,
        args=export_args,
        kwargs=export_kwargs if export_kwargs else None,
        dynamic_shapes=export_dynamic_shapes,
    )

    prep.composite_input_names, prep.composite_output_names = (
        _derive_composite_io_names(ep)
    )

    return ep


def _finalize_module_export(
    prep: _PreparedModule,
    exported_program: ExportedProgram,
) -> _ExternalizedExportedProgram:
    """Step 5: Pack a ``_PreparedModule`` and its exported program into an ``_ExternalizedExportedProgram``.

    Also registers the program in the session so nested children can find
    their parent's custom op nodes.
    """
    if prep._program_registry is not None:
        prep._program_registry._programs[prep.name] = exported_program
        # Also register under the original module path so nested children
        # can find their parent via _find_program_for / _ancestor_paths.
        # When a module instance has multiple call sites, this key is
        # overwritten once per call site (last-writer-wins). That's safe
        # only because every call site's sub-export carries the same set of
        # nested custom-op nodes for this instance's children — if a future
        # change ever produced call-site-specific sub-exports (e.g. differing
        # graph shapes per site), this would silently pick an arbitrary
        # parent EP for nested lookups.
        if prep.module_path != prep.name:
            prep._program_registry._programs[prep.module_path] = exported_program
    return _ExternalizedExportedProgram(
        name=prep.name,
        op_name=prep.op_name,
        exported_program=exported_program,
        composite_op_name=prep.composite_op_name,
        composite_decl_attrs=prep.composite_decl_attrs,
        composite_input_names=prep.composite_input_names,
        composite_output_names=prep.composite_output_names,
        source_nodes=[n.name for n in prep.source_nodes],
    )


def _prepare_module(
    model: torch.nn.Module,
    submodule: torch.nn.Module,
    op_name_suffix: str,
) -> None:
    """Step 1b: Replace ``submodule.forward`` with a ``torch.library.custom_op``.

    Called by ``_mark_externalize`` for each matching submodule. Saves the
    original forward as ``_original_forward`` and attaches ``_externalize_name``
    and ``_externalize_op_name`` markers for later steps.

    ``op_name_suffix`` (shared by every submodule marked in one
    ``_patch_model_for_externalization`` call) is appended to the sanitized module
    path to form the ``torch.library`` op name. PyTorch library registrations
    are **process-global** and keyed only by this name, so without a per-call
    suffix, marking two different models whose submodules resolve to the same
    dotted path (e.g. both have a ``.norm``) would collide even though each
    call individually restores its own patches.
    """
    name = _resolve_name(model, submodule)
    op_name = f"{_sanitize_op_name(name)}_{op_name_suffix}"
    qualified_name = f"{_EXTERNALIZE_NAMESPACE}::{op_name}"

    if hasattr(submodule, "_original_forward"):
        raise RuntimeError(
            f"submodule '{name}' is already marked for externalization "
            f"(missing a restore from a prior _patch_model_for_externalization call). "
            f"Restore it via _subexport_and_restore before re-marking."
        )

    original_forward = submodule.forward
    submodule._original_forward = original_forward  # type: ignore[attr-defined]
    submodule._externalize_name = name  # type: ignore[attr-defined]
    submodule._externalize_op_name = op_name  # type: ignore[attr-defined]

    custom_op_fn = torch.library.custom_op(
        qualified_name, original_forward, mutates_args=()
    )
    torch.library.register_fake(qualified_name, original_forward)
    # torch.library registrations are process-global and have no
    # unregistration API, so this custom_op/register_fake/register_autograd
    # triple stays registered for the rest of the process's lifetime. The
    # per-call op_name_suffix keeps names unique across calls (no collisions),
    # but many convert cycles in one long-lived process will accumulate
    # registrations — an intentional, unavoidable tradeoff, not a bug.

    def _setup_context_impl(ctx, inputs, keyword_only_inputs, output):  # type: ignore[no-untyped-def]
        ctx.save_for_backward(
            *[x.detach().clone() for x in inputs if isinstance(x, torch.Tensor)]
        )
        ctx.keyword_only_inputs = keyword_only_inputs
        ctx.non_tensor_inputs = [
            (i, x) for i, x in enumerate(inputs) if not isinstance(x, torch.Tensor)
        ]
        ctx.input_requires_grad = [
            isinstance(x, torch.Tensor) and x.requires_grad for x in inputs
        ]

    # torch.library.register_autograd fixes the setup_context calling
    # convention per-op based on the op's schema: ops with a keyword-only
    # parameter (i.e. the wrapped forward has one after a bare `*`) are
    # always called as (ctx, inputs, keyword_only_inputs, output); all
    # other ops are always called as (ctx, inputs, output). This is
    # decided once per op, not per call, so we must pick the matching
    # signature up front rather than hardcode either form.
    has_kwonly_args = any(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for p in inspect.signature(original_forward).parameters.values()
    )

    if has_kwonly_args:

        def _setup_context(ctx, inputs, keyword_only_inputs, output):  # type: ignore[no-untyped-def]
            _setup_context_impl(ctx, inputs, keyword_only_inputs, output)
    else:

        def _setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def]
            _setup_context_impl(ctx, inputs, {}, output)

    def _backward(ctx, *grad_outputs):  # type: ignore[no-untyped-def]
        # Re-runs original_forward under enable_grad to reconstruct the inner
        # autograd graph. Cost is ~1.5x a normal backward (forward runs twice).
        # Stateful submodules (BN running stats, dropout, RNG) are observed
        # twice per training step — _patch_model_for_externalization is not a
        # transparent drop-in for training loops with stateful submodules.
        saved = list(ctx.saved_tensors)
        non_tensor_map = dict(ctx.non_tensor_inputs)
        # Reconstruct the positional input list interleaving tensors and
        # non-tensors. Keyword-only inputs are kept separate (see
        # ctx.keyword_only_inputs) and are not differentiated — torch
        # disallows kwarg-only Tensor args for register_autograd, so they
        # are always plain config values.
        inputs: list[Any] = []
        tensor_iter = iter(saved)
        for i, needs_grad in enumerate(ctx.input_requires_grad):
            if i in non_tensor_map:
                inputs.append(non_tensor_map[i])
            else:
                t = next(tensor_iter)
                inputs.append(t.detach().requires_grad_(needs_grad))

        with torch.enable_grad():
            out = original_forward(*inputs, **ctx.keyword_only_inputs)

        torch.autograd.backward(
            [out] if isinstance(out, torch.Tensor) else list(out),
            grad_outputs,
        )
        return tuple(
            x.grad if isinstance(x, torch.Tensor) and x.requires_grad else None
            for x in inputs
        )

    torch.library.register_autograd(
        qualified_name, _backward, setup_context=_setup_context
    )

    def patched_forward(*args, _op=custom_op_fn, **kwargs):  # type: ignore[no-untyped-def]
        return _op(*args, **kwargs)

    submodule.forward = patched_forward  # type: ignore[assignment]


def _prepare_module_export(
    submodule: torch.nn.Module,
    exported_program: ExportedProgram,
) -> list[_PreparedModule]:
    """Step 3: Extract call-site info from the whole-model export for one submodule.

    Finds all FX nodes that call this submodule's custom op, extracts fake
    inputs and dynamic shapes from each, restores the original forward, and
    returns one ``_PreparedModule`` per call site (node).
    """
    name: str = submodule._externalize_name  # type: ignore[attr-defined]
    op_name: str = submodule._externalize_op_name  # type: ignore[attr-defined]

    all_nodes = _find_all_custom_op_nodes(exported_program, op_name)
    if not all_nodes:
        raise ValueError(
            f"Custom op '{_EXTERNALIZE_NAMESPACE}.{op_name}.default' "
            f"not found in the provided exported program"
        )

    # Restore original forward so the caller can export the submodule
    submodule.forward = submodule._original_forward  # type: ignore[union-attr]

    # Collect composite metadata from config
    config: ExternalizeSpec | None = getattr(submodule, "_externalize_config", None)
    if config is not None and config.composite_op_name is not None:
        attrs = {
            attr: getattr(submodule, attr) for attr in (config.composite_attrs or [])
        }
        composite_op_name: str | None = config.composite_op_name
        composite_decl_attrs: dict[str, Any] = attrs
    else:
        composite_op_name = None
        composite_decl_attrs = {}

    # One _PreparedModule per call site (node).  Even when two call sites have the same
    # argument count, each must get its own noinline graph so the runtime does
    # not deduplicate invocations of the same graph symbol.
    preps: list[_PreparedModule] = []
    for node in all_nodes:
        preps.append(
            _PreparedModule(
                name=f"{name}_{uuid4().hex[:8]}",
                op_name=op_name,
                module_path=name,
                module=submodule,
                fake_inputs=_fake_inputs_from_node(node),
                dynamic_shapes=_dynamic_shapes_from_node(node),
                composite_op_name=composite_op_name,
                composite_decl_attrs=composite_decl_attrs,
                source_nodes=[node],
            )
        )

    return preps


def _find_marked_submodules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Walk ``model`` and collect submodules previously patched by ``_patch_model_for_externalization``.

    Marked submodules are identified by the ``_externalize_name`` stamp left
    by ``_prepare_module`` — there is no separate handle object tracking
    them, so ``_subexport_and_restore`` rediscovers them this way each time
    it runs.
    """
    return [
        mod
        for name, mod in model.named_modules()
        if name and hasattr(mod, "_externalize_name")
    ]


def _mark_externalize(
    module: torch.nn.Module,
    targets: list[type | ExternalizeSpec],
) -> None:
    """Step 1: Walk ``module`` and patch every matching submodule's forward.

    For each submodule whose class matches a target, calls
    ``_prepare_module`` (replacing forward with a custom op) and stores
    the ``ExternalizeSpec`` config on the instance. Must be called **before**
    ``torch.export.export``.

    Emits a ``UserWarning`` if any target does not match at least one
    submodule. This is a warning rather than an error because callers
    legitimately pass a superset of specs across model variants (e.g.
    ``[Qwen2Attn, MixtralAttn]`` where only one applies per checkpoint).
    """
    configs: list[ExternalizeSpec] = [
        ExternalizeSpec(target_class=t) if isinstance(t, type) else t for t in targets
    ]
    matched = [False] * len(configs)
    op_name_suffix = uuid4().hex[:8]

    try:
        for name, mod in module.named_modules():
            if not name:
                continue
            for i, config in enumerate(configs):
                if isinstance(mod, config.target_class):
                    _prepare_module(module, mod, op_name_suffix)
                    mod._externalize_config = config  # type: ignore[attr-defined]
                    matched[i] = True
                    break
    except Exception:
        _restore_externalized(_find_marked_submodules(module))
        raise

    unmatched = [
        configs[i].target_class.__name__ for i, ok in enumerate(matched) if not ok
    ]
    if unmatched:
        warnings.warn(
            f"externalize_modules: the following target class(es) did not "
            f"match any submodule in the model: {', '.join(unmatched)}. "
            f"No externalization will happen for these classes. If intentional "
            f"(e.g. passing a superset across model variants), this warning "
            f"is safe to ignore. Otherwise, check for typos or stale class "
            f"references.",
            stacklevel=2,
        )


class _PreparedModules:
    """Iterator for step 3: yields one ``_PreparedModule`` per call site.

    Iterates marked submodules in shallowest-first order. For nested
    externalization, resolves the correct parent ``ExportedProgram`` so each
    submodule's custom op nodes can be found.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        exported_program: ExportedProgram,
    ) -> None:
        self._programs: dict[str, ExportedProgram] = {"": exported_program}
        self._wrapped: list[torch.nn.Module] = sorted(
            _find_marked_submodules(model),
            key=lambda mod: mod._externalize_name.count("."),  # type: ignore[attr-defined]
        )

    def __len__(self) -> int:
        return len(self._wrapped)

    def __iter__(self) -> Iterator[_PreparedModule]:
        for mod in self._wrapped:
            name: str = mod._externalize_name  # type: ignore[attr-defined]
            op_name: str = mod._externalize_op_name  # type: ignore[attr-defined]
            parent_ep = _find_program_for(name, op_name, self._programs)
            if parent_ep is None:
                # Marked submodule is not invoked in the exported graph.
                # Skip it; its forward will be restored by _restore_externalized.
                warnings.warn(
                    f"coreai_torch.externalize: skipping unused submodule '{name}'. "
                    f"It matched an externalize_modules target class but is not "
                    f"reachable from the exported graph. Action: remove it from "
                    f"the model passed to add_pytorch_module, or ignore if "
                    f"intentional.",
                    stacklevel=3,
                )
                continue
            preps = _prepare_module_export(mod, parent_ep)
            for prep in preps:
                prep._program_registry = self
                yield prep


def _drop_missing_call_sites(
    model: torch.nn.Module,
    exported_program: ExportedProgram,
) -> None:
    """Warn on and restore top-level marked submodules absent from ``exported_program``.

    A marked submodule may not appear in ``exported_program`` if the model
    was exported before ``_patch_model_for_externalization`` was called. Rather than
    letting ``_prepare_module_export`` raise an opaque ``ValueError`` later,
    warn and restore it (and any of its nested marked children, whose call
    sites only exist inside the missing parent's own sub-export) in place —
    the next ``_find_marked_submodules(model)`` call naturally excludes them.

    Only top-level marked modules are checked directly; nested ones do not
    appear in the top-level ``exported_program`` — they show up in the
    sub-export of their enclosing module.
    """
    marked = _find_marked_submodules(model)
    marked_names = {mod._externalize_name for mod in marked}  # type: ignore[attr-defined]

    def _has_ancestor_in(name: str, names: set[str]) -> bool:
        return any(ancestor in names for ancestor in _ancestor_paths(name))

    missing_names = {
        mod._externalize_name  # type: ignore[attr-defined]
        for mod in marked
        if not _has_ancestor_in(mod._externalize_name, marked_names)  # type: ignore[attr-defined]
        and not _find_all_custom_op_nodes(exported_program, mod._externalize_op_name)  # type: ignore[attr-defined]
    }
    if not missing_names:
        return

    for name in sorted(missing_names):
        warnings.warn(
            f"No call sites found for externalized submodule '{name}' in "
            "exported_program. It will not be externalized. The model may "
            "have been exported before _patch_model_for_externalization was called.",
            UserWarning,
            stacklevel=3,
        )

    # Also drop nested modules whose top-level parent is absent — their
    # call sites only appear in the parent's sub-export, which won't run.
    to_remove = [
        mod
        for mod in marked
        if mod._externalize_name in missing_names  # type: ignore[attr-defined]
        or _has_ancestor_in(mod._externalize_name, missing_names)  # type: ignore[attr-defined]
    ]
    _restore_externalized(to_remove)


def _subexport_and_restore(
    model: torch.nn.Module,
    exported_program: ExportedProgram,
    *,
    progress_bar: _ProgressBar | None = None,
) -> list[_ExternalizedExportedProgram]:
    """Run sub-export (Phases 2–3) and restore all forward patches.

    Rediscovers the submodules ``_patch_model_for_externalization`` patched by
    walking ``model`` for the stamps ``_prepare_module`` left behind, finds
    every custom-op call site for each in ``exported_program``, runs
    ``torch.export.export`` on each patched submodule standalone, and
    returns the resulting :class:`_ExternalizedExportedProgram` list. Marked submodules
    with no call site in ``exported_program`` are skipped with a
    ``UserWarning`` rather than raising. The original ``forward`` methods
    are always restored in a ``finally`` block regardless of whether the
    export succeeds.

    Calling this a second time on an already-restored ``model`` finds no
    marked submodules and returns ``[]``.

    Pass ``progress_bar`` (e.g. ``TorchConverter``'s own bar, from inside
    a ``to_coreai()`` call) to report per-submodule progress; omit it to
    run silently, which is the right default when calling this outside
    such a context.

    Pass the returned list to
    :meth:`TorchConverter.add_exported_program`'s ``_externalized_exported_programs``
    argument.
    """
    try:
        _drop_missing_call_sites(model, exported_program)
        preps = _PreparedModules(model, exported_program)
        exts: list[_ExternalizedExportedProgram] = []
        stream = (
            progress_bar.stream("Externalizing submodules")
            if progress_bar is not None
            else nullcontext(None)
        )
        with stream as advance:
            for prep in preps:
                # By now exported_program (the whole model) already
                # contained this submodule's custom-op call site, so a
                # failure re-exporting it standalone is a coreai-torch
                # bug, not a caller error.
                try:
                    inner_ep = _torch_export_module(prep)
                except Exception as e:
                    raise RuntimeError(
                        f"Internal error: failed to export submodule "
                        f"'{prep.name}': {e}\n"
                        f"This is a coreai-torch bug. Please report it."
                    ) from e
                inner_ep = inner_ep.run_decompositions()
                exts.append(_finalize_module_export(prep, inner_ep))
                if advance is not None:
                    advance()
        return exts
    finally:
        # _drop_missing_call_sites already restored the modules it dropped
        # (and warned about them); restore everything still marked here.
        _restore_externalized(_find_marked_submodules(model))


def _patch_model_for_externalization(
    model: torch.nn.Module,
    targets: list[type | ExternalizeSpec],
) -> None:
    """Patch matching submodules immediately.

    Each matching submodule's ``forward`` is replaced with a
    ``torch.library.custom_op`` so that any export or quantization pass run
    afterwards captures the call sites as opaque FX nodes that survive every
    downstream transform. Patch details (original forward, op name, spec)
    are stamped directly onto each matched submodule instance — there is no
    separate handle object tracking them, and nothing is returned;
    :func:`_subexport_and_restore` rediscovers marked submodules by walking
    the model itself.

    After exporting (or quantizing) the patched model, call
    :func:`_subexport_and_restore` with the model and the resulting
    ``ExportedProgram`` to get back a ``list[_ExternalizedExportedProgram]``, then pass
    that to :meth:`TorchConverter.add_exported_program`::

        _patch_model_for_externalization(model, [
            ExternalizeSpec(RMSNorm, composite_op_name="rms_norm",
                            composite_attrs=["eps"]),
        ])
        ep = my_export_or_quantize_pipeline(model)
        _externalized_exported_programs = _subexport_and_restore(model, ep)

        coreai_program = (
            TorchConverter()
            .add_exported_program(ep, _externalized_exported_programs=_externalized_exported_programs)
            .to_coreai()
        )
    """
    _mark_externalize(model, targets)


def _restore_externalized(marked: list[torch.nn.Module]) -> None:
    """Step 6: Undo all patches from step 1.

    Restores original forward methods and removes all externalization markers.
    Called after the pipeline completes (or on error via ``finally``).
    """
    for mod in marked:
        original = getattr(mod, "_original_forward", None)
        if original is not None:
            mod.forward = original
            del mod._original_forward
            del mod._externalize_name
            del mod._externalize_op_name
            if hasattr(mod, "_externalize_config"):
                del mod._externalize_config
