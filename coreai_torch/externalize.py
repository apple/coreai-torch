# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Internals for module externalization and composite op support.

Implementation Details
======================

The externalize pipeline runs in ``to_coreai()`` when ``externalize_modules``
is provided. It works in five phases:

Phase 1: Mark & Re-export (``_mark_externalize``)
--------------------------------------------------
1. Walk ``model.named_modules()`` and for each matching instance: resolve the
   module path, sanitize the op name, save the original forward as
   ``_original_forward``, register a ``torch.library.custom_op`` from the
   submodule's ``forward``, register the original forward as the fake
   implementation via ``register_fake``, patch ``submodule.forward`` to call
   the custom op, and stamp ``_externalize_config = ExternalizeSpec(...)``
   on the module.
2. Re-export via ``export_fn(model)`` and ``run_decompositions(decomp_table)``.
   The FX graph now contains opaque ``call_function`` nodes for each custom op
   call site.

Phase 2: Prepare (``_prepare_externalized`` / ``_prepare_module_export``)
-------------------------------------------------------------------------
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

Phase 4: Emit Core AI IR (``_perform_externalization``)
------------------------------------------------------
Process ``_ExportedModule`` objects deepest-first so inner lowerings exist
when parent graphs are built. For each module, build a ``coreai.GraphOp``
(``noinline`` for all, ``private`` + ``composite_decl`` for composite ops)
and register per-node lowerings keyed by FX node name.

Phase 5: Cleanup (``_restore_externalized``)
---------------------------------------------
Remove all markers (``_original_forward``, ``_externalize_name``,
``_externalize_op_name``, ``_externalize_config``) from every patched module.
The user's model is left unmodified.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterator
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
    _dynamic_shapes_from_node,
    _fake_inputs_from_node,
    _find_all_custom_op_nodes,
    _find_program_for,
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
class _ExportedModule:
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
) -> _ExportedModule:
    """Step 5: Pack a ``_PreparedModule`` and its exported program into an ``_ExportedModule``.

    Also registers the program in the session so nested children can find
    their parent's custom op nodes.
    """
    if prep._program_registry is not None:
        prep._program_registry._programs[prep.name] = exported_program
        # Also register under the original module path so nested children
        # can find their parent via _find_program_for / _ancestor_paths.
        if prep.module_path != prep.name:
            prep._program_registry._programs[prep.module_path] = exported_program
    return _ExportedModule(
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
) -> None:
    """Step 1b: Replace ``submodule.forward`` with a ``torch.library.custom_op``.

    Called by ``_mark_externalize`` for each matching submodule. Saves the
    original forward as ``_original_forward`` and attaches ``_externalize_name``
    and ``_externalize_op_name`` markers for later steps.
    """
    name = _resolve_name(model, submodule)
    op_name = _sanitize_op_name(name)
    qualified_name = f"{_EXTERNALIZE_NAMESPACE}::{op_name}"

    original_forward = submodule.forward
    submodule._original_forward = original_forward  # type: ignore[attr-defined]
    submodule._externalize_name = name  # type: ignore[attr-defined]
    submodule._externalize_op_name = op_name  # type: ignore[attr-defined]

    custom_op_fn = torch.library.custom_op(
        qualified_name, original_forward, mutates_args=()
    )
    torch.library.register_fake(qualified_name, original_forward)

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

    for name, mod in module.named_modules():
        if not name:
            continue
        for i, config in enumerate(configs):
            if isinstance(mod, config.target_class):
                _prepare_module(module, mod)
                mod._externalize_config = config  # type: ignore[attr-defined]
                matched[i] = True
                break

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
        self._model = model
        self._programs: dict[str, ExportedProgram] = {"": exported_program}
        self._wrapped: list[tuple[str, torch.nn.Module]] = sorted(
            (
                (name, mod)
                for name, mod in model.named_modules()
                if hasattr(mod, "_externalize_name")
            ),
            key=lambda x: x[0].count("."),
        )

    def __len__(self) -> int:
        return len(self._wrapped)

    def __iter__(self) -> Iterator[_PreparedModule]:
        for _name, mod in self._wrapped:
            name: str = mod._externalize_name  # type: ignore[attr-defined]
            op_name: str = mod._externalize_op_name  # type: ignore[attr-defined]
            parent_ep = _find_program_for(name, op_name, self._programs)
            if parent_ep is None:
                # Marked submodule is not invoked in the exported graph.
                # Skip it; its forward will be restored by _restore_externalized.
                warnings.warn(
                    f"\n[WARN] coreai_torch.externalize: skipping unused submodule '{name}'.\n"
                    f"       It matched an externalize_modules target class but is not "
                    f"reachable from the exported graph.\n"
                    f"       Action: remove it from the model passed to add_pytorch_module, "
                    f"or ignore if intentional.\n",
                    stacklevel=2,
                )
                continue
            preps = _prepare_module_export(mod, parent_ep)
            for prep in preps:
                prep._program_registry = self
                yield prep


def _prepare_externalized(
    model: torch.nn.Module,
    exported_program: ExportedProgram,
) -> Iterator[_PreparedModule]:
    """Step 3 entry point: yield a ``_PreparedModule`` for every call site.

    Wraps ``_PreparedModules`` which handles shallowest-first ordering and
    nested program lookup.
    """
    return _PreparedModules(model, exported_program)


def _restore_externalized(module: torch.nn.Module) -> None:
    """Step 6: Undo all patches from step 1.

    Restores original forward methods and removes all externalization markers.
    Called after the pipeline completes (or on error via ``finally``).
    """
    for _name, mod in module.named_modules():
        original = getattr(mod, "_original_forward", None)
        if original is not None:
            mod.forward = original
            del mod._original_forward
            del mod._externalize_name
            del mod._externalize_op_name
            if hasattr(mod, "_externalize_config"):
                del mod._externalize_config
