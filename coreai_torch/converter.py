# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, cast

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir  # type: ignore[attr-defined]
import ml_dtypes
import torch
import torch.fx as fx
from coreai._compiler.dialects import coreai
from coreai._compiler.ir import (
    ArrayAttr,
    Attribute,
    DictAttr,
    InsertionPoint,
    Location,
    Module,
    OpResultList,
    StringAttr,
    Type,
    Value,
)
from coreai.authoring import AIProgram
from coreai.authoring import Context as _CoreAIAuthoringContext
from torch import Tensor
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind
from typing_extensions import Self

from .__version__ import __version__
from ._aten_to_core import _aten_to_core_resolver, _higher_order_resolver
from ._composite_declaration import generate_composite_decl
from ._compression.utils import inject_subbyte_tensors
from ._custom_to_core import _custom_to_core_resolver
from ._debug_locations import _DebugInfoRecorder
from ._torch_metal_kernel import TorchMetalKernel
from ._utils import (
    _NARROW_TORCH_DTYPE,
    _get_mutation_output_name,
    _get_verify_debuginfo_locations_enabled,
    _ProgressBar,
    _resolve_io_names,
    check_result_type,
    get_invoke_from_graph,
    get_namespace,
    get_operand,
    get_result_types,
    get_target,
    get_tensor_type,
    preprocess_graph,
    scalar_constant,
    strip_variant_from_target,
    validate_and_cast_numpy_array,
)
from ._validate import validate_exported_program
from .externalize import (
    ExternalizeSpec,
    _ExternalizedExportedProgram,
    _find_marked_submodules,
    _patch_model_for_externalization,
    _restore_externalized,
    _subexport_and_restore,
)


@dataclass
class _StagedEntry:
    """Per-program state staged for conversion."""

    exported_program: ExportedProgram
    input_names: Sequence[str]
    output_names: Sequence[str]
    state_names: Sequence[str]
    entrypoint_name: str  # IR graph name only (coreai.graph @name)
    # nn.Module-path fields (None for ExportedProgram path):
    module: torch.nn.Module | None = None
    export_fn: Callable[..., Any] | None = None
    externalize_modules: list[type | ExternalizeSpec] | None = None
    _externalized_exported_programs: list[_ExternalizedExportedProgram] | None = None


class Context(_CoreAIAuthoringContext):
    def __init__(self) -> None:
        super().__init__()
        self._location: Location = Location.unknown(self._mlir_context)

    def __enter__(self) -> Self:
        super().__enter__()
        self._location.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._location.__exit__(*args)
        super().__exit__(*args)


class TorchConverter:
    class Mode(Enum):
        """Controls the level of debug information embedded in the converted asset.

        Attributes:
            RELEASE: Lightweight mode that records only operation IDs without
                stack traces.
            DEBUG: Includes full torch stack traces for comprehensive source
                mapping and debugging.
        """

        DEBUG = "debug"
        RELEASE = "release"

    @staticmethod
    def _create_debug_info_recorder(
        mode: "TorchConverter.Mode",
    ) -> _DebugInfoRecorder:
        """Create and configure a DebugInfoRecorder based on the converter mode.

        Args:
            mode: The converter mode that determines whether stack traces are
                included.

        Returns:
            A configured _DebugInfoRecorder instance.
        """
        include_stack_trace = mode == TorchConverter.Mode.DEBUG
        debug_config = _DebugInfoRecorder.Config(
            include_stack_trace=include_stack_trace,
            verify_debuginfo_locations=_get_verify_debuginfo_locations_enabled(),
        )
        return _DebugInfoRecorder(config=debug_config)

    def __init__(self, *, mode: "TorchConverter.Mode" = Mode.DEBUG) -> None:
        """Create a reusable converter engine.

        Args:
            mode: Controls the level of debug information embedded in the
                converted asset. Use ``TorchConverter.Mode.RELEASE``
                for lightweight operation-ID-only tracking, or ``TorchConverter.Mode.DEBUG`` (default)
                for full torch stack traces. Call
                :func:`coreai_torch.debugging.debug_info.strip_debug_info`
                to remove debug metadata from an already-converted program.

        Reusable state (custom op lowerings) is retained across calls to
        ``to_coreai()``.  Per-conversion transient state is reset each time.
        """
        self._mode = mode
        self.context = Context()

        # user defined torch op lowering (reusable across conversions)
        self._user_defined_torch_lowering: dict[str, Callable[..., Any]] = {}

        # staged programs awaiting conversion
        self._staged: list[_StagedEntry] = []

        # active progress bar during to_coreai() (None outside conversion)
        self._progress_bar: _ProgressBar | None = None

        # Debug info recorder for comprehensive debug tracking
        self._debug_info_recorder = self._create_debug_info_recorder(mode)

    def _init_conversion_state(self) -> None:
        """Reset per-conversion transient state."""
        # constants in a pytorch model
        self._constants_map: dict[str, Tensor] = {}
        self._parameters_map: dict[str, Tensor] = {}
        self._buffers_map: dict[str, torch.Tensor] = {}
        # io maps
        self._inputs_map: OrderedDict[str, Tensor] = OrderedDict()
        self._outputs_map: OrderedDict[str, Tensor] = OrderedDict()

        self._values_map: dict[str, Value] = {}

        # per-node lowerings for externalized submodules, keyed by FX node name
        self._externalized_lowerings: dict[str, Callable[..., Any]] = {}

        # externalized modules
        self._externalized_exported_programs: list[_ExternalizedExportedProgram] = []

        self.user_input_names: Sequence[str] = []
        self.user_output_names: Sequence[str] = []
        self.user_state_names: Sequence[str] = []

        # module-path state
        self._module: torch.nn.Module | None = None
        self._export_fn: Callable[[torch.nn.Module], ExportedProgram] | None = None
        self._externalize_modules: list[type | ExternalizeSpec] | None = None

    def add_exported_program(
        self,
        exported_program: ExportedProgram,
        *,
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        state_names: Sequence[str] | None = None,
        entrypoint_name: str = "main",
        _externalized_exported_programs: list[_ExternalizedExportedProgram]
        | None = None,
    ) -> Self:
        """Stage a pre-exported ``ExportedProgram`` for conversion.

        The caller is responsible for calling ``torch.export.export()`` and
        ``run_decompositions()`` before passing the program.

        Args:
            input_names: Non-stateful ``forward()`` arg names.
            output_names: Return value names (not mutation outputs).
            state_names: One name per state (buffers then mutated inputs).
                Defaults to FX placeholder names.
            _externalized_exported_programs: Result of
                :func:`coreai_torch.externalize._subexport_and_restore`.
                When set, emits composite ``coreai.graph``s for the patched
                call sites in ``exported_program``.

        Returns ``self`` for chaining.
        """
        if any(e.entrypoint_name == entrypoint_name for e in self._staged):
            raise ValueError(
                f"A program with entrypoint_name={entrypoint_name!r} is already staged. "
                f"Each staged program must have a unique entrypoint_name."
            )
        # Promote uint8 compression constants (indices, quantized data, masks)
        # to their correct sub-byte representations before the MLIR import.
        inject_subbyte_tensors(exported_program)
        validate_exported_program(exported_program, self._user_defined_torch_lowering)
        self._staged.append(
            _StagedEntry(
                exported_program=exported_program,
                input_names=input_names or [],
                output_names=output_names or [],
                state_names=state_names or [],
                entrypoint_name=entrypoint_name,
                _externalized_exported_programs=_externalized_exported_programs,
            )
        )
        return self

    def add_pytorch_module(
        self,
        model: torch.nn.Module,
        *,
        export_fn: Callable[[torch.nn.Module], ExportedProgram],
        externalize_modules: list[type | ExternalizeSpec] | None = None,
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        state_names: Sequence[str] | None = None,
        entrypoint_name: str = "main",
    ) -> Self:
        """Stage an nn.Module for conversion.

        ``export_fn`` must return an ``ExportedProgram`` that has already been
        decomposed via ``run_decompositions()``.  Use
        :func:`coreai_torch.get_decomp_table` for the
        recommended decomposition table::

            export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
                coreai_torch.get_decomp_table()
            )

        The model is not mutated (externalization patches are restored after
        ``to_coreai``).

        Args:
            input_names: Non-stateful forward() arg names. See
                :meth:`add_exported_program`.
            output_names: Return value names. See :meth:`add_exported_program`.
            state_names: One name per state. See :meth:`add_exported_program`.

        Returns ``self`` for chaining.
        """
        if any(e.entrypoint_name == entrypoint_name for e in self._staged):
            raise ValueError(
                f"A program with entrypoint_name={entrypoint_name!r} is already staged. "
                f"Each staged program must have a unique entrypoint_name."
            )
        # Validate exportability eagerly — fails fast with a user error
        try:
            ep = export_fn(model)
        except Exception as e:
            raise RuntimeError(
                f"Your model failed to export: {e}\n"
                f"Ensure the model is exportable via torch.export before "
                f"passing it to TorchConverter.add_pytorch_module."
            ) from e

        if not externalize_modules:
            inject_subbyte_tensors(ep)
        validate_exported_program(ep, self._user_defined_torch_lowering)

        self._staged.append(
            _StagedEntry(
                exported_program=ep,
                input_names=input_names or [],
                output_names=output_names or [],
                state_names=state_names or [],
                entrypoint_name=entrypoint_name,
                module=model,
                export_fn=export_fn,
                externalize_modules=externalize_modules,
            )
        )
        return self

    def _run_externalize_pipeline_from_module(self) -> None:
        """Externalize from a live ``nn.Module`` + ``export_fn`` (Phases 1-3).

        Used by :meth:`add_pytorch_module`. Marks matching submodules with
        custom ops, re-exports the whole model, then delegates Phases 2-3
        (sub-export) to :func:`_subexport_and_restore`, which stashes the
        results in ``self._externalized_exported_programs`` for
        :meth:`_perform_externalization` to emit and restores the model
        patch (``try/finally``) internally.
        """
        assert self._module is not None
        assert self._export_fn is not None
        assert self._externalize_modules is not None

        _patch_model_for_externalization(self._module, self._externalize_modules)

        try:
            with self._progress_bar.status("Re-exporting for externalization..."):
                whole_ep: ExportedProgram = self._export_fn(self._module)
            inject_subbyte_tensors(whole_ep)
        except Exception as e:
            # Re-export after marking failed — our bug, not the user's. Restore
            # here since _subexport_and_restore never gets a chance to.
            _restore_externalized(_find_marked_submodules(self._module))
            raise RuntimeError(
                f"Internal error: re-export after externalization failed: {e}\n"
                f"This is a coreai-torch bug. Please report it."
            ) from e

        # The user's export_fn may preserve composite ops like
        # aten.scaled_dot_product_attention so they survive in the
        # *whole-model* graph for externalization detection; decompositions
        # inside each submodule's own export happen in _subexport_and_restore.
        self._externalized_exported_programs = _subexport_and_restore(
            self._module, whole_ep, progress_bar=self._progress_bar
        )
        self.exported_program = whole_ep

    def _perform_externalization(self, context) -> None:
        """Emit externalized submodule graphs and register their lowerings.

        Modules are processed deepest-first so that inner lowerings are
        available when parent graphs are built (nested externalization).

        For each :class:`_ExternalizedExportedProgram`, this method:

        1. Temporarily swaps ``self.exported_program`` to the submodule's export.
        2. Builds a ``coreai.graph noinline`` via :meth:`_get_graph_op`
           (with ``composite_decl`` and ``private`` for composite op modules).
        3. Registers a lowering that maps the custom op to ``coreai.invoke``.
        4. Restores the whole-model program.
        """
        if not self._externalized_exported_programs:
            return

        whole_program: ExportedProgram = self.exported_program

        # Process deepest modules first so their lowerings exist
        # when parent modules are built.
        sorted_exts: list[_ExternalizedExportedProgram] = sorted(
            self._externalized_exported_programs,
            key=lambda ext: ext.name.count("."),
            reverse=True,
        )

        for ext in self._progress_bar.track(
            sorted_exts,
            description="Converting submodules",
            transient=True,
        ):
            self.exported_program = ext.exported_program
            composite_decl_attrs = {
                k: v for k, v in ext.composite_decl_attrs.items() if v is not None
            }

            composite_decl_attr: Attribute | None = None
            if ext.composite_op_name is not None:
                composite_decl_attr = generate_composite_decl(
                    context,
                    ext.composite_op_name,
                    ext.composite_input_names,
                    ext.composite_output_names,
                    composite_decl_attrs,
                )

            graph_op: coreai.GraphOp = self._get_graph_op(
                name=ext.name,
                no_inline=True,
                private=ext.composite_op_name is not None,
                composite_decl=composite_decl_attr,
            )

            # Register per-node lowering: node.name → coreai.invoke @graph
            for node_name in ext.source_nodes:
                self._externalized_lowerings[node_name] = (
                    lambda values_map, node, loc, _gop=graph_op: get_invoke_from_graph(
                        values_map, node, loc, _gop
                    )
                )

        self.exported_program = whole_program

    def _clean(self) -> None:
        """Reset all internal state dictionaries to empty.

        This method clears all mappings including constants, parameters, buffers,
        inputs, outputs, and Core AI values. It is called before processing a new graph
        to ensure a clean state for conversion.
        """
        self._constants_map = {}
        self._parameters_map = {}
        self._buffers_map = {}
        self._inputs_map = OrderedDict()
        self._outputs_map = OrderedDict()
        self._values_map = {}

    def _register_io(self) -> None:
        """
        This function registers the inputs and outputs to the coreai graph. It selects which inputs
        from exported_program.graph_signature is considered for coreai.graph

        We consider placeholder ops in graph that is present in following dict as inputs to coreai.graph
            - graph_signature.user_inputs (inputs provided by user)
            - graph_signature.buffer_to_mutate (buffers which mutate)

        All the inputs are stored in:

        """
        user_inputs = self.exported_program.graph_signature.user_inputs
        buffers_to_mutate = self.exported_program.graph_signature.buffers_to_mutate
        inputs_to_buffer = self.exported_program.graph_signature.inputs_to_buffers
        for node in self.exported_program.graph_module.graph.nodes:
            if node.op == "placeholder" and node.target in user_inputs:
                self._inputs_map[node.target] = node.meta["val"]
            elif (
                node.op == "placeholder"
                and inputs_to_buffer.get(node.target) in buffers_to_mutate.values()
            ):
                self._inputs_map[node.target] = node.meta["val"]

            if node.op == "output":
                for out_producer in node.args[0]:
                    self._outputs_map[out_producer.name] = out_producer.meta["val"]

    def _register_constants(self) -> None:
        """
        This function registers all the constants in pytorch model. It chooses which constants are to be
        considered.

        We iterate through input specs in exported_program.graph_signature and if it satisfies either of
        the following conditions it is constant:
            - spec.kind is constant tensor : read tensor from _constants
            - spec.kind is parameter : read tensor from parameters()
            - spec.kind is buffer and it can mutate : read tensor from buffers()

        Note: Uses parameters() and buffers() instead of state_dict to capture both persistent
        and non-persistent buffers (registered with persistent=False).
        """
        input_specs = self.exported_program.graph_signature.input_specs
        buffers_to_mutate = self.exported_program.graph_signature.buffers_to_mutate
        inputs_to_buffers = self.exported_program.graph_signature.inputs_to_buffers

        # Build lookup maps using graph_signature names with parameters() and buffers()
        # This approach (from Core AI) captures ALL parameters and buffers including non-persistent ones
        parameters_and_buffers: dict[str, Tensor] = {}

        for name, parameter in zip(
            self.exported_program.graph_signature.parameters,
            self.exported_program.parameters(),
            strict=False,
        ):
            parameters_and_buffers[name] = parameter

        for name, buffer in zip(
            self.exported_program.graph_signature.buffers,
            self.exported_program.buffers(),
            strict=False,
        ):
            parameters_and_buffers[name] = buffer

        for spec in input_specs:
            if spec.kind == InputKind.CONSTANT_TENSOR:
                self._constants_map[spec.arg.name] = (
                    cast(Tensor, self.exported_program._constants[spec.target])
                    .detach()
                    .cpu()
                )

            if spec.kind == InputKind.PARAMETER:
                self._parameters_map[spec.arg.name] = (
                    cast(Tensor, parameters_and_buffers[spec.target]).detach().cpu()
                )

            if spec.kind == InputKind.BUFFER and (
                inputs_to_buffers.get(spec.arg.name) not in buffers_to_mutate.values()
            ):
                self._buffers_map[spec.arg.name] = cast(
                    Tensor, parameters_and_buffers[spec.target]
                )

    def _get_input_names_and_types(self) -> tuple[list[str], list[Type]]:
        """Extract input names and their corresponding Core AI types from the exported program.

        This method iterates through placeholder nodes in the graph and collects those
        that are registered as inputs in _inputs_map. For each input, it retrieves the
        name and converts the PyTorch tensor type to a RankedTensorType.

        Returns:
            A tuple containing:
                - A list of input names (strings)
                - A list of corresponding Core AI Types
        """
        input_names: list[str] = []
        input_types: list[Type] = []

        for node in self.exported_program.graph.nodes:
            if node.op == "placeholder":
                target = get_target(node)
                if target in self._inputs_map:
                    input_names.append(target)
                    input_types.append(
                        get_tensor_type(
                            self._inputs_map[target],
                            self._location,
                        )
                    )
        return input_names, input_types

    # Reduced-precision float types that numpy cannot represent natively.
    # These must be handled via raw uint8 bytes + DenseResourceElementsAttr.
    _REDUCED_PRECISION_FLOAT_DTYPES = {
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e8m0fnu,
        getattr(torch, "float4_e2m1fn_x2", None),
    }

    def _constant_from_tensor(self, tensor: Tensor) -> Value:
        """Create a coreai.constant value.

        Handle complications such as sub-byte dtypes, dtype narrowing, etc.
        """
        torch_dtype = getattr(tensor, "future_dtype", tensor.dtype)
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        tensor_type = get_tensor_type(tensor, self._location, dtype=torch_dtype)

        if hasattr(tensor, "elem"):
            # Our tensor subclasses use bit-packed byte representation
            # for sub-byte dtypes such as ui1, ui2, ui4, si4, fp4, ...
            # the bit-packed bytes are stored in .elem member as uint8 tensor
            tensor = tensor.elem
        cpu_tensor = tensor.detach().cpu().contiguous()

        if torch_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e8m0fnu,
        ):
            # numpy does not natively support float8,
            # so need to detour from float16 then downcast to ml_dtypes
            cpu_tensor = cpu_tensor.to(torch.float16)
            np_array = cpu_tensor.numpy()
            np_dtype = {
                torch.float8_e4m3fn: ml_dtypes.float8_e4m3fn,
                torch.float8_e5m2: ml_dtypes.float8_e5m2,
                torch.float8_e8m0fnu: ml_dtypes.float8_e8m0fnu,
            }[torch_dtype]
            np_array = np_array.astype(np_dtype)
        elif torch_dtype == torch.bfloat16:
            # numpy does not natively support bfloat16,
            # so need to detour from float32 then downcast to ml_dtypes
            cpu_tensor = cpu_tensor.to(torch.float32)
            np_array = cpu_tensor.numpy().astype(ml_dtypes.bfloat16)
        elif torch_dtype in _NARROW_TORCH_DTYPE:
            # Core AI supports up to 32-bit dtypes, i.e. no 64-bit dtypes
            # such as int64 / float64, so they need to be downcast to 32-bit
            cpu_tensor = cpu_tensor.to(_NARROW_TORCH_DTYPE[torch_dtype])
            np_array = cpu_tensor.numpy()
        else:
            np_array = cpu_tensor.numpy()

        # We prefer to create DenseElementsAttr whenever possible, because
        # DenseElementsAttr has better support than DenseResourceElementsAttr,
        # e.g. Core AI compiler can only check if a DenseElementsAttr is splat,
        # i.e. cannot check the opaque DenseResourceElementsAttr.
        # As of 2026-04-10 coreai.constant API is not as good at this as
        # create_elements_attr + ConstantOp APIs, e.g. it uses
        # DenseResourceElementsAttr for reduced-precision float dtypes.
        # TODO: Once coreai.constant reaches parity, migrate to it.
        value_attr = _mlir.dialects.coreai.create_elements_attr(
            self._location,
            tensor_type,
            validate_and_cast_numpy_array(np_array),
        )
        return coreai.ConstantOp(value=value_attr).result

    def _handle_placeholder_op(self, node: fx.Node) -> None:
        """Handle placeholder nodes by converting them to Core AI constant values.

        Placeholder nodes represent inputs to the computation graph. This method checks
        if the placeholder corresponds to a constant, parameter, or buffer, and creates
        the appropriate Core AI constant operation. If the placeholder is a graph input
        (already in _values_map), it does nothing.

        Args:
            node: An FX placeholder node to process

        Raises:
            ValueError: If the placeholder cannot be resolved from any available source
        """
        target: str = get_target(node)
        if target in self._constants_map:
            self._values_map[target] = self._constant_from_tensor(
                self._constants_map[target]
            )
        elif target in self._parameters_map:
            self._values_map[target] = self._constant_from_tensor(
                self._parameters_map[target]
            )
        elif target in self._buffers_map:
            self._values_map[target] = self._constant_from_tensor(
                self._buffers_map[target]
            )
        elif target in self._values_map:
            return None
        else:
            raise ValueError(f"Could not resolve placeholder op: {node}")

    def _handle_call_function_op(self, node: fx.Node) -> None:
        """Handle call_function nodes by converting them to Core AI operations.

        This method dispatches function calls to the appropriate converter based on the
        operator's namespace:
        - "aten" or None: Standard PyTorch operators, converted via _aten_to_core_resolver
        - "module_transform": Submodule calls, converted to coreai.invoke operations

        The resulting Core AI Values are stored in _values_map with the node name as key.
        For operations returning multiple values, keys are suffixed with "#i".

        Args:
            node: An FX call_function node to process

        Raises:
            ValueError: If the namespace is unsupported or a module graph is not found
        """
        target: str = get_target(node)
        namespace: str | None = get_namespace(node)
        qualified_target: str = f"{namespace}::{target}"
        variantless_target: str = strip_variant_from_target(target)
        key: tuple[str, str] | str = (str(namespace), variantless_target)

        if node.name in self._externalized_lowerings:
            with self._location:
                results = self._externalized_lowerings[node.name](
                    self._values_map, node, self._location
                )
            if not isinstance(results, (list, tuple, OpResultList)):
                results = [results]

        elif qualified_target in self._user_defined_torch_lowering:
            with self._location:
                results = self._user_defined_torch_lowering[qualified_target](
                    self._values_map, node, self._location
                )
            if not isinstance(results, (list, tuple)):
                results = [results]

        elif namespace is None or namespace == "aten":
            if target not in _aten_to_core_resolver:
                raise ValueError(
                    f"Unsupported ATen op: {target}. "
                    f"Use register_torch_lowering() to provide a custom lowering."
                )
            with self._location:
                results = _aten_to_core_resolver[target](
                    self._values_map, node, self._location
                )
            if not isinstance(results, (list, tuple, OpResultList)):
                results = [results]

        elif namespace in ("coreai", "coreaix"):
            handler = _custom_to_core_resolver[variantless_target]
            results = [handler(self._values_map, node, self._location)]
        elif namespace == "higher_order":
            if target in _higher_order_resolver:
                with self._location:
                    results = _higher_order_resolver[target](
                        self._values_map,
                        node,
                        graph_module=self.exported_program.graph_module,
                    )
            if not isinstance(results, (list, tuple, OpResultList)):
                results = [results]

        else:
            raise ValueError(
                f"unable to handle call function op: target: {target}, namespace: {namespace}"
            )

        for i, op_result in enumerate(results):
            key = node.name if len(results) == 1 else f"{node.name}#{i}"
            self._values_map[key] = op_result
            val = node.meta.get("val")
            if val is not None:
                check_result_type(
                    op_result,
                    val[i] if isinstance(val, (list, tuple)) else val,
                    node,
                    i,
                )
        self._debug_info_recorder._op_results = results

    def _get_operation(self, node: fx.Node) -> None:
        """Convert an FX node to its corresponding Core AI operation.

        This is the main dispatcher method that routes different node types to their
        appropriate handlers. It uses DebugInfoRecorder to track debug information,
        then processes the node based on its operation type:
        - "placeholder": Handled by _handle_placeholder_op
        - "call_function": Handled by _handle_call_function_op
        - "get_attr": Skipped (subgraph references used by higher-order ops like cond)
        - "output": No operation needed (handled separately during graph finalization)

        Args:
            node: An FX node from the exported program's graph

        Raises:
            ValueError: If the node's operation type is not supported
        """
        with self._debug_info_recorder.record_operation(node) as (_, location):
            self._location = location

            if node.op == "placeholder":
                return self._handle_placeholder_op(node)
            elif node.op == "call_function":
                return self._handle_call_function_op(node)
            elif node.op == "get_attr":
                return None
            elif node.op == "output":
                return None
            else:
                raise ValueError(f"Could not create operation for the fx node {node}")

    def _get_graph_op(
        self,
        name: str,
        no_inline: bool = False,
        externalize: bool = False,
        private: bool = False,
        primary_entrypoint: bool = False,
        composite_decl: Any | None = None,
    ) -> coreai.GraphOp:
        """Create a Core AI graph operation from the current exported program.

        This method orchestrates the conversion of a PyTorch FX graph to a Core AI graph
        operation. It performs the following steps:
        1. Cleans internal state
        2. Registers constants, parameters, and buffers
        3. Registers inputs and outputs
        4. Creates a coreai.GraphOp with appropriate attributes
        5. Converts each FX node to Core AI operations
        6. Sets up output specifications and buffer mutation attributes

        Args:
            name: The name for the graph operation
            no_inline: If True, prevents the graph from being inlined during optimization
            externalize: If True, marks the graph for externalization

        Returns:
            A coreai.GraphOp representing the converted PyTorch graph
        """
        self._clean()
        self._register_constants()
        self._register_io()
        graph_module = preprocess_graph(self.exported_program.graph_module)
        self._location = Location.current if Location.current else Location.unknown()

        user_input_names = self.user_input_names if primary_entrypoint else []
        user_output_names = self.user_output_names if primary_entrypoint else []
        user_state_names = self.user_state_names if primary_entrypoint else []

        original_input_names, input_types = self._get_input_names_and_types()
        graph_output_names = list(self._outputs_map.keys())

        graph_input_names, resolved_output_names, fx_to_output = _resolve_io_names(
            self.exported_program.graph_signature,
            original_input_names,
            graph_output_names,
            user_input_names,
            user_output_names,
            user_state_names,
        )

        # creating and populating coreai.graph
        graph_op: coreai.GraphOp = coreai.GraphOp(
            name=name,
            input_types=input_types,
            result_types=[],
            input_names=graph_input_names,
            loc=self._location,
            no_inline=no_inline,
            externalize=externalize,
            private=private,
            composite_decl=composite_decl,
        )

        with self._debug_info_recorder.record_graph(graph_op):
            for i, orig_name in enumerate(original_input_names):
                self._values_map[orig_name] = graph_op.arguments[i]

            description = (
                f"Converting {name}" if primary_entrypoint else "Converting submodule"
            )
            for node in self._progress_bar.track(
                graph_module.graph.nodes,
                description=description,
                transient=not primary_entrypoint,
            ):
                with graph_op.block:
                    self._get_operation(node)

            # Assemble outputs with resolved names
            outputs_name_value: list[tuple[str, Value]] = [
                (resolved_name, self._values_map[fx_name])
                for fx_name, resolved_name in zip(
                    graph_output_names, resolved_output_names
                )
            ]

            graph_op.set_outputs_spec_from_dict(
                OrderedDict(outputs_name_value),
            )

            # Set MutableBuffers.buffer_mutation attrs on stateful inputs
            existing_arg_attrs: list = (
                list(graph_op.arg_attrs) if graph_op.arg_attrs else []
            )
            inputs_to_buffers = self.exported_program.graph_signature.inputs_to_buffers
            buffers_to_mutate = self.exported_program.graph_signature.buffers_to_mutate
            user_inputs_to_mutate = (
                self.exported_program.graph_signature.user_inputs_to_mutate
            )
            arg_attr: list = []
            for i, input_name in enumerate(graph_input_names):
                attr_dict = {}
                for named_attr in existing_arg_attrs[i]:
                    attr_dict[named_attr.name] = named_attr.attr
                orig_name = original_input_names[i]
                mutation_output = _get_mutation_output_name(
                    orig_name,
                    inputs_to_buffers,
                    buffers_to_mutate,
                    user_inputs_to_mutate,
                )
                if mutation_output is not None:
                    resolved_name = fx_to_output.get(mutation_output, mutation_output)
                    attr_dict["MutableBuffers.buffer_mutation"] = StringAttr.get(
                        resolved_name
                    )
                arg_attr.append(DictAttr.get(attr_dict))
            graph_op.arg_attrs = ArrayAttr.get(arg_attr)

        return graph_op

    def to_coreai(self, *, entrypoints: Sequence[str] | None = None) -> AIProgram:
        """Convert staged programs to a Core AI AIProgram.

        This is the main entry point for converting PyTorch models to Core AI format.
        It creates a Core AI module, processes staged entries (from ``add_exported_program``
        and ``add_pytorch_module`` calls), and generates graph operations.

        Staged programs persist after conversion. Call ``clear()`` to remove them.

        Args:
            entrypoints: If provided, only convert programs with these entrypoint names.
                  If None, convert all staged programs.

        Returns:
            An AIProgram containing the converted Core AI model

        Raises:
            RuntimeError: If no programs have been staged via ``add_exported_program()``
                or ``add_pytorch_module()``.
        """
        entries = self._staged
        if entrypoints is not None:
            names = set(entrypoints)
            entries = [e for e in entries if e.entrypoint_name in names]
        if not entries:
            raise RuntimeError(
                "No programs to convert. Call add_exported_program() or "
                "add_pytorch_module() first."
            )

        with self.context, _ProgressBar() as bar:
            self._progress_bar = bar
            bar.print(
                f"[bold cyan]coreai-torch[/] [dim]{__version__}[/]: "
                f"converting {len(entries)} program(s) to Core AI"
            )
            module: Module = Module.create()
            with self._debug_info_recorder.record_module(module):
                with InsertionPoint(module.body):
                    for entry in bar.track(entries, description="Entries"):
                        self._init_conversion_state()
                        self.exported_program = entry.exported_program
                        self.user_input_names = entry.input_names
                        self.user_output_names = entry.output_names
                        self.user_state_names = entry.state_names

                        # Handle externalization for nn.Module path
                        if entry.module is not None and entry.externalize_modules:
                            self._module = entry.module
                            self._export_fn = entry.export_fn
                            self._externalize_modules = entry.externalize_modules
                            self._run_externalize_pipeline_from_module()
                            self._perform_externalization(module.context)

                        # Handle externalization for pre-marked ExportedProgram path
                        elif entry._externalized_exported_programs is not None:
                            self._externalized_exported_programs = (
                                entry._externalized_exported_programs
                            )
                            self._perform_externalization(module.context)

                        self._get_graph_op(
                            entry.entrypoint_name, primary_entrypoint=True
                        )

        return AIProgram._from_mlir_module(module)

    def clear(self, *, entrypoints: Sequence[str] | None = None) -> None:
        """Remove staged programs. If entrypoints given, remove only those; else remove all.
        Custom lowerings are always preserved."""
        if entrypoints is None:
            self._staged.clear()
        else:
            names = set(entrypoints)
            self._staged = [e for e in self._staged if e.entrypoint_name not in names]

    def __repr__(self) -> str:
        lines = ["TorchConverter("]
        lines.append("  programs:")
        if not self._staged:
            lines.append("    (none)")
        for entry in self._staged:
            kind = "nn.Module" if entry.module is not None else "ExportedProgram"
            inputs = list(entry.input_names) if entry.input_names else []
            outputs = list(entry.output_names) if entry.output_names else []
            parts = [f"    {entry.entrypoint_name}: {kind} {inputs} -> {outputs}"]
            if entry.externalize_modules:
                names = [
                    s.target_class.__name__
                    if isinstance(s, ExternalizeSpec)
                    else s.__name__
                    for s in entry.externalize_modules
                ]
                parts.append(f"externalize={names}")
            lines.append(", ".join(parts))
        lowerings = list(self._user_defined_torch_lowering.keys())
        lines.append(f"  custom_lowerings: {lowerings}")
        lines.append(")")
        return "\n".join(lines)

    def register_torch_lowering(
        self: Self,
        qualified_name: str,
        allow_override: Optional[bool] = False,
    ) -> Callable:
        """Register a custom FX node lowering function with this converter.

        Used as a decorator. The decorated function receives ``(values_map, node, loc)``
        and must return a Core AI ``Value`` or list of ``Value``s.

        Args:
            qualified_name: Op name in ``"namespace::op_name"`` form, matching the
                torch op's qualified name (e.g. ``"my_lib::my_op"``).
            allow_override: If ``True``, silently replaces an existing lowering for
                the same op. Defaults to ``False``.

        Raises:
            ValueError: If ``qualified_name`` is not in ``"namespace::op_name"`` form,
                if the namespace is reserved (``aten``, ``higher_order``, ``coreai``, ``coreaix``),
                or if a lowering for the op already exists and ``allow_override`` is ``False``.
        """

        def decorator(func: Callable) -> Callable:
            parts = qualified_name.split("::")
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    f"qualified_name must be 'namespace::op_name', got {qualified_name!r}"
                )
            namespace, target = parts

            if not allow_override:
                _reserved = {
                    "aten": _aten_to_core_resolver,
                    "higher_order": _higher_order_resolver,
                    "coreai": _custom_to_core_resolver,
                    "coreaix": _custom_to_core_resolver,
                }
                resolver = _reserved.get(namespace)
                if (resolver is not None and target in resolver) or (
                    qualified_name in self._user_defined_torch_lowering
                ):
                    raise ValueError(
                        f"{qualified_name!r} is already registered; set allow_override=True to replace it"
                    )

            self._user_defined_torch_lowering[qualified_name] = func
            return func

        return decorator

    def register_custom_kernels(
        self: Self,
        kernels: Sequence[TorchMetalKernel],
    ) -> Self:
        """Register :class:`TorchMetalKernel` lowerings with this converter.

        For each kernel, a ``register_torch_lowering`` handler is registered
        that extracts input Core AI values and result types from the FX node,
        then delegates to :meth:`CustomMetalKernel.construct_kernel_op` to
        emit ``coreai.metal4_kernel`` ops.

        Returns ``self`` for chaining.
        """
        for kernel in kernels:

            @self.register_torch_lowering(
                f"coreai_metal_kernels::{kernel.name}.default"
            )
            def _(
                values_map: dict[str, Value],
                node: fx.Node,
                loc: Location,
                _k: TorchMetalKernel = kernel,
            ) -> Value | list[Value]:
                input_values: list[Value] = []
                scalar_values: dict[str, Any] = {}
                for idx in range(len(node.args)):
                    arg = node.args[idx]
                    scalar_type: type | None = None
                    if idx < len(_k.input_names) and not isinstance(arg, fx.Node):
                        scalar_type = _k._scalar_input_types.get(_k.input_names[idx])
                    if scalar_type is not None:
                        input_values.append(scalar_constant(scalar_type, arg))
                        scalar_values[_k.input_names[idx]] = arg
                    else:
                        input_values.append(get_operand(values_map, node, idx))
                _k._scalar_values_for_call = scalar_values
                try:
                    results = _k._construct_kernel_op(
                        input_values,
                        get_result_types(node),
                    )
                finally:
                    _k._scalar_values_for_call = {}
                return results[0] if len(results) == 1 else results

        return self
