# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utilities for debugging PyTorch models using FX graph inspection."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union, cast

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
import numpy as np
import torch
import torch.fx
from coreai._compiler.ir import Operation, WalkResult
from coreai.authoring import AIProgram
from torch.export import ExportedProgram

from coreai_torch._compression._intx import SubbyteTensor

from .debug_info import (
    CompilationMappings,
    OutputMapping,
    SourceInfo,
)

logger = logging.getLogger(__name__)


@dataclass
class DebugTrace:
    """
    Container for intermediate values from model execution.

    Attributes:
        inputs: Dictionary mapping input node names to their tensors
        outputs: Dictionary mapping output node names to their tensors
        intermediates: Dictionary mapping intermediate node names to their tensors
        mappings: Optional compilation mappings (torch→coreai, torch→odix)

    """

    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]
    intermediates: dict[str, torch.Tensor]
    mappings: CompilationMappings | None = None


class _TorchFXNodeValueInterpreter(torch.fx.Interpreter):
    """
    Custom FX Interpreter that invokes a callback for intermediate node values during execution.

    This interpreter extends torch.fx.Interpreter to call a user-provided function
    with each node and its result during graph execution.
    """

    VIEW_OPS = {"aten.view"}

    def __init__(
        self,
        module: torch.nn.Module,
        callback: Callable[[torch.fx.Node, Any], None] | None = None,
        garbage_collect_values: bool = True,
        enable_autocast: bool = False,
    ) -> None:
        super().__init__(module, garbage_collect_values)
        self._callback = callback
        self._enable_autocast = enable_autocast

    @staticmethod
    def _make_tensors_contiguous(arg: Any) -> Any:
        """
        Recursively walk arg (which may be a Node Argument: tuple/list/dict/scalar/Tensor)
        and return a structure where Tensor objects are contiguous.
        """
        if isinstance(arg, torch.Tensor):
            return arg if arg.is_contiguous() else arg.contiguous()
        elif isinstance(arg, tuple):
            return tuple(
                _TorchFXNodeValueInterpreter._make_tensors_contiguous(a) for a in arg
            )
        elif isinstance(arg, list):
            return [
                _TorchFXNodeValueInterpreter._make_tensors_contiguous(a) for a in arg
            ]
        elif isinstance(arg, dict):
            return {
                k: _TorchFXNodeValueInterpreter._make_tensors_contiguous(v)
                for k, v in arg.items()
            }
        return arg

    @staticmethod
    def _unpack_subbyte_tensors(args: tuple[Any, ...]) -> tuple[Any, ...]:
        """
        Unpack SubbyteTensor arguments to avoid infinite recursion during execution.

        SubbyteTensor is a custom tensor subclass that implements __torch_dispatch__
        to intercept PyTorch operations. When SubbyteTensor objects are passed through
        the FX interpreter, their __torch_dispatch__ method gets called, which can
        lead to infinite recursion because:

        1. The FX interpreter calls a PyTorch operation (e.g., aten.view)
        2. SubbyteTensor.__torch_dispatch__ intercepts this call
        3. The dispatch logic may call back into PyTorch operations
        4. This triggers the FX interpreter again, creating a recursive loop

        By unpacking SubbyteTensor objects to regular tensors before execution,
        we bypass the dispatch mechanism entirely and work with the underlying
        tensor data directly, preventing infinite recursion while preserving
        the correct tensor values and shapes.

        Args:
            args: Tuple of arguments that may contain SubbyteTensor objects

        Returns:
            Tuple of arguments with SubbyteTensor objects unpacked to regular tensors
        """
        unpacked_args = []
        for arg in args:
            if isinstance(arg, SubbyteTensor):
                # Unpack SubbyteTensor to avoid __torch_dispatch__ infinite recursion
                unpacked_args.append(
                    arg.unpack_func(arg.elem, arg.tensor_shape, arg.nbits)
                )
            else:
                unpacked_args.append(arg)
        return tuple(unpacked_args)

    def fetch_args_kwargs_from_env(
        self, node: torch.fx.Node
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Fetch arguments and keyword arguments for a node from the environment.

        Overrides the parent method to automatically unpack SubbyteTensor objects
        to prevent infinite recursion during execution. This ensures that all
        SubbyteTensor objects are converted to regular tensors before any operations
        are performed on them.

        Args:
            node: The FX node to fetch arguments for

        Returns:
            Tuple of (args, kwargs) with SubbyteTensor objects unpacked to regular tensors
        """
        args, kwargs = super().fetch_args_kwargs_from_env(node)

        # Unpack SubbyteTensor arguments to avoid __torch_dispatch__ infinite recursion
        args = self._unpack_subbyte_tensors(args)

        # Also unpack any SubbyteTensor objects in kwargs
        unpacked_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, SubbyteTensor):
                unpacked_kwargs[key] = value.unpack_func(
                    value.elem, value.tensor_shape, value.nbits
                )
            else:
                unpacked_kwargs[key] = value

        return args, unpacked_kwargs

    def run_node(
        self,
        node: torch.fx.Node,
    ) -> Any:
        """
        Execute a node and invoke the callback with its result.

        Args:
            node: The FX node to execute

        Returns:
            The result of executing the node

        """
        # Resolve concrete argument values (automatically unpacks SubbyteTensor objects)
        args, kwargs = self.fetch_args_kwargs_from_env(node)

        # Log operation details and input types
        arg_types = [type(arg).__name__ for arg in args]
        kwarg_types = {k: type(v).__name__ for k, v in kwargs.items()}
        logger.debug(
            f"Executing {node.op} '{node.name}' (target: {node.target}) with args: {arg_types}, kwargs: {kwarg_types}"
        )

        # If we handle the node here (call_function), execute and put result in env
        if node.op == "call_function" and any(
            str(node.target).startswith(view_op) for view_op in self.VIEW_OPS
        ):
            # Make any torch.Tensor instances contiguous (returns new objects when needed)
            resolved_args = _TorchFXNodeValueInterpreter._make_tensors_contiguous(args)
            resolved_kwargs = _TorchFXNodeValueInterpreter._make_tensors_contiguous(
                kwargs
            )
            out = getattr(self, node.op)(node.target, resolved_args, resolved_kwargs)
            self.env[node] = out
            result = out
        else:
            # For other ops, delegate to the base implementation which will use self.env
            result = super().run_node(node)

        if self._callback is not None:
            try:
                self._callback(node, result)
            except Exception as e:
                # Re-raise with context about which node failed
                msg = f"Callback failed for node '{node.name}' (op: {node.op})"
                raise RuntimeError(msg) from e
        return result

    def run(
        self,
        *args: Any,
        initial_env: dict[torch.fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        """
        Run the graph with given inputs, invoking the callback for each node.

        Args:
            *args: Variable positional arguments (input tensors)
            initial_env: Optional initial environment mapping nodes to values
            enable_io_processing: Whether to enable I/O processing

        Returns:
            The final output of the graph

        """
        if self._enable_autocast:
            with torch.autocast("cpu", enabled=True):
                return super().run(
                    *args,
                    initial_env=initial_env,
                    enable_io_processing=enable_io_processing,
                )
        else:
            return super().run(
                *args,
                initial_env=initial_env,
                enable_io_processing=enable_io_processing,
            )


def fetch_intermediate_values(
    exported_program: ExportedProgram,
    inputs: Union[tuple[Any, ...], list[Any]],
    callback: Callable[[torch.fx.Node, Any], None],
    enable_autocast: bool = False,
) -> Any:
    """
    Execute an ExportedProgram and invoke a callback for each intermediate value.

    This function runs the exported program with the given inputs and calls the
    provided callback function for each node in the computation graph with the
    node and its computed result.

    Args:
        exported_program: The ExportedProgram to execute and inspect
        inputs: Input tensors to feed into the program. Can be a tuple or list.
        callback: Callable that takes (node: torch.fx.Node, result: Any) and
                 performs any desired operation (e.g., store values, print info,
                 compute statistics, etc.). Called for every node during execution.
        enable_autocast: Whether to enable automatic mixed precision during execution.
                        Default is False. Set to True to handle mixed precision models.
                        Uses CPU for autocast operations.

    Returns:
        The final output of the exported program

    Raises:
        Exception: Any exception raised by the callback will propagate and halt execution
    """
    # Convert inputs to tuple if it's a list
    if isinstance(inputs, list):
        inputs = tuple(inputs)

    # Get the module from the exported program (this has parameters properly bound)
    module = exported_program.module()
    interpreter = _TorchFXNodeValueInterpreter(
        module,
        callback=callback,
        enable_autocast=enable_autocast,
    )
    output = interpreter.run(inputs)

    return output


def _identify_output_nodes(graph: torch.fx.Graph) -> set[str]:
    """
    Identify output nodes from the FX graph.

    Args:
        graph: The FX graph to analyze

    Returns:
        Set of node names that are outputs

    """
    output_nodes: set[str] = set()

    for node in graph.nodes:
        if node.op == "output":
            # The output node's args contain references to the actual output nodes
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    output_nodes.add(arg.name)
                elif isinstance(arg, (list, tuple)):
                    # Handle multiple outputs
                    for item in arg:
                        if isinstance(item, torch.fx.Node):
                            output_nodes.add(item.name)

    return output_nodes


def _create_node_metadata(
    node: torch.fx.Node,
    tensor: torch.Tensor,
    filename: str,
) -> dict[str, Any]:
    """
    Create metadata dictionary for a node's tensor result.

    Args:
        node: The FX node
        result: The tensor result
        filename: Filename where the tensor will be saved

    Returns:
        Dictionary containing node metadata

    """
    # NumPy doesn't support BFloat16, convert to Float32 for numpy conversion
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)

    numpy_array = tensor.numpy()

    metadata = {
        "node_name": node.name,
        "node_op": node.op,
        "node_target": str(node.target),
        "data_file": f"data/{filename}",
        "shape": list(numpy_array.shape),
        "torch_dtype": str(tensor.dtype),
        "numel": int(numpy_array.size),
    }

    return metadata


def _save_tensor_to_file(
    tensor: torch.Tensor,
    filepath: Path,
) -> np.ndarray[Any, Any]:
    """
    Save a tensor to a numpy file.

    Args:
        tensor: The tensor to save
        filepath: Path where to save the file

    Returns:
        The numpy array that was saved

    """
    # NumPy doesn't support BFloat16, convert to Float32
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)

    numpy_array = tensor.numpy()
    np.save(str(filepath), numpy_array)
    return numpy_array


def _extract_source_identifiers_from_coreai_program(
    coreai_program: AIProgram,
) -> dict[str, dict[str, SourceInfo]]:
    """
    Extract source identifiers from operations in the Core AI program.

    This extracts the source dialect and operation identifiers (e.g., "aten.topk")
    for each operation that has source information.

    Args:
        coreai_program: The AIProgram to extract source info from

    Returns:
        Dictionary mapping dialects to operation IDs to SourceInfo objects

    """
    source_mappings: dict[str, dict[str, SourceInfo]] = defaultdict(dict)

    def extract_source_info(operation: Operation) -> WalkResult:
        # Get source info from operation location
        source_infos = _mlir.get_source_info(operation.location)  # type: ignore[attr-defined]

        for source_info in source_infos:
            # Get the source name (dialect)
            source_name = getattr(source_info, "name", None)
            if not source_name:
                continue

            # Get identifiers
            identifiers = getattr(source_info, "identifiers", None)
            if identifiers:
                identifier_list = [str(ident) for ident in identifiers]
            else:
                identifier_list = []

            # Get operation ID for this source level
            op_id = _mlir.get_operation_id(operation.location, source_name)  # type: ignore[attr-defined]
            if op_id is not None:
                op_id_value = getattr(op_id, "value", None)
                if op_id_value is not None:
                    source_mappings[source_name][str(op_id_value)] = SourceInfo(
                        dialect=source_name,
                        id=op_id_value,
                        identifiers=identifier_list,
                    )

        return WalkResult.ADVANCE

    # Walk the Core AI program to extract source info
    coreai_program._module._mlir_module.operation.walk(extract_source_info)

    return dict(source_mappings)


def _extract_output_mappings_from_coreai_program(
    coreai_program: AIProgram,
) -> list[OutputMapping]:
    """
    Get operation output mappings from an AIProgram.

    Extracts mappings showing how values flow through the compilation pipeline
    (e.g., torch → coreai, torch → odix). These mappings are created by the
    InferOutputMappings pass during compilation.

    Args:
        coreai_program: The AIProgram to extract mappings from

    Returns:
        List of OutputMapping objects showing source and target operation outputs

    """
    output_mappings = []

    # Use LocationAPI to get all output mappings from the module
    output_maps = _mlir.get_all_output_maps_from_module(
        coreai_program._module._mlir_module
    )  # type: ignore[attr-defined]

    for output_map in output_maps:
        # Extract mapping fields
        source_level = getattr(output_map, "source_level", None)
        source_op_id = getattr(output_map, "source_op_id", None)
        source_output = getattr(output_map, "source_output", None)
        target_level = getattr(output_map, "target_level", None)
        target_op_id = getattr(output_map, "target_op_id", None)
        target_output = getattr(output_map, "target_output", None)

        if all(
            v is not None
            for v in [
                source_level,
                source_op_id,
                source_output,
                target_level,
                target_op_id,
                target_output,
            ]
        ):
            # Cast to int to satisfy mypy after None check
            output_mappings.append(
                OutputMapping(
                    source_level=cast("str", source_level),
                    source_op_id=cast("int", source_op_id),
                    source_output=cast("int", source_output),
                    target_level=cast("str", target_level),
                    target_op_id=cast("int", target_op_id),
                    target_output=cast("int", target_output),
                ),
            )

    return output_mappings


def _extract_torch_source_info_from_location(location: Any) -> tuple[int, str] | None:
    """
    Extract torch operation ID and single identifier from a location.

    The C++ layer automatically selects the identifier with the highest embedded
    torch_op_id when multiple identifiers exist (from fused operations).
    Identifiers are returned as clean names without the ID suffix.

    Args:
        location: Core AI location to extract from

    Returns:
        Tuple of (operation_id, identifier) or None if torch source not found

    """
    source_infos = _mlir.get_source_info(location)  # type: ignore[attr-defined]

    for source_info in source_infos:
        source_name = getattr(source_info, "name", None)
        if source_name != "torch":
            continue

        # Get operation ID - already parsed from embedded identifiers in C++
        op_id_value = getattr(source_info, "id", None)

        # Get identifiers - C++ returns list but we take the first/only one
        # (C++ layer has already selected the highest ID for fused ops)
        identifiers = getattr(source_info, "identifiers", None)
        if identifiers and len(identifiers) > 0:
            identifier = str(identifiers[0])
        else:
            identifier = None

        # Only return if we have valid values
        if op_id_value is not None and identifier is not None:
            return (op_id_value, identifier)

    return None


def _resolve_compiled_mappings(
    mappings_list: list[OutputMapping],
) -> OutputMapping | None:
    """
    Resolve multiple compiled operation mappings for a single torch identifier.

    When multiple mappings exist, selects the one with the highest target_op_id,
    as this typically represents the most downstream operation in the pipeline.
    The C++ resolveOutputMaps pass should have already removed ambiguous cases.

    Args:
        mappings_list: List of OutputMapping objects for this identifier

    Returns:
        Resolved OutputMapping object with highest target_op_id, or None if empty

    """
    if len(mappings_list) == 0:
        return None

    if len(mappings_list) == 1:
        return mappings_list[0]

    # Select mapping with the highest target operation ID
    return max(mappings_list, key=lambda m: m.target_op_id)


def _deduplicate_torch_identifiers_by_coreai_output(
    mappings: dict[str, OutputMapping],
) -> dict[str, OutputMapping]:
    """
    Deduplicate torch identifiers that map to the same coreai operation output.

    When multiple torch identifiers map to the same coreai output (e.g., from fused
    operations like view+matmul), keeps only the identifier with the highest
    torch_op_id since IDs are assigned topologically.

    Args:
        mappings: Dictionary mapping torch identifiers to OutputMapping objects

    Returns:
        Deduplicated dictionary with only one identifier per coreai output

    """
    # Group by coreai output: (target_op_id, target_output) -> (source_op_id, identifier)
    coreai_to_torch: dict[tuple[int, int], tuple[int, str]] = {}

    for identifier, mapping in mappings.items():
        coreai_key = (mapping.target_op_id, mapping.target_output)

        if coreai_key in coreai_to_torch:
            # Already have a mapping to this coreai output
            existing_torch_op_id, _ = coreai_to_torch[coreai_key]
            # Keep the one with higher torch_op_id (topologically latest)
            if mapping.source_op_id > existing_torch_op_id:
                coreai_to_torch[coreai_key] = (mapping.source_op_id, identifier)
        else:
            coreai_to_torch[coreai_key] = (mapping.source_op_id, identifier)

    # Build result with only the selected identifiers
    result: dict[str, OutputMapping] = {}
    selected_identifiers = {identifier for _, identifier in coreai_to_torch.values()}

    for identifier in selected_identifiers:
        if identifier in mappings:
            result[identifier] = mappings[identifier]

    return result


def get_torch_to_coreai_output_mapping(
    coreai_program: AIProgram,
) -> dict[str, OutputMapping]:
    """
    Get mappings from torch FX node identifiers to coreai operation outputs.

    Creates a mapping from torch operation identifiers (FX node names) to their
    corresponding coreai operation output mappings. This is useful for tracing
    how torch operations are lowered to coreai operations.

    When multiple coreai operations map to a single torch identifier, this function
    applies resolution logic using connected graph analysis and topological sorting
    to select the most appropriate mapping. Identifiers that cannot be resolved
    are excluded from the result.

    When multiple torch identifiers map to the same coreai operation output (from
    fused operations), only the identifier with the highest torch_op_id is kept.

    Args:
        coreai_program: The AIProgram to extract mappings from

    Returns:
        Dictionary mapping torch identifier strings to OutputMapping objects

    """
    result: dict[str, list[OutputMapping]] = defaultdict(list)

    # Get all output mappings from the module
    all_mappings = _extract_output_mappings_from_coreai_program(coreai_program)

    # Filter by coreai target level
    filtered_mappings = [
        mapping for mapping in all_mappings if mapping.target_level == "coreai"
    ]

    # Build mapping from torch operation IDs to identifiers
    torch_op_id_to_identifier: dict[int, str] = {}

    def collect_torch_identifiers(operation: Operation) -> WalkResult:
        """Collect torch operation IDs and their single identifier."""
        result_info = _extract_torch_source_info_from_location(operation.location)
        if result_info is not None:
            torch_op_id, identifier = result_info
            torch_op_id_to_identifier[torch_op_id] = identifier
        return WalkResult.ADVANCE

    module = coreai_program._module._mlir_module
    module.operation.walk(collect_torch_identifiers)

    # Map torch identifiers to output mappings
    for mapping in filtered_mappings:
        if (
            mapping.source_level == "torch"
            and mapping.source_op_id in torch_op_id_to_identifier
        ):
            identifier = torch_op_id_to_identifier[mapping.source_op_id]
            result[identifier].append(mapping)

    # Resolve mappings for identifiers with multiple target operations
    resolved_result: dict[str, OutputMapping] = {}
    for identifier, mappings_list in result.items():
        resolved_mapping = _resolve_compiled_mappings(mappings_list)
        if resolved_mapping is not None:
            resolved_result[identifier] = resolved_mapping

    # Deduplicate: keep only one identifier per coreai output (highest torch_op_id)
    final_result = _deduplicate_torch_identifiers_by_coreai_output(resolved_result)

    return final_result


def get_torch_to_ops_mapping(
    coreai_program: AIProgram,
) -> dict[str, list[Operation]]:
    """
    Create mapping from torch identifiers to Core AI operations.

    Walks all operations in the Core AI program and extracts torch identifiers
    from their source info to build the mapping.

    Args:
        coreai_program: The AIProgram to analyze

    Returns:
        Dictionary mapping torch FX node identifiers to lists of Operation objects

    """
    mapping: defaultdict[str, list[Operation]] = defaultdict(list)

    def collect_ops(operation: Operation) -> WalkResult:
        """Collect operations and their torch identifiers."""
        # Get source info from operation location
        source_infos = _mlir.get_source_info(operation.location)  # type: ignore[attr-defined]

        for source_info in source_infos:
            # Get identifiers from source info
            identifiers = getattr(source_info, "identifiers", None)
            if identifiers:
                for identifier in identifiers:
                    identifier_str = str(identifier)
                    mapping[identifier_str].append(operation)

        return WalkResult.ADVANCE

    coreai_program._module._mlir_module.operation.walk(collect_ops)

    return dict(mapping)


class _IntermediateDumper:
    """Helper class to handle dumping of intermediate values during model execution."""

    def __init__(
        self,
        data_path: Path,
        output_nodes: set[str],
        node_filter: Callable[[torch.fx.Node, Any], bool] | None = None,
    ) -> None:
        self.data_path = data_path
        self.output_nodes = output_nodes
        self.node_filter = node_filter
        self.file_counter = 0
        self.inputs_metadata: dict[str, dict[str, Any]] = {}
        self.outputs_metadata: dict[str, dict[str, Any]] = {}
        self.intermediates_metadata: dict[str, dict[str, Any]] = {}

    def __call__(self, node: torch.fx.Node, result: Any) -> None:
        """Invoke callback for each node during execution."""
        # Apply filter if provided
        if self.node_filter is not None and not self.node_filter(node, result):
            return

        # Only dump tensors
        if not isinstance(result, torch.Tensor):
            return

        try:
            if result.is_quantized:
                result = result.dequantize()

            # Convert to CPU and detach
            result = result.detach().cpu()

            # Generate filename for this node
            filename = f"{node.name}_{self.file_counter:04d}.npy"
            self.file_counter += 1
            filepath = self.data_path / filename

            # Save tensor to file
            _save_tensor_to_file(result, filepath)
        except Exception as e:
            logger.debug(
                "Skipping non-tensor result: type=%s, error=%r",
                type(result).__name__,
                e,
            )
            return

        # Create metadata
        node_metadata = _create_node_metadata(
            node=node,
            tensor=result,
            filename=filename,
        )

        # Categorize by node operation type
        if node.op == "placeholder":
            self.inputs_metadata[node.name] = node_metadata
        elif node.name in self.output_nodes:
            self.outputs_metadata[node.name] = node_metadata
        else:
            self.intermediates_metadata[node.name] = node_metadata


def _default_node_filter(node: torch.fx.Node, result: Any) -> bool:
    """
    Default node filter that excludes tensor transformation operations.

    This filter excludes operations that primarily transform tensor shapes or
    metadata without changing the core tensor values, helping to reduce
    storage and focus on more meaningful intermediate values.

    Args:
        node: The FX node to evaluate
        result: The result of executing the node

    Returns:
        False for tensor transformation ops (don't save), True otherwise (save)
    """
    # Only filter tensor results
    if not isinstance(result, torch.Tensor):
        return True

    # Define tensor transformation operation prefixes to exclude
    TENSOR_TRANSFORM_PREFIXES = {
        # Shape operations
        "aten.view",
        "aten.reshape",
        "aten.transpose",
        "aten.permute",
        "aten.squeeze",
        "aten.unsqueeze",
        "aten.flatten",
        "aten.unflatten",
        # Memory layout operations
        "aten.contiguous",
        "aten.clone",
        "aten.detach",
        "aten.as_strided",
        # Type operations
        "aten.to",
        "aten.type_as",
        # Slice operations (often just change views)
        "aten.slice",
        "aten.select",
        "aten.narrow",
        # Expansion operations
        "aten.expand",
        "aten.repeat",
    }

    # Define built-in operations to exclude (data access, not computation)
    BUILTIN_ACCESS_OPS = {
        # Data access operations
        "__getitem__",
        "getitem",  # Accessing elements from tuples, lists, tensors
    }

    # Check if this is a tensor transformation operation using prefix matching
    if node.op == "call_function":
        target_str = str(node.target)
        if any(target_str.startswith(prefix) for prefix in TENSOR_TRANSFORM_PREFIXES):
            return False

        # Check for built-in data access operations
        if any(builtin_op in target_str for builtin_op in BUILTIN_ACCESS_OPS):
            return False

    return True


def _setup_output_directories(output_dir: Union[str, Path]) -> tuple[Path, Path]:
    """
    Create output directories for intermediate dumps.

    Args:
        output_dir: Base output directory

    Returns:
        Tuple of (output_path, data_path)

    Raises:
        FileExistsError: If the output directory already exists

    """
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        raise FileExistsError(
            f"Output directory already exists: {output_path}. "
            "Please choose a different output directory, remove the existing one, or "
            "use a different model_name parameter."
        ) from e
    data_path = output_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    return output_path, data_path


def _serialize_mappings_to_dict(
    source_identifiers: dict[str, dict[str, SourceInfo]] | None,
    output_mappings_list: list[OutputMapping] | None,
) -> dict[str, Any]:
    """
    Serialize mappings to dictionary format for JSON.

    Args:
        source_identifiers: Dictionary of source identifiers by dialect
        output_mappings_list: List of output mappings

    Returns:
        Dictionary with serialized mappings

    """
    mappings: dict[str, Any] = {}
    if source_identifiers is not None:
        # Convert SourceInfo objects to dicts for JSON serialization
        sources_dict = {}
        for dialect, ops in source_identifiers.items():
            sources_dict[dialect] = {
                op_id: source_info.to_dict() for op_id, source_info in ops.items()
            }
        mappings["sources"] = sources_dict
    if output_mappings_list is not None:
        # Convert OutputMapping objects to dicts for JSON serialization
        mappings["outputs"] = [mapping.to_dict() for mapping in output_mappings_list]
    return mappings


def save_intermediates(  # noqa: PLR0913
    program: ExportedProgram,
    inputs: Union[tuple[Any, ...], list[Any]],
    output_dir: Union[str, Path],
    node_filter: Callable[[torch.fx.Node, Any], bool] = _default_node_filter,
    coreai_program: AIProgram | None = None,
    enable_autocast: bool = False,
    model_name: str = "main",
) -> str:
    """
    Execute a PyTorch ExportedProgram and dump intermediate values to disk.

     This function runs the program and saves intermediate tensor values
     to numpy files in the specified directory, along with metadata in a JSON file.
     The metadata is organized into inputs, outputs, and intermediates.

    Args:
         program: ExportedProgram to execute and inspect.
         inputs: Input tensors to feed into the program. Can be a tuple or list.
         output_dir: Directory path where to save the intermediate values.
                     Will be created if it doesn't exist.
         node_filter: Optional callable that takes (node: torch.fx.Node, result: Any)
                     and returns True if the node's value should be dumped.
                     If None and exclude_tensor_transforms is True, uses default filter
                     that excludes tensor transformation ops. If None and
                     exclude_tensor_transforms is False, saves all tensor nodes.
         coreai_program: Optional AIProgram to extract source info from.
                           If provided, variable information from source locations
                           will be added to the metadata.
         enable_autocast: Whether to enable automatic mixed precision during execution.
                         Default is False. Set to True to handle mixed precision models
                        and avoid dtype mismatch errors. Uses CPU for autocast operations.
        model_name: Name for the model-specific output directory. Default is "main".
                   Creates a directory named "{model_name}.aimodelintermediates" within
                   the specified output_dir for better organization of outputs.

    Returns:
        Path to the generated metadata JSON file

    Examples:
         >>> import torch
         >>> from torch.export import export
         >>>
         >>> # Create and export a simple model
         >>> class MyModel(torch.nn.Module):
         ...     def __init__(self):
         ...         super().__init__()
         ...         self.conv = torch.nn.Conv2d(3, 16, 3)
         ...         self.linear = torch.nn.Linear(16, 10)
         ...
         ...     def forward(self, x):
         ...         x = self.conv(x)
         ...         x = x.flatten(1)
         ...         return self.linear(x)
         >>>
         >>> model = MyModel()
         >>> example_input = (torch.randn(1, 3, 32, 32),)
         >>> exported = export(model, example_input)
         >>>
         >>> # Dump all intermediate tensors
         >>> metadata_path = dump_intermediates(
         ...     exported,
         ...     example_input,
         ...     './debug_output'
         ... )
         >>>
         >>> # Dump only convolution outputs
         >>> metadata_path = dump_intermediates(
         ...     exported,
         ...     example_input,
         ...     './debug_output',
         ...     node_filter=lambda node, result: 'conv' in node.name and isinstance(result, torch.Tensor)
         ... )

    """
    # Use the exported program directly
    exported_program = program

    # Setup output directories with model-specific .aimodelintermediates directory
    model_output_dir = Path(output_dir) / f"{model_name}.aimodelintermediates"
    output_path, data_path = _setup_output_directories(model_output_dir)

    # Extract mappings from Core AI program if provided
    source_identifiers = None
    output_mappings_list = None
    if coreai_program is not None:
        source_identifiers = _extract_source_identifiers_from_coreai_program(
            coreai_program,
        )
        output_mappings_list = _extract_output_mappings_from_coreai_program(
            coreai_program,
        )

    # Identify output nodes from the graph
    graph = exported_program.graph_module.graph
    output_nodes = _identify_output_nodes(graph)

    # Create dumper and run the model
    dumper = _IntermediateDumper(data_path, output_nodes, node_filter)
    fetch_intermediate_values(
        exported_program,
        inputs,
        dumper,
        enable_autocast=enable_autocast,
    )

    # Create structured metadata
    metadata = {
        "inputs": dumper.inputs_metadata,
        "outputs": dumper.outputs_metadata,
        "intermediates": dumper.intermediates_metadata,
    }

    # Add mappings if available
    if source_identifiers is not None or output_mappings_list is not None:
        metadata["mappings"] = _serialize_mappings_to_dict(
            source_identifiers,
            output_mappings_list,
        )

    # Save metadata to JSON file
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(metadata_file)


def _parse_torch_dtype(dtype_str: str) -> torch.dtype | None:
    """
    Convert a torch dtype string to a torch.dtype object.

    Args:
        dtype_str: String representation of a torch dtype (e.g., "torch.float32")

    Returns:
        The corresponding torch.dtype object, or None if not recognized

    """
    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float": torch.float32,
        "torch.float64": torch.float64,
        "torch.double": torch.float64,
        "torch.float16": torch.float16,
        "torch.half": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.long": torch.int64,
        "torch.int32": torch.int32,
        "torch.int": torch.int32,
        "torch.int16": torch.int16,
        "torch.short": torch.int16,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
        "torch.complex64": torch.complex64,
        "torch.cfloat": torch.complex64,
        "torch.complex128": torch.complex128,
        "torch.cdouble": torch.complex128,
    }
    return dtype_map.get(dtype_str)


def _validate_metadata_format(metadata: Any) -> None:
    """
    Validate that metadata has the expected format.

    Args:
        metadata: The metadata to validate

    Raises:
        ValueError: If metadata format is invalid

    """
    if (
        not isinstance(metadata, dict)
        or "inputs" not in metadata
        or "outputs" not in metadata
        or "intermediates" not in metadata
    ):
        msg = (
            "Invalid metadata format. Expected structure with 'inputs', 'outputs', "
            "and 'intermediates' keys."
        )
        raise ValueError(msg)


def _load_single_tensor(
    node_info: dict[str, Any],
    output_dir: Path,
    device: Union[str, torch.device] | None,
) -> torch.Tensor:
    """
    Load a single tensor from metadata and numpy file.

    Args:
        node_info: Metadata dictionary for the node
        output_dir: Directory containing the data files
        device: Optional device to load tensor onto

    Returns:
        The loaded tensor

    Raises:
        FileNotFoundError: If the data file doesn't exist

    """
    # Get the numpy file path
    data_file = node_info["data_file"]
    numpy_path = output_dir / data_file

    # Check if numpy file exists
    if not numpy_path.exists():
        msg = f"Data file not found: {numpy_path}"
        raise FileNotFoundError(msg)

    # Load numpy array
    numpy_array = np.load(str(numpy_path))

    # Convert to torch tensor
    tensor = torch.from_numpy(numpy_array)

    # Convert to the original torch dtype if stored
    if "torch_dtype" in node_info:
        target_dtype = _parse_torch_dtype(node_info["torch_dtype"])
        if target_dtype is not None:
            tensor = tensor.to(target_dtype)

    # Move to specified device if provided
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def _load_tensor_dict(
    metadata_dict: dict[str, dict[str, Any]],
    output_dir: Path,
    device: Union[str, torch.device] | None,
) -> dict[str, torch.Tensor]:
    """
    Load all tensors from a metadata dictionary.

    Args:
        metadata_dict: Dictionary mapping node names to metadata
        output_dir: Directory containing the data files
        device: Optional device to load tensors onto

    Returns:
        Dictionary mapping node names to tensors

    """
    result: dict[str, torch.Tensor] = {}
    for node_name, node_info in metadata_dict.items():
        result[node_name] = _load_single_tensor(node_info, output_dir, device)
    return result


def load_intermediates(
    metadata_path: Union[str, Path],
    device: Union[str, torch.device] | None = None,
) -> DebugTrace:
    """
    Load intermediate values from disk into an DebugTrace object.

    This function reads the metadata JSON file and associated numpy files
    created by dump_intermediates() and reconstructs an DebugTrace
    object with inputs, outputs, and intermediate tensors.

    Args:
        metadata_path: Path to the metadata JSON file (metadata.json)
                      or the directory containing it.
        device: Optional device to load tensors onto (e.g., 'cpu', 'cuda').
               If None, tensors are loaded to CPU.

    Returns:
        DebugTrace object with inputs, outputs, and intermediates attributes

    Raises:
        ValueError: If metadata is not in the expected format with 'inputs',
                   'outputs', and 'intermediates' keys.

    Example:
        >>> # After dumping intermediates
        >>> metadata_path = dump_intermediates(exported, inputs, './debug_output')
        >>>
        >>> # Load them back
        >>> loaded = load_intermediates('./debug_output/metadata.json')
        >>> for name, tensor in loaded.intermediates.items():
        ...     print(f"{name}: {tensor.shape}")
        >>>
        >>> # Load to specific device
        >>> loaded = load_intermediates(metadata_path, device='cuda:0')

    """
    # Convert to Path
    metadata_path = Path(metadata_path)

    # If a .aimodelintermediates directory is provided, look for metadata.json in it
    if metadata_path.is_dir():
        if not metadata_path.name.endswith(".aimodelintermediates"):
            raise ValueError(
                f"Expected a .aimodelintermediates directory, but got: {metadata_path}. "
                "Please provide the path to a .aimodelintermediates directory created by save_intermediates()."
            )
        metadata_path = metadata_path / "metadata.json"

    # Check if file exists
    if not metadata_path.exists():
        msg = f"Metadata file not found: {metadata_path}"
        raise FileNotFoundError(msg)

    # Get the directory containing the numpy files
    output_dir = metadata_path.parent

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Validate metadata format
    _validate_metadata_format(metadata)

    # Load all tensors into separate dictionaries
    inputs = _load_tensor_dict(metadata["inputs"], output_dir, device)
    outputs = _load_tensor_dict(metadata["outputs"], output_dir, device)
    intermediates = _load_tensor_dict(metadata["intermediates"], output_dir, device)

    # Load mappings if available
    mappings = None
    if "mappings" in metadata:
        mappings_data = metadata["mappings"]

        # Load sources and convert to SourceInfo objects
        sources_dict: dict[str, dict[str, SourceInfo]] = {}
        if "sources" in mappings_data:
            for dialect, ops in mappings_data["sources"].items():
                sources_dict[dialect] = {
                    op_id: SourceInfo.from_dict(op_data)
                    for op_id, op_data in ops.items()
                }

        # Load outputs and convert to OutputMapping objects
        outputs_list: list[OutputMapping] = []
        if "outputs" in mappings_data:
            outputs_list = [
                OutputMapping.from_dict(mapping_data)
                for mapping_data in mappings_data["outputs"]
            ]

        mappings = CompilationMappings(
            sources=sources_dict,
            outputs=outputs_list,
        )

    return DebugTrace(
        inputs=inputs,
        outputs=outputs,
        intermediates=intermediates,
        mappings=mappings,
    )
