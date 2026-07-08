# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Helpers for constructing debug info locations."""

from __future__ import annotations

import hashlib
import logging
import pathlib
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import torch.fx as fx
from coreai._compiler._mlir_libs._coreaiIR._bindings.mlir import (
    set_block_arg_location,
    set_op_location,
)
from coreai._compiler.dialects.debuginfo.attrs import (
    compile_unit_attr,
    file_attr,
    location_attr,
    metadata_attr,
    subprogram_attr,
)
from coreai._compiler.ir import (
    ArrayAttr,
    Attribute,
    Context,
    DictAttr,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    Operation,
    OpResult,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
)
from typing_extensions import Self

from ._utils import (
    _get_module_hierarchy,
    _ModuleInstanceRegistry,
    parse_traceback,
)

# Set up logger
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Debug Information
# =============================================================================


@dataclass(frozen=True)
class Source:
    """Source information for debug tracking."""

    id: int  # Source ID for this operation
    name: str  # Dialect/stage name: "torch", "coreai", "odix"
    identifiers: list[str]  # Original op names: ["aten.matmul"]


@dataclass(frozen=True)
class OperationID:
    """Operation identifier for tracking operations through compilation."""

    type: str  # Dialect/stage name (e.g., "torch", "coreai")
    value: int  # Unique operation identifier


@dataclass(frozen=True)
class FileLineColLoc:
    """File location information for debug info.

    Can be converted to Core AI Location when context is available.
    """

    filename: str
    line: int
    col: int = 0


@dataclass(frozen=True)
class OutputMap:
    """Output mapping between pipeline stages.

    Represents value-level data flow tracking.
    """

    @dataclass
    class Value:
        """Represents a value at a specific pipeline stage."""

        level: str  # Pipeline level (e.g., "torch", "coreai")
        output: int  # Output index
        op_id: int | None = None  # Operation ID (optional)

    source: "OutputMap.Value"  # Source value
    target: "OutputMap.Value"  # Target value

    @classmethod
    def create_torch_mapping(
        cls,
        source_output: int,
        target_output: int,
        source_op_id: int,
        target_op_id: Optional[int],
        source_level: str = "torch",
        target_level: str = "coreai",
    ) -> "OutputMap":
        """Create an OutputMap with default torch levels for source and target."""
        source_value = cls.Value(
            level=source_level, output=source_output, op_id=source_op_id
        )
        target_value = cls.Value(
            level=target_level, output=target_output, op_id=target_op_id
        )
        return cls(source=source_value, target=target_value)


@dataclass
class DebugInfo:
    """Comprehensive debug information for an operation.

    Combines operation ID, source info, file locations, output maps, and module
    hierarchy.
    """

    operation_id: OperationID | None  # Operation identifier
    source: Source | None  # Source information (optional)
    file_locations: list[FileLineColLoc]  # Stack trace file locations
    output_maps: list[OutputMap]  # Output mappings between stages
    call_stack: list[str]  # Module call stack hierarchy


# =============================================================================
# Core Debug Location Creation Functions
# =============================================================================


def _create_debug_location_from_debug_info(
    debug_info: DebugInfo,
    scope: Attribute,
    context: Context,
    unknown_src: Location,
) -> location_attr:
    """
    Create a debug location from DebugInfo dataclass.

    Uses file_locations as the base location and attaches all other info as
    metadata.

    Args:
        debug_info: DebugInfo dataclass instance containing all debug
            information
        scope: Debug scope attribute (SubprogramAttr, CompileUnitAttr, or
            UnitAttr)
        context: Core AI context
        unknown_src: Unknown location.

    Returns:
        LocationAttr with file location as base and metadata from other
            fields
    """
    # Create metadata from DebugInfo components
    metadata_attrs = []

    # Add source metadata (includes source and source_id) if source exists
    if debug_info.source:
        source_metadata_list = _create_source_metadatas(debug_info.source, context)
        metadata_attrs.extend(source_metadata_list)

    # Add operation ID metadata
    if debug_info.operation_id:
        op_id_metadata = _create_operation_id_metadata(debug_info.operation_id, context)
        metadata_attrs.append(op_id_metadata)

    # Add output maps metadata if present
    if debug_info.output_maps:
        output_maps_metadata = _create_output_maps_metadata(
            debug_info.output_maps, context
        )
        metadata_attrs.append(output_maps_metadata)

    # Add module hierarchy metadata if present
    if debug_info.call_stack:
        call_stack_metadata = _create_call_stack_metadata(
            debug_info.call_stack, context
        )
        metadata_attrs.append(call_stack_metadata)

    # Create location using file_locations as base and metadata for rest
    return _create_stack_trace_debug_location(
        file_locations=debug_info.file_locations,
        scope=scope,
        context=context,
        metadata_attrs=metadata_attrs,
        unknown_src=unknown_src,
    )


# =============================================================================
# Integer Attribute Helper Functions
# =============================================================================


def _create_int_attr(
    value: int, bit_width: int, signed: bool, context: Context
) -> IntegerAttr:
    """Create IntegerAttr with specified type."""
    if signed:
        int_type = IntegerType.get_signed(bit_width, context=context)
    else:
        int_type = IntegerType.get_unsigned(bit_width, context=context)
    return IntegerAttr.get(int_type, value)


def _create_uint64_attr(value: int, context: Context) -> IntegerAttr:
    """Create unsigned 64-bit IntegerAttr."""
    return _create_int_attr(value, 64, False, context)


def _create_uint32_attr(value: int, context: Context) -> IntegerAttr:
    """Create unsigned 32-bit IntegerAttr."""
    return _create_int_attr(value, 32, False, context)


# =============================================================================
# Metadata Creation Functions
# =============================================================================


def _create_operation_id_metadata(
    operation_id: OperationID, context: Context, name: str = "op_id"
) -> metadata_attr:
    """Create operation ID metadata from OperationID dataclass."""
    attrs = {
        "type": StringAttr.get(operation_id.type, context=context),
        "value": _create_uint64_attr(operation_id.value, context),
    }

    return metadata_attr(
        name=name, data=DictAttr.get(attrs, context=context), context=context
    )


def _create_call_stack_metadata(
    module_hierarchy: list[str], context: Context
) -> metadata_attr:
    """Create call_stack metadata from module hierarchy."""
    if not module_hierarchy:
        array_data = ArrayAttr.get([], context=context)
    else:
        string_attrs = [
            StringAttr.get(name, context=context) for name in module_hierarchy
        ]
        array_data = ArrayAttr.get(string_attrs, context=context)

    return metadata_attr(name="call_stack", data=array_data, context=context)


def _source_to_dictionary(source: Source, context: Context) -> DictAttr:
    """Create a Source dictionary."""
    attrs = {
        "name": StringAttr.get(source.name, context=context),
        "id": _create_uint64_attr(source.id, context),
    }

    if source.identifiers:
        id_attrs = [
            StringAttr.get(ident, context=context) for ident in source.identifiers
        ]
        attrs["identifiers"] = ArrayAttr.get(id_attrs, context=context)

    return DictAttr.get(attrs, context=context)


def _create_source_metadatas(source: Source, context: Context) -> list[metadata_attr]:
    """Create source metadata attributes from Source dataclass."""
    metadata_attrs = []

    # Create source dictionary and metadata
    source_dict = _source_to_dictionary(source, context)
    sources_array = ArrayAttr.get([source_dict], context=context)

    metadata_attrs.append(
        metadata_attr(name="sources", data=sources_array, context=context)
    )

    # Create source ID metadata
    source_operation_id = OperationID(type=source.name, value=source.id)
    source_id_metadata = _create_operation_id_metadata(source_operation_id, context)
    metadata_attrs.append(source_id_metadata)

    return metadata_attrs


def _value_to_dictionary(value: OutputMap.Value, context: Context) -> DictAttr:
    """Convert OutputMap.Value to dictionary."""
    attrs = {
        "level": StringAttr.get(value.level, context=context),
        "output": _create_uint32_attr(value.output, context),
    }

    if value.op_id is not None:
        attrs["id"] = _create_uint64_attr(value.op_id, context)

    return DictAttr.get(attrs, context=context)


def _output_map_to_dictionary(output_map: OutputMap, context: Context) -> DictAttr:
    """Convert OutputMap to dictionary."""
    attrs = {
        "source": _value_to_dictionary(output_map.source, context),
        "target": _value_to_dictionary(output_map.target, context),
    }

    return DictAttr.get(attrs, context=context)


def _create_output_maps_metadata(
    output_maps: list[OutputMap], context: Context
) -> metadata_attr:
    """Create output_maps metadata from a list of OutputMap dataclasses."""
    map_dicts = [
        _output_map_to_dictionary(output_map, context) for output_map in output_maps
    ]
    output_maps_array = ArrayAttr.get(map_dicts, context=context)

    return metadata_attr(name="output_maps", data=output_maps_array, context=context)


def _create_unknown_location(
    unknown_src: Location,
    context: Context,
    metadata_attrs: list[metadata_attr] | None = None,
) -> location_attr:
    """Create an unknown debuginfo location with optional metadata.

    Args:
        unknown_src: A pre-created unknown source Location (e.g. filename="", line=0).
        context: Core AI context.
        metadata_attrs: Optional metadata attributes to attach.

    Returns:
        A location_attr with UnitAttr scope and the given metadata.
    """
    scope = UnitAttr.get(context=context)
    return location_attr(
        src=unknown_src,
        scope=scope,
        metadata=metadata_attrs,
        context=context,
    )


def _create_unknown_location_with_operation_id(
    operation_id: OperationID | None,
    unknown_src: Location,
    context: Context,
) -> location_attr:
    """Create an unknown debuginfo location preserving only an operation ID.

    Args:
        operation_id: Optional operation ID to embed as metadata.
        unknown_src: A pre-created unknown source Location.
        context: Core AI context.

    Returns:
        A location_attr with UnitAttr scope and operation_id metadata.
    """
    metadata_attrs: list[metadata_attr] = []
    if operation_id is not None:
        metadata_attrs.append(_create_operation_id_metadata(operation_id, context))
    return _create_unknown_location(
        unknown_src, context, metadata_attrs if metadata_attrs else None
    )


# =============================================================================
# Location Creation Functions
# =============================================================================


def _create_stack_trace_file_locations(
    node: fx.Node,
) -> list[FileLineColLoc]:
    """
    Extract stack trace from FX node and convert to FileLineColLoc dataclasses.

    Args:
        node: FX node containing stack_trace metadata

    Returns:
        List of FileLineColLoc objects from stack trace
    """
    if "stack_trace" not in node.meta:
        return []

    file_locations = []
    traceback_list = parse_traceback(node.meta["stack_trace"])
    for entry in reversed(traceback_list):
        file_locations.append(
            FileLineColLoc(filename=entry.file_path, line=entry.line_number, col=0)
        )

    return file_locations


def _create_debug_info_from_node(
    node: fx.Node,
    source_operation_id: int,
    operation_id: int | None,
    module_registry: _ModuleInstanceRegistry,
) -> DebugInfo:
    """
    Create a DebugInfo object from a torch.fx.Node and operation ID without
    OutputMap.

    Args:
        node: FX node containing metadata for debug info
        operation_id: Unique operation identifier
        module_registry: Module instance registry for hierarchy extraction

    Returns:
        DebugInfo object with operation ID, source info, file locations,
            and module hierarchy, but with empty output_maps list
    """
    source = Source(id=source_operation_id, name="torch", identifiers=[node.name])

    # Extract file locations from stack trace
    file_locations = _create_stack_trace_file_locations(node)

    # Extract module hierarchy/call stack
    call_stack = _get_module_hierarchy(node, module_registry)
    return DebugInfo(
        operation_id=operation_id,
        source=source,
        file_locations=file_locations,
        output_maps=[],
        call_stack=call_stack,
    )


def _create_stack_trace_debug_location(
    file_locations: list[FileLineColLoc],
    scope: Attribute,
    context: Context,
    unknown_src: Location,
    metadata_attrs: list[metadata_attr] | None = None,
) -> location_attr:
    """
    Create debug location from FileLineColLoc list using location_attr with
    fused_with.

    Args:
        file_locations: List of FileLineColLoc objects from stack trace
        scope: Debug scope attribute (SubprogramAttr, CompileUnitAttr, or
            UnitAttr)
        context: Core AI context
        unknown_src: Unknown location.
        metadata_attrs: Optional list of metadata attributes to attach

    Returns:
        DebugInfo LocationAttr from stack trace, or unknown location if no
            locations
    """
    if not file_locations:
        unit_scope = UnitAttr.get(context=context)
        return location_attr(
            src=unknown_src,
            scope=unit_scope,
            metadata=metadata_attrs,
            context=context,
        )

    # Convert first location to Core AI Location for src
    first_location = Location.file(
        filename=file_locations[0].filename,
        line=file_locations[0].line,
        col=file_locations[0].col,
        context=context,
    )

    # Convert remaining locations for fused_with - each must be a DebugInfo
    # location
    fused_with_locations = []
    for file_loc in file_locations[1:]:
        loc = Location.file(
            filename=file_loc.filename,
            line=file_loc.line,
            col=file_loc.col,
            context=context,
        )
        # Create DebugInfo location for each fused location
        debug_info_loc = location_attr(
            src=loc,
            scope=scope,
            fused_with=None,  # No nested fusing to avoid recursion
            metadata=None,  # No metadata on fused locations
            context=context,
        )
        fused_with_locations.append(debug_info_loc)

    # Use location_attr with fused_with and metadata
    return location_attr(
        src=first_location,
        scope=scope,
        fused_with=fused_with_locations if fused_with_locations else None,
        metadata=metadata_attrs,
        context=context,
    )


# =============================================================================
# File and Compile Unit Creation Functions
# =============================================================================


def _get_file_attr(filename: str, context: Context) -> file_attr:
    """Get a FileAttr with SHA256 hashing."""
    # Extract directory and filename from path
    path = pathlib.Path(filename).resolve().absolute()
    directory = str(path.parent)
    file_path = str(path.name)

    # Open the file if we can to hash it. If not, simply return an empty hash.
    sha_hex_str = ""
    try:
        with open(path, "rb") as f:
            file_content = f.read()
            hash_obj = hashlib.sha256(file_content)
            sha_hex_str = hash_obj.hexdigest()
    except (OSError, IOError):
        # If we can't read the file, use empty hash
        pass

    # Construct the attribute and return it
    return file_attr(
        filename=file_path,
        directory=directory,
        sha256_sum=sha_hex_str,
        context=context,
    )


def _get_compile_unit_attr(
    file: file_attr,
    producer: str = "torch",
    optimized: bool = False,
    all_files: Optional[list[file_attr]] = None,
) -> compile_unit_attr:
    """Create a compile unit attribute from a file attribute."""
    if all_files is None:
        all_files = []

    return compile_unit_attr(
        source_language="mlir",
        file=file,
        producer=producer,
        optimized=optimized,
        flags="",
        all_files=all_files,
        context=file.context,
    )


# =============================================================================
# Utility Functions for Operations and Scopes
# =============================================================================


def _get_nested_operations(operation: Operation) -> Iterator[Operation]:
    """Iteratively yield all nested operations from regions/blocks.

    Args:
        operation: The operation to extract nested operations from

    Yields:
        All operations from nested regions and blocks (recursively)
    """
    # Use a stack for iterative traversal to handle deeply nested structures
    operation_stack: list[Operation] = [operation]

    while operation_stack:
        current_op = operation_stack.pop()
        # Iterate through all regions of the current operation
        for region in current_op.regions:
            # Iterate through all blocks in the region
            for block in region:
                # Iterate through all operations in the block
                for nested_op in block:
                    yield nested_op
                    # Add nested operation to stack for further traversal
                    operation_stack.append(nested_op)


# =============================================================================
# Scope Resolution Functions
# =============================================================================


def _get_scope(operation: Operation) -> Optional[Attribute]:
    """Get SymbolRefAttr from operation's sym_name or return None."""
    if operation.attributes and "sym_name" in operation.attributes:
        sym_attr = operation.attributes["sym_name"]
        # SymbolRefAttr.get expects a sequence of strings, not StringAttr
        symbol_name = str(sym_attr).strip('"')
        return SymbolRefAttr.get([symbol_name], context=operation.context)
    return None


def _get_parent_scope(operation: Operation) -> Optional[Attribute]:
    """Get SymbolRefAttr from parent operation with symbol table trait or return None."""
    current = operation.parent
    while current is not None:
        scope = _get_scope(current)
        if scope is not None:
            return scope
        current = current.parent
    return None


def _get_symbol_name(operation: Operation, default_name: str = "unknown") -> str:
    """Get symbol name from operation's sym_name attribute."""
    if (
        hasattr(operation, "attributes")
        and operation.attributes
        and "sym_name" in operation.attributes
    ):
        sym_name = operation.attributes["sym_name"]
        if sym_name:
            return str(sym_name).strip('"')
    return default_name


def _is_debuginfo_location(location: Location) -> bool:
    """Check if location is a DebugInfo LocationAttr by checking the string representation."""
    if location is None:
        return False

    # Check if the location string representation starts with #debuginfo.location
    location_str = str(location)
    return location_str.startswith("#debuginfo.location")


# =============================================================================
# Custom Map Class for Operation to DebugInfo Mapping
# =============================================================================


class _DebugInfoMap:
    """Custom map that assigns unique IDs to Operations for stable identification."""

    def __init__(self):
        self._map: dict[int, DebugInfo] = {}
        self._ops_with_debug_id: set[Operation] = (
            set()
        )  # Track operations with __debug_id
        self._next_id = 0

    def _has_id(self, operation: Operation) -> bool:
        """Check if an operation has a debug ID without assigning one."""
        debug_attr_name = "__debug_id"
        return operation.attributes and debug_attr_name in operation.attributes

    def _get_or_assign_id(self, operation: Operation) -> int:
        """Get or assign a unique ID to an operation using attributes."""
        debug_attr_name = "__debug_id"

        # Check if operation has our debug ID attribute
        if operation.attributes and debug_attr_name in operation.attributes:
            return int(operation.attributes[debug_attr_name])

        # Assign new ID
        current_id = self._next_id
        self._next_id += 1

        # Store in operation attributes using existing helper
        debug_attr = _create_uint64_attr(current_id, operation.context)
        operation.attributes[debug_attr_name] = debug_attr

        # Track this operation so we can clean up later
        self._ops_with_debug_id.add(operation)

        return current_id

    def __getitem__(self, operation: Operation) -> DebugInfo:
        """Get DebugInfo for an operation."""
        return self._map[self._get_or_assign_id(operation)]

    def __setitem__(self, operation: Operation, debug_info: DebugInfo) -> None:
        """Set DebugInfo for an operation."""
        self._map[self._get_or_assign_id(operation)] = debug_info

    def __contains__(self, operation: Operation) -> bool:
        """Check if operation has debug info."""
        if not self._has_id(operation):
            return False
        operation_id = int(operation.attributes["__debug_id"])
        return operation_id in self._map

    def get(
        self, operation: Operation, default: DebugInfo | None = None
    ) -> DebugInfo | None:
        """Get DebugInfo for operation with optional default."""
        if not self._has_id(operation):
            return default
        operation_id = int(operation.attributes["__debug_id"])
        return self._map.get(operation_id, default)

    def __len__(self) -> int:
        """Get number of entries in the map."""
        return len(self._map)

    def clear(self) -> None:
        """Clear all entries and clean up __debug_id attributes."""
        # Clean up __debug_id attributes from operations we tracked
        debug_attr_name = "__debug_id"
        for operation in self._ops_with_debug_id:
            if operation.attributes and debug_attr_name in operation.attributes:
                del operation.attributes[debug_attr_name]

        # Clear all tracking
        self._map.clear()
        self._ops_with_debug_id.clear()
        self._next_id = 0


# =============================================================================
# Debug Info Recorder Class
# =============================================================================


class _DebugInfoRecorder:
    """Recorder for recording debug information with node and results."""

    @dataclass(frozen=True)
    class Config:
        """Configuration for _DebugInfoRecorder."""

        include_stack_trace: bool
        verify_debuginfo_locations: bool

    def __init__(
        self: Self,
        config: "Config" = Config(
            include_stack_trace=True,
            verify_debuginfo_locations=False,
        ),
    ):
        self.config = config
        self.module_registry = _ModuleInstanceRegistry()
        self._operation_id = 0
        self._source_operation_id = 0
        self._debug_info_map = _DebugInfoMap()
        self._current_module: Module | None = None
        self._current_graph: Operation | None = None
        self._op_results: Iterable[OpResult] | None = (
            None  # Track results for deriving new operations
        )
        self._current_node: fx.Node | None = None
        self._file_cache: OrderedDict[str, file_attr] = OrderedDict()
        self._unknown_src: Location | None = None

    def _get_unknown_src(self: Self, context: Context) -> Location:
        """Get or create the shared unknown source location.

        Creates the Location once on first call and reuses it for all
        subsequent calls within the same recorder instance.

        Args:
            context: Core AI context

        Returns:
            A shared Location with filename="", line=0, col=0
        """
        if self._unknown_src is None:
            self._unknown_src = Location.file(
                filename="",
                line=0,
                col=0,
                context=context,
            )
        return self._unknown_src

    def _get_unknown_location(
        self: Self,
        context: Context,
        metadata_attrs: list[metadata_attr] | None = None,
    ) -> location_attr:
        """Create an unknown location reusing the cached unknown_src.

        Args:
            context: Core AI context
            metadata_attrs: Optional list of metadata attributes to attach

        Returns:
            LocationAttr with cached unknown_src and UnitAttr scope
        """
        unknown_src = self._get_unknown_src(context)
        return _create_unknown_location(unknown_src, context, metadata_attrs)

    def _get_unknown_location_with_operation_id(
        self: Self, debug_info: DebugInfo, context: Context
    ) -> location_attr:
        """Create an unknown location with operation_id metadata if available.

        Reuses the cached unknown_src and attaches operation ID as metadata.

        Args:
            debug_info: Debug information containing optional operation_id
            context: Core AI context

        Returns:
            LocationAttr with cached unknown_src and operation_id metadata
        """
        unknown_src = self._get_unknown_src(context)
        return _create_unknown_location_with_operation_id(
            debug_info.operation_id, unknown_src, context
        )

    def _create_operation_location(
        self: Self,
        debug_info: DebugInfo,
        context: Context,
        scope: Attribute | None = None,
    ) -> location_attr:
        """Create a debuginfo location_attr for an operation.

        Args:
            debug_info: Debug information for the location
            context: Core AI context
            scope: Optional scope attribute

        Returns:
            location_attr with debug info, or unknown location with operation
            ID metadata when stack traces are disabled
        """
        if not self.config.include_stack_trace:
            return self._get_unknown_location_with_operation_id(debug_info, context)
        if scope is None:
            scope = UnitAttr.get(context=context)
        return _create_debug_location_from_debug_info(
            debug_info, scope, context, unknown_src=self._get_unknown_src(context)
        )

    def _populate_file_cache_from_debug_info(
        self: Self, debug_info: DebugInfo, context: Context
    ) -> None:
        """Populate file cache from DebugInfo file locations.

        Args:
            debug_info: DebugInfo containing file locations
            context: Core AI context for creating file attributes
        """
        for file_loc in debug_info.file_locations:
            if file_loc.filename and file_loc.filename not in self._file_cache:
                try:
                    file_attr_obj = _get_file_attr(file_loc.filename, context)
                    self._file_cache[file_loc.filename] = file_attr_obj
                except Exception:
                    continue

    def _get_empty_file(self: Self, context: Context) -> file_attr:
        """Create an empty file attribute for use as main file.

        Args:
            context: Core AI context

        Returns:
            Empty file_attr
        """
        return file_attr(
            filename="-",
            directory="",
            sha256_sum="",
            context=context,
        )

    def _get_files_for_graph(
        self: Self,
        graph_operation: Operation,
    ) -> list[file_attr]:
        """Collect all files referenced by operations in this graph.

        Args:
            graph_operation: The graph operation to collect files for

        Returns:
            List of file_attr objects referenced by the graph
        """
        context = graph_operation.context

        # Collect all files referenced by operations in this graph
        graph_files: list[file_attr] = []

        # Collect files from all operations in the graph
        for nested_op in _get_nested_operations(graph_operation):
            if nested_op not in self._debug_info_map:
                continue
            debug_info = self._debug_info_map[nested_op]
            for file_loc in debug_info.file_locations:
                if file_loc.filename and file_loc.filename in self._file_cache:
                    file_attr_obj = self._file_cache[file_loc.filename]
                    if file_attr_obj not in graph_files:
                        graph_files.append(file_attr_obj)

        # Use empty file if no files found
        if len(graph_files) == 0:
            graph_files.append(self._get_empty_file(context=context))

        return graph_files

    def _set_graph_location(self: Self, graph_operation: Operation) -> None:
        """Set debug location for a graph operation.

        Args:
            graph_operation: The graph operation to set location for
        """
        context = graph_operation.context

        # Create operation ID metadata for the graph operation
        graph_operation_id = OperationID(type="coreai", value=self._operation_id)
        op_id_metadata = _create_operation_id_metadata(graph_operation_id, context)
        self._operation_id += 1

        if not self.config.include_stack_trace:
            # Create unknown location with metadata if locations are disabled
            debug_location = self._get_unknown_location(context, [op_id_metadata])
        else:
            # Get the graph name using the helper function
            graph_name = _get_symbol_name(graph_operation, "")

            # Check for parent scope first
            parent_scope = _get_parent_scope(graph_operation)

            if parent_scope is not None:
                # Use parent scope and main file from graph
                graph_files = self._get_files_for_graph(graph_operation)
                main_file = graph_files[0]
                compile_unit_or_scope = parent_scope
            else:
                # Create compile unit only when no parent scope exists
                graph_files = self._get_files_for_graph(graph_operation)
                main_file = graph_files[0]
                compile_unit_or_scope = _get_compile_unit_attr(
                    file=main_file, all_files=graph_files
                )

            # Create subprogram with parent scope or compile unit
            subprogram = subprogram_attr(
                mlir_type=graph_operation.type,
                compile_unit=compile_unit_or_scope,
                file=main_file,
                name=graph_name,
                context=context,
            )

            # Create a proper file location for the graph operation using the main file
            src_location = Location.file(
                filename=f"{main_file.directory}/{main_file.filename}"
                if main_file.directory
                else main_file.filename,
                line=0,
                col=0,
                context=context,
            )

            debug_location = location_attr(
                src=src_location,
                scope=subprogram,
                metadata=[op_id_metadata],
                context=context,
            )

        # Set the location on the graph operation
        set_op_location(graph_operation, debug_location)

    def _set_module_location(self: Self, module: Module) -> None:
        """Set debug location for a module.

        Args:
            module: The Core AI Module to set location for
        """
        # Get the module operation
        module_op = module.operation
        context = module_op.context

        if not self.config.include_stack_trace:
            # Create unknown location if locations are disabled
            debug_location = self._get_unknown_location(context)
        else:
            # Create compile unit for module using files from the graph if available
            if self._file_cache:
                # Use the first file from cache as main file
                main_file = next(iter(self._file_cache.values()))
                all_files = list(self._file_cache.values())
            else:
                # Fallback to empty file if no files in cache
                main_file = self._get_empty_file(context)
                all_files = [main_file]

            compile_unit = _get_compile_unit_attr(
                file=main_file, producer="torch", optimized=False, all_files=all_files
            )

            # Create a proper file location for the module operation
            src_location = Location.file(
                filename=f"{main_file.directory}/{main_file.filename}"
                if main_file.directory
                else main_file.filename,
                line=0,
                col=0,
                context=context,
            )

            debug_location = location_attr(
                src=src_location,
                scope=compile_unit,  # Use compile unit directly as scope
                context=context,
            )

        # Set the location on the module operation
        set_op_location(module_op, debug_location)

    def update_output_maps(
        self: Self,
        source_op_id: int,
        op_results: Iterable[OpResult],
    ) -> None:
        """Update OutputMaps from torch fx node outputs to Core AI OpResults.

        Args:
            node: The torch FX node (source)
            op_results: Core AI OpResults (target)
            source_op_id: Operation ID for the source FX node

        Note:
            If there's a mismatch between FX node outputs and Core AI OpResults
            count, logs the mismatch but continues without adding output maps.
        """
        for i, target_result in enumerate(op_results):
            target_op = target_result.owner
            if not isinstance(target_op, Operation):
                continue

            target_debug_info = self._debug_info_map.get(target_op)
            target_op_id = (
                target_debug_info.operation_id.value if target_debug_info else None
            )

            if target_op_id is None:
                continue

            output_map = OutputMap.create_torch_mapping(
                source_output=i,
                target_output=i,
                source_op_id=source_op_id,
                target_op_id=None,
                source_level="torch",
                target_level="coreai",
            )
            target_debug_info.output_maps.append(output_map)

    @contextmanager
    def record_module(self: Self, module: Module):
        """Context manager for module-level debug recording.

        Args:
            module: The Core AI Module to record debug info for

        Yields:
            The current Core AI module being processed
        """
        # Save the current module
        prev_module = self._current_module
        self._current_module = module

        yield module
        # Set module location before restoring context
        self._set_module_location(module)

        # Verify that each location is a debuginfo location when enabled
        if self.config.verify_debuginfo_locations:
            self._verify_debuginfo_locations(module)

        # Restore previous module and clear caches
        self._current_module = prev_module
        self._file_cache.clear()

    @contextmanager
    def record_graph(self, graph_operation: Operation):
        """Context manager for graph-level debug recording.

        Args:
            graph_operation: The graph Operation to record debug info for

        Yields:
            The current graph operation being processed
        """
        # Store current graph operation
        prev_graph_operation = self._current_graph
        self._current_graph = graph_operation

        yield graph_operation
        # Set graph location first so it can be reused by block arguments
        self._set_graph_location(graph_operation)

        # Ensure ALL operations in the graph have debug locations with operation IDs
        self._ensure_all_operations_have_debug_locations(graph_operation)

        # Clear debug info map (also cleans up __debug_id attributes)
        self._debug_info_map.clear()

        # Clear op_results as they're tied to the current graph context
        self._op_results = None

        # Restore previous graph context
        self._current_graph = prev_graph_operation

    def _create_location_for_node(
        self: Self,
        debug_info: DebugInfo,
    ) -> location_attr:
        """Create debug location for a node.

        Args:
            debug_info: Debug information for the node

        Returns:
            LocationAttr for the node
        """
        if not self._current_graph:
            # Fallback to unknown location if no graph context
            raise RuntimeError("Cannot create location without a current graph context")

        context = self._current_graph.context

        # Create location based on the specified mode and enable_locations setting
        scope = _get_scope(self._current_graph)
        return self._create_operation_location(debug_info, context, scope)

    def _find_new_operations(self: Self) -> list[Operation]:
        """Find operations that were added during the current operation context.

        Uses op_results to find candidate operations, then checks all nested operations.

        Returns:
            List of newly added operations that don't have debug info yet
        """
        if not self._current_graph or not self._op_results:
            return []

        def should_process_op(op) -> bool:
            """Check if an operation should be processed (valid Operation not already seen)."""
            return (
                op is not None
                and isinstance(op, Operation)
                and op not in seen_ops
                and op not in self._debug_info_map
            )

        operations = []
        seen_ops = set()
        for result in self._op_results:
            op = result.owner
            while should_process_op(op):
                seen_ops.add(op)
                operations.append(op)
                op = op.parent

        added_operations = []
        for op in operations:
            added_operations.append(op)
            for nested_op in _get_nested_operations(op):
                added_operations.append(nested_op)

        return added_operations

    def _assign_debug_info_to_operations(
        self: Self,
        operations: list[Operation],
        base_debug_info: DebugInfo,
    ) -> None:
        """Assign debug information to a list of operations.

        Args:
            operations: List of operations to assign debug info to
            base_debug_info: Base debug information to copy and modify
        """
        for operation in operations:
            operation_id = self._operation_id
            op_debug_info = DebugInfo(
                operation_id=OperationID(type="coreai", value=operation_id),
                source=base_debug_info.source,
                file_locations=base_debug_info.file_locations,
                output_maps=[],
                call_stack=base_debug_info.call_stack,
            )
            self._debug_info_map[operation] = op_debug_info
            self._operation_id += 1

    def _process_terminating_operations(self: Self, source_operation_id: int) -> None:
        """Process terminating operations and update output maps.

        Args:
            operations: List of operations to find terminating operations from
            node: Source FX node
            source_operation_id: Source operation ID
        """
        # Use op_results directly since they represent results from terminating operations
        if self._op_results:
            self.update_output_maps(
                source_op_id=source_operation_id,
                op_results=self._op_results,
            )

    @contextmanager
    def record_operation(self: Self, node: fx.Node):
        """Context manager for operation-level debug recording.

        Args:
            node: FX node for the operation

        Yields:
            Tuple of (node, location) where location is created from DebugInfo
        """
        # Store current node
        prev_node = self._current_node
        self._current_node = node
        # Create debug info for this node
        debug_info = _create_debug_info_from_node(
            node=node,
            source_operation_id=self._source_operation_id,
            operation_id=None,
            module_registry=self.module_registry,
        )

        # Populate file cache from debug info file locations
        if self._current_graph:
            self._populate_file_cache_from_debug_info(
                debug_info, self._current_graph.context
            )

        source_op_id = self._source_operation_id
        self._source_operation_id += 1

        # Create location from debug info
        location = self._create_location_for_node(debug_info)

        yield (node, location)
        # Restore previous node
        self._current_node = prev_node
        added_operations = self._find_new_operations()
        if added_operations:
            # Assign debug info to all new operations
            self._assign_debug_info_to_operations(added_operations, debug_info)
            # Process terminating operations and update output maps
            self._process_terminating_operations(source_op_id)

            for operation in added_operations:
                debug_info = self._debug_info_map[operation]
                context = operation.context

                if not self.config.include_stack_trace:
                    # Create unknown location with metadata if locations are disabled
                    location = self._get_unknown_location_with_operation_id(
                        debug_info, context
                    )
                else:
                    # Create location based on the specified mode
                    scope = _get_parent_scope(operation)
                    location = self._create_operation_location(
                        debug_info, context, scope
                    )

                set_op_location(operation, location)

                # Set block argument locations using the operation's location
                self._set_block_argument_locations(operation)

        self._op_results = None

    def _ensure_all_operations_have_debug_locations(
        self: Self, graph_operation: Operation
    ) -> None:
        """Ensure all operations in the graph have debug locations with operation IDs."""
        context = graph_operation.context
        default_scope = UnitAttr.get(context=context)

        # Set locations on block arguments
        self._set_block_argument_locations(graph_operation)

        # Find operations without debug info
        operations_without_debug_info = []
        for nested_op in _get_nested_operations(graph_operation):
            if nested_op not in self._debug_info_map:
                operations_without_debug_info.append(nested_op)

        # Create debug info for operations that don't have it
        for operation in operations_without_debug_info:
            # Create minimal debug info with operation ID (no source needed)
            operation_id = self._operation_id
            debug_info = DebugInfo(
                operation_id=OperationID(type="coreai", value=operation_id),
                source=None,  # No source for operations without FX node
                file_locations=[],
                output_maps=[],
                call_stack=[],
            )
            self._debug_info_map[operation] = debug_info
            self._operation_id += 1

            # Create debug location
            if not self.config.include_stack_trace:
                # Create unknown location with metadata if locations are disabled
                location = self._get_unknown_location_with_operation_id(
                    debug_info, context
                )
            else:
                # Create location based on the specified mode
                scope = _get_scope(graph_operation) or default_scope
                location = self._create_operation_location(debug_info, context, scope)

            set_op_location(operation, location)

            # Set block argument locations using the operation's location
            self._set_block_argument_locations(operation)

    def _set_block_argument_locations(self: Self, operation: Operation) -> None:
        """Set debug locations on block arguments using the operation's location."""
        # Use the operation's location directly
        arg_location = operation.location

        # Set location on all block arguments in all regions
        for region in operation.regions:
            for block in region:
                for arg in block.arguments:
                    set_block_arg_location(arg, arg_location)

    def _verify_debuginfo_locations(self: Self, module: Module) -> None:
        """Verify that each location in the module is a debuginfo location.

        Args:
            module: The Core AI Module to verify locations for

        Raises:
            ValueError: If a non-debuginfo location is found
        """
        module_op = module.operation
        # Check module operation location
        if not _is_debuginfo_location(module_op.location):
            raise ValueError(
                f"Non-debuginfo location found in module: "
                f"Expected debuginfo location, got {str(module_op.location)}"
            )

        # Check all nested operations
        for nested_op in _get_nested_operations(module_op):
            op_name = nested_op.name

            if not _is_debuginfo_location(nested_op.location):
                raise ValueError(
                    f"Non-debuginfo location found in operation {op_name}: "
                    f"Expected debuginfo location, got {str(nested_op.location)}"
                )

            # Check block argument locations
            for region in nested_op.regions:
                for block_idx, block in enumerate(region):
                    for arg_idx, arg in enumerate(block.arguments):
                        if arg.location is not None and not _is_debuginfo_location(
                            arg.location
                        ):
                            raise ValueError(
                                f"Non-debuginfo location found in block argument {arg_idx} "
                                f"in block {block_idx} of operation {op_name}: "
                                f"Expected debuginfo location, got {str(arg.location)}"
                            )
