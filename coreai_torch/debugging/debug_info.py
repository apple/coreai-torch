# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parser for debug_infos metadata structure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
from coreai._compiler._mlir_libs._coreaiIR._bindings.mlir import (
    set_block_arg_location,
    set_op_location,
)
from coreai._compiler.ir import Location, Operation, WalkResult
from coreai.authoring import AIProgram

from coreai_torch._debug_locations import (
    OperationID,
    _create_unknown_location,
    _create_unknown_location_with_operation_id,
    _get_nested_operations,
)


@dataclass
class SourceInfo:
    """
    Information about a source operation.

    Attributes:
        dialect: The dialect name (e.g., "torch", "coreai")
        id: The operation ID
        identifiers: List of operation identifiers

    """

    dialect: str
    id: int
    identifiers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dialect": self.dialect,
            "id": self.id,
            "identifiers": self.identifiers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceInfo:
        """Create from dictionary representation."""
        return cls(
            dialect=data["dialect"],
            id=data["id"],
            identifiers=tuple(data["identifiers"]),
        )


@dataclass
class OutputMapping:
    """
    Mapping showing value flow between pipeline stages.

    Attributes:
        source_level: Source dialect level (e.g., "torch")
        source_op_id: Source operation ID
        source_output: Source output index
        target_level: Target dialect level (e.g., "coreai")
        target_op_id: Target operation ID
        target_output: Target output index

    """

    source_level: str
    source_op_id: int
    source_output: int
    target_level: str
    target_op_id: int
    target_output: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_level": self.source_level,
            "source_op_id": self.source_op_id,
            "source_output": self.source_output,
            "target_level": self.target_level,
            "target_op_id": self.target_op_id,
            "target_output": self.target_output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputMapping:
        """Create from dictionary representation."""
        return cls(
            source_level=data["source_level"],
            source_op_id=data["source_op_id"],
            source_output=data["source_output"],
            target_level=data["target_level"],
            target_op_id=data["target_op_id"],
            target_output=data["target_output"],
        )


@dataclass
class CompilationMappings:
    """
    Container for compilation pipeline mappings.

    Attributes:
        sources: Dictionary mapping dialects to operation IDs to their source info
        outputs: List of output mappings showing value flow between pipeline stages

    """

    sources: dict[str, dict[str, SourceInfo]]
    outputs: list[OutputMapping]


@dataclass(frozen=True)
class SourceLocation:
    """Represents a source code location."""

    file_name: str
    line: int
    column: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceLocation":
        """Parse from dictionary."""
        return cls(
            file_name=data.get("fileName", ""),
            line=data.get("line", 0),
            column=data.get("column", 0),
        )


@dataclass(frozen=True)
class Metadata:
    """Represents a key-value metadata pair."""

    @dataclass(frozen=True)
    class Value:
        """Represents a metadata value with its type."""

        value: int | str | list["Metadata.Value"] | dict[str, "Metadata.Value"] | None
        value_type: str  # 'integer', 'string', 'array', 'dictionary', 'unit'

        @classmethod
        def _parse_array(cls, array_data: Any) -> list["Metadata.Value"] | None:
            """Parse array metadata value."""
            if array_data is None:
                return None
            parsed_values: list["Metadata.Value"] = []
            for v in array_data:
                parsed = cls.from_dict(v)
                if parsed is not None:
                    parsed_values.append(parsed)
            return parsed_values

        @classmethod
        def _parse_dict(cls, dict_data: Any) -> dict[str, "Metadata.Value"] | None:
            """Parse dictionary metadata value."""
            if dict_data is None:
                return None
            parsed_dict: dict[str, "Metadata.Value"] = {}
            for k, raw_v in dict_data.items():
                parsed = cls.from_dict(raw_v)
                if parsed is not None:
                    parsed_dict[k] = parsed
            return parsed_dict

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> "Metadata.Value | None":
            """Parse metadata value from nested dictionary."""
            # Map type keys to their parsing logic
            if "integer" in data:
                value = (
                    next(iter(data["integer"].values())) if data["integer"] else None
                )
                return cls(value=value, value_type="integer")
            if "string" in data:
                value = next(iter(data["string"].values())) if data["string"] else None
                return cls(value=value, value_type="string")
            if "array" in data:
                array_data = (
                    next(iter(data["array"].values())) if data["array"] else None
                )
                parsed_values = cls._parse_array(array_data)
                return (
                    cls(value=parsed_values, value_type="array")
                    if parsed_values is not None
                    else None
                )
            if "dictionary" in data:
                dict_data = (
                    next(iter(data["dictionary"].values()))
                    if data["dictionary"]
                    else None
                )
                parsed_dict = cls._parse_dict(dict_data)
                return (
                    cls(value=parsed_dict, value_type="dictionary")
                    if parsed_dict is not None
                    else None
                )
            if "unit" in data:
                return cls(value=None, value_type="unit")
            return None

    key: str
    value: "Metadata.Value"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Metadata | None":
        """Parse from dictionary."""
        value = Metadata.Value.from_dict(data["value"])
        if value is None:
            return None
        return cls(
            key=data["key"],
            value=value,
        )


@dataclass(frozen=True)
class DebugInfo:
    """Represents debug information for a single operation."""

    odix_id: int
    name: str
    source_locations: tuple[SourceLocation, ...]
    metadatas: tuple[Metadata, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebugInfo":
        """Parse from dictionary."""
        return cls(
            odix_id=data.get("odixId", 0),
            name=data.get("name", ""),
            source_locations=tuple(
                SourceLocation.from_dict(loc) for loc in data.get("sourceLocations", [])
            ),
            metadatas=tuple(
                m
                for raw_m in data.get("metadatas", [])
                if (m := Metadata.from_dict(raw_m)) is not None
            ),
        )

    def get_metadata(self, key: str) -> Metadata.Value | None:
        """Get metadata value by key."""
        for m in self.metadatas:
            if m.key == key:
                return m.value
        return None

    def get_op_id(self, level: str) -> int | None:
        """
        Get the first operation ID for a given dialect level.

        Args:
        ----
            level: Dialect level name (e.g., "torch", "coreai")

        Returns:
        -------
            Operation ID if present, None otherwise

        """
        ids = self.get_op_ids(level)
        return ids[0] if ids else None

    def get_op_ids(self, level: str) -> list[int]:
        """
        Get all operation IDs for a given dialect level.

        Collects every "op_id" metadata entry whose "type" field matches
        *level* and returns the corresponding "value" fields. Only integer
        values are produced (see :meth:`_get_int_field`).

        Args:
        ----
            level: Dialect level name (e.g., "torch", "coreai")

        Returns:
        -------
            List of operation IDs matching the given level

        """
        ids: list[int] = []
        for metadata in self.metadatas:
            if metadata.key != "op_id":
                continue

            if metadata.value.value_type != "dictionary" or not isinstance(
                metadata.value.value,
                dict,
            ):
                continue

            type_field = self._get_str_field(metadata.value.value, "type")
            if type_field != level:
                continue

            value_field = self._get_int_field(metadata.value.value, "value")
            if value_field is not None:
                ids.append(value_field)

        return ids

    def get_source(self) -> str | None:
        """Get source identifier if present."""
        value = self.get_metadata("source")
        if value and value.value_type == "string" and isinstance(value.value, str):
            return value.value
        return None

    @staticmethod
    def _get_int_field(d: dict[str, Metadata.Value], key: str) -> int | None:
        """Extract integer value from dictionary field."""
        val = d.get(key)
        return (
            val.value
            if val and val.value_type == "integer" and isinstance(val.value, int)
            else None
        )

    @staticmethod
    def _get_str_field(d: dict[str, Metadata.Value], key: str) -> str | None:
        """Extract string value from dictionary field."""
        val = d.get(key)
        return (
            val.value
            if val and val.value_type == "string" and isinstance(val.value, str)
            else None
        )

    @staticmethod
    def _get_dict_field(
        d: dict[str, Metadata.Value],
        key: str,
    ) -> dict[str, Metadata.Value] | None:
        """Extract dictionary value from dictionary field."""
        val = d.get(key)
        return (
            val.value
            if val and val.value_type == "dictionary" and isinstance(val.value, dict)
            else None
        )

    def _parse_mapping_from_dict(
        self,
        mapping_dict: dict[str, Metadata.Value],
        source_level: str,
    ) -> OutputMapping | None:
        """
        Parse OutputMapping from dictionary with 'source' and 'target' keys.

        Args:
        ----
            mapping_dict: Dictionary containing 'source' and 'target' metadata values
            source_level: Expected source dialect level

        Returns:
        -------
            OutputMapping if successfully parsed, None otherwise

        """
        source_dict = self._get_dict_field(mapping_dict, "source")
        target_dict = self._get_dict_field(mapping_dict, "target")

        if not source_dict or not target_dict:
            return None

        # Extract and validate source
        src_level = self._get_str_field(source_dict, "level")
        src_output = self._get_int_field(source_dict, "output")
        src_id = self._get_int_field(source_dict, "id")

        if src_level != source_level or src_output is None or src_id is None:
            return None

        # Extract and validate target
        tgt_level = self._get_str_field(target_dict, "level")
        tgt_output = self._get_int_field(target_dict, "output")
        tgt_id = self._get_int_field(target_dict, "id")

        if not tgt_level or tgt_output is None:
            return None

        # Fallback for tgt_id if not provided in dictionary
        if tgt_id is None:
            op_id = self.get_op_id(tgt_level)
            tgt_id = op_id if isinstance(op_id, int) else None

        if tgt_id is None:
            return None

        return OutputMapping(
            source_level=source_level,
            source_op_id=src_id,
            source_output=src_output,
            target_level=tgt_level,
            target_op_id=tgt_id,
            target_output=tgt_output,
        )

    def get_all_metadata(self, key: str) -> list[Metadata.Value]:
        """Get all metadata values matching the given key.

        Unlike :meth:`get_metadata` which returns only the first match,
        this method collects every ``Metadata`` entry whose key equals
        *key* and returns the corresponding values.

        Args:
        ----
            key: Metadata key to search for.

        Returns:
        -------
            List of matching :class:`Metadata.Value` instances (may be empty).

        """
        return [m.value for m in self.metadatas if m.key == key]

    def get_output_mappings(self, source_level: str) -> list[OutputMapping]:
        """
        Get output mappings from source level by parsing metadata.

        Parses all metadata entries with key "output_maps" containing an
        array of dictionaries, each with 'source' and 'target' fields
        containing level, output, and id.

        Args:
        ----
            source_level: Source dialect level (e.g., "torch", "coreai")

        Returns:
        -------
            List of OutputMapping objects

        """
        mappings: list[OutputMapping] = []
        for output_maps_metadata in self.get_all_metadata("output_maps"):
            if output_maps_metadata.value_type != "array" or not isinstance(
                output_maps_metadata.value, list
            ):
                continue

            for elem in output_maps_metadata.value:
                if elem.value_type == "dictionary" and isinstance(elem.value, dict):
                    mapping = self._parse_mapping_from_dict(elem.value, source_level)
                    if mapping:
                        mappings.append(mapping)

        return mappings


@dataclass(frozen=True)
class DebugInfoRecord:
    """Represents a collection of operation debug infos with an identifier."""

    identifier: str
    operations: tuple[DebugInfo, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebugInfoRecord":
        """Parse from dictionary."""
        return cls(
            identifier=data["identifier"],
            operations=tuple(
                DebugInfo.from_dict(info) for info in data.get("operations", [])
            ),
        )

    def find_by_odix_id(self, odix_id: int) -> DebugInfo | None:
        """Find operation debug info by ODIX ID."""
        for info in self.operations:
            if info.odix_id == odix_id:
                return info
        return None

    def find_by_torch_op_id(self, torch_op_id: int) -> list[DebugInfo]:
        """Find all operation debug infos with given torch operation ID."""
        return [
            info for info in self.operations if info.get_op_id("torch") == torch_op_id
        ]


def parse_debug_infos(debug_infos_bytes: bytes) -> list[DebugInfoRecord]:
    """
    Parse debug_infos bytes into structured objects.

    Args:
    ----
        debug_infos_bytes: JSON bytes from library.debug_infos or model.debug_infos

    Returns:
    -------
        List of DebugInfoRecord objects

    """
    debug_infos_str = debug_infos_bytes.decode("utf-8")
    debug_infos_data = json.loads(debug_infos_str)

    return [DebugInfoRecord.from_dict(item) for item in debug_infos_data]


def _build_coreai_op_map(program: AIProgram) -> dict[int, "Operation"]:
    """Build a mapping from coreai operation ID to MLIR ``Operation``.

    Walks all operations in *program* and collects each one that carries
    a ``"coreai"`` operation ID in its debug location metadata.

    Args:
        program: The AIProgram to inspect.

    Returns:
        Dictionary mapping coreai op ID to the MLIR ``Operation``.
    """
    op_map: dict[int, Operation] = {}

    def _collect(operation: Operation) -> WalkResult:
        op_id = _mlir.get_operation_id(operation.location, "coreai")
        if op_id is not None:
            op_map[op_id.value] = operation
        return WalkResult.ADVANCE

    program._mlir_module.operation.walk(_collect)
    return op_map


def strip_debug_info(program: AIProgram) -> None:
    """Strip debugging information from all operations in the program.

    This is useful for reducing asset size when full debug traces are
    no longer needed.

    Args:
        program: The AIProgram to strip debug info from. Modified in place.
    """
    module = program._mlir_module
    module_op = module.operation
    context = module_op.context

    # Create a shared unknown source location for reuse
    unknown_src = Location.file(filename="", line=0, col=0, context=context)

    # Set module operation location (no operation ID for the module itself)
    module_location = _create_unknown_location(unknown_src, context)
    set_op_location(module_op, module_location)

    # Walk all nested operations and assign fresh sequential IDs
    operation_id = 0
    for nested_op in _get_nested_operations(module_op):
        op_id = OperationID(type="coreai", value=operation_id)
        operation_id += 1

        loc = _create_unknown_location_with_operation_id(op_id, unknown_src, context)
        set_op_location(nested_op, loc)

        # Update block argument locations
        for region in nested_op.regions:
            for block in region:
                for arg in block.arguments:
                    set_block_arg_location(arg, loc)
