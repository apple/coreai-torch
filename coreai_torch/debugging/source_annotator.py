# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Generic source annotation for Core AI operations.
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TextIO

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
from coreai._compiler.ir import Operation
from coreai.authoring import AIProgram

from .annotations import (
    _AnnotatedListing,
    _Annotation,
    _AnnotationCallback,
    _TextAnnotation,
    _write_line,
)
from .utils import LocationInfo, _walk_operations, get_operation_locations

logger = logging.getLogger(__name__)

__all__ = [
    "_Annotation",
    "_AnnotationCallback",
    "ModulePath",
    "_SourceAnnotator",
    "_TextAnnotation",
    "_annotate_operations",
    "_annotate_source",
    "_operation_in_module",
    "_should_exclude_location",
]


# Module instance path from an operation's stack trace, outermost first (e.g.
# ("HierarchicalModel$1", "SubModel$2")). Disambiguates instances of the same
# submodule that share a source file/line.
ModulePath = tuple[str, ...]


@dataclass(frozen=True)
class _AttributedAnnotation:
    """An annotation together with the module instance it was attributed to."""

    annotation: _Annotation
    """The annotation produced by the callback."""

    module: ModulePath
    """Stack-trace module path of the originating operation (may be empty)."""


def _distinct_files_innermost_first(
    locations: list[LocationInfo],
) -> list[str]:
    """
    List the distinct source files of an operation, innermost first.

    Walks locations from innermost outward, collapsing runs of consecutive
    same-file locations into a single entry.

    Args:
        locations: Operation locations (outermost first, innermost last).

    Returns:
        Distinct filenames ordered innermost first.

    """
    distinct_files: list[str] = []
    for loc in reversed(locations):
        if not distinct_files or distinct_files[-1] != loc.filename:
            distinct_files.append(loc.filename)
    return distinct_files


def _trim_excluded_leaf_modules(
    module: ModulePath,
    locations: list[LocationInfo],
    exclude: Callable[[LocationInfo], bool],
) -> ModulePath:
    """
    Drop trailing module frames whose innermost source files are excluded.

    Leaf module frames (e.g. ``torch.nn.Linear``) often lack their own
    annotatable file. For each innermost distinct source file that is excluded,
    one trailing frame is dropped so the operation collapses into the parent
    module that owns the annotatable line.

    Args:
        module: Stack-trace frames, outermost first.
        locations: Operation locations (outermost first, innermost last).
        exclude: Decides whether a source location is excluded.

    Returns:
        ``module`` with its excluded leaf frames removed.

    """
    excluded_leaves = 0
    for filename in _distinct_files_innermost_first(locations):
        if not exclude(LocationInfo(filename=filename, line=0, col=0)):
            break
        excluded_leaves += 1

    if not excluded_leaves:
        return module
    return module[: max(0, len(module) - excluded_leaves)]


def _get_module_path(
    operation: Operation,
    exclude: Callable[[LocationInfo], bool],
) -> ModulePath:
    """
    Extract the trimmed module instance path from an operation's stack trace.

    Each frame identifies a module instance (e.g. ``"SubModel$2"``), so the full
    path distinguishes instances sharing a source location. Trailing (innermost)
    frames whose source files are excluded are dropped, collapsing leaf modules
    without their own annotatable file into the parent that owns the line.

    Args:
        operation: Operation to extract the module path from.
        exclude: Decides whether a source location is excluded.

    Returns:
        Stack-trace frames (outermost first), trimmed of trailing frames without
        their own annotatable file. Empty when there is no stack trace.

    """
    stack_trace = _mlir.get_stack_trace(operation.location)  # type: ignore[attr-defined]
    if not stack_trace:
        return ()

    return _trim_excluded_leaf_modules(
        tuple(stack_trace),
        get_operation_locations(operation),
        exclude,
    )


def _operation_in_module(operation: Operation, module: ModulePath) -> bool:
    """
    Check whether an operation belongs to a given module instance path.

    Matches ``module`` against the prefix of the operation's stack trace
    (outermost first), so an operation is considered part of ``module`` when it
    originates from that module or any of its descendants. An empty ``module``
    matches every operation.

    Args:
        operation: Operation whose stack trace is inspected.
        module: Module instance path (outermost first) to match against.

    Returns:
        True if the operation belongs to ``module`` (or one of its children).

    """
    stack_trace = tuple(_mlir.get_stack_trace(operation.location) or ())  # type: ignore[attr-defined]
    return stack_trace[: len(module)] == module


def _should_exclude_location(location: LocationInfo) -> bool:
    """
    Default exclusion: torch package files, ``exported_program.py``, and ``"-"``.

    Args:
        location: LocationInfo to check.

    Returns:
        True if location should be excluded.

    """
    file_path = Path(location.filename)

    torch_packages = {"torch", "torchaudio", "torchvision", "torchtext", "torchdata"}
    has_torch_package = any(part in torch_packages for part in file_path.parts)

    return (
        has_torch_package
        or file_path.name == "exported_program.py"
        or location.filename == "-"
    )


def _read_source_file(file_path: Path, output: TextIO) -> list[str] | None:
    """
    Read source file lines, or write an error and return None.

    Args:
        file_path: Path to the source file to read.
        output: Text stream to write errors to.

    Returns:
        Source lines, or None if the file couldn't be read.

    """
    try:
        with open(file_path) as f:
            return f.readlines()
    except FileNotFoundError:
        _write_line(output, f"# Error: File not found: {file_path}")
        return None
    except Exception as e:
        _write_line(output, f"# Error reading file: {e}")
        return None


def _normalize_path(filename: str | Path) -> str:
    """
    Normalize a filename to a canonical absolute path string.

    Used as the attribution dict key so lookups match regardless of how the path
    is expressed. Falls back to the unresolved absolute path if resolution fails.

    Args:
        filename: Source filename to normalize.

    Returns:
        Canonical absolute path string for ``filename``.

    """
    path = Path(filename)
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


class _SourceAnnotator:
    """
    Attribute per-operation annotations to source locations and render them.

    Walks a collection of operations, asks a callback for each operation's
    :class:`_Annotation`, and attributes it to the operation's dominant
    (innermost, non-excluded) source location whose file exists on disk. Supports
    :meth:`get_annotation` (look up annotations for a file/line) and
    :meth:`write` (write annotated source with each annotation above its line).
    """

    def __init__(
        self: Self,
        operations: Iterable[Operation],
        annotate: _AnnotationCallback,
        *,
        exclude: Callable[[LocationInfo], bool] | None = None,
    ) -> None:
        """
        Build the annotation attribution map.

        Args:
            operations: Operations to annotate.
            annotate: Maps each operation to its :class:`_Annotation`, or
                ``None`` to skip it.
            exclude: Optional source-location filter. Defaults to excluding torch
                files, ``exported_program.py``, and ``"-"``.

        """
        self._exclude = exclude if exclude is not None else _should_exclude_location

        # filename -> {line -> attributed annotations}
        self._file_line_annotations: dict[
            str,
            dict[int, list[_AttributedAnnotation]],
        ] = defaultdict(lambda: defaultdict(list))
        # filename -> attribution count (for dominance).
        self._file_counts: dict[str, int] = defaultdict(int)

        for operation in operations:
            annotation = annotate(operation)
            if annotation is None:
                continue
            self._attribute(operation, annotation)

    def _attribute(self: Self, operation: Operation, annotation: _Annotation) -> None:
        """
        Attribute an operation's annotation to its dominant source location.

        Attributes to the innermost non-excluded location whose file exists. The
        originating module instance is stored alongside so different instances of
        the same submodule can be told apart.

        Args:
            operation: Operation the annotation belongs to.
            annotation: _Annotation to attribute.

        """
        locations = get_operation_locations(operation)
        valid_locations = [loc for loc in locations if not self._exclude(loc)]
        if not valid_locations:
            return

        # Innermost valid location.
        last_loc = valid_locations[-1]

        # Only attribute to existing files so we can read them later.
        if not Path(last_loc.filename).exists():
            return

        attributed = _AttributedAnnotation(
            annotation=annotation,
            module=_get_module_path(operation, self._exclude),
        )
        key = _normalize_path(last_loc.filename)
        self._file_line_annotations[key][last_loc.line].append(attributed)
        self._file_counts[key] += 1

    @property
    def files(self: Self) -> list[str]:
        """
        Source files with at least one attributed annotation.

        Returns:
            Filenames ordered by attribution count (descending); the first, when
            present, is the dominant file.

        """
        return sorted(
            self._file_counts,
            key=lambda filename: self._file_counts[filename],
            reverse=True,
        )

    @property
    def dominant_file(self: Self) -> str | None:
        """
        The single most frequently attributed source file.

        Returns:
            The dominant filename, or ``None`` if nothing was attributed.

        """
        files = self.files
        return files[0] if files else None

    def get_annotation(
        self: Self,
        filename: str | Path,
        line: int,
        *,
        module: ModulePath | None = None,
    ) -> list[_Annotation]:
        """
        Get the annotations attributed to a given file and line.

        Annotations at a file/line may come from different module instances; pass
        ``module`` to restrict the result to one instance.

        Args:
            filename: Source file to look up.
            line: 1-based line number within ``filename``.
            module: Optional module instance path (from :meth:`modules_at`) to
                filter by. None returns all instances at that location.

        Returns:
            Annotations attributed to that file/line (and module, when given).

        """
        line_annotations = self._file_line_annotations.get(_normalize_path(filename))
        if line_annotations is None:
            return []
        return [
            attributed.annotation
            for attributed in line_annotations.get(line, [])
            if module is None or attributed.module == module
        ]

    def modules_at(self: Self, filename: str | Path, line: int) -> list[ModulePath]:
        """
        Get the distinct module instance paths attributed to a file and line.

        Args:
            filename: Source file to look up.
            line: 1-based line number within ``filename``.

        Returns:
            Unique module paths (order preserved) with annotations there.

        """
        line_annotations = self._file_line_annotations.get(_normalize_path(filename))
        if line_annotations is None:
            return []
        return list(
            OrderedDict.fromkeys(
                attributed.module for attributed in line_annotations.get(line, [])
            ),
        )

    def _modules_in_file(self: Self, filename: str | Path) -> list[ModulePath]:
        """
        Get the distinct module instance paths attributed anywhere in a file.

        Args:
            filename: Source file to look up.

        Returns:
            Unique module paths (order preserved) with annotations in the file.

        """
        line_annotations = self._file_line_annotations.get(_normalize_path(filename))
        if line_annotations is None:
            return []
        modules: OrderedDict[ModulePath, None] = OrderedDict()
        for attributed_list in line_annotations.values():
            for attributed in attributed_list:
                modules[attributed.module] = None
        return list(modules)

    def write(
        self: Self,
        output: TextIO | None = None,
        *,
        annotate_all_files: bool = False,
    ) -> None:
        """
        Write annotated source to a text stream.

        Reads the relevant source file(s) and writes them back with each
        attributed annotation rendered above the corresponding line. A file
        reached through multiple module instances is written once per instance,
        each copy showing only that instance's annotations.

        Args:
            output: Text stream to write to. Defaults to ``sys.stdout``.
            annotate_all_files: When True, annotate every attributed file
                (ordered by count, descending). When False, only the dominant
                file.

        """
        if output is None:
            output = sys.stdout

        files = self.files
        if not files:
            _write_line(output, "# No valid locations found in operations")
            return

        if not annotate_all_files:
            files = files[:1]

        for filename in files:
            for module in self._modules_in_file(filename):
                self._write_file(Path(filename), output, module=module)

    def _build_listing(
        self: Self,
        file_path: Path,
        source_lines: list[str],
        *,
        module: ModulePath,
    ) -> _AnnotatedListing:
        """
        Build the listing for one source file and module instance.

        Args:
            file_path: Path to the annotated source file.
            source_lines: Lines of that file, as read from disk.
            module: Module instance whose annotations are included.

        Returns:
            The listing, with each annotation attached above its source line.

        """
        module_label = " -> ".join(module) if module else "<unknown>"
        listing = _AnnotatedListing(
            lines=[line.rstrip("\n") for line in source_lines],
            header=f"\n# === {file_path} [{module_label}] ===",
        )

        line_annotations = self._file_line_annotations.get(
            _normalize_path(file_path),
            {},
        )
        for line_number, attributed_annotations in line_annotations.items():
            listing.extend(
                line_number,
                (
                    attributed.annotation
                    for attributed in attributed_annotations
                    if attributed.module == module
                ),
            )
        return listing

    def _write_file(
        self: Self,
        file_path: Path,
        output: TextIO,
        *,
        module: ModulePath,
    ) -> None:
        """
        Write a single source file with one module instance's annotations.

        Args:
            file_path: Path to the source file to annotate.
            output: Text stream to write the annotated source to.
            module: Module instance whose annotations are rendered (also shown in
                the header).

        """
        source_lines = _read_source_file(file_path, output)
        if source_lines is None:
            return

        self._build_listing(file_path, source_lines, module=module).write(output)


def _annotate_operations(
    operations: Iterable[Operation],
    annotate: _AnnotationCallback,
    output: TextIO | None = None,
    *,
    exclude: Callable[[LocationInfo], bool] | None = None,
    annotate_all_files: bool = False,
) -> None:
    """
    Annotate source for a collection of operations.

    Convenience wrapper that builds a :class:`_SourceAnnotator` and writes its
    output.

    Args:
        operations: Operations to annotate.
        annotate: Maps each operation to its :class:`_Annotation`, or ``None`` to
            skip it.
        output: Text stream to write to. Defaults to ``sys.stdout``.
        exclude: Optional source-location filter. Defaults to excluding torch
            files, ``exported_program.py``, and ``"-"``.
        annotate_all_files: When True, annotate every attributed file (ordered by
            count, descending). When False, only the dominant file.

    """
    annotator = _SourceAnnotator(operations, annotate, exclude=exclude)
    annotator.write(output, annotate_all_files=annotate_all_files)


def _annotate_source(
    coreai_program: AIProgram,
    annotate: _AnnotationCallback,
    output: TextIO | None = None,
    *,
    exclude: Callable[[LocationInfo], bool] | None = None,
    annotate_all_files: bool = False,
) -> None:
    """
    Annotate the source of an AIProgram.

    Walks every operation in ``coreai_program`` and delegates to
    :func:`_annotate_operations`.

    Args:
        coreai_program: AIProgram whose source should be annotated.
        annotate: Maps each operation to its :class:`_Annotation`, or ``None`` to
            skip it.
        output: Text stream to write to. Defaults to ``sys.stdout``.
        exclude: Optional source-location filter. Defaults to excluding torch
            files, ``exported_program.py``, and ``"-"``.
        annotate_all_files: When True, annotate every attributed file (ordered by
            count, descending). When False, only the dominant file.

    """
    _annotate_operations(
        _walk_operations(coreai_program),
        annotate,
        output,
        exclude=exclude,
        annotate_all_files=annotate_all_files,
    )
