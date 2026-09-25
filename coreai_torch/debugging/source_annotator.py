# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Generic source annotation for Core AI operations.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, TextIO

import coreai._compiler._mlir_libs._coreaiIR._bindings.mlir as _mlir
from coreai._compiler.ir import Operation
from coreai.authoring import AIProgram

from .annotations import (
    _HEADING_STYLE,
    _AnnotatedListing,
    _Annotation,
    _AnnotationCallback,
    _AnnotationLine,
    _TextAnnotation,
    _write_line,
)
from .debug_info import _build_coreai_op_map, get_operation_id
from .utils import (
    LocationInfo,
    _walk_operations,
    get_operation_locations,
    split_module_frame,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_Annotation",
    "module_paths_by_op_id",
    "source_locations_by_op_id",
    "_AnnotationCallback",
    "ModulePath",
    "_SourceAnnotator",
    "_TextAnnotation",
    "_annotate_operations",
    "_annotate_source",
    "annotate_source_with_operations",
    "_operation_in_module",
    "_should_exclude_location",
]


# Module instance path from an operation's stack trace, outermost first (e.g.
# ("HierarchicalModel$1", "SubModel$2")). Disambiguates instances of the same
# submodule that share a source file/line.
ModulePath = tuple[str, ...]

_COREAI_LEVEL = "coreai"
"""Debug-info level the operation ids are read from."""


def _module_step(frame: str) -> dict[str, Any]:
    """
    One step of a module path, as plain values.

    Args:
        frame: Module frame, e.g. ``"Linear$3"``.

    Returns:
        The frame verbatim, plus its type and instance number.

    """
    type_name, instance = split_module_frame(frame)
    return {"name": frame, "type_name": type_name, "instance": instance}


@dataclass(frozen=True)
class _AttributedAnnotation:
    """An annotation together with the module instance it was attributed to."""

    annotation: _Annotation
    """The annotation produced by the callback."""

    module: ModulePath
    """Stack-trace module path of the originating operation (may be empty)."""


@dataclass(frozen=True)
class _ModuleGroup:
    """Every annotation from one module instance, under a single heading.

    Lets one copy of a file carry every instance's annotations, instead of the file
    being written once per instance. Writing per instance duplicates the whole file once
    per layer, and separates what a reader most wants side by side: the same source line
    under two different instances.

    Grouped rather than one label per annotation: a heavily-annotated line carries the
    same path a dozen times otherwise, and the module is the axis a reader scans by, so
    it belongs at the head of the group rather than buried at the start of each line.
    """

    label: str
    """Module instance path, already shortened against the file's common prefix."""

    annotations: tuple[_Annotation, ...]
    """The annotations attributed to this instance, in order."""

    def lines(self: Self) -> Iterable[_AnnotationLine]:
        """
        Return the heading followed by each annotation, indented beneath it.

        Returns:
            The lines, in display order.

        """
        rendered = [_AnnotationLine(f"[{self.label}]", _HEADING_STYLE)]
        for annotation in self.annotations:
            for line in annotation.lines():
                rendered.append(
                    _AnnotationLine(line.text, line.style, "  " + line.indent)
                )
        return rendered

    def data(self: Self) -> dict[str, Any]:
        """
        Return the group as plain values.

        Returns:
            The module path and the annotations attributed to it.

        """
        return {
            "module": self.label,
            "annotations": [annotation.data() for annotation in self.annotations],
        }


def _shorten_against(module: ModulePath, common: ModulePath) -> str:
    """Render *module* with the prefix it shares with every other path removed.

    Every path in a file usually starts at the same top-level module, so printing it on
    each annotation is noise: `TwoLayerDecoder$1 -> TransformerBlock$2 -> RMSNorm$4` says
    little more than `TransformerBlock$2 -> RMSNorm$4`, and the shared part is already in
    the file header.

    Args:
        module: The path to render.
        common: The prefix shared by every path in this file.

    Returns:
        The shortened path, or ``<unknown>`` for an operation with no stack trace.
    """
    if not module:
        return "<unknown>"
    trimmed = module[len(common) :] if module[: len(common)] == common else module
    # A path that *is* the common prefix keeps its last step, so it is not blank.
    return " -> ".join(trimmed) if trimmed else module[-1]


def _common_prefix(modules: Iterable[ModulePath]) -> ModulePath:
    """The longest module path prefix shared by all of *modules*.

    Empty paths are ignored: an operation with no stack trace should not collapse the
    prefix to nothing for every other operation in the file.

    Args:
        modules: The paths to compare.

    Returns:
        The shared prefix, possibly empty.
    """
    paths = [m for m in modules if m]
    if not paths:
        return ()
    prefix = list(paths[0])
    for path in paths[1:]:
        keep = 0
        for left, right in zip(prefix, path):
            if left != right:
                break
            keep += 1
        prefix = prefix[:keep]
        if not prefix:
            break
    return tuple(prefix)


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


def _keep_every_location(_location: LocationInfo) -> bool:
    """
    Exclude nothing, so no frame is trimmed from a module path.

    Args:
        _location: Ignored.

    Returns:
        False, always.

    """
    return False


def module_paths_by_op_id(
    program: AIProgram,
    *,
    exclude: Callable[[LocationInfo], bool] | None = None,
) -> dict[int, ModulePath]:
    """
    Which module instance each Core AI operation came from, by op id.

    Answers "which layer is this?" for a tool that holds op ids rather than operations --
    a placement diff or an intermediates report -- so a finding can name `self.b` instead of
    leaving a reader to infer it from operation order. Paths identify module *instances*, so
    two structurally identical layers are told apart, which operation names cannot do.

    Args:
        program: Program to read stack traces from. Must have been converted with
            `include_stack_trace=True`; without it every operation has no stack trace and the
            result is empty, which is why an operation absent here is not the same as an
            operation at the top level.
        exclude: Source-location filter. Defaults to excluding nothing, which is
            deliberately *not* the annotator's `_should_exclude_location`: that policy trims
            leaf frames whose source file is excluded, so a model built from stock `nn`
            modules collapses to the top-level module and every operation reports the same
            path. Trimming is right for annotating a user's file and wrong for
            provenance, where the module path is the answer being asked for.

    Returns:
        Op id to module path, holding only the operations that have one. Absent means no
        stack trace was recorded, not that the operation belongs to no module.

    """

    policy = exclude if exclude is not None else _keep_every_location
    paths = {}
    for op_id, operation in _build_coreai_op_map(program).items():
        module = _get_module_path(operation, policy)
        if module:
            paths[op_id] = module
    return paths


def source_locations_by_op_id(
    program: AIProgram,
    *,
    exclude: Callable[[LocationInfo], bool] | None = None,
) -> dict[int, LocationInfo]:
    """
    Where in the user's code each Core AI operation came from, by op id.

    The companion to `module_paths_by_op_id`, and the two want opposite policies. This one
    excludes by default: unfiltered, the innermost frame of an operation from a stock module
    is inside torch -- `linear.py:134` -- which is no use to a reader wanting to change
    something. Filtered, it is the line in their own file that called it.

    Kept as `LocationInfo` rather than a formatted string so a caller can open the file;
    render with `format_source_locations` when a line of text is what is wanted.

    A location does not replace a module path: three layers applied on one line share this
    location, and only the module path tells them apart.

    Args:
        program: Program to read locations from.
        exclude: Source-location filter, defaulting to `_should_exclude_location`, so this
            names the same line the annotator would annotate.

    Returns:
        Op id to its innermost non-excluded location -- the annotator's own choice of
        dominant location. Holds only operations that have one: an operation whose every
        frame is excluded is absent rather than mapped to a location outside the user's code.

    """

    policy = exclude if exclude is not None else _should_exclude_location
    locations = {}
    for op_id, operation in _build_coreai_op_map(program).items():
        # Innermost valid location, as `_SourceAnnotator._attribute` picks. Its extra
        # requirement that the file exist is deliberately not applied: that is needed to
        # read the file back for annotation, while a report only has to name the line.
        valid = [
            location
            for location in get_operation_locations(operation)
            if not policy(location)
        ]
        if valid:
            locations[op_id] = valid[-1]
    return locations


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

    def annotation_data(self: Self, filename: str | Path) -> list[dict[str, Any]]:
        """
        Every annotation in a file as plain values, ready to serialize.

        One call per file rather than per line, so a consumer that decorates source
        -- an editor, say -- gets the whole file's annotations without walking it.

        Args:
            filename: Source file to describe.

        Returns:
            One entry per annotation, each holding its 1-based ``line``, the
            ``module`` instance path it was attributed to, and the annotation's own
            fields from :meth:`_Annotation.data`. Ordered by line.

            Each step of ``module`` is split into ``name``, ``type_name`` and
            ``instance``, so a consumer can group instances of one module type
            without parsing ``Linear$3`` itself.

            Identical entries are collapsed. Every member of a fused dispatch
            attributes separately, and members often share a line, so the same
            record would otherwise repeat -- once drawn per member, or counted once
            per member by anything adding the durations up.

        """
        line_annotations = self._file_line_annotations.get(_normalize_path(filename))
        if line_annotations is None:
            return []

        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line in sorted(line_annotations):
            for attributed in line_annotations[line]:
                entry = {
                    "line": line,
                    "module": [_module_step(frame) for frame in attributed.module],
                    **attributed.annotation.data(),
                }
                # The whole entry is the identity: for a timing annotation that is
                # line, module and op ids, and for any other kind whatever it
                # reports. Serialized because the values include lists.
                key = json.dumps(entry, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(entry)
        return entries

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
        combine_modules: bool = True,
    ) -> None:
        """
        Write annotated source to a text stream.

        Reads the relevant source file(s) and writes them back with each
        attributed annotation rendered above the corresponding line.

        Args:
            output: Text stream to write to. Defaults to ``sys.stdout``.
            annotate_all_files: When True, annotate every attributed file
                (ordered by count, descending). When False, only the dominant
                file.
            combine_modules: When True (the default), write each file once and
                label every annotation with the module instance it came from.
                When False, write the file once per module instance, each copy
                showing only that instance's annotations.

                Combining is the default because the per-instance form duplicates
                the whole file, once per instance, and separates what a reader most
                wants side by side -- the same source line under two different
                instances. Set False to isolate one instance.

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
            if combine_modules:
                self._write_file(Path(filename), output, module=None)
                continue
            for module in self._modules_in_file(filename):
                self._write_file(Path(filename), output, module=module)

    def _build_listing(
        self: Self,
        file_path: Path,
        source_lines: list[str],
        *,
        module: ModulePath | None,
    ) -> _AnnotatedListing:
        """
        Build the listing for one source file.

        Args:
            file_path: Path to the annotated source file.
            source_lines: Lines of that file, as read from disk.
            module: Module instance whose annotations are included, or None to
                include every instance and label each annotation with its own.

        Returns:
            The listing, with each annotation attached above its source line.

        """
        modules = self._modules_in_file(file_path)
        if module is None:
            common = _common_prefix(modules)
            shared = " -> ".join(common) if common else "all modules"
            header = f"\n# === {file_path} [{shared}] ==="
        else:
            module_label = " -> ".join(module) if module else "<unknown>"
            header = f"\n# === {file_path} [{module_label}] ==="
            common = ()

        listing = _AnnotatedListing(
            lines=[line.rstrip("\n") for line in source_lines],
            header=header,
            # Follow the indentation of the annotated line, so a comment sits with
            # the code it describes instead of breaking out to column zero.
            align_to_line=True,
        )

        line_annotations = self._file_line_annotations.get(
            _normalize_path(file_path),
            {},
        )
        for line_number, attributed_annotations in line_annotations.items():
            if module is not None:
                listing.extend(
                    line_number,
                    (
                        attributed.annotation
                        for attributed in attributed_annotations
                        if attributed.module == module
                    ),
                )
                continue
            # One heading per module instance, its annotations indented beneath, so a
            # line reached through a dozen instances reads as a dozen short groups
            # rather than a dozen repetitions of the same path.
            grouped: OrderedDict[ModulePath, list[_Annotation]] = OrderedDict()
            for attributed in sorted(attributed_annotations, key=lambda a: a.module):
                grouped.setdefault(attributed.module, []).append(attributed.annotation)
            listing.extend(
                line_number,
                (
                    _ModuleGroup(_shorten_against(path, common), tuple(annotations))
                    for path, annotations in grouped.items()
                ),
            )
        return listing

    def _write_file(
        self: Self,
        file_path: Path,
        output: TextIO,
        *,
        module: ModulePath | None,
    ) -> None:
        """
        Write a single source file's annotations.

        Args:
            file_path: Path to the source file to annotate.
            output: Text stream to write the annotated source to.
            module: Module instance whose annotations are rendered (also shown in
                the header), or None to render every instance's, each labelled.

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


def annotate_source_with_operations(
    coreai_program: AIProgram,
    output: TextIO | None = None,
    *,
    exclude: Callable[[LocationInfo], bool] | None = None,
    annotate_all_files: bool = False,
) -> None:
    """
    Annotate the program's source with the Core AI operations each line produced.

    The plainest question the debug info answers: *what did this line of my model
    become?* Every other annotated-source view layers something on top: a device from a
    :class:`~coreai_torch.debugging.compute_plan.ComputePlan`, a duration from the
    benchmarker. Those need a compiled model and a run. This needs neither, so it works
    on a program straight out of the converter.

    Operations are grouped by the module instance that produced them, so a layer used
    twice shows both instances against the same source line.

    Args:
        coreai_program: Program to annotate. Must have been converted with
            ``include_stack_trace=True``; without it no operation has a source location
            and nothing is annotated.
        output: Text stream to write to. Defaults to ``sys.stdout``.
        exclude: Optional source-location filter. Defaults to excluding torch files,
            ``exported_program.py`` and ``"-"``, so the annotated file is the model's
            own rather than the framework's.
        annotate_all_files: When True, annotate every attributed file (ordered by
            count, descending). When False, only the dominant file.

    """

    def annotate(operation: Operation) -> _Annotation | None:
        op_id = get_operation_id(operation, _COREAI_LEVEL)
        if op_id is None:
            # No coreai id means nothing downstream can refer to this operation, so
            # naming it here would offer a handle that does not work.
            return None
        return _TextAnnotation(f"coreai:{op_id:<5} {operation.name}")

    _annotate_source(
        coreai_program,
        annotate,
        output,
        exclude=exclude,
        annotate_all_files=annotate_all_files,
    )


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
