# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Annotation rendering shared by the source and program annotators.

An :class:`_Annotation` renders one operation's information as comment lines. The
annotators differ in *where* those lines go -- above a Python source line, or
beside a line of the printed Core AI program -- but not in how they are rendered,
so :class:`_AnnotatedListing` owns the rendering for both.

Styling is expressed as `rich` style names rather than ANSI escapes, so colour is
applied for a terminal and dropped for a file or in-memory stream.

:class:`_Annotation`, :class:`_TextAnnotation`, and :data:`_AnnotationCallback` are
re-exported by :mod:`~coreai_torch.debugging.source_annotator`; everything else
here is internal to the annotators.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TextIO, runtime_checkable

from coreai._compiler.ir import Operation
from rich.console import Console
from rich.text import Text
from typing_extensions import Self

from .table_writer import _make_console

_ANNOTATION_STYLE = "green"
"""Default style for an annotation comment."""

_DETAIL_STYLE = "yellow"
"""Style for a secondary line under an annotation (e.g. a validation message)."""

# Rendering one line at a time would build a console per call, so consoles are
# kept per output stream. The cache stays tiny: an annotator writes to one stream.
_CONSOLES: dict[int, tuple[TextIO, Console]] = {}


def _console_for(output: TextIO) -> Console:
    """
    Get a console writing to *output*, reusing one per stream.

    Args:
        output: Destination stream.

    Returns:
        A console for that stream.

    """
    key = id(output)
    cached = _CONSOLES.get(key)
    if cached is not None and cached[0] is output:
        return cached[1]
    if len(_CONSOLES) > 8:
        _CONSOLES.clear()
    console = _make_console(output, soft_wrap=True)
    _CONSOLES[key] = (output, console)
    return console


def _write_line(output: TextIO, text: str, style: str = "") -> None:
    """
    Write one styled line to *output*.

    Lines are never wrapped, so a long annotation stays on one line and source
    text is reproduced as-is.

    Args:
        output: Destination stream.
        text: Line contents, without a trailing newline.
        style: `rich` style name, or empty for unstyled output.

    """
    _console_for(output).print(Text(text, style=style or None))


@dataclass(frozen=True)
class _AnnotationLine:
    """One line of an annotation, before a comment prefix or indent is applied."""

    text: str
    """Line contents, without a comment marker."""

    style: str = _ANNOTATION_STYLE
    """`rich` style name. Empty string renders the line unstyled."""

    indent: str = ""
    """Extra indentation, for a line subordinate to the one above it."""


@runtime_checkable
class _Annotation(Protocol):
    """
    Something that can describe itself as annotation lines.

    An annotation supplies text; the listing decides the comment marker and the
    indentation, so the same annotation reads correctly above Python source and
    beside a line of printed Core AI.
    """

    def lines(self: Self) -> Iterable[_AnnotationLine]:
        """
        Return the lines describing this annotation.

        Returns:
            The lines, in display order.

        """
        ...


@dataclass(frozen=True)
class _TextAnnotation:
    """Default :class:`_Annotation`: a single comment line."""

    text: str
    """Annotation text, rendered after the listing's comment prefix."""

    color: str = _ANNOTATION_STYLE
    """`rich` style name for the comment. Empty string renders it unstyled."""

    def lines(self: Self) -> Iterable[_AnnotationLine]:
        """
        Return this annotation as a single line.

        Returns:
            One :class:`_AnnotationLine` holding the text.

        """
        return (_AnnotationLine(self.text, self.color),)

    def write(self: Self, output: TextIO) -> None:
        """
        Write the annotation as a standalone comment line.

        Convenience for rendering one annotation outside a listing; a listing
        applies its own prefix and indent instead.

        Args:
            output: Text stream to write the annotation to.

        """
        _write_line(output, f"# {self.text}", self.color)


# Maps an operation to its _Annotation, or None to skip it.
_AnnotationCallback = Callable[[Operation], "_Annotation | None"]


class _Placement(Enum):
    """Where an annotation goes relative to the line it describes."""

    ABOVE = "above"
    """On its own line(s) before the annotated line, as in annotated source."""

    TRAILING = "trailing"
    """After the annotated line's text, as in an annotated program listing."""


@dataclass
class _AnnotatedListing:
    """
    Text lines together with the annotations attached to them.

    Both annotators reduce to this: a list of lines, and a mapping from 1-based
    line number to the annotations describing that line.
    """

    lines: list[str]
    """Lines of text, without trailing newlines."""

    annotations: dict[int, list[_Annotation]] = field(default_factory=dict)
    """Annotations per 1-based line number, in the order they should appear."""

    header: str | None = None
    """Optional header written above the listing."""

    placement: _Placement = _Placement.ABOVE
    """Whether annotations precede their line or follow it."""

    comment_prefix: str = "#"
    """Comment marker for annotation lines. ``"//"`` for a Core AI program."""

    align_to_line: bool = False
    """Whether to indent an annotation to the line it describes.

    Reads naturally in a program listing, where an annotation sits inside the
    region its operation belongs to. Left off for source, where annotations
    precede the line and a fixed column keeps them scannable.
    """

    def annotate(self: Self, number: int, annotation: _Annotation) -> None:
        """
        Attach an annotation to a line.

        Args:
            number: 1-based line number the annotation describes.
            annotation: _Annotation to attach.

        """
        self.annotations.setdefault(number, []).append(annotation)

    def extend(self: Self, number: int, annotations: Iterable[_Annotation]) -> None:
        """
        Attach several annotations to a line.

        Args:
            number: 1-based line number the annotations describe.
            annotations: Annotations to attach, in display order.

        """
        for annotation in annotations:
            self.annotate(number, annotation)

    def _write_annotations(
        self: Self,
        output: TextIO,
        annotations: Iterable[_Annotation],
        described: str,
    ) -> None:
        """
        Write the annotations describing one line.

        Args:
            output: Text stream to write to.
            annotations: Annotations to write, in display order.
            described: The line they describe, whose indentation they follow when
                :attr:`align_to_line` is set.

        """
        margin = ""
        if self.align_to_line:
            margin = described[: len(described) - len(described.lstrip())]
        for annotation in annotations:
            for line in annotation.lines():
                _write_line(
                    output,
                    f"{margin}{self.comment_prefix}{line.indent} {line.text}",
                    line.style,
                )

    def write(self: Self, output: TextIO) -> None:
        """
        Write the listing with its annotations.

        Args:
            output: Text stream to write to.

        """
        if self.header is not None:
            _write_line(output, self.header, _ANNOTATION_STYLE)

        for number, text in enumerate(self.lines, start=1):
            attached = self.annotations.get(number, ())
            if self.placement is _Placement.ABOVE:
                self._write_annotations(output, attached, text)
                _write_line(output, text)
            elif self.placement is _Placement.TRAILING:
                _write_line(output, text)
                self._write_annotations(output, attached, text)
