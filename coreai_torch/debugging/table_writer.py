# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Shared table rendering for the debugging tools.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Protocol, TextIO, runtime_checkable

from rich.console import Console, RenderableType
from rich.table import Table
from typing_extensions import Self

_DEFAULT_WIDTH = 150
"""Console width used when the destination stream has no detectable width."""


@dataclass(frozen=True)
class _Column:
    """One column of a :class:`_TableSpec`."""

    header: str
    """Column heading."""

    justify: str = "left"
    """Cell alignment: ``"left"``, ``"center"``, or ``"right"``."""

    style: str = ""
    """`rich` style applied to every cell in the column (e.g. ``"bold green"``)."""

    no_wrap: bool = False
    """Whether to keep cells on one line. Defaults to False so a value that does
    not fit wraps within its column instead of being cut off."""

    overflow: str = "fold"
    """How to handle content wider than the column: ``"fold"`` (wrap onto the
    next line), ``"ellipsis"``, or ``"crop"``. Defaults to folding so no text is
    lost.
    """


@dataclass(frozen=True)
class _Row:
    """One row of a :class:`_TableSpec`."""

    cells: tuple[str, ...]
    """Cell values, one per :class:`_Column`, already formatted as text."""

    style: str = ""
    """`rich` style applied to the whole row (e.g. ``"red"`` for a removed op)."""


@runtime_checkable
class _TableRow(Protocol):
    """
    Something that can render itself as a table row.

    Implementing this keeps a single formatting definition per record type, so a
    row's cells cannot drift from the header that describes them.
    """

    def to_row(self: Self) -> _Row:
        """
        Return this object's cells.

        Returns:
            The :class:`_Row` describing this object.

        """
        ...


@dataclass
class _TableSpec:
    """A table to render: a title, its columns, and its rows."""

    title: str
    """Table title, written above the table."""

    columns: tuple[_Column, ...]
    """Column definitions, left to right."""

    rows: list[_Row] = field(default_factory=list)
    """Rows, in display order."""

    caption: str | None = None
    """Optional caption written below the table (e.g. a truncation note)."""

    show_lines: bool = False
    """Whether to rule between rows. Turn this on when cells wrap, so it stays
    clear where one row ends and the next begins."""

    row_spacing: int = 0
    """Blank lines below each row's content. One line of spacing keeps rows of
    wrapped text from reading as a single block."""

    def add(self: Self, row: _Row | _TableRow) -> None:
        """
        Append a row, accepting either a :class:`_Row` or a :class:`_TableRow`.

        Args:
            row: _Row to append, or an object that can produce one.

        """
        self.rows.append(row if isinstance(row, _Row) else row.to_row())


def _make_console(
    output: TextIO | None = None,
    *,
    width: int | None = None,
    color: bool | None = None,
    soft_wrap: bool = False,
) -> Console:
    """
    Build a `rich` console writing to *output*.

    Args:
        output: Destination stream. Defaults to ``sys.stdout`` when None.
        width: Console width. Defaults to :data:`_DEFAULT_WIDTH`, since an
            in-memory stream has no width to detect.
        color: Force color on or off. When None, `rich` decides from the stream
            (colors for a terminal, plain text otherwise).
        soft_wrap: When True, lines are emitted whole instead of being wrapped to
            the console width. Use it for content whose line breaks matter, such
            as annotated source.

    Returns:
        A console configured for the given stream.

    """
    stream = sys.stdout if output is None else output
    return Console(
        file=stream,
        width=_DEFAULT_WIDTH if width is None else width,
        no_color=None if color is None else not color,
        highlight=False,
        soft_wrap=soft_wrap,
        markup=False,
    )


def _write_renderable(
    renderable: RenderableType,
    output: TextIO | None = None,
    *,
    width: int | None = None,
    color: bool | None = None,
) -> None:
    """
    Write any `rich` renderable (table, tree, text) to *output*.

    Args:
        renderable: The `rich` renderable to write.
        output: Destination stream. Defaults to ``sys.stdout`` when None.
        width: Console width. Defaults to :data:`_DEFAULT_WIDTH`.
        color: Force color on or off. When None, `rich` decides from the stream.

    """
    _make_console(output, width=width, color=color).print(renderable)


def _build_table(spec: _TableSpec) -> Table:
    """
    Build a `rich` table from *spec* without rendering it.

    Useful when the table has to be composed with other renderables before being
    written.

    Args:
        spec: Description of the table.

    Returns:
        The `rich` table.

    """
    table = Table(
        title=spec.title,
        title_justify="left",
        caption=spec.caption,
        caption_justify="left",
        header_style="bold",
        show_lines=spec.show_lines,
        # (top, right, bottom, left): keep the usual one-space side padding and
        # add the requested blank lines below each row's content.
        padding=(0, 1, spec.row_spacing, 1),
    )
    for column in spec.columns:
        table.add_column(
            column.header,
            justify=column.justify,
            style=column.style or None,
            no_wrap=column.no_wrap,
            overflow=column.overflow,
        )
    for row in spec.rows:
        table.add_row(*row.cells, style=row.style or None)
    return table


def _write_table(
    spec: _TableSpec,
    output: TextIO | None = None,
    *,
    width: int | None = None,
    color: bool | None = None,
) -> None:
    """
    Render *spec* to *output*.

    Args:
        spec: Description of the table.
        output: Destination stream. Defaults to ``sys.stdout`` when None.
        width: Console width. Defaults to :data:`_DEFAULT_WIDTH`.
        color: Force color on or off. When None, `rich` decides from the stream.

    """
    _write_renderable(_build_table(spec), output, width=width, color=color)
