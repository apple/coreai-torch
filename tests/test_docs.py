# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Verify the documentation build emits the llms.txt artifacts.

The ``sphinx_llm.txt`` extension (declared in the ``docs`` extra) writes
``llms.txt`` (index), ``llms-full.txt`` (corpus), and per-page ``*.html.md``
files described by https://llmstxt.org/, which power the "Copy page" / "View as
Markdown" affordance in the Shibuya theme. It produces them in a *separate*
``sphinx-build -b markdown`` subprocess that it spawns, which re-reads
``docs/conf.py``.

The test is skipped where the ``docs`` extra is absent (e.g. a plain
``uv sync --extra test``); it runs in CI because ``pr-checks-linux`` syncs
``--extra docs``. Notebook execution is disabled via the ``NB_EXECUTION_MODE``
environment variable rather than a ``-D`` override: the env var is inherited by
both the parent build and sphinx-llm's child markdown subprocess (``-D`` reaches
only the parent), so the Metal-kernel guide notebooks — which cannot run on
Linux CI — are skipped in the build that actually writes the artifacts.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sphinx_llm")

_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
_ARTIFACTS = ("llms.txt", "llms-full.txt", "index.html.md")


def test_docs_build_emits_llms_artifacts(tmp_path: Path) -> None:
    """A docs build generates the llms.txt index, full corpus, and per-page Markdown."""
    output_dir = tmp_path / "html"
    # NB_EXECUTION_MODE=off is inherited by sphinx-llm's child markdown build
    # (a `-D` flag would not reach it); the Metal-kernel notebooks can't run on Linux CI.
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", str(_DOCS_DIR), str(output_dir)],
        capture_output=True,
        text=True,
        env={**os.environ, "NB_EXECUTION_MODE": "off"},
    )
    assert result.returncode == 0, (
        f"sphinx-build failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    for artifact in _ARTIFACTS:
        path = output_dir / artifact
        assert path.exists() and path.stat().st_size > 0, (
            f"Expected non-empty docs artifact {path}.\n"
            f"sphinx-build output:\n{result.stdout}\n{result.stderr}"
        )
    assert "# coreai-torch" in (output_dir / "llms.txt").read_text()
