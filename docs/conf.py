# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import json
import os
from pathlib import Path

project = "coreai-torch"
author = "Apple - On-Device Machine Learning"
copyright = "2026, Apple Inc."

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
]

try:
    import sphinx_llm  # noqa: F401

    extensions.append("sphinx_llm.txt")
except ImportError:
    pass

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

# Notebook execution
nb_execution_mode = "auto"
nb_remove_code_outputs = True

# Theme
html_theme = "shibuya"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["sidebar-wrap.js", "copy-page-button.js"]
templates_path = ["_templates"]
html_show_sourcelink = False

# Pygments (syntax highlighting) style
pygments_style = "friendly"

# llms.txt / per-page Markdown generation (powers the "Copy page" / "View as
# Markdown" affordance in the copy-page-button override).
llms_txt_description = "coreai-torch converts PyTorch models (torch.export ExportedProgram) to Core AI format."
llms_txt_build_parallel = False

html_theme_options = {
    "accent_color": "indigo",
    "dark_code": False,
    "globaltoc_expand_depth": 1,
    "show_ai_links": True,
    "github_url": "https://github.com/apple/coreai-torch",
    "nav_links": [
        {"title": "Getting Started", "url": "getting-started/installation"},
        {"title": "Guides", "url": "guides/conversion-workflows"},
        {"title": "API", "url": "api/TorchConverter"},
        {"title": "coreai-core", "url": "coreai-core/index"},
    ],
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
    ],
}

# Version switcher configuration (overridable via env vars).
# - VERSION_MATCH=local (default for local builds) injects an extra
#   "Local preview" entry at the top of the switcher and highlights it.
#   The entry's URL is a no-op so clicks stay on the local page.
# - deploy-integration.sh sets VERSION_MATCH=integration so the deployed
#   build highlights the integration row and omits the local one.
version_match = os.environ.get("VERSION_MATCH", "local")

_versions_file = Path(__file__).parent / "_static" / "versions.json"
try:
    _versions_data = json.loads(_versions_file.read_text())
except (FileNotFoundError, json.JSONDecodeError):
    _versions_data = []
if version_match == "local":
    _versions_data = [{"name": "local preview", "version": "local", "url": "#"}] + [
        v for v in _versions_data if v.get("version") != "local"
    ]
versions_json_inline = json.dumps(_versions_data).replace("</", r"<\/")

html_context = {
    "source_type": "github",
    "source_user": "apple",
    "source_repo": "coreai-torch",
    "source_version": "main",
    "source_docs_path": "/docs/",
    "version_match": version_match,
    "versions_json_inline": versions_json_inline,
}

# Source file suffixes
source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Suppress warnings for missing cross-reference targets in notebooks
suppress_warnings = ["myst.xref_missing", "misc.highlighting_failure"]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

root_doc = "index"
