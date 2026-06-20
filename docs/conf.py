"""Sphinx configuration for the PyQCA documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qca import __version__  # noqa: E402

project = "PyQCA"
author = "taishi-yamasaki"
copyright = "2026, taishi-yamasaki"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"PyQCA {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/t-yamsaki/PyQCA/",
    "source_branch": "main",
    "source_directory": "docs/",
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

language = "en"
pygments_style = "sphinx"
pygments_dark_style = "monokai"
