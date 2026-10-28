"""Execute introductory documentation against the public API."""

import re
import textwrap
import tomllib
from pathlib import Path

import pandas as pd
import pytest

import qca

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "document",
    ["getting_started/quickstart.rst", "user_guide/calibration.rst"],
)
def test_documentation_examples(document: str) -> None:
    source = (ROOT / "docs" / document).read_text(encoding="utf-8")
    blocks = re.findall(
        r"\.\. code-block:: python\n\n((?:(?:   .*|)\n)+)", source
    )
    assert blocks, "Expected executable Python examples"
    namespace = {
        "data": pd.DataFrame(
            {
                "raw_policy": range(10),
                "raw_capacity": range(10),
                "raw_support": range(10),
            }
        )
    }
    for block in blocks:
        exec(compile(textwrap.dedent(block), document, "exec"), namespace)


def test_release_versions_agree() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_version = re.search(r"^version: (.+)$", citation, re.MULTILINE)
    assert citation_version is not None
    assert metadata["project"]["version"] == qca.__version__
    assert citation_version.group(1) == qca.__version__
