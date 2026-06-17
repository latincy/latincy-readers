"""Validate that all demo notebooks conform to the nbformat schema.

This guards against malformed notebooks (e.g. a code cell missing the
required ``outputs`` property) that render as "Invalid Notebook" on GitHub.
The check is lightweight: it only parses and validates the JSON structure,
so it does not import the package or download any models/corpora.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbformat.validator import normalize, validate

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks present")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid(path: Path) -> None:
    """Each notebook parses and validates against the nbformat schema."""
    nb = nbformat.read(path, as_version=4)
    # validate() raises nbformat.ValidationError on the first schema violation.
    validate(nb)


@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks present")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_normalized(path: Path) -> None:
    """Notebooks are already normalized (no changes needed to make them valid).

    ``normalize`` returns the number of changes required to repair the
    notebook; a non-zero count means the on-disk file is malformed and would
    be flagged as an invalid notebook by GitHub's renderer.
    """
    nb = nbformat.read(path, as_version=4)
    changes, _ = normalize(nb)
    assert changes == 0, f"{path.name} is not normalized ({changes} change(s) needed)"
