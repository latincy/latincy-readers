"""Pytest fixtures for latincy-readers tests."""

import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tesserae_dir(fixtures_dir) -> Path:
    """Path to Tesserae test fixtures."""
    return fixtures_dir / "tesserae"


@pytest.fixture
def sample_tess_file(tesserae_dir) -> Path:
    """Path to sample .tess file."""
    return tesserae_dir / "tesserae.tess"


@pytest.fixture
def tei_dir(fixtures_dir) -> Path:
    """Path to TEI test fixtures."""
    return fixtures_dir / "tei"


@pytest.fixture
def sample_tei_file(tei_dir) -> Path:
    """Path to sample TEI XML file."""
    return tei_dir / "sample.xml"


@pytest.fixture
def camena_dir(fixtures_dir) -> Path:
    """Path to CAMENA test fixtures."""
    return fixtures_dir / "camena"


@pytest.fixture
def txtdown_dir(fixtures_dir) -> Path:
    """Path to txtdown test fixtures."""
    return fixtures_dir / "txtdown"


@pytest.fixture
def greek_tesserae_dir(fixtures_dir) -> Path:
    """Path to Greek Tesserae test fixtures."""
    return fixtures_dir / "greek_tesserae"


@pytest.fixture
def ud_dir(fixtures_dir) -> Path:
    """Path to UD (Universal Dependencies) test fixtures."""
    return fixtures_dir / "ud"


@pytest.fixture
def wikisource_dir(fixtures_dir) -> Path:
    """Path to WikiSource test fixtures."""
    return fixtures_dir / "wikisource"


@pytest.fixture
def digilibt_dir(fixtures_dir) -> Path:
    """Path to digilibLT test fixtures."""
    return fixtures_dir / "digilibt"


@pytest.fixture
def pta_dir(fixtures_dir) -> Path:
    """Path to PTA test fixtures."""
    return fixtures_dir / "pta"


@pytest.fixture
def csel_dir(fixtures_dir) -> Path:
    """Path to CSEL test fixtures."""
    return fixtures_dir / "csel"


@pytest.fixture
def formulae_dir(fixtures_dir) -> Path:
    """Path to Formulae-Litterae-Chartae test fixtures."""
    return fixtures_dir / "formulae"


@pytest.fixture
def epistolae_dir(fixtures_dir) -> Path:
    """Path to Epistolae test fixtures."""
    return fixtures_dir / "epistolae"


@pytest.fixture
def edh_dir(fixtures_dir) -> Path:
    """Path to EDH test fixtures."""
    return fixtures_dir / "edh"


@pytest.fixture
def gutenberg_dir(fixtures_dir) -> Path:
    """Path to Project Gutenberg test fixtures."""
    return fixtures_dir / "gutenberg"
