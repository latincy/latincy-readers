"""Acceptance: txtdown routed through the shared cache + correction choke-point.

Mirrors the plaintext-ll workflow (a .txtd source) on a small committed fixture:
build → cache → correct → re-read (overlay) → survive clear → base stays silver.
"""

from pathlib import Path

import pytest

from latincyreaders import AnnotationLevel
from latincyreaders.cache.disk import CacheConfig
from latincyreaders.readers.txtdown import TxtdownReader

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "txtdown"


@pytest.fixture
def reader(tmp_path):
    return TxtdownReader(
        root=FIXTURE_DIR,
        fileids="mini.txtd",
        annotation_level=AnnotationLevel.MINIMAL,  # blank pipeline, fast
        cache_config=CacheConfig(
            cache_dir=tmp_path / "cache", persist=True, collection="txtd-mini",
        ),
        corrections_dir=tmp_path / "corrections",
    )


def test_txtdown_now_uses_docbin_cache(reader):
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid))
    assert len(doc) > 0
    assert doc[0]._.token_id == "t0000"
    # The DocBin base cache was populated (txtdown previously had none).
    assert reader._disk_cache.has(fileid)

    # Second read (after dropping the LRU) is served from the DocBin base.
    reader.clear_cache()
    misses_before = reader.cache_stats()["misses"]
    doc2 = next(reader.docs(fileid))
    assert reader.cache_stats()["misses"] == misses_before  # no re-parse
    assert doc2[0]._.token_id == "t0000"


def test_txtdown_correction_overlays_and_survives_clear(reader):
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid))
    tid = doc[2]._.token_id  # "cano"
    assert doc[2].text == "cano"
    reader.correct(fileid, tid, "lemma", "cano", evidence="acceptance")

    doc2 = next(reader.docs(fileid))
    assert doc2[2].lemma_ == "cano"
    assert doc2[2]._.corrected is True

    # Wipe both caches — the correction lives outside them.
    reader.clear_cache()
    reader._disk_cache.clear()
    doc3 = next(reader.docs(fileid))
    assert doc3[2].lemma_ == "cano"
    assert doc3[2]._.corrected is True


def test_txtdown_base_docbin_stays_silver(reader):
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid))
    tid = doc[2]._.token_id
    original = doc[2].lemma_
    reader.correct(fileid, tid, "lemma", "GOLD_ONLY")

    reader.clear_cache()
    raw = reader._disk_cache.get(fileid, reader.vocab, generator=None)
    assert raw is not None
    assert raw[2].lemma_ == original  # base uncorrected


def test_txtdown_spans_and_metadata_survive_docbin(reader):
    """The generic token-extension + span preservation keeps txtdown's structure."""
    fileid = reader.fileids()[0]
    next(reader.docs(fileid))  # populate DocBin

    reader.clear_cache()
    raw = reader._disk_cache.get(fileid, reader.vocab, generator=None)
    assert raw is not None
    # Section/line citation spans built during production survive the round-trip.
    assert "sections" in raw.spans or "lines" in raw.spans
    assert raw._.metadata is not None
