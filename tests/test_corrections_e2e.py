"""End-to-end: correct() through a real reader, overlaid on the DocBin base cache."""

import pytest

from latincyreaders import AnnotationLevel, TesseraeReader
from latincyreaders.cache.disk import CacheConfig


@pytest.fixture
def reader(tesserae_dir, tmp_path):
    return TesseraeReader(
        root=tesserae_dir,
        fileids="*.tess",
        annotation_level=AnnotationLevel.MINIMAL,  # blank pipeline, fast
        cache_config=CacheConfig(
            cache_dir=tmp_path / "cache", persist=True, collection="tess-corr",
        ),
        corrections_dir=tmp_path / "corrections",
    )


def test_correction_overlays_on_next_read(reader):
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid))
    assert len(doc) > 0
    tid = doc[0]._.token_id
    assert tid == "t0000"
    original_lemma = doc[0].lemma_

    # Lock in a correction on the first token's lemma.
    reader.correct(fileid, tid, "lemma", "CORRECTED_LEMMA", evidence="test")

    # Next read surfaces the gold value and flags the token.
    doc2 = next(reader.docs(fileid))
    assert doc2[0].lemma_ == "CORRECTED_LEMMA"
    assert doc2[0]._.corrected is True
    assert original_lemma != "CORRECTED_LEMMA"


def test_correction_survives_cache_clear(reader):
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid))
    tid = doc[0]._.token_id
    reader.correct(fileid, tid, "lemma", "STICKY", evidence="test")

    # Wipe both the LRU and the DocBin cache — corrections live elsewhere.
    reader.clear_cache()
    reader._disk_cache.clear()

    doc2 = next(reader.docs(fileid))  # rebuilt from source, overlay re-applied
    assert doc2[0].lemma_ == "STICKY"


def test_correction_survives_generator_rebuild(reader, monkeypatch):
    """A model-version change forces a DocBin rebuild; the correction is carried
    through the read path (old DocBin captured, re-pointed, re-overlaid)."""
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid))
    tid = doc[0]._.token_id
    reader.correct(fileid, tid, "lemma", "REBUILT")

    # Simulate a model upgrade: the generator stamp changes, so the DocBin base
    # is stamp-stale and gets rebuilt on the next read.
    monkeypatch.setattr(reader, "_active_generator", lambda: ("la_core_web_lg", "9.9.9"))
    reader.clear_cache()
    doc2 = next(reader.docs(fileid))
    assert doc2[0].lemma_ == "REBUILT"
    assert doc2[0]._.corrected is True


def test_base_docbin_stays_silver(reader):
    """The overlay never mutates the base cache: a raw DocBin load is uncorrected."""
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid))
    tid = doc[0]._.token_id
    original = doc[0].lemma_
    reader.correct(fileid, tid, "lemma", "GOLD_ONLY")

    # Read the DocBin directly, bypassing the reader's overlay.
    reader.clear_cache()
    raw = reader._disk_cache.get(fileid, reader.vocab, generator=None)
    assert raw is not None
    assert raw[0].lemma_ == original  # base is still silver
