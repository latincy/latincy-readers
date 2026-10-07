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


def test_persist_cache_does_not_bake_corrections_into_disk(reader):
    """persist_cache() force-flushes the in-memory LRU to disk. The LRU entry
    carries the overlaid gold value (overlay() mutates in place), so a naive
    flush would write the corrected value into the "silver" DocBin cache."""
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid))
    tid = doc[0]._.token_id
    original = doc[0].lemma_
    reader.correct(fileid, tid, "lemma", "GOLD_ONLY")

    doc2 = next(reader.docs(fileid))  # LRU hit, overlay applied in place
    assert doc2[0].lemma_ == "GOLD_ONLY"
    assert reader._cache[fileid][0].lemma_ == "GOLD_ONLY"  # confirms the risk

    reader.persist_cache()

    # The in-memory doc must still read as corrected afterward...
    assert reader._cache[fileid][0].lemma_ == "GOLD_ONLY"

    # ...but the disk base cache must still be silver.
    raw = reader._disk_cache.get(fileid, reader.vocab, generator=None)
    assert raw is not None
    assert raw[0].lemma_ == original


def test_second_correction_keeps_machine_value(reader):
    """Correcting the same token twice: `was` stays the machine value, and
    persist_cache() writes that value (not the first gold value) to disk."""
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid))
    tid = doc[0]._.token_id
    original = doc[0].lemma_
    reader.correct(fileid, tid, "lemma", "GOLD_A")
    rec = reader.correct(fileid, tid, "lemma", "GOLD_B")
    assert rec.was == original

    doc2 = next(reader.docs(fileid))
    assert doc2[0].lemma_ == "GOLD_B"
    reader.persist_cache()
    assert reader._cache[fileid][0].lemma_ == "GOLD_B"
    raw = reader._disk_cache.get(fileid, reader.vocab, generator=None)
    assert raw[0].lemma_ == original
    assert "_lr_silver" not in raw.user_data


def test_persist_cache_restores_current_silver_not_stale_was(reader):
    """After the base changes (e.g. a model upgrade), persist_cache() must
    write the doc's actual pre-overlay value, not the record's older `was`."""
    fileid = reader.fileids()[0]
    tid = next(reader.docs(fileid))[0]._.token_id
    reader.correct(fileid, tid, "lemma", "GOLD")
    store = reader._get_correction_store()
    cset = store.load(fileid)
    cset.corrections[0].was = "OLD_MODEL_LEMMA"  # simulate stale provenance
    store.save(cset)

    doc = next(reader.docs(fileid))
    current_silver = reader._disk_cache.get(fileid, reader.vocab, generator=None)[0].lemma_
    assert doc[0].lemma_ == "GOLD"
    reader.persist_cache()
    raw = reader._disk_cache.get(fileid, reader.vocab, generator=None)
    assert raw[0].lemma_ == current_silver != "OLD_MODEL_LEMMA"


def test_persist_cache_failure_keeps_overlay(reader, monkeypatch):
    """If the disk write raises, the live LRU doc must keep its corrections."""
    fileid = reader.fileids()[0]
    tid = next(reader.docs(fileid))[0]._.token_id
    reader.correct(fileid, tid, "lemma", "GOLD")
    next(reader.docs(fileid))

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(reader._disk_cache, "put", boom)
    with pytest.raises(OSError):
        reader.persist_cache()
    assert reader._cache[fileid][0].lemma_ == "GOLD"


def _overlaid(reader, value="GOLD_A"):
    fileid = reader.fileids()[0]
    tid = next(reader.docs(fileid))[0]._.token_id
    original = reader._disk_cache.get(fileid, reader.vocab, generator=None)[0].lemma_
    reader.correct(fileid, tid, "lemma", value)
    doc = next(reader.docs(fileid))
    assert doc[0].lemma_ == value
    return fileid, tid, original, doc


def test_persist_cache_silver_after_record_changed_behind_reader(reader):
    """The record changes after this reader overlaid its LRU doc (another reader,
    or a direct store write): the disk must still get the machine value."""
    fileid, tid, original, _ = _overlaid(reader)
    store = reader._get_correction_store()
    raw = reader._disk_cache.get(fileid, reader.vocab, generator=None)
    store.record(fileid, raw, tid, "lemma", "GOLD_B")
    reader.persist_cache()
    assert reader._disk_cache.get(fileid, reader.vocab, generator=None)[0].lemma_ == original


def test_persist_cache_silver_after_corrections_deleted(reader):
    fileid, _, original, _ = _overlaid(reader)
    store = reader._get_correction_store()
    for f in store._dir.glob("*.corr.json"):
        f.unlink()
    reader.persist_cache()
    assert reader._disk_cache.get(fileid, reader.vocab, generator=None)[0].lemma_ == original


def test_persist_cache_silver_when_store_unreadable(reader, monkeypatch):
    fileid, _, original, _ = _overlaid(reader)
    store = reader._get_correction_store()

    def broken(*a, **k):
        raise ValueError("corrupt corrections file")

    monkeypatch.setattr(store, "load", broken)
    monkeypatch.setattr(store, "overlay", lambda *a, **k: 0)
    reader.persist_cache()
    assert reader._disk_cache.get(fileid, reader.vocab, generator=None)[0].lemma_ == original


def test_disk_put_refuses_overlaid_doc(reader):
    """put() of an overlaid doc would write gold as silver: it must refuse."""
    fileid, _, original, doc = _overlaid(reader)
    with pytest.raises(ValueError, match="overlaid"):
        reader._disk_cache.put(fileid, doc, generator="x@1")
    raw = reader._disk_cache.load_raw(fileid, reader.vocab)
    assert raw[0].lemma_ == original and "_lr_silver" not in raw.user_data


def test_persist_cache_overlay_failure_does_not_leave_silver_in_lru(reader, monkeypatch):
    """If re-overlay fails after the write, the LRU entry is dropped (not left
    silver) and the remaining entries are still persisted."""
    fileid, _, original, _ = _overlaid(reader)
    store = reader._get_correction_store()

    def broken(*a, **k):
        raise ValueError("corrupt corrections file")

    monkeypatch.setattr(store, "overlay", broken)
    reader.persist_cache()
    assert fileid not in reader._cache
    assert reader._disk_cache.get(fileid, reader.vocab, generator=None)[0].lemma_ == original


def test_lru_cached_before_downstream_failure(reader, monkeypatch):
    """The freshly-annotated doc must be cached before disk_cache.put() runs,
    so a downstream failure (disk full, permissions) doesn't discard NLP
    work that already succeeded."""
    fileid = reader.fileids()[0]
    reader.clear_cache()
    reader._disk_cache.clear()

    def boom(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(reader._disk_cache, "put", boom)

    with pytest.raises(OSError):
        next(reader.docs(fileid))

    assert fileid in reader._cache  # not lost despite the downstream raise


def test_repoint_corrections_invalidates_disk_cache(reader):
    """repoint_corrections() is documented to force a re-anchor; without
    invalidating the disk entry, docs() would just re-serve the same
    (unchanged) cached doc, making the old/new alignment a no-op."""
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid))
    tid = doc[0]._.token_id
    reader.correct(fileid, tid, "lemma", "ANCHORED")

    assert reader._disk_cache.has(fileid)
    reader.repoint_corrections(fileid)
    # A fresh disk entry was written by the forced rebuild inside docs().
    assert reader._disk_cache.has(fileid)

    doc2 = next(reader.docs(fileid))
    assert doc2[0].lemma_ == "ANCHORED"
