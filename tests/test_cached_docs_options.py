"""docs(cache=..., annotation_level=...) for readers using _cached_docs."""

import warnings

import pytest

from latincyreaders import AnnotationLevel, EDHReader, TxtdownReader
from latincyreaders.cache import CacheConfig


@pytest.fixture(params=["txtdown", "edh"])
def reader(request, txtdown_dir, edh_dir, tmp_path):
    cls, root = {
        "txtdown": (TxtdownReader, txtdown_dir),
        "edh": (EDHReader, edh_dir),
    }[request.param]
    return cls(
        root=root,
        annotation_level=AnnotationLevel.MINIMAL,
        cache_config=CacheConfig(
            cache_dir=tmp_path / "cache", persist=True, collection="opts-test"
        ),
        corrections_dir=tmp_path / "corrections",
    )


def test_cache_false_stores_nothing(reader):
    fileid = reader.fileids()[0]
    doc = next(reader.docs(fileid, cache=False))
    assert len(doc) > 0
    assert fileid not in reader._cache
    assert not reader._disk_cache.has(fileid)


def test_cache_false_still_overlays_corrections(reader):
    fileid = reader.fileids()[0]
    tid = next(reader.docs(fileid))[0]._.token_id
    reader.correct(fileid, tid, "lemma", "GOLD")
    fresh = next(reader.docs(fileid, cache=False))
    assert fresh[0].lemma_ == "GOLD"
    assert fresh[0]._.corrected


def test_cache_false_bypasses_cached_doc(reader):
    fileid = reader.fileids()[0]
    cached = next(reader.docs(fileid))
    fresh = next(reader.docs(fileid, cache=False))
    assert fresh is not cached


def test_level_override_warns(reader):
    fileid = reader.fileids()[0]
    with pytest.warns(UserWarning, match="ignored"):
        next(reader.docs(fileid, annotation_level=AnnotationLevel.FULL))


def test_level_override_as_string_warns_not_crashes(reader):
    fileid = reader.fileids()[0]
    with pytest.warns(UserWarning, match="ignored"):
        next(reader.docs(fileid, annotation_level="full"))


def test_same_level_as_string_does_not_warn(reader):
    fileid = reader.fileids()[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        next(reader.docs(fileid, annotation_level="minimal"))


def test_cache_false_repoints_against_stale_disk_entry(reader):
    """cache=False still re-points corrections whose ids belong to an older
    tokenization (stale DocBin entry) before overlaying them."""
    from spacy.tokens import Doc

    fileid = reader.fileids()[0]
    real = next(reader.docs(fileid, cache=False))
    words = [t.text for t in real]
    assert len(words) > 3
    # Older tokenization: first token missing, so ids are shifted by one.
    old = Doc(real.vocab, words=words[1:], spaces=[True] * (len(words) - 2) + [False])
    old._.fileid = fileid
    reader._ensure_token_ids(old)
    reader._get_correction_store().record(fileid, old, "t0000", "lemma", "GOLD")
    reader._disk_cache.put(fileid, old, generator="old@0")

    fresh = next(reader.docs(fileid, cache=False))
    assert fresh[1].lemma_ == "GOLD"  # followed the shift
    assert fresh[0].lemma_ != "GOLD"

    # Further reads (uncached and cached) must not re-point a second time.
    again = next(reader.docs(fileid, cache=False))
    assert again[1].lemma_ == "GOLD" and again[2].lemma_ != "GOLD"
    cached = next(reader.docs(fileid))
    assert cached[1].lemma_ == "GOLD" and cached[2].lemma_ != "GOLD"
    rec = reader._get_correction_store().load(fileid).corrections[0]
    assert rec.token_id == "t0001"


def test_same_level_does_not_warn(reader):
    fileid = reader.fileids()[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        next(reader.docs(fileid, annotation_level=AnnotationLevel.MINIMAL))
