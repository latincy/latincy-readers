"""Auto re-point on rebuild: the read-path trigger + DiskCache.load_raw."""

import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab

from latincyreaders import AnnotationLevel, TesseraeReader
from latincyreaders.cache.disk import CacheConfig, DiskCache


@pytest.fixture(autouse=True)
def _exts():
    for name, default in (("token_id", None), ("corrected", False)):
        if not Token.has_extension(name):
            Token.set_extension(name, default=default)


@pytest.fixture
def vocab():
    return Vocab()


def _doc(vocab, words):
    doc = Doc(vocab, words=words, spaces=[True] * (len(words) - 1) + [False])
    return TesseraeReader._ensure_token_ids(doc)


@pytest.fixture
def reader(tesserae_dir, tmp_path):
    return TesseraeReader(
        root=tesserae_dir,
        fileids="*.tess",
        annotation_level=AnnotationLevel.MINIMAL,
        cache_config=CacheConfig(
            cache_dir=tmp_path / "cache", persist=True, collection="rp",
        ),
        corrections_dir=tmp_path / "corrections",
    )


def test_repoint_on_rebuild_carries_correction_across_drift(reader, vocab):
    store = reader._get_correction_store()
    old = _doc(vocab, ["Arma", "virumque", "cano"])
    store.record("f.tess", old, "t0002", "lemma", "canō")

    # A rebuild re-tokenizes: a leading token is inserted, shifting "cano".
    new = _doc(vocab, ["«", "Arma", "virumque", "cano"])
    reader._repoint_on_rebuild("f.tess", new, old)

    rec = store.load("f.tess").corrections[0]
    assert rec.token_id == "t0003"  # followed the shift


def test_repoint_on_rebuild_noop_when_forms_stable(reader, vocab):
    store = reader._get_correction_store()
    old = _doc(vocab, ["Arma", "virumque", "cano"])
    store.record("f.tess", old, "t0002", "lemma", "canō")

    new = _doc(vocab, ["Arma", "virumque", "cano"])  # identical tokenization
    reader._repoint_on_rebuild("f.tess", new, old)

    assert store.load("f.tess").corrections[0].token_id == "t0002"  # unchanged


def test_repoint_on_rebuild_noop_without_old_doc(reader, vocab):
    store = reader._get_correction_store()
    old = _doc(vocab, ["Arma", "virumque", "cano"])
    store.record("f.tess", old, "t0002", "lemma", "canō")

    new = _doc(vocab, ["«", "Arma", "virumque", "cano"])
    reader._repoint_on_rebuild("f.tess", new, None)  # no old doc → no-op
    assert store.load("f.tess").corrections[0].token_id == "t0002"


def test_load_raw_ignores_generator_stamp(tmp_path, vocab):
    cache = DiskCache(CacheConfig(cache_dir=tmp_path, persist=True, collection="lr"))
    doc = _doc(vocab, ["Arma", "cano"])
    cache.put("f.tess", doc, generator="m@1")

    # A stamp mismatch makes get() miss, but load_raw still returns the doc.
    assert cache.get("f.tess", vocab, generator="m@2") is None
    raw = cache.load_raw("f.tess", vocab)
    assert raw is not None
    assert [t._.token_id for t in raw] == ["t0000", "t0001"]