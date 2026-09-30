"""Gold correction store: record, overlay, and drift re-pointing."""

import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab

from latincyreaders.cache.correction_store import (
    CorrectionStore,
    _fileid_to_filename,
)
from latincyreaders.core.base import BaseCorpusReader


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
    BaseCorpusReader._ensure_token_ids(doc)
    return doc


@pytest.fixture
def store(tmp_path):
    return CorrectionStore("test", store_root=tmp_path)


def test_record_and_load_roundtrip(store, vocab):
    doc = _doc(vocab, ["Arma", "virumque", "cano"])
    rec = store.record("f.tess", doc, "t0002", "lemma", "canō", generator="m@1")

    assert rec.token_id == "t0002"
    assert rec.form == "cano"
    assert rec.value == "canō"
    assert rec.ctx["forms"] == ["Arma", "virumque", "cano"]
    assert rec.ctx["i"] == 2

    loaded = store.load("f.tess")
    assert loaded.generator == "m@1"
    assert loaded.count == 1
    assert loaded.corrections[0].value == "canō"


def test_record_replaces_same_field(store, vocab):
    doc = _doc(vocab, ["Arma", "virumque", "cano"])
    store.record("f.tess", doc, "t0002", "lemma", "cano")
    store.record("f.tess", doc, "t0002", "lemma", "canō")  # supersede
    loaded = store.load("f.tess")
    assert loaded.count == 1
    assert loaded.corrections[0].value == "canō"


def test_overlay_applies_gold_and_flags(store, vocab):
    src = _doc(vocab, ["Arma", "virumque", "cano"])
    store.record("f.tess", src, "t0002", "lemma", "canō")

    fresh = _doc(vocab, ["Arma", "virumque", "cano"])  # simulates DocBin reload
    n = store.overlay("f.tess", fresh)
    assert n == 1
    assert fresh[2].lemma_ == "canō"
    assert fresh[2]._.corrected is True
    assert fresh[0]._.corrected is False


def test_overlay_skips_on_form_mismatch(store, vocab):
    src = _doc(vocab, ["Arma", "virumque", "cano"])
    store.record("f.tess", src, "t0002", "lemma", "canō")

    changed = _doc(vocab, ["Arma", "virumque", "canit"])  # t0002 form differs
    assert store.overlay("f.tess", changed) == 0
    assert changed[2]._.corrected is False


def test_repoint_fast_path_with_old_doc(store, vocab):
    old = _doc(vocab, ["Arma", "virumque", "cano"])
    store.record("f.tess", old, "t0002", "lemma", "canō")

    new = _doc(vocab, ["«", "Arma", "virumque", "cano"])  # leading token inserted
    ok, quar = store.repoint("f.tess", new, old_doc=old)
    assert (ok, quar) == (1, 0)

    rec = store.load("f.tess").corrections[0]
    assert rec.token_id == "t0003"  # cano moved
    # overlay now lands on the right token
    assert store.overlay("f.tess", new) == 1
    assert new[3].lemma_ == "canō"


def test_repoint_fallback_via_sentence_ctx(store, vocab):
    old = _doc(vocab, ["Arma", "virumque", "cano"])
    store.record("f.tess", old, "t0002", "lemma", "canō")

    # Old DocBin is gone; re-point using only the stored sentence context.
    new = _doc(vocab, ["«", "Arma", "virumque", "cano"])
    ok, quar = store.repoint("f.tess", new, old_doc=None)
    assert (ok, quar) == (1, 0)
    assert store.load("f.tess").corrections[0].token_id == "t0003"


def test_repoint_records_migration_ledger(store, vocab):
    old = _doc(vocab, ["Arma", "virumque", "cano"])
    store.record("f.tess", old, "t0002", "lemma", "canō", generator="lg@3.9.4")

    new = _doc(vocab, ["«", "Arma", "virumque", "cano"])
    store.repoint("f.tess", new, old_doc=old, to_generator="lg@3.9.6")

    revs = store.migrations()
    assert len(revs) == 1
    assert revs[0]["from_generator"] == "lg@3.9.4"
    assert revs[0]["to_generator"] == "lg@3.9.6"
    assert revs[0]["repointed"] == 1
    assert revs[0]["quarantined"] == 0
    assert (store.store_dir / "head.json").exists()
    # The set is re-stamped to the new generator.
    assert store.load("f.tess").generator == "lg@3.9.6"


def test_repoint_quarantines_vanished_word(store, vocab):
    old = _doc(vocab, ["Arma", "virumque", "cano"])
    store.record("f.tess", old, "t0002", "lemma", "canō")

    # 'cano' is replaced by a different word — must quarantine, not mis-apply.
    new = _doc(vocab, ["Arma", "virumque", "canebat"])
    ok, quar = store.repoint("f.tess", new, old_doc=old)
    assert (ok, quar) == (0, 1)
    assert store.load("f.tess").count == 0
    assert (store.store_dir / "corrections_unresolved.log").exists()


def test_fileid_to_filename_is_collision_free():
    """A naive slash-flatten scheme maps 'a/b' and 'a--b' to the same
    filename; the hash-based scheme must not."""
    a = _fileid_to_filename("a/b", ".corr.json")
    b = _fileid_to_filename("a--b", ".corr.json")
    assert a != b


def test_ctx_alignment_scopes_to_matching_sentence_not_whole_doc(store, vocab):
    """A repeated formulaic phrase (two identical sentences) must not let
    the self-anchoring fallback drift from the correction's own sentence
    to an unrelated later occurrence of the same words."""
    words = ["Pax", "et", "amor", ".", "Pax", "et", "amor", "."]
    doc = Doc(
        vocab, words=words,
        spaces=[True, True, False, True, True, True, False, False],
        sent_starts=[True, False, False, False, True, False, False, False],
    )
    sents = list(doc.sents)
    assert len(sents) == 2

    forms = [t.text for t in sents[0]]
    best = CorrectionStore._best_matching_sentence(forms, doc)
    assert best.start == sents[0].start and best.end == sents[0].end


def test_record_rejects_invalid_correction_value(store, vocab):
    """An invalid value must fail loudly at record time, not be silently
    stored and permanently fail to apply on every future overlay()."""
    doc = _doc(vocab, ["Arma", "virumque", "cano"])
    with pytest.raises(ValueError):
        store.record("f.tess", doc, "t0002", "upos", "not-a-real-upos-tag")
    assert store.load("f.tess") is None

    # The token itself is left unmutated by the failed validation attempt.
    token = doc[2]
    assert token.pos_ != "not-a-real-upos-tag"
