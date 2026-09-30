"""Durable token ids: minting + DocBin user_data round-trip."""

import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab

from latincyreaders.cache.disk import CacheConfig, DiskCache
from latincyreaders.core.base import BaseCorpusReader


@pytest.fixture(autouse=True)
def _token_id_ext():
    if not Token.has_extension("token_id"):
        Token.set_extension("token_id", default=None)


@pytest.fixture
def vocab():
    return Vocab()


@pytest.fixture
def doc(vocab):
    return Doc(vocab, words=["Arma", "virumque", "cano"], spaces=[True, True, False])


def test_ensure_token_ids_mints_positional(doc):
    BaseCorpusReader._ensure_token_ids(doc)
    assert [t._.token_id for t in doc] == ["t0000", "t0001", "t0002"]


def test_ensure_token_ids_only_mints_when_missing(doc):
    doc[1]._.token_id = "kept"
    BaseCorpusReader._ensure_token_ids(doc)
    assert [t._.token_id for t in doc] == ["t0000", "kept", "t0002"]


def test_token_ids_survive_docbin_roundtrip(tmp_path, vocab, doc):
    BaseCorpusReader._ensure_token_ids(doc)
    cache = DiskCache(CacheConfig(cache_dir=tmp_path, persist=True, collection="t"))
    cache.put("f.tess", doc)

    loaded = cache.get("f.tess", vocab)
    assert loaded is not None
    assert [t._.token_id for t in loaded] == ["t0000", "t0001", "t0002"]


def test_no_token_ids_no_stash(tmp_path, vocab, doc):
    """A doc without ids round-trips cleanly (nothing stashed)."""
    cache = DiskCache(CacheConfig(cache_dir=tmp_path, persist=True, collection="t"))
    cache.put("f.tess", doc)
    loaded = cache.get("f.tess", vocab)
    assert loaded is not None
    assert all(t._.token_id is None for t in loaded)
