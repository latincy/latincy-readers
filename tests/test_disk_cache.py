"""Tests for persistent disk caching."""

import time

import pytest
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab

from latincyreaders.cache.disk import CacheConfig, DiskCache, _fileid_hash


class TestCacheConfig:
    """Tests for CacheConfig defaults and behaviour."""

    def test_default_values(self):
        cfg = CacheConfig()
        assert cfg.persist is False
        assert cfg.ttl_days is None
        assert cfg.collection is None
        assert cfg.cache_dir.name == ".latincy_cache"

    def test_custom_values(self, tmp_path):
        cfg = CacheConfig(
            cache_dir=tmp_path / "my_cache",
            persist=True,
            ttl_days=30,
            collection="tesserae",
        )
        assert cfg.persist is True
        assert cfg.ttl_days == 30
        assert cfg.collection == "tesserae"


class TestDiskCache:
    """Tests for DiskCache round-trip and lifecycle."""

    @pytest.fixture
    def cache(self, tmp_path):
        cfg = CacheConfig(
            cache_dir=tmp_path / "cache",
            persist=True,
            collection="test",
        )
        return DiskCache(cfg)

    @pytest.fixture
    def vocab(self):
        return Vocab()

    @pytest.fixture
    def sample_doc(self, vocab):
        doc = Doc(vocab, words=["Arma", "virumque", "cano"], spaces=[True, True, False])
        doc._.fileid = "vergil.aen.tess"
        return doc

    def test_put_and_get(self, cache, vocab, sample_doc):
        """Round-trip: put then get returns equivalent doc."""
        cache.put("vergil.aen.tess", sample_doc)
        loaded = cache.get("vergil.aen.tess", vocab)

        assert loaded is not None
        assert loaded.text == sample_doc.text
        assert len(loaded) == len(sample_doc)

    def test_has(self, cache, sample_doc):
        assert cache.has("vergil.aen.tess") is False
        cache.put("vergil.aen.tess", sample_doc)
        assert cache.has("vergil.aen.tess") is True

    def test_get_missing_returns_none(self, cache, vocab):
        assert cache.get("nonexistent.tess", vocab) is None

    def test_invalidate(self, cache, vocab, sample_doc):
        cache.put("vergil.aen.tess", sample_doc)
        assert cache.has("vergil.aen.tess") is True

        cache.invalidate("vergil.aen.tess")
        assert cache.has("vergil.aen.tess") is False
        assert cache.get("vergil.aen.tess", vocab) is None

    def test_clear(self, cache, vocab, sample_doc):
        cache.put("file1.tess", sample_doc)
        cache.put("file2.tess", sample_doc)
        assert cache.stats()["entries"] == 2

        cache.clear()
        assert cache.stats()["entries"] == 0

    def test_stats(self, cache, sample_doc):
        stats = cache.stats()
        assert stats["collection"] == "test"
        assert stats["entries"] == 0

        cache.put("vergil.aen.tess", sample_doc)
        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["size_bytes"] > 0

    def test_persist_false_no_op(self, tmp_path, vocab):
        """When persist=False, put/get are no-ops."""
        cfg = CacheConfig(cache_dir=tmp_path / "cache", persist=False)
        cache = DiskCache(cfg)

        doc = Doc(vocab, words=["test"], spaces=[False])
        cache.put("test.tess", doc)
        assert cache.get("test.tess", vocab) is None

    def test_extra_metadata_in_manifest(self, cache, sample_doc, tmp_path):
        cache.put("test.tess", sample_doc, annotation_level="FULL", model_name="la_core_web_lg")
        manifest = cache._load_manifest()
        entry = manifest["test.tess"]
        assert entry["annotation_level"] == "FULL"
        assert entry["model_name"] == "la_core_web_lg"


class TestDiskCacheTTL:
    """Tests for TTL expiration."""

    @pytest.fixture
    def cache_with_ttl(self, tmp_path):
        cfg = CacheConfig(
            cache_dir=tmp_path / "cache",
            persist=True,
            ttl_days=1,
            collection="ttl_test",
        )
        return DiskCache(cfg)

    def test_fresh_entry_not_expired(self, cache_with_ttl, tmp_path):
        vocab = Vocab()
        doc = Doc(vocab, words=["test"], spaces=[False])
        cache_with_ttl.put("fresh.tess", doc)

        assert cache_with_ttl.has("fresh.tess") is True
        assert cache_with_ttl.refresh_check("fresh.tess") is False

    def test_stale_entry_expired(self, cache_with_ttl, tmp_path):
        vocab = Vocab()
        doc = Doc(vocab, words=["test"], spaces=[False])
        cache_with_ttl.put("stale.tess", doc)

        # Manually set timestamp to 2 days ago
        manifest = cache_with_ttl._load_manifest()
        manifest["stale.tess"]["timestamp"] = time.time() - 2 * 86400
        cache_with_ttl._save_manifest(manifest)
        cache_with_ttl._manifest = None  # force reload

        assert cache_with_ttl.has("stale.tess") is False
        assert cache_with_ttl.refresh_check("stale.tess") is True
        assert cache_with_ttl.get("stale.tess", vocab) is None


class TestRemorphRoundtrip:
    """Tests for remorph stash/restore through DocBin."""

    @pytest.fixture
    def cache(self, tmp_path):
        cfg = CacheConfig(
            cache_dir=tmp_path / "cache", persist=True, collection="test"
        )
        return DiskCache(cfg)

    def test_remorph_survives_roundtrip(self, cache):
        from spacy.tokens import Token

        if not Token.has_extension("remorph"):
            Token.set_extension("remorph", default=None)

        vocab = Vocab()
        doc = Doc(vocab, words=["est", "arma", "cepit"], spaces=[True, True, False])
        doc[0]._.remorph = "present"
        # doc[1] has no remorph (None) — should stay None
        doc[2]._.remorph = "perfect"

        cache.put("test.tess", doc)

        loaded = cache.get("test.tess", vocab)
        assert loaded is not None
        assert loaded[0]._.remorph == "present"
        assert loaded[1]._.remorph is None
        assert loaded[2]._.remorph == "perfect"

    def test_no_remorph_extension_no_error(self, cache):
        """Docs without remorph data should round-trip cleanly."""
        vocab = Vocab()
        doc = Doc(vocab, words=["test"], spaces=[False])

        cache.put("no_remorph.tess", doc)
        loaded = cache.get("no_remorph.tess", vocab)
        assert loaded is not None
        assert loaded[0].text == "test"


class TestFileidHash:
    def test_deterministic(self):
        h1 = _fileid_hash("vergil.aen.tess")
        h2 = _fileid_hash("vergil.aen.tess")
        assert h1 == h2

    def test_different_inputs(self):
        h1 = _fileid_hash("file1.tess")
        h2 = _fileid_hash("file2.tess")
        assert h1 != h2

    def test_length(self):
        h = _fileid_hash("any_file.tess")
        assert len(h) == 16
