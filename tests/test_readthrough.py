"""Tests for the read-through cache path and content-hash staleness detection.

Exercises the full lookup chain:
    LRU → DocBin (disk) → .conlluc (canonical) → pipeline
and verifies that upstream corrections to .conlluc files automatically
invalidate the DocBin cache.
"""

import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from latincyreaders.cache.canonical import CanonicalAnnotationStore, CanonicalConfig
from latincyreaders.cache.conlluc import write_conlluc
from latincyreaders.cache.disk import CacheConfig, DiskCache


class TestContentHash:
    """Tests for CanonicalAnnotationStore.content_hash()."""

    @pytest.fixture
    def vocab(self):
        return Vocab()

    @pytest.fixture
    def store(self, tmp_path):
        cfg = CanonicalConfig(
            store_root=tmp_path / "canonical",
            collection="hash-test",
        )
        return CanonicalAnnotationStore(cfg)

    @pytest.fixture
    def sample_doc(self, vocab):
        doc = Doc(
            vocab,
            words=["Gallia", "est", "omnis", "divisa"],
            spaces=[True, True, True, False],
        )
        doc._.fileid = "caesar.gal.tess"
        return doc

    def test_content_hash_returns_none_for_missing(self, store):
        assert store.content_hash("nonexistent") is None

    def test_content_hash_returns_string(self, store, sample_doc):
        store.save("caesar.gal.tess", sample_doc)
        h = store.content_hash("caesar.gal.tess")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_content_hash_deterministic(self, store, sample_doc):
        store.save("caesar.gal.tess", sample_doc)
        h1 = store.content_hash("caesar.gal.tess")
        h2 = store.content_hash("caesar.gal.tess")
        assert h1 == h2

    def test_content_hash_changes_on_correction(self, store, vocab, sample_doc):
        store.save("caesar.gal.tess", sample_doc)
        h_before = store.content_hash("caesar.gal.tess")

        # "Correct" the annotation by saving a different doc
        corrected = Doc(
            vocab,
            words=["Gallia", "est", "omnis", "diuisa"],  # v→u correction
            spaces=[True, True, True, False],
        )
        store.save("caesar.gal.tess", corrected)
        # Must clear cached manifest to see updated file
        store._manifest = None

        h_after = store.content_hash("caesar.gal.tess")
        assert h_before != h_after


class TestDiskCacheStaleness:
    """Tests for source_hash staleness detection in DiskCache."""

    @pytest.fixture
    def vocab(self):
        return Vocab()

    @pytest.fixture
    def cache(self, tmp_path):
        cfg = CacheConfig(
            cache_dir=tmp_path / "cache",
            persist=True,
            collection="stale-test",
        )
        return DiskCache(cfg)

    @pytest.fixture
    def sample_doc(self, vocab):
        return Doc(
            vocab,
            words=["Arma", "virumque", "cano"],
            spaces=[True, True, False],
        )

    def test_get_without_source_hash_ignores_staleness(self, cache, vocab, sample_doc):
        """Backwards compatible: no source_hash → no staleness check."""
        cache.put("test.tess", sample_doc, source_hash="abc123")
        loaded = cache.get("test.tess", vocab)
        assert loaded is not None

    def test_get_with_matching_source_hash(self, cache, vocab, sample_doc):
        cache.put("test.tess", sample_doc, source_hash="abc123")
        loaded = cache.get("test.tess", vocab, source_hash="abc123")
        assert loaded is not None
        assert loaded.text == sample_doc.text

    def test_get_with_mismatched_source_hash_returns_none(self, cache, vocab, sample_doc):
        """Simulates upstream correction: hash changed → cache miss."""
        cache.put("test.tess", sample_doc, source_hash="abc123")
        loaded = cache.get("test.tess", vocab, source_hash="def456")
        assert loaded is None

    def test_get_entry_without_stored_hash_is_stale(self, cache, vocab, sample_doc):
        """Old cache entries without source_hash are stale when hash is requested."""
        cache.put("test.tess", sample_doc)  # no source_hash
        loaded = cache.get("test.tess", vocab, source_hash="abc123")
        assert loaded is None


class TestReadThroughPath:
    """Integration tests for the full read-through cache chain.

    Uses DiskCache + CanonicalAnnotationStore together to verify:
    - Canonical store hit warms disk cache
    - Upstream corrections invalidate disk cache
    """

    @pytest.fixture
    def vocab(self):
        return Vocab()

    @pytest.fixture
    def canonical_store(self, tmp_path):
        cfg = CanonicalConfig(
            store_root=tmp_path / "canonical",
            collection="readthrough-test",
        )
        return CanonicalAnnotationStore(cfg)

    @pytest.fixture
    def disk_cache(self, tmp_path):
        cfg = CacheConfig(
            cache_dir=tmp_path / "cache",
            persist=True,
            collection="readthrough-test",
        )
        return DiskCache(cfg)

    @pytest.fixture
    def sample_doc(self, vocab):
        doc = Doc(
            vocab,
            words=["Arma", "virumque", "cano"],
            spaces=[True, True, False],
        )
        doc._.fileid = "vergil.aen.tess"
        return doc

    def test_canonical_hit_warms_disk_cache(
        self, canonical_store, disk_cache, vocab, sample_doc,
    ):
        """When canonical has data, reading it should allow warming disk cache."""
        fileid = "vergil.aen.tess"

        # Save to canonical store
        canonical_store.save(fileid, sample_doc, model_name="la_core_web_lg")
        source_hash = canonical_store.content_hash(fileid)

        # Disk cache is empty
        assert disk_cache.get(fileid, vocab) is None

        # Load from canonical
        doc = canonical_store.load(fileid, vocab)
        assert doc is not None

        # Warm disk cache with source_hash
        disk_cache.put(fileid, doc, source_hash=source_hash)

        # Now disk cache should hit with same hash
        cached = disk_cache.get(fileid, vocab, source_hash=source_hash)
        assert cached is not None
        assert cached.text == sample_doc.text

    def test_upstream_correction_invalidates_disk_cache(
        self, canonical_store, disk_cache, vocab, sample_doc,
    ):
        """Simulates: save → cache → correct upstream → cache stale."""
        fileid = "vergil.aen.tess"

        # 1. Save original to canonical
        canonical_store.save(fileid, sample_doc, model_name="la_core_web_lg")
        original_hash = canonical_store.content_hash(fileid)

        # 2. Load and warm disk cache
        doc = canonical_store.load(fileid, vocab)
        disk_cache.put(fileid, doc, source_hash=original_hash)

        # 3. Verify disk cache works
        assert disk_cache.get(fileid, vocab, source_hash=original_hash) is not None

        # 4. Simulate upstream correction (someone edits the .conlluc)
        corrected = Doc(
            vocab,
            words=["Arma", "uirumque", "cano"],  # v→u
            spaces=[True, True, False],
        )
        corrected._.fileid = fileid
        canonical_store.save(fileid, corrected, model_name="la_core_web_lg")
        canonical_store._manifest = None  # force manifest reload
        new_hash = canonical_store.content_hash(fileid)

        # 5. Hash changed
        assert original_hash != new_hash

        # 6. Disk cache is now stale
        assert disk_cache.get(fileid, vocab, source_hash=new_hash) is None

        # 7. Re-read from canonical and re-warm
        fresh = canonical_store.load(fileid, vocab)
        disk_cache.put(fileid, fresh, source_hash=new_hash)
        assert disk_cache.get(fileid, vocab, source_hash=new_hash) is not None
        assert disk_cache.get(fileid, vocab, source_hash=new_hash).text == "Arma uirumque cano"
