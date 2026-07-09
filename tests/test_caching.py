"""Tests for document caching."""

import pytest
from unittest.mock import patch, MagicMock

from latincyreaders import TesseraeReader, AnnotationLevel


class TestDocumentCaching:
    """Tests for the document caching feature."""

    @pytest.fixture
    def reader_with_cache(self, tesserae_dir):
        """Create a TesseraeReader with caching enabled."""
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.MINIMAL,
            cache=True,
        )

    @pytest.fixture
    def reader_without_cache(self, tesserae_dir):
        """Create a TesseraeReader with caching disabled."""
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.MINIMAL,
            cache=False,
        )

    def test_cache_enabled_by_default(self, tesserae_dir):
        """Caching is enabled by default."""
        reader = TesseraeReader(
            root=tesserae_dir,
            annotation_level=AnnotationLevel.MINIMAL,
        )
        assert reader.cache_enabled is True

    def test_cache_can_be_disabled(self, reader_without_cache):
        """Caching can be disabled via parameter."""
        assert reader_without_cache.cache_enabled is False

    def test_cached_docs_are_same_object(self, reader_with_cache):
        """Accessing same doc twice returns cached object."""
        fileid = reader_with_cache.fileids()[0]

        doc1 = next(reader_with_cache.docs(fileid))
        doc2 = next(reader_with_cache.docs(fileid))

        # Should be the exact same object (cached)
        assert doc1 is doc2

    def test_uncached_docs_are_different_objects(self, reader_without_cache):
        """Without caching, each call creates new doc."""
        fileid = reader_without_cache.fileids()[0]

        doc1 = next(reader_without_cache.docs(fileid))
        doc2 = next(reader_without_cache.docs(fileid))

        # Should be different objects (not cached)
        assert doc1 is not doc2

    def test_cache_stats(self, reader_with_cache):
        """cache_stats() returns hit/miss information."""
        fileid = reader_with_cache.fileids()[0]

        # Initial state
        stats = reader_with_cache.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

        # First access (miss)
        _ = next(reader_with_cache.docs(fileid))
        stats = reader_with_cache.cache_stats()
        assert stats["misses"] == 1
        assert stats["size"] == 1

        # Second access (hit)
        _ = next(reader_with_cache.docs(fileid))
        stats = reader_with_cache.cache_stats()
        assert stats["hits"] == 1

    def test_clear_cache(self, reader_with_cache):
        """clear_cache() empties the cache."""
        fileid = reader_with_cache.fileids()[0]

        # Populate cache
        _ = next(reader_with_cache.docs(fileid))
        assert reader_with_cache.cache_stats()["size"] == 1

        # Clear
        reader_with_cache.clear_cache()
        assert reader_with_cache.cache_stats()["size"] == 0

    def test_cache_respects_fileids(self, reader_with_cache):
        """Different fileids are cached separately."""
        fileids = reader_with_cache.fileids()
        if len(fileids) < 2:
            pytest.skip("Need at least 2 files for this test")

        doc1 = next(reader_with_cache.docs(fileids[0]))
        doc2 = next(reader_with_cache.docs(fileids[1]))

        assert doc1 is not doc2
        assert reader_with_cache.cache_stats()["size"] == 2

    def test_cache_with_file_selector(self, reader_with_cache):
        """Caching works with FileSelector."""
        selection = reader_with_cache.select().match("tesserae")

        # First iteration
        docs1 = list(reader_with_cache.docs(selection))

        # Second iteration should hit cache
        docs2 = list(reader_with_cache.docs(selection))

        assert len(docs1) == len(docs2)
        for d1, d2 in zip(docs1, docs2):
            assert d1 is d2  # Same cached objects


class TestCacheMaxSize:
    """Tests for cache size limits."""

    @pytest.fixture
    def reader_small_cache(self, tesserae_dir):
        """Create reader with small cache."""
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.MINIMAL,
            cache=True,
            cache_maxsize=2,
        )

    def test_cache_maxsize_parameter(self, tesserae_dir):
        """cache_maxsize limits cache size."""
        reader = TesseraeReader(
            root=tesserae_dir,
            annotation_level=AnnotationLevel.MINIMAL,
            cache=True,
            cache_maxsize=5,
        )
        assert reader._cache_maxsize == 5

    def test_cache_evicts_oldest(self, reader_small_cache):
        """Cache evicts oldest entries when full."""
        fileids = reader_small_cache.fileids()
        assert len(fileids) >= 3, "Need 3 test files for eviction test"

        # Access first two files - cache size is 2
        _ = next(reader_small_cache.docs(fileids[0]))
        _ = next(reader_small_cache.docs(fileids[1]))
        assert reader_small_cache.cache_stats()["size"] == 2

        # Access third file - should evict first
        _ = next(reader_small_cache.docs(fileids[2]))
        assert reader_small_cache.cache_stats()["size"] == 2  # Still at max

        # First file should be evicted, accessing it should be a miss
        misses_before = reader_small_cache.cache_stats()["misses"]
        _ = next(reader_small_cache.docs(fileids[0]))
        misses_after = reader_small_cache.cache_stats()["misses"]
        assert misses_after > misses_before  # Re-loaded from disk


class TestCacheIntegration:
    """Integration tests for caching with other features."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        """Create reader with cache."""
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.BASIC,
            cache=True,
        )

    def test_sents_uses_cached_docs(self, reader):
        """sents() benefits from cached docs."""
        fileid = reader.fileids()[0]

        # Access via docs first
        doc = next(reader.docs(fileid))
        initial_misses = reader.cache_stats()["misses"]

        # Access via sents - should use cached doc
        sents = list(reader.sents(fileid))
        assert len(sents) > 0

        # Should have hit cache, not created new miss
        assert reader.cache_stats()["misses"] == initial_misses

    def test_tokens_uses_cached_docs(self, reader):
        """tokens() benefits from cached docs."""
        fileid = reader.fileids()[0]

        # Access via docs first
        _ = next(reader.docs(fileid))
        initial_misses = reader.cache_stats()["misses"]

        # Access via tokens - should use cached doc
        tokens = list(reader.tokens(fileid))
        assert len(tokens) > 0

        # Should have hit cache
        assert reader.cache_stats()["misses"] == initial_misses

    def test_concordance_uses_cached_docs(self, reader):
        """concordance() benefits from cached docs."""
        fileid = reader.fileids()[0]

        # First concordance call
        conc1 = reader.concordance(fileids=fileid)

        # Second call should use cache
        hits_before = reader.cache_stats()["hits"]
        conc2 = reader.concordance(fileids=fileid)
        hits_after = reader.cache_stats()["hits"]

        assert hits_after > hits_before
