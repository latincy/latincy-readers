"""Tests for the annotation-model generator stamp.

A DocBin cache built under one model version must never be served after the
model is upgraded (the silent-stale failure). These tests cover:

- ``DiskCache`` generator staleness (stamp-keyed ephemeral cache).
- ``CanonicalAnnotationStore`` emitting ``AnnotationModelMismatchWarning`` when
  the stored model differs from the active one.
- The user's exact scenario: la_core_web_lg 3.9.4 → 3.9.6.
"""

import warnings

import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from latincyreaders import AnnotationLevel, TesseraeReader
from latincyreaders.cache.canonical import CanonicalAnnotationStore, CanonicalConfig
from latincyreaders.cache.disk import CacheConfig, DiskCache
from latincyreaders.warnings import AnnotationModelMismatchWarning


@pytest.fixture
def vocab():
    return Vocab()


@pytest.fixture
def sample_doc(vocab):
    doc = Doc(
        vocab,
        words=["Arma", "virumque", "cano"],
        spaces=[True, True, False],
    )
    doc._.fileid = "vergil.aen.tess"
    return doc


class TestDiskCacheGeneratorStaleness:
    """DiskCache invalidates on generator (model version) mismatch."""

    @pytest.fixture
    def cache(self, tmp_path):
        cfg = CacheConfig(
            cache_dir=tmp_path / "cache",
            persist=True,
            collection="gen-test",
        )
        return DiskCache(cfg)

    def test_matching_generator_hits(self, cache, vocab, sample_doc):
        cache.put("f.tess", sample_doc, generator="la_core_web_lg@3.9.4")
        loaded = cache.get("f.tess", vocab, generator="la_core_web_lg@3.9.4")
        assert loaded is not None
        assert loaded.text == sample_doc.text

    def test_mismatched_generator_misses(self, cache, vocab, sample_doc):
        """The core fix: a model upgrade is a miss, not a silent-stale read."""
        cache.put("f.tess", sample_doc, generator="la_core_web_lg@3.9.4")
        assert cache.get("f.tess", vocab, generator="la_core_web_lg@3.9.6") is None

    def test_no_generator_arg_ignores_stamp(self, cache, vocab, sample_doc):
        """Backwards compatible: no generator requested → no stamp check."""
        cache.put("f.tess", sample_doc, generator="la_core_web_lg@3.9.4")
        assert cache.get("f.tess", vocab) is not None

    def test_legacy_entry_without_stamp_is_stale(self, cache, vocab, sample_doc):
        """A pre-fix entry (no stored generator) is rebuilt once when a
        generator is requested."""
        cache.put("f.tess", sample_doc)  # no generator stamp
        assert cache.get("f.tess", vocab, generator="la_core_web_lg@3.9.6") is None

    def test_stamp_and_source_hash_compose(self, cache, vocab, sample_doc):
        cache.put(
            "f.tess", sample_doc,
            generator="la_core_web_lg@3.9.4", source_hash="abc123",
        )
        # Matching both → hit
        assert cache.get(
            "f.tess", vocab, generator="la_core_web_lg@3.9.4", source_hash="abc123",
        ) is not None
        # Generator matches but content changed → miss
        assert cache.get(
            "f.tess", vocab, generator="la_core_web_lg@3.9.4", source_hash="def456",
        ) is None


class TestCanonicalMismatchWarning:
    """CanonicalAnnotationStore warns loudly when the base model changed."""

    @pytest.fixture
    def store(self, tmp_path):
        cfg = CanonicalConfig(
            store_root=tmp_path / "canonical",
            collection="warn-test",
        )
        return CanonicalAnnotationStore(cfg)

    def test_matching_generator_no_warning(self, store, vocab, sample_doc):
        store.save(
            "vergil.aen.tess", sample_doc,
            model_name="la_core_web_lg", model_version="3.9.6",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", AnnotationModelMismatchWarning)
            doc = store.load(
                "vergil.aen.tess", vocab,
                expected_generator=("la_core_web_lg", "3.9.6"),
            )
        assert doc is not None

    def test_mismatched_generator_warns(self, store, vocab, sample_doc):
        """The user's exact scenario: base built under 3.9.4, active is 3.9.6."""
        store.save(
            "vergil.aen.tess", sample_doc,
            model_name="la_core_web_lg", model_version="3.9.4",
        )
        with pytest.warns(AnnotationModelMismatchWarning, match="3.9.4"):
            store.load(
                "vergil.aen.tess", vocab,
                expected_generator=("la_core_web_lg", "3.9.6"),
            )

    def test_warns_once_per_fileid(self, store, vocab, sample_doc):
        store.save(
            "vergil.aen.tess", sample_doc,
            model_name="la_core_web_lg", model_version="3.9.4",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            store.load("vergil.aen.tess", vocab, expected_generator=("la_core_web_lg", "3.9.6"))
            store.load("vergil.aen.tess", vocab, expected_generator=("la_core_web_lg", "3.9.6"))
        mismatch = [w for w in caught if issubclass(w.category, AnnotationModelMismatchWarning)]
        assert len(mismatch) == 1

    def test_no_expected_generator_no_warning(self, store, vocab, sample_doc):
        """Callers that don't pass expected_generator keep old behaviour."""
        store.save(
            "vergil.aen.tess", sample_doc,
            model_name="la_core_web_lg", model_version="3.9.4",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", AnnotationModelMismatchWarning)
            assert store.load("vergil.aen.tess", vocab) is not None

    def test_unknown_active_version_no_warning(self, store, vocab, sample_doc):
        """When the active version is unknown, there is nothing reliable to
        compare against, so we stay quiet rather than cry wolf."""
        store.save(
            "vergil.aen.tess", sample_doc,
            model_name="la_core_web_lg", model_version="3.9.4",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", AnnotationModelMismatchWarning)
            assert store.load(
                "vergil.aen.tess", vocab,
                expected_generator=("la_core_web_lg", "unknown"),
            ) is not None


class TestTesseraeReaderDiskInvalidation:
    """End-to-end through TesseraeReader.docs() — the previously-blind path.

    Before the fix, TesseraeReader's DocBin path passed no staleness key at all,
    so a model upgrade served the stale blob. It must now invalidate and rebuild.
    """

    def _reader(self, tesserae_dir, tmp_path):
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.MINIMAL,  # blank pipeline, fast
            cache_config=CacheConfig(
                cache_dir=tmp_path / "cache", persist=True, collection="tess-gen",
            ),
        )

    def test_disk_cache_rebuilds_on_model_upgrade(
        self, tesserae_dir, tmp_path, monkeypatch,
    ):
        reader = self._reader(tesserae_dir, tmp_path)
        fileid = reader.fileids()[0]

        # 1. Build the cache under model version 3.9.4.
        monkeypatch.setattr(
            reader, "_active_generator", lambda: ("la_core_web_lg", "3.9.4"),
        )
        list(reader.docs(fileid))
        entry = reader._disk_cache._load_manifest()[fileid]
        assert entry["generator"] == "la_core_web_lg@3.9.4"

        # 2. Upgrade the model to 3.9.6 and drop the in-memory LRU so the read
        #    falls through to the disk cache (the layer that used to go stale).
        reader.clear_cache()
        reader._disk_cache._manifest = None  # force manifest reload
        monkeypatch.setattr(
            reader, "_active_generator", lambda: ("la_core_web_lg", "3.9.6"),
        )
        list(reader.docs(fileid))

        # 3. The stale 3.9.4 blob was not served — the entry was rebuilt and
        #    now carries the active generator stamp.
        reader._disk_cache._manifest = None
        entry = reader._disk_cache._load_manifest()[fileid]
        assert entry["generator"] == "la_core_web_lg@3.9.6"
