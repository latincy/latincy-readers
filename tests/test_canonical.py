"""Tests for canonical annotation store."""

import json

import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from latincyreaders.cache.canonical import CanonicalAnnotationStore, CanonicalConfig


class TestCanonicalAnnotationStore:
    """Tests for CRUD operations on canonical annotations."""

    @pytest.fixture
    def vocab(self):
        return Vocab()

    @pytest.fixture
    def store(self, tmp_path):
        cfg = CanonicalConfig(
            store_root=tmp_path / "canonical",
            collection="test-collection",
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

    def test_save_and_load(self, store, vocab, sample_doc):
        store.save("caesar.gal.tess", sample_doc)
        loaded = store.load("caesar.gal.tess", vocab)

        assert loaded is not None
        assert loaded.text == sample_doc.text
        assert len(loaded) == len(sample_doc)

    def test_has(self, store, sample_doc):
        assert store.has("caesar.gal.tess") is False
        store.save("caesar.gal.tess", sample_doc)
        assert store.has("caesar.gal.tess") is True

    def test_remove(self, store, vocab, sample_doc):
        store.save("caesar.gal.tess", sample_doc)
        store.remove("caesar.gal.tess")
        assert store.has("caesar.gal.tess") is False
        assert store.load("caesar.gal.tess", vocab) is None

    def test_fileids(self, store, sample_doc):
        assert store.fileids() == []
        store.save("file1.tess", sample_doc)
        store.save("file2.tess", sample_doc)
        assert sorted(store.fileids()) == ["file1.tess", "file2.tess"]

    def test_stats(self, store, sample_doc):
        stats = store.stats()
        assert stats["collection"] == "test-collection"
        assert stats["entries"] == 0

        store.save("file.tess", sample_doc)
        stats = store.stats()
        assert stats["entries"] == 1
        assert stats["size_bytes"] > 0

    def test_extra_metadata(self, store, sample_doc):
        store.save("file.tess", sample_doc, model_name="la_core_web_lg")
        manifest = store._load_manifest()
        entry = manifest["files"]["file.tess"]
        assert entry["model_name"] == "la_core_web_lg"


class TestCanonicalDiff:
    """Tests for comparing canonical vs dynamic annotations."""

    @pytest.fixture
    def vocab(self):
        return Vocab()

    @pytest.fixture
    def store(self, tmp_path):
        cfg = CanonicalConfig(
            store_root=tmp_path / "canonical",
            collection="diff-test",
        )
        return CanonicalAnnotationStore(cfg)

    def test_diff_identical_docs(self, store, vocab):
        doc = Doc(vocab, words=["arma", "virumque"], spaces=[True, False])
        store.save("test.tess", doc)

        diffs = store.diff("test.tess", doc)
        assert diffs == []

    def test_diff_different_lemmas(self, store, vocab):
        # Canonical doc
        canonical = Doc(vocab, words=["arma", "virum"], spaces=[True, False])
        canonical[0].lemma_ = "armum"
        canonical[1].lemma_ = "vir"
        store.save("test.tess", canonical)

        # Dynamic doc with different lemma
        dynamic = Doc(vocab, words=["arma", "virum"], spaces=[True, False])
        dynamic[0].lemma_ = "arma"
        dynamic[1].lemma_ = "vir"

        diffs = store.diff("test.tess", dynamic)
        assert len(diffs) == 1
        assert diffs[0]["index"] == 0
        assert diffs[0]["lemma"]["canonical"] == "armum"
        assert diffs[0]["lemma"]["dynamic"] == "arma"

    def test_diff_missing_canonical(self, store, vocab):
        doc = Doc(vocab, words=["test"], spaces=[False])
        diffs = store.diff("nonexistent.tess", doc)
        assert len(diffs) == 1
        assert "error" in diffs[0]

    def test_diff_length_mismatch(self, store, vocab):
        short = Doc(vocab, words=["arma"], spaces=[False])
        long = Doc(vocab, words=["arma", "virumque"], spaces=[True, False])
        store.save("test.tess", short)

        diffs = store.diff("test.tess", long)
        assert any("length_mismatch" in d for d in diffs)


class TestCanonicalExportImport:
    """Tests for exporting and importing collections."""

    @pytest.fixture
    def vocab(self):
        return Vocab()

    def test_export_import_round_trip(self, tmp_path, vocab):
        # Create and populate source store
        src_cfg = CanonicalConfig(
            store_root=tmp_path / "src",
            collection="export-test",
        )
        src_store = CanonicalAnnotationStore(src_cfg)

        doc = Doc(vocab, words=["Gallia", "est"], spaces=[True, False])
        src_store.save("file1.tess", doc)
        src_store.save("file2.tess", doc)

        # Export
        export_dir = tmp_path / "exported"
        src_store.export_collection(export_dir)

        # Import into new store
        dst_cfg = CanonicalConfig(
            store_root=tmp_path / "dst",
            collection="imported",
        )
        dst_store = CanonicalAnnotationStore(dst_cfg)
        count = dst_store.import_collection(export_dir)

        assert count == 2
        assert sorted(dst_store.fileids()) == ["file1.tess", "file2.tess"]

        loaded = dst_store.load("file1.tess", vocab)
        assert loaded is not None
        assert loaded.text == doc.text

    def test_export_existing_dir_raises(self, tmp_path, vocab):
        cfg = CanonicalConfig(store_root=tmp_path / "src", collection="test")
        store = CanonicalAnnotationStore(cfg)
        doc = Doc(vocab, words=["test"], spaces=[False])
        store.save("file.tess", doc)

        existing = tmp_path / "existing"
        existing.mkdir()

        with pytest.raises(FileExistsError):
            store.export_collection(existing)

    def test_import_missing_manifest_raises(self, tmp_path):
        cfg = CanonicalConfig(store_root=tmp_path / "dst", collection="test")
        store = CanonicalAnnotationStore(cfg)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            store.import_collection(empty_dir)
