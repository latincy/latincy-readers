"""Tests for BaseCorpusReader vector integration (build_vectors, find_similar)."""

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from spacy.vocab import Vocab

from latincyreaders.cache.vectors import SentenceVectorConfig, SentenceVectorStore
from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel


class _VectorTestReader(BaseCorpusReader):
    """Minimal reader for testing vector integration.

    Uses NONE annotation level by default to avoid loading a model.
    We override docs() to return pre-built Doc objects with vectors.
    """

    @classmethod
    def _default_file_pattern(cls) -> str:
        return "*.tess"

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        text = path.read_text(encoding=self._encoding)
        yield text, {"filename": path.name}


def _make_vector_reader(tesserae_dir, tmp_path, annotation_level=AnnotationLevel.FULL):
    """Create a test reader pointing at tesserae fixtures."""
    return _VectorTestReader(
        root=tesserae_dir,
        annotation_level=annotation_level,
        model_name="la_core_web_lg",
        cache=False,
    )


class TestBuildVectors:
    """Test BaseCorpusReader.build_vectors()."""

    def test_build_vectors_returns_store(self, tesserae_dir, tmp_path):
        reader = _make_vector_reader(tesserae_dir, tmp_path)
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="test")
        store = reader.build_vectors(config=cfg)

        assert isinstance(store, SentenceVectorStore)
        stats = store.stats()
        assert stats["sentences"] > 0
        assert stats["vector_dim"] > 0

    def test_build_vectors_default_config(self, tesserae_dir, tmp_path):
        reader = _make_vector_reader(tesserae_dir, tmp_path)
        store = reader.build_vectors()
        assert isinstance(store, SentenceVectorStore)

    def test_build_vectors_specific_fileids(self, tesserae_dir, tmp_path):
        reader = _make_vector_reader(tesserae_dir, tmp_path)
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="subset")

        # Build with just one file
        fids = reader.fileids()
        assert len(fids) >= 1
        store = reader.build_vectors(config=cfg, fileids=[fids[0]])

        stats = store.stats()
        assert stats["sentences"] > 0

    def test_build_vectors_persists_to_disk(self, tesserae_dir, tmp_path):
        reader = _make_vector_reader(tesserae_dir, tmp_path)
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="persist")
        store = reader.build_vectors(config=cfg)

        # Verify files exist on disk
        store_dir = tmp_path / "persist"
        assert (store_dir / "vectors.npy").exists()
        assert (store_dir / "index.json").exists()

        # Verify a new store instance can read them
        store2 = SentenceVectorStore(cfg)
        assert store2.stats()["sentences"] == store.stats()["sentences"]


class TestFindSimilar:
    """Test BaseCorpusReader.find_similar()."""

    def test_find_similar_returns_results(self, tesserae_dir, tmp_path):
        reader = _make_vector_reader(tesserae_dir, tmp_path)
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="search")

        # Pre-build the store
        reader.build_vectors(config=cfg)

        results = reader.find_similar("amicitia", config=cfg)
        assert len(results) > 0
        assert "score" in results[0]
        assert "text" in results[0]
        assert "fileid" in results[0]

    def test_find_similar_top_k(self, tesserae_dir, tmp_path):
        reader = _make_vector_reader(tesserae_dir, tmp_path)
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="topk")
        reader.build_vectors(config=cfg)

        results = reader.find_similar("amicitia", top_k=3, config=cfg)
        assert len(results) <= 3

    def test_find_similar_scores_descending(self, tesserae_dir, tmp_path):
        reader = _make_vector_reader(tesserae_dir, tmp_path)
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="order")
        reader.build_vectors(config=cfg)

        results = reader.find_similar("amicitia", top_k=5, config=cfg)
        if len(results) >= 2:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_find_similar_no_nlp_raises(self, tesserae_dir, tmp_path):
        reader = _VectorTestReader(
            root=tesserae_dir,
            annotation_level=AnnotationLevel.NONE,
            cache=False,
        )
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="nolp")

        with pytest.raises(ValueError, match="NLP pipeline"):
            reader.find_similar("test", config=cfg)

    def test_find_similar_auto_build(self, tesserae_dir, tmp_path):
        """find_similar with auto_build=True should build store if missing."""
        reader = _make_vector_reader(tesserae_dir, tmp_path)
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="auto")

        # No pre-build — should auto-build
        results = reader.find_similar("amicitia", config=cfg, auto_build=True)
        assert len(results) > 0

        # Verify store was built
        assert (tmp_path / "auto" / "vectors.npy").exists()

    def test_find_similar_empty_store_no_auto_build_raises(self, tesserae_dir, tmp_path):
        """find_similar without auto_build on empty store raises with guidance."""
        reader = _make_vector_reader(tesserae_dir, tmp_path)
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="empty")

        with pytest.raises(ValueError, match="No vector index found"):
            reader.find_similar("amicitia", config=cfg)
