"""Tests for sentence vector store."""

import json

import numpy as np
import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from latincyreaders.cache.vectors import SentenceVectorConfig, SentenceVectorStore


class TestSentenceVectorConfig:
    def test_defaults(self):
        cfg = SentenceVectorConfig()
        assert cfg.collection == "default"
        assert cfg.vector_source == "spacy"

    def test_custom(self, tmp_path):
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="tesserae")
        assert cfg.collection == "tesserae"
        assert cfg.store_root == tmp_path


class TestSentenceVectorStore:
    """Tests for building and querying the vector store."""

    @pytest.fixture
    def vocab(self):
        v = Vocab()
        # Add vectors to vocab so docs have them
        v.reset_vectors(width=8)
        return v

    @pytest.fixture
    def store(self, tmp_path):
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="test")
        return SentenceVectorStore(cfg)

    def _make_doc_with_vectors(self, vocab, words, vector_values):
        """Create a doc with artificial sentence vectors."""
        # Set word vectors via vocab so tokens pick them up
        vec = np.array(vector_values, dtype=np.float32)
        for word in words:
            vocab.set_vector(word, vec)

        # Mark first token as sentence start
        sent_starts = [True] + [False] * (len(words) - 1)
        doc = Doc(vocab, words=words, spaces=[True] * (len(words) - 1) + [False], sent_starts=sent_starts)
        doc._.fileid = "test.tess"
        return doc

    def test_add_doc_and_stats(self, store, vocab):
        doc = self._make_doc_with_vectors(vocab, ["arma", "virumque"], [1.0] * 8)
        count = store.add_doc(doc)
        assert count > 0

        stats = store.stats()
        assert stats["sentences"] > 0
        assert stats["vector_dim"] == 8

    def test_similar_basic(self, store, vocab):
        # Add a doc
        doc = self._make_doc_with_vectors(vocab, ["arma", "virumque"], [1.0] * 8)
        store.add_doc(doc)

        # Query with same vector
        query = np.ones(8, dtype=np.float32)
        results = store.similar(query, top_k=5)
        assert len(results) > 0
        assert results[0]["score"] > 0.99  # Should be very similar

    def test_similar_ranking(self, tmp_path):
        # Use separate vocabs to give different vector values to each doc
        cfg = SentenceVectorConfig(store_root=tmp_path / "ranking", collection="test")
        store = SentenceVectorStore(cfg)

        # Doc1 with vectors pointing in dim 0
        vocab1 = Vocab()
        vocab1.reset_vectors(width=8)
        vec1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vocab1.set_vector("arma", vec1)
        vocab1.set_vector("virumque", vec1)
        doc1 = Doc(vocab1, words=["arma", "virumque"], spaces=[True, False], sent_starts=[True, False])
        doc1._.fileid = "doc1.tess"
        store.add_doc(doc1)

        # Doc2 with vectors pointing in dim 1
        vocab2 = Vocab()
        vocab2.reset_vectors(width=8)
        vec2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vocab2.set_vector("Gallia", vec2)
        vocab2.set_vector("est", vec2)
        doc2 = Doc(vocab2, words=["Gallia", "est"], spaces=[True, False], sent_starts=[True, False])
        doc2._.fileid = "doc2.tess"
        store.add_doc(doc2)

        # Query closer to doc1
        query = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.similar(query, top_k=2)

        assert len(results) == 2
        assert results[0]["fileid"] == "doc1.tess"

    def test_similar_to_doc_sent(self, store, vocab):
        doc = self._make_doc_with_vectors(vocab, ["arma", "virumque"], [1.0] * 8)
        store.add_doc(doc)

        results = store.similar_to_doc_sent("test.tess", 0, top_k=5)
        assert len(results) > 0

    def test_similar_to_doc_sent_missing(self, store):
        results = store.similar_to_doc_sent("nonexistent.tess", 0)
        assert results == []

    def test_clear(self, store, vocab):
        doc = self._make_doc_with_vectors(vocab, ["test"], [1.0] * 8)
        store.add_doc(doc)
        assert store.stats()["sentences"] > 0

        store.clear()
        assert store.stats()["sentences"] == 0

    def test_empty_query(self, store):
        """Querying empty store returns empty list."""
        query = np.zeros(8, dtype=np.float32)
        results = store.similar(query, top_k=5)
        assert results == []

    def test_zero_vector_query(self, store, vocab):
        """Zero-magnitude query returns empty (can't normalise)."""
        doc = self._make_doc_with_vectors(vocab, ["test"], [1.0] * 8)
        store.add_doc(doc)

        query = np.zeros(8, dtype=np.float32)
        results = store.similar(query, top_k=5)
        assert results == []

    def test_add_doc_appends(self, store, vocab):
        """Adding multiple docs accumulates vectors."""
        doc1 = self._make_doc_with_vectors(vocab, ["arma"], [1.0] * 8)
        doc1._.fileid = "doc1.tess"
        store.add_doc(doc1)

        doc2 = self._make_doc_with_vectors(vocab, ["Gallia"], [0.5] * 8)
        doc2._.fileid = "doc2.tess"
        store.add_doc(doc2)

        stats = store.stats()
        assert stats["sentences"] == 2

    def test_persistence(self, tmp_path, vocab):
        """Vectors persist across store instances."""
        cfg = SentenceVectorConfig(store_root=tmp_path, collection="persist")

        store1 = SentenceVectorStore(cfg)
        doc = self._make_doc_with_vectors(vocab, ["test"], [1.0] * 8)
        store1.add_doc(doc)

        # New store instance reads from disk
        store2 = SentenceVectorStore(cfg)
        assert store2.stats()["sentences"] == 1

        query = np.ones(8, dtype=np.float32)
        results = store2.similar(query, top_k=5)
        assert len(results) == 1
