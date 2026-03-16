"""Sentence-level vector store for similarity search.

Stores sentence embeddings as memory-mapped NumPy arrays for efficient
similarity search across large collections. Vectors are derived from
spaCy model vectors (mean of token vectors).

Requires the ``vectors`` optional dependency::

    pip install latincyreaders[vectors]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Doc

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# Default vector store root
_DEFAULT_STORE_ROOT = Path.home() / "latincy_data" / "vectors"


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise ImportError(
            "numpy is required for SentenceVectorStore. "
            "Install it with: pip install latincyreaders[vectors]"
        )


@dataclass
class SentenceVectorConfig:
    """Configuration for sentence vector storage.

    Attributes:
        store_root: Root directory for vector stores.
        collection: Collection name for organising vectors.
        vector_source: Source of vectors (currently only ``"spacy"``).
    """

    store_root: Path = field(default_factory=lambda: _DEFAULT_STORE_ROOT)
    collection: str = "default"
    vector_source: str = "spacy"

    def __post_init__(self) -> None:
        self.store_root = Path(self.store_root)


class SentenceVectorStore:
    """Efficient array-based sentence vector store for similarity search.

    Stores sentence vectors as memory-mapped ``.npy`` files alongside a
    JSON index that maps each row to its source (fileid, sentence index,
    citation, text preview).

    Similarity search uses cosine distance computed with NumPy — no heavy
    external dependencies required.

    Example::

        >>> from latincyreaders.cache.vectors import SentenceVectorStore, SentenceVectorConfig
        >>> cfg = SentenceVectorConfig(collection="tesserae")
        >>> store = SentenceVectorStore(cfg)
        >>> store.build(reader)
        >>> results = store.similar_to_sent("arma virumque cano", nlp)
        >>> for r in results:
        ...     print(f"{r['citation']}: {r['text'][:60]}  (sim={r['score']:.3f})")
    """

    def __init__(self, config: SentenceVectorConfig) -> None:
        _require_numpy()
        self._config = config
        self._dir = config.store_root / config.collection
        self._vectors_path = self._dir / "vectors.npy"
        self._index_path = self._dir / "index.json"
        self._vectors: Any | None = None  # np.ndarray, lazy
        self._index: list[dict[str, Any]] | None = None

    @property
    def _np(self) -> Any:
        """Lazy numpy import."""
        _require_numpy()
        return np

    # ------------------------------------------------------------------
    # Building the store
    # ------------------------------------------------------------------

    def build(
        self,
        reader: Any,
        fileids: list[str] | None = None,
        batch_size: int | None = None,
    ) -> int:
        """Compute and store sentence vectors from a reader.

        Iterates over all docs, computes mean-of-token vectors for each
        sentence, and writes them to disk.

        Args:
            reader: Corpus reader to index.
            fileids: Files to include, or None for all.
            batch_size: If set, flush vectors to disk every N files.
                Provides crash resilience and lower peak memory for
                large corpora. If None, writes once at the end.

        Returns the number of sentences indexed.
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        # Clear any existing store for a fresh build
        self.clear()

        fids = fileids if fileids is not None else reader.fileids()
        total_count = 0
        batch_vecs: list[Any] = []
        batch_index: list[dict[str, Any]] = []
        files_in_batch = 0

        for doc in reader.docs(fids):
            fileid = doc._.fileid or "unknown"
            for sent_idx, sent in enumerate(doc.sents):
                vec = sent.vector
                if vec is not None and self._np.any(vec):
                    batch_vecs.append(vec)
                    batch_index.append({
                        "fileid": fileid,
                        "sent_idx": sent_idx,
                        "citation": self._get_citation(doc, sent),
                        "text": sent.text[:200],
                    })

            files_in_batch += 1

            if batch_size is not None and files_in_batch >= batch_size:
                total_count += self._flush_batch(batch_vecs, batch_index)
                batch_vecs = []
                batch_index = []
                files_in_batch = 0

        # Flush remaining
        if batch_vecs:
            total_count += self._flush_batch(batch_vecs, batch_index)

        return total_count

    def _flush_batch(
        self,
        vectors_list: list[Any],
        index_list: list[dict[str, Any]],
    ) -> int:
        """Append a batch of vectors and index entries to disk."""
        if not vectors_list:
            return 0

        new_vecs = self._np.stack(vectors_list).astype(self._np.float32)

        # Append to existing on disk
        existing_vecs = self._load_vectors()
        existing_index = self._load_index()

        if existing_vecs is not None and len(existing_vecs) > 0:
            combined = self._np.concatenate([existing_vecs, new_vecs], axis=0)
        else:
            combined = new_vecs

        self._np.save(str(self._vectors_path), combined)

        full_index = existing_index + index_list
        self._index_path.write_text(
            json.dumps(full_index, ensure_ascii=False),
            encoding="utf-8",
        )

        # Reset lazy cache so next _load picks up new data
        self._vectors = None
        self._index = None

        return len(index_list)

    @staticmethod
    def _get_citation(doc: "Doc", sent: Any) -> str:
        """Extract citation string for a sentence span."""
        if hasattr(sent._, "citation") and sent._.citation:
            return sent._.citation
        for span_key in doc.spans:
            for span in doc.spans[span_key]:
                if span.start <= sent.start < span.end:
                    c = getattr(span._, "citation", None)
                    if c:
                        return c
        return ""

    def add_doc(self, doc: "Doc") -> int:
        """Add vectors for a single doc's sentences to the store.

        If the store already has data, new vectors are appended.
        Returns the number of sentences added.
        """
        vectors_list: list[Any] = []
        index_list: list[dict[str, Any]] = []

        fileid = doc._.fileid or "unknown"
        for sent_idx, sent in enumerate(doc.sents):
            vec = sent.vector
            if vec is not None and self._np.any(vec):
                vectors_list.append(vec)
                citation = getattr(sent._, "citation", None) or ""
                index_list.append({
                    "fileid": fileid,
                    "sent_idx": sent_idx,
                    "citation": citation,
                    "text": sent.text[:200],
                })

        if not vectors_list:
            return 0

        self._dir.mkdir(parents=True, exist_ok=True)
        new_vecs = self._np.stack(vectors_list).astype(self._np.float32)

        # Append to existing
        existing_vecs = self._load_vectors()
        existing_index = self._load_index()

        if existing_vecs is not None and len(existing_vecs) > 0:
            combined = self._np.concatenate([existing_vecs, new_vecs], axis=0)
        else:
            combined = new_vecs

        self._np.save(str(self._vectors_path), combined)

        full_index = existing_index + index_list
        self._index_path.write_text(
            json.dumps(full_index, ensure_ascii=False),
            encoding="utf-8",
        )

        self._vectors = None
        self._index = None

        return len(index_list)

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def similar(
        self,
        query_vector: Any,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find the most similar sentences by cosine similarity.

        Args:
            query_vector: 1-D array (same dim as stored vectors).
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: fileid, sent_idx, citation, text, score.
        """
        vectors = self._load_vectors()
        index = self._load_index()
        if vectors is None or len(vectors) == 0:
            return []

        query = self._np.asarray(query_vector, dtype=self._np.float32)

        # Normalise
        q_norm = self._np.linalg.norm(query)
        if q_norm == 0:
            return []
        query_normed = query / q_norm

        v_norms = self._np.linalg.norm(vectors, axis=1, keepdims=True)
        v_norms = self._np.where(v_norms == 0, 1, v_norms)
        vectors_normed = vectors / v_norms

        scores = vectors_normed @ query_normed
        top_indices = self._np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            entry = index[idx].copy()
            entry["score"] = float(scores[idx])
            results.append(entry)

        return results

    def similar_to_sent(
        self,
        text: str,
        nlp: Any,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find sentences similar to a query text.

        Vectorises the query through the provided spaCy pipeline,
        then searches the store.

        Args:
            text: Query text.
            nlp: A spaCy Language pipeline (must include vectors).
            top_k: Number of results.
        """
        doc = nlp(text)
        return self.similar(doc.vector, top_k=top_k)

    def similar_to_doc_sent(
        self,
        fileid: str,
        sent_idx: int,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find sentences similar to an already-indexed sentence.

        Args:
            fileid: File identifier of the source sentence.
            sent_idx: Sentence index within that document.
            top_k: Number of results.
        """
        index = self._load_index()
        vectors = self._load_vectors()
        if vectors is None or len(vectors) == 0:
            return []

        # Find the row
        for row_idx, entry in enumerate(index):
            if entry["fileid"] == fileid and entry["sent_idx"] == sent_idx:
                return self.similar(vectors[row_idx], top_k=top_k)

        return []

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        index = self._load_index()
        vectors = self._load_vectors()
        total_bytes = 0
        if self._vectors_path.exists():
            total_bytes += self._vectors_path.stat().st_size
        if self._index_path.exists():
            total_bytes += self._index_path.stat().st_size

        return {
            "collection": self._config.collection,
            "sentences": len(index),
            "vector_dim": int(vectors.shape[1]) if vectors is not None and len(vectors) > 0 else 0,
            "size_bytes": total_bytes,
            "store_dir": str(self._dir),
        }

    def clear(self) -> None:
        """Remove all stored vectors and index."""
        self._vectors_path.unlink(missing_ok=True)
        self._index_path.unlink(missing_ok=True)
        self._vectors = None
        self._index = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_vectors(self) -> Any | None:
        """Load vectors array (memory-mapped for efficiency)."""
        if self._vectors is not None:
            return self._vectors

        if not self._vectors_path.exists():
            return None

        self._vectors = self._np.load(str(self._vectors_path), mmap_mode="r")
        return self._vectors

    def _load_index(self) -> list[dict[str, Any]]:
        if self._index is not None:
            return self._index

        if not self._index_path.exists():
            self._index = []
            return self._index

        try:
            self._index = json.loads(
                self._index_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            self._index = []

        return self._index
