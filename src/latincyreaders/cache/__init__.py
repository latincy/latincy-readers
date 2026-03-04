"""Persistent caching, canonical annotations, and vector stores.

This subpackage provides three complementary persistence layers:

- :class:`DiskCache` — save/load annotated Docs to disk (DocBin format)
- :class:`CanonicalAnnotationStore` — version-controlled expert annotations
  stored in ``.conlluc`` (CoNLL-U Cache) format
- :class:`SentenceVectorStore` — sentence-level embedding search (requires numpy)

The ``.conlluc`` format is CoNLL-U with mandatory file-level metadata marking
content as silver-standard.  See :mod:`latincyreaders.cache.conlluc` for details.
"""

from latincyreaders.cache.disk import CacheConfig, DiskCache
from latincyreaders.cache.canonical import CanonicalAnnotationStore, CanonicalConfig
from latincyreaders.cache.conlluc import (
    CONLLUC_EXTENSION,
    doc_to_conlluc,
    conlluc_to_doc,
    read_conlluc,
    write_conlluc,
    validate_conlluc_header,
)

__all__ = [
    "CacheConfig",
    "DiskCache",
    "CanonicalAnnotationStore",
    "CanonicalConfig",
    # .conlluc format
    "CONLLUC_EXTENSION",
    "doc_to_conlluc",
    "conlluc_to_doc",
    "read_conlluc",
    "write_conlluc",
    "validate_conlluc_header",
]

# SentenceVectorStore requires numpy — import lazily
try:
    from latincyreaders.cache.vectors import SentenceVectorConfig, SentenceVectorStore

    __all__ += ["SentenceVectorConfig", "SentenceVectorStore"]
except ImportError:
    pass
