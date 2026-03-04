"""Persistent caching, canonical annotations, and vector stores.

This subpackage provides three complementary persistence layers:

- :class:`DiskCache` — save/load annotated Docs to disk (DocBin format)
- :class:`CanonicalAnnotationStore` — version-controlled expert annotations
- :class:`SentenceVectorStore` — sentence-level embedding search (requires numpy)
"""

from latincyreaders.cache.disk import CacheConfig, DiskCache
from latincyreaders.cache.canonical import CanonicalAnnotationStore, CanonicalConfig

__all__ = [
    "CacheConfig",
    "DiskCache",
    "CanonicalAnnotationStore",
    "CanonicalConfig",
]

# SentenceVectorStore requires numpy — import lazily
try:
    from latincyreaders.cache.vectors import SentenceVectorConfig, SentenceVectorStore

    __all__ += ["SentenceVectorConfig", "SentenceVectorStore"]
except ImportError:
    pass
