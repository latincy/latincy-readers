# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-03-16

### Added

- **Sentence vector search** — semantic search across Latin texts using sentence-level embeddings
  - `SentenceVectorStore` for building and querying vector indices with cosine similarity
  - `SentenceVectorConfig` for collection-based index organization
  - `reader.find_similar()` shortcut with `auto_build=True` for lazy index creation
  - `reader.build_vectors()` for building indices from any reader
  - Memory-mapped NumPy arrays for efficient search (no external vector DB required)
  - Stored at `~/latincy_data/vectors/<collection>/` by default
- **Vector search CLI** (`cli/vector_search.py`) with `build`, `query`, and `stats` subcommands
- **Vector search demo notebook** (`notebooks/vector-search-demo.ipynb`)
- **3-tier annotation caching** — read-through path: LRU → DocBin → .conlluc → NLP pipeline
  - `DiskCache` for persistent DocBin storage
  - `CanonicalAnnotationStore` for version-controlled expert annotations in `.conlluc` format
  - CoNLL-U Cache format (`.conlluc`) — CoNLL-U with mandatory silver-standard metadata
- **Lazy model loading** — lightweight vocab for cache deserialization avoids ~7s model load
  when all documents are served from cache (8x speedup)
- **Streamlit annotation editor** with tabbed layer view (Lemma | UPOS | XPOS | Morph | NER)
- **NLP backend abstraction** (`NLPBackend`, `SpaCyBackend`) for future multi-backend support
- **WikiSourceReader** for la.wikisource.org
- **AGLDT-to-UD tree converter** for Prague-style dependency annotations

## [1.3.0] - 2026-02-15

### Added

- WikiSourceReader for la.wikisource.org (49 tests)
- NLP backend abstraction (SpaCyBackend, stubs for Stanza/Flair)
- 478+ total tests

## [1.2.0] - 2026-01-20

### Added

- GreekTesseraeReader with OdyCy integration
- Universal Dependencies readers (PROIEL, Perseus, ITTB, LLCT, UDante, CIRCSE)
- LatinUDReader composite reader for all 6 Latin UD treebanks
- FileSelector fluent API for complex file queries
- MetadataManager with schema validation
- CombinedReader for multi-reader composition
- Search API: find_sents(), search(), concordance(), kwic(), ngrams(), skipgrams()

## [1.1.0] - 2025-12-15

### Added

- TesseraeReader, PlaintextReader, LatinLibraryReader
- TEIReader, PerseusReader, CamenaReader
- TxtdownReader
- AnnotationLevel enum (NONE, TOKENIZE, BASIC, FULL)
- Auto-download support for corpora
- Document caching with LRU eviction
