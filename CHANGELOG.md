# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-04-27

### Added

- **DigilibLTReader** for the [digilibLT](http://digiliblt.uniupo.it) corpus
  (Digital Library of Late-Antique Latin Texts) — chapter-aware reader for all
  structural patterns in the collection (flat `<p>`, `<div type="cap">`, nested
  `lib`/`cap`, `section` with `<head>`, verse `<lg>/<l>`)
  - Chapter-level structure exposed as named spans (`doc.spans["chapters"]`)
  - Rich metadata extraction: DLT ID, author (via `persName[@type='usualname']`),
    source bibliography, creation date
  - `use_symbols=True` (default) strips text-critical marks (`< >`, `[ ]`, `{ }`,
    `†`, `***`) and expands abbreviations (`M(arcus)` → `Marcus`) before NLP
  - `chapters(as_text=True)` yields `(citation, text)` tuples with zero NLP overhead

### Changed

- **Model installation moved from extras to documented URLs.** The `[la]`,
  `[grc]`, and `[all]` install extras (added in 1.4.1 but never published — they
  used direct-URL refs that PyPI rejects on upload) have been removed. Install
  LatinCy models separately via their Hugging Face wheel URLs — see the README
  *Models* section. This mirrors spaCy's own pattern for language models.

### Fixed

- Project URLs in `pyproject.toml` corrected from `github.com/diyclassics/...`
  to `github.com/latincy/...` (the actual repo location).

## [1.4.1] - 2026-03-20

### Added

- **Corrections module** for tracking token-level human corrections across model
  upgrades — extract, save, load, and apply correction workflow
- **Install extras** for [LatinCy](https://github.com/diyclassics/latincy) model
  wheels (hosted on Hugging Face): `[la]` (la_core_web_lg 3.9.0),
  `[grc]` (grc_dep_web_lg 3.8.1), and `[all]` for both

### Changed

- `token._.remorph` is now persisted through `DocBin` serialization (stashed in
  `doc.user_data`, restored on load) so cached docs preserve remorph annotations
- README install instructions updated for the new model extras
- Greek model switched from OdyCy to LatinCy `grc_dep_web_lg` (merged from
  `update-greek-model-v1.5`)

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
- **NLP backend abstraction** (`NLPBackend`, `SpaCyBackend`) for future multi-backend support
- **WikiSourceReader** for la.wikisource.org

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
