# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-05-28

### Added

- **Text-critical markup support in `TxtdownReader`** — all standard West (1973)
  apparatus conventions are now stripped before NLP and recorded in a new
  `doc._.textcrit` dict:
  - Cruxes `†text†` → text preserved, `Token._.is_crux = True`
  - Editorial additions `<text>` → text preserved, `Token._.is_addition = True`
  - Expansions `M(arcus)` → `Marcus`, `Token._.is_expansion = True`
  - Deletions `{spurious}` → stripped entirely (recorded in `textcrit["deletions"]` with original and text, no token span)
  - Lacunae `[text]` → left as-is (NLP processes as-found)
- **`doc._.textcrit`** — new Doc-level dict on all Docs produced by `TxtdownReader`:
  `{"cruxes": [...], "additions": [...], "expansions": [...], "deletions": [...]}`
  Each non-deletion entry carries `original`, `text`, and `span` (token index tuple).
- **`Token._.is_crux`**, **`Token._.is_addition`**, **`Token._.is_expansion`** —
  new boolean token extensions, registered globally for all readers.
- **`Token._.newline_after`** — `True` on the last token of each source line;
  set by `mark_newlines_from_spans()` after line spans are built.
- **`Token._.text_with_nl`** — getter returning `token.text + "\n"` or
  `token.whitespace_`; `"".join(t._.text_with_nl for t in doc)` reconstructs
  line-structured source text.
- **`latincyreaders.utils.text_utils.find_line_in_doc_text()`** — shared utility
  for NLP-tolerant line-to-Doc alignment (handles J/I, V/U normalization and
  whitespace inserted around punctuation by LatinCy).

### Changed

- **`CamenaReader.root` is now a required argument.** The `auto_download`
  parameter and its fallback pathway have been removed from the public API.
  `CORPUS_URL` and `download()` remain available for manual corpus setup.
  Pass an explicit `root=` path (or set `CAMENA_ROOT`) to construct the reader.

### Fixed

- `mark_newlines_from_spans()` no longer appends a spurious trailing `\n` after
  the final line span, matching `str.splitlines(keepends=True)` semantics.
- Single-line sections in `TxtdownReader` no longer have their line citation
  silently overwritten by the section citation. (spaCy stores span extension
  values keyed by token range; when a section and its only line share the same
  range, the last write wins — line citations are now re-applied after all
  section spans are set.)

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
