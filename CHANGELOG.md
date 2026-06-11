# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.6.1] - 2026-06-11

### Fixed

- **TesseraeReader** now downloads the maintained LatinCy fork
  (`github.com/latincy/lat_text_tesserae`) instead of the CLTK upstream
  (`github.com/cltk/lat_text_tesserae`), so it picks up the LatinCy cleanup
  releases (v0.3 capitalization, v0.4 whitespace/Unicode, v0.5 accent/Greek
  normalization).
- Fixed the Tesserae download/read layout: clones to a clean
  `~/latincy_data/lat_text_tesserae` folder (was `…/lat_text_tesserae/texts`,
  which nested the repo's own `texts/` one level too deep) and reads `.tess`
  recursively from the repo's `texts/` subdirectory.

## [1.6.0] - 2026-06-08

### Added

- **ProjectGutenbergReader** — fetch plain-text files from Project Gutenberg
  by numeric ID, cache to disk, strip standard PG boilerplate (START/END markers),
  and expose the full corpus-reader interface
  - `model_name` and `lang` are explicit constructor params so the same reader
    works for Latin (`la_core_web_lg`) and English (`en_core_web_sm`) without
    subclassing
  - Caches to `~/.latincy_cache/gutenberg/` by default; subsequent reads are instant
  - Metadata per Doc: `pg_id`, `filename`, `path`

- **CSELReader** for the [Corpus Scriptorum Ecclesiasticorum Latinorum](https://github.com/OpenGreekAndLatin/csel-dev)
  — chapter-aware reader for the CSEL digital edition (Open Greek and Latin Project),
  CC-BY-SA 4.0
  - Handles the two-level `book`/`section` textpart hierarchy; each
    `<div subtype="section">` becomes a span in `doc.spans["chapters"]`
  - Citations follow the form `"book 1, section 3"` (uses `subtype=` attribute)
  - Metadata per Doc: `author`, `title` (prefers `xml:lang="lat"`), `cts_urn`, `filename`
  - Inherits critical mark normalization from DigilibLTReader (`use_symbols=True`)
  - `<note>` elements stripped from body text by default
  - `headers()` and `chapters(as_text=True)` for zero-NLP-overhead iteration
  - File pattern: `**/*.opp-lat1.xml`

- **PTAReader** for the [Patristic Text Archive](https://pta.bbaw.de) (PTA)
  — section-aware reader for ~210 Greek texts (~2.3M tokens) and Latin texts,
  all CC-BY 4.0
  - Each `<div type="textpart">` section yields a separate Doc, preserving
    CTS URN, language (`lat`/`grc`), author, title, div_type, div_n, and
    citation in `doc._.metadata`
  - Auto-download via `DownloadableCorpusMixin` (clones from GitHub into
    `~/latincy_data/pta_data` or `$PTA_PATH`)
  - `<note>` elements stripped from body text by default
  - Language detection from `xml:lang` attribute with filename-suffix fallback

- **Text-critical markup support for TxtdownReader** (West 1973 conventions)
  - Cruxes (`†text†`), additions (`<text>`), expansions (`M(arcus)` → `Marcus`),
    deletions (`{text}`), and lacunae (`[text]`) handled before NLP
  - Token extensions: `Token._.is_crux`, `Token._.is_addition`, `Token._.is_expansion`
  - `doc._.textcrit` records all occurrences with original form and clean text
  - `TxtdownReader._strip_critical_markup()` available as a static method

- **`Token._.newline_after`** extension — tracks verse line boundaries lost
  after LatinCy tokenization; populated by `mark_newlines_from_spans()`

- **`latincyreaders.utils.text_utils.find_line_in_doc_text`** — shared utility
  for NLP-normalization-tolerant line-to-span matching (J/I, V/U, whitespace)

### Fixed

- `mark_newlines_from_spans()`: skip final span (no spurious trailing newline);
  `list()` SpanGroup before slicing to avoid index errors
- Unpicklable extension getter under `nlp.pipe(n_process>1)` — lambda replaced
  with a named function so the getter survives multiprocessing serialization
- `auto_download` parameter removed from `CamenaReader` and `PTAReader` public
  API (never functional; silently ignored)

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
