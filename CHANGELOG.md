# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.9.0] - 2026-10-02

### Added

- **Gold correction layer.** NLP annotations can now be corrected and the fixes
  *locked in*, so a cached text carries trustworthy annotations and a
  training-ready record of human judgements. `reader.correct(fileid, token_id,
  field, value, evidence=...)` records a correction; it is applied as a read-time
  overlay on every subsequent `docs()` call. Corrections live in a durable store
  (`~/latincy_data/corrections/<collection>/`, or a `corrections_dir` you pass)
  **separate from the DocBin cache**, so they survive `clear_cache()` and full
  re-annotation. Supported fields: `lemma`, `upos`, `xpos`, `feats`, `deprel`.
- **Durable opaque token ids** (`Token._.token_id`, e.g. `t0042`) — the join key
  between the DocBin base cache and the correction layer. Minted positionally,
  persisted through DocBin via `user_data`, and surfaced in `.conlluc` exports as
  `TokenId=` in the MISC column.
- **Token-drift migration** (`latincyreaders.cache.migrate`) — a pure form-anchored
  aligner (ported from latincy-viewer) that re-points corrections across
  tokenization changes (split/merge/shift), verifying each remap against the
  recorded surface form and quarantining anything that cannot be placed safely to
  `corrections_unresolved.log` rather than mis-applying it.
- **Automatic re-pointing on rebuild.** When a model upgrade (generator-stamp
  mismatch) rebuilds a text's DocBin base and the tokenization changed, its
  corrections are re-pointed onto the new tokens automatically — the pre-rebuild
  DocBin is captured (`DiskCache.load_raw`) as the alignment source. Force it
  explicitly with `reader.repoint_corrections(fileid)`.
- **Migration ledger.** Each drift re-pointing is recorded as a revision
  (`migrations.jsonl` + `head.json` in the correction store) capturing the
  `from → to` generator, revision number, and repointed/quarantined counts —
  an auditable trail of "this collection was migrated from lg-3.9.4 to 3.9.6."
  Inspect via `CorrectionStore.migrations()`.
- **EDHReader, FormulaeReader, EpistolaeReader.** None of the three download the
  corpus — `root` must point at a local checkout the user has acquired
  themselves. Designed for use with the following open collections:
  - `EDHReader` — Epigraphic Database Heidelberg EpiDoc TEI-XML
    (~82K Latin inscriptions; CC BY-SA 4.0).
  - `FormulaeReader` — Formulae-Litterae-Chartae TEI-XML charters
    (CC BY 4.0).
  - `EpistolaeReader` — Epistolae medieval women's Latin letters, Hugo
    Markdown source (CC BY-NC-SA 4.0).

### Changed

- **Corrections coexist with the base, never overwrite it.** The overlay surfaces
  the gold value on the in-memory Doc and flags `Token._.corrected`; the DocBin
  base cache is never mutated (it stays silver), and the correction record keeps
  the machine value it overrode in `was`.
- Local `scratch/` working files are now excluded from git and from the
  sdist (previously picked up by the build regardless of git-tracking,
  since packaging is filesystem-based, not git-aware).
- README: Readers table and Corpora Supported now list EDHReader,
  FormulaeReader, EpistolaeReader and WikiSourceReader; Bibliography adds a
  Collections section with references for each supported corpus.

### Fixed

- **EDHReader now uses the shared cache/correction pipeline.** `docs()`
  previously called the NLP pipeline directly, bypassing disk caching,
  token-id minting, and the gold-correction overlay entirely —
  `reader.correct()` always raised on EDH docs.
- **EDHReader compound abbreviation expansion.** A multi-pair `<expan>`
  (e.g. `co(n)s(ul)`) collapsed to only its last `abbr`/`ex` pair; all
  pairs are now accumulated in document order.
- **FormulaeReader `remove_notes`.** The `remove_notes=True` default had no
  effect — `<note>` editorial commentary was not being stripped from the
  extracted edition text.
- **Correction store: collision-safe filenames.** Two fileids that
  flattened to the same path under the old naive scheme (e.g. `a/b` and
  `a--b`) could collide onto the same `.corr.json` file; correction
  filenames now reuse the same sha256 hash as the DocBin cache.
- **Correction store: invalid values now rejected at record time.**
  `reader.correct()` previously accepted any value silently, even one that
  could never actually apply (e.g. a non-UD `upos` tag), leaving the
  correction permanently inert. It now raises immediately.
- **Correction store: self-anchoring re-point no longer drifts across
  repeated phrases.** The fallback re-pointer (used when the pre-rebuild
  DocBin is unavailable) now scopes its alignment to the correction's own
  sentence rather than the whole document — a repeated formulaic phrase
  could previously re-anchor a correction to the wrong occurrence.
- **`persist_cache()` no longer bakes gold corrections into the disk
  cache.** The base DocBin cache is documented to stay "silver"
  (uncorrected); force-flushing the in-memory cache could previously write
  already-corrected values to disk.
- **`repoint_corrections()` now actually forces a re-anchor.** It
  previously only cleared the in-memory cache, so a still-fresh disk cache
  entry was just reloaded unchanged, making the re-point a no-op.
- **Freshly-annotated docs are no longer lost on a downstream caching
  failure.** A failure in the disk-cache write, canonical-store save, or
  correction re-point/overlay step (e.g. disk full) previously discarded
  NLP work that had already completed; the doc is now cached first. Fixed
  in both the shared caching path and `TesseraeReader`, which carries its
  own duplicate of the same pipeline.
- **`Token._.remorph` is now always registered on disk-cache load,** even
  for a doc where every token is at the default (nothing to restore) —
  previously such a doc could leave the extension unregistered
  process-wide.

### Notes

- **Shared cache+overlay choke-point** (`BaseCorpusReader._cached_docs`). Readers
  route their per-file production through it to get the DocBin base cache +
  correction overlay uniformly. `TxtdownReader` and `WikiSourceReader` now use it
  (previously they reprocessed NLP on every read and carried no corrections);
  `BaseCorpusReader`/`TesseraeReader` keep their own read-through (they add the
  canonical `.conlluc` tier), and `DigilibtReader` inherits via `super().docs()`.
- **DocBin now preserves all known custom token extensions** (remorph, durable
  ids, text-critical flags, verse/newline/speaker markers) via a generalized
  `user_data` stash — previously only `remorph` survived a round-trip, so cached
  docs silently lost the rest.

## [1.8.0] - 2026-07-26

### Added

- **Annotation-model generator stamp.** Every cache write now records the model
  that produced it (`model_name` + `model_version`, read from the installed model
  package via `importlib.metadata`, falling back to a loaded pipeline's `meta`).
  Previously `model_version` was plumbed through the code but never populated, so
  cached artifacts recorded `model_version = unknown`.
- **`AnnotationModelMismatchWarning`** (in `latincyreaders.warnings`). When a
  canonical `.conlluc` was built by a different model/version than the one in use,
  loading it now emits a loud warning (once per fileid) instead of silently
  serving stale annotations. The canonical store is *not* auto-invalidated — it
  may be intentionally pinned or community-corrected — so the caller decides
  whether to rebuild.

### Changed

- **DocBin (`.spacy`) disk cache is now a stamp-keyed ephemeral layer.** Its
  read-through staleness check compares the active model generator against the
  stored stamp; a mismatch is a self-healing miss + rebuild. This closes the
  silent-stale failure where a DocBin cache built under `la_core_web_lg` 3.9.4
  kept serving stale sentence boundaries after an upgrade to 3.9.6.
- **`TesseraeReader.docs()` DocBin path now participates in staleness.** It
  previously passed no staleness key at all (no `source_hash`, no generator), so
  its cached blobs never invalidated; it now passes both.

### Migration

- Legacy cache entries written before this release carry no generator stamp and
  are therefore treated as stale on first access under a known model version —
  they are rebuilt once, then hit normally. No manual cache deletion is needed.

## [1.7.0] - 2026-07-09

### Added

- **`AnnotationLevel.MINIMAL`** — new level between `NONE` and `TOKENIZE`. Uses a
  blank spaCy model with a rule-based sentencizer (punctuation only); no model
  download required. Replaces the old `TOKENIZE` behavior.
- **Local metadata merge.** `BaseCorpusReader` accepts a `local_metadata` path to
  a private JSON file whose fields are merged fill-only (public corpus fields
  always win). `TesseraeReader` auto-detects
  `~/latincy_data/lat_text_tesserae_local.json`, so private annotations survive
  corpus re-downloads.
- **Version-aware corpus updates.** On init, a reader checks the remote for a
  newer release tag and offers to update (5s timeout, silent on failure).
  `installed_version()` / the `corpus_version` property report the git tag or
  commit actually on disk, so an analysis can record exactly which corpus
  produced it.

### Changed

- **`AnnotationLevel.TOKENIZE`** now loads `la_core_web_lg` with only the
  tokenizer, `tok2vec`, and neural `senter` enabled. This is far more accurate
  for Latin than the previous rule-based sentencizer and is the minimum
  recommended level for any sentence-aware work.
- Minimum recommended LatinCy model bumped to **3.9.6**.
- **Metadata discovery** now globs `**/metadata/*.json` by default, so metadata
  is found regardless of corpus nesting depth. Resolution is first-file-wins per
  field across all matched files. `TesseraeReader` normalizes bare-filename
  metadata keys to `texts/<name>` to match its `texts/`-nested file IDs.
- **`download()` updates in place.** An existing git checkout is refreshed with
  `git fetch` + checkout instead of being skipped, preserving untracked and
  gitignored files (e.g. `metadata_local.json`) across updates. The Tesserae
  corpus pin moves **v0.5 → v0.6**.
- Corpus download/update prompts default to **No** when stdin is not a TTY, so
  headless callers (CI, render pipelines, cron) get the no-op path instead of an
  `EOFError`.

## [1.6.2] - 2026-06-20

### Added

- **Txtdown 0.2 integration.** `TxtdownReader` now depends on the published
  [`txtdown>=0.2.0`](https://pypi.org/project/txtdown/) package and surfaces its
  two new markup features through spaCy custom extensions:
  - **Speaker markup** (`@Name:` for dramatic/dialogue texts) → `span._.speaker`
    and `token._.speaker`. `sents_with_citations()` reports the sentence's
    `speaker` (single value when unambiguous) plus a `speakers` list when a
    sentence spans a speaker change. Section-level speaker is set only when the
    whole section is a single voice.
  - **Cross-source quotation** (leading `>`, formerly an inline "blockquote") →
    `span._.is_quote` and `token._.is_quote`. In 0.2 a `>` line is a verbatim
    quotation of another source: the marker is stripped and the line is kept on
    its own line (it is no longer joined inline with the preceding authorial
    line). `sents_with_citations()` reports `is_quote=True` for sentences whose
    covered lines are entirely quotation.
- **Per-line indentation.** Verse works indent some lines (e.g. elegiac
  pentameters). Leading indentation is now normalized out of the `texts()`
  output (previously it leaked into the raw text) and captured instead as a
  per-line datum: `sents_with_citations()` reports `is_indented=True` for
  sentences whose source line was indented, so a reader can render the indent.
  A no-op for flush-left prose; the `docs()` token spine is unaffected.
- **Reproducible corpus pinning.** `DownloadableCorpusMixin` now supports an
  optional `CORPUS_VERSION` class attribute; when set, the corpus is cloned at
  that git tag/branch (`git clone --branch …`) instead of the default-branch
  HEAD. This makes "I ran X on corpus Y" reproducible across time.
  - **TesseraeReader** is pinned to `CORPUS_VERSION = "v0.5"`. Override per
    instance with `TesseraeReader(corpus_version="v0.6.2")` (or `"main"`), or
    point `TESSERAE_PATH` at a local checkout.
  - `reader.corpus_version` reports the release actually on disk (via
    `git describe`), so a notebook/paper can record exactly which corpus
    produced a result; returns `None` for non-git checkouts.
  - `DownloadableCorpusMixin.installed_version()` and `download(ref=…)` added.
  - On a version mismatch (requested ≠ on-disk), the reader warns and uses the
    existing copy rather than silently serving the wrong data or clobbering it.
  - Backward compatible: readers without `CORPUS_VERSION` (e.g.
    GreekTesseraeReader) keep tracking the default branch.

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
