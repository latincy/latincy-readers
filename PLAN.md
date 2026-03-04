# Plan: Cached Collection Annotations, Canonical Annotations & Sentence Vector Stores

## Overview

Three interrelated features to make annotations persistent, shareable, and searchable:

1. **Persistent Disk Cache** — Save/load annotations to disk (beyond in-memory LRU)
2. **Canonical Annotations** — Version-controlled expert annotations for known collections (starting with Tesserae)
3. **Sentence Vector Store** — Efficient array-based sentence embeddings for similarity search

---

## Architecture: The Canonical/Dynamic Split

The UDReader already demonstrates the canonical/dynamic pattern: it constructs `Doc` objects directly from gold-standard CoNLL-U annotations rather than running the spaCy pipeline. We extend this principle:

- **Canonical annotations** = pre-computed, community-corrected, version-controlled (like UD treebanks)
- **Dynamic annotations** = computed on-the-fly via spaCy pipeline (current behavior for Tesserae, plaintext, etc.)
- A reader can fall back from canonical → dynamic when canonical data is missing

---

## Feature 1: Persistent Disk Cache (`cache/` subpackage)

### New module: `src/latincyreaders/cache/__init__.py`

Exports: `DiskCache`, `CacheConfig`

### New module: `src/latincyreaders/cache/disk.py`

**`CacheConfig` dataclass:**
- `cache_dir: Path | None` — directory for cached annotations (default: `~/.latincy_cache/`)
- `persist: bool` — whether to save to disk (default: `False`, opt-in)
- `ttl_days: int | None` — expiration/refresh period in days (default: `None` = never expires)
- `collection: str | None` — collection name for organizing cache (e.g., "tesserae", "proiel")

**`DiskCache` class:**
- Serializes spaCy `Doc` objects to disk using `DocBin` (spaCy's native binary format — compact, fast, includes all annotations)
- Keyed by `(collection, fileid)` → stored as `{cache_dir}/{collection}/{fileid_hash}.spacy`
- Manifest file: `{cache_dir}/{collection}/manifest.json` — maps fileid → hash, annotation_level, model_name, timestamp, version
- Methods:
  - `get(fileid, vocab) -> Doc | None` — load from disk if exists and not expired
  - `put(fileid, doc) -> None` — serialize to disk
  - `has(fileid) -> bool` — check existence + expiry
  - `invalidate(fileid) -> None` — remove specific entry
  - `clear(collection=None) -> None` — clear all or specific collection
  - `stats() -> dict` — cache statistics (size on disk, entry count, etc.)
  - `refresh_check(fileid) -> bool` — True if entry is stale per ttl_days

### Integration with `BaseCorpusReader`

- Add `cache_config: CacheConfig | None = None` parameter to `__init__`
- In `docs()`, check disk cache before processing: disk hit → deserialize → put in LRU → yield
- On LRU miss + disk miss → process → put in both LRU and disk
- New method: `persist_cache() -> int` — force-save all LRU entries to disk, return count
- New method: `warm_cache(fileids=None) -> int` — pre-process and cache all/selected files

### New file: `tests/test_disk_cache.py`

Tests for DiskCache round-trip, TTL expiry, invalidation, manifest integrity.

---

## Feature 2: Canonical Annotations (`cache/canonical.py`)

Modeled after the UDReader's approach of constructing Docs from gold data rather than running the pipeline.

### New module: `src/latincyreaders/cache/canonical.py`

**`CanonicalAnnotationStore` class:**
- Stores pre-computed, community-corrected annotations as `DocBin` files
- Directory structure: `{store_root}/{collection}/` with `.spacy` files per text + `manifest.json`
- Manifest includes: version, model_name, annotation_level, contributor info, upstream URL
- Methods:
  - `load(fileid, vocab) -> Doc | None` — load canonical annotations
  - `has(fileid) -> bool` — check if canonical annotation exists
  - `save(fileid, doc) -> None` — write canonical annotations (for building/correcting)
  - `export_collection(output_dir) -> None` — export for sharing/versioning
  - `import_collection(source_dir) -> None` — import from shared source
  - `diff(fileid, doc) -> list[dict]` — compare canonical vs. dynamic annotations (token-level diffs in lemma, POS, etc.)

**`CanonicalConfig` dataclass:**
- `store_root: Path` — location of canonical annotation store
- `collection: str` — collection identifier (e.g., "cltk-tesserae")
- `prefer_canonical: bool` — when True (default), use canonical over dynamic
- `auto_download: bool` — download canonical annotations if not present

### Integration with readers

- `TesseraeReader` gets a new `canonical: CanonicalConfig | None = None` parameter
- In `docs()`: if canonical exists and `prefer_canonical=True`, load from canonical store; otherwise fall back to dynamic pipeline — same pattern as UDReader constructing from gold data
- New method: `build_canonical(fileids=None) -> int` — process all files and save to canonical store
- New method: `compare_annotations(fileid) -> list[dict]` — show diffs between canonical and dynamic

### CLTK-Tesserae canonical data

- Default store location: `~/latincy_data/canonical/cltk-tesserae/`
- Environment variable: `TESSERAE_CANONICAL_PATH`
- Future: a separate GitHub repo (`diyclassics/latincy-canonical-tesserae`) with version-controlled `.spacy` files + manifest, downloadable via `DownloadableCorpusMixin`

### New file: `tests/test_canonical.py`

Tests for store CRUD, canonical-vs-dynamic comparison, import/export.

---

## Feature 3: Sentence Vector Store (`cache/vectors.py`)

### New module: `src/latincyreaders/cache/vectors.py`

**`SentenceVectorStore` class:**
- Stores sentence-level vectors as NumPy `.npy` memory-mapped arrays for efficient similarity search
- Directory structure: `{store_root}/{collection}/vectors/`
  - `{fileid_hash}_vectors.npy` — (n_sents, vector_dim) float32 array
  - `{fileid_hash}_index.json` — maps array index → (fileid, sent_idx, citation, text_preview)
- Methods:
  - `build(reader, fileids=None) -> int` — compute and store sentence vectors from a reader
  - `add_doc(doc) -> None` — add vectors for a single doc's sentences
  - `similar(query_vector, top_k=10) -> list[dict]` — find most similar sentences by cosine similarity
  - `similar_to_sent(sent_text, nlp, top_k=10) -> list[dict]` — convenience: vectorize query text then search
  - `similar_to_doc_sent(fileid, sent_idx, top_k=10) -> list[dict]` — find similar to an existing indexed sentence
  - `stats() -> dict` — total sentences indexed, vector dimensions, disk size
  - `clear() -> None` — remove all stored vectors

**Vector computation:**
- Default: mean of token vectors from spaCy model (already available via `doc.vector`, `sent.vector`)
- Uses `la_core_web_lg` vectors by default (300-dim)
- Similarity via cosine distance (NumPy dot product on normalized vectors — no heavy dependencies needed)
- Memory-mapped arrays allow searching millions of sentences without loading everything into RAM

**`SentenceVectorConfig` dataclass:**
- `store_root: Path` — default `~/latincy_data/vectors/`
- `collection: str` — collection name
- `vector_source: str` — "spacy" (default) for model vectors

### Integration with `BaseCorpusReader`

- New method on `BaseCorpusReader`: `build_vectors(config: SentenceVectorConfig, fileids=None) -> SentenceVectorStore`
- New method: `find_similar(text, top_k=10, config=None) -> list[dict]` — convenience for similarity search

### New optional dependency

- `numpy` added to `[project.optional-dependencies]` as `vectors = ["numpy>=1.24"]`
- Import guarded: `SentenceVectorStore` raises `ImportError` with helpful message if numpy not installed

### New file: `tests/test_vectors.py`

Tests for build, similarity search, persistence, memory-mapped access.

---

## File Changes Summary

### New files:
1. `src/latincyreaders/cache/__init__.py` — exports DiskCache, CanonicalAnnotationStore, SentenceVectorStore
2. `src/latincyreaders/cache/disk.py` — DiskCache + CacheConfig
3. `src/latincyreaders/cache/canonical.py` — CanonicalAnnotationStore + CanonicalConfig
4. `src/latincyreaders/cache/vectors.py` — SentenceVectorStore + SentenceVectorConfig
5. `tests/test_disk_cache.py`
6. `tests/test_canonical.py`
7. `tests/test_vectors.py`

### Modified files:
1. `src/latincyreaders/core/base.py` — add `cache_config` param, disk cache integration in `docs()`, `persist_cache()`, `warm_cache()`, `build_vectors()`, `find_similar()` methods
2. `src/latincyreaders/readers/tesserae.py` — add `canonical` param, canonical annotation support in `docs()`, `build_canonical()`, `compare_annotations()` methods
3. `src/latincyreaders/nlp/pipeline.py` — no changes needed (extensions already registered)
4. `pyproject.toml` — add `vectors` optional dependency group, bump version to 1.3.0

---

## Implementation Order

1. `cache/disk.py` + `cache/__init__.py` — foundation for persistence
2. Integrate disk cache into `BaseCorpusReader.docs()`
3. `cache/canonical.py` — canonical annotation store
4. Integrate canonical into `TesseraeReader.docs()`
5. `cache/vectors.py` — sentence vector store
6. Integrate vectors into `BaseCorpusReader`
7. Tests for all three features
8. Update `pyproject.toml` (version bump + optional deps)
