"""Base corpus reader class.

This module provides the abstract base class that all corpus readers inherit from.
It handles common functionality like file discovery, NLP pipeline management,
and the standard iteration interface.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

from tqdm import tqdm

from latincyreaders.nlp.pipeline import AnnotationLevel, get_nlp

if TYPE_CHECKING:
    from spacy import Language
    from spacy.tokens import Doc, Span, Token
    from latincyreaders.cache.disk import CacheConfig, DiskCache
    from latincyreaders.cache.canonical import CanonicalAnnotationStore, CanonicalConfig
    from latincyreaders.cache.correction_store import CorrectionStore
    from latincyreaders.cache.vectors import SentenceVectorConfig, SentenceVectorStore
    from latincyreaders.core.selector import FileSelector
    from latincyreaders.nlp.backends import NLPBackend

# Re-export for convenience
__all__ = ["BaseCorpusReader", "AnnotationLevel"]


class BaseCorpusReader(ABC):
    """Abstract base class for all Latin corpus readers.

    To create a new reader, subclass and implement:

    Required:
        - _parse_file(path) -> yields (text, metadata) tuples

    Optional overrides:
        - _normalize_text(text) -> cleaned text
        - _default_file_pattern() -> glob pattern for files

    Example:
        class MyReader(BaseCorpusReader):
            @classmethod
            def _default_file_pattern(cls) -> str:
                return "*.txt"

            def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
                yield path.read_text(), {"filename": path.name}
    """

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        metadata_pattern: str = "**/metadata/*.json",
        local_metadata: "Path | str | None" = None,
        cache: bool = True,
        cache_maxsize: int = 128,
        model_name: str = "la_core_web_lg",
        lang: str = "la",
        backend: "NLPBackend | None" = None,
        cache_config: "CacheConfig | None" = None,
        canonical_config: "CanonicalConfig | None" = None,
        corrections_dir: "Path | str | None" = None,
        enable: list[str] | None = None,
        disable: list[str] | None = None,
    ):
        """Initialize the corpus reader.

        Args:
            root: Root directory containing corpus files.
            fileids: Glob pattern for selecting files. If None, uses class default.
            encoding: Text encoding for reading files.
            annotation_level: How much NLP annotation to apply. Ignored when
                *enable* or *disable* is provided (except NONE/TOKENIZE).
            metadata_pattern: Glob pattern for metadata JSON files, relative to the corpus
                root. Defaults to ``**/metadata/*.json`` to find a ``metadata/`` folder at
                any depth. Set to None to disable.
            local_metadata: Path to a local JSON metadata file that will be merged with
                corpus metadata. Local fields are added only for keys not already present
                in the public metadata (fill-only, never overrides). Useful for private
                annotations that cannot be committed to the public corpus.
            cache: If True (default), cache processed Doc objects for reuse.
            cache_maxsize: Maximum number of documents to cache (default 128).
            model_name: Name of the spaCy model to load for BASIC/FULL levels.
            lang: Language code for blank model in TOKENIZE level.
            backend: Optional NLP backend. If provided, used instead of creating
                a pipeline from model_name/lang/annotation_level.
            cache_config: Optional disk cache configuration. When provided with
                ``persist=True``, annotations are saved to disk for fast reloading
                across sessions.
            canonical_config: Optional canonical annotation store configuration.
                When provided, the reader will check the canonical store for
                pre-computed ``.conlluc`` annotations before running the NLP
                pipeline.  Combined with *cache_config*, the read-through
                path is: LRU → DocBin (disk) → .conlluc (canonical) → pipeline.
            enable: Component names to enable (additive). Backbone components
                (tok2vec/transformer, senter) are always included. Mutually
                exclusive with *disable*.
            disable: Component names to disable (subtractive). Backbone
                components cannot be disabled. Mutually exclusive with *enable*.
        """
        self._root = Path(root).resolve()
        self._fileids_pattern = fileids or self._default_file_pattern()
        self._encoding = encoding
        self._annotation_level = annotation_level
        self._model_name = model_name
        self._lang = lang
        self._backend = backend
        self._enable = enable
        self._disable = disable
        self._nlp: Language | None = None  # Lazy loaded
        self._metadata_pattern = metadata_pattern
        self._local_metadata_path = Path(local_metadata) if local_metadata else None
        self._metadata: dict[str, dict[str, Any]] | None = None  # Lazy loaded

        # Caching
        self._cache_enabled = cache
        self._cache_maxsize = cache_maxsize
        self._cache: OrderedDict[str, "Doc"] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # Disk cache
        self._cache_config = cache_config
        self._disk_cache: "DiskCache | None" = None
        if cache_config is not None and cache_config.persist:
            from latincyreaders.cache.disk import DiskCache

            self._disk_cache = DiskCache(cache_config)

        # Canonical annotation store
        self._canonical_config = canonical_config
        self._canonical_store: "CanonicalAnnotationStore | None" = None
        if canonical_config is not None:
            from latincyreaders.cache.canonical import CanonicalAnnotationStore

            self._canonical_store = CanonicalAnnotationStore(canonical_config)

        # Gold correction layer — enabled when a collection name is known (from the
        # cache or canonical config). Corrections live outside the ephemeral DocBin
        # cache so they survive clear_cache() and full re-annotation.
        self._collection: str | None = None
        if cache_config is not None and cache_config.collection:
            self._collection = cache_config.collection
        elif canonical_config is not None and canonical_config.collection:
            self._collection = canonical_config.collection
        self._corrections_dir = Path(corrections_dir) if corrections_dir else None
        self._correction_store: "CorrectionStore | None" = None

    @property
    def root(self) -> Path:
        """Root directory of the corpus."""
        return self._root

    @property
    def nlp(self) -> Language | None:
        """spaCy pipeline (lazy loaded on first access).

        If a backend was provided, delegates to the backend's nlp property.
        """
        if self._backend is not None:
            return self._backend.nlp
        if self._nlp is None and self._annotation_level != AnnotationLevel.NONE:
            self._nlp = get_nlp(
                self._annotation_level,
                model_name=self._model_name,
                lang=self._lang,
                enable=self._enable,
                disable=self._disable,
            )
        return self._nlp

    @property
    def vocab(self):
        """Lightweight vocab for deserializing cached docs without loading the full model.

        Returns the model's vocab if already loaded, otherwise creates a
        minimal blank vocab.  This avoids a ~7s model load when all
        documents are served from .conlluc or other caches.
        """
        if self._nlp is not None:
            return self._nlp.vocab
        if not hasattr(self, "_vocab"):
            import spacy
            self._vocab = spacy.blank(self._lang).vocab
        return self._vocab

    def _active_generator(self) -> tuple[str, str]:
        """Return ``(model_name, model_version)`` for the active annotation model.

        This is the generator stamp written into cache entries and compared on
        read.  The version is read from an already-loaded pipeline's ``meta``
        when available; otherwise from the installed model package via
        :mod:`importlib.metadata` (no pipeline load, so cache-only readers stay
        lazy); failing both, ``"unknown"``.
        """
        nlp = self._backend.nlp if self._backend is not None else self._nlp
        if nlp is not None:
            return self._model_name, str(nlp.meta.get("version", "unknown"))
        try:
            return self._model_name, importlib.metadata.version(self._model_name)
        except importlib.metadata.PackageNotFoundError:
            return self._model_name, "unknown"

    @staticmethod
    def _ensure_token_ids(doc: "Doc") -> "Doc":
        """Ensure every token carries a durable opaque id in ``Token._.token_id``.

        Ids are positional at mint time (``t0000``, ``t0001``, …) and are the join
        key to the JSON correction layer. They are minted only where missing, so a
        DocBin-loaded doc keeps the ids restored from ``user_data`` and a freshly
        produced doc gets fresh ones. Across a re-tokenization the *corrections*
        are re-pointed (via :mod:`latincyreaders.cache.migrate`), not the token
        strings frozen — so positional minting here is always safe.
        """
        from spacy.tokens import Token

        if not Token.has_extension("token_id"):
            Token.set_extension("token_id", default=None)
        for token in doc:
            if token._.token_id is None:
                token._.token_id = f"t{token.i:04d}"
        return doc

    def _get_correction_store(self) -> "CorrectionStore | None":
        """The gold correction store for this collection (lazy), or None."""
        if self._collection is None:
            return None
        if self._correction_store is None:
            from latincyreaders.cache.correction_store import (
                _DEFAULT_STORE_ROOT,
                CorrectionStore,
            )

            root = self._corrections_dir or _DEFAULT_STORE_ROOT
            self._correction_store = CorrectionStore(self._collection, store_root=root)
        return self._correction_store

    def _overlay_corrections(self, fileid: str, doc: "Doc") -> "Doc":
        """Overlay gold corrections onto *doc* in memory (base cache untouched)."""
        store = self._get_correction_store()
        if store is not None:
            store.overlay(fileid, doc)
        return doc

    def _repoint_on_rebuild(
        self, fileid: str, new_doc: "Doc", old_doc: "Doc | None"
    ) -> None:
        """Re-point this file's corrections when a rebuild re-tokenized it.

        Called on a DocBin miss where a stale entry existed (*old_doc* is the
        pre-rebuild doc). If the tokenization actually changed, align old→new and
        carry the corrections' durable ids across (via
        :meth:`CorrectionStore.repoint`); survivors re-anchor, and anything whose
        word-content vanished is quarantined to ``corrections_unresolved.log``.
        No-op when there are no corrections or the token forms are unchanged.
        """
        if old_doc is None:
            return
        store = self._get_correction_store()
        if store is None:
            return
        cset = store.load(fileid)
        if cset is None or not cset.corrections:
            return
        if [t.text for t in old_doc] == [t.text for t in new_doc]:
            return  # tokenization stable — ids still valid
        model_name, model_version = self._active_generator()
        store.repoint(
            fileid, new_doc, old_doc=old_doc,
            to_generator=f"{model_name}@{model_version}",
        )

    def repoint_corrections(self, fileid: str) -> tuple[int, int]:
        """Explicitly re-point *fileid*'s corrections onto the current model's
        tokenization. Returns ``(repointed, quarantined)``.

        Uses the pre-rebuild DocBin (if present) as the alignment source, else the
        corrections' stored sentence context. Normally triggered automatically on
        a rebuild; call this to force it (e.g. after a manual cache change).
        """
        store = self._get_correction_store()
        if store is None or store.load(fileid) is None:
            return (0, 0)
        vocab = self.nlp.vocab if self.nlp is not None else self.vocab
        old_doc = (
            self._disk_cache.load_raw(fileid, vocab)
            if self._disk_cache is not None else None
        )
        self._cache.pop(fileid, None)
        if self._disk_cache is not None:
            # Force a genuine rebuild: a fresh (non-invalidated) disk entry
            # would just be reloaded as-is, making old_doc and new_doc
            # identical and the re-point below a no-op.
            self._disk_cache.invalidate(fileid)
        new_doc = next(self.docs(fileid))
        model_name, model_version = self._active_generator()
        return store.repoint(
            fileid, new_doc, old_doc=old_doc,
            to_generator=f"{model_name}@{model_version}",
        )

    def _store_lru(self, fileid: str, doc: "Doc") -> None:
        """Evict-and-store a doc in the in-memory LRU cache, if enabled."""
        if self._cache_enabled:
            while len(self._cache) >= self._cache_maxsize:
                self._cache.popitem(last=False)
            self._cache[fileid] = doc

    def _cached_docs(self, fileids, produce) -> Iterator["Doc"]:
        """Shared read path: LRU → stamped DocBin → produce, with correction overlay.

        The single caching + correction choke-point for readers whose ``docs()``
        does not need the canonical ``.conlluc`` tier. *produce* is a callable
        ``(fileid, path) -> Iterator[Doc]`` running the reader-specific NLP + span
        building on a cache miss.

        Invariant: every disk write uses the pre-overlay (silver) doc; durable
        token ids are attached and gold corrections overlaid as the final in-memory
        step, so the DocBin base cache never carries correction state.
        """
        model_name, model_version = self._active_generator()
        generator = f"{model_name}@{model_version}"

        for path in self._iter_paths(fileids):
            fileid = str(path.relative_to(self._root))

            # 1. LRU
            if self._cache_enabled and fileid in self._cache:
                self._cache_hits += 1
                self._cache.move_to_end(fileid)
                yield self._cache[fileid]
                continue

            vocab = self.nlp.vocab if self.nlp is not None else self.vocab
            source_hash: str | None = None
            if self._canonical_store is not None:
                source_hash = self._canonical_store.content_hash(fileid)

            # 2. DocBin base cache (stamp-keyed)
            old_doc = None
            if self._disk_cache is not None:
                disk_doc = self._disk_cache.get(
                    fileid, vocab, source_hash=source_hash, generator=generator,
                )
                if disk_doc is not None:
                    self._cache_hits += 1
                    self._ensure_token_ids(disk_doc)
                    self._overlay_corrections(fileid, disk_doc)
                    self._store_lru(fileid, disk_doc)
                    yield disk_doc
                    continue
                # Miss with a stale entry present → capture it for re-pointing.
                old_doc = self._disk_cache.load_raw(fileid, vocab)

            # 3. Cache miss — produce (reader-specific), persist silver, overlay
            if self._cache_enabled:
                self._cache_misses += 1

            for doc in produce(fileid, path):
                if doc._.fileid is None:
                    doc._.fileid = fileid
                self._ensure_token_ids(doc)

                # Cache immediately, before any step that could raise (disk
                # write, repoint, overlay) — otherwise a failure there would
                # discard NLP work that already succeeded. Later in-place
                # mutations still apply, since this is the same object.
                self._store_lru(fileid, doc)

                if self._disk_cache is not None:
                    self._disk_cache.put(
                        fileid, doc,
                        annotation_level=self._annotation_level.name,
                        model_name=self._model_name,
                        model_version=model_version,
                        generator=generator,
                        **({"source_hash": source_hash} if source_hash else {}),
                    )

                self._repoint_on_rebuild(fileid, doc, old_doc)
                old_doc = None  # only the first produced doc pairs with the old one
                self._overlay_corrections(fileid, doc)
                yield doc

    def correct(
        self,
        fileid: str,
        token_id: str,
        field: str,
        value: str,
        *,
        evidence: str = "",
    ) -> Any:
        """Record a gold correction for one token's field and refresh the overlay.

        The correction is stored in the durable correction layer (outside the
        DocBin cache), keyed by the token's durable id. The base annotations are
        never mutated; the correction is surfaced on every subsequent ``docs()``
        read via the read-time overlay.

        Args:
            fileid: File identifier.
            token_id: Durable ``Token._.token_id`` of the target token.
            field: One of ``lemma``, ``upos``, ``xpos``, ``feats``, ``deprel``.
            value: The corrected value.
            evidence: Optional free-text justification.

        Returns:
            The persisted correction record.
        """
        store = self._get_correction_store()
        if store is None:
            raise ValueError(
                "Corrections require a collection name. Pass cache_config or "
                "canonical_config with a `collection` set."
            )
        doc = next(self.docs(fileid))
        model_name, model_version = self._active_generator()
        record = store.record(
            fileid, doc, token_id, field, value,
            evidence=evidence, generator=f"{model_name}@{model_version}",
        )
        self._cache.pop(fileid, None)  # drop LRU so next read re-overlays
        return record

    @property
    def annotation_level(self) -> AnnotationLevel:
        """Current annotation level."""
        return self._annotation_level

    @property
    def cache_enabled(self) -> bool:
        """Whether document caching is enabled."""
        return self._cache_enabled

    def cache_stats(self) -> dict[str, int]:
        """Return cache statistics.

        Returns:
            Dict with keys:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - size: Current number of cached documents
                - maxsize: Maximum cache size
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "maxsize": self._cache_maxsize,
        }

    def clear_cache(self) -> None:
        """Clear the document cache and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        """Load and aggregate metadata from JSON files.

        Resolution policy: first file wins per field. Files are processed in
        sorted order, so ``metadata.json`` takes precedence over
        ``metadata_local.json`` for any field present in both. The local_metadata
        path (if set) is also merged with the same fill-only semantics.

        Returns:
            Dict mapping fileid -> metadata dict.
        """
        merged: dict[str, dict[str, Any]] = {}

        if self._metadata_pattern is not None:
            for json_file in sorted(self._root.glob(self._metadata_pattern)):
                try:
                    data = json.loads(json_file.read_text(encoding=self._encoding))
                    if isinstance(data, dict):
                        for fileid, meta in data.items():
                            if isinstance(meta, dict):
                                entry = merged.setdefault(fileid, {})
                                for k, v in meta.items():
                                    entry.setdefault(k, v)  # first file wins per field
                except (json.JSONDecodeError, OSError):
                    continue

        if self._local_metadata_path is not None and self._local_metadata_path.exists():
            try:
                data = json.loads(self._local_metadata_path.read_text(encoding=self._encoding))
                if isinstance(data, dict):
                    for fileid, local_meta in data.items():
                        if isinstance(local_meta, dict):
                            entry = merged.setdefault(fileid, {})
                            for k, v in local_meta.items():
                                entry.setdefault(k, v)  # fill only, never override
            except (json.JSONDecodeError, OSError):
                pass

        return merged

    def get_metadata(self, fileid: str) -> dict[str, Any]:
        """Get metadata for a specific file.

        Args:
            fileid: File identifier.

        Returns:
            Metadata dict for the file, or empty dict if not found.
        """
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata.get(fileid, {})

    def metadata(
        self,
        fileids: str | list[str] | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield (fileid, metadata) pairs.

        Args:
            fileids: Files to get metadata for, or None for all.

        Yields:
            Tuples of (fileid, metadata_dict).
        """
        for fileid in self._resolve_fileids(fileids):
            yield fileid, self.get_metadata(fileid)

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Default glob pattern for this corpus type. Override in subclasses."""
        return "*.*"

    def _normalize_text(self, text: str) -> str:
        """Normalize text. Override for corpus-specific cleaning.

        Args:
            text: Raw text from file.

        Returns:
            Normalized text.
        """
        return unicodedata.normalize("NFC", text)

    def fileids(self, match: str | None = None) -> list[str]:
        """Return list of file identifiers matching the pattern.

        Args:
            match: Optional regex pattern to filter filenames.

        Returns:
            Naturally sorted list of matching file identifiers (relative paths).
        """
        from natsort import natsorted

        pattern = self._fileids_pattern
        files = self._root.glob(pattern)

        # Convert to relative paths as strings
        result = [str(f.relative_to(self._root)) for f in files if f.is_file()]

        # Apply optional regex filter
        if match:
            regex = re.compile(match, re.IGNORECASE)
            result = [f for f in result if regex.search(f)]

        # Natural sort (handles numbers correctly: part.1, part.2, ..., part.10)
        return natsorted(result)

    def select(self) -> "FileSelector":
        """Create a FileSelector for fluent file filtering.

        Returns a FileSelector that allows chaining filters on filenames
        and metadata. The resulting selection can be passed to docs(),
        texts(), sents(), etc.

        Returns:
            A new FileSelector instance.

        Example:
            >>> # Select epic poetry by Vergil
            >>> selection = reader.select().where(author="Vergil", genre="epic")
            >>> for doc in reader.docs(selection):
            ...     print(doc._.fileid)

            >>> # Select files by date range
            >>> augustan = reader.select().date_range(-50, 50)
            >>> print(f"Found {len(augustan)} Augustan texts")
        """
        from latincyreaders.core.selector import FileSelector

        return FileSelector(self)

    def _resolve_fileids(
        self, fileids: str | list[str] | "FileSelector" | None
    ) -> list[str]:
        """Resolve fileids argument to a list of file identifiers.

        Args:
            fileids: Single fileid, list of fileids, FileSelector, or None for all files.

        Returns:
            List of file identifiers.
        """
        if fileids is None:
            return self.fileids()
        if isinstance(fileids, str):
            return [fileids]
        # Handle any iterable (including FileSelector)
        return list(fileids)

    def _iter_paths(self, fileids: str | list[str] | None = None) -> Iterator[Path]:
        """Iterate over file paths for the given fileids.

        Args:
            fileids: Files to iterate over.

        Yields:
            Path objects for each file.
        """
        for fid in self._resolve_fileids(fileids):
            yield self._root / fid

    @abstractmethod
    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a single file. Yield (text_chunk, metadata) pairs.

        This is the main extension point for subclasses. Implement this method
        to handle the specific file format of your corpus.

        Args:
            path: Path to the file to parse.

        Yields:
            Tuples of (text, metadata_dict) for each logical unit in the file.
        """
        ...

    # -------------------------------------------------------------------------
    # Core iteration methods
    # -------------------------------------------------------------------------

    def texts(self, fileids: str | list[str] | None = None) -> Iterator[str]:
        """Yield raw text strings. Zero NLP overhead.

        This is the fastest way to iterate over corpus content when you
        don't need any linguistic annotation.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Raw text strings.
        """
        for path in self._iter_paths(fileids):
            for text, _metadata in self._parse_file(path):
                yield self._normalize_text(text)

    def docs(self, fileids: str | list[str] | None = None) -> Iterator["Doc"]:
        """Yield spaCy Doc objects with annotations.

        The level of annotation depends on the reader's annotation_level setting.
        Metadata from JSON files is merged with any metadata from _parse_file().

        When caching is enabled (default), documents are stored after first access
        and returned from cache on subsequent requests for the same fileid.

        Lookup order:
            1. LRU memory cache (instant)
            2. DocBin disk cache — checked against canonical content hash
               *and* the active model generator stamp, so upstream corrections
               and model-version changes both auto-invalidate (fast, ~ms)
            3. Canonical ``.conlluc`` store — parse text, warm DocBin cache
               for next time (~10-100ms)
            4. NLP pipeline from source — write ``.conlluc`` + DocBin (~seconds)

        Args:
            fileids: Files to process, or None for all.

        Yields:
            spaCy Doc objects.
        """
        nlp = self.nlp
        if nlp is None:
            raise ValueError(
                "Cannot create Docs with annotation_level=NONE. "
                "Use texts() for raw strings, or set a higher annotation level."
            )

        # Generator stamp for the active model — written into cache entries and
        # compared on read so a model (version) change can never serve stale.
        model_name, model_version = self._active_generator()
        generator = f"{model_name}@{model_version}"

        for path in self._iter_paths(fileids):
            fileid = str(path.relative_to(self._root))

            # 1. Check LRU memory cache
            if self._cache_enabled and fileid in self._cache:
                self._cache_hits += 1
                self._cache.move_to_end(fileid)
                yield self._cache[fileid]
                continue

            # Compute canonical content hash for staleness detection
            source_hash: str | None = None
            if self._canonical_store is not None:
                source_hash = self._canonical_store.content_hash(fileid)

            # 2. Check disk cache (staleness: canonical content hash + generator)
            old_doc = None
            if self._disk_cache is not None:
                disk_doc = self._disk_cache.get(
                    fileid, nlp.vocab, source_hash=source_hash,
                    generator=generator,
                )
                if disk_doc is not None:
                    self._cache_hits += 1
                    # Base DocBin is silver; attach ids and overlay gold in memory.
                    self._ensure_token_ids(disk_doc)
                    self._overlay_corrections(fileid, disk_doc)
                    if self._cache_enabled:
                        while len(self._cache) >= self._cache_maxsize:
                            self._cache.popitem(last=False)
                        self._cache[fileid] = disk_doc
                    yield disk_doc
                    continue
                # Miss with a stale entry present → capture it for re-pointing.
                old_doc = self._disk_cache.load_raw(fileid, nlp.vocab)

            # 3. Check canonical store (.conlluc)
            if self._canonical_store is not None:
                canonical_doc = self._canonical_store.load(
                    fileid, nlp.vocab,
                    expected_generator=(model_name, model_version),
                )
                if canonical_doc is not None:
                    self._cache_hits += 1
                    self._ensure_token_ids(canonical_doc)

                    # Warm the disk cache (silver) for fast access next time —
                    # before overlaying, so the base DocBin stays uncorrected.
                    if self._disk_cache is not None and source_hash is not None:
                        self._disk_cache.put(
                            fileid, canonical_doc,
                            annotation_level=self._annotation_level.name,
                            model_name=self._model_name,
                            model_version=model_version,
                            generator=generator,
                            source_hash=source_hash,
                        )

                    self._overlay_corrections(fileid, canonical_doc)
                    if self._cache_enabled:
                        while len(self._cache) >= self._cache_maxsize:
                            self._cache.popitem(last=False)
                        self._cache[fileid] = canonical_doc

                    yield canonical_doc
                    continue

            # 4. Cache miss — process the file through NLP pipeline
            if self._cache_enabled:
                self._cache_misses += 1

            json_metadata = self.get_metadata(fileid)

            for text, file_metadata in self._parse_file(path):
                text = self._normalize_text(text)
                doc = nlp(text)
                doc._.fileid = fileid
                doc._.metadata = {**json_metadata, **file_metadata}
                self._ensure_token_ids(doc)

                # Cache the freshly-annotated doc immediately, before any
                # disk/canonical write or repoint/overlay step that could
                # raise — otherwise a failure there would discard NLP work
                # that already succeeded. Later in-place mutations (repoint,
                # overlay) still apply, since this is the same object.
                if self._cache_enabled:
                    while len(self._cache) >= self._cache_maxsize:
                        self._cache.popitem(last=False)
                    self._cache[fileid] = doc

                # Write canonical .conlluc (silver — before overlay)
                if self._canonical_store is not None:
                    self._canonical_store.save(
                        fileid, doc,
                        model_name=self._model_name,
                        model_version=model_version,
                    )
                    source_hash = self._canonical_store.content_hash(fileid)

                # Persist to disk cache (silver; source_hash + generator staleness)
                if self._disk_cache is not None:
                    self._disk_cache.put(
                        fileid, doc,
                        annotation_level=self._annotation_level.name,
                        model_name=self._model_name,
                        model_version=model_version,
                        generator=generator,
                        **({"source_hash": source_hash} if source_hash else {}),
                    )

                # Re-point corrections if this rebuild changed tokenization, then
                # overlay gold in memory (base cache stays silver) and cache.
                self._repoint_on_rebuild(fileid, doc, old_doc)
                old_doc = None
                self._overlay_corrections(fileid, doc)

                yield doc

    def persist_cache(self) -> int:
        """Force-save all in-memory cached documents to disk.

        Requires a ``cache_config`` with ``persist=True``.

        Returns:
            Number of documents persisted.
        """
        if self._disk_cache is None:
            raise ValueError(
                "No disk cache configured. Pass cache_config=CacheConfig(persist=True) "
                "to the reader constructor."
            )

        model_name, model_version = self._active_generator()
        generator = f"{model_name}@{model_version}"
        store = self._get_correction_store()
        count = 0
        for fileid, doc in self._cache.items():
            # Cached docs may carry an in-memory gold overlay (see
            # _overlay_corrections); revert it before writing so the disk
            # base cache never picks up corrected values as if they were
            # raw model output, then reapply so the live LRU entry is
            # unaffected.
            if store is not None:
                store.revert(fileid, doc)
            self._disk_cache.put(
                fileid, doc,
                annotation_level=self._annotation_level.name,
                model_name=self._model_name,
                model_version=model_version,
                generator=generator,
            )
            if store is not None:
                store.overlay(fileid, doc)
            count += 1
        return count

    def warm_cache(self, fileids: str | list[str] | None = None) -> int:
        """Pre-process and cache all (or selected) files.

        Iterates over every file, triggering NLP processing and caching.
        Useful for building a complete disk cache in one pass.

        Returns:
            Number of documents processed.
        """
        count = 0
        for _doc in self.docs(fileids):
            count += 1
        return count

    def build_vectors(
        self,
        config: "SentenceVectorConfig | None" = None,
        fileids: str | list[str] | None = None,
    ) -> "SentenceVectorStore":
        """Build a sentence vector store from this reader's documents.

        Args:
            config: Vector store configuration. If None, uses defaults.
            fileids: Files to include, or None for all.

        Returns:
            The populated SentenceVectorStore.
        """
        from latincyreaders.cache.vectors import SentenceVectorConfig, SentenceVectorStore

        if config is None:
            config = SentenceVectorConfig()
        store = SentenceVectorStore(config)
        store.build(self, fileids)
        return store

    def find_similar(
        self,
        text: str,
        top_k: int = 10,
        config: "SentenceVectorConfig | None" = None,
        auto_build: bool = False,
    ) -> list[dict]:
        """Find sentences similar to query text using stored vectors.

        Args:
            text: Query text.
            top_k: Number of results.
            config: Vector store configuration. If None, uses defaults.
            auto_build: If True, build the vector store automatically when
                no existing store is found. Defaults to False.

        Returns:
            List of result dicts with fileid, citation, text, score.
        """
        from latincyreaders.cache.vectors import SentenceVectorConfig, SentenceVectorStore

        if config is None:
            config = SentenceVectorConfig()

        nlp = self.nlp
        if nlp is None:
            raise ValueError("find_similar requires NLP pipeline for vectorisation.")

        store = SentenceVectorStore(config)

        if store.stats()["sentences"] == 0:
            if auto_build:
                store.build(self)
            else:
                raise ValueError(
                    f"No vector index found for collection "
                    f"{config.collection!r}. Build one first with "
                    f"reader.build_vectors() or pass auto_build=True."
                )

        return store.similar_to_sent(text, nlp, top_k=top_k)

    def sents(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Span | str"]:
        """Yield sentences from documents.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Span objects.

        Yields:
            Sentence Spans (or strings if as_text=True).
        """
        for doc in self.docs(fileids):
            for sent in doc.sents:
                yield sent.text if as_text else sent

    def tokens(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Token | str"]:
        """Yield individual tokens from documents.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Token objects.

        Yields:
            Tokens (or strings if as_text=True).
        """
        for doc in self.docs(fileids):
            for token in doc:
                yield token.text if as_text else token

    # -------------------------------------------------------------------------
    # Text analysis methods
    # -------------------------------------------------------------------------

    def _get_token_citation(self, doc: "Doc", token: "Token", token_idx: int) -> str:
        """Get citation for a token, checking spans if not on token directly.

        Args:
            doc: The document containing the token.
            token: The token to get citation for.
            token_idx: Index of the token in the document.

        Returns:
            Citation string, or fileid:idx fallback.
        """
        # First check token-level citation
        citation = getattr(token._, "citation", None)
        if citation is not None:
            return citation

        # Check if token is within a span that has a citation (e.g., Tesserae lines)
        for span_key in doc.spans:
            for span in doc.spans[span_key]:
                if span.start <= token.i < span.end:
                    span_citation = getattr(span._, "citation", None)
                    if span_citation is not None:
                        return span_citation

        # Fallback to fileid:index
        fileid = doc._.fileid or "unknown"
        return f"{fileid}:{token_idx}"

    def concordance(
        self,
        fileids: str | list[str] | None = None,
        basis: str = "lemma",
        only_alpha: bool = True,
    ) -> dict[str, list[str]]:
        """Build a concordance mapping words to their citation locations.

        A concordance is a dictionary where keys are word forms and values
        are lists of citations/locations where that word appears.

        Args:
            fileids: Files to process, or None for all.
            basis: How to key the concordance:
                - "lemma": group by lemma (default, recommended)
                - "norm": group by normalized form (spaCy's norm_)
                - "text": group by exact surface form
            only_alpha: If True, skip non-alphabetic tokens (punctuation, numbers).

        Returns:
            Dict mapping word form -> list of citation strings.
            Citations are in format "<citation>" if available, else "fileid:token_idx".

        Example:
            >>> conc = reader.concordance(basis="lemma")
            >>> conc["amor"]
            ['<catull. 1.1>', '<catull. 1.3>', '<verg. aen. 4.1>']
        """
        from collections import defaultdict

        concordance_dict: defaultdict[str, list[str]] = defaultdict(list)

        for doc in self.docs(fileids):
            for i, token in enumerate(doc):
                # Skip non-alphabetic tokens if requested
                if only_alpha and not token.is_alpha:
                    continue

                # Determine the key based on basis
                if basis == "lemma":
                    key = token.lemma_
                elif basis == "norm":
                    key = token.norm_
                else:  # "text" or fallback
                    key = token.text

                citation = self._get_token_citation(doc, token, i)
                concordance_dict[key].append(citation)

        # Sort by key and return as regular dict
        return dict(sorted(concordance_dict.items()))

    def kwic(
        self,
        keyword: str,
        fileids: str | list[str] | None = None,
        window: int = 5,
        ignore_case: bool = True,
        by_lemma: bool = False,
        limit: int | None = None,
    ) -> Iterator[dict[str, str]]:
        """Find keyword in context (KWIC) across the corpus.

        Returns matches with surrounding context, useful for studying
        word usage patterns.

        Args:
            keyword: Word to search for.
            fileids: Files to search, or None for all.
            window: Number of tokens on each side for context.
            ignore_case: If True, match case-insensitively.
            by_lemma: If True, match against lemma instead of surface form.
            limit: Maximum number of results to return.

        Yields:
            Dicts with keys:
                - left: left context (string)
                - match: matched token (string)
                - right: right context (string)
                - citation: citation string if available
                - fileid: file identifier

        Example:
            >>> for hit in reader.kwic("amor", window=3, by_lemma=True):
            ...     print(f"{hit['left']} [{hit['match']}] {hit['right']}")
            ...     print(f"  -- {hit['citation']}")
        """
        target = keyword.lower() if ignore_case else keyword
        count = 0

        for doc in self.docs(fileids):
            fileid = doc._.fileid or "unknown"
            tokens = list(doc)

            for i, token in enumerate(tokens):
                # Determine what to match against
                if by_lemma:
                    token_value = token.lemma_.lower() if ignore_case else token.lemma_
                else:
                    token_value = token.text.lower() if ignore_case else token.text

                if token_value == target:
                    # Build context windows
                    left_start = max(0, i - window)
                    right_end = min(len(tokens), i + window + 1)

                    left_tokens = tokens[left_start:i]
                    right_tokens = tokens[i + 1:right_end]

                    left_text = " ".join(t.text for t in left_tokens)
                    right_text = " ".join(t.text for t in right_tokens)

                    citation = self._get_token_citation(doc, token, i)

                    yield {
                        "left": left_text,
                        "match": token.text,
                        "right": right_text,
                        "citation": citation,
                        "fileid": fileid,
                    }

                    count += 1
                    if limit is not None and count >= limit:
                        return

    def ngrams(
        self,
        n: int = 2,
        fileids: str | list[str] | None = None,
        filter_stops: bool = False,
        filter_punct: bool = True,
        filter_nums: bool = False,
        basis: str = "text",
        as_tuples: bool = False,
    ) -> Iterator[str | tuple["Token", ...]]:
        """Extract n-grams from the corpus.

        N-grams are contiguous sequences of n tokens. Useful for
        collocations, frequency analysis, and language modeling.

        Args:
            n: Size of n-grams (2 for bigrams, 3 for trigrams, etc.).
            fileids: Files to process, or None for all.
            filter_stops: If True, exclude n-grams containing stop words.
            filter_punct: If True, exclude n-grams containing punctuation.
            filter_nums: If True, exclude n-grams containing numbers.
            basis: How to represent tokens in output strings:
                - "text": surface form (default) - "amat te"
                - "lemma": lemmatized form - "amo tu"
                - "norm": normalized form (spaCy's norm_)
            as_tuples: If True, yield tuples of Token objects instead of strings.
                When True, basis is ignored.

        Yields:
            N-gram strings like "arma virumque" (default), or tuples of
            Token objects if as_tuples=True.

        Example:
            >>> # Get all bigrams from Catullus
            >>> bigrams = list(reader.ngrams(n=2, fileids="catullus.*"))
            >>> print(bigrams[:5])
            ['Cui dono', 'dono lepidum', 'lepidum novum', ...]

            >>> # Get bigrams by lemma for better frequency analysis
            >>> lemma_bigrams = list(reader.ngrams(n=2, basis="lemma"))
            >>> print(lemma_bigrams[:5])
            ['qui do', 'do lepidus', 'lepidus novus', ...]

            >>> # Get trigrams as token tuples for linguistic analysis
            >>> for gram in reader.ngrams(n=3, as_tuples=True, fileids="catullus.*"):
            ...     print([(t.text, t.pos_) for t in gram])
        """
        import textacy.extract

        for doc in self.docs(fileids):
            ngram_spans = textacy.extract.ngrams(
                doc,
                n=n,
                filter_stops=filter_stops,
                filter_punct=filter_punct,
                filter_nums=filter_nums,
            )

            for span in ngram_spans:
                if as_tuples:
                    yield tuple(token for token in span)
                else:
                    if basis == "lemma":
                        yield " ".join(t.lemma_ for t in span)
                    elif basis == "norm":
                        yield " ".join(t.norm_ for t in span)
                    else:  # "text" or fallback
                        yield span.text

    def skipgrams(
        self,
        n: int = 2,
        k: int = 1,
        fileids: str | list[str] | None = None,
        filter_stops: bool = False,
        filter_punct: bool = True,
        filter_nums: bool = False,
        basis: str = "text",
        as_tuples: bool = False,
    ) -> Iterator[str | tuple["Token", ...]]:
        """Extract skipgrams from the corpus.

        Skipgrams are like n-grams but allow gaps between tokens.
        For example, a (2,1)-skipgram from "the quick brown fox" includes
        both "the quick" and "the brown" (skipping "quick").

        Args:
            n: Number of tokens in each skipgram.
            k: Maximum number of tokens to skip between included tokens.
            fileids: Files to process, or None for all.
            filter_stops: If True, exclude skipgrams containing stop words.
            filter_punct: If True, exclude skipgrams containing punctuation.
            filter_nums: If True, exclude skipgrams containing numbers.
            basis: How to represent tokens in output strings:
                - "text": surface form (default)
                - "lemma": lemmatized form
                - "norm": normalized form (spaCy's norm_)
            as_tuples: If True, yield tuples of Token objects instead of strings.
                When True, basis is ignored.

        Yields:
            Skipgram strings (default), or tuples of Token objects if as_tuples=True.

        Example:
            >>> # Bigrams with 1 skip - captures non-adjacent word pairs
            >>> for sg in reader.skipgrams(n=2, k=1, fileids="catullus.*"):
            ...     print(sg)

            >>> # Skipgrams by lemma
            >>> for sg in reader.skipgrams(n=2, k=1, basis="lemma"):
            ...     print(sg)
        """
        for doc in self.docs(fileids):
            # Filter tokens first
            tokens = [t for t in doc if self._token_passes_filters(
                t, filter_stops, filter_punct, filter_nums
            )]

            for i in range(len(tokens)):
                for skip in range(k + 1):
                    # Build skipgram indices
                    indices = []
                    pos = i
                    for _ in range(n):
                        if pos >= len(tokens):
                            break
                        indices.append(pos)
                        pos += skip + 1

                    if len(indices) == n:
                        gram_tokens = tuple(tokens[idx] for idx in indices)
                        if as_tuples:
                            yield gram_tokens
                        else:
                            if basis == "lemma":
                                yield " ".join(t.lemma_ for t in gram_tokens)
                            elif basis == "norm":
                                yield " ".join(t.norm_ for t in gram_tokens)
                            else:  # "text" or fallback
                                yield " ".join(t.text for t in gram_tokens)

    def _token_passes_filters(
        self,
        token: "Token",
        filter_stops: bool,
        filter_punct: bool,
        filter_nums: bool,
    ) -> bool:
        """Check if a token passes the specified filters."""
        if filter_stops and token.is_stop:
            return False
        if filter_punct and token.is_punct:
            return False
        if filter_nums and token.like_num:
            return False
        return True

    # -------------------------------------------------------------------------
    # Sentence search methods
    # -------------------------------------------------------------------------

    def _get_citation_for_span(self, doc: "Doc", span: "Span") -> str:
        """Get citation for a span (sentence).

        Override in subclasses for format-specific citations.

        Args:
            doc: The document containing the span.
            span: The span to get citation for.

        Returns:
            Citation string.
        """
        # Check if span has a citation attribute
        citation = getattr(span._, "citation", None)
        if citation is not None:
            return citation

        # Check if span overlaps with any citation-bearing spans
        for span_key in doc.spans:
            for labeled_span in doc.spans[span_key]:
                if labeled_span.start <= span.start < labeled_span.end:
                    span_citation = getattr(labeled_span._, "citation", None)
                    if span_citation is not None:
                        return span_citation

        # Fallback to fileid:sent_index
        fileid = doc._.fileid or "unknown"
        sents = list(doc.sents)
        for i, s in enumerate(sents):
            if s.start == span.start:
                return f"{fileid}:sent{i}"
        return f"{fileid}:sent?"

    def find_sents(
        self,
        pattern: str | None = None,
        forms: list[str] | None = None,
        lemma: str | list[str] | None = None,
        matcher_pattern: list[dict] | None = None,
        fileids: str | list[str] | None = None,
        ignore_case: bool = True,
        context: bool = False,
        show_progress: bool = False,
    ) -> Iterator[dict]:
        """Find sentences containing specific words/patterns/lemmas.

        This is the main search method for extracting sentences for annotation.

        Args:
            pattern: Regex pattern to match.
            forms: List of exact word forms to match.
            lemma: Lemma or list of lemmas to match (requires NLP - slower).
            matcher_pattern: spaCy Matcher pattern for advanced queries.
            fileids: Files to search, or None for all.
            ignore_case: Whether to ignore case (default True for pattern/forms).
            context: If True, include surrounding sentences.
            show_progress: If True, show tqdm progress bar for file iteration.

        Yields:
            Dicts with keys: fileid, citation, sentence, matches, (prev_sent, next_sent).

        Example:
            >>> for hit in reader.find_sents(pattern=r"\\bTheb\\w+\\b"):
            ...     print(f"{hit['citation']}: {hit['sentence']}")

            >>> for hit in reader.find_sents(lemma=["bellum", "pax"]):
            ...     print(hit['sentence'])
        """
        if matcher_pattern is not None:
            yield from self._find_sents_by_matcher(matcher_pattern, fileids, context, show_progress)
        elif lemma is not None:
            lemmas = [lemma] if isinstance(lemma, str) else lemma
            yield from self._find_sents_by_lemma(lemmas, fileids, context, show_progress)
        else:
            yield from self._find_sents_by_pattern(pattern, forms, fileids, ignore_case, context, show_progress)

    @staticmethod
    def _normalize_sent_text(text: str) -> str:
        """Normalize sentence text by replacing newlines with spaces."""
        # Replace \r\n, \r, \n with space, then collapse multiple spaces
        return " ".join(text.split())

    def _find_sents_by_pattern(
        self,
        pattern: str | None,
        forms: list[str] | None,
        fileids: str | list[str] | None,
        ignore_case: bool,
        context: bool,
        show_progress: bool = False,
    ) -> Iterator[dict]:
        """Find sentences by regex pattern (fast path)."""
        if pattern is None and forms is None:
            raise ValueError("Must provide either pattern or forms")

        if forms is not None:
            escaped = [re.escape(f) for f in forms]
            pattern = r"\b(" + "|".join(escaped) + r")\b"

        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)

        # Get fileids list for progress bar
        if show_progress:
            fids = self._resolve_fileids(fileids)
            doc_iter = tqdm(self.docs(fids), total=len(fids), desc="Files", unit="file")
        else:
            doc_iter = self.docs(fileids)

        for doc in doc_iter:
            sents = list(doc.sents)
            for i, sent in enumerate(sents):
                matches = regex.findall(sent.text)
                if matches:
                    result = {
                        "fileid": doc._.fileid,
                        "citation": self._get_citation_for_span(doc, sent),
                        "sentence": self._normalize_sent_text(sent.text),
                        "matches": matches,
                    }
                    if context:
                        result["prev_sent"] = self._normalize_sent_text(sents[i - 1].text) if i > 0 else None
                        result["next_sent"] = self._normalize_sent_text(sents[i + 1].text) if i < len(sents) - 1 else None
                    yield result

    def _find_sents_by_lemma(
        self,
        lemmas: list[str],
        fileids: str | list[str] | None,
        context: bool,
        show_progress: bool = False,
    ) -> Iterator[dict]:
        """Find sentences by lemma(s) (uses NLP)."""
        target_lemmas = {lem.lower() for lem in lemmas}

        # Get fileids list for progress bar
        if show_progress:
            fids = self._resolve_fileids(fileids)
            doc_iter = tqdm(self.docs(fids), total=len(fids), desc="Files", unit="file")
        else:
            doc_iter = self.docs(fileids)

        for doc in doc_iter:
            sents = list(doc.sents)
            for i, sent in enumerate(sents):
                matches = [t.text for t in sent if t.lemma_.lower() in target_lemmas]
                if matches:
                    matched_lemmas = [
                        t.lemma_.lower() for t in sent if t.lemma_.lower() in target_lemmas
                    ]
                    result = {
                        "fileid": doc._.fileid,
                        "citation": self._get_citation_for_span(doc, sent),
                        "sentence": self._normalize_sent_text(sent.text),
                        "matches": matches,
                        "lemmas": list(set(matched_lemmas)),
                    }
                    if context:
                        result["prev_sent"] = self._normalize_sent_text(sents[i - 1].text) if i > 0 else None
                        result["next_sent"] = self._normalize_sent_text(sents[i + 1].text) if i < len(sents) - 1 else None
                    yield result

    def _find_sents_by_matcher(
        self,
        matcher_pattern: list[dict],
        fileids: str | list[str] | None,
        context: bool,
        show_progress: bool = False,
    ) -> Iterator[dict]:
        """Find sentences using spaCy Matcher patterns."""
        from spacy.matcher import Matcher

        nlp = self.nlp
        if nlp is None:
            raise ValueError("Matcher patterns require NLP pipeline")

        matcher = Matcher(nlp.vocab)
        matcher.add("PATTERN", [matcher_pattern])

        # Get fileids list for progress bar
        if show_progress:
            fids = self._resolve_fileids(fileids)
            doc_iter = tqdm(self.docs(fids), total=len(fids), desc="Files", unit="file")
        else:
            doc_iter = self.docs(fileids)

        for doc in doc_iter:
            sents = list(doc.sents)
            matches = matcher(doc)

            matched_sents: dict[int, list[str]] = {}
            for _, start, end in matches:
                match_span = doc[start:end]
                for i, sent in enumerate(sents):
                    if sent.start <= start < sent.end:
                        if i not in matched_sents:
                            matched_sents[i] = []
                        matched_sents[i].append(match_span.text)
                        break

            for i, match_texts in matched_sents.items():
                sent = sents[i]
                result = {
                    "fileid": doc._.fileid,
                    "citation": self._get_citation_for_span(doc, sent),
                    "sentence": self._normalize_sent_text(sent.text),
                    "matches": match_texts,
                }
                if context:
                    result["prev_sent"] = self._normalize_sent_text(sents[i - 1].text) if i > 0 else None
                    result["next_sent"] = self._normalize_sent_text(sents[i + 1].text) if i < len(sents) - 1 else None
                yield result
