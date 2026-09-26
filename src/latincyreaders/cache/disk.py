"""Persistent disk cache for spaCy Doc annotations.

Serializes annotated Doc objects to disk using spaCy's DocBin format,
providing fast load times and compact storage. Complements the in-memory
LRU cache in BaseCorpusReader.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab

# Default cache root
_DEFAULT_CACHE_DIR = Path.home() / ".latincy_cache"


def _sanitize_user_data(doc: Doc) -> None:
    """Convert non-serializable user_data values to strings.

    LatinCy models store MorphAnalysis objects in custom extensions
    (e.g. ``remorph``). These aren't msgpack-serializable, so we
    convert them to their string representation before DocBin export.
    """
    from spacy.tokens.morphanalysis import MorphAnalysis

    for key, val in doc.user_data.items():
        if isinstance(val, MorphAnalysis):
            doc.user_data[key] = str(val)


# Custom Token extensions preserved through DocBin, which otherwise drops custom
# token attributes. Excludes read-time/transient markers (e.g. ``corrected``): the
# base cache is always silver, so correction state never enters it.
_STASHED_TOKEN_EXTS = (
    "remorph",
    "token_id",
    "ud",
    "citation",
    "speaker",
    "is_quote",
    "is_crux",
    "is_addition",
    "is_expansion",
    "is_supplement",
    "newline_after",
)


def _stash_token_extensions(doc: Doc) -> None:
    """Stash custom Token extension values into ``doc.user_data`` for DocBin.

    DocBin serializes ``doc.user_data`` (``store_user_data=True``) but drops custom
    *token* extensions, so we save the non-default values of the known data
    extensions (:data:`_STASHED_TOKEN_EXTS`) sparsely, keyed by token index, and
    restore them after load. This preserves rich annotations — durable ids,
    remorph, text-critical flags, verse/newline markers — across the round-trip.
    """
    from spacy.tokens import Token

    payload: dict[str, dict[str, Any]] = {}
    for name in _STASHED_TOKEN_EXTS:
        if not Token.has_extension(name):
            continue
        sparse: dict[str, Any] = {}
        for token in doc:
            val = getattr(token._, name)
            if val is None or val is False or val == "":
                continue
            sparse[str(token.i)] = val
        if sparse:
            payload[name] = sparse
    if payload:
        doc.user_data["_tok_exts"] = payload


def _restore_token_extensions(doc: Doc) -> None:
    """Restore custom Token extension values stashed by _stash_token_extensions."""
    from spacy.tokens import Token

    payload = doc.user_data.pop("_tok_exts", None)
    # Legacy sparse keys from earlier cache entries.
    legacy_remorph = doc.user_data.pop("_remorph", None)
    legacy_ids = doc.user_data.pop("_token_ids", None)

    if payload:
        for name, sparse in payload.items():
            if not Token.has_extension(name):
                Token.set_extension(name, default=None)
            for idx_str, val in sparse.items():
                idx = int(idx_str)
                if idx < len(doc):
                    setattr(doc[idx]._, name, val)

    if legacy_remorph:
        if not Token.has_extension("remorph"):
            Token.set_extension("remorph", default=None)
        for idx_str, val in legacy_remorph.items():
            idx = int(idx_str)
            if idx < len(doc):
                doc[idx]._.remorph = val
    if legacy_ids:
        if not Token.has_extension("token_id"):
            Token.set_extension("token_id", default=None)
        for idx, val in enumerate(legacy_ids):
            if idx < len(doc) and val is not None:
                doc[idx]._.token_id = val


@dataclass
class CacheConfig:
    """Configuration for persistent disk caching.

    Attributes:
        cache_dir: Root directory for cached annotations.
        persist: Whether to persist annotations to disk.
        ttl_days: Time-to-live in days. None means never expires.
        collection: Collection name for organising cache entries.
    """

    cache_dir: Path = field(default_factory=lambda: _DEFAULT_CACHE_DIR)
    persist: bool = False
    ttl_days: int | None = None
    collection: str | None = None

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)


def _fileid_hash(fileid: str) -> str:
    """Deterministic short hash for a fileid."""
    return hashlib.sha256(fileid.encode()).hexdigest()[:16]


class DiskCache:
    """Persistent disk cache for spaCy Doc objects.

    Uses spaCy's ``DocBin`` for serialisation — compact, fast, and
    preserves all token-level annotations plus custom extension data
    stored via ``Doc._.``.

    Directory layout::

        {cache_dir}/{collection}/{hash}.spacy   — serialised DocBin
        {cache_dir}/{collection}/manifest.json  — fileid→hash map + metadata

    Example::

        >>> from latincyreaders.cache.disk import DiskCache, CacheConfig
        >>> cfg = CacheConfig(persist=True, collection="tesserae")
        >>> cache = DiskCache(cfg)
        >>> cache.put("vergil.aen.tess", doc)
        >>> loaded = cache.get("vergil.aen.tess", nlp.vocab)
    """

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._collection = config.collection or "_default"
        self._dir = config.cache_dir / self._collection
        self._manifest_path = self._dir / "manifest.json"
        self._manifest: dict[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        fileid: str,
        vocab: Vocab,
        source_hash: str | None = None,
        generator: str | None = None,
    ) -> Doc | None:
        """Load a cached Doc from disk.

        Returns ``None`` if the entry does not exist, has expired, or is
        stale relative to *source_hash* or *generator*.

        Args:
            fileid: File identifier.
            vocab: spaCy Vocab for Doc reconstruction.
            source_hash: If provided, the cache entry is considered stale
                when its stored ``source_hash`` differs.  This enables
                upstream correction detection: when a ``.conlluc`` file
                changes, its content hash changes, and the DocBin cache
                auto-invalidates.
            generator: If provided (e.g. ``"la_core_web_lg@3.9.6"``), the
                entry is considered stale when its stored ``generator``
                differs.  This makes the DocBin cache a *stamp-keyed
                ephemeral* layer: it can never serve annotations produced by
                a different model (version), so a model upgrade is a
                self-healing miss+rebuild rather than a silent-stale read.
                A legacy entry with no stored ``generator`` counts as a
                mismatch and is rebuilt once.
        """
        if not self._config.persist:
            return None

        manifest = self._load_manifest()
        entry = manifest.get(fileid)
        if entry is None:
            return None

        if self._is_expired(entry):
            return None

        # Staleness check against upstream source (e.g. .conlluc content hash)
        if source_hash is not None:
            stored_hash = entry.get("source_hash")
            if stored_hash != source_hash:
                return None

        # Staleness check against the annotation model generator stamp.
        # A missing stored generator (legacy entry) never matches → rebuild.
        if generator is not None:
            stored_generator = entry.get("generator")
            if stored_generator != generator:
                return None

        path = self._dir / entry["filename"]
        if not path.exists():
            return None

        doc_bin = DocBin().from_disk(path)
        docs = list(doc_bin.get_docs(vocab))
        if not docs:
            return None
        doc = docs[0]
        _restore_token_extensions(doc)
        return doc

    def load_raw(self, fileid: str, vocab: Vocab) -> Doc | None:
        """Load a cached Doc **ignoring** staleness (generator/hash/TTL).

        Used at rebuild time to recover the *pre-rebuild* tokenization (with its
        durable token ids) so corrections can be re-pointed onto the new one.
        Returns ``None`` if no entry/file exists.
        """
        if not self._config.persist:
            return None
        manifest = self._load_manifest()
        entry = manifest.get(fileid)
        if entry is None:
            return None
        path = self._dir / entry["filename"]
        if not path.exists():
            return None
        doc_bin = DocBin().from_disk(path)
        docs = list(doc_bin.get_docs(vocab))
        if not docs:
            return None
        doc = docs[0]
        _restore_token_extensions(doc)
        return doc

    def put(self, fileid: str, doc: Doc, **extra_meta: Any) -> None:
        """Persist a Doc to disk.

        Args:
            fileid: Identifier for the cached document.
            doc: spaCy Doc to serialise.
            **extra_meta: Additional metadata to store in the manifest
                (e.g. ``annotation_level``, ``model_name``, ``model_version``,
                ``generator``, ``source_hash``).  Store ``generator`` here to
                make later ``get(..., generator=...)`` calls stamp-aware.
        """
        if not self._config.persist:
            return

        self._dir.mkdir(parents=True, exist_ok=True)

        h = _fileid_hash(fileid)
        filename = f"{h}.spacy"
        path = self._dir / filename

        doc_bin = DocBin(store_user_data=True)
        _stash_token_extensions(doc)
        _sanitize_user_data(doc)
        doc_bin.add(doc)
        doc_bin.to_disk(path)

        manifest = self._load_manifest()
        manifest[fileid] = {
            "filename": filename,
            "hash": h,
            "timestamp": time.time(),
            **extra_meta,
        }
        self._save_manifest(manifest)

    def has(self, fileid: str) -> bool:
        """Check if a valid (non-expired) entry exists."""
        manifest = self._load_manifest()
        entry = manifest.get(fileid)
        if entry is None:
            return False
        if self._is_expired(entry):
            return False
        path = self._dir / entry["filename"]
        return path.exists()

    def invalidate(self, fileid: str) -> None:
        """Remove a specific cache entry."""
        manifest = self._load_manifest()
        entry = manifest.pop(fileid, None)
        if entry is not None:
            path = self._dir / entry["filename"]
            path.unlink(missing_ok=True)
            self._save_manifest(manifest)

    def clear(self) -> None:
        """Remove all entries for this collection."""
        manifest = self._load_manifest()
        for entry in manifest.values():
            path = self._dir / entry["filename"]
            path.unlink(missing_ok=True)
        manifest.clear()
        self._save_manifest(manifest)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        manifest = self._load_manifest()
        total_bytes = 0
        for entry in manifest.values():
            path = self._dir / entry["filename"]
            if path.exists():
                total_bytes += path.stat().st_size

        return {
            "collection": self._collection,
            "entries": len(manifest),
            "size_bytes": total_bytes,
            "cache_dir": str(self._dir),
        }

    def refresh_check(self, fileid: str) -> bool:
        """Return True if the entry is stale and should be refreshed."""
        manifest = self._load_manifest()
        entry = manifest.get(fileid)
        if entry is None:
            return True
        return self._is_expired(entry)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        if self._config.ttl_days is None:
            return False
        ts = entry.get("timestamp", 0)
        age_days = (time.time() - ts) / 86400
        return age_days > self._config.ttl_days

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        if self._manifest is not None:
            return self._manifest

        if self._manifest_path.exists():
            try:
                self._manifest = json.loads(
                    self._manifest_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                self._manifest = {}
        else:
            self._manifest = {}
        return self._manifest

    def _save_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest = manifest
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
