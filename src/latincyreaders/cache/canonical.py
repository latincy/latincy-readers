"""Canonical annotation store for known collections.

Stores pre-computed, community-corrected annotations that can be shared
and version-controlled. Uses the ``.conlluc`` (CoNLL-U Cache) format —
a human-readable, silver-standard format that is clearly marked as
machine-generated and not suitable for model training.

Canonical annotations live in a directory structure::

    {store_root}/{collection}/
        manifest.json        — fileid map, version, contributor info
        {hash}.conlluc       — CoNLL-U Cache files (silver annotations)

They can be exported and imported for sharing via git repositories.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copytree
from typing import Any

from spacy.tokens import Doc
from spacy.vocab import Vocab

from latincyreaders.cache.conlluc import (
    CONLLUC_EXTENSION,
    doc_to_conlluc,
    read_conlluc,
    validate_conlluc_header,
    write_conlluc,
)
from latincyreaders.cache.disk import _fileid_hash


# Default canonical data root
_DEFAULT_STORE_ROOT = Path.home() / "latincy_data" / "canonical"


@dataclass
class CanonicalConfig:
    """Configuration for canonical annotation loading.

    Attributes:
        store_root: Root directory of canonical annotation stores.
        collection: Collection identifier (e.g. ``"cltk-tesserae"``).
        prefer_canonical: When True, prefer canonical over dynamic annotations.
        auto_download: Download canonical annotations if not present.
    """

    store_root: Path = field(default_factory=lambda: _DEFAULT_STORE_ROOT)
    collection: str = "cltk-tesserae"
    prefer_canonical: bool = True
    auto_download: bool = False

    def __post_init__(self) -> None:
        self.store_root = Path(self.store_root)


class CanonicalAnnotationStore:
    """Store and retrieve canonical (expert/community) annotations.

    Canonical annotations are pre-computed and version-controlled.
    They serve as a gold-standard reference that can be community-corrected
    upstream, similar to how UDReader loads gold-standard CoNLL-U data.

    Annotations are stored in ``.conlluc`` format — human-readable CoNLL-U
    with file-level metadata marking them as silver-standard.

    Example::

        >>> store = CanonicalAnnotationStore(
        ...     CanonicalConfig(collection="cltk-tesserae")
        ... )
        >>> # Save canonical annotations
        >>> store.save("vergil.aen.tess", doc, model_name="la_core_web_lg")
        >>> # Load them later
        >>> doc = store.load("vergil.aen.tess", nlp.vocab)
    """

    def __init__(self, config: CanonicalConfig) -> None:
        self._config = config
        self._dir = config.store_root / config.collection
        self._manifest_path = self._dir / "manifest.json"
        self._manifest: dict[str, Any] | None = None

    @property
    def collection(self) -> str:
        return self._config.collection

    @property
    def store_dir(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def load(self, fileid: str, vocab: Vocab) -> Doc | None:
        """Load canonical annotations for a fileid.

        Returns None if no canonical annotation exists.
        """
        manifest = self._load_manifest()
        files = manifest.get("files", {})
        entry = files.get(fileid)
        if entry is None:
            return None

        path = self._dir / entry["filename"]
        if not path.exists():
            return None

        doc, _meta = read_conlluc(path, vocab)
        return doc

    def has(self, fileid: str) -> bool:
        """Check if canonical annotations exist for a fileid."""
        manifest = self._load_manifest()
        files = manifest.get("files", {})
        entry = files.get(fileid)
        if entry is None:
            return False
        return (self._dir / entry["filename"]).exists()

    def save(self, fileid: str, doc: Doc, **extra_meta: Any) -> None:
        """Write canonical annotations for a fileid.

        Args:
            fileid: File identifier.
            doc: Annotated spaCy Doc.
            **extra_meta: Extra metadata (e.g. ``model_name``, ``annotation_level``).
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        h = _fileid_hash(fileid)
        filename = f"{h}{CONLLUC_EXTENSION}"
        path = self._dir / filename

        content = doc_to_conlluc(
            doc,
            fileid=fileid,
            collection=self._config.collection,
            model_name=str(extra_meta.get("model_name", "")),
            model_version=str(extra_meta.get("model_version", "")),
            corrections=int(extra_meta.get("corrections", 0)),
        )
        write_conlluc(path, content)

        manifest = self._load_manifest()
        files = manifest.setdefault("files", {})
        files[fileid] = {
            "filename": filename,
            "hash": h,
            "timestamp": time.time(),
            **extra_meta,
        }
        self._save_manifest(manifest)

    def remove(self, fileid: str) -> None:
        """Remove canonical annotations for a fileid."""
        manifest = self._load_manifest()
        files = manifest.get("files", {})
        entry = files.pop(fileid, None)
        if entry is not None:
            path = self._dir / entry["filename"]
            path.unlink(missing_ok=True)
            self._save_manifest(manifest)

    def fileids(self) -> list[str]:
        """Return list of fileids with canonical annotations."""
        manifest = self._load_manifest()
        return list(manifest.get("files", {}).keys())

    # ------------------------------------------------------------------
    # Collection-level operations
    # ------------------------------------------------------------------

    def build_from_reader(
        self,
        reader: Any,
        fileids: list[str] | None = None,
        model_name: str | None = None,
    ) -> int:
        """Build canonical annotations from a reader.

        Processes all (or selected) files through the reader's NLP pipeline
        and saves the results as canonical annotations.

        Returns the number of documents saved.
        """
        count = 0
        fids = fileids if fileids is not None else reader.fileids()
        for doc in reader.docs(fids):
            fid = doc._.fileid
            if fid:
                extra = {}
                if model_name:
                    extra["model_name"] = model_name
                self.save(fid, doc, **extra)
                count += 1
        return count

    def diff(self, fileid: str, doc: Doc) -> list[dict[str, Any]]:
        """Compare canonical annotations against a dynamically-annotated Doc.

        Returns a list of token-level differences. Each diff entry contains
        the token index, text, and the fields that differ.
        """
        vocab = doc.vocab
        canonical_doc = self.load(fileid, vocab)
        if canonical_doc is None:
            return [{"error": f"No canonical annotation for {fileid}"}]

        diffs: list[dict[str, Any]] = []
        max_len = min(len(canonical_doc), len(doc))

        for i in range(max_len):
            ct = canonical_doc[i]
            dt = doc[i]
            token_diff: dict[str, Any] = {}

            if ct.text != dt.text:
                token_diff["text"] = {"canonical": ct.text, "dynamic": dt.text}
            if ct.lemma_ != dt.lemma_:
                token_diff["lemma"] = {"canonical": ct.lemma_, "dynamic": dt.lemma_}
            if ct.pos_ != dt.pos_:
                token_diff["pos"] = {"canonical": ct.pos_, "dynamic": dt.pos_}
            if ct.tag_ != dt.tag_:
                token_diff["tag"] = {"canonical": ct.tag_, "dynamic": dt.tag_}
            if ct.dep_ != dt.dep_:
                token_diff["dep"] = {"canonical": ct.dep_, "dynamic": dt.dep_}

            if token_diff:
                token_diff["index"] = i
                token_diff["token"] = dt.text
                diffs.append(token_diff)

        if len(canonical_doc) != len(doc):
            diffs.append({
                "length_mismatch": {
                    "canonical": len(canonical_doc),
                    "dynamic": len(doc),
                }
            })

        return diffs

    def export_collection(self, output_dir: str | Path) -> None:
        """Export the entire collection to a directory for sharing."""
        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        copytree(self._dir, output_dir)

    def import_collection(self, source_dir: str | Path) -> int:
        """Import canonical annotations from a shared source directory.

        Returns the number of files imported.
        """
        source_dir = Path(source_dir)
        source_manifest_path = source_dir / "manifest.json"
        if not source_manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in {source_dir}")

        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        source_files = source_manifest.get("files", {})

        self._dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        files = manifest.setdefault("files", {})

        count = 0
        for fileid, entry in source_files.items():
            src_path = source_dir / entry["filename"]
            if src_path.exists():
                dst_path = self._dir / entry["filename"]
                dst_path.write_bytes(src_path.read_bytes())
                files[fileid] = entry
                count += 1

        self._save_manifest(manifest)
        return count

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        manifest = self._load_manifest()
        files = manifest.get("files", {})
        total_bytes = 0
        for entry in files.values():
            path = self._dir / entry["filename"]
            if path.exists():
                total_bytes += path.stat().st_size

        return {
            "collection": self._config.collection,
            "entries": len(files),
            "size_bytes": total_bytes,
            "store_dir": str(self._dir),
            "version": manifest.get("version"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
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

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest = manifest
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
