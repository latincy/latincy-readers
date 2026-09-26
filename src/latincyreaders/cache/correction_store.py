"""Gold correction layer — a small JSON overlay on the DocBin base cache.

NLP annotations contain errors. This store lets users *lock in* corrections that
(a) make the cached annotations trustworthy for project work and (b) are a
training-ready record of human judgements. Corrections are kept **separate from
the regenerable DocBin cache** so they survive ``clear_cache()`` and full
re-annotation — "protect corrections like source."

Design
------
- **Keyed by durable opaque token id** (``Token._.token_id``, e.g. ``t0042``), the
  join key to the DocBin base cache. One flat record per ``(token_id, field)``.
- **Coexist, never overwrite.** A correction is overlaid onto the *in-memory* Doc
  at read time (surfacing the gold value); the base DocBin bytes are never
  mutated, and the record keeps the machine value it overrode in ``was``.
- **Self-anchoring.** Each record carries its enclosing sentence's token forms
  (captured for free at correction time). When tokenization drifts, corrections
  re-point via :mod:`latincyreaders.cache.migrate`: against the old DocBin when it
  is present (fast path), else against the stored sentence context (fallback) —
  the same aligner either way. A correction whose word-content genuinely vanished
  is *quarantined* to a log for manual re-anchoring, never silently mis-applied.

Layout::

    ~/latincy_data/corrections/<collection>/
        manifest.json
        <fileid>.corr.json
        corrections_unresolved.log
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from spacy.tokens import Doc

from latincyreaders.cache.migrate import TokenRef, align_section, repoint_correction

# Scalar CoNLL-U fields a correction can set on a token in this increment.
# (Structural head/dep edits are a follow-up — see the design doc.)
CORRECTABLE_FIELDS = ("lemma", "upos", "xpos", "feats", "deprel")

_DEFAULT_STORE_ROOT = Path.home() / "latincy_data" / "corrections"


def _today() -> str:
    return datetime.date.today().isoformat()


def _fileid_to_filename(fileid: str, suffix: str) -> str:
    """Flatten path separators into a safe single filename."""
    return fileid.replace("/", "--").replace("\\", "--") + suffix


@dataclass
class CorrectionRecord:
    """One human correction of a single field on a single token.

    Attributes:
        token_id: Durable opaque id of the target token at authoring time.
        form: Surface form of the token (validation on re-apply).
        field: One of :data:`CORRECTABLE_FIELDS`.
        value: The corrected (gold) value.
        was: The machine value this overrode (provenance / training signal).
        agent: ``"human"`` for a hand correction.
        evidence: Free-text justification (optional).
        created: ISO date.
        ctx: Self-anchor context — ``{"sent": int, "forms": [str, ...], "i": int}``
            (the enclosing sentence's forms and the target's index within it).
    """

    token_id: str
    form: str
    field: str
    value: str
    was: str = ""
    agent: str = "human"
    evidence: str = ""
    created: str = field(default_factory=_today)
    ctx: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionSet:
    """All corrections for one fileid."""

    fileid: str
    generator: str = ""  # model the corrections were authored against
    corrections: list[CorrectionRecord] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.corrections)


class CorrectionStore:
    """Load, save, apply, and re-point gold corrections for a collection."""

    def __init__(
        self,
        collection: str,
        store_root: Path | str = _DEFAULT_STORE_ROOT,
    ) -> None:
        self._collection = collection
        self._dir = Path(store_root) / collection
        self._manifest_path = self._dir / "manifest.json"
        self._unresolved_log = self._dir / "corrections_unresolved.log"

    @property
    def store_dir(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _path(self, fileid: str) -> Path:
        return self._dir / _fileid_to_filename(fileid, ".corr.json")

    def load(self, fileid: str) -> CorrectionSet | None:
        """Load corrections for *fileid*, or None if none exist."""
        path = self._path(fileid)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        records = [CorrectionRecord(**r) for r in data.get("corrections", [])]
        return CorrectionSet(
            fileid=data.get("fileid", fileid),
            generator=data.get("generator", ""),
            corrections=records,
        )

    def save(self, cset: CorrectionSet) -> Path:
        """Persist a correction set and register it in the manifest."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(cset.fileid)
        path.write_text(
            json.dumps(
                {
                    "fileid": cset.fileid,
                    "generator": cset.generator,
                    "corrections": [asdict(r) for r in cset.corrections],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        manifest = self._load_manifest()
        manifest.setdefault("files", {})[cset.fileid] = {"filename": path.name}
        self._save_manifest(manifest)
        return path

    def fileids(self) -> list[str]:
        return list(self._load_manifest().get("files", {}).keys())

    # ------------------------------------------------------------------
    # Recording a correction
    # ------------------------------------------------------------------

    def record(
        self,
        fileid: str,
        doc: Doc,
        token_id: str,
        field_name: str,
        value: str,
        *,
        evidence: str = "",
        generator: str = "",
    ) -> CorrectionRecord:
        """Record a correction against a token in *doc* and persist it.

        Captures the token's current value (``was``) and its enclosing-sentence
        context for self-anchoring. Replaces any existing record for the same
        ``(token_id, field)``.
        """
        if field_name not in CORRECTABLE_FIELDS:
            raise ValueError(
                f"field {field_name!r} not correctable; "
                f"expected one of {CORRECTABLE_FIELDS}"
            )
        token = self._find_token(doc, token_id)
        if token is None:
            raise KeyError(f"no token {token_id!r} in doc for {fileid!r}")

        record = CorrectionRecord(
            token_id=token_id,
            form=token.text,
            field=field_name,
            value=value,
            was=_read_field(token, field_name),
            evidence=evidence,
            ctx=self._sentence_ctx(doc, token),
        )

        cset = self.load(fileid) or CorrectionSet(fileid=fileid, generator=generator)
        if generator:
            cset.generator = generator
        cset.corrections = [
            r for r in cset.corrections
            if not (r.token_id == token_id and r.field == field_name)
        ]
        cset.corrections.append(record)
        self.save(cset)
        return record

    # ------------------------------------------------------------------
    # Overlay (read-time application)
    # ------------------------------------------------------------------

    def overlay(self, fileid: str, doc: Doc) -> int:
        """Apply corrections for *fileid* onto *doc* in memory. Returns count.

        Coexist semantics: the gold value is surfaced on the Doc and the token is
        flagged ``Token._.corrected``; the base DocBin is never touched. Records
        whose token id is not present (stale, awaiting re-point) are skipped.
        """
        cset = self.load(fileid)
        if cset is None or not cset.corrections:
            return 0
        by_id = {t._.token_id: t for t in doc if t._.token_id is not None}
        applied = 0
        for rec in cset.corrections:
            token = by_id.get(rec.token_id)
            if token is None or token.text != rec.form:
                continue
            if _apply_field(token, rec.field, rec.value):
                token._.corrected = True
                applied += 1
        return applied

    # ------------------------------------------------------------------
    # Re-pointing across tokenization drift
    # ------------------------------------------------------------------

    def repoint(
        self,
        fileid: str,
        new_doc: Doc,
        old_doc: Doc | None = None,
        to_generator: str = "",
    ) -> tuple[int, int]:
        """Re-point corrections onto *new_doc*'s tokenization. Returns (ok, quar).

        Fast path: align the *old_doc* (from the pre-rebuild DocBin, carrying the
        old ids) against *new_doc*. Fallback: when *old_doc* is None, align each
        record's stored sentence context against *new_doc*. Survivors get the new
        token id; failures are logged to ``corrections_unresolved.log`` and dropped
        from the active set.

        When *to_generator* is given, the set is re-stamped to it and the
        migration is recorded in the ledger (``migrations.jsonl`` + ``head.json``)
        for an auditable ``from → to`` drift trail.
        """
        cset = self.load(fileid)
        if cset is None or not cset.corrections:
            return (0, 0)

        from_generator = cset.generator
        kept: list[CorrectionRecord] = []
        quarantined = 0
        alignment = None
        if old_doc is not None:
            alignment = align_section(
                fileid, _refs(old_doc), _refs(new_doc)
            )

        for rec in cset.corrections:
            algn = (
                alignment if alignment is not None
                else self._ctx_alignment(rec, new_doc)
            )
            result = repoint_correction(
                {"target": rec.token_id, "form": rec.form}, algn
            ) if algn is not None else None
            if result is not None and result.status == "clean":
                rec.token_id = result.correction["target"]
                # refresh the anchor context against the new tokenization
                token = self._find_token(new_doc, rec.token_id)
                if token is not None:
                    rec.form = token.text
                    rec.ctx = self._sentence_ctx(new_doc, token)
                kept.append(rec)
            else:
                reason = result.reason if result is not None else "no anchor context"
                self._log_unresolved(rec, reason)
                quarantined += 1

        cset.corrections = kept
        if to_generator:
            cset.generator = to_generator
            self._record_migration(
                fileid, from_generator, to_generator, len(kept), quarantined,
            )
        self.save(cset)
        return (len(kept), quarantined)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_token(doc: Doc, token_id: str):
        for token in doc:
            if token._.token_id == token_id:
                return token
        return None

    @staticmethod
    def _sentence_ctx(doc: Doc, token) -> dict[str, Any]:
        """Capture the target's enclosing-sentence forms + local index."""
        try:
            sent = token.sent
        except (ValueError, AttributeError):
            sent = doc[:]
        forms = [t.text for t in sent]
        return {"forms": forms, "i": token.i - sent.start}

    def _ctx_alignment(self, rec: CorrectionRecord, new_doc: Doc):
        """Build an alignment from a record's stored sentence context vs new_doc.

        The stored sentence carries the record's own ``token_id`` at index ``i``
        (synthetic ids elsewhere); aligning it against the whole new doc lets the
        target re-point by content even with the old DocBin gone.
        """
        forms = rec.ctx.get("forms")
        i = rec.ctx.get("i")
        if not forms or i is None or i >= len(forms):
            return None
        old_refs = [
            TokenRef(rec.token_id if k == i else f"_ctx{k}", f)
            for k, f in enumerate(forms)
        ]
        return align_section(rec.token_id, old_refs, _refs(new_doc))

    # ------------------------------------------------------------------
    # Migration ledger (auditable drift trail)
    # ------------------------------------------------------------------

    def _record_migration(
        self,
        fileid: str,
        from_generator: str,
        to_generator: str,
        repointed: int,
        quarantined: int,
    ) -> None:
        """Append a revision to ``migrations.jsonl`` and advance ``head.json``."""
        self._dir.mkdir(parents=True, exist_ok=True)
        head = self._load_head()
        revision = int(head.get("head", 0)) + 1
        record = {
            "revision": revision,
            "fileid": fileid,
            "from_generator": from_generator,
            "to_generator": to_generator,
            "repointed": repointed,
            "quarantined": quarantined,
            "created": _today(),
        }
        with (self._dir / "migrations.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        (self._dir / "head.json").write_text(
            json.dumps(
                {"head": revision, "applied": _today(), "generator": to_generator},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _load_head(self) -> dict[str, Any]:
        path = self._dir / "head.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def migrations(self) -> list[dict[str, Any]]:
        """Return the migration revisions recorded for this collection, in order."""
        path = self._dir / "migrations.jsonl"
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def _log_unresolved(self, rec: CorrectionRecord, reason: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        line = (
            f"[{_today()}] UNRESOLVED: token_id={rec.token_id} form={rec.form!r} "
            f"field={rec.field} value={rec.value!r} :: {reason}\n"
        )
        with self._unresolved_log.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Field read/write on a spaCy token
# ---------------------------------------------------------------------------

def _refs(doc: Doc) -> list[TokenRef]:
    return [TokenRef(t._.token_id or f"t{t.i:04d}", t.text) for t in doc]


def _read_field(token, field_name: str) -> str:
    if field_name == "lemma":
        return token.lemma_
    if field_name == "upos":
        return token.pos_
    if field_name == "xpos":
        return token.tag_
    if field_name == "feats":
        return str(token.morph)
    if field_name == "deprel":
        return token.dep_
    return ""


def _apply_field(token, field_name: str, value: str) -> bool:
    """Set *field_name* to *value* on *token*. Returns False if it couldn't apply."""
    try:
        if field_name == "lemma":
            token.lemma_ = value
        elif field_name == "upos":
            token.pos_ = value
        elif field_name == "xpos":
            token.tag_ = value
        elif field_name == "feats":
            token.set_morph(value if value not in ("_", "") else "")
        elif field_name == "deprel":
            token.dep_ = value
        else:
            return False
        return True
    except (KeyError, ValueError):
        return False
