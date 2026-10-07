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
import difflib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from spacy.tokens import Doc

from latincyreaders.cache.disk import _fileid_hash
from latincyreaders.cache.migrate import TokenRef, align_section, repoint_correction

# Scalar CoNLL-U fields a correction can set on a token in this increment.
# (Structural head/dep edits are a follow-up — see the design doc.)
CORRECTABLE_FIELDS = ("lemma", "upos", "xpos", "feats", "deprel")

_DEFAULT_STORE_ROOT = Path.home() / "latincy_data" / "corrections"


def _today() -> str:
    return datetime.date.today().isoformat()


# doc.user_data key holding each overlaid token's pre-overlay (machine) value,
# keyed "token_id|field". Lets revert() restore exactly what overlay() replaced
# rather than trusting CorrectionRecord.was (which can be stale after a model
# upgrade). Removed by revert(), so it never reaches the DocBin cache.
_SILVER_KEY = "_lr_silver"


def _silver_key(token_id: str, field_name: str) -> str:
    return f"{token_id}|{field_name}"


def _fileid_to_filename(fileid: str, suffix: str) -> str:
    """Derive a collision-free filename for a fileid.

    Reuses disk.py's sha256-based hash (also used for the DocBin base
    cache) rather than a naive path-separator flatten, which admits
    collisions: e.g. ``"a/b"`` and ``"a--b"`` both flatten to ``"a--b"``.
    """
    return _fileid_hash(fileid) + suffix


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
        ctx: Self-anchor context — ``{"forms": [str, ...], "i": int, "rep": int}``
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

        current = _read_field(token, field_name)
        # On an overlaid doc the token already shows a gold value; provenance
        # must be the machine value it replaced, not an earlier correction.
        silver = doc.user_data.get(_SILVER_KEY, {})
        was = silver.get(_silver_key(token_id, field_name), current)
        # Validate the value actually applies before persisting it, so a
        # correction can never be silently inert (overlay() would otherwise
        # fail the same _apply_field() call at every future read with no
        # way for the caller to discover it).
        if not _apply_field(token, field_name, value):
            raise ValueError(
                f"invalid value {value!r} for field {field_name!r} "
                f"on token {token.text!r}"
            )
        # record() only validates; overlay() is the sole in-memory mutator.
        _apply_field(token, field_name, current)

        record = CorrectionRecord(
            token_id=token_id,
            form=token.text,
            field=field_name,
            value=value,
            was=was,
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
        silver = doc.user_data.setdefault(_SILVER_KEY, {})
        applied = 0
        for rec in cset.corrections:
            token = by_id.get(rec.token_id)
            if token is None or token.text != rec.form:
                continue
            key = _silver_key(rec.token_id, rec.field)
            current = _read_field(token, rec.field)
            if _apply_field(token, rec.field, rec.value):
                silver.setdefault(key, current)
                token._.corrected = True
                applied += 1
        return applied

    def revert(self, fileid: str, doc: Doc) -> int:
        """Undo an in-memory :meth:`overlay` on *doc*, restoring machine values.

        Used before writing a cached Doc back to the disk base cache (e.g.
        :meth:`BaseCorpusReader.persist_cache`), so an overlaid, gold-corrected
        in-memory Doc never contaminates the "silver" DocBin cache. Restores
        exactly the values :meth:`overlay` replaced (stashed on the doc), so it
        does not depend on the current correction records, which may have
        changed since. A doc with no stash (never overlaid by this version)
        falls back to the records' ``was``, touching only tokens whose value
        still matches the record.
        """
        by_id = {t._.token_id: t for t in doc if t._.token_id is not None}
        silver = doc.user_data.get(_SILVER_KEY)
        if silver is not None:
            # Restore exactly what overlay() replaced. Independent of the current
            # correction set, so a record changed or deleted since the overlay
            # (or an unreadable store) cannot leave a gold value in place.
            reverted = 0
            for key, original in silver.items():
                token_id, _, field_name = key.partition("|")
                token = by_id.get(token_id)
                if token is not None and _apply_field(token, field_name, original):
                    token._.corrected = False
                    reverted += 1
            del doc.user_data[_SILVER_KEY]
            return reverted

        # No stash: doc was never overlaid by this version; fall back to records.
        cset = self.load(fileid)
        if cset is None or not cset.corrections:
            return 0
        reverted = 0
        for rec in cset.corrections:
            token = by_id.get(rec.token_id)
            if token is None:
                continue
            if _read_field(token, rec.field) != rec.value:
                continue  # not currently overlaid (or overlay never applied)
            if _apply_field(token, rec.field, rec.was):
                token._.corrected = False
                reverted += 1
        return reverted

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

        occurrences = self._occurrence_index(new_doc)
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
                    rec.ctx = self._sentence_ctx(new_doc, token, occurrences)
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
    def _sentence_ctx(doc: Doc, token, occurrences: dict | None = None) -> dict[str, Any]:
        """Capture the target's enclosing-sentence forms + local index.

        ``rep`` is which occurrence of this exact sentence the target is in
        (0 = first): it disambiguates verbatim-repeated sentences and, unlike an
        absolute sentence index, survives sentences added or removed elsewhere.
        Pass *occurrences* from :meth:`_occurrence_index` when calling per token.
        """
        try:
            sent = token.sent
        except (ValueError, AttributeError):
            sent = doc[:]
        forms = [t.text for t in sent]
        if occurrences is None:
            occurrences = CorrectionStore._occurrence_index(doc)
        rep = occurrences.get(sent.start, 0)
        return {"forms": forms, "i": token.i - sent.start, "rep": rep}

    @staticmethod
    def _occurrence_index(doc: Doc) -> dict[int, int]:
        """Map each sentence start to its occurrence rank among identical sentences."""
        try:
            sents = list(doc.sents)
        except ValueError:
            return {}
        seen: dict[tuple[str, ...], int] = {}
        index: dict[int, int] = {}
        for s in sents:
            key = tuple(t.text for t in s)
            index[s.start] = seen.get(key, 0)
            seen[key] = index[s.start] + 1
        return index

    def _ctx_alignment(self, rec: CorrectionRecord, new_doc: Doc):
        """Build an alignment from a record's stored sentence context vs new_doc.

        The stored sentence carries the record's own ``token_id`` at index ``i``
        (synthetic ids elsewhere). Aligning it against the *best-matching
        sentence* (not the whole doc) avoids mis-anchoring on a repeated
        formulaic phrase that occurs more than once in the document.
        """
        forms = rec.ctx.get("forms")
        i = rec.ctx.get("i")
        if not forms or i is None or i >= len(forms):
            return None
        old_refs = [
            TokenRef(rec.token_id if k == i else f"_ctx{k}", f)
            for k, f in enumerate(forms)
        ]
        sent = self._best_matching_sentence(forms, new_doc, rec.ctx.get("rep", 0))
        if sent is None:
            return None
        return align_section(rec.token_id, old_refs, _refs(sent))

    @staticmethod
    def _best_matching_sentence(forms: list[str], new_doc: Doc, rep: int = 0):
        """Return the sentence in *new_doc* most similar to *forms*, or None.

        Guards against aligning a short, repeated context (e.g. a formulaic
        opening/closing phrase) against the wrong occurrence by scoping the
        alignment to one sentence instead of the whole document. When several
        sentences tie for best (a verbatim-repeated sentence), *rep* picks
        which occurrence, in document order; if that occurrence no longer
        exists, None (the record is quarantined rather than moved).
        """
        try:
            sents = list(new_doc.sents)
        except ValueError:
            sents = [new_doc[:]]  # no sentence boundaries set; treat as one unit
        scored = [
            (
                difflib.SequenceMatcher(None, forms, [t.text for t in s]).ratio(),
                s,
            )
            for s in sents
        ]
        best = max((r for r, _ in scored), default=0.0)
        if best == 0.0:
            return None
        tied = [s for r, s in scored if r == best]
        if rep >= len(tied):
            return None  # recorded occurrence gone: quarantine, don't guess
        return tied[rep]

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
