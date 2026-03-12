"""Correction submission and review system for .conlluc annotations.

Corrections are stored as lightweight JSON files in a pending directory.
A maintainer reviews and applies them, which updates the .conlluc file
and adds ``Corrected=<fields>`` to the MISC column of affected tokens.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from latincyreaders.editor.validation import (
    validate_lemma,
    validate_morph,
    validate_ner,
    validate_upos,
    validate_xpos,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TokenCorrection:
    """A single token-level correction."""

    sent_idx: int
    token_idx: int
    form: str  # read-only context — not editable

    # Original values (for display / diff)
    old_lemma: str = ""
    old_upos: str = ""
    old_xpos: str = ""
    old_feats: str = ""
    old_ner: str = ""

    # New values (empty string = no change)
    new_lemma: str = ""
    new_upos: str = ""
    new_xpos: str = ""
    new_feats: str = ""
    new_ner: str = ""

    reason: str = ""

    @property
    def changed_fields(self) -> list[str]:
        """Return list of field names that were actually changed."""
        changed = []
        if self.new_lemma and self.new_lemma != self.old_lemma:
            changed.append("lemma")
        if self.new_upos and self.new_upos != self.old_upos:
            changed.append("upos")
        if self.new_xpos and self.new_xpos != self.old_xpos:
            changed.append("xpos")
        if self.new_feats and self.new_feats != self.old_feats:
            changed.append("feats")
        if self.new_ner and self.new_ner != self.old_ner:
            changed.append("ner")
        return changed

    @property
    def has_changes(self) -> bool:
        return len(self.changed_fields) > 0


@dataclass
class CorrectionSubmission:
    """A batch of corrections submitted by a user for a single file."""

    submission_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    fileid: str = ""
    collection: str = ""
    submitted_by: str = "anonymous"
    submitted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    status: str = "pending"  # pending | accepted | rejected
    reviewer_notes: str = ""
    corrections: list[TokenCorrection] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate all corrections. Returns list of error/warning messages."""
        messages: list[str] = []
        for i, corr in enumerate(self.corrections):
            if not corr.has_changes:
                continue
            prefix = f"Token {corr.sent_idx}:{corr.token_idx} ({corr.form})"
            if corr.new_lemma:
                err = validate_lemma(corr.new_lemma)
                if err:
                    messages.append(f"{prefix}: {err}")
            if corr.new_upos:
                err = validate_upos(corr.new_upos)
                if err:
                    messages.append(f"{prefix}: {err}")
            if corr.new_xpos:
                err = validate_xpos(corr.new_xpos)
                if err:
                    messages.append(f"{prefix}: {err}")
            if corr.new_feats:
                from latincyreaders.editor.validation import feats_from_str
                feats = feats_from_str(corr.new_feats)
                for w in validate_morph(feats):
                    messages.append(f"{prefix}: {w}")
            if corr.new_ner:
                err = validate_ner(corr.new_ner)
                if err:
                    messages.append(f"{prefix}: {err}")
        return messages

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorrectionSubmission:
        corrections = [TokenCorrection(**c) for c in data.pop("corrections", [])]
        return cls(**data, corrections=corrections)


# ---------------------------------------------------------------------------
# Corrections store (filesystem-based)
# ---------------------------------------------------------------------------

class CorrectionStore:
    """Manages pending corrections as JSON files on disk.

    Directory layout::

        {store_root}/
            pending/
                {submission_id}.json
            accepted/
                {submission_id}.json
            rejected/
                {submission_id}.json
    """

    def __init__(self, store_root: str | Path) -> None:
        self.root = Path(store_root)
        self._pending = self.root / "pending"
        self._accepted = self.root / "accepted"
        self._rejected = self.root / "rejected"
        for d in (self._pending, self._accepted, self._rejected):
            d.mkdir(parents=True, exist_ok=True)

    def submit(self, submission: CorrectionSubmission) -> Path:
        """Save a new correction submission to the pending directory."""
        # Only include corrections that actually have changes
        submission.corrections = [c for c in submission.corrections if c.has_changes]
        if not submission.corrections:
            raise ValueError("No actual changes in submission")

        errors = submission.validate()
        hard_errors = [e for e in errors if not e.startswith("Unusual")]
        if hard_errors:
            raise ValueError(
                f"Validation errors:\n" + "\n".join(hard_errors)
            )

        path = self._pending / f"{submission.submission_id}.json"
        path.write_text(
            json.dumps(submission.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def list_pending(self, fileid: str | None = None) -> list[CorrectionSubmission]:
        """List all pending submissions, optionally filtered by fileid."""
        results = []
        for path in sorted(self._pending.glob("*.json")):
            sub = self._load(path)
            if fileid is None or sub.fileid == fileid:
                results.append(sub)
        return results

    def list_accepted(self) -> list[CorrectionSubmission]:
        return [self._load(p) for p in sorted(self._accepted.glob("*.json"))]

    def list_rejected(self) -> list[CorrectionSubmission]:
        return [self._load(p) for p in sorted(self._rejected.glob("*.json"))]

    def get(self, submission_id: str) -> CorrectionSubmission | None:
        """Load a submission by ID from any status directory."""
        for d in (self._pending, self._accepted, self._rejected):
            path = d / f"{submission_id}.json"
            if path.exists():
                return self._load(path)
        return None

    def accept(self, submission_id: str, reviewer_notes: str = "") -> CorrectionSubmission:
        """Move a submission from pending to accepted."""
        return self._move(submission_id, "accepted", reviewer_notes)

    def reject(self, submission_id: str, reviewer_notes: str = "") -> CorrectionSubmission:
        """Move a submission from pending to rejected."""
        return self._move(submission_id, "rejected", reviewer_notes)

    def _move(self, submission_id: str, target: str, notes: str) -> CorrectionSubmission:
        src = self._pending / f"{submission_id}.json"
        if not src.exists():
            raise FileNotFoundError(f"No pending submission: {submission_id}")
        sub = self._load(src)
        sub.status = target
        sub.reviewer_notes = notes
        dst_dir = self._accepted if target == "accepted" else self._rejected
        dst = dst_dir / f"{submission_id}.json"
        dst.write_text(
            json.dumps(sub.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        src.unlink()
        return sub

    @staticmethod
    def _load(path: Path) -> CorrectionSubmission:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CorrectionSubmission.from_dict(data)


# ---------------------------------------------------------------------------
# Apply accepted corrections to .conlluc files
# ---------------------------------------------------------------------------

def apply_corrections_to_conlluc(
    conlluc_text: str,
    corrections: list[TokenCorrection],
) -> str:
    """Apply accepted corrections to raw .conlluc text.

    Updates token fields and adds/extends ``Corrected=<fields>`` in the MISC
    column. Returns the modified .conlluc text.
    """
    from latincyreaders.editor.validation import feats_from_str

    # Index corrections by (sent_idx, token_idx)
    corr_map: dict[tuple[int, int], TokenCorrection] = {}
    for c in corrections:
        if c.has_changes:
            corr_map[(c.sent_idx, c.token_idx)] = c

    if not corr_map:
        return conlluc_text

    lines = conlluc_text.splitlines()
    out: list[str] = []
    sent_idx = -1
    in_header = True
    corrections_count = 0

    for line in lines:
        # Blank line
        if not line.strip():
            out.append(line)
            in_header = False
            continue

        # Comment line
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                sent_idx += 1
            # Update corrections count in header
            if in_header and line.startswith("# corrections = "):
                # We'll rewrite this at the end
                out.append(line)
                continue
            out.append(line)
            continue

        # Token line
        parts = line.split("\t")
        if len(parts) != 10:
            out.append(line)
            continue

        # Skip multi-word tokens
        if "-" in parts[0]:
            out.append(line)
            continue

        tok_idx = int(parts[0]) - 1  # 0-based
        key = (sent_idx, tok_idx)

        if key in corr_map:
            corr = corr_map[key]
            changed = corr.changed_fields

            # Apply changes
            if "lemma" in changed:
                parts[2] = corr.new_lemma
            if "upos" in changed:
                parts[3] = corr.new_upos
            if "xpos" in changed:
                parts[4] = corr.new_xpos
            if "feats" in changed:
                parts[5] = corr.new_feats
            # NER goes in MISC as NER=<label>
            if "ner" in changed:
                misc = _parse_misc_str(parts[9])
                misc["NER"] = corr.new_ner
                parts[9] = _format_misc_str(misc)

            # Add/update Corrected= in MISC
            misc = _parse_misc_str(parts[9])
            existing = misc.get("Corrected", "").split(",") if "Corrected" in misc else []
            existing = [e for e in existing if e]
            for f in changed:
                if f not in existing:
                    existing.append(f)
            misc["Corrected"] = ",".join(sorted(existing))
            parts[9] = _format_misc_str(misc)

            corrections_count += 1

        out.append("\t".join(parts))

    # Update the corrections count in the header
    result = "\n".join(out)
    # Find and update the corrections line
    import re
    old_match = re.search(r"^# corrections = (\d+)", result, re.MULTILINE)
    if old_match:
        old_count = int(old_match.group(1))
        result = result.replace(
            old_match.group(0),
            f"# corrections = {old_count + corrections_count}",
        )

    return result


# ---------------------------------------------------------------------------
# Lightweight .conlluc parser for the editor (no spaCy needed)
# ---------------------------------------------------------------------------

def parse_conlluc_for_editor(text: str) -> dict[str, Any]:
    """Parse .conlluc text into a structure the editor can display.

    Returns dict with keys: file_meta, sentences.
    Each sentence has: meta (dict), tokens (list of dicts).
    """
    file_meta: dict[str, str] = {}
    sentences: list[dict[str, Any]] = []
    current_tokens: list[dict[str, Any]] = []
    current_meta: dict[str, str] = {}
    in_header = True

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line:
            if current_tokens:
                sentences.append({
                    "meta": current_meta,
                    "tokens": current_tokens,
                })
                current_tokens = []
                current_meta = {}
            in_header = False
            continue

        if line.startswith("#"):
            key_value = line[1:].strip()
            if " = " in key_value:
                key, value = key_value.split(" = ", 1)
                if in_header and not current_tokens:
                    file_meta[key.strip()] = value.strip()
                else:
                    current_meta[key.strip()] = value.strip()
                    in_header = False
            continue

        in_header = False
        parts = line.split("\t")
        if len(parts) != 10:
            continue
        if "-" in parts[0]:
            continue

        misc = _parse_misc_str(parts[9])
        current_tokens.append({
            "id": int(parts[0]),
            "form": parts[1],
            "lemma": parts[2] if parts[2] != "_" else "",
            "upos": parts[3] if parts[3] != "_" else "",
            "xpos": parts[4] if parts[4] != "_" else "",
            "feats": parts[5] if parts[5] != "_" else "",
            "head": parts[6],
            "deprel": parts[7],
            "deps": parts[8],
            "misc_raw": parts[9],
            "misc": misc,
            "ner": misc.get("NER", "O"),
            "corrected": misc.get("Corrected", "").split(",") if misc.get("Corrected") else [],
        })

    if current_tokens:
        sentences.append({
            "meta": current_meta,
            "tokens": current_tokens,
        })

    return {"file_meta": file_meta, "sentences": sentences}


def _parse_misc_str(misc: str) -> dict[str, str]:
    if misc == "_" or not misc:
        return {}
    result = {}
    for pair in misc.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def _format_misc_str(misc: dict[str, str]) -> str:
    if not misc:
        return "_"
    return "|".join(f"{k}={v}" for k, v in sorted(misc.items()))
