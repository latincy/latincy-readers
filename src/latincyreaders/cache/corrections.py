"""Correction tracking for canonical annotations.

Records token-level human corrections as a sidecar JSON file alongside
the ``.conlluc`` canonical annotations.  Corrections are keyed by
sentence ID + token form + position so they survive model re-annotation.

Workflow:
    1. User edits ``.conlluc`` directly (best UX)
    2. ``extract_corrections()`` diffs the edited file against the baseline
    3. ``save_corrections()`` writes sidecar ``.corrections.json``
    4. After re-annotating with a new model, ``apply_corrections()``
       re-applies the stored overrides
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# CoNLL-U fields that can be corrected (columns 3–8 in 0-indexed token dict)
CORRECTABLE_FIELDS = ("lemma", "upos", "xpos", "feats", "head", "deprel")


@dataclass
class TokenCorrection:
    """A single token-level correction.

    Attributes:
        sent_id: Sentence identifier (e.g. ``"silius_italicus.punica.part.1.tess:4"``).
        token_idx: 1-based token index within the sentence (CoNLL-U ID column).
        token_form: Surface form of the token (for verification on re-apply).
        changes: Mapping of field name to ``{"from": old, "to": new}``.
    """

    sent_id: str
    token_idx: int
    token_form: str
    changes: dict[str, dict[str, Any]]


@dataclass
class CorrectionSet:
    """All corrections for a single fileid.

    Attributes:
        fileid: File identifier.
        model_name: Model that produced the baseline annotations.
        model_version: Version of that model.
        corrections: List of token-level corrections.
    """

    fileid: str
    model_name: str = ""
    model_version: str = ""
    corrections: list[TokenCorrection] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.corrections)


# ---------------------------------------------------------------------------
# Extract corrections by diffing two parsed .conlluc files
# ---------------------------------------------------------------------------

def _parse_conlluc_sentences(path: Path) -> tuple[dict[str, str], list[tuple[dict[str, str], list[dict[str, Any]]]]]:
    """Parse a .conlluc file into file metadata + list of (sent_meta, tokens).

    Returns (file_meta, [(sent_meta, [token_dict, ...]), ...]).
    """
    file_meta: dict[str, str] = {}
    result: list[tuple[dict[str, str], list[dict[str, Any]]]] = []
    current_tokens: list[dict[str, Any]] = []
    current_meta: dict[str, str] = {}
    in_header = True

    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line:
            if current_tokens:
                result.append((current_meta, current_tokens))
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

        tok_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = parts
        current_tokens.append({
            "id": int(tok_id),
            "form": form,
            "lemma": lemma,
            "upos": upos,
            "xpos": xpos,
            "feats": feats,
            "head": head,
            "deprel": deprel,
            "deps": deps,
            "misc": misc,
        })

    if current_tokens:
        result.append((current_meta, current_tokens))

    return file_meta, result


def extract_corrections(
    baseline_path: Path,
    corrected_path: Path,
) -> CorrectionSet:
    """Diff a baseline .conlluc against a user-corrected version.

    Compares sentence-by-sentence, token-by-token, and records any
    differences in correctable fields.

    Args:
        baseline_path: Path to the original model-generated .conlluc.
        corrected_path: Path to the user-edited .conlluc.

    Returns:
        CorrectionSet with all detected corrections.
    """
    base_meta, base_sents = _parse_conlluc_sentences(baseline_path)
    corr_meta, corr_sents = _parse_conlluc_sentences(corrected_path)

    fileid = corr_meta.get("fileid", base_meta.get("fileid", ""))
    model_name = base_meta.get("model_name", "")
    model_version = base_meta.get("model_version", "")

    corrections: list[TokenCorrection] = []

    for sent_idx, (base_pair, corr_pair) in enumerate(
        zip(base_sents, corr_sents)
    ):
        base_sent_meta, base_tokens = base_pair
        corr_sent_meta, corr_tokens = corr_pair
        sent_id = corr_sent_meta.get("sent_id", base_sent_meta.get("sent_id", f"sent_{sent_idx}"))

        for base_tok, corr_tok in zip(base_tokens, corr_tokens):
            changes: dict[str, dict[str, Any]] = {}
            for fld in CORRECTABLE_FIELDS:
                base_val = base_tok.get(fld, "_")
                corr_val = corr_tok.get(fld, "_")
                if base_val != corr_val:
                    changes[fld] = {"from": base_val, "to": corr_val}

            if changes:
                corrections.append(TokenCorrection(
                    sent_id=sent_id,
                    token_idx=base_tok["id"],
                    token_form=base_tok["form"],
                    changes=changes,
                ))

    cset = CorrectionSet(
        fileid=fileid,
        model_name=model_name,
        model_version=model_version,
        corrections=corrections,
    )

    if corrections:
        logger.info(
            "Extracted %d corrections for '%s' (baseline model: %s %s)",
            len(corrections), fileid, model_name, model_version,
        )

    return cset


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _corrections_path(conlluc_path: Path) -> Path:
    """Derive the sidecar corrections path from a .conlluc path."""
    return conlluc_path.with_suffix(".corrections.json")


def save_corrections(corrections: CorrectionSet, conlluc_path: Path) -> Path:
    """Write corrections to a sidecar JSON file.

    Args:
        corrections: The correction set to persist.
        conlluc_path: Path to the .conlluc file (sidecar is derived from this).

    Returns:
        Path to the written corrections file.
    """
    out_path = _corrections_path(conlluc_path)
    data = {
        "fileid": corrections.fileid,
        "model_name": corrections.model_name,
        "model_version": corrections.model_version,
        "corrections": [asdict(c) for c in corrections.corrections],
    }
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved %d corrections to %s", corrections.count, out_path)
    return out_path


def load_corrections(conlluc_path: Path) -> CorrectionSet | None:
    """Load corrections from a sidecar JSON file.

    Returns None if no corrections file exists.
    """
    corr_path = _corrections_path(conlluc_path)
    if not corr_path.exists():
        return None

    data = json.loads(corr_path.read_text(encoding="utf-8"))
    corrections = [
        TokenCorrection(
            sent_id=c["sent_id"],
            token_idx=c["token_idx"],
            token_form=c["token_form"],
            changes=c["changes"],
        )
        for c in data.get("corrections", [])
    ]
    return CorrectionSet(
        fileid=data.get("fileid", ""),
        model_name=data.get("model_name", ""),
        model_version=data.get("model_version", ""),
        corrections=corrections,
    )


# ---------------------------------------------------------------------------
# Apply corrections to a .conlluc file
# ---------------------------------------------------------------------------

def apply_corrections(
    conlluc_path: Path,
    corrections: CorrectionSet,
) -> tuple[int, int]:
    """Apply stored corrections to a (re-annotated) .conlluc file in place.

    Matches corrections by sentence ID + token index. Verifies the token
    form matches before applying. Skips corrections where the new model
    already agrees with the corrected value.

    Args:
        conlluc_path: Path to the .conlluc file to patch.
        corrections: Correction set to apply.

    Returns:
        ``(applied, skipped)`` — number of corrections applied vs skipped
        (either because the model already agrees or the token couldn't be matched).
    """
    # Build lookup: sent_id → {token_idx: TokenCorrection}
    lookup: dict[str, dict[int, TokenCorrection]] = {}
    for corr in corrections.corrections:
        lookup.setdefault(corr.sent_id, {})[corr.token_idx] = corr

    # Field name → column index in CoNLL-U (0-indexed in the 10-column row)
    field_col = {
        "lemma": 2,
        "upos": 3,
        "xpos": 4,
        "feats": 5,
        "head": 6,
        "deprel": 7,
    }

    lines = conlluc_path.read_text(encoding="utf-8").splitlines()
    current_sent_id: str = ""
    applied = 0
    skipped = 0
    already_correct = 0

    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip()

        # Track sentence ID from comments
        if line.startswith("#") and "sent_id = " in line:
            current_sent_id = line.split("sent_id = ", 1)[1].strip()
            continue

        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) != 10 or "-" in parts[0]:
            continue

        tok_idx = int(parts[0])
        sent_corrections = lookup.get(current_sent_id)
        if sent_corrections is None:
            continue

        corr = sent_corrections.get(tok_idx)
        if corr is None:
            continue

        # Verify token form matches
        if parts[1] != corr.token_form:
            logger.warning(
                "Token form mismatch at %s token %d: expected '%s', got '%s'. "
                "Skipping correction.",
                current_sent_id, tok_idx, corr.token_form, parts[1],
            )
            skipped += 1
            continue

        # Apply each field change
        token_changed = False
        for fld, change in corr.changes.items():
            col = field_col.get(fld)
            if col is None:
                continue

            current_val = parts[col]
            target_val = str(change["to"])

            if current_val == target_val:
                already_correct += 1
                continue

            parts[col] = target_val
            token_changed = True

        if token_changed:
            # Mark as corrected in MISC column
            misc = parts[9]
            if "Corrected=Yes" not in misc:
                if misc == "_":
                    parts[9] = "Corrected=Yes"
                else:
                    parts[9] = misc + "|Corrected=Yes"

            lines[i] = "\t".join(parts)
            applied += 1
        else:
            skipped += 1

    # Update corrections count in header
    for i, line in enumerate(lines):
        if line.startswith("# corrections = "):
            lines[i] = f"# corrections = {applied}"
            break

    conlluc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if already_correct:
        logger.info(
            "%d field(s) already matched the correction (new model agrees).",
            already_correct,
        )
    logger.info(
        "Applied %d corrections, skipped %d for '%s'.",
        applied, skipped, corrections.fileid,
    )

    return applied, skipped
