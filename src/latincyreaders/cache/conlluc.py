"""CoNLL-U Cache (.conlluc) format for silver-standard annotations.

The ``.conlluc`` format is CoNLL-U with mandatory file-level metadata that
marks the content as **silver-standard, machine-generated annotations**.

.. warning::

    Files in this format are **NOT gold-standard** and **MUST NOT** be used
    for model training.  The custom ``.conlluc`` extension exists specifically
    to prevent accidental ingestion by training pipelines that glob for
    ``.conllu`` files.

File-level metadata (written as ``# key = value`` comments before the first
sentence) always includes:

- ``generator``: producing library (``latincy-readers``)
- ``annotation_status``: always ``silver``
- ``do_not_use_for_training``: always ``true``
- ``model_name``: spaCy model that produced the annotations
- ``model_version``: version string of that model (if available)
- ``generated``: ISO-8601 timestamp
- ``collection``: collection identifier
- ``fileid``: original file identifier
- ``corrections``: number of human corrections applied (``0`` for fresh output)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spacy.tokens import Doc
from spacy.vocab import Vocab


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONLLUC_EXTENSION = ".conlluc"

_REQUIRED_FILE_META = frozenset({
    "generator",
    "annotation_status",
    "do_not_use_for_training",
})

# CoNLL-U column order
_COLUMNS = ("id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def doc_to_conlluc(
    doc: Doc,
    *,
    fileid: str = "",
    collection: str = "",
    model_name: str = "",
    model_version: str = "",
    corrections: int = 0,
    extra_meta: dict[str, str] | None = None,
) -> str:
    """Serialize a spaCy Doc to ``.conlluc`` (CoNLL-U Cache) text.

    Parameters
    ----------
    doc:
        The annotated spaCy Doc.
    fileid:
        Original file identifier.
    collection:
        Collection this annotation belongs to.
    model_name:
        Name of the spaCy model used (e.g. ``"la_core_web_lg"``).
    model_version:
        Version of the model.
    corrections:
        Number of human corrections applied (0 for raw output).
    extra_meta:
        Additional key-value pairs for the file header.

    Returns
    -------
    str
        Full ``.conlluc`` file content (UTF-8 text).
    """
    lines: list[str] = []

    # -- file-level metadata ------------------------------------------------
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    file_meta: dict[str, str] = {
        "generator": "latincy-readers",
        "annotation_status": "silver",
        "do_not_use_for_training": "true",
        "model_name": model_name or "unknown",
        "model_version": model_version or "unknown",
        "generated": now,
        "collection": collection,
        "fileid": fileid,
        "corrections": str(corrections),
    }
    if extra_meta:
        file_meta.update(extra_meta)

    for key, value in file_meta.items():
        lines.append(f"# {key} = {value}")
    lines.append("")  # blank line separates file header from sentences

    # -- sentences ----------------------------------------------------------
    # Handle docs without sentence boundaries: treat entire doc as one sentence
    try:
        sents = list(doc.sents)
    except ValueError:
        # No sentence boundaries set — wrap entire doc as a single sentence
        sents = [doc[:]] if len(doc) > 0 else []

    for sent_idx, sent in enumerate(sents):
        # Sentence-level metadata
        lines.append(f"# sent_id = {fileid}:{sent_idx + 1}")
        lines.append(f"# text = {sent.text}")

        for local_idx, token in enumerate(sent, start=1):
            form = token.text
            lemma = token.lemma_ or "_"
            upos = token.pos_ or "_"
            xpos = token.tag_ or "_"
            feats = _format_feats(token)
            head = _resolve_head(token, sent, local_idx)
            deprel = token.dep_ or "_"
            deps = "_"  # enhanced deps not tracked
            misc = _format_misc(token)

            row = "\t".join([
                str(local_idx),
                form,
                lemma,
                upos,
                xpos,
                feats,
                str(head),
                deprel,
                deps,
                misc,
            ])
            lines.append(row)

        lines.append("")  # blank line after each sentence

    return "\n".join(lines)


def conlluc_to_doc(
    text: str,
    vocab: Vocab,
) -> tuple[Doc | None, dict[str, str]]:
    """Deserialize ``.conlluc`` text back into a spaCy Doc.

    Parameters
    ----------
    text:
        The ``.conlluc`` file content.
    vocab:
        Shared spaCy Vocab for Doc construction.

    Returns
    -------
    tuple[Doc | None, dict[str, str]]
        ``(doc, file_meta)`` where *file_meta* contains all file-level
        metadata.  Returns ``(None, file_meta)`` if the file contains no
        sentences.
    """
    file_meta: dict[str, str] = {}
    sentences: list[list[dict[str, Any]]] = []
    sent_meta: list[dict[str, str]] = []
    current_tokens: list[dict[str, Any]] = []
    current_meta: dict[str, str] = {}
    in_header = True

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Blank line: end of header or end of sentence
        if not line:
            if current_tokens:
                sentences.append(current_tokens)
                sent_meta.append(current_meta)
                current_tokens = []
                current_meta = {}
            in_header = False
            continue

        # Comment line
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

        # Token line
        in_header = False
        parts = line.split("\t")
        if len(parts) != 10:
            continue

        # Skip multi-word tokens (e.g. "1-2")
        if "-" in parts[0]:
            continue

        tok_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = parts
        current_tokens.append({
            "id": int(tok_id),
            "form": form,
            "lemma": lemma if lemma != "_" else "",
            "upos": upos if upos != "_" else "",
            "xpos": xpos if xpos != "_" else "",
            "feats": _parse_feats(feats),
            "head": int(head) if head not in ("_", "0") else 0,
            "deprel": deprel if deprel != "_" else "",
            "deps": deps,
            "misc": _parse_misc(misc),
        })

    # Flush final sentence
    if current_tokens:
        sentences.append(current_tokens)
        sent_meta.append(current_meta)

    if not sentences:
        return None, file_meta

    # -- Reconstruct Doc ----------------------------------------------------
    words: list[str] = []
    spaces: list[bool] = []
    sent_starts: list[bool] = []
    token_data: list[dict[str, Any]] = []
    sent_offsets: list[tuple[int, int]] = []

    for sent_tokens in sentences:
        start = len(words)
        for i, tok in enumerate(sent_tokens):
            words.append(tok["form"])
            has_space = tok["misc"].get("SpaceAfter", "Yes") != "No"
            spaces.append(has_space)
            sent_starts.append(i == 0)
            token_data.append(tok)
        sent_offsets.append((start, len(words)))

    doc = Doc(vocab, words=words, spaces=spaces, sent_starts=sent_starts)

    # Populate token attributes
    for sent_idx, (start, end) in enumerate(sent_offsets):
        for local_idx, token in enumerate(doc[start:end]):
            td = token_data[start + local_idx]
            token.lemma_ = td["lemma"]
            token.pos_ = td["upos"]
            token.tag_ = td["xpos"]
            token.dep_ = td["deprel"]

            # Resolve head
            head_idx = td["head"]
            if head_idx == 0:
                token.head = token
            else:
                target = start + head_idx - 1
                if 0 <= target < len(doc):
                    token.head = doc[target]
                else:
                    token.head = token

    # Set doc-level extensions (if registered)
    fileid = file_meta.get("fileid", "")
    if Doc.has_extension("fileid"):
        doc._.fileid = fileid
    if Doc.has_extension("metadata"):
        doc._.metadata = {
            "source": "conlluc",
            "annotation_status": file_meta.get("annotation_status", "silver"),
            "model_name": file_meta.get("model_name", ""),
        }

    return doc, file_meta


def write_conlluc(path: str | Path, content: str) -> None:
    """Write ``.conlluc`` content to disk."""
    path = Path(path)
    path.write_text(content, encoding="utf-8")


def read_conlluc(path: str | Path, vocab: Vocab) -> tuple[Doc | None, dict[str, str]]:
    """Read a ``.conlluc`` file from disk and parse it.

    Returns ``(doc, file_meta)``."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return conlluc_to_doc(text, vocab)


def validate_conlluc_header(file_meta: dict[str, str]) -> list[str]:
    """Check that required file-level metadata is present.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []
    for key in _REQUIRED_FILE_META:
        if key not in file_meta:
            errors.append(f"Missing required metadata: {key}")

    if file_meta.get("annotation_status") not in (None, "silver"):
        if file_meta.get("annotation_status") != "silver":
            errors.append(
                f"annotation_status must be 'silver', "
                f"got '{file_meta['annotation_status']}'"
            )

    if file_meta.get("do_not_use_for_training") not in (None, "true"):
        if file_meta.get("do_not_use_for_training") != "true":
            errors.append("do_not_use_for_training must be 'true'")

    return errors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_feats(token: Any) -> str:
    """Format morphological features as UD FEATS string."""
    morph = str(token.morph) if hasattr(token, "morph") and token.morph else ""
    return morph if morph else "_"


def _resolve_head(token: Any, sent: Any, local_idx: int) -> int:
    """Convert spaCy head to CoNLL-U 1-based head index within the sentence.

    Returns 0 for root tokens.
    """
    if token.head == token:
        return 0
    # Head offset within the sentence
    head_offset = token.head.i - sent.start
    if head_offset < 0 or head_offset >= len(sent):
        return 0  # fallback for cross-sentence heads
    return head_offset + 1  # 1-based


def _format_misc(token: Any) -> str:
    """Format MISC field (SpaceAfter, Corrected, NER)."""
    parts: dict[str, str] = {}
    if not token.whitespace_:
        parts["SpaceAfter"] = "No"
    # Preserve Corrected= if stored in token's custom data
    if hasattr(token, "_") and hasattr(token._, "ud"):
        ud = token._.ud
        if isinstance(ud, dict):
            misc = ud.get("misc", {})
            if isinstance(misc, dict):
                if "Corrected" in misc:
                    parts["Corrected"] = misc["Corrected"]
                if "NER" in misc:
                    parts["NER"] = misc["NER"]
    # NER from spaCy ent_type_
    if not parts.get("NER") and hasattr(token, "ent_type_") and token.ent_type_:
        iob = getattr(token, "ent_iob_", "O")
        if iob in ("B", "I"):
            parts["NER"] = f"{iob}-{token.ent_type_}"
        elif iob == "O":
            pass  # don't clutter with O tags
    if not parts:
        return "_"
    return "|".join(f"{k}={v}" for k, v in sorted(parts.items()))


def _parse_feats(feats_str: str) -> dict[str, str]:
    """Parse UD FEATS string into a dict."""
    if feats_str == "_" or not feats_str:
        return {}
    result = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def _parse_misc(misc_str: str) -> dict[str, str]:
    """Parse UD MISC string into a dict."""
    if misc_str == "_" or not misc_str:
        return {}
    result = {}
    for pair in misc_str.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result
