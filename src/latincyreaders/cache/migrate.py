"""Token-drift migration — form-anchored alignment + correction re-pointing.

When a text is re-annotated under a new model (e.g. ``la_core_web_lg`` 3.9.4 →
3.9.6) or its source is edited, tokenization can change: a token after a
punctuation split/merge keeps its surface form but lands at a new index, so the
durable opaque token id (``t0004`` = the 4th token at mint time) now names a
different word. Corrections in the JSON correction layer are keyed by those ids,
so they must be **re-pointed** onto the new tokenization.

This module is the "Alembic for the cache": it builds a reversible map
``old token id → new token id`` by aligning the OLD token-form sequence against
the NEW one, then re-points corrections through it. Every remap is **verified
against the correction's recorded surface form** before it is accepted; anything
that doesn't verify (orphaned, merged-away, ambiguous, or landed on a different
word) is quarantined for human review rather than silently mis-applied.

Two callers in latincy-readers:

- **DocBin rebuild** (the fast path): read the *old* DocBin's ``(token_id, form)``
  sequence, align it against the freshly re-annotated doc, carry ids forward, and
  re-point corrections. Cheap and exact because the whole old spine is present.
- **Self-anchoring fallback**: when the old DocBin is gone, a correction re-points
  itself by aligning its stored *enclosing-sentence* forms against the new doc's
  sentences — the same ``align_section`` machinery, scoped to one sentence.

Pure functions over form sequences: no spaCy, no model, no reader imports — so the
aligner is safe to call from anywhere (and latincy-viewer can re-import it, since
readers is the lower layer). Ported from ``latincy_viewer.migrate``; the API is
kept identical to ease that future reconciliation. The ``section`` parameter is a
free-form label (readers passes a fileid or sentence id).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenRef:
    """A token's durable id and its surface form (the alignment anchor)."""

    id: str
    form: str


@dataclass
class SectionAlignment:
    """The alignment of one unit's old token sequence against the new one.

    ``id_map`` holds only the *clean* re-pointable maps (old id → new id) — the
    1:1 ``keep`` pairings plus a confidently-detected split/merge core. Tokens in
    ``removed`` have no clean new counterpart; ``added`` are genuinely new tokens.
    ``ops`` is the ordered, human-readable op log persisted in the revision.
    """

    section: str
    ops: list[dict[str, Any]]
    id_map: dict[str, str]
    removed: list[str]
    added: list[str]
    new_forms: dict[str, str] = field(default_factory=dict)
    old_forms: dict[str, str] = field(default_factory=dict)


@dataclass
class Repoint:
    new_id: str | None
    status: str  # "keep" | "orphan"


@dataclass
class RepointResult:
    correction: dict[str, Any] | None
    status: str  # "clean" | "quarantine"
    reason: str


def _alpha_core(tokens: list[TokenRef]) -> TokenRef | None:
    """The single alphabetic token in a split/merge block, or None if ambiguous.

    A punctuation split (``augurem,`` → ``augurem`` + ``,``) has exactly one
    alphabetic piece — that piece inherits the old id. Zero or several alphabetic
    pieces means we cannot pick a head safely, so we refuse to guess.
    """
    alpha = [t for t in tokens if any(c.isalpha() for c in t.form)]
    return alpha[0] if len(alpha) == 1 else None


#: Latin enclitics that attach to a host word (``Estne`` = Est + -ne, ``virumque`` =
#: virum + -que). In a split, the NON-enclitic piece is the head that inherits the id.
_ENCLITICS = {"que", "ne", "ve"}


def _split_head(tokens: list[TokenRef]) -> TokenRef | None:
    """The head piece of a split/merge block — the token that inherits the old id.

    First tries a punctuation split (exactly one alphabetic piece → that piece).
    Failing that, handles an ENCLITIC split (``Estne`` → ``Est`` + ``ne``): when
    exactly one alphabetic piece is NON-enclitic and the rest are enclitics, the
    non-enclitic piece is the head. Still refuses (returns None) when the head is
    genuinely ambiguous — never guesses."""
    core = _alpha_core(tokens)
    if core is not None:
        return core
    alpha = [t for t in tokens if any(c.isalpha() for c in t.form)]
    non_enclitic = [t for t in alpha if t.form.lower() not in _ENCLITICS]
    return non_enclitic[0] if len(non_enclitic) == 1 else None


def _handle_replace(
    olds: list[TokenRef],
    news: list[TokenRef],
    ops: list[dict[str, Any]],
    id_map: dict[str, str],
    added: list[str],
    removed: list[str],
) -> None:
    """Classify a SequenceMatcher 'replace' block as split / merge / generic."""
    o_concat = "".join(t.form for t in olds)
    n_concat = "".join(t.form for t in news)
    same_chars = o_concat == n_concat

    if same_chars and len(olds) == 1 and len(news) > 1:  # split: 1 → many
        core = _split_head(news)
        ops.append(
            {
                "op": "split",
                "old": olds[0].id,
                "old_form": olds[0].form,
                "new": [n.id for n in news],
                "forms": [n.form for n in news],
                "core": core.id if core else None,
            }
        )
        if core is not None:
            id_map[olds[0].id] = core.id
            added.extend(n.id for n in news if n.id != core.id)
        else:
            removed.append(olds[0].id)
            added.extend(n.id for n in news)
        return

    if same_chars and len(news) == 1 and len(olds) > 1:  # merge: many → 1
        core = _split_head(olds)
        ops.append(
            {
                "op": "merge",
                "old": [o.id for o in olds],
                "forms": [o.form for o in olds],
                "new": news[0].id,
                "new_form": news[0].form,
                "core": core.id if core else None,
            }
        )
        # Every constituent is part of the merged token, so each maps to it. A
        # multi-token span that covered the merge then collapses to one id (the
        # re-pointer dedups), and the recorded surface is re-verified — e.g. the
        # entity span ['Sp','.','Mummium'] → ['Sp.','Mummium'].
        for o in olds:
            id_map[o.id] = news[0].id
        return

    # genuine substitution — conservative: orphan every old, flag every new.
    ops.append(
        {
            "op": "replace",
            "old": [o.id for o in olds],
            "old_forms": [o.form for o in olds],
            "new": [n.id for n in news],
            "new_forms": [n.form for n in news],
        }
    )
    removed.extend(o.id for o in olds)
    added.extend(n.id for n in news)


def align_section(
    section: str, old: list[TokenRef], new: list[TokenRef]
) -> SectionAlignment:
    """Align one unit's old token sequence against the new one, by form.

    Pure function. ``equal`` runs become clean ``keep`` maps (covering pure index
    shifts after a punctuation insert/delete); ``insert``/``delete`` become
    ``add``/``remove``; ``replace`` is classified as split/merge/substitution.
    """
    old_forms = [t.form for t in old]
    new_forms = [t.form for t in new]
    ops: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}
    removed: list[str] = []
    added: list[str] = []

    sm = difflib.SequenceMatcher(a=old_forms, b=new_forms, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                o, n = old[i1 + k], new[j1 + k]
                ops.append({"op": "keep", "old": o.id, "new": n.id, "form": o.form})
                id_map[o.id] = n.id
        elif tag == "insert":
            for n in new[j1:j2]:
                ops.append({"op": "add", "new": n.id, "form": n.form})
                added.append(n.id)
        elif tag == "delete":
            for o in old[i1:i2]:
                ops.append({"op": "remove", "old": o.id, "form": o.form})
                removed.append(o.id)
        else:  # replace
            _handle_replace(old[i1:i2], new[j1:j2], ops, id_map, added, removed)

    return SectionAlignment(
        section=section,
        ops=ops,
        id_map=id_map,
        removed=removed,
        added=added,
        new_forms={t.id: t.form for t in new},
        old_forms={t.id: t.form for t in old},
    )


def alignment_to_dict(alignment: SectionAlignment) -> dict[str, Any]:
    """Serialize a section alignment for the on-disk revision (JSON-friendly)."""
    return {
        "ops": alignment.ops,
        "id_map": alignment.id_map,
        "removed": alignment.removed,
        "added": alignment.added,
        "new_forms": alignment.new_forms,
        "old_forms": alignment.old_forms,
    }


def alignment_from_dict(section: str, data: dict[str, Any]) -> SectionAlignment:
    """Reconstruct a section alignment from its persisted revision entry."""
    return SectionAlignment(
        section=section,
        ops=data.get("ops", []),
        id_map=data.get("id_map", {}),
        removed=data.get("removed", []),
        added=data.get("added", []),
        new_forms=data.get("new_forms", {}),
        old_forms=data.get("old_forms", {}),
    )


def repoint_id(old_id: str, alignment: SectionAlignment) -> Repoint:
    """Re-point a single token id through an alignment (None if orphaned)."""
    new_id = alignment.id_map.get(old_id)
    return Repoint(new_id, "keep" if new_id is not None else "orphan")


def _word_content(forms: list[str]) -> str:
    """The case-folded alphabetic content of a list of token forms.

    This is the verification invariant: a correction still covers the same
    *word-content* iff the letters of its span are preserved. It deliberately
    ignores punctuation, whitespace, gap words outside the span, and case — so a
    pure index shift, an ``Sp`` + ``.`` → ``Sp.`` merge, a discontinuous mention
    that skips a particle, and a sentence-initial capitalisation all verify, while
    a remap onto a genuinely different word (different letters) does not.
    """
    return "".join(c.lower() for f in forms for c in f if c.isalpha())


def repoint_correction(
    corr: dict[str, Any], alignment: SectionAlignment
) -> RepointResult:
    """Re-point one correction's ``target``/``span`` and verify its word-content.

    A correction is accepted only when every id remaps cleanly AND the alphabetic
    content of its span is preserved across the remap (``_word_content``). A merge
    collapses several old ids onto one (the span dedups); anything that lands on a
    different word, or whose token was orphaned, is quarantined — never silently
    re-applied. The verification compares the OLD span's letters against the NEW
    span's letters, so it depends on nothing but the alignment itself.
    """
    old_ids = list(corr.get("span") or [corr.get("target")])
    if not old_ids or old_ids[0] is None:
        return RepointResult(None, "quarantine", "correction has no target/span")

    new_ids: list[str] = []
    for oid in old_ids:
        new_id = alignment.id_map.get(oid)
        if new_id is None:
            return RepointResult(None, "quarantine", f"token {oid} orphaned (no clean remap)")
        if new_id not in new_ids:  # a merge collapses several old ids onto one
            new_ids.append(new_id)

    old_content = _word_content([alignment.old_forms.get(oid, "") for oid in old_ids])
    new_content = _word_content([alignment.new_forms.get(nid, "") for nid in new_ids])
    if old_content != new_content:
        return RepointResult(
            None,
            "quarantine",
            f"word-content changed: {old_content!r} → {new_content!r}",
        )

    new_corr = dict(corr)
    if "span" in corr:
        new_corr["span"] = new_ids
    if "target" in corr:
        new_corr["target"] = new_ids[0]

    # Re-point the head of a dependency edge — a SECOND token id the correction names
    # that also shifts under the same retokenization; carrying only target/span would
    # leave a stale, mis-pointed arc. ``head_id`` lives top-level in the correction
    # (a syntax fix) and inside ``body`` in a stamped record — handle both. A root
    # self-loop (head_id == target) re-points to the new target id.
    def _remap_head(old_head: str) -> str | None:
        return alignment.id_map.get(old_head)

    if new_corr.get("head_id"):
        nh = _remap_head(new_corr["head_id"])
        if nh is None:
            return RepointResult(None, "quarantine", f"head_id {new_corr['head_id']} orphaned")
        new_corr["head_id"] = nh
    body = new_corr.get("body")
    if isinstance(body, dict) and body.get("head_id"):
        nh = _remap_head(body["head_id"])
        if nh is None:
            return RepointResult(None, "quarantine", f"body.head_id {body['head_id']} orphaned")
        new_corr["body"] = {**body, "head_id": nh}
    return RepointResult(new_corr, "clean", "")
