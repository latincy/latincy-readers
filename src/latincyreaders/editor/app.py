"""Streamlit annotation editor for .conlluc files.

Launch with::

    streamlit run src/latincyreaders/editor/app.py -- --store /path/to/corrections

Or via the package entry point::

    latincy-editor --store /path/to/corrections
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from latincyreaders.editor.validation import (
    MORPH_FEATURES,
    NER_IOB_LABELS,
    NER_DESCRIPTIONS,
    UPOS_DESCRIPTIONS,
    UPOS_TAGS,
    XPOS_DESCRIPTIONS,
    XPOS_TAGS,
    feats_from_str,
    feats_to_str,
    validate_lemma,
    validate_morph,
    validate_ner,
    validate_upos,
    validate_xpos,
)
from latincyreaders.editor.corrections import (
    CorrectionStore,
    CorrectionSubmission,
    TokenCorrection,
    apply_corrections_to_conlluc,
    parse_conlluc_for_editor,
)



# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="LatinCy Annotation Editor",
        page_icon="\U0001F3DB",  # classical building
        layout="wide",
    )

    st.title("LatinCy Annotation Editor")

    # Sidebar: configuration
    with st.sidebar:
        st.header("Configuration")

        corrections_dir = st.text_input(
            "Corrections directory",
            value=str(Path.home() / ".latincy_data" / "corrections"),
            help="Where pending corrections are stored as JSON files",
        )
        store = CorrectionStore(corrections_dir)

        submitted_by = st.text_input("Your name", value="anonymous")

        st.divider()
        st.header("Navigation")
        mode = st.radio("Mode", ["Edit annotations", "Review corrections"])

    if mode == "Edit annotations":
        _edit_mode(store, submitted_by)
    else:
        _review_mode(store)


def _edit_mode(store: CorrectionStore, submitted_by: str) -> None:
    """Main editing interface."""

    # File upload
    uploaded = st.file_uploader(
        "Upload a .conlluc file",
        type=["conlluc"],
        help="Upload a .conlluc file to edit its annotations",
    )

    if uploaded is None:
        st.info("Upload a .conlluc file to get started, or paste content below.")
        pasted = st.text_area(
            "Or paste .conlluc content",
            height=200,
            placeholder="# generator = latincy-readers\n# annotation_status = silver\n...",
        )
        if not pasted.strip():
            return
        raw_text = pasted
        filename = "pasted.conlluc"
    else:
        raw_text = uploaded.read().decode("utf-8")
        filename = uploaded.name

    # Parse
    parsed = parse_conlluc_for_editor(raw_text)
    file_meta = parsed["file_meta"]
    sentences = parsed["sentences"]

    if not sentences:
        st.error("No sentences found in file.")
        return

    # File info
    with st.expander("File metadata", expanded=False):
        cols = st.columns(3)
        cols[0].metric("Sentences", len(sentences))
        cols[1].metric("Model", file_meta.get("model_name", "unknown"))
        cols[2].metric("Corrections", file_meta.get("corrections", "0"))
        st.json(file_meta)

    fileid = file_meta.get("fileid", filename)
    collection = file_meta.get("collection", "")

    # Sentence navigator
    st.subheader("Sentences")

    # Initialize session state for corrections
    if "pending_corrections" not in st.session_state:
        st.session_state.pending_corrections = []

    # Sentence selector
    sent_options = []
    for i, sent in enumerate(sentences):
        text = sent["meta"].get("text", " ".join(t["form"] for t in sent["tokens"]))
        preview = text[:80] + "..." if len(text) > 80 else text
        sent_options.append(f"{i + 1}. {preview}")

    selected_sent_idx = st.selectbox(
        "Select sentence",
        range(len(sentences)),
        format_func=lambda i: sent_options[i],
    )

    sent = sentences[selected_sent_idx]
    sent_text = sent["meta"].get("text", " ".join(t["form"] for t in sent["tokens"]))

    # Display sentence text
    st.markdown(f"**Full text:** *{sent_text}*")

    # Token table with inline editing
    st.subheader("Token annotations")

    _render_token_editor(sent, selected_sent_idx)

    # Pending corrections summary
    if st.session_state.pending_corrections:
        st.divider()
        st.subheader(f"Pending changes ({len(st.session_state.pending_corrections)})")
        for corr in st.session_state.pending_corrections:
            changed = corr.changed_fields
            if changed:
                st.markdown(
                    f"- **{corr.form}** (sent {corr.sent_idx + 1}, "
                    f"token {corr.token_idx + 1}): {', '.join(changed)}"
                )

        col1, col2 = st.columns(2)
        with col1:
            reason = st.text_area(
                "Reason for corrections (optional)",
                placeholder="e.g. 'Fixed lemma for deponent verb'",
            )
        with col2:
            if st.button("Submit corrections", type="primary"):
                # Apply reason to all corrections
                for corr in st.session_state.pending_corrections:
                    corr.reason = reason

                submission = CorrectionSubmission(
                    fileid=fileid,
                    collection=collection,
                    submitted_by=submitted_by,
                    corrections=st.session_state.pending_corrections,
                )
                errors = submission.validate()
                hard_errors = [e for e in errors if "Invalid" in e]
                if hard_errors:
                    for err in hard_errors:
                        st.error(err)
                else:
                    if errors:
                        for w in errors:
                            st.warning(w)
                    try:
                        path = store.submit(submission)
                        st.success(
                            f"Submitted {len(submission.corrections)} correction(s). "
                            f"ID: {submission.submission_id}"
                        )
                        st.session_state.pending_corrections = []
                    except ValueError as e:
                        st.error(str(e))

        if st.button("Clear all pending"):
            st.session_state.pending_corrections = []
            st.rerun()


def _render_token_editor(sent: dict[str, Any], sent_idx: int) -> None:
    """Render tabbed annotation editor — one layer per tab across all tokens."""
    tokens = sent["tokens"]
    pending = st.session_state.get("pending_corrections", [])

    # Collect new values in session state dicts keyed by (sent_idx, tok_idx)
    if "edits" not in st.session_state:
        st.session_state.edits = {}

    tab_lemma, tab_upos, tab_xpos, tab_morph, tab_ner = st.tabs(
        ["Lemma", "UPOS", "XPOS", "Morph", "NER"]
    )

    # --- Helper: status badge for a token ---
    def _badge(tok: dict[str, Any], tok_idx: int) -> str:
        is_pending = any(
            c.sent_idx == sent_idx and c.token_idx == tok_idx
            for c in pending
        )
        if is_pending:
            return "🟡"
        if tok["corrected"]:
            return "🟢"
        return ""

    # === LEMMA TAB ===
    with tab_lemma:
        for tok_idx, tok in enumerate(tokens):
            badge = _badge(tok, tok_idx)
            key = f"s{sent_idx}_t{tok_idx}_lemma"
            cols = st.columns([2, 3])
            cols[0].markdown(f"{badge} **{tok['form']}**")
            cols[1].text_input(
                "lemma",
                value=tok["lemma"],
                key=key,
                label_visibility="collapsed",
            )

    # === UPOS TAB ===
    with tab_upos:
        upos_options = [""] + UPOS_TAGS
        for tok_idx, tok in enumerate(tokens):
            badge = _badge(tok, tok_idx)
            key = f"s{sent_idx}_t{tok_idx}_upos"
            upos_idx = (
                upos_options.index(tok["upos"])
                if tok["upos"] in upos_options else 0
            )
            cols = st.columns([2, 3])
            cols[0].markdown(f"{badge} **{tok['form']}**")
            cols[1].selectbox(
                "upos",
                upos_options,
                index=upos_idx,
                key=key,
                label_visibility="collapsed",
                format_func=lambda x: (
                    f"{x} — {UPOS_DESCRIPTIONS[x]}"
                    if x in UPOS_DESCRIPTIONS else x or "—"
                ),
            )

    # === XPOS TAB ===
    with tab_xpos:
        xpos_options = ["_"] + [x for x in XPOS_TAGS if x != "_"]
        for tok_idx, tok in enumerate(tokens):
            badge = _badge(tok, tok_idx)
            key = f"s{sent_idx}_t{tok_idx}_xpos"
            xpos_idx = (
                xpos_options.index(tok["xpos"])
                if tok["xpos"] in xpos_options else 0
            )
            cols = st.columns([2, 3])
            cols[0].markdown(f"{badge} **{tok['form']}**")
            cols[1].selectbox(
                "xpos",
                xpos_options,
                index=xpos_idx,
                key=key,
                label_visibility="collapsed",
                format_func=lambda x: (
                    f"{x} — {XPOS_DESCRIPTIONS[x]}"
                    if x in XPOS_DESCRIPTIONS else x
                ),
            )

    # === MORPH TAB ===
    with tab_morph:
        for tok_idx, tok in enumerate(tokens):
            badge = _badge(tok, tok_idx)
            feats_str = tok["feats"] or "_"
            current_feats = feats_from_str(feats_str)
            key_prefix = f"s{sent_idx}_t{tok_idx}"

            with st.expander(
                f"{badge} **{tok['form']}** — `{feats_str}`",
            ):
                built: dict[str, str] = {}
                morph_cols = st.columns(3)
                for i, (feat, values) in enumerate(
                    sorted(MORPH_FEATURES.items())
                ):
                    col = morph_cols[i % 3]
                    cur = current_feats.get(feat, "—")
                    if cur not in values:
                        cur = "—"
                    val = col.selectbox(
                        feat,
                        ["—"] + values,
                        index=(["—"] + values).index(cur),
                        key=f"{key_prefix}_morph_{feat}",
                    )
                    if val != "—":
                        built[feat] = val
                new_feats = feats_to_str(built)
                if new_feats != feats_str:
                    st.info(f"`{feats_str}` → `{new_feats}`")

    # === NER TAB ===
    with tab_ner:
        ner_options = NER_IOB_LABELS
        for tok_idx, tok in enumerate(tokens):
            badge = _badge(tok, tok_idx)
            key = f"s{sent_idx}_t{tok_idx}_ner"
            ner_idx = (
                ner_options.index(tok["ner"])
                if tok["ner"] in ner_options else 0
            )
            cols = st.columns([2, 3])
            cols[0].markdown(f"{badge} **{tok['form']}**")
            cols[1].selectbox(
                "ner",
                ner_options,
                index=ner_idx,
                key=key,
                label_visibility="collapsed",
                format_func=lambda x: (
                    f"{x} — {NER_DESCRIPTIONS.get(x.split('-')[-1], '')}"
                    if x.startswith(("B-", "I-"))
                    else NER_DESCRIPTIONS.get(x, x)
                ) if x else "—",
            )

    # === STAGE ALL CHANGES button ===
    st.divider()
    _stage_all_changes(tokens, sent_idx)


def _stage_all_changes(
    tokens: list[dict[str, Any]], sent_idx: int
) -> None:
    """Detect and stage all changes across tabs for this sentence."""
    changes: list[tuple[int, dict[str, Any], dict[str, str]]] = []

    for tok_idx, tok in enumerate(tokens):
        key_prefix = f"s{sent_idx}_t{tok_idx}"
        feats_str = tok["feats"] or "_"

        new_lemma = st.session_state.get(f"{key_prefix}_lemma", tok["lemma"])
        new_upos = st.session_state.get(f"{key_prefix}_upos", tok["upos"])
        new_xpos = st.session_state.get(f"{key_prefix}_xpos", tok["xpos"])
        new_ner = st.session_state.get(f"{key_prefix}_ner", tok["ner"])

        # Reconstruct morph from individual selectors
        built: dict[str, str] = {}
        for feat in sorted(MORPH_FEATURES):
            val = st.session_state.get(f"{key_prefix}_morph_{feat}", "—")
            if val != "—":
                built[feat] = val
        new_feats = feats_to_str(built)

        diffs: dict[str, str] = {}
        if new_lemma != tok["lemma"]:
            diffs["lemma"] = new_lemma
        if new_upos != tok["upos"]:
            diffs["upos"] = new_upos
        if new_xpos != tok["xpos"]:
            diffs["xpos"] = new_xpos
        if new_feats != feats_str:
            diffs["feats"] = new_feats
        if new_ner != tok["ner"]:
            diffs["ner"] = new_ner

        if diffs:
            changes.append((tok_idx, tok, diffs))

    if not changes:
        st.caption("No changes to stage.")
        return

    st.markdown(f"**{len(changes)} token(s) modified:**")
    for tok_idx, tok, diffs in changes:
        parts = [
            f"{k}: `{tok.get(k, tok.get('feats') or '_') if k != 'feats' else (tok['feats'] or '_')}` → `{v}`"
            for k, v in diffs.items()
        ]
        st.markdown(f"- **{tok['form']}** — {' · '.join(parts)}")

    if st.button("Stage all changes", type="primary"):
        errors: list[str] = []
        for tok_idx, tok, diffs in changes:
            form = tok["form"]
            if "lemma" in diffs:
                err = validate_lemma(diffs["lemma"])
                if err:
                    errors.append(f"{form}: {err}")
            if "upos" in diffs:
                err = validate_upos(diffs["upos"])
                if err:
                    errors.append(f"{form}: {err}")
            if "feats" in diffs:
                for w in validate_morph(feats_from_str(diffs["feats"])):
                    errors.append(f"{form}: {w}")

        if errors:
            for err in errors:
                st.error(err)
            return

        feats_str_map = {
            tok_idx: tok["feats"] or "_"
            for tok_idx, tok, _ in changes
        }
        for tok_idx, tok, diffs in changes:
            feats_str = feats_str_map[tok_idx]
            corr = TokenCorrection(
                sent_idx=sent_idx,
                token_idx=tok_idx,
                form=tok["form"],
                old_lemma=tok["lemma"],
                old_upos=tok["upos"],
                old_xpos=tok["xpos"],
                old_feats=feats_str,
                old_ner=tok["ner"],
                new_lemma=diffs.get("lemma", ""),
                new_upos=diffs.get("upos", ""),
                new_xpos=diffs.get("xpos", ""),
                new_feats=diffs.get("feats", ""),
                new_ner=diffs.get("ner", ""),
            )
            st.session_state.pending_corrections = [
                c for c in st.session_state.pending_corrections
                if not (
                    c.sent_idx == sent_idx
                    and c.token_idx == tok_idx
                )
            ]
            st.session_state.pending_corrections.append(corr)
        st.rerun()


def _review_mode(store: CorrectionStore) -> None:
    """Review interface for pending corrections."""
    st.subheader("Pending corrections")

    pending = store.list_pending()

    if not pending:
        st.info("No pending corrections to review.")
        return

    for sub in pending:
        with st.expander(
            f"{sub.fileid} — {len(sub.corrections)} correction(s) "
            f"by {sub.submitted_by} ({sub.submitted_at})",
            expanded=True,
        ):
            # Show each correction
            for corr in sub.corrections:
                changed = corr.changed_fields
                if not changed:
                    continue

                st.markdown(f"**{corr.form}** (sent {corr.sent_idx + 1}, token {corr.token_idx + 1})")

                diff_cols = st.columns(2)
                with diff_cols[0]:
                    st.markdown("**Before:**")
                    if "lemma" in changed:
                        st.markdown(f"- Lemma: `{corr.old_lemma}`")
                    if "upos" in changed:
                        st.markdown(f"- UPOS: `{corr.old_upos}`")
                    if "xpos" in changed:
                        st.markdown(f"- XPOS: `{corr.old_xpos}`")
                    if "feats" in changed:
                        st.markdown(f"- Feats: `{corr.old_feats}`")
                    if "ner" in changed:
                        st.markdown(f"- NER: `{corr.old_ner}`")

                with diff_cols[1]:
                    st.markdown("**After:**")
                    if "lemma" in changed:
                        st.markdown(f"- Lemma: `{corr.new_lemma}`")
                    if "upos" in changed:
                        st.markdown(f"- UPOS: `{corr.new_upos}`")
                    if "xpos" in changed:
                        st.markdown(f"- XPOS: `{corr.new_xpos}`")
                    if "feats" in changed:
                        st.markdown(f"- Feats: `{corr.new_feats}`")
                    if "ner" in changed:
                        st.markdown(f"- NER: `{corr.new_ner}`")

                if corr.reason:
                    st.caption(f"Reason: {corr.reason}")

            # Action buttons
            col1, col2, col3 = st.columns(3)
            reviewer_notes = col1.text_input(
                "Reviewer notes",
                key=f"review_notes_{sub.submission_id}",
                placeholder="Optional notes...",
            )
            if col2.button(
                "Accept",
                key=f"accept_{sub.submission_id}",
                type="primary",
            ):
                store.accept(sub.submission_id, reviewer_notes)
                st.success(f"Accepted submission {sub.submission_id}")
                st.rerun()
            if col3.button(
                "Reject",
                key=f"reject_{sub.submission_id}",
            ):
                store.reject(sub.submission_id, reviewer_notes)
                st.warning(f"Rejected submission {sub.submission_id}")
                st.rerun()

    # Accepted corrections history
    st.divider()
    st.subheader("History")
    tab1, tab2 = st.tabs(["Accepted", "Rejected"])
    with tab1:
        accepted = store.list_accepted()
        if accepted:
            for sub in accepted:
                st.markdown(
                    f"- **{sub.fileid}** — {len(sub.corrections)} corrections "
                    f"by {sub.submitted_by} ({sub.submitted_at})"
                )
                if sub.reviewer_notes:
                    st.caption(f"Notes: {sub.reviewer_notes}")
        else:
            st.caption("No accepted corrections yet.")
    with tab2:
        rejected = store.list_rejected()
        if rejected:
            for sub in rejected:
                st.markdown(
                    f"- ~~{sub.fileid}~~ — {len(sub.corrections)} corrections "
                    f"by {sub.submitted_by} ({sub.submitted_at})"
                )
                if sub.reviewer_notes:
                    st.caption(f"Notes: {sub.reviewer_notes}")
        else:
            st.caption("No rejected corrections.")


if __name__ == "__main__":
    main()
