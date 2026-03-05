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
    """Render the token annotation table with inline editing."""
    tokens = sent["tokens"]

    # Color coding legend
    with st.expander("Legend"):
        st.markdown(
            "🔵 **Silver** (pipeline output) · "
            "🟢 **Corrected** (human-verified) · "
            "🟡 **Modified** (unsaved change)"
        )

    for tok_idx, tok in enumerate(tokens):
        corrected_fields = tok["corrected"]
        form = tok["form"]

        # Status indicator
        if corrected_fields:
            status = "🟢"
            status_text = f"Corrected: {', '.join(corrected_fields)}"
        else:
            status = "🔵"
            status_text = "Silver (pipeline)"

        with st.container():
            cols = st.columns([0.5, 1.5, 2, 1.5, 1, 2, 1.5, 0.8])

            # Token ID and form
            cols[0].markdown(f"**{tok['id']}**")
            cols[1].markdown(f"{status} **{form}**")

            # Lemma
            key_prefix = f"s{sent_idx}_t{tok_idx}"
            new_lemma = cols[2].text_input(
                "Lemma",
                value=tok["lemma"],
                key=f"{key_prefix}_lemma",
                label_visibility="collapsed",
                placeholder="lemma",
            )

            # UPOS dropdown
            upos_options = [""] + UPOS_TAGS
            upos_idx = upos_options.index(tok["upos"]) if tok["upos"] in upos_options else 0
            new_upos = cols[3].selectbox(
                "UPOS",
                upos_options,
                index=upos_idx,
                key=f"{key_prefix}_upos",
                label_visibility="collapsed",
                format_func=lambda x: f"{x} ({UPOS_DESCRIPTIONS[x]})" if x in UPOS_DESCRIPTIONS else x or "—",
            )

            # XPOS dropdown
            xpos_options = ["_"] + [x for x in XPOS_TAGS if x != "_"]
            xpos_idx = xpos_options.index(tok["xpos"]) if tok["xpos"] in xpos_options else 0
            new_xpos = cols[4].selectbox(
                "XPOS",
                xpos_options,
                index=xpos_idx,
                key=f"{key_prefix}_xpos",
                label_visibility="collapsed",
                format_func=lambda x: f"{x} ({XPOS_DESCRIPTIONS[x]})" if x in XPOS_DESCRIPTIONS else x,
            )

            # Morph features (compact display with popover-style expander)
            feats_str = tok["feats"] or "_"
            new_feats = cols[5].text_input(
                "Feats",
                value=feats_str,
                key=f"{key_prefix}_feats",
                label_visibility="collapsed",
                placeholder="Case=Nom|Number=Sing",
            )

            # NER
            ner_options = NER_IOB_LABELS
            ner_idx = ner_options.index(tok["ner"]) if tok["ner"] in ner_options else 0
            new_ner = cols[6].selectbox(
                "NER",
                ner_options,
                index=ner_idx,
                key=f"{key_prefix}_ner",
                label_visibility="collapsed",
                format_func=lambda x: (
                    NER_DESCRIPTIONS.get(x.split("-")[-1], x)
                    if x.startswith(("B-", "I-")) else
                    NER_DESCRIPTIONS.get(x, x)
                ) if x else "—",
            )

            # Save button for this token
            has_change = (
                new_lemma != tok["lemma"]
                or new_upos != tok["upos"]
                or new_xpos != tok["xpos"]
                or new_feats != feats_str
                or new_ner != tok["ner"]
            )

            if has_change:
                if cols[7].button("💾", key=f"{key_prefix}_save", help="Stage this correction"):
                    # Validate
                    errors = []
                    if new_lemma != tok["lemma"]:
                        err = validate_lemma(new_lemma)
                        if err:
                            errors.append(err)
                    if new_upos != tok["upos"]:
                        err = validate_upos(new_upos)
                        if err:
                            errors.append(err)
                    if new_feats != feats_str:
                        feats = feats_from_str(new_feats)
                        errors.extend(validate_morph(feats))

                    if errors:
                        for err in errors:
                            st.error(err)
                    else:
                        corr = TokenCorrection(
                            sent_idx=sent_idx,
                            token_idx=tok_idx,
                            form=form,
                            old_lemma=tok["lemma"],
                            old_upos=tok["upos"],
                            old_xpos=tok["xpos"],
                            old_feats=feats_str,
                            old_ner=tok["ner"],
                            new_lemma=new_lemma if new_lemma != tok["lemma"] else "",
                            new_upos=new_upos if new_upos != tok["upos"] else "",
                            new_xpos=new_xpos if new_xpos != tok["xpos"] else "",
                            new_feats=new_feats if new_feats != feats_str else "",
                            new_ner=new_ner if new_ner != tok["ner"] else "",
                        )
                        # Remove any existing correction for this token
                        st.session_state.pending_corrections = [
                            c for c in st.session_state.pending_corrections
                            if not (c.sent_idx == sent_idx and c.token_idx == tok_idx)
                        ]
                        st.session_state.pending_corrections.append(corr)
                        st.rerun()

    # Morph feature helper (expandable)
    with st.expander("Morphological feature builder"):
        st.markdown("Build a feature string by selecting values:")
        morph_cols = st.columns(4)
        built_feats: dict[str, str] = {}
        for i, (feat, values) in enumerate(sorted(MORPH_FEATURES.items())):
            col = morph_cols[i % 4]
            val = col.selectbox(
                feat,
                ["—"] + values,
                key=f"morph_builder_{feat}",
            )
            if val != "—":
                built_feats[feat] = val
        if built_feats:
            result = feats_to_str(built_feats)
            st.code(result)
            st.caption("Copy this string into the Feats field above.")


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
