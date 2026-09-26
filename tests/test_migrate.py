"""Token-drift migration: form-anchored alignment + correction re-pointing.

Ported from latincy-viewer's test_migrate.py (retargeted to
``latincyreaders.cache.migrate``) so the aligner's behaviour is verified
identically, plus a readers-specific case for the sentence-scoped self-anchoring
fallback used when the old DocBin is gone. The aligner is a pure function over
form sequences — no model needed.
"""

from latincyreaders.cache.migrate import (
    SectionAlignment,
    TokenRef,
    align_section,
    alignment_from_dict,
    alignment_to_dict,
    repoint_correction,
    repoint_id,
)


def toks(*pairs: tuple[str, str]) -> list[TokenRef]:
    return [TokenRef(tid, form) for tid, form in pairs]


def test_no_drift_maps_every_id_to_itself():
    old = toks(("t000", "Q."), ("t001", "Mucius"), ("t002", "augur"))
    new = toks(("t000", "Q."), ("t001", "Mucius"), ("t002", "augur"))
    a = align_section("f", old, new)
    assert a.id_map == {"t000": "t000", "t001": "t001", "t002": "t002"}
    assert a.added == []
    assert a.removed == []


def test_punctuation_insert_shifts_following_ids_cleanly():
    # A new model splits a trailing comma into its own token; everything after
    # shifts by one index but keeps its form, so SequenceMatcher pairs 'equal'.
    old = toks(("t000", "amicis"), ("t001", "multa"), ("t002", "narrare"))
    new = toks(
        ("t000", "amicis"),
        ("t001", ","),
        ("t002", "multa"),
        ("t003", "narrare"),
    )
    a = align_section("f", old, new)
    assert a.id_map["t000"] == "t000"  # unchanged
    assert a.id_map["t001"] == "t002"  # multa shifted +1
    assert a.id_map["t002"] == "t003"  # narrare shifted +1
    assert "t001" in a.added           # the new comma token
    assert a.removed == []


def test_punctuation_merge_orphans_the_removed_token():
    old = toks(("t000", "amicis"), ("t001", ","), ("t002", "multa"))
    new = toks(("t000", "amicis"), ("t001", "multa"))
    a = align_section("f", old, new)
    assert a.id_map["t000"] == "t000"
    assert a.id_map["t002"] == "t001"  # multa shifted back
    assert "t001" in a.removed         # the comma is gone
    assert repoint_id("t001", a).new_id is None  # orphaned, no clean remap


def test_repoint_correction_follows_a_shift_and_verifies_form():
    old = toks(("t098", "de"), ("t099", "augurem"))
    new = toks(("t098", ","), ("t099", "de"), ("t100", "augurem"))
    a = align_section("f", old, new)
    corr = {"target": "t099", "form": "augurem", "value": "augur"}
    r = repoint_correction(corr, a)
    assert r.status == "clean"
    assert r.correction["target"] == "t100"
    assert r.correction["value"] == "augur"  # untouched payload


def test_word_substitution_is_quarantined():
    # A remap that lands the correction on a DIFFERENT word (different letters)
    # must never be applied silently.
    a = SectionAlignment(
        section="f",
        ops=[],
        id_map={"t099": "t100"},
        removed=[],
        added=[],
        new_forms={"t100": "auspex"},
        old_forms={"t099": "augurem"},
    )
    corr = {"target": "t099", "form": "augurem"}
    r = repoint_correction(corr, a)
    assert r.correction is None
    assert r.status == "quarantine"


def test_split_maps_correction_to_the_alpha_core():
    # A new model splits the glued comma off the word.
    old = toks(("t099", "augurem,"))
    new = toks(("t099", "augurem"), ("t100", ","))
    a = align_section("f", old, new)
    assert a.id_map["t099"] == "t099"  # the word keeps the id
    assert "t100" in a.added           # the comma is new
    corr = {"target": "t099", "form": "augurem"}
    r = repoint_correction(corr, a)
    assert r.status == "clean"
    assert r.correction["target"] == "t099"


def test_enclitic_split_keeps_head_id():
    # 'Estne' -> 'Est' + 'ne': the non-enclitic piece inherits the id.
    old = toks(("t000", "Estne"))
    new = toks(("t000", "Est"), ("t001", "ne"))
    a = align_section("f", old, new)
    assert a.id_map["t000"] == "t000"
    assert "t001" in a.added


def test_merge_collapses_span_to_the_merged_token():
    # 'Sp' + '.' merge into one token 'Sp.'; the 3-token span
    # ['t047','t048','t049'] collapses to ['t047','t048'], re-verified by form.
    old = toks(("t047", "Sp"), ("t048", "."), ("t049", "Mummium"))
    new = toks(("t047", "Sp."), ("t048", "Mummium"))
    a = align_section("f", old, new)
    corr = {
        "target": "t047",
        "span": ["t047", "t048", "t049"],
        "form": "Sp. Mummium",
    }
    r = repoint_correction(corr, a)
    assert r.status == "clean"
    assert r.correction["span"] == ["t047", "t048"]
    assert r.correction["target"] == "t047"


def test_dependency_head_repoints_with_the_token():
    old = toks(("t000", "amo"), ("t001", "te"))
    new = toks(("t000", ","), ("t001", "amo"), ("t002", "te"))
    a = align_section("f", old, new)
    corr = {"target": "t001", "form": "te", "field": "head", "head_id": "t000"}
    r = repoint_correction(corr, a)
    assert r.status == "clean"
    assert r.correction["target"] == "t002"
    assert r.correction["head_id"] == "t001"  # head followed its token


def test_alignment_round_trips_through_json_dict():
    old = toks(("t000", "amicis"), ("t001", "multa"))
    new = toks(("t000", "amicis"), ("t001", ","), ("t002", "multa"))
    a = align_section("f", old, new)
    b = alignment_from_dict("f", alignment_to_dict(a))
    assert b.id_map == a.id_map
    assert b.added == a.added
    assert b.removed == a.removed
    corr = {"target": "t001", "form": "multa"}
    assert repoint_correction(corr, b).correction["target"] == "t002"


def test_sentence_scoped_self_anchor_fallback():
    """Readers-specific: when the old DocBin is gone, a correction re-points by
    aligning its stored enclosing-sentence forms against the new sentence.

    The stored context carries the correction's *own* ids; align it against the
    freshly tokenized sentence (which carries the new ids) and the target follows.
    """
    # Correction was authored on 'cano' (t002) in "Arma virumque cano".
    stored_sentence = toks(("t000", "Arma"), ("t001", "virumque"), ("t002", "cano"))
    # New tokenization of the same sentence inserts a leading marker.
    new_sentence = toks(
        ("n000", "«"), ("n001", "Arma"), ("n002", "virumque"), ("n003", "cano")
    )
    a = align_section("s", stored_sentence, new_sentence)
    corr = {"target": "t002", "form": "cano", "field": "lemma", "value": "canō"}
    r = repoint_correction(corr, a)
    assert r.status == "clean"
    assert r.correction["target"] == "n003"
    assert r.correction["value"] == "canō"
