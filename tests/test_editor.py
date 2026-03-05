"""Tests for the annotation editor: validation, corrections, and conlluc apply."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latincyreaders.editor.validation import (
    UPOS_TAGS,
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
)
from latincyreaders.editor.corrections import parse_conlluc_for_editor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONLLUC = """\
# generator = latincy-readers
# annotation_status = silver
# do_not_use_for_training = true
# model_name = la_core_web_lg
# model_version = 3.8.0
# generated = 2025-01-01T00:00:00+00:00
# collection = test
# fileid = test.tess
# corrections = 0

# sent_id = test.tess:1
# text = Arma virumque cano.
1	Arma	arma	NOUN	n	Case=Acc|Gender=Neut|Number=Plur	3	obj	_	_
2	virumque	vir	NOUN	n	Case=Acc|Gender=Masc|Number=Sing	3	obj	_	_
3	cano	canō	VERB	t	Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	_	SpaceAfter=No
4	.	.	PUNCT	_	_	3	punct	_	_

# sent_id = test.tess:2
# text = Troiae qui primus ab oris.
1	Troiae	Trōia	PROPN	n	Case=Gen|Gender=Fem|Number=Sing	5	nmod	_	_
2	qui	quī	PRON	p	Case=Nom|Gender=Masc|Number=Sing|PronType=Rel	0	root	_	_
3	primus	prīmus	ADJ	a	Case=Nom|Degree=Pos|Gender=Masc|Number=Sing	2	amod	_	_
4	ab	ab	ADP	r	_	5	case	_	_
5	oris	ōra	NOUN	n	Case=Abl|Gender=Neut|Number=Plur	2	obl	_	SpaceAfter=No
6	.	.	PUNCT	_	_	2	punct	_	_
"""


@pytest.fixture
def tmp_store(tmp_path: Path) -> CorrectionStore:
    return CorrectionStore(tmp_path / "corrections")


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_upos_tags(self):
        for tag in UPOS_TAGS:
            assert validate_upos(tag) is None

    def test_invalid_upos(self):
        assert validate_upos("INVALID") is not None

    def test_empty_upos_ok(self):
        assert validate_upos("") is None
        assert validate_upos("_") is None

    def test_valid_xpos(self):
        assert validate_xpos("n") is None
        assert validate_xpos("t") is None
        assert validate_xpos("_") is None

    def test_unusual_xpos_warns(self):
        result = validate_xpos("ZZZ")
        assert result is not None
        assert "Unusual" in result

    def test_valid_lemma(self):
        assert validate_lemma("arma") is None
        assert validate_lemma("canō") is None

    def test_empty_lemma_rejected(self):
        assert validate_lemma("") is not None
        assert validate_lemma("   ") is not None

    def test_valid_morph(self):
        feats = {"Case": "Nom", "Number": "Sing", "Gender": "Masc"}
        assert validate_morph(feats) == []

    def test_invalid_morph_feature(self):
        feats = {"Case": "Nom", "Foo": "Bar"}
        warnings = validate_morph(feats)
        assert len(warnings) == 1
        assert "Unknown feature" in warnings[0]

    def test_invalid_morph_value(self):
        feats = {"Case": "Xyz"}
        warnings = validate_morph(feats)
        assert len(warnings) == 1
        assert "Invalid value" in warnings[0]

    def test_valid_ner(self):
        assert validate_ner("O") is None
        assert validate_ner("B-PER") is None
        assert validate_ner("I-LOC") is None

    def test_invalid_ner(self):
        assert validate_ner("PERSON") is not None

    def test_feats_roundtrip(self):
        original = {"Case": "Acc", "Gender": "Neut", "Number": "Plur"}
        s = feats_to_str(original)
        assert s == "Case=Acc|Gender=Neut|Number=Plur"
        parsed = feats_from_str(s)
        assert parsed == original

    def test_feats_empty(self):
        assert feats_to_str({}) == "_"
        assert feats_from_str("_") == {}
        assert feats_from_str("") == {}


# ---------------------------------------------------------------------------
# Corrections tests
# ---------------------------------------------------------------------------

class TestTokenCorrection:
    def test_changed_fields(self):
        c = TokenCorrection(
            sent_idx=0, token_idx=0, form="arma",
            old_lemma="arma", new_lemma="armum",
            old_upos="NOUN", new_upos="NOUN",
        )
        assert c.changed_fields == ["lemma"]
        assert c.has_changes

    def test_no_changes(self):
        c = TokenCorrection(
            sent_idx=0, token_idx=0, form="arma",
            old_lemma="arma", new_lemma="arma",
        )
        assert c.changed_fields == []
        assert not c.has_changes

    def test_multiple_changes(self):
        c = TokenCorrection(
            sent_idx=0, token_idx=0, form="arma",
            old_lemma="arma", new_lemma="armum",
            old_upos="NOUN", new_upos="VERB",
            old_ner="O", new_ner="B-PER",
        )
        assert set(c.changed_fields) == {"lemma", "upos", "ner"}


class TestCorrectionSubmission:
    def test_validate_good(self):
        sub = CorrectionSubmission(
            fileid="test.tess",
            corrections=[
                TokenCorrection(
                    sent_idx=0, token_idx=0, form="arma",
                    old_lemma="arma", new_lemma="armum",
                ),
            ],
        )
        assert sub.validate() == []

    def test_validate_bad_upos(self):
        sub = CorrectionSubmission(
            fileid="test.tess",
            corrections=[
                TokenCorrection(
                    sent_idx=0, token_idx=0, form="arma",
                    old_upos="NOUN", new_upos="INVALID",
                ),
            ],
        )
        errors = sub.validate()
        assert any("Invalid UPOS" in e for e in errors)

    def test_roundtrip_dict(self):
        sub = CorrectionSubmission(
            fileid="test.tess",
            submitted_by="tester",
            corrections=[
                TokenCorrection(
                    sent_idx=0, token_idx=1, form="virum",
                    old_lemma="vir", new_lemma="virus",
                ),
            ],
        )
        d = sub.to_dict()
        sub2 = CorrectionSubmission.from_dict(d)
        assert sub2.fileid == sub.fileid
        assert sub2.corrections[0].new_lemma == "virus"


class TestCorrectionStore:
    def test_submit_and_list(self, tmp_store: CorrectionStore):
        sub = CorrectionSubmission(
            fileid="test.tess",
            corrections=[
                TokenCorrection(
                    sent_idx=0, token_idx=0, form="arma",
                    old_lemma="arma", new_lemma="armum",
                ),
            ],
        )
        tmp_store.submit(sub)
        pending = tmp_store.list_pending()
        assert len(pending) == 1
        assert pending[0].fileid == "test.tess"

    def test_submit_empty_rejected(self, tmp_store: CorrectionStore):
        sub = CorrectionSubmission(
            fileid="test.tess",
            corrections=[
                TokenCorrection(
                    sent_idx=0, token_idx=0, form="arma",
                    old_lemma="arma", new_lemma="arma",  # no change
                ),
            ],
        )
        with pytest.raises(ValueError, match="No actual changes"):
            tmp_store.submit(sub)

    def test_accept(self, tmp_store: CorrectionStore):
        sub = CorrectionSubmission(
            fileid="test.tess",
            corrections=[
                TokenCorrection(
                    sent_idx=0, token_idx=0, form="arma",
                    old_lemma="arma", new_lemma="armum",
                ),
            ],
        )
        tmp_store.submit(sub)
        result = tmp_store.accept(sub.submission_id, "Looks good")
        assert result.status == "accepted"
        assert result.reviewer_notes == "Looks good"
        assert len(tmp_store.list_pending()) == 0
        assert len(tmp_store.list_accepted()) == 1

    def test_reject(self, tmp_store: CorrectionStore):
        sub = CorrectionSubmission(
            fileid="test.tess",
            corrections=[
                TokenCorrection(
                    sent_idx=0, token_idx=0, form="arma",
                    old_lemma="arma", new_lemma="armum",
                ),
            ],
        )
        tmp_store.submit(sub)
        result = tmp_store.reject(sub.submission_id, "Wrong lemma")
        assert result.status == "rejected"
        assert len(tmp_store.list_pending()) == 0
        assert len(tmp_store.list_rejected()) == 1

    def test_get_by_id(self, tmp_store: CorrectionStore):
        sub = CorrectionSubmission(
            fileid="test.tess",
            corrections=[
                TokenCorrection(
                    sent_idx=0, token_idx=0, form="arma",
                    old_lemma="arma", new_lemma="armum",
                ),
            ],
        )
        tmp_store.submit(sub)
        found = tmp_store.get(sub.submission_id)
        assert found is not None
        assert found.fileid == "test.tess"

    def test_filter_by_fileid(self, tmp_store: CorrectionStore):
        for fid in ("a.tess", "b.tess", "a.tess"):
            sub = CorrectionSubmission(
                fileid=fid,
                corrections=[
                    TokenCorrection(
                        sent_idx=0, token_idx=0, form="x",
                        old_lemma="x", new_lemma="y",
                    ),
                ],
            )
            tmp_store.submit(sub)
        assert len(tmp_store.list_pending("a.tess")) == 2
        assert len(tmp_store.list_pending("b.tess")) == 1


# ---------------------------------------------------------------------------
# Apply corrections to .conlluc
# ---------------------------------------------------------------------------

class TestApplyCorrections:
    def test_apply_lemma_correction(self):
        corrections = [
            TokenCorrection(
                sent_idx=0, token_idx=1, form="virumque",
                old_lemma="vir", new_lemma="virus",
            ),
        ]
        result = apply_corrections_to_conlluc(SAMPLE_CONLLUC, corrections)

        # The lemma should be changed
        assert "virus" in result
        # Corrected= should be in MISC
        assert "Corrected=lemma" in result
        # corrections count should be updated
        assert "# corrections = 1" in result

    def test_apply_upos_correction(self):
        corrections = [
            TokenCorrection(
                sent_idx=0, token_idx=0, form="Arma",
                old_upos="NOUN", new_upos="VERB",
            ),
        ]
        result = apply_corrections_to_conlluc(SAMPLE_CONLLUC, corrections)
        # Find the Arma line
        for line in result.splitlines():
            if line.startswith("1\tArma"):
                parts = line.split("\t")
                assert parts[3] == "VERB"
                assert "Corrected=upos" in parts[9]
                break
        else:
            pytest.fail("Arma token not found")

    def test_apply_multiple_fields(self):
        corrections = [
            TokenCorrection(
                sent_idx=0, token_idx=2, form="cano",
                old_lemma="canō", new_lemma="canis",
                old_upos="VERB", new_upos="NOUN",
                old_feats="Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin|Voice=Act",
                new_feats="Case=Nom|Gender=Masc|Number=Sing",
            ),
        ]
        result = apply_corrections_to_conlluc(SAMPLE_CONLLUC, corrections)
        for line in result.splitlines():
            if line.startswith("3\tcano"):
                parts = line.split("\t")
                assert parts[2] == "canis"
                assert parts[3] == "NOUN"
                assert parts[5] == "Case=Nom|Gender=Masc|Number=Sing"
                assert "Corrected=feats,lemma,upos" in parts[9]
                break
        else:
            pytest.fail("cano token not found")

    def test_apply_ner_correction(self):
        corrections = [
            TokenCorrection(
                sent_idx=1, token_idx=0, form="Troiae",
                old_ner="O", new_ner="B-GPE",
            ),
        ]
        result = apply_corrections_to_conlluc(SAMPLE_CONLLUC, corrections)
        for line in result.splitlines():
            if line.startswith("1\tTroiae"):
                parts = line.split("\t")
                assert "NER=B-GPE" in parts[9]
                assert "Corrected=ner" in parts[9]
                break
        else:
            pytest.fail("Troiae token not found")

    def test_apply_no_changes(self):
        result = apply_corrections_to_conlluc(SAMPLE_CONLLUC, [])
        assert result == SAMPLE_CONLLUC

    def test_corrections_across_sentences(self):
        corrections = [
            TokenCorrection(
                sent_idx=0, token_idx=0, form="Arma",
                old_lemma="arma", new_lemma="armum",
            ),
            TokenCorrection(
                sent_idx=1, token_idx=0, form="Troiae",
                old_lemma="Trōia", new_lemma="Troia",
            ),
        ]
        result = apply_corrections_to_conlluc(SAMPLE_CONLLUC, corrections)
        assert "armum" in result
        assert "Troia" in result
        assert "# corrections = 2" in result


# ---------------------------------------------------------------------------
# Editor parser tests
# ---------------------------------------------------------------------------

class TestEditorParser:
    def test_parse_file_meta(self):
        parsed = parse_conlluc_for_editor(SAMPLE_CONLLUC)
        assert parsed["file_meta"]["generator"] == "latincy-readers"
        assert parsed["file_meta"]["model_name"] == "la_core_web_lg"

    def test_parse_sentences(self):
        parsed = parse_conlluc_for_editor(SAMPLE_CONLLUC)
        assert len(parsed["sentences"]) == 2

    def test_parse_tokens(self):
        parsed = parse_conlluc_for_editor(SAMPLE_CONLLUC)
        sent0 = parsed["sentences"][0]
        assert len(sent0["tokens"]) == 4
        assert sent0["tokens"][0]["form"] == "Arma"
        assert sent0["tokens"][0]["lemma"] == "arma"
        assert sent0["tokens"][0]["upos"] == "NOUN"
        assert sent0["tokens"][0]["xpos"] == "n"

    def test_parse_morph_feats(self):
        parsed = parse_conlluc_for_editor(SAMPLE_CONLLUC)
        token = parsed["sentences"][0]["tokens"][0]
        assert token["feats"] == "Case=Acc|Gender=Neut|Number=Plur"

    def test_parse_ner_default(self):
        parsed = parse_conlluc_for_editor(SAMPLE_CONLLUC)
        # No NER in MISC, should default to "O"
        assert parsed["sentences"][0]["tokens"][0]["ner"] == "O"

    def test_parse_corrected_field(self):
        conlluc_with_correction = SAMPLE_CONLLUC.replace(
            "1\tArma\tarma\tNOUN\tn\tCase=Acc|Gender=Neut|Number=Plur\t3\tobj\t_\t_",
            "1\tArma\tarmum\tNOUN\tn\tCase=Acc|Gender=Neut|Number=Plur\t3\tobj\t_\tCorrected=lemma",
        )
        parsed = parse_conlluc_for_editor(conlluc_with_correction)
        token = parsed["sentences"][0]["tokens"][0]
        assert "lemma" in token["corrected"]
        assert token["lemma"] == "armum"

    def test_parse_sent_meta(self):
        parsed = parse_conlluc_for_editor(SAMPLE_CONLLUC)
        assert parsed["sentences"][0]["meta"]["text"] == "Arma virumque cano."
        assert parsed["sentences"][1]["meta"]["sent_id"] == "test.tess:2"
