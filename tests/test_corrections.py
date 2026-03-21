"""Tests for correction tracking (extract, save, load, apply)."""

import json

import pytest

from latincyreaders.cache.corrections import (
    CorrectionSet,
    TokenCorrection,
    apply_corrections,
    extract_corrections,
    load_corrections,
    save_corrections,
)


# -- Fixtures ---------------------------------------------------------------

BASELINE_CONLLUC = """\
# generator = latincy-readers
# annotation_status = silver
# do_not_use_for_training = true
# model_name = la_core_web_lg
# model_version = 3.8.0
# generated = 2026-03-20T10:00:00+00:00
# collection = tesserae
# fileid = test.tess
# corrections = 0

# sent_id = test.tess:1
# text = ter Marte sinistro
1	ter	ter	ADV	adverb	_	0	ROOT	_	_
2	Marte	Mars	PROPN	proper_noun	Case=Abl|Gender=Masc|Number=Sing	1	obl	_	NER=B-PERSON
3	sinistro	sinister	ADJ	adjective	Case=Abl|Gender=Masc|Number=Sing	2	amod	_	_

# sent_id = test.tess:2
# text = arma uirumque cano
1	arma	arma	NOUN	noun	Case=Acc|Gender=Neut|Number=Plur	3	obj	_	_
2	uirumque	uir	NOUN	noun	Case=Acc|Gender=Masc|Number=Sing	3	obj	_	_
3	cano	cano	VERB	verb	Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin|Voice=Act	0	ROOT	_	_
"""

# User corrected: Marte PROPN→NOUN, Mars→mars
CORRECTED_CONLLUC = """\
# generator = latincy-readers
# annotation_status = silver
# do_not_use_for_training = true
# model_name = la_core_web_lg
# model_version = 3.8.0
# generated = 2026-03-20T10:00:00+00:00
# collection = tesserae
# fileid = test.tess
# corrections = 0

# sent_id = test.tess:1
# text = ter Marte sinistro
1	ter	ter	ADV	adverb	_	0	ROOT	_	_
2	Marte	mars	NOUN	noun	Case=Abl|Gender=Masc|Number=Sing	1	obl	_	_
3	sinistro	sinister	ADJ	adjective	Case=Abl|Gender=Masc|Number=Sing	2	amod	_	_

# sent_id = test.tess:2
# text = arma uirumque cano
1	arma	arma	NOUN	noun	Case=Acc|Gender=Neut|Number=Plur	3	obj	_	_
2	uirumque	uir	NOUN	noun	Case=Acc|Gender=Masc|Number=Sing	3	obj	_	_
3	cano	cano	VERB	verb	Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin|Voice=Act	0	ROOT	_	_
"""

# Re-annotated with new model (different baseline, but same Marte error)
NEW_MODEL_CONLLUC = """\
# generator = latincy-readers
# annotation_status = silver
# do_not_use_for_training = true
# model_name = la_core_web_lg
# model_version = 3.9.0
# generated = 2026-03-20T12:00:00+00:00
# collection = tesserae
# fileid = test.tess
# corrections = 0

# sent_id = test.tess:1
# text = ter Marte sinistro
1	ter	ter	ADV	adverb	_	0	ROOT	_	_
2	Marte	Mars	PROPN	proper_noun	Case=Abl|Gender=Masc|Number=Sing	1	obl	_	NER=B-PERSON
3	sinistro	sinister	ADJ	adjective	Case=Abl|Gender=Masc|Number=Sing	2	amod	_	_

# sent_id = test.tess:2
# text = arma uirumque cano
1	arma	arma	NOUN	noun	Case=Acc|Gender=Neut|Number=Plur	3	obj	_	_
2	uirumque	uir	NOUN	noun	Case=Acc|Gender=Masc|Number=Sing	3	obj	_	_
3	cano	cano	VERB	verb	Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin|Voice=Act	0	ROOT	_	_
"""

# New model already agrees with the correction on lemma but not upos
NEW_MODEL_PARTIAL_AGREE = """\
# generator = latincy-readers
# annotation_status = silver
# do_not_use_for_training = true
# model_name = la_core_web_lg
# model_version = 3.9.0
# generated = 2026-03-20T12:00:00+00:00
# collection = tesserae
# fileid = test.tess
# corrections = 0

# sent_id = test.tess:1
# text = ter Marte sinistro
1	ter	ter	ADV	adverb	_	0	ROOT	_	_
2	Marte	mars	PROPN	proper_noun	Case=Abl|Gender=Masc|Number=Sing	1	obl	_	NER=B-PERSON
3	sinistro	sinister	ADJ	adjective	Case=Abl|Gender=Masc|Number=Sing	2	amod	_	_

# sent_id = test.tess:2
# text = arma uirumque cano
1	arma	arma	NOUN	noun	Case=Acc|Gender=Neut|Number=Plur	3	obj	_	_
2	uirumque	uir	NOUN	noun	Case=Acc|Gender=Masc|Number=Sing	3	obj	_	_
3	cano	cano	VERB	verb	Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin|Voice=Act	0	ROOT	_	_
"""


@pytest.fixture
def baseline_file(tmp_path):
    p = tmp_path / "test.conlluc"
    p.write_text(BASELINE_CONLLUC, encoding="utf-8")
    return p


@pytest.fixture
def corrected_file(tmp_path):
    p = tmp_path / "test_corrected.conlluc"
    p.write_text(CORRECTED_CONLLUC, encoding="utf-8")
    return p


# -- Tests -------------------------------------------------------------------


class TestExtractCorrections:
    def test_detects_lemma_and_upos_change(self, baseline_file, corrected_file):
        cset = extract_corrections(baseline_file, corrected_file)
        assert cset.count == 1
        corr = cset.corrections[0]
        assert corr.sent_id == "test.tess:1"
        assert corr.token_idx == 2
        assert corr.token_form == "Marte"
        assert corr.changes["lemma"] == {"from": "Mars", "to": "mars"}
        assert corr.changes["upos"] == {"from": "PROPN", "to": "NOUN"}
        assert corr.changes["xpos"] == {"from": "proper_noun", "to": "noun"}

    def test_preserves_model_info(self, baseline_file, corrected_file):
        cset = extract_corrections(baseline_file, corrected_file)
        assert cset.model_name == "la_core_web_lg"
        assert cset.model_version == "3.8.0"
        assert cset.fileid == "test.tess"

    def test_no_changes_returns_empty(self, baseline_file):
        cset = extract_corrections(baseline_file, baseline_file)
        assert cset.count == 0


class TestSaveLoadCorrections:
    def test_roundtrip(self, tmp_path):
        conlluc_path = tmp_path / "test.conlluc"
        conlluc_path.write_text("dummy", encoding="utf-8")

        cset = CorrectionSet(
            fileid="test.tess",
            model_name="la_core_web_lg",
            model_version="3.8.0",
            corrections=[
                TokenCorrection(
                    sent_id="test.tess:1",
                    token_idx=2,
                    token_form="Marte",
                    changes={"upos": {"from": "PROPN", "to": "NOUN"}},
                ),
            ],
        )

        out = save_corrections(cset, conlluc_path)
        assert out.suffix == ".json"
        assert out.exists()

        loaded = load_corrections(conlluc_path)
        assert loaded is not None
        assert loaded.count == 1
        assert loaded.corrections[0].token_form == "Marte"
        assert loaded.corrections[0].changes["upos"]["to"] == "NOUN"

    def test_load_missing_returns_none(self, tmp_path):
        conlluc_path = tmp_path / "nonexistent.conlluc"
        assert load_corrections(conlluc_path) is None


class TestApplyCorrections:
    def test_apply_to_new_model(self, baseline_file, corrected_file, tmp_path):
        # Extract corrections from old baseline vs user edit
        cset = extract_corrections(baseline_file, corrected_file)
        assert cset.count == 1

        # Write new model output
        new_file = tmp_path / "new_model.conlluc"
        new_file.write_text(NEW_MODEL_CONLLUC, encoding="utf-8")

        # Apply corrections
        applied, skipped = apply_corrections(new_file, cset)
        assert applied == 1
        assert skipped == 0

        # Verify the file was patched
        content = new_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        # Find the Marte line
        marte_line = [l for l in lines if l.startswith("2\tMarte")][0]
        parts = marte_line.split("\t")
        assert parts[2] == "mars"      # lemma corrected
        assert parts[3] == "NOUN"      # upos corrected
        assert parts[4] == "noun"      # xpos corrected
        assert "Corrected=Yes" in parts[9]

        # corrections count updated in header
        assert "# corrections = 1" in content

    def test_skip_when_model_agrees(self, baseline_file, corrected_file, tmp_path):
        cset = extract_corrections(baseline_file, corrected_file)

        # New model already partially agrees
        new_file = tmp_path / "partial.conlluc"
        new_file.write_text(NEW_MODEL_PARTIAL_AGREE, encoding="utf-8")

        applied, skipped = apply_corrections(new_file, cset)
        # Still applied because upos/xpos differ (even if lemma agrees)
        assert applied == 1

        content = new_file.read_text(encoding="utf-8")
        marte_line = [l for l in content.splitlines() if l.startswith("2\tMarte")][0]
        parts = marte_line.split("\t")
        assert parts[2] == "mars"   # already correct, kept
        assert parts[3] == "NOUN"   # corrected
        assert parts[4] == "noun"   # corrected

    def test_token_form_mismatch_skips(self, tmp_path, caplog):
        cset = CorrectionSet(
            fileid="test.tess",
            corrections=[
                TokenCorrection(
                    sent_id="test.tess:1",
                    token_idx=2,
                    token_form="WRONG_FORM",
                    changes={"upos": {"from": "PROPN", "to": "NOUN"}},
                ),
            ],
        )
        new_file = tmp_path / "test.conlluc"
        new_file.write_text(NEW_MODEL_CONLLUC, encoding="utf-8")

        import logging
        with caplog.at_level(logging.WARNING):
            applied, skipped = apply_corrections(new_file, cset)

        assert applied == 0
        assert skipped == 1
        assert "form mismatch" in caplog.text


class TestFullWorkflow:
    """End-to-end: edit → extract → save → reload → apply to new model."""

    def test_full_migration(self, tmp_path):
        # 1. Original baseline from old model
        baseline = tmp_path / "baseline.conlluc"
        baseline.write_text(BASELINE_CONLLUC, encoding="utf-8")

        # 2. User edits the file
        corrected = tmp_path / "silius.conlluc"
        corrected.write_text(CORRECTED_CONLLUC, encoding="utf-8")

        # 3. Extract corrections
        cset = extract_corrections(baseline, corrected)
        assert cset.count == 1

        # 4. Save corrections
        save_corrections(cset, corrected)

        # 5. Re-annotate with new model (overwrite the file)
        corrected.write_text(NEW_MODEL_CONLLUC, encoding="utf-8")

        # 6. Load corrections and re-apply
        loaded = load_corrections(corrected)
        assert loaded is not None
        applied, skipped = apply_corrections(corrected, loaded)
        assert applied == 1

        # 7. Verify final state
        content = corrected.read_text(encoding="utf-8")
        marte_line = [l for l in content.splitlines() if l.startswith("2\tMarte")][0]
        parts = marte_line.split("\t")
        assert parts[2] == "mars"
        assert parts[3] == "NOUN"
        assert "Corrected=Yes" in parts[9]
