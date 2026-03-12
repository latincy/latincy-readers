"""Tests for the .conlluc (CoNLL-U Cache) format."""

import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from latincyreaders.cache.conlluc import (
    CONLLUC_EXTENSION,
    conlluc_to_doc,
    doc_to_conlluc,
    read_conlluc,
    validate_conlluc_header,
    write_conlluc,
)


class TestDocToConlluc:
    """Tests for serializing spaCy Docs to .conlluc format."""

    @pytest.fixture
    def vocab(self):
        return Vocab()

    @pytest.fixture
    def annotated_doc(self, vocab):
        doc = Doc(
            vocab,
            words=["Gallia", "est", "omnis", "divisa", "."],
            spaces=[True, True, True, False, False],
            sent_starts=[True, False, False, False, False],
        )
        doc[0].lemma_ = "Gallia"
        doc[0].pos_ = "PROPN"
        doc[1].lemma_ = "sum"
        doc[1].pos_ = "AUX"
        doc[2].lemma_ = "omnis"
        doc[2].pos_ = "ADJ"
        doc[3].lemma_ = "divido"
        doc[3].pos_ = "VERB"
        doc[4].lemma_ = "."
        doc[4].pos_ = "PUNCT"
        doc._.fileid = "caesar.gal.tess"
        return doc

    def test_file_header_present(self, annotated_doc):
        text = doc_to_conlluc(annotated_doc, fileid="caesar.gal.tess")
        assert "# generator = latincy-readers" in text
        assert "# annotation_status = silver" in text
        assert "# do_not_use_for_training = true" in text

    def test_model_info_in_header(self, annotated_doc):
        text = doc_to_conlluc(
            annotated_doc,
            fileid="caesar.gal.tess",
            model_name="la_core_web_lg",
            model_version="3.7.0",
        )
        assert "# model_name = la_core_web_lg" in text
        assert "# model_version = 3.7.0" in text

    def test_collection_in_header(self, annotated_doc):
        text = doc_to_conlluc(
            annotated_doc,
            fileid="caesar.gal.tess",
            collection="cltk-tesserae",
        )
        assert "# collection = cltk-tesserae" in text

    def test_corrections_in_header(self, annotated_doc):
        text = doc_to_conlluc(annotated_doc, corrections=5)
        assert "# corrections = 5" in text

    def test_sentence_metadata(self, annotated_doc):
        text = doc_to_conlluc(annotated_doc, fileid="test")
        assert "# sent_id = test:1" in text
        assert "# text = " in text

    def test_token_columns(self, annotated_doc):
        text = doc_to_conlluc(annotated_doc, fileid="test")
        lines = text.strip().split("\n")
        # Find first token line (after header + blank + sent metadata)
        token_lines = [l for l in lines if l and not l.startswith("#")]
        assert len(token_lines) == 5  # 5 tokens
        parts = token_lines[0].split("\t")
        assert len(parts) == 10  # CoNLL-U has 10 columns
        assert parts[0] == "1"  # id
        assert parts[1] == "Gallia"  # form
        assert parts[2] == "Gallia"  # lemma
        assert parts[3] == "PROPN"  # upos

    def test_space_after_no(self, annotated_doc):
        text = doc_to_conlluc(annotated_doc)
        lines = text.strip().split("\n")
        # "divisa" has no trailing space, and "." has no trailing space
        token_lines = [l for l in lines if l and not l.startswith("#")]
        divisa_misc = token_lines[3].split("\t")[9]
        assert divisa_misc == "SpaceAfter=No"

    def test_empty_doc(self, vocab):
        doc = Doc(vocab, words=[], spaces=[])
        text = doc_to_conlluc(doc)
        # Should still have header
        assert "# generator = latincy-readers" in text
        assert "# annotation_status = silver" in text


class TestConllucToDoc:
    """Tests for deserializing .conlluc text back to spaCy Docs."""

    @pytest.fixture
    def vocab(self):
        return Vocab()

    @pytest.fixture
    def sample_conlluc(self):
        return (
            "# generator = latincy-readers\n"
            "# annotation_status = silver\n"
            "# do_not_use_for_training = true\n"
            "# model_name = la_core_web_lg\n"
            "# model_version = 3.7.0\n"
            "# generated = 2026-03-04T00:00:00+00:00\n"
            "# collection = test\n"
            "# fileid = caesar.gal.tess\n"
            "# corrections = 0\n"
            "\n"
            "# sent_id = caesar.gal.tess:1\n"
            "# text = Gallia est omnis divisa.\n"
            "1\tGallia\tGallia\tPROPN\t_\t_\t4\tnsubj\t_\t_\n"
            "2\test\tsum\tAUX\t_\t_\t4\tcop\t_\t_\n"
            "3\tomnis\tomnis\tADJ\t_\t_\t4\tamod\t_\t_\n"
            "4\tdivisa\tdivido\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No\n"
            "5\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\tSpaceAfter=No\n"
            "\n"
        )

    def test_roundtrip_text(self, vocab, sample_conlluc):
        doc, meta = conlluc_to_doc(sample_conlluc, vocab)
        assert doc is not None
        assert doc.text == "Gallia est omnis divisa."

    def test_file_meta_parsed(self, vocab, sample_conlluc):
        _doc, meta = conlluc_to_doc(sample_conlluc, vocab)
        assert meta["generator"] == "latincy-readers"
        assert meta["annotation_status"] == "silver"
        assert meta["do_not_use_for_training"] == "true"
        assert meta["model_name"] == "la_core_web_lg"
        assert meta["fileid"] == "caesar.gal.tess"

    def test_token_annotations(self, vocab, sample_conlluc):
        doc, _meta = conlluc_to_doc(sample_conlluc, vocab)
        assert doc[0].lemma_ == "Gallia"
        assert doc[0].pos_ == "PROPN"
        assert doc[1].lemma_ == "sum"
        assert doc[1].pos_ == "AUX"
        assert doc[3].lemma_ == "divido"
        assert doc[3].pos_ == "VERB"

    def test_dep_heads(self, vocab, sample_conlluc):
        doc, _meta = conlluc_to_doc(sample_conlluc, vocab)
        # Token 0 (Gallia) head = 4 → index 3 (divisa)
        assert doc[0].head.text == "divisa"
        assert doc[0].dep_ == "nsubj"
        # Token 3 (divisa) head = 0 → root (self)
        assert doc[3].head == doc[3]

    def test_doc_extensions(self, vocab, sample_conlluc):
        doc, _meta = conlluc_to_doc(sample_conlluc, vocab)
        assert doc._.fileid == "caesar.gal.tess"
        assert doc._.metadata["annotation_status"] == "silver"

    def test_empty_input(self, vocab):
        doc, meta = conlluc_to_doc("", vocab)
        assert doc is None
        assert meta == {}

    def test_header_only(self, vocab):
        text = (
            "# generator = latincy-readers\n"
            "# annotation_status = silver\n"
            "# do_not_use_for_training = true\n"
            "\n"
        )
        doc, meta = conlluc_to_doc(text, vocab)
        assert doc is None
        assert meta["generator"] == "latincy-readers"


class TestRoundTrip:
    """Test full serialize → deserialize round-trips."""

    @pytest.fixture
    def vocab(self):
        return Vocab()

    def test_roundtrip_preserves_text(self, vocab):
        doc = Doc(
            vocab,
            words=["Arma", "virumque", "cano", "."],
            spaces=[True, True, False, False],
            sent_starts=[True, False, False, False],
        )
        for token in doc:
            token.lemma_ = token.text.lower()
            token.pos_ = "NOUN"

        text = doc_to_conlluc(doc, fileid="vergil.aen")
        loaded, meta = conlluc_to_doc(text, vocab)

        assert loaded is not None
        assert loaded.text == doc.text
        assert len(loaded) == len(doc)

    def test_roundtrip_preserves_annotations(self, vocab):
        doc = Doc(
            vocab,
            words=["Gallia", "est"],
            spaces=[True, False],
            sent_starts=[True, False],
        )
        doc[0].lemma_ = "Gallia"
        doc[0].pos_ = "PROPN"
        doc[1].lemma_ = "sum"
        doc[1].pos_ = "AUX"

        text = doc_to_conlluc(doc, fileid="test", model_name="la_core_web_lg")
        loaded, meta = conlluc_to_doc(text, vocab)

        assert loaded[0].lemma_ == "Gallia"
        assert loaded[0].pos_ == "PROPN"
        assert loaded[1].lemma_ == "sum"
        assert loaded[1].pos_ == "AUX"
        assert meta["model_name"] == "la_core_web_lg"

    def test_roundtrip_multi_sentence(self, vocab):
        doc = Doc(
            vocab,
            words=["Arma", "cano", ".", "Musa", "dic", "."],
            spaces=[True, False, True, True, False, False],
            sent_starts=[True, False, False, True, False, False],
        )
        for t in doc:
            t.lemma_ = t.text.lower()
            t.pos_ = "X"

        text = doc_to_conlluc(doc, fileid="multi")
        loaded, _meta = conlluc_to_doc(text, vocab)

        assert loaded is not None
        sents = list(loaded.sents)
        assert len(sents) == 2
        assert sents[0].text.startswith("Arma")
        assert sents[1].text.startswith("Musa")


class TestDiskIO:
    """Tests for reading/writing .conlluc files to disk."""

    @pytest.fixture
    def vocab(self):
        return Vocab()

    def test_write_and_read(self, tmp_path, vocab):
        doc = Doc(
            vocab,
            words=["Gallia", "est"],
            spaces=[True, False],
            sent_starts=[True, False],
        )
        doc[0].lemma_ = "Gallia"
        doc[0].pos_ = "PROPN"

        path = tmp_path / f"test{CONLLUC_EXTENSION}"
        content = doc_to_conlluc(doc, fileid="test", model_name="la_core_web_lg")
        write_conlluc(path, content)

        loaded, meta = read_conlluc(path, vocab)
        assert loaded is not None
        assert loaded.text == "Gallia est"
        assert meta["model_name"] == "la_core_web_lg"

    def test_file_extension(self):
        assert CONLLUC_EXTENSION == ".conlluc"


class TestValidateHeader:
    """Tests for header validation."""

    def test_valid_header(self):
        meta = {
            "generator": "latincy-readers",
            "annotation_status": "silver",
            "do_not_use_for_training": "true",
        }
        errors = validate_conlluc_header(meta)
        assert errors == []

    def test_missing_required_fields(self):
        errors = validate_conlluc_header({})
        assert len(errors) == 3  # 3 required fields

    def test_wrong_annotation_status(self):
        meta = {
            "generator": "latincy-readers",
            "annotation_status": "gold",
            "do_not_use_for_training": "true",
        }
        errors = validate_conlluc_header(meta)
        assert any("annotation_status" in e for e in errors)

    def test_wrong_training_flag(self):
        meta = {
            "generator": "latincy-readers",
            "annotation_status": "silver",
            "do_not_use_for_training": "false",
        }
        errors = validate_conlluc_header(meta)
        assert any("do_not_use_for_training" in e for e in errors)
