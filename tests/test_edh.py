"""Tests for EDHReader (Epigraphic Database Heidelberg)."""

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# fileids
# ---------------------------------------------------------------------------

class TestEDHReaderFileids:
    def test_fileids_returns_xml_files(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        ids = reader.fileids()
        assert len(ids) == 2
        assert any("HD000001.xml" in f for f in ids)
        assert any("HD000002.xml" in f for f in ids)

    def test_root_is_path(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=str(edh_dir), annotation_level=0)
        assert len(reader.fileids()) == 2


# ---------------------------------------------------------------------------
# texts — Leiden normalization
# ---------------------------------------------------------------------------

class TestEDHReaderTexts:
    def test_texts_yields_strings(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        texts = list(reader.texts())
        assert len(texts) >= 1
        assert all(isinstance(t, str) for t in texts)

    def test_texts_skip_greek_inscription(self, edh_dir):
        """HD000002.xml has only a Greek edition div; texts() should skip it."""
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        texts = list(reader.texts())
        assert len(texts) == 1

    def test_texts_expand_abbreviations(self, edh_dir):
        """D(is) M(anibus) → 'Dis Manibus'."""
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        text = list(reader.texts())[0]
        assert "Dis" in text
        assert "Manibus" in text

    def test_texts_include_supplied_text(self, edh_dir):
        """<supplied reason='lost'>Gaio</supplied> → 'Gaio'."""
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        text = list(reader.texts())[0]
        assert "Gaio" in text

    def test_texts_exclude_erased_text(self, edh_dir):
        """<del>erasum</del> → not in output."""
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        text = list(reader.texts())[0]
        assert "erasum" not in text

    def test_texts_contain_plain_latin(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        text = list(reader.texts())[0]
        assert "parentibus" in text
        assert "pientissimis" in text

    def test_texts_no_xml_tags(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        for text in reader.texts():
            assert "<" not in text
            assert ">" not in text


# ---------------------------------------------------------------------------
# headers — zero-NLP metadata
# ---------------------------------------------------------------------------

class TestEDHReaderHeaders:
    def test_headers_yields_dicts(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        headers = list(reader.headers())
        assert len(headers) >= 1
        assert all(isinstance(h, dict) for h in headers)

    def test_headers_skip_non_latin(self, edh_dir):
        """Only Latin inscriptions should appear in headers()."""
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        headers = list(reader.headers())
        assert len(headers) == 1

    def test_headers_extract_hd_nr(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        header = list(reader.headers())[0]
        assert header.get("hd_nr") == "HD000001"

    def test_headers_extract_not_before(self, edh_dir):
        """notBefore-custom='0071' → not_before='71'."""
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        header = list(reader.headers())[0]
        assert header.get("not_before") == "71"

    def test_headers_extract_not_after(self, edh_dir):
        """notAfter-custom='0130' → not_after='130'."""
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        header = list(reader.headers())[0]
        assert header.get("not_after") == "130"

    def test_headers_extract_province(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        header = list(reader.headers())[0]
        assert header.get("province") == "Latium et Campania (Regio I)"

    def test_headers_extract_type_of_inscription(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        header = list(reader.headers())[0]
        assert header.get("type_of_inscription") == "epitaph"

    def test_headers_include_filename(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir, annotation_level=0)
        header = list(reader.headers())[0]
        assert "HD000001.xml" in header.get("filename", "")


# ---------------------------------------------------------------------------
# docs — NLP + line spans
# ---------------------------------------------------------------------------

class TestEDHReaderDocs:
    def test_docs_yield_spacy_docs(self, edh_dir):
        from latincyreaders import EDHReader
        from spacy.tokens import Doc
        reader = EDHReader(root=edh_dir)
        docs = list(reader.docs())
        assert len(docs) == 1
        assert isinstance(docs[0], Doc)

    def test_docs_have_fileid(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir)
        doc = list(reader.docs())[0]
        assert doc._.fileid is not None
        assert "HD000001" in doc._.fileid

    def test_docs_have_hd_nr_in_metadata(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir)
        doc = list(reader.docs())[0]
        assert doc._.metadata is not None
        assert doc._.metadata.get("hd_nr") == "HD000001"

    def test_docs_have_line_spans(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir)
        doc = list(reader.docs())[0]
        assert "lines" in doc.spans
        assert len(doc.spans["lines"]) >= 1

    def test_line_spans_have_citations(self, edh_dir):
        """Line citations follow 'HD000001.N' pattern."""
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir)
        doc = list(reader.docs())[0]
        citations = [span._.citation for span in doc.spans["lines"]]
        assert "HD000001.1" in citations
        assert "HD000001.2" in citations

    def test_line_spans_cover_abbreviated_words(self, edh_dir):
        """Line 1 span should contain 'Dis' or 'Manibus'."""
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir)
        doc = list(reader.docs())[0]
        line1 = next(s for s in doc.spans["lines"] if s._.citation == "HD000001.1")
        assert any(tok.text in ("Dis", "Manibus") for tok in line1)

    def test_docs_contain_tokens(self, edh_dir):
        from latincyreaders import EDHReader
        reader = EDHReader(root=edh_dir)
        doc = list(reader.docs())[0]
        assert len(doc) > 0
