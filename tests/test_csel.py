"""Tests for CSELReader."""

import pytest
from pathlib import Path

from latincyreaders import AnnotationLevel


class TestCSELReaderFileids:
    """File listing and root handling."""

    @pytest.fixture
    def reader(self, csel_dir):
        from latincyreaders import CSELReader
        return CSELReader(root=csel_dir, fileids="*.opp-lat1.xml")

    def test_fileids_returns_opp_lat1_files(self, reader):
        ids = reader.fileids()
        assert isinstance(ids, list)
        assert len(ids) == 1
        assert ids[0].endswith(".opp-lat1.xml")

    def test_root_is_path(self, reader, csel_dir):
        assert reader.root == csel_dir.resolve()


class TestCSELReaderTexts:
    """Raw text extraction."""

    @pytest.fixture
    def reader(self, csel_dir):
        from latincyreaders import CSELReader
        return CSELReader(root=csel_dir, fileids="*.opp-lat1.xml")

    def test_texts_yields_strings(self, reader):
        texts = list(reader.texts())
        assert len(texts) == 1
        assert isinstance(texts[0], str)

    def test_texts_contains_latin(self, reader):
        texts = list(reader.texts())
        assert "Magnus" in texts[0]
        assert "Recordari" in texts[0]

    def test_texts_removes_notes_by_default(self, reader):
        texts = list(reader.texts())
        assert "Ps. 144" not in texts[0]

    def test_texts_normalizes_critical_marks(self, reader):
        """<supplied> markup should be stripped, preserving word."""
        texts = list(reader.texts())
        assert "isto" in texts[0]
        assert "<supplied" not in texts[0]


class TestCSELReaderHeaders:
    """Metadata extraction from teiHeader."""

    @pytest.fixture
    def reader(self, csel_dir):
        from latincyreaders import CSELReader
        return CSELReader(root=csel_dir, fileids="*.opp-lat1.xml")

    def test_headers_yields_dicts(self, reader):
        headers = list(reader.headers())
        assert len(headers) == 1
        assert isinstance(headers[0], dict)

    def test_headers_extract_author(self, reader):
        headers = list(reader.headers())
        assert headers[0]["author"] == "Augustine"

    def test_headers_extract_title(self, reader):
        headers = list(reader.headers())
        assert headers[0]["title"] == "Confessiones"

    def test_headers_extract_cts_urn(self, reader):
        headers = list(reader.headers())
        assert headers[0]["cts_urn"] == "urn:cts:latinLit:stoa0040.stoa001.opp-lat1"

    def test_headers_include_filename(self, reader):
        headers = list(reader.headers())
        assert headers[0]["filename"] == "stoa0040.stoa001.opp-lat1.xml"


class TestCSELReaderDocs:
    """spaCy Doc output and chapter spans."""

    @pytest.fixture
    def reader(self, csel_dir):
        from latincyreaders import CSELReader
        return CSELReader(
            root=csel_dir,
            fileids="*.opp-lat1.xml",
            annotation_level=AnnotationLevel.MINIMAL,
        )

    def test_docs_yields_spacy_docs(self, reader):
        from spacy.tokens import Doc
        docs = list(reader.docs())
        assert len(docs) == 1
        assert isinstance(docs[0], Doc)

    def test_docs_have_fileid(self, reader):
        for doc in reader.docs():
            assert doc._.fileid is not None
            assert doc._.fileid.endswith(".opp-lat1.xml")

    def test_docs_have_chapter_spans(self, reader):
        docs = list(reader.docs())
        doc = docs[0]
        assert "chapters" in doc.spans
        assert len(doc.spans["chapters"]) == 4

    def test_docs_metadata_has_author(self, reader):
        docs = list(reader.docs())
        assert docs[0]._.metadata["author"] == "Augustine"

    def test_docs_metadata_has_title(self, reader):
        docs = list(reader.docs())
        assert docs[0]._.metadata["title"] == "Confessiones"

    def test_docs_metadata_has_cts_urn(self, reader):
        docs = list(reader.docs())
        assert docs[0]._.metadata["cts_urn"] == "urn:cts:latinLit:stoa0040.stoa001.opp-lat1"

    def test_docs_metadata_does_not_leak_private_keys(self, reader):
        """_chapters key should be cleaned from public metadata."""
        for doc in reader.docs():
            assert "_chapters" not in (doc._.metadata or {})


class TestCSELReaderChapters:
    """Chapter citation format and iteration."""

    @pytest.fixture
    def reader(self, csel_dir):
        from latincyreaders import CSELReader
        return CSELReader(
            root=csel_dir,
            fileids="*.opp-lat1.xml",
            annotation_level=AnnotationLevel.MINIMAL,
        )

    def test_chapter_citations_use_subtype_labels(self, reader):
        docs = list(reader.docs())
        citations = [s._.citation for s in docs[0].spans["chapters"]]
        assert "book 1, section 1" in citations
        assert "book 1, section 2" in citations
        assert "book 2, section 1" in citations
        assert "book 2, section 2" in citations

    def test_chapter_text_content(self, reader):
        docs = list(reader.docs())
        ch_map = {s._.citation: s.text for s in docs[0].spans["chapters"]}
        assert "Magnus" in ch_map["book 1, section 1"]
        assert "Recordari" in ch_map["book 2, section 1"]

    def test_chapters_method_yields_spans(self, reader):
        from spacy.tokens import Span
        spans = list(reader.chapters())
        assert len(spans) == 4
        assert all(isinstance(s, Span) for s in spans)

    def test_chapters_as_text_yields_tuples(self, reader):
        items = list(reader.chapters(as_text=True))
        assert len(items) == 4
        citations = [c for c, _ in items]
        assert "book 1, section 1" in citations
        texts = [t for _, t in items]
        assert all(isinstance(t, str) for t in texts)
