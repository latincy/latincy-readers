"""Tests for TxtdownReader."""

import pytest
from pathlib import Path

from latincyreaders import TxtdownReader, AnnotationLevel


class TestTxtdownReader:
    """Test suite for TxtdownReader."""

    @pytest.fixture
    def reader(self, txtdown_dir):
        """Create a TxtdownReader with test fixtures."""
        return TxtdownReader(
            root=txtdown_dir,
            fileids="sample.txtd",
            annotation_level=AnnotationLevel.BASIC,
        )

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of .txtd files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".txtd") for f in fileids)

    def test_fileids_contains_test_file(self, reader):
        """Test fixture file is discovered."""
        fileids = reader.fileids()
        assert "sample.txtd" in fileids

    def test_root_is_path(self, reader, txtdown_dir):
        """root property returns correct Path."""
        assert reader.root == txtdown_dir.resolve()

    # -------------------------------------------------------------------------
    # Text access
    # -------------------------------------------------------------------------

    def test_texts_yields_strings(self, reader):
        """texts() yields raw strings."""
        texts = list(reader.texts())
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_texts_contains_latin(self, reader):
        """Text content is Latin."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        # Check for known content from test file
        assert "Vivamus" in all_text or "Lesbia" in all_text

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def test_metadata_from_front_matter(self, reader):
        """Metadata is extracted from YAML front matter."""
        doc = next(reader.docs())
        assert doc._.metadata.get("author") == "Catullus"
        assert doc._.metadata.get("work") == "Carmina"
        assert doc._.metadata.get("source") == "Latin Library"

    def test_metadata_includes_sections(self, reader):
        """Metadata includes section information."""
        doc = next(reader.docs())
        sections = doc._.metadata.get("sections", [])
        assert len(sections) == 2
        assert sections[0]["id"] == "1"
        assert sections[0]["title"] == "Carmen I"
        assert sections[1]["id"] == "2"
        assert sections[1]["title"] == "Carmen II"

    # -------------------------------------------------------------------------
    # spaCy Doc access
    # -------------------------------------------------------------------------

    def test_docs_yields_spacy_docs(self, reader):
        """docs() yields spaCy Doc objects."""
        from spacy.tokens import Doc

        docs = list(reader.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_have_fileid(self, reader):
        """Docs have fileid custom attribute."""
        doc = next(reader.docs())
        assert hasattr(doc._, "fileid")
        assert doc._.fileid == "sample.txtd"

    def test_docs_have_metadata(self, reader):
        """Docs have metadata custom attribute."""
        doc = next(reader.docs())
        assert hasattr(doc._, "metadata")
        assert isinstance(doc._.metadata, dict)

    def test_sents_yields_spans(self, reader):
        """sents() yields sentence Spans."""
        from spacy.tokens import Span

        sents = list(reader.sents())
        assert len(sents) > 0
        assert all(isinstance(s, Span) for s in sents)

    def test_tokens_yields_tokens(self, reader):
        """tokens() yields Token objects."""
        from spacy.tokens import Token

        tokens = list(reader.tokens())
        assert len(tokens) > 0
        assert all(isinstance(t, Token) for t in tokens)

    # -------------------------------------------------------------------------
    # Annotation levels
    # -------------------------------------------------------------------------

    def test_annotation_level_none_blocks_docs(self, txtdown_dir):
        """annotation_level=NONE prevents docs() usage."""
        reader = TxtdownReader(
            root=txtdown_dir,
            fileids="*.txtd",
            annotation_level=AnnotationLevel.NONE,
        )
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            next(reader.docs())

    def test_annotation_level_none_allows_texts(self, txtdown_dir):
        """annotation_level=NONE still allows texts()."""
        reader = TxtdownReader(
            root=txtdown_dir,
            fileids="*.txtd",
            annotation_level=AnnotationLevel.NONE,
        )
        texts = list(reader.texts())
        assert len(texts) > 0


class TestTxtdownSentsWithCitations:
    """Tests for sents_with_citations method."""

    @pytest.fixture
    def reader(self, txtdown_dir):
        """Create a TxtdownReader with sample.txtd fixture."""
        return TxtdownReader(
            root=txtdown_dir,
            fileids="sample.txtd",
            annotation_level=AnnotationLevel.BASIC,
        )

    def test_sents_with_citations_yields_dicts(self, reader):
        """sents_with_citations() yields dicts."""
        sents = list(reader.sents_with_citations())
        assert len(sents) > 0
        assert all(isinstance(s, dict) for s in sents)

    def test_sents_with_citations_has_sentence_text(self, reader):
        """Each result has sentence text."""
        sent = next(reader.sents_with_citations())
        assert "sentence" in sent
        assert isinstance(sent["sentence"], str)
        assert len(sent["sentence"]) > 0

    def test_sents_with_citations_has_metadata(self, reader):
        """Each result has metadata."""
        sent = next(reader.sents_with_citations())
        assert "metadata" in sent
        assert "fileid" in sent

    def test_sents_with_citations_has_author(self, reader):
        """Metadata includes author from front matter."""
        sent = next(reader.sents_with_citations())
        assert sent["metadata"].get("author") == "Catullus"


class TestTxtdownBlockquotes:
    """Tests for blockquote handling in txtdown files."""

    @pytest.fixture
    def reader(self, txtdown_dir):
        """Create a TxtdownReader with blockquote fixture."""
        return TxtdownReader(
            root=txtdown_dir,
            fileids="blockquote.txtd",
            annotation_level=AnnotationLevel.BASIC,
        )

    def test_blockquote_stripped_from_text(self, reader):
        """Blockquote markers (>) are stripped from text."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert ">" not in all_text

    def test_blockquote_joins_with_preceding(self, reader):
        """Blockquote line joins with preceding text."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        # The blockquote should join to form a continuous sentence
        assert "per aras Sanguine" in all_text

    def test_consecutive_blockquotes_join(self, reader):
        """Multiple consecutive blockquote lines join together."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        # Should have continuation joined
        assert "continuation that spans" in all_text

    def test_blockquote_sentence_segmentation(self, reader):
        """Sentence segmentation works correctly across blockquotes."""
        sents = [s.text for s in reader.sents()]
        # Find the sentence containing the Virgil quote
        virgil_sent = None
        for sent in sents:
            if "Priamum" in sent and "Sanguine" in sent:
                virgil_sent = sent
                break
        assert virgil_sent is not None, "Blockquote should join into single sentence"
        # Should be one continuous sentence without > marker
        assert ">" not in virgil_sent


class TestTxtdownImportError:
    """Tests for when txtdown package is not available."""

    def test_import_error_message(self, monkeypatch, txtdown_dir):
        """Clear error message when txtdown not installed."""
        # Simulate txtdown not being available
        import latincyreaders.readers.txtdown as txtdown_module
        monkeypatch.setattr(txtdown_module, "TXTDOWN_AVAILABLE", False)

        with pytest.raises(ImportError, match="txtdown package required"):
            TxtdownReader(root=txtdown_dir)
