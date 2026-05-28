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


    def test_blockquote_section_id_not_none(self, reader):
        """Blockquote sentences must have a valid section_id, not None.

        Regression test: blockquote lines were not found in normalized doc text
        because the > marker was not stripped before searching, causing
        section_id to be lost for sentences containing or following blockquotes.
        """
        sents = list(reader.sents_with_citations())
        for sent in sents:
            assert sent["section_id"] is not None, (
                f"section_id is None for sentence: {sent['sentence']!r}"
            )

    def test_blockquote_line_spans_created(self, reader):
        """Line spans are created for blockquote lines in the Doc."""
        doc = next(reader.docs())
        line_spans = doc.spans.get("lines", [])
        # The blockquote fixture has lines including blockquoted ones;
        # all should produce line spans
        citations = [span._.citation for span in line_spans]
        # Section 1 has 2 lines (one regular, one blockquote)
        assert "1.1" in citations, f"Missing line 1.1 in {citations}"
        assert "1.2" in citations, f"Missing blockquote line 1.2 in {citations}"
        # Section 2 has 3 lines (one regular, two blockquotes)
        assert "2.1" in citations, f"Missing line 2.1 in {citations}"
        assert "2.2" in citations, f"Missing blockquote line 2.2 in {citations}"
        assert "2.3" in citations, f"Missing blockquote line 2.3 in {citations}"

    def test_blockquote_sents_have_correct_section(self, reader):
        """Sentences with blockquote content report the correct section_id."""
        sents = list(reader.sents_with_citations())
        for sent in sents:
            if "Sanguine" in sent["sentence"] or "Priamum" in sent["sentence"]:
                assert sent["section_id"] == "1", (
                    f"Blockquote sentence in section 1 has wrong section_id: "
                    f"{sent['section_id']}"
                )
            if "continuation" in sent["sentence"]:
                assert sent["section_id"] == "2", (
                    f"Blockquote sentence in section 2 has wrong section_id: "
                    f"{sent['section_id']}"
                )


class TestTxtdownCriticalMarkup:
    """Tests for text-critical markup: cruxes (†text†) and additions (<text>)."""

    @pytest.fixture
    def reader(self, txtdown_dir):
        return TxtdownReader(
            root=txtdown_dir,
            fileids="critical_markup.txtd",
            annotation_level=AnnotationLevel.BASIC,
        )

    def test_crux_daggers_stripped_from_text(self, reader):
        """Crux markers (†) are stripped from text output."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "†" not in all_text

    def test_crux_word_preserved_in_text(self, reader):
        """Text inside crux markers is preserved after stripping daggers."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "protulerant" in all_text

    def test_addition_angle_brackets_stripped_from_text(self, reader):
        """Addition markers (<>) are stripped from text output."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "<et>" not in all_text

    def test_addition_word_preserved_in_text(self, reader):
        """Text inside addition markers is preserved after stripping brackets."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "et" in all_text

    def test_docs_process_crux_lines(self, reader):
        """NLP can process lines containing crux markers without error."""
        docs = list(reader.docs())
        assert len(docs) > 0

    def test_line_spans_created_for_crux_lines(self, reader):
        """Line spans are created for lines containing crux markup."""
        doc = next(reader.docs())
        line_spans = doc.spans.get("lines", [])
        citations = [span._.citation for span in line_spans]
        assert "1.4" in citations, f"Missing crux line 1.4 in {citations}"

    def test_line_spans_created_for_addition_lines(self, reader):
        """Line spans are created for lines containing addition markup."""
        doc = next(reader.docs())
        line_spans = doc.spans.get("lines", [])
        citations = [span._.citation for span in line_spans]
        assert "2.1" in citations, f"Missing addition line 2.1 in {citations}"

    def test_section_ids_not_none_with_critical_markup(self, reader):
        """Sentences containing critical markup have valid section_id."""
        sents = list(reader.sents_with_citations())
        for sent in sents:
            assert sent["section_id"] is not None, (
                f"section_id is None for sentence: {sent['sentence']!r}"
            )

    # -- expansions -----------------------------------------------------------

    def test_expansion_combines_prefix_and_content(self, reader):
        """M(arcus) expands to Marcus in text output."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "Marcus" in all_text

    def test_expansion_multi_char_prefix(self, reader):
        """C(aius) expands to Caius."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "Caius" in all_text

    def test_expansion_parentheses_removed(self, reader):
        """Expansion parentheses are not present in output."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "M(arcus)" not in all_text
        assert "C(aius)" not in all_text

    # -- deletions {X} --------------------------------------------------------

    def test_deletion_braces_stripped(self, reader):
        """Deletion markers {} are stripped from text output."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "{" not in all_text
        assert "}" not in all_text

    def test_deletion_content_stripped(self, reader):
        """Content inside {} is removed entirely."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "interpolatione" not in all_text

    # -- lacunae [X] left alone -----------------------------------------------

    def test_lacuna_brackets_preserved(self, reader):
        """Lacuna markers [] are left in place for NLP to handle as-is."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "[" in all_text


class TestTxtdownTextcrit:
    """Tests for doc._.textcrit and derived token attributes."""

    @pytest.fixture
    def doc(self, txtdown_dir):
        reader = TxtdownReader(
            root=txtdown_dir,
            fileids="critical_markup.txtd",
            annotation_level=AnnotationLevel.BASIC,
        )
        return next(reader.docs())

    # -- doc._.textcrit structure ---------------------------------------------

    def test_textcrit_is_dict(self, doc):
        assert isinstance(doc._.textcrit, dict)

    def test_textcrit_has_expected_keys(self, doc):
        for key in ("cruxes", "additions", "expansions", "deletions"):
            assert key in doc._.textcrit

    # -- cruxes ---------------------------------------------------------------

    def test_textcrit_crux_count(self, doc):
        assert len(doc._.textcrit["cruxes"]) == 1

    def test_textcrit_crux_original(self, doc):
        assert doc._.textcrit["cruxes"][0]["original"] == "†protulerant†"

    def test_textcrit_crux_text(self, doc):
        assert doc._.textcrit["cruxes"][0]["text"] == "protulerant"

    def test_textcrit_crux_has_span(self, doc):
        span = doc._.textcrit["cruxes"][0]["span"]
        assert span is not None
        assert isinstance(span, tuple)
        assert len(span) == 2

    def test_token_is_crux(self, doc):
        start, end = doc._.textcrit["cruxes"][0]["span"]
        tokens = list(doc[start:end])
        assert all(t._.is_crux for t in tokens)

    def test_non_crux_tokens_not_flagged(self, doc):
        start, end = doc._.textcrit["cruxes"][0]["span"]
        others = [t for t in doc if t.i < start or t.i >= end]
        assert not any(t._.is_crux for t in others)

    # -- additions ------------------------------------------------------------

    def test_textcrit_addition_count(self, doc):
        assert len(doc._.textcrit["additions"]) == 1

    def test_textcrit_addition_original(self, doc):
        assert doc._.textcrit["additions"][0]["original"] == "<et>"

    def test_textcrit_addition_has_span(self, doc):
        assert doc._.textcrit["additions"][0]["span"] is not None

    def test_token_is_addition(self, doc):
        start, end = doc._.textcrit["additions"][0]["span"]
        assert all(t._.is_addition for t in doc[start:end])

    # -- expansions -----------------------------------------------------------

    def test_textcrit_expansion_count(self, doc):
        assert len(doc._.textcrit["expansions"]) == 2

    def test_textcrit_expansion_texts(self, doc):
        texts = {e["text"] for e in doc._.textcrit["expansions"]}
        assert texts == {"Caius", "Marcus"}

    def test_textcrit_expansion_originals(self, doc):
        originals = {e["original"] for e in doc._.textcrit["expansions"]}
        assert originals == {"C(aius)", "M(arcus)"}

    def test_token_is_expansion(self, doc):
        for entry in doc._.textcrit["expansions"]:
            start, end = entry["span"]
            assert all(t._.is_expansion for t in doc[start:end])

    # -- deletions ------------------------------------------------------------

    def test_textcrit_deletion_count(self, doc):
        assert len(doc._.textcrit["deletions"]) == 1

    def test_textcrit_deletion_original(self, doc):
        assert doc._.textcrit["deletions"][0]["original"] == "{hac interpolatione}"

    def test_textcrit_deletion_text(self, doc):
        assert doc._.textcrit["deletions"][0]["text"] == "hac interpolatione"

    def test_textcrit_deletion_has_no_span(self, doc):
        assert "span" not in doc._.textcrit["deletions"][0]


class TestTxtdownImportError:
    """Tests for when txtdown package is not available."""

    def test_import_error_message(self, monkeypatch, txtdown_dir):
        """Clear error message when txtdown not installed."""
        # Simulate txtdown not being available
        import latincyreaders.readers.txtdown as txtdown_module
        monkeypatch.setattr(txtdown_module, "TXTDOWN_AVAILABLE", False)

        with pytest.raises(ImportError, match="txtdown package required"):
            TxtdownReader(root=txtdown_dir)
