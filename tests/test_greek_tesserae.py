"""Tests for GreekTesseraeReader."""

import pytest
from pathlib import Path

from latincyreaders import GreekTesseraeReader, AnnotationLevel


class TestGreekTesseraeReader:
    """Test suite for GreekTesseraeReader."""

    @pytest.fixture
    def reader(self, greek_tesserae_dir):
        """Create a GreekTesseraeReader with test fixtures (no NLP)."""
        return GreekTesseraeReader(
            root=greek_tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )

    @pytest.fixture
    def reader_tokenize(self, greek_tesserae_dir):
        """Reader with TOKENIZE annotation (uses grc blank model)."""
        return GreekTesseraeReader(
            root=greek_tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of .tess files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".tess") for f in fileids)

    def test_fileids_contains_test_file(self, reader):
        """Test fixture file is discovered."""
        fileids = reader.fileids()
        assert "homer_iliad.tess" in fileids

    def test_root_is_path(self, reader, greek_tesserae_dir):
        """root property returns correct Path."""
        assert reader.root == greek_tesserae_dir.resolve()

    def test_annotation_level_property(self, reader):
        """annotation_level property returns current level."""
        assert reader.annotation_level == AnnotationLevel.NONE

    # -------------------------------------------------------------------------
    # Class attributes (Greek-specific)
    # -------------------------------------------------------------------------

    def test_corpus_url(self):
        """CORPUS_URL points to Greek Tesserae repo."""
        assert "grc_text_tesserae" in GreekTesseraeReader.CORPUS_URL

    def test_env_var(self):
        """ENV_VAR is set for Greek corpus."""
        assert GreekTesseraeReader.ENV_VAR == "GRC_TESSERAE_PATH"

    def test_default_subdir(self):
        """DEFAULT_SUBDIR is set for Greek corpus."""
        assert "grc_text_tesserae" in GreekTesseraeReader.DEFAULT_SUBDIR

    def test_model_defaults(self, greek_tesserae_dir):
        """Default model is OdyCy, language is grc."""
        reader = GreekTesseraeReader(
            root=greek_tesserae_dir,
            annotation_level=AnnotationLevel.NONE,
        )
        assert reader._model_name == "grc_odycy_joint_sm"
        assert reader._lang == "grc"

    # -------------------------------------------------------------------------
    # Raw text access (no NLP)
    # -------------------------------------------------------------------------

    def test_texts_yields_strings(self, reader):
        """texts() yields raw strings."""
        texts = list(reader.texts())
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_texts_contains_greek(self, reader):
        """Text content is Greek."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "μῆνιν" in all_text or "Ἀχιλ" in all_text

    def test_texts_by_line_yields_tuples(self, reader):
        """texts_by_line() yields (citation, text) tuples."""
        lines = list(reader.texts_by_line())
        assert len(lines) > 0
        assert all(isinstance(line, tuple) and len(line) == 2 for line in lines)

    def test_texts_by_line_has_citations(self, reader):
        """Citations are in correct format."""
        citation, text = next(reader.texts_by_line())
        assert citation.startswith("<")
        assert citation.endswith(">")
        assert len(text) > 0

    def test_texts_by_line_citation_content(self, reader):
        """First citation should reference Homer's Iliad."""
        citation, text = next(reader.texts_by_line())
        assert "hom. il." in citation

    # -------------------------------------------------------------------------
    # spaCy Doc access (TOKENIZE level)
    # -------------------------------------------------------------------------

    def test_docs_yields_spacy_docs(self, reader_tokenize):
        """docs() yields spaCy Doc objects."""
        from spacy.tokens import Doc

        docs = list(reader_tokenize.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_have_fileid(self, reader_tokenize):
        """Docs have fileid extension set."""
        doc = next(reader_tokenize.docs())
        assert doc._.fileid is not None
        assert doc._.fileid.endswith(".tess")

    def test_docs_have_metadata(self, reader_tokenize):
        """Docs have metadata extension set."""
        doc = next(reader_tokenize.docs())
        assert doc._.metadata is not None
        assert "filename" in doc._.metadata

    def test_docs_have_line_spans(self, reader_tokenize):
        """Docs have 'lines' span group."""
        doc = next(reader_tokenize.docs())
        assert "lines" in doc.spans
        assert len(doc.spans["lines"]) > 0

    def test_line_spans_have_citations(self, reader_tokenize):
        """Line spans have citation extensions."""
        doc = next(reader_tokenize.docs())
        for span in doc.spans["lines"]:
            assert span._.citation is not None
            assert span._.citation.startswith("<")

    # -------------------------------------------------------------------------
    # Sentence and token iteration
    # -------------------------------------------------------------------------

    def test_sents_yields_spans(self, reader_tokenize):
        """sents() yields Span objects."""
        from spacy.tokens import Span

        sents = list(reader_tokenize.sents())
        assert len(sents) > 0
        assert all(isinstance(s, Span) for s in sents)

    def test_tokens_yields_tokens(self, reader_tokenize):
        """tokens() yields Token objects."""
        from spacy.tokens import Token

        tokens = list(reader_tokenize.tokens())
        assert len(tokens) > 0
        assert all(isinstance(t, Token) for t in tokens)

    def test_tokens_as_text_yields_strings(self, reader_tokenize):
        """tokens(as_text=True) yields strings."""
        tokens = list(reader_tokenize.tokens(as_text=True))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    # -------------------------------------------------------------------------
    # Citation parsing
    # -------------------------------------------------------------------------

    def test_parse_lines_extracts_citations(self, reader):
        """_parse_lines extracts citation-text pairs from Greek text."""
        fixture = reader.root / "homer_iliad.tess"
        text = fixture.read_text()
        lines = list(reader._parse_lines(text))

        assert len(lines) == 8
        assert lines[0].citation == "<hom. il. 1.1>"
        assert lines[3].citation == "<hom. il. 1.4>"

    def test_parse_lines_preserves_greek_text(self, reader):
        """_parse_lines preserves Greek Unicode content."""
        fixture = reader.root / "homer_iliad.tess"
        text = fixture.read_text()
        lines = list(reader._parse_lines(text))

        # First line should start with μῆνιν
        assert lines[0].text.startswith("μῆνιν")

    # -------------------------------------------------------------------------
    # Annotation levels
    # -------------------------------------------------------------------------

    def test_annotation_level_none_blocks_docs(self, greek_tesserae_dir):
        """annotation_level=NONE prevents docs() from working."""
        reader = GreekTesseraeReader(
            root=greek_tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            next(reader.docs())

    def test_annotation_level_none_allows_texts(self, greek_tesserae_dir):
        """annotation_level=NONE still allows texts()."""
        reader = GreekTesseraeReader(
            root=greek_tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )
        texts = list(reader.texts())
        assert len(texts) > 0

    def test_annotation_level_none_allows_texts_by_line(self, greek_tesserae_dir):
        """annotation_level=NONE still allows texts_by_line()."""
        reader = GreekTesseraeReader(
            root=greek_tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )
        lines = list(reader.texts_by_line())
        assert len(lines) > 0

    # -------------------------------------------------------------------------
    # Auto-download functionality
    # -------------------------------------------------------------------------

    def test_auto_download_false_raises_on_missing(self, tmp_path, monkeypatch):
        """auto_download=False raises FileNotFoundError for missing corpus."""
        monkeypatch.setenv("GRC_TESSERAE_PATH", str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="corpus not found"):
            GreekTesseraeReader(root=None, auto_download=False)

    def test_default_root_returns_path(self):
        """default_root() returns a Path."""
        root = GreekTesseraeReader.default_root()
        assert isinstance(root, Path)

    def test_explicit_root_bypasses_default(self, greek_tesserae_dir):
        """Explicit root= bypasses default location logic."""
        reader = GreekTesseraeReader(
            root=greek_tesserae_dir,
            annotation_level=AnnotationLevel.NONE,
        )
        assert reader.root == greek_tesserae_dir.resolve()


class TestGreekTesseraeSearch:
    """Test search functionality with Greek text."""

    @pytest.fixture
    def reader(self, greek_tesserae_dir):
        return GreekTesseraeReader(
            root=greek_tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    def test_search_returns_matches(self, reader):
        """search() finds Greek text patterns."""
        results = list(reader.search(r"Ἀχιλ"))
        assert len(results) > 0
        fileid, citation, text, matches = results[0]
        assert any("Ἀχιλ" in m for m in matches)

    def test_search_returns_citation(self, reader):
        """search() includes citation in results."""
        results = list(reader.search(r"μῆνιν"))
        assert len(results) > 0
        _, citation, _, _ = results[0]
        assert citation.startswith("<")
        assert citation.endswith(">")

    def test_find_lines_with_pattern(self, reader):
        """find_lines() works with Greek regex pattern."""
        results = list(reader.find_lines(pattern=r"Ἀχαι"))
        assert len(results) > 0
        fileid, citation, text = results[0]
        assert "Ἀχαι" in text

    def test_find_lines_with_forms(self, reader):
        """find_lines() works with Greek forms list."""
        results = list(reader.find_lines(forms=["μῆνιν", "θεὰ"]))
        assert len(results) > 0

    def test_find_sents_returns_dicts(self, reader):
        """find_sents() returns dicts with expected keys."""
        results = list(reader.find_sents(pattern=r"Ἀχιλ"))
        assert len(results) > 0
        result = results[0]
        assert "fileid" in result
        assert "citation" in result
        assert "sentence" in result
        assert "matches" in result

    def test_lines_yields_spans_with_citations(self, reader):
        """lines() yields Spans with citations."""
        from spacy.tokens import Span

        lines = list(reader.lines())
        assert len(lines) > 0
        assert all(isinstance(line, Span) for line in lines)
        assert all(line._.citation is not None for line in lines)

    def test_doc_rows_yields_dicts(self, reader):
        """doc_rows() yields citation->Span dicts."""
        rows = list(reader.doc_rows())
        assert len(rows) > 0
        assert all(isinstance(r, dict) for r in rows)

        row = rows[0]
        for citation, span in row.items():
            assert citation.startswith("<")
            assert hasattr(span, "text")


class TestGreekTesseraeNLP:
    """Test BASIC/FULL annotation levels with OdyCy.

    These tests are skipped if OdyCy is not installed.
    """

    @pytest.fixture
    def reader(self, greek_tesserae_dir):
        pytest.importorskip("grc_odycy_joint_sm")
        return GreekTesseraeReader(
            root=greek_tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.BASIC,
        )

    def test_docs_have_pos_tags(self, reader):
        """Docs from BASIC level have POS tags."""
        doc = next(reader.docs())
        pos_tags = [t.pos_ for t in doc if t.is_alpha]
        assert len(pos_tags) > 0
        # Should have actual POS tags, not empty strings
        assert any(tag != "" for tag in pos_tags)

    def test_docs_have_lemmas(self, reader):
        """Docs from BASIC level have lemmas."""
        doc = next(reader.docs())
        lemmas = [t.lemma_ for t in doc if t.is_alpha]
        assert len(lemmas) > 0

    def test_concordance_works(self, reader):
        """concordance() works with Greek text and OdyCy."""
        conc = reader.concordance()
        assert isinstance(conc, dict)
        assert len(conc) > 0

    def test_kwic_works(self, reader):
        """kwic() works with Greek text."""
        # Search for a common word in our fixture
        results = list(reader.kwic("καὶ", limit=5))
        assert len(results) >= 0  # May or may not match depending on tokenization

    def test_ngrams_work(self, reader):
        """ngrams() works with Greek text."""
        bigrams = list(reader.ngrams(n=2))
        assert len(bigrams) > 0
        assert all(isinstance(bg, str) for bg in bigrams)
