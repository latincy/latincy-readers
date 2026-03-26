"""Tests for DigilibLTReader."""

import pytest
from pathlib import Path

from latincyreaders import DigilibLTReader, AnnotationLevel


class TestDigilibLTReader:
    """Test suite for DigilibLTReader."""

    @pytest.fixture
    def reader(self, digilibt_dir):
        """Create a DigilibLTReader with test fixtures."""
        return DigilibLTReader(root=digilibt_dir, fileids="*.xml")

    @pytest.fixture
    def reader_tokenize(self, digilibt_dir):
        """Reader with minimal annotation for faster tests."""
        return DigilibLTReader(
            root=digilibt_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------

    def test_fileids_returns_all_fixtures(self, reader):
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) == 4
        assert all(f.endswith(".xml") for f in fileids)

    def test_root_is_path(self, reader, digilibt_dir):
        assert reader.root == digilibt_dir.resolve()

    # -------------------------------------------------------------------------
    # Raw text
    # -------------------------------------------------------------------------

    def test_texts_yields_strings(self, reader):
        texts = list(reader.texts())
        assert len(texts) == 4
        assert all(isinstance(t, str) for t in texts)

    def test_texts_contains_latin(self, reader):
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "Nobis" in all_text or "Aduersantur" in all_text

    def test_texts_removes_notes_by_default(self, reader):
        """Notes should be stripped from text."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "Textual note here" not in all_text

    # -------------------------------------------------------------------------
    # Metadata extraction
    # -------------------------------------------------------------------------

    def test_headers_yields_dicts(self, reader):
        headers = list(reader.headers())
        assert len(headers) == 4
        assert all(isinstance(h, dict) for h in headers)

    def test_headers_extract_dlt_id(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert headers["prose_caps.xml"]["dlt_id"] == "DLT000405"
        assert headers["prose_flat.xml"]["dlt_id"] == "DLT000001"
        assert headers["nested_with_verse.xml"]["dlt_id"] == "DLT000150"

    def test_headers_extract_title(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert headers["prose_caps.xml"]["title"] == "Peregrinatio Egeriae"
        assert headers["prose_flat.xml"]["title"] == "De controuersiis agrorum"

    def test_headers_extract_author_via_persname(self, reader):
        """Author should be extracted from persName[@type='usualname']."""
        headers = {h["filename"]: h for h in reader.headers()}
        assert headers["prose_flat.xml"]["author"] == "Agennius Vrbicus"
        assert headers["nested_with_verse.xml"]["author"] == "Donatus, Aelius"

    def test_headers_extract_source(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert "Franceschini" in headers["prose_caps.xml"]["source"]

    def test_headers_extract_creation_date(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert headers["prose_caps.xml"]["creation_date"] == "0384"

    # -------------------------------------------------------------------------
    # spaCy Docs
    # -------------------------------------------------------------------------

    def test_docs_yields_spacy_docs(self, reader_tokenize):
        from spacy.tokens import Doc

        docs = list(reader_tokenize.docs())
        assert len(docs) == 4
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_have_fileid(self, reader_tokenize):
        for doc in reader_tokenize.docs():
            assert doc._.fileid is not None
            assert doc._.fileid.endswith(".xml")

    def test_docs_have_metadata_with_dlt_id(self, reader_tokenize):
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        caps_doc = docs["prose_caps.xml"]
        assert caps_doc._.metadata["dlt_id"] == "DLT000405"

    def test_docs_metadata_does_not_leak_private_keys(self, reader_tokenize):
        """_chapters key should be cleaned from metadata."""
        for doc in reader_tokenize.docs():
            assert "_chapters" not in (doc._.metadata or {})

    # -------------------------------------------------------------------------
    # Chapter spans — prose_caps.xml (flat <div type="cap">)
    # -------------------------------------------------------------------------

    def test_caps_file_has_chapter_spans(self, reader_tokenize):
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        doc = docs["prose_caps.xml"]
        assert "chapters" in doc.spans
        assert len(doc.spans["chapters"]) == 3

    def test_caps_chapter_citations(self, reader_tokenize):
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        doc = docs["prose_caps.xml"]
        citations = [s._.citation for s in doc.spans["chapters"]]
        assert citations == ["cap. 1", "cap. 2", "cap. 3"]

    def test_caps_chapter_text_content(self, reader_tokenize):
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        doc = docs["prose_caps.xml"]
        ch1 = doc.spans["chapters"][0]
        assert "Nobis" in ch1.text
        assert "uallem infinitam" in ch1.text

    # -------------------------------------------------------------------------
    # Chapter spans — nested_with_verse.xml (lib → cap, with <lg>/<l>)
    # -------------------------------------------------------------------------

    def test_nested_file_has_chapter_spans(self, reader_tokenize):
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        doc = docs["nested_with_verse.xml"]
        assert "chapters" in doc.spans
        assert len(doc.spans["chapters"]) == 3

    def test_nested_chapter_citations_are_hierarchical(self, reader_tokenize):
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        doc = docs["nested_with_verse.xml"]
        citations = [s._.citation for s in doc.spans["chapters"]]
        assert citations == ["lib. V, cap. 1", "lib. V, cap. 2", "lib. VI, cap. 1"]

    def test_nested_verse_lines_included_in_text(self, reader_tokenize):
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        doc = docs["nested_with_verse.xml"]
        ch2 = doc.spans["chapters"][1]  # lib. V, cap. 2 has <lg>
        assert "Suspendit" in ch2.text
        assert "tabella" in ch2.text

    # -------------------------------------------------------------------------
    # Chapter spans — prose_flat.xml (no divs)
    # -------------------------------------------------------------------------

    def test_flat_file_has_no_chapter_spans(self, reader_tokenize):
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        doc = docs["prose_flat.xml"]
        assert len(doc.spans.get("chapters", [])) == 0

    def test_flat_file_still_has_text(self, reader_tokenize):
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        doc = docs["prose_flat.xml"]
        assert "Aduersantur" in doc.text

    # -------------------------------------------------------------------------
    # Chapter spans — section_with_heads.xml (section div with <head>)
    # -------------------------------------------------------------------------

    def test_section_file_has_chapter_spans(self, reader_tokenize):
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        doc = docs["section_with_heads.xml"]
        assert "chapters" in doc.spans
        assert len(doc.spans["chapters"]) == 1

    def test_section_chapter_uses_head_as_citation(self, reader_tokenize):
        """Section without n attribute should use <head> for citation."""
        docs = {doc._.fileid: doc for doc in reader_tokenize.docs()}
        doc = docs["section_with_heads.xml"]
        ch = doc.spans["chapters"][0]
        assert "de arte metrica" in ch._.citation

    # -------------------------------------------------------------------------
    # chapters() method
    # -------------------------------------------------------------------------

    def test_chapters_yields_spans(self, reader_tokenize):
        from spacy.tokens import Span

        chapters = list(reader_tokenize.chapters())
        assert len(chapters) > 0
        assert all(isinstance(ch, Span) for ch in chapters)

    def test_chapters_as_text_yields_tuples(self, reader):
        """chapters(as_text=True) should work without NLP."""
        results = list(reader.chapters(as_text=True))
        assert len(results) > 0
        for citation, text in results:
            assert isinstance(citation, str)
            assert isinstance(text, str)

    def test_chapters_as_text_no_nlp_overhead(self, digilibt_dir):
        """chapters(as_text=True) should work at annotation_level=NONE."""
        reader = DigilibLTReader(
            root=digilibt_dir,
            annotation_level=AnnotationLevel.NONE,
        )
        results = list(reader.chapters(as_text=True))
        assert len(results) > 0

    def test_chapters_for_specific_file(self, reader_tokenize):
        chapters = list(reader_tokenize.chapters(fileids="prose_caps.xml"))
        assert len(chapters) == 3

    # -------------------------------------------------------------------------
    # Sentence and token iteration
    # -------------------------------------------------------------------------

    def test_sents_yields_spans(self, reader_tokenize):
        from spacy.tokens import Span

        sents = list(reader_tokenize.sents())
        assert len(sents) > 0
        assert all(isinstance(s, Span) for s in sents)

    def test_tokens_yields_tokens(self, reader_tokenize):
        from spacy.tokens import Token

        tokens = list(reader_tokenize.tokens())
        assert len(tokens) > 0
        assert all(isinstance(t, Token) for t in tokens)

    # -------------------------------------------------------------------------
    # Annotation levels
    # -------------------------------------------------------------------------

    def test_annotation_level_none_blocks_docs(self, digilibt_dir):
        reader = DigilibLTReader(
            root=digilibt_dir,
            annotation_level=AnnotationLevel.NONE,
        )
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            next(reader.docs())

    def test_annotation_level_none_allows_texts(self, digilibt_dir):
        reader = DigilibLTReader(
            root=digilibt_dir,
            annotation_level=AnnotationLevel.NONE,
        )
        texts = list(reader.texts())
        assert len(texts) == 4

    # -------------------------------------------------------------------------
    # Text-critical symbol stripping (use_symbols)
    # -------------------------------------------------------------------------

    def test_use_symbols_strips_angle_brackets(self, digilibt_dir):
        """<word> marks should be stripped, preserving the word."""
        reader = DigilibLTReader(
            root=digilibt_dir,
            fileids="prose_caps.xml",
            use_symbols=True,
        )
        # prose_caps.xml has <supplied>locus</supplied> which via itertext
        # gives "locus" directly. But let's test the normalizer directly.
        text = reader._normalize_text("Ipse autem <locus> in extremo est.")
        assert "<" not in text
        assert ">" not in text
        assert "locus" in text

    def test_use_symbols_removes_square_brackets_and_content(self, digilibt_dir):
        """[word] should be removed entirely (secluded text)."""
        reader = DigilibLTReader(root=digilibt_dir, use_symbols=True)
        text = reader._normalize_text("hoc [verbum] est")
        assert "[" not in text
        assert "verbum" not in text
        assert "hoc" in text
        assert "est" in text

    def test_use_symbols_strips_curly_brackets(self, digilibt_dir):
        reader = DigilibLTReader(root=digilibt_dir, use_symbols=True)
        text = reader._normalize_text("hoc {verbum} est")
        assert "{" not in text
        assert "verbum" in text

    def test_use_symbols_strips_daggers(self, digilibt_dir):
        reader = DigilibLTReader(root=digilibt_dir, use_symbols=True)
        text = reader._normalize_text("hoc †verbum† est")
        assert "†" not in text
        assert "verbum" in text

    def test_use_symbols_removes_lacuna_markers(self, digilibt_dir):
        reader = DigilibLTReader(root=digilibt_dir, use_symbols=True)
        text = reader._normalize_text("ante *** post")
        assert "***" not in text
        assert "ante" in text
        assert "post" in text

    def test_use_symbols_collapses_whitespace(self, digilibt_dir):
        reader = DigilibLTReader(root=digilibt_dir, use_symbols=True)
        text = reader._normalize_text("ante *** post")
        assert "  " not in text

    def test_use_symbols_expands_abbreviations(self, digilibt_dir):
        """M(arcus) → Marcus, s(alutem) → salutem."""
        reader = DigilibLTReader(root=digilibt_dir, use_symbols=True)
        text = reader._normalize_text("M(arcus) Cicero s(alutem) d(icit)")
        assert text == "Marcus Cicero salutem dicit"

    def test_use_symbols_preserves_standalone_parens(self, digilibt_dir):
        """(word) without preceding letter is kept as-is."""
        reader = DigilibLTReader(root=digilibt_dir, use_symbols=True)
        text = reader._normalize_text("hoc (verbum) est")
        assert "(verbum)" in text

    def test_use_symbols_false_preserves_marks(self, digilibt_dir):
        """When use_symbols=False, marks should be preserved."""
        reader = DigilibLTReader(root=digilibt_dir, use_symbols=False)
        text = reader._normalize_text("Ipse autem <locus> in extremo est.")
        assert "<locus>" in text

    def test_use_symbols_default_is_true(self, digilibt_dir):
        """Default should strip symbols."""
        reader = DigilibLTReader(root=digilibt_dir)
        text = reader._normalize_text("hoc <est> bonum")
        assert "<" not in text
        assert "est" in text

    # -------------------------------------------------------------------------
    # Paragraph iteration (inherited from TEIReader)
    # -------------------------------------------------------------------------

    def test_paras_as_text(self, reader_tokenize):
        paras = list(reader_tokenize.paras(as_text=True))
        assert len(paras) > 0
        assert all(isinstance(p, str) for p in paras)
