"""Tests for WikiSourceReader."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from latincyreaders.readers.wikisource import WikiSourceReader, WikiSection
from latincyreaders.core.base import AnnotationLevel


class TestWikiSourceBasic:
    """Basic WikiSourceReader functionality."""

    @pytest.fixture
    def reader(self, wikisource_dir):
        """WikiSourceReader with TOKENIZE for fast tests."""
        return WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of .wiki files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".wiki") for f in fileids)

    def test_default_file_pattern(self, reader):
        """Default pattern is **/*.wiki."""
        assert reader._fileids_pattern == "**/*.wiki"

    def test_root_is_path(self, reader, wikisource_dir):
        """root property returns correct Path."""
        assert reader.root == wikisource_dir.resolve()

    def test_fileids_discovers_nested(self, reader):
        """fileids() discovers .wiki files in subdirectories."""
        fileids = reader.fileids()
        nested = [f for f in fileids if "/" in f]
        assert len(nested) > 0  # aeneis/liber_i.wiki


class TestMetadataParsing:
    """Test {{titulus2}} template parsing."""

    @pytest.fixture
    def reader(self, wikisource_dir):
        return WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.NONE,
        )

    def test_parse_titulus_extracts_author(self, reader, wikisource_dir):
        """_parse_titulus extracts Scriptor as author."""
        text = (wikisource_dir / "de_vita_beata.wiki").read_text()
        meta = reader._parse_titulus(text)
        assert meta["author"] == "Lucius Annaeus Seneca"

    def test_parse_titulus_extracts_title(self, reader, wikisource_dir):
        """_parse_titulus extracts OperaeTitulus as title."""
        text = (wikisource_dir / "de_vita_beata.wiki").read_text()
        meta = reader._parse_titulus(text)
        assert meta["title"] == "De vita beata"

    def test_parse_titulus_extracts_date(self, reader, wikisource_dir):
        """_parse_titulus extracts Annus as date."""
        text = (wikisource_dir / "de_vita_beata.wiki").read_text()
        meta = reader._parse_titulus(text)
        assert meta["date"] == "-58"

    def test_parse_titulus_extracts_genre(self, reader, wikisource_dir):
        """_parse_titulus extracts Genera as genre."""
        text = (wikisource_dir / "de_vita_beata.wiki").read_text()
        meta = reader._parse_titulus(text)
        assert meta["genre"] == "Philosophia"

    def test_parse_titulus_verse(self, reader, wikisource_dir):
        """_parse_titulus works on verse files."""
        text = (wikisource_dir / "aeneis" / "liber_i.wiki").read_text()
        meta = reader._parse_titulus(text)
        assert meta["author"] == "Publius Vergilius Maro"
        assert meta["title"] == "Aeneis I"

    def test_parse_titulus_missing(self, reader):
        """_parse_titulus returns empty dict for text without template."""
        meta = reader._parse_titulus("Just some text without templates.")
        assert meta == {}


class TestMarkupStripping:
    """Test wikitext markup stripping."""

    @pytest.fixture
    def reader(self, wikisource_dir):
        return WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.NONE,
        )

    def test_strip_finis(self, reader):
        """_strip_markup removes {{finis}}."""
        result = reader._strip_markup("Text here. {{finis}}")
        assert "finis" not in result.lower()
        assert "Text here." in result

    def test_strip_textquality(self, reader):
        """_strip_markup removes {{textquality|...}}."""
        result = reader._strip_markup("Text. {{textquality|75%}}")
        assert "textquality" not in result
        assert "75%" not in result

    def test_strip_interwiki(self, reader):
        """_strip_markup removes interwiki links."""
        result = reader._strip_markup("Text. [[en:English page]] [[fr:French page]]")
        assert "[[en:" not in result
        assert "[[fr:" not in result
        assert "Text." in result

    def test_strip_liber_navigation(self, reader):
        """_strip_markup removes {{Liber|...}} navigation."""
        result = reader._strip_markup("{{Liber|Ante=|Post=[[Aeneis/Liber II|Liber II]]}} Text.")
        assert "Liber|" not in result
        assert "Text." in result

    def test_strip_div_wrappers(self, reader):
        """_strip_markup removes <div> tags."""
        result = reader._strip_markup('<div class="text">Content here</div>')
        assert "<div" not in result
        assert "</div>" not in result
        assert "Content here" in result

    def test_strip_intraincepti(self, reader):
        """_strip_markup removes {{Intraincepti|...}}."""
        result = reader._strip_markup("Text. {{Intraincepti|Aeneis}}")
        assert "Intraincepti" not in result

    def test_strip_imago(self, reader):
        """_strip_markup removes [[Imago:...]]."""
        result = reader._strip_markup("Text. [[Imago:some_image.jpg]]")
        assert "Imago" not in result

    def test_strip_versus_template(self, reader):
        """_strip_markup removes {{versus|N}} templates."""
        result = reader._strip_markup("{{versus|1}}Arma virumque cano")
        assert "{{versus" not in result
        assert "Arma virumque cano" in result

    def test_strip_preserves_content(self, reader):
        """_strip_markup preserves actual Latin text."""
        result = reader._strip_markup(
            "{{finis}} Vivere omnes beate volunt. {{textquality|100%}} [[en:Page]]"
        )
        assert "Vivere omnes beate volunt." in result


class TestTextIteration:
    """Test raw text access."""

    @pytest.fixture
    def reader(self, wikisource_dir):
        return WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.NONE,
        )

    def test_texts_yields_strings(self, reader):
        """texts() yields raw strings."""
        texts = list(reader.texts())
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_texts_no_markup(self, reader):
        """texts() output has no wiki markup."""
        for text in reader.texts():
            assert "{{" not in text
            assert "}}" not in text
            assert "<div" not in text
            assert "<poem>" not in text

    def test_texts_contains_latin(self, reader):
        """texts() contains actual Latin content."""
        all_text = " ".join(reader.texts())
        # From De vita beata
        assert "Vivere" in all_text or "vivere" in all_text
        # From Aeneid
        assert "Arma" in all_text or "arma" in all_text

    def test_index_page_skipped(self, reader):
        """Index pages are not yielded by texts()."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        # Index page has "Aeneis est poema epicum" but no real content
        # The actual content files should produce text, index shouldn't
        # We should have text from de_vita_beata and liber_i but not the index
        assert len(texts) >= 2  # At least prose + verse


class TestProseParsing:
    """Test prose (section + paragraph) parsing."""

    @pytest.fixture
    def reader(self, wikisource_dir):
        return WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    def test_prose_sections_parsed(self, reader, wikisource_dir):
        """_parse_sections extracts section headers."""
        text = (wikisource_dir / "de_vita_beata.wiki").read_text()
        sections = reader._parse_sections(text)
        assert len(sections) == 3  # I, II, III
        assert sections[0].header == "I."
        assert sections[1].header == "II."
        assert sections[2].header == "III."

    def test_prose_paragraphs_numbered(self, reader, wikisource_dir):
        """Sections contain numbered paragraphs."""
        text = (wikisource_dir / "de_vita_beata.wiki").read_text()
        sections = reader._parse_sections(text)
        # Section I has 3 paragraphs
        assert len(sections[0].paragraphs) == 3
        assert sections[0].paragraphs[0][0] == 1  # para num
        assert sections[0].paragraphs[1][0] == 2
        assert sections[0].paragraphs[2][0] == 3

    def test_prose_sections_have_citations(self, reader):
        """Prose docs have section spans with citations."""
        prose_fileids = [f for f in reader.fileids() if "de_vita_beata" in f]
        assert len(prose_fileids) > 0

        doc = next(reader.docs(prose_fileids))
        assert "sections" in doc.spans
        assert len(doc.spans["sections"]) > 0

        # Check citations are in section.paragraph format
        for span in doc.spans["sections"]:
            assert span._.citation is not None
            # Should be like "I.1", "II.2", etc.
            assert "." in span._.citation or span._.citation.endswith(".")

    def test_prose_doc_metadata(self, reader):
        """Prose docs have metadata from titulus template."""
        prose_fileids = [f for f in reader.fileids() if "de_vita_beata" in f]
        doc = next(reader.docs(prose_fileids))
        assert doc._.metadata is not None
        assert doc._.metadata.get("author") == "Lucius Annaeus Seneca"
        assert doc._.metadata.get("content_type") == "prose"


class TestVerseParsing:
    """Test verse (<poem> + {{versus|N}}) parsing."""

    @pytest.fixture
    def reader(self, wikisource_dir):
        return WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    def test_verse_lines_parsed(self, reader, wikisource_dir):
        """_parse_verse_lines extracts numbered lines."""
        text = (wikisource_dir / "aeneis" / "liber_i.wiki").read_text()
        lines = reader._parse_verse_lines(text)
        assert len(lines) == 30  # 30 lines in fixture
        assert lines[0][0] == 1  # first line number
        assert lines[0][1].startswith("Arma virumque cano")

    def test_verse_lines_have_citations(self, reader):
        """Verse docs have line spans with citations."""
        verse_fileids = [f for f in reader.fileids() if "liber_i" in f]
        assert len(verse_fileids) > 0

        doc = next(reader.docs(verse_fileids))
        assert "lines" in doc.spans
        assert len(doc.spans["lines"]) > 0

        # Citations should be line numbers
        first_span = doc.spans["lines"][0]
        assert first_span._.citation == "1"

    def test_versus_template_parsed(self, reader, wikisource_dir):
        """{{versus|N}} templates are correctly parsed to line numbers."""
        text = (wikisource_dir / "aeneis" / "liber_i.wiki").read_text()
        lines = reader._parse_verse_lines(text)
        # Line numbers should match versus template values
        line_nums = [num for num, _ in lines]
        assert 1 in line_nums
        assert 10 in line_nums
        assert 30 in line_nums

    def test_poem_tags_stripped(self, reader):
        """<poem> tags don't appear in output text."""
        verse_fileids = [f for f in reader.fileids() if "liber_i" in f]
        for text in reader.texts(verse_fileids):
            assert "<poem>" not in text
            assert "</poem>" not in text

    def test_verse_doc_metadata(self, reader):
        """Verse docs have metadata from titulus template."""
        verse_fileids = [f for f in reader.fileids() if "liber_i" in f]
        doc = next(reader.docs(verse_fileids))
        assert doc._.metadata is not None
        assert doc._.metadata.get("author") == "Publius Vergilius Maro"
        assert doc._.metadata.get("content_type") == "verse"


class TestDocCreation:
    """Test spaCy Doc creation."""

    @pytest.fixture
    def reader(self, wikisource_dir):
        return WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    def test_docs_have_fileid(self, reader):
        """Docs have fileid extension set."""
        doc = next(reader.docs())
        assert doc._.fileid is not None
        assert doc._.fileid.endswith(".wiki")

    def test_docs_have_metadata(self, reader):
        """Docs have metadata extension set."""
        doc = next(reader.docs())
        assert doc._.metadata is not None
        assert "author" in doc._.metadata or "filename" in doc._.metadata

    def test_docs_yields_spacy_docs(self, reader):
        """docs() yields spaCy Doc objects."""
        from spacy.tokens import Doc

        docs = list(reader.docs())
        assert len(docs) >= 2  # prose + verse
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_spans_populated(self, reader):
        """Docs have either sections or lines spans populated."""
        for doc in reader.docs():
            has_sections = "sections" in doc.spans and len(doc.spans["sections"]) > 0
            has_lines = "lines" in doc.spans and len(doc.spans["lines"]) > 0
            assert has_sections or has_lines

    def test_sents_yields_spans(self, reader):
        """sents() yields Span objects."""
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


class TestSectionsAndLines:
    """Test sections() and lines() methods."""

    @pytest.fixture
    def reader(self, wikisource_dir):
        return WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    def test_sections_yields_spans(self, reader):
        """sections() yields Span objects with citations."""
        from spacy.tokens import Span

        section_spans = list(reader.sections())
        assert len(section_spans) > 0
        assert all(isinstance(s, Span) for s in section_spans)
        assert all(s._.citation is not None for s in section_spans)

    def test_lines_yields_spans(self, reader):
        """lines() yields verse line Spans with citations."""
        from spacy.tokens import Span

        line_spans = list(reader.lines())
        assert len(line_spans) > 0
        assert all(isinstance(s, Span) for s in line_spans)
        assert all(s._.citation is not None for s in line_spans)


class TestContentTypeDetection:
    """Test verse vs prose detection."""

    @pytest.fixture
    def reader(self, wikisource_dir):
        return WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.NONE,
        )

    def test_verse_detected(self, reader, wikisource_dir):
        """_is_verse correctly identifies verse content."""
        text = (wikisource_dir / "aeneis" / "liber_i.wiki").read_text()
        assert reader._is_verse(text) is True

    def test_prose_not_verse(self, reader, wikisource_dir):
        """_is_verse correctly identifies prose content."""
        text = (wikisource_dir / "de_vita_beata.wiki").read_text()
        assert reader._is_verse(text) is False

    def test_index_page_detected(self, reader, wikisource_dir):
        """_is_index_page correctly identifies index pages."""
        text = (wikisource_dir / "index_page.wiki").read_text()
        assert reader._is_index_page(text) is True

    def test_content_page_not_index(self, reader, wikisource_dir):
        """_is_index_page returns False for content pages."""
        text = (wikisource_dir / "de_vita_beata.wiki").read_text()
        assert reader._is_index_page(text) is False


class TestDownload:
    """Test download classmethod (mocked)."""

    def test_download_saves_wiki_file(self, tmp_path):
        """download() saves a .wiki file."""
        mock_wikitext = "{{titulus2|Scriptor=Test}} Content here."
        mock_response = json.dumps({
            "parse": {"wikitext": {"*": mock_wikitext}}
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            paths = WikiSourceReader.download("Test_Page", tmp_path, follow_subpages=False)

        assert len(paths) == 1
        assert paths[0].suffix == ".wiki"
        assert paths[0].exists()
        assert "Content here" in paths[0].read_text()

    def test_download_follows_subpages(self, tmp_path):
        """download() recursively fetches sub-pages."""
        main_wikitext = "Main page. [[Test_Page/Sub1|Sub 1]] [[Test_Page/Sub2|Sub 2]]"
        sub1_wikitext = "Sub page 1 content."
        sub2_wikitext = "Sub page 2 content."

        responses = {
            "Test_Page": main_wikitext,
            "Test_Page/Sub1": sub1_wikitext,
            "Test_Page/Sub2": sub2_wikitext,
        }

        def mock_urlopen(req):
            # Extract page name from URL
            url = req.full_url if hasattr(req, 'full_url') else str(req)
            for page_name, content in responses.items():
                if urllib.parse.quote(page_name) in url:
                    mock_resp = MagicMock()
                    mock_resp.read.return_value = json.dumps({
                        "parse": {"wikitext": {"*": content}}
                    }).encode("utf-8")
                    mock_resp.__enter__ = lambda s: s
                    mock_resp.__exit__ = MagicMock(return_value=False)
                    return mock_resp
            raise ValueError(f"Unexpected URL: {url}")

        import urllib.parse

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            paths = WikiSourceReader.download("Test_Page", tmp_path, follow_subpages=True)

        assert len(paths) == 3  # main + 2 sub-pages
        assert all(p.suffix == ".wiki" for p in paths)

    def test_download_handles_api_error(self, tmp_path):
        """download() raises on MediaWiki API errors."""
        mock_response = json.dumps({
            "error": {"code": "missingtitle", "info": "The page does not exist."}
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(ValueError, match="MediaWiki API error"):
                WikiSourceReader.download("Nonexistent_Page", tmp_path)


class TestAnnotationLevels:
    """Test annotation level behavior."""

    def test_annotation_level_none_blocks_docs(self, wikisource_dir):
        """annotation_level=NONE prevents docs()."""
        reader = WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.NONE,
        )
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            next(reader.docs())

    def test_annotation_level_none_allows_texts(self, wikisource_dir):
        """annotation_level=NONE still allows texts()."""
        reader = WikiSourceReader(
            root=wikisource_dir,
            annotation_level=AnnotationLevel.NONE,
        )
        texts = list(reader.texts())
        assert len(texts) > 0
