"""Tests for ProjectGutenbergReader."""

import pytest
from pathlib import Path

from latincyreaders import AnnotationLevel


@pytest.fixture
def gutenberg_dir(fixtures_dir) -> Path:
    """Path to Project Gutenberg test fixtures."""
    return fixtures_dir / "gutenberg"


class TestProjectGutenbergReader:
    """Test suite for ProjectGutenbergReader."""

    @pytest.fixture
    def reader(self, gutenberg_dir):
        """Create a reader using the local fixture cache dir (no network)."""
        from latincyreaders import ProjectGutenbergReader

        return ProjectGutenbergReader(
            ids=[28233],
            cache_dir=gutenberg_dir,
            annotation_level=AnnotationLevel.NONE,
        )

    # -------------------------------------------------------------------------
    # Boilerplate stripping
    # -------------------------------------------------------------------------

    def test_strip_boilerplate_removes_header(self):
        """_strip_boilerplate removes everything up to and including the START marker line."""
        from latincyreaders.readers.gutenberg import ProjectGutenbergReader

        raw = (
            "Project Gutenberg preamble line\n"
            "\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***\n"
            "\n"
            "Actual content here.\n"
        )
        result = ProjectGutenbergReader._strip_boilerplate(raw)
        assert "preamble" not in result
        assert "START OF" not in result
        assert "Actual content here." in result

    def test_strip_boilerplate_removes_footer(self):
        """_strip_boilerplate removes the END marker and everything after."""
        from latincyreaders.readers.gutenberg import ProjectGutenbergReader

        raw = (
            "*** START OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***\n"
            "\n"
            "Actual content here.\n"
            "\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***\n"
            "\n"
            "License footer here.\n"
        )
        result = ProjectGutenbergReader._strip_boilerplate(raw)
        assert "License footer" not in result
        assert "END OF" not in result
        assert "Actual content here." in result

    def test_strip_boilerplate_no_markers_returns_text(self):
        """_strip_boilerplate returns text unchanged if markers are absent."""
        from latincyreaders.readers.gutenberg import ProjectGutenbergReader

        raw = "Just some plain text without any markers."
        result = ProjectGutenbergReader._strip_boilerplate(raw)
        assert result == raw.strip()

    # -------------------------------------------------------------------------
    # texts() interface
    # -------------------------------------------------------------------------

    def test_texts_yields_strings(self, reader):
        """texts() yields raw strings."""
        texts = list(reader.texts())
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_texts_contains_latin_content(self, reader):
        """texts() yields the Latin content from the fixture."""
        text = next(reader.texts())
        assert "Quantitas Materiae" in text

    def test_texts_excludes_pg_header(self, reader):
        """texts() does not include PG preamble text."""
        text = next(reader.texts())
        assert "Project Gutenberg License" not in text
        assert "START OF THE PROJECT GUTENBERG" not in text

    def test_texts_excludes_pg_footer(self, reader):
        """texts() does not include PG footer text."""
        text = next(reader.texts())
        assert "END OF THE PROJECT GUTENBERG" not in text
        assert "Updated editions will replace" not in text

    # -------------------------------------------------------------------------
    # fileids()
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a non-empty list."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0

    def test_fileids_match_requested_ids(self, reader):
        """fileids() includes files for the requested PG IDs."""
        fileids = reader.fileids()
        assert any("28233" in f for f in fileids)

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def test_metadata_includes_pg_id(self, reader):
        """Parsed metadata includes the pg_id field."""
        reader._annotation_level = AnnotationLevel.TOKENIZE
        reader._nlp = None

        doc = next(reader.docs())
        assert "pg_id" in doc._.metadata
        assert doc._.metadata["pg_id"] == "28233"

    # -------------------------------------------------------------------------
    # Language / model params
    # -------------------------------------------------------------------------

    def test_model_name_and_lang_are_explicit_params(self):
        """model_name and lang appear as named params in the constructor signature."""
        import inspect
        from latincyreaders import ProjectGutenbergReader

        sig = inspect.signature(ProjectGutenbergReader.__init__)
        assert "model_name" in sig.parameters
        assert "lang" in sig.parameters

    def test_accepts_custom_model_name(self, gutenberg_dir):
        """model_name param is stored and forwarded correctly."""
        from latincyreaders import ProjectGutenbergReader

        reader = ProjectGutenbergReader(
            ids=[28233],
            cache_dir=gutenberg_dir,
            annotation_level=AnnotationLevel.NONE,
            model_name="en_core_web_sm",
            lang="en",
        )
        assert reader._model_name == "en_core_web_sm"
        assert reader._lang == "en"

    def test_default_model_is_latin(self, gutenberg_dir):
        """Default model_name and lang are for Latin."""
        from latincyreaders import ProjectGutenbergReader

        reader = ProjectGutenbergReader(
            ids=[28233],
            cache_dir=gutenberg_dir,
            annotation_level=AnnotationLevel.NONE,
        )
        assert reader._model_name == "la_core_web_lg"
        assert reader._lang == "la"

    # -------------------------------------------------------------------------
    # URL pattern
    # -------------------------------------------------------------------------

    def test_url_pattern(self):
        """BASE_URL formats correctly for a given ID."""
        from latincyreaders.readers.gutenberg import ProjectGutenbergReader

        url = ProjectGutenbergReader.BASE_URL.format(id="28233")
        assert url == "https://www.gutenberg.org/cache/epub/28233/pg28233.txt"

    # -------------------------------------------------------------------------
    # No-download path (file already cached)
    # -------------------------------------------------------------------------

    def test_does_not_fetch_when_file_exists(self, gutenberg_dir, monkeypatch):
        """If the file is already in cache_dir, _fetch_text is never called."""
        from latincyreaders import ProjectGutenbergReader

        fetch_called = []

        def mock_fetch(self, pg_id, dest):
            fetch_called.append(pg_id)

        monkeypatch.setattr(ProjectGutenbergReader, "_fetch_text", mock_fetch)

        ProjectGutenbergReader(
            ids=[28233],
            cache_dir=gutenberg_dir,
            annotation_level=AnnotationLevel.NONE,
        )
        assert fetch_called == [], "Should not fetch when file already exists"
