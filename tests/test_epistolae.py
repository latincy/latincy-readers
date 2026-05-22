"""Tests for EpistolaeReader."""

import pytest
from pathlib import Path

from latincyreaders import AnnotationLevel


class TestEpistolaeReaderFileids:
    """File listing."""

    @pytest.fixture
    def reader(self, epistolae_dir):
        from latincyreaders import EpistolaeReader
        return EpistolaeReader(root=epistolae_dir)

    def test_fileids_returns_md_files(self, reader):
        ids = reader.fileids()
        assert isinstance(ids, list)
        assert len(ids) == 2
        assert all(f.endswith(".html.md") for f in ids)

    def test_root_is_path(self, reader, epistolae_dir):
        assert reader.root == epistolae_dir.resolve()


class TestEpistolaeReaderTexts:
    """Latin-only text extraction."""

    @pytest.fixture
    def reader(self, epistolae_dir):
        from latincyreaders import EpistolaeReader
        return EpistolaeReader(root=epistolae_dir)

    def test_texts_yields_strings(self, reader):
        texts = list(reader.texts())
        assert len(texts) == 2
        assert all(isinstance(t, str) for t in texts)

    def test_texts_contain_latin(self, reader):
        all_text = " ".join(reader.texts())
        assert "venerabilis" in all_text
        assert "Domino" in all_text

    def test_texts_exclude_english_translation(self, reader):
        all_text = " ".join(reader.texts())
        assert "venerable father" not in all_text
        assert "should not be extracted" not in all_text

    def test_texts_exclude_historical_context(self, reader):
        all_text = " ".join(reader.texts())
        assert "celebrated letters" not in all_text
        assert "early correspondence" not in all_text

    def test_texts_exclude_html_tags(self, reader):
        all_text = " ".join(reader.texts())
        assert "<h2" not in all_text
        assert "<p>" not in all_text
        assert "<em>" not in all_text


class TestEpistolaeReaderHeaders:
    """Metadata from YAML frontmatter."""

    @pytest.fixture
    def reader(self, epistolae_dir):
        from latincyreaders import EpistolaeReader
        return EpistolaeReader(root=epistolae_dir)

    def test_headers_yields_dicts(self, reader):
        headers = list(reader.headers())
        assert len(headers) == 2
        assert all(isinstance(h, dict) for h in headers)

    def test_headers_extract_letter_id(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert headers["1.html.md"]["letter_id"] == "10001"
        assert headers["2.html.md"]["letter_id"] == "10002"

    def test_headers_extract_date(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert headers["1.html.md"]["date"] == "1146"
        assert headers["2.html.md"]["date"] == "1133"

    def test_headers_extract_senders(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert headers["1.html.md"]["senders"] == ["Hildegard of Bingen"]
        assert headers["2.html.md"]["senders"] == ["Heloise"]

    def test_headers_extract_receivers(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert headers["1.html.md"]["receivers"] == ["Bernard of Clairvaux"]
        assert headers["2.html.md"]["receivers"] == ["Peter Abelard"]

    def test_headers_extract_title(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert "Hildegard" in headers["1.html.md"]["title"]

    def test_headers_include_filename(self, reader):
        headers = list(reader.headers())
        assert all("filename" in h for h in headers)


class TestEpistolaeReaderDocs:
    """spaCy Doc output and metadata."""

    @pytest.fixture
    def reader(self, epistolae_dir):
        from latincyreaders import EpistolaeReader
        return EpistolaeReader(
            root=epistolae_dir,
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    def test_docs_yield_spacy_docs(self, reader):
        from spacy.tokens import Doc
        docs = list(reader.docs())
        assert len(docs) == 2
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_have_fileid(self, reader):
        for doc in reader.docs():
            assert doc._.fileid is not None
            assert doc._.fileid.endswith(".html.md")

    def test_docs_metadata_has_letter_id(self, reader):
        ids = {doc._.fileid: doc._.metadata.get("letter_id") for doc in reader.docs()}
        assert ids["1.html.md"] == "10001"
        assert ids["2.html.md"] == "10002"

    def test_docs_metadata_has_senders(self, reader):
        for doc in reader.docs():
            assert "senders" in doc._.metadata
            assert isinstance(doc._.metadata["senders"], list)

    def test_docs_contain_latin_tokens(self, reader):
        all_tokens = [t.text for doc in reader.docs() for t in doc]
        assert "venerabilis" in all_tokens
        assert "Domino" in all_tokens
