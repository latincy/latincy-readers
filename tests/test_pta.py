"""Tests for PTAReader (Patristic Text Archive)."""

import pytest
from pathlib import Path

from latincyreaders import PTAReader, AnnotationLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reader(pta_dir):
    """PTAReader over all fixture files, TOKENIZE level."""
    return PTAReader(
        root=pta_dir,
        fileids="*.xml",
        annotation_level=AnnotationLevel.MINIMAL,
    )


@pytest.fixture
def reader_lat(pta_dir):
    """PTAReader restricted to Latin fixture files."""
    return PTAReader(
        root=pta_dir,
        fileids="*lat*.xml",
        annotation_level=AnnotationLevel.MINIMAL,
    )


@pytest.fixture
def reader_grc(pta_dir):
    """PTAReader restricted to Greek fixture files."""
    return PTAReader(
        root=pta_dir,
        fileids="*grc*.xml",
        annotation_level=AnnotationLevel.MINIMAL,
        model_name="grc_dep_treebanks_trf",
        lang="grc",
    )


@pytest.fixture
def reader_minimal(pta_dir):
    """PTAReader over the minimal synthetic fixture only."""
    return PTAReader(
        root=pta_dir,
        fileids="pta9999*.xml",
        annotation_level=AnnotationLevel.MINIMAL,
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


class TestPTAReaderFileids:
    def test_fileids_returns_list(self, reader):
        """fileids() returns a non-empty list of XML files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".xml") for f in fileids)

    def test_lat_file_discovered(self, pta_dir):
        """Latin fixture file is discovered."""
        reader = PTAReader(root=pta_dir, fileids="*lat*.xml",
                           annotation_level=AnnotationLevel.NONE)
        fids = reader.fileids()
        assert any("lat" in f for f in fids)

    def test_grc_file_discovered(self, pta_dir):
        """Greek fixture file is discovered."""
        reader = PTAReader(root=pta_dir, fileids="*grc*.xml",
                           annotation_level=AnnotationLevel.NONE)
        fids = reader.fileids()
        assert any("grc" in f for f in fids)


# ---------------------------------------------------------------------------
# Raw text extraction
# ---------------------------------------------------------------------------


class TestPTAReaderTexts:
    def test_latin_file_yields_text(self, reader_lat):
        """Latin PTA files yield non-empty text strings."""
        texts = list(reader_lat.texts())
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)
        assert all(len(t) > 0 for t in texts)

    def test_greek_file_yields_text(self, reader_grc):
        """Greek PTA files yield non-empty text strings."""
        texts = list(reader_grc.texts())
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)
        assert all(len(t) > 0 for t in texts)

    def test_no_note_text_in_output(self, reader_minimal):
        """Body <note> elements are stripped from extracted text."""
        texts = list(reader_minimal.texts())
        assert len(texts) > 0
        combined = " ".join(texts)
        assert "do not include this note text" not in combined

    def test_latin_text_contains_latin(self, reader_lat):
        """Extracted Latin text contains expected Latin words."""
        text = next(reader_lat.texts())
        # The Latin fixture (pta0001.pta014.pta-lat1.xml) is a homily —
        # check for a distinctive Latin word from the known text.
        assert "Abraham" in text or "Dominus" in text or "Christus" in text


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestPTAReaderMetadata:
    def test_docs_have_metadata(self, reader_lat):
        """Docs have expected metadata keys present."""
        doc = next(reader_lat.docs())
        meta = doc._.metadata
        assert meta is not None
        for key in ("filename", "path", "urn", "language", "author", "title"):
            assert key in meta, f"missing key: {key}"

    def test_language_detected_latin(self, reader_lat):
        """Language is detected as 'lat' for Latin files."""
        doc = next(reader_lat.docs())
        assert doc._.metadata["language"] == "lat"

    def test_language_detected_greek(self, reader_grc):
        """Language is detected as 'grc' for Greek files."""
        doc = next(reader_grc.docs())
        assert doc._.metadata["language"] == "grc"

    def test_cts_urn_in_metadata(self, reader_lat):
        """CTS URN is present and well-formed in metadata."""
        doc = next(reader_lat.docs())
        urn = doc._.metadata["urn"]
        assert urn.startswith("urn:cts:pta:")
        assert "pta0001.pta014" in urn

    def test_author_in_metadata(self, reader_lat):
        """Author name is extracted from teiHeader."""
        doc = next(reader_lat.docs())
        author = doc._.metadata["author"]
        assert author
        assert "Severianus" in author or "Gabalensis" in author

    def test_title_in_metadata(self, reader_lat):
        """Title is extracted from teiHeader."""
        doc = next(reader_lat.docs())
        title = doc._.metadata["title"]
        assert title
        assert len(title) > 0

    def test_citation_in_metadata(self, reader_lat):
        """Each doc chunk has a citation (section number) in metadata."""
        doc = next(reader_lat.docs())
        assert "citation" in doc._.metadata
        citation = doc._.metadata["citation"]
        assert citation  # non-empty

    def test_div_type_in_metadata(self, reader_lat):
        """div_type is present in metadata."""
        doc = next(reader_lat.docs())
        assert "div_type" in doc._.metadata

    def test_div_n_in_metadata(self, reader_lat):
        """div_n is present in metadata."""
        doc = next(reader_lat.docs())
        assert "div_n" in doc._.metadata
        assert doc._.metadata["div_n"] == "1"


# ---------------------------------------------------------------------------
# Per-section chunking
# ---------------------------------------------------------------------------


class TestPTAReaderChunking:
    def test_greek_file_yields_multiple_docs(self, reader_grc):
        """Greek file (15 sections) yields multiple Doc objects."""
        docs = list(reader_grc.docs())
        assert len(docs) > 1

    def test_latin_file_yields_at_least_one_doc(self, reader_lat):
        """Latin file (1 section) yields at least one Doc."""
        docs = list(reader_lat.docs())
        assert len(docs) >= 1

    def test_each_doc_has_distinct_citation(self, reader_grc):
        """Each Doc from a multi-section file has a distinct citation."""
        docs = list(reader_grc.docs())
        citations = [d._.metadata.get("citation") for d in docs]
        assert len(citations) == len(set(citations)), "citations should be unique per section"
