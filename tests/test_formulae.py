"""Tests for FormulaeReader."""

import pytest
from pathlib import Path

from latincyreaders import AnnotationLevel


class TestFormulaeReaderFileids:
    """File listing."""

    @pytest.fixture
    def reader(self, formulae_dir):
        from latincyreaders import FormulaeReader
        return FormulaeReader(root=formulae_dir)

    def test_fileids_returns_lat_xml_files(self, reader):
        ids = reader.fileids()
        assert isinstance(ids, list)
        assert len(ids) == 2
        assert all(".lat" in f for f in ids)

    def test_fileids_excludes_capitains(self, reader):
        ids = reader.fileids()
        assert not any("__capitains__" in f for f in ids)

    def test_root_is_path(self, reader, formulae_dir):
        assert reader.root == formulae_dir.resolve()


class TestFormulaeReaderTexts:
    """Text extraction from <w>-tokenized TEI."""

    @pytest.fixture
    def reader(self, formulae_dir):
        from latincyreaders import FormulaeReader
        return FormulaeReader(root=formulae_dir)

    def test_texts_yields_strings(self, reader):
        texts = list(reader.texts())
        assert len(texts) == 2
        assert all(isinstance(t, str) for t in texts)

    def test_texts_contain_latin_words(self, reader):
        all_text = " ".join(reader.texts())
        assert "Notum" in all_text
        assert "Multipliciter" in all_text

    def test_texts_exclude_french_front_matter(self, reader):
        all_text = " ".join(reader.texts())
        assert "français" not in all_text
        assert "Résumé" not in all_text
        assert "Carta Fulconis" not in all_text

    def test_texts_strip_lemmaref_attributes(self, reader):
        """lemmaRef attribute values must not appear as text."""
        all_text = " ".join(reader.texts())
        assert "lemmaRef" not in all_text
        assert "misericordia" in all_text  # word text preserved


class TestFormulaeReaderHeaders:
    """Metadata extraction from teiHeader and body."""

    @pytest.fixture
    def reader(self, formulae_dir):
        from latincyreaders import FormulaeReader
        return FormulaeReader(root=formulae_dir)

    def test_headers_yields_dicts(self, reader):
        headers = list(reader.headers())
        assert len(headers) == 2
        assert all(isinstance(h, dict) for h in headers)

    def test_headers_extract_cts_urn(self, reader):
        urns = {h["filename"]: h.get("cts_urn") for h in reader.headers()}
        assert urns["redon.courson0001.lat001.xml"] == "urn:cts:formulae:redon.courson0001.lat001"
        assert urns["anjou.marchegay0001.lat001.xml"] == "urn:cts:formulae:anjou.marchegay0001.lat001"

    def test_headers_extract_collection(self, reader):
        colls = {h["filename"]: h.get("collection") for h in reader.headers()}
        assert colls["redon.courson0001.lat001.xml"] == "redon"
        assert colls["anjou.marchegay0001.lat001.xml"] == "anjou"

    def test_headers_extract_title(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert "Redon" in headers["redon.courson0001.lat001.xml"]["title"]

    def test_headers_extract_date(self, reader):
        headers = {h["filename"]: h for h in reader.headers()}
        assert headers["redon.courson0001.lat001.xml"]["date"] == "832"
        assert headers["anjou.marchegay0001.lat001.xml"]["date"] == "1000"

    def test_headers_include_filename(self, reader):
        headers = list(reader.headers())
        assert all("filename" in h for h in headers)


class TestFormulaeReaderDocs:
    """spaCy Doc output and metadata."""

    @pytest.fixture
    def reader(self, formulae_dir):
        from latincyreaders import FormulaeReader
        return FormulaeReader(
            root=formulae_dir,
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
            assert ".lat" in doc._.fileid

    def test_docs_have_cts_urn(self, reader):
        for doc in reader.docs():
            assert "cts_urn" in doc._.metadata
            assert doc._.metadata["cts_urn"].startswith("urn:cts:formulae:")

    def test_docs_have_collection(self, reader):
        colls = {doc._.fileid: doc._.metadata.get("collection") for doc in reader.docs()}
        assert "redon" in colls.values()
        assert "anjou" in colls.values()

    def test_docs_contain_latin_tokens(self, reader):
        all_tokens = [t.text for doc in reader.docs() for t in doc]
        assert "Notum" in all_tokens
        assert "Multipliciter" in all_tokens
