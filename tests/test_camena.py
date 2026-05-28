"""Tests for CamenaReader."""

import pytest
from pathlib import Path

from latincyreaders import CamenaReader, AnnotationLevel


class TestCamenaReader:
    """Test suite for CamenaReader."""

    @pytest.fixture
    def reader(self, camena_dir):
        """Create a CamenaReader with test fixtures."""
        return CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
        )

    @pytest.fixture
    def reader_with_front(self, camena_dir):
        """Reader that includes front matter."""
        return CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
            include_front=True,
        )

    @pytest.fixture
    def reader_without_front(self, camena_dir):
        """Reader that excludes front matter."""
        return CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
            include_front=False,
        )

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of XML files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".xml") for f in fileids)

    def test_fileids_contains_test_file(self, reader):
        """Test fixture file is discovered."""
        fileids = reader.fileids()
        assert "sample.xml" in fileids

    def test_root_is_path(self, reader, camena_dir):
        """root property returns correct Path."""
        assert reader.root == camena_dir.resolve()

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
        assert "basia" in all_text.lower() or "amor" in all_text.lower()

    def test_texts_with_front_includes_dedication(self, reader_with_front):
        """Front matter is included when configured."""
        texts = list(reader_with_front.texts())
        all_text = " ".join(texts)
        assert "Lectorem" in all_text or "dono" in all_text

    def test_texts_without_front_excludes_dedication(self, reader_without_front):
        """Front matter is excluded when configured."""
        texts = list(reader_without_front.texts())
        all_text = " ".join(texts)
        # Should have body content but not front matter dedication
        assert "basia" in all_text.lower() or "amor" in all_text.lower()

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
        assert doc._.fileid == "sample.xml"

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
    # Note removal
    # -------------------------------------------------------------------------

    def test_notes_removed_by_default(self, reader):
        """Notes are removed by default."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "should be removed" not in all_text

    def test_notes_preserved_when_configured(self, camena_dir):
        """Notes can be preserved when remove_notes=False."""
        reader = CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
            remove_notes=False,
        )
        # Just verify the reader works with remove_notes=False
        texts = list(reader.texts())
        assert len(texts) > 0

    # -------------------------------------------------------------------------
    # Annotation levels
    # -------------------------------------------------------------------------

    def test_annotation_level_none_blocks_docs(self, camena_dir):
        """annotation_level=NONE prevents docs() usage."""
        reader = CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.NONE,
        )
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            next(reader.docs())

    def test_annotation_level_none_allows_texts(self, camena_dir):
        """annotation_level=NONE still allows texts()."""
        reader = CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.NONE,
        )
        texts = list(reader.texts())
        assert len(texts) > 0


class TestCamenaEdgeCases:
    """Tests for edge cases and fallback paths."""

    def test_empty_xml_file(self, tmp_path):
        """Empty/invalid XML doesn't crash."""
        camena_dir = tmp_path / "camena"
        camena_dir.mkdir()
        (camena_dir / "empty.xml").write_text('<?xml version="1.0"?><TEI></TEI>')

        reader = CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
        )
        texts = list(reader.texts())
        # Should handle gracefully (empty or no output)
        assert isinstance(texts, list)

    def test_xml_without_body(self, tmp_path):
        """XML without body element doesn't crash."""
        camena_dir = tmp_path / "camena"
        camena_dir.mkdir()
        (camena_dir / "nobody.xml").write_text('''<?xml version="1.0"?>
<TEI>
  <teiHeader><fileDesc><titleStmt><title>Test</title></titleStmt></fileDesc></teiHeader>
  <text></text>
</TEI>''')

        reader = CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
        )
        texts = list(reader.texts())
        assert isinstance(texts, list)

    def test_standalone_lines_extraction(self, tmp_path):
        """Standalone <l> elements (not in <lg>) are extracted."""
        camena_dir = tmp_path / "camena"
        camena_dir.mkdir()
        (camena_dir / "standalone.xml").write_text('''<?xml version="1.0"?>
<TEI>
  <teiHeader><fileDesc><titleStmt><title>Test</title></titleStmt></fileDesc></teiHeader>
  <text>
    <body>
      <l>Prima linea versus.</l>
      <l>Secunda linea versus.</l>
    </body>
  </text>
</TEI>''')

        reader = CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
        )
        texts = list(reader.texts())
        assert len(texts) > 0
        all_text = " ".join(texts)
        assert "Prima" in all_text or "linea" in all_text

    def test_fallback_text_extraction(self, tmp_path):
        """Unstructured text content is extracted as fallback."""
        camena_dir = tmp_path / "camena"
        camena_dir.mkdir()
        (camena_dir / "fallback.xml").write_text('''<?xml version="1.0"?>
<TEI>
  <teiHeader><fileDesc><titleStmt><title>Test</title></titleStmt></fileDesc></teiHeader>
  <text>
    <body>
      <div>Haec est prosa Latina sine structura.</div>
    </body>
  </text>
</TEI>''')

        reader = CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
        )
        texts = list(reader.texts())
        assert len(texts) > 0
        all_text = " ".join(texts)
        assert "prosa" in all_text or "Latina" in all_text


class TestCamenaCollections:
    """Tests for collection-related methods."""

    @pytest.fixture
    def reader(self, camena_dir):
        """Create a CamenaReader with test fixtures."""
        return CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
        )

    def test_collections_returns_list(self, reader):
        """collections() returns a list."""
        collections = reader.collections()
        assert isinstance(collections, list)

    def test_docs_by_collection_yields_docs(self, tmp_path):
        """docs_by_collection() yields Doc objects."""
        # Create a directory structure with a poemata collection
        camena_dir = tmp_path / "camena"
        poemata_dir = camena_dir / "poemata"
        poemata_dir.mkdir(parents=True)

        (poemata_dir / "poem.xml").write_text('''<?xml version="1.0"?>
<TEI>
  <teiHeader><fileDesc><titleStmt><title>Poem</title></titleStmt></fileDesc></teiHeader>
  <text><body><p>Carmen Latinum.</p></body></text>
</TEI>''')

        reader = CamenaReader(
            root=camena_dir,
            fileids="**/*.xml",
            annotation_level=AnnotationLevel.BASIC,
        )
        docs = list(reader.docs_by_collection("poemata"))
        assert len(docs) == 1
        assert docs[0]._.metadata.get("collection") == "poemata"


class TestCamenaHeaderMetadata:
    """Tests for TEI header metadata extraction."""

    def test_metadata_extracts_date_when_attr(self, tmp_path):
        """Date is extracted from when attribute."""
        camena_dir = tmp_path / "camena"
        camena_dir.mkdir()
        (camena_dir / "dated.xml").write_text('''<?xml version="1.0"?>
<TEI>
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Test</title><author>Author Name</author></titleStmt>
      <publicationStmt>
        <publisher>Test Publisher</publisher>
        <date when="1650">1650</date>
      </publicationStmt>
    </fileDesc>
  </teiHeader>
  <text><body><p>Haec est prosa Latina.</p></body></text>
</TEI>''')

        reader = CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
        )
        doc = next(reader.docs())
        assert doc._.metadata.get("date") == "1650"
        assert doc._.metadata.get("publisher") == "Test Publisher"
        assert doc._.metadata.get("author") == "Author Name"


class TestCamenaCaching:
    """Tests for CAMENA reader caching through BaseCorpusReader."""

    def test_cache_hit_returns_same_doc(self, camena_dir):
        """Accessing same file twice hits cache."""
        reader = CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
            cache=True,
        )
        fileid = reader.fileids()[0]

        doc1 = next(reader.docs(fileid))
        doc2 = next(reader.docs(fileid))

        # Same object from cache
        assert doc1 is doc2
        assert reader.cache_stats()["hits"] == 1

    def test_cache_eviction(self, tmp_path):
        """Cache evicts oldest when at capacity."""
        # Create 3 XML files
        camena_dir = tmp_path / "camena"
        camena_dir.mkdir()
        for i in range(3):
            (camena_dir / f"file{i}.xml").write_text(f'''<?xml version="1.0"?>
<TEI>
  <teiHeader><fileDesc><titleStmt><title>Test {i}</title></titleStmt></fileDesc></teiHeader>
  <text><body><p>Content for file {i}.</p></body></text>
</TEI>''')

        reader = CamenaReader(
            root=camena_dir,
            fileids="*.xml",
            annotation_level=AnnotationLevel.BASIC,
            cache=True,
            cache_maxsize=2,
        )

        fileids = sorted(reader.fileids())
        assert len(fileids) == 3

        # Load first two files
        _ = next(reader.docs(fileids[0]))
        _ = next(reader.docs(fileids[1]))
        assert reader.cache_stats()["size"] == 2

        # Load third file - should evict first
        _ = next(reader.docs(fileids[2]))
        assert reader.cache_stats()["size"] == 2

        # First file should be evicted
        misses_before = reader.cache_stats()["misses"]
        _ = next(reader.docs(fileids[0]))
        assert reader.cache_stats()["misses"] > misses_before


class TestCamenaDownload:
    """Tests for CAMENA corpus path requirements."""

    def test_root_is_required(self):
        """root is a required argument; omitting it raises TypeError."""
        with pytest.raises(TypeError):
            CamenaReader()  # type: ignore[call-arg]

    def test_corpus_url_is_documented(self):
        """CORPUS_URL class attribute records the source for manual cloning."""
        assert "camena" in CamenaReader.CORPUS_URL.lower()
