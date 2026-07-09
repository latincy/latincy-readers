"""Tests for FileSelector fluent API."""

import pytest
from pathlib import Path

from latincyreaders import TesseraeReader, AnnotationLevel
from latincyreaders.core import FileSelector


class TestFileSelectorBasics:
    """Test basic FileSelector operations."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        """Create a TesseraeReader with test fixtures."""
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )

    def test_select_returns_file_selector(self, reader):
        """reader.select() returns a FileSelector instance."""
        selector = reader.select()
        assert isinstance(selector, FileSelector)

    def test_selector_is_iterable(self, reader):
        """FileSelector is iterable."""
        selector = reader.select()
        fileids = list(selector)
        assert isinstance(fileids, list)
        assert len(fileids) > 0

    def test_selector_len(self, reader):
        """len(FileSelector) returns count of matching files."""
        selector = reader.select()
        assert len(selector) > 0
        assert len(selector) == len(list(selector))

    def test_selector_count(self, reader):
        """FileSelector.count() returns count of matching files."""
        selector = reader.select()
        assert selector.count() > 0
        assert selector.count() == len(selector)

    def test_selector_to_list(self, reader):
        """FileSelector.to_list() materializes results."""
        selector = reader.select()
        result = selector.to_list()
        assert isinstance(result, list)
        assert result == list(selector)

    def test_selector_preview(self, reader):
        """FileSelector.preview(n) returns first n matches."""
        selector = reader.select()
        preview = selector.preview(1)
        assert isinstance(preview, list)
        assert len(preview) <= 1


class TestFileSelectorMatch:
    """Test FileSelector.match() regex filtering."""

    @pytest.fixture
    def reader_with_multiple_files(self, tmp_path):
        """Create reader with multiple test files."""
        # Create test .tess files
        (tmp_path / "vergil.aeneid.tess").write_text("<verg. aen. 1.1> Arma virumque cano\n")
        (tmp_path / "vergil.eclogues.tess").write_text("<verg. ecl. 1.1> Tityre tu patulae\n")
        (tmp_path / "ovid.metamorphoses.tess").write_text("<ov. met. 1.1> In nova fert animus\n")
        (tmp_path / "catullus.carmina.tess").write_text("<catull. 1.1> Cui dono lepidum\n")

        return TesseraeReader(
            root=tmp_path,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )

    def test_match_filters_by_pattern(self, reader_with_multiple_files):
        """match() filters files by regex pattern."""
        selector = reader_with_multiple_files.select().match("vergil")
        result = selector.to_list()
        assert len(result) == 2
        assert all("vergil" in f for f in result)

    def test_match_is_case_insensitive(self, reader_with_multiple_files):
        """match() is case-insensitive."""
        upper = reader_with_multiple_files.select().match("VERGIL").to_list()
        lower = reader_with_multiple_files.select().match("vergil").to_list()
        assert upper == lower

    def test_match_chaining(self, reader_with_multiple_files):
        """match() can be chained."""
        selector = reader_with_multiple_files.select().match("vergil").match("aeneid")
        result = selector.to_list()
        assert len(result) == 1
        assert "aeneid" in result[0]


class TestFileSelectorWhere:
    """Test FileSelector.where() metadata filtering."""

    @pytest.fixture
    def reader_with_metadata(self, tmp_path):
        """Create reader with files and metadata."""
        # Create test files under texts/ (the real Tesserae layout)
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "vergil.aeneid.tess").write_text("<verg. aen. 1.1> Arma virumque cano\n")
        (texts_dir / "vergil.eclogues.tess").write_text("<verg. ecl. 1.1> Tityre tu patulae\n")
        (texts_dir / "ovid.metamorphoses.tess").write_text("<ov. met. 1.1> In nova fert animus\n")
        (texts_dir / "catullus.carmina.tess").write_text("<catull. 1.1> Cui dono lepidum\n")

        # Create metadata (bare filename keys, normalized to texts/<name>)
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        import json
        (metadata_dir / "authors.json").write_text(json.dumps({
            "vergil.aeneid.tess": {"author": "Vergil", "genre": "epic", "date": -19},
            "vergil.eclogues.tess": {"author": "Vergil", "genre": "pastoral", "date": -37},
            "ovid.metamorphoses.tess": {"author": "Ovid", "genre": "epic", "date": 8},
            "catullus.carmina.tess": {"author": "Catullus", "genre": "lyric", "date": -54},
        }))

        return TesseraeReader(
            root=tmp_path,
            fileids="**/*.tess",
            annotation_level=AnnotationLevel.NONE,
        )

    def test_where_exact_match(self, reader_with_metadata):
        """where(field=value) filters by exact metadata match."""
        selector = reader_with_metadata.select().where(author="Vergil")
        result = selector.to_list()
        assert len(result) == 2
        assert all("vergil" in f for f in result)

    def test_where_multiple_fields(self, reader_with_metadata):
        """where() with multiple fields ANDs them together."""
        selector = reader_with_metadata.select().where(author="Vergil", genre="epic")
        result = selector.to_list()
        assert len(result) == 1
        assert "aeneid" in result[0]

    def test_where_in_operator(self, reader_with_metadata):
        """where(field__in=[...]) matches any of the values."""
        selector = reader_with_metadata.select().where(author__in=["Vergil", "Ovid"])
        result = selector.to_list()
        assert len(result) == 3  # 2 Vergil + 1 Ovid

    def test_where_chaining(self, reader_with_metadata):
        """where() can be chained."""
        selector = reader_with_metadata.select().where(author="Vergil").where(genre="epic")
        result = selector.to_list()
        assert len(result) == 1
        assert "aeneid" in result[0]

    def test_where_unknown_field_excludes_file(self, reader_with_metadata):
        """Files missing the metadata field are excluded."""
        selector = reader_with_metadata.select().where(unknown_field="value")
        result = selector.to_list()
        assert len(result) == 0


class TestFileSelectorWhereOperators:
    """Test where() operator validation."""

    @pytest.fixture
    def reader_with_metadata(self, tmp_path):
        """Create reader with metadata."""
        (tmp_path / "test.tess").write_text("<test. 1> Test\n")
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        import json
        (metadata_dir / "meta.json").write_text(json.dumps({
            "test.tess": {"author": "Test", "date": -50}
        }))
        return TesseraeReader(
            root=tmp_path,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )

    def test_unknown_operator_raises(self, reader_with_metadata):
        """Unknown operators raise ValueError."""
        with pytest.raises(ValueError, match="Unknown operator"):
            reader_with_metadata.select().where(author__unknown=["value"])

    def test_conflicting_filters_raises(self, reader_with_metadata):
        """Conflicting filters on same field raise ValueError."""
        with pytest.raises(ValueError, match="[Cc]onflict"):
            reader_with_metadata.select().where(author="Test").where(author__in=["Other"])


class TestFileSelectorWhereBetween:
    """Test FileSelector.where_between() numeric range filtering."""

    @pytest.fixture
    def reader_with_dates(self, tmp_path):
        """Create reader with dated files."""
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "early.tess").write_text("<early. 1> Early text\n")
        (texts_dir / "middle.tess").write_text("<middle. 1> Middle text\n")
        (texts_dir / "late.tess").write_text("<late. 1> Late text\n")

        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        import json
        (metadata_dir / "dates.json").write_text(json.dumps({
            "early.tess": {"date": -100, "lines": 50},
            "middle.tess": {"date": 0, "lines": 100},
            "late.tess": {"date": 100, "lines": 200},
        }))

        return TesseraeReader(
            root=tmp_path,
            fileids="**/*.tess",
            annotation_level=AnnotationLevel.NONE,
        )

    def test_where_between_inclusive(self, reader_with_dates):
        """where_between() is inclusive on both ends."""
        selector = reader_with_dates.select().where_between("date", -100, 0)
        result = selector.to_list()
        assert len(result) == 2
        assert "texts/early.tess" in result
        assert "texts/middle.tess" in result

    def test_where_between_any_field(self, reader_with_dates):
        """where_between() works on any numeric field."""
        selector = reader_with_dates.select().where_between("lines", 75, 150)
        result = selector.to_list()
        assert len(result) == 1
        assert "texts/middle.tess" in result

    def test_where_between_missing_field(self, reader_with_dates):
        """Files missing the field are excluded."""
        selector = reader_with_dates.select().where_between("unknown", 0, 100)
        result = selector.to_list()
        assert len(result) == 0


class TestFileSelectorDateRange:
    """Test FileSelector.date_range() convenience method."""

    @pytest.fixture
    def reader_with_dates(self, tmp_path):
        """Create reader with dated files."""
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "republic.tess").write_text("<rep. 1> Republic era\n")
        (texts_dir / "augustan.tess").write_text("<aug. 1> Augustan era\n")
        (texts_dir / "imperial.tess").write_text("<imp. 1> Imperial era\n")

        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        import json
        (metadata_dir / "dates.json").write_text(json.dumps({
            "republic.tess": {"date": -50},
            "augustan.tess": {"date": 10},
            "imperial.tess": {"date": 100},
        }))

        return TesseraeReader(
            root=tmp_path,
            fileids="**/*.tess",
            annotation_level=AnnotationLevel.NONE,
        )

    def test_date_range_filters_by_date(self, reader_with_dates):
        """date_range() filters by 'date' metadata field."""
        selector = reader_with_dates.select().date_range(-50, 50)
        result = selector.to_list()
        assert len(result) == 2
        assert "texts/republic.tess" in result
        assert "texts/augustan.tess" in result

    def test_date_range_is_convenience_for_where_between(self, reader_with_dates):
        """date_range() is equivalent to where_between('date', ...)."""
        via_date_range = reader_with_dates.select().date_range(-50, 50).to_list()
        via_where_between = reader_with_dates.select().where_between("date", -50, 50).to_list()
        assert via_date_range == via_where_between


class TestFileSelectorEmptyResults:
    """Test FileSelector behavior with empty results."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        """Create a TesseraeReader."""
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )

    def test_empty_result_iterates_empty(self, reader):
        """Empty results iterate as empty (not an error)."""
        selector = reader.select().match("nonexistent_pattern_xyz")
        result = list(selector)
        assert result == []

    def test_empty_result_len_zero(self, reader):
        """Empty results have len() == 0."""
        selector = reader.select().match("nonexistent_pattern_xyz")
        assert len(selector) == 0

    def test_empty_result_count_zero(self, reader):
        """Empty results have count() == 0."""
        selector = reader.select().match("nonexistent_pattern_xyz")
        assert selector.count() == 0

    def test_empty_result_preview_empty(self, reader):
        """Empty results preview() returns empty list."""
        selector = reader.select().match("nonexistent_pattern_xyz")
        assert selector.preview(5) == []


class TestFileSelectorIntegration:
    """Test FileSelector integration with reader methods."""

    @pytest.fixture
    def reader_with_metadata(self, tmp_path):
        """Create reader with multiple files and metadata."""
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "vergil.aeneid.tess").write_text("<verg. aen. 1.1> Arma virumque cano\n")
        (texts_dir / "ovid.metamorphoses.tess").write_text("<ov. met. 1.1> In nova fert animus\n")

        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        import json
        (metadata_dir / "meta.json").write_text(json.dumps({
            "vergil.aeneid.tess": {"author": "Vergil"},
            "ovid.metamorphoses.tess": {"author": "Ovid"},
        }))

        return TesseraeReader(
            root=tmp_path,
            fileids="**/*.tess",
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    def test_docs_accepts_selector(self, reader_with_metadata):
        """reader.docs() accepts a FileSelector."""
        from spacy.tokens import Doc

        selector = reader_with_metadata.select().where(author="Vergil")
        docs = list(reader_with_metadata.docs(selector))

        assert len(docs) == 1
        assert isinstance(docs[0], Doc)
        assert "vergil" in docs[0]._.fileid.lower()

    def test_texts_accepts_selector(self, reader_with_metadata):
        """reader.texts() accepts a FileSelector."""
        selector = reader_with_metadata.select().where(author="Ovid")
        texts = list(reader_with_metadata.texts(selector))

        assert len(texts) == 1
        assert "nova" in texts[0].lower()

    def test_sents_accepts_selector(self, reader_with_metadata):
        """reader.sents() accepts a FileSelector."""
        selector = reader_with_metadata.select().match("vergil")
        sents = list(reader_with_metadata.sents(selector))

        assert len(sents) >= 1

    def test_tokens_accepts_selector(self, reader_with_metadata):
        """reader.tokens() accepts a FileSelector."""
        selector = reader_with_metadata.select().match("ovid")
        tokens = list(reader_with_metadata.tokens(selector))

        assert len(tokens) >= 1


class TestFileSelectorChaining:
    """Test complex chaining scenarios."""

    @pytest.fixture
    def reader_with_metadata(self, tmp_path):
        """Create reader with complex test data."""
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "vergil.aeneid.tess").write_text("<verg. aen. 1.1> Arma virumque\n")
        (texts_dir / "vergil.eclogues.tess").write_text("<verg. ecl. 1.1> Tityre\n")
        (texts_dir / "ovid.metamorphoses.tess").write_text("<ov. met. 1.1> In nova\n")
        (texts_dir / "lucan.pharsalia.tess").write_text("<luc. phar. 1.1> Bella per\n")

        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        import json
        (metadata_dir / "meta.json").write_text(json.dumps({
            "vergil.aeneid.tess": {"author": "Vergil", "genre": "epic", "date": -19},
            "vergil.eclogues.tess": {"author": "Vergil", "genre": "pastoral", "date": -37},
            "ovid.metamorphoses.tess": {"author": "Ovid", "genre": "epic", "date": 8},
            "lucan.pharsalia.tess": {"author": "Lucan", "genre": "epic", "date": 65},
        }))

        return TesseraeReader(
            root=tmp_path,
            fileids="**/*.tess",
            annotation_level=AnnotationLevel.NONE,
        )

    def test_complex_chain_match_where_date(self, reader_with_metadata):
        """Complex chain: match + where + date_range."""
        selector = (
            reader_with_metadata.select()
            .where(genre="epic")
            .date_range(-50, 50)
        )
        result = selector.to_list()
        assert len(result) == 2
        assert "texts/vergil.aeneid.tess" in result
        assert "texts/ovid.metamorphoses.tess" in result

    def test_fluent_api_returns_new_selector(self, reader_with_metadata):
        """Each method returns a new FileSelector (immutable chaining)."""
        s1 = reader_with_metadata.select()
        s2 = s1.match("vergil")
        s3 = s2.where(genre="epic")

        # Each should be a different selector
        assert len(s1) == 4
        assert len(s2) == 2
        assert len(s3) == 1
