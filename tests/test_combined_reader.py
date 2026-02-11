"""Tests for CombinedReader."""

import pytest
from unittest.mock import MagicMock
from collections.abc import Iterator

from latincyreaders.core.combined import CombinedReader, combine


# ---------------------------------------------------------------------------
# Helpers — stub readers with controllable class names
# ---------------------------------------------------------------------------


def _make_stub_reader(cls_name, fileids_list, texts_data=None, metadata_data=None):
    """Create a MagicMock whose type has the desired __name__.

    Dynamically creates a MagicMock subclass so that
    ``type(reader).__name__`` returns the desired class name.
    """
    klass = type(cls_name, (MagicMock,), {})
    reader = klass()

    reader.fileids.return_value = fileids_list

    if texts_data is not None:
        reader.texts.side_effect = lambda fileids=None: iter(texts_data)
    else:
        reader.texts.side_effect = lambda fileids=None: iter(
            [f"text from {fid}" for fid in fileids_list]
        )

    if metadata_data is not None:
        reader.metadata.side_effect = lambda fileids=None: iter(
            metadata_data.items()
        )
    else:
        reader.metadata.side_effect = lambda fileids=None: iter(
            [(fid, {"source": cls_name}) for fid in fileids_list]
        )

    reader.docs.side_effect = lambda fileids=None: iter([])
    reader.sents.side_effect = lambda fileids=None, as_text=False: iter([])
    reader.tokens.side_effect = lambda fileids=None, as_text=False: iter([])
    reader.find_sents.side_effect = lambda fileids=None, **kw: iter([])
    reader.kwic.side_effect = lambda keyword, fileids=None, **kw: iter([])
    reader.ngrams.side_effect = lambda n=2, fileids=None, **kw: iter([])
    reader.skipgrams.side_effect = lambda n=2, k=1, fileids=None, **kw: iter([])
    reader.concordance.side_effect = lambda fileids=None, **kw: {}

    return reader


@pytest.fixture
def reader_a():
    """Stub reader simulating TesseraeReader."""
    return _make_stub_reader(
        "TesseraeReader",
        ["vergil.aeneid.tess", "ovid.met.tess"],
        texts_data=["Arma virumque cano", "In nova fert animus"],
    )


@pytest.fixture
def reader_b():
    """Stub reader simulating LatinLibraryReader."""
    return _make_stub_reader(
        "LatinLibraryReader",
        ["cicero.txt", "caesar.txt"],
        texts_data=["Quo usque tandem", "Gallia est omnis divisa"],
    )


@pytest.fixture
def reader_empty():
    """Stub reader with no files."""
    return _make_stub_reader("EmptyReader", [])


# ===========================================================================
# Constructor tests
# ===========================================================================


class TestConstructor:
    """Test CombinedReader constructor styles."""

    def test_auto_prefix_from_class_name(self, reader_a):
        """Auto-prefix strips 'Reader' suffix and lowercases."""
        cr = CombinedReader(reader_a)
        assert "tesserae" in cr.readers

    def test_auto_prefix_strips_reader_suffix(self, reader_b):
        """LatinLibraryReader -> 'latinlibrary'."""
        cr = CombinedReader(reader_b)
        assert "latinlibrary" in cr.readers

    def test_explicit_prefix_via_tuple(self, reader_a):
        """Explicit prefix overrides auto-derivation."""
        cr = CombinedReader(("tess", reader_a))
        assert "tess" in cr.readers
        assert "tesserae" not in cr.readers

    def test_mixed_auto_and_explicit(self, reader_a, reader_b):
        """Can mix auto-prefix and explicit tuple styles."""
        cr = CombinedReader(reader_a, ("ll", reader_b))
        assert "tesserae" in cr.readers
        assert "ll" in cr.readers

    def test_prefixes_kwarg(self, reader_a, reader_b):
        """prefixes dict overrides auto-derivation for specific readers."""
        cr = CombinedReader(reader_a, reader_b, prefixes={reader_a: "tess"})
        assert "tess" in cr.readers
        assert "latinlibrary" in cr.readers

    def test_readers_property_returns_dict(self, reader_a, reader_b):
        """readers property returns dict mapping prefix -> reader."""
        cr = CombinedReader(reader_a, reader_b)
        readers = cr.readers
        assert isinstance(readers, dict)
        assert len(readers) == 2

    def test_readers_preserves_instances(self, reader_a):
        """readers property returns the original reader instances."""
        cr = CombinedReader(reader_a)
        assert cr.readers["tesserae"] is reader_a


# ===========================================================================
# fileids() tests
# ===========================================================================


class TestFileids:
    """Test namespaced file ID generation."""

    def test_fileids_are_namespaced(self, reader_a):
        """fileids() prepends prefix to each file ID."""
        cr = CombinedReader(reader_a)
        fids = cr.fileids()
        assert all(f.startswith("tesserae/") for f in fids)

    def test_fileids_combines_all_readers(self, reader_a, reader_b):
        """fileids() includes files from all readers."""
        cr = CombinedReader(reader_a, reader_b)
        fids = cr.fileids()
        assert len(fids) == 4
        assert "tesserae/vergil.aeneid.tess" in fids
        assert "latinlibrary/cicero.txt" in fids

    def test_fileids_match_filters(self, reader_a, reader_b):
        """fileids(match=...) filters the namespaced IDs by regex."""
        cr = CombinedReader(reader_a, reader_b)
        fids = cr.fileids(match="vergil")
        assert "tesserae/vergil.aeneid.tess" in fids
        assert "tesserae/ovid.met.tess" not in fids

    def test_fileids_empty_reader(self, reader_empty):
        """Empty reader contributes no fileids."""
        cr = CombinedReader(reader_empty)
        assert cr.fileids() == []

    def test_fileids_mixed_empty_and_nonempty(self, reader_a, reader_empty):
        """Empty readers don't break combination with non-empty readers."""
        cr = CombinedReader(reader_a, reader_empty)
        fids = cr.fileids()
        assert len(fids) == 2


# ===========================================================================
# _resolve_fileids() tests
# ===========================================================================


class TestResolveFileids:
    """Test file ID routing to correct readers."""

    def test_none_returns_all_readers(self, reader_a, reader_b):
        """None fileids means all readers get None."""
        cr = CombinedReader(reader_a, reader_b)
        resolved = cr._resolve_fileids(None)
        assert len(resolved) == 2
        assert all(fids is None for _, _, fids in resolved)

    def test_string_fileid_routes_to_correct_reader(self, reader_a, reader_b):
        """Single namespaced fileid routes to correct reader."""
        cr = CombinedReader(reader_a, reader_b)
        resolved = cr._resolve_fileids("tesserae/vergil.aeneid.tess")
        assert len(resolved) == 1
        prefix, reader, fids = resolved[0]
        assert prefix == "tesserae"
        assert reader is reader_a
        assert fids == ["vergil.aeneid.tess"]

    def test_list_fileids_groups_by_prefix(self, reader_a, reader_b):
        """List of fileids groups correctly by reader prefix."""
        cr = CombinedReader(reader_a, reader_b)
        resolved = cr._resolve_fileids([
            "tesserae/vergil.aeneid.tess",
            "latinlibrary/cicero.txt",
        ])
        assert len(resolved) == 2
        prefixes = {prefix for prefix, _, _ in resolved}
        assert prefixes == {"tesserae", "latinlibrary"}

    def test_unknown_prefix_excluded(self, reader_a):
        """Fileids with unknown prefix are silently excluded."""
        cr = CombinedReader(reader_a)
        resolved = cr._resolve_fileids("unknown/file.txt")
        assert len(resolved) == 0


# ===========================================================================
# Core iteration methods
# ===========================================================================


class TestCoreIteration:
    """Test that iteration methods chain across readers."""

    def test_texts_chains_all_readers(self, reader_a, reader_b):
        """texts() yields from all readers."""
        cr = CombinedReader(reader_a, reader_b)
        texts = list(cr.texts())
        assert len(texts) == 4
        assert "Arma virumque cano" in texts
        assert "Gallia est omnis divisa" in texts

    def test_texts_with_fileids_routes_to_correct_reader(self, reader_a, reader_b):
        """texts(fileids=...) only queries targeted reader."""
        cr = CombinedReader(reader_a, reader_b)
        # Consume the iterator to trigger the call
        list(cr.texts(fileids="tesserae/vergil.aeneid.tess"))
        reader_a.texts.assert_called_with(fileids=["vergil.aeneid.tess"])
        reader_b.texts.assert_not_called()

    def test_docs_chains(self, reader_a, reader_b):
        """docs() chains across readers."""
        cr = CombinedReader(reader_a, reader_b)
        # Consume to trigger lazy evaluation
        list(cr.docs())
        reader_a.docs.assert_called_once_with(fileids=None)
        reader_b.docs.assert_called_once_with(fileids=None)

    def test_sents_passes_as_text(self, reader_a):
        """sents(as_text=True) forwards kwarg."""
        cr = CombinedReader(reader_a)
        list(cr.sents(as_text=True))
        reader_a.sents.assert_called_once_with(fileids=None, as_text=True)

    def test_tokens_passes_as_text(self, reader_a):
        """tokens(as_text=True) forwards kwarg."""
        cr = CombinedReader(reader_a)
        list(cr.tokens(as_text=True))
        reader_a.tokens.assert_called_once_with(fileids=None, as_text=True)

    def test_metadata_namespaces_fileids(self, reader_a):
        """metadata() prefixes fileids in output."""
        cr = CombinedReader(reader_a)
        meta_list = list(cr.metadata())
        assert len(meta_list) == 2
        assert all(fid.startswith("tesserae/") for fid, _ in meta_list)

    def test_returns_iterators(self, reader_a):
        """All iteration methods return iterators, not lists."""
        cr = CombinedReader(reader_a)
        assert isinstance(cr.texts(), Iterator)
        assert isinstance(cr.docs(), Iterator)
        assert isinstance(cr.sents(), Iterator)
        assert isinstance(cr.tokens(), Iterator)
        assert isinstance(cr.metadata(), Iterator)


# ===========================================================================
# Search & analysis methods
# ===========================================================================


class TestSearchMethods:
    """Test search and analysis method delegation."""

    def test_find_sents_chains(self, reader_a, reader_b):
        """find_sents() chains results from all readers."""
        reader_a.find_sents.side_effect = lambda fileids=None, **kw: iter(
            [{"sentence": "s1"}]
        )
        reader_b.find_sents.side_effect = lambda fileids=None, **kw: iter(
            [{"sentence": "s2"}]
        )
        cr = CombinedReader(reader_a, reader_b)
        results = list(cr.find_sents(pattern=r"test"))
        assert len(results) == 2

    def test_find_sents_passes_kwargs(self, reader_a):
        """find_sents() forwards all kwargs."""
        cr = CombinedReader(reader_a)
        list(cr.find_sents(pattern=r"\btest\b", ignore_case=False, context=True))
        reader_a.find_sents.assert_called_once_with(
            fileids=None, pattern=r"\btest\b", ignore_case=False, context=True
        )

    def test_kwic_chains(self, reader_a, reader_b):
        """kwic() chains results from all readers."""
        reader_a.kwic.side_effect = lambda kw, fileids=None, **kwargs: iter(
            [{"match": "amor"}]
        )
        reader_b.kwic.side_effect = lambda kw, fileids=None, **kwargs: iter(
            [{"match": "amor"}]
        )
        cr = CombinedReader(reader_a, reader_b)
        results = list(cr.kwic("amor"))
        assert len(results) == 2

    def test_kwic_passes_kwargs(self, reader_a):
        """kwic() forwards all kwargs."""
        cr = CombinedReader(reader_a)
        list(cr.kwic("amor", window=3, by_lemma=True))
        reader_a.kwic.assert_called_once_with(
            "amor", fileids=None, window=3, by_lemma=True
        )

    def test_ngrams_chains(self, reader_a, reader_b):
        """ngrams() chains across readers."""
        reader_a.ngrams.side_effect = lambda n=2, fileids=None, **kw: iter(
            ["arma virum"]
        )
        reader_b.ngrams.side_effect = lambda n=2, fileids=None, **kw: iter(
            ["gallia est"]
        )
        cr = CombinedReader(reader_a, reader_b)
        results = list(cr.ngrams(n=2))
        assert results == ["arma virum", "gallia est"]

    def test_skipgrams_chains(self, reader_a, reader_b):
        """skipgrams() chains across readers."""
        reader_a.skipgrams.side_effect = lambda n=2, k=1, fileids=None, **kw: iter(
            ["arma cano"]
        )
        reader_b.skipgrams.side_effect = lambda n=2, k=1, fileids=None, **kw: iter(
            ["gallia divisa"]
        )
        cr = CombinedReader(reader_a, reader_b)
        results = list(cr.skipgrams(n=2, k=1))
        assert results == ["arma cano", "gallia divisa"]

    def test_concordance_merges_dicts(self, reader_a, reader_b):
        """concordance() merges dicts, concatenating citation lists."""
        reader_a.concordance.side_effect = lambda fileids=None, **kw: {
            "amor": ["verg.1.1", "verg.1.5"],
            "bellum": ["verg.1.2"],
        }
        reader_b.concordance.side_effect = lambda fileids=None, **kw: {
            "amor": ["cic.1.3"],
            "pax": ["cic.2.1"],
        }
        cr = CombinedReader(reader_a, reader_b)
        conc = cr.concordance()

        assert conc["amor"] == ["verg.1.1", "verg.1.5", "cic.1.3"]
        assert conc["bellum"] == ["verg.1.2"]
        assert conc["pax"] == ["cic.2.1"]

    def test_concordance_passes_kwargs(self, reader_a):
        """concordance() forwards kwargs to each reader."""
        cr = CombinedReader(reader_a)
        cr.concordance(basis="text", only_alpha=False)
        reader_a.concordance.assert_called_once_with(
            fileids=None, basis="text", only_alpha=False
        )


# ===========================================================================
# combine() convenience function
# ===========================================================================


class TestCombineFunction:
    """Test the combine() convenience function."""

    def test_returns_combined_reader(self, reader_a, reader_b):
        """combine() returns a CombinedReader instance."""
        cr = combine(reader_a, reader_b)
        assert isinstance(cr, CombinedReader)

    def test_passes_kwargs(self, reader_a):
        """combine() passes kwargs to CombinedReader."""
        cr = combine(reader_a, prefixes={reader_a: "tess"})
        assert "tess" in cr.readers


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_reader(self, reader_a):
        """CombinedReader works with a single reader."""
        cr = CombinedReader(reader_a)
        fids = cr.fileids()
        assert len(fids) == 2

    def test_empty_reader(self, reader_empty):
        """CombinedReader works with an empty reader."""
        cr = CombinedReader(reader_empty)
        assert cr.fileids() == []
        assert list(cr.texts()) == []

    def test_no_readers(self):
        """CombinedReader works with zero readers."""
        cr = CombinedReader()
        assert cr.fileids() == []

    def test_repr(self, reader_a, reader_b):
        """__repr__ shows useful information."""
        cr = CombinedReader(reader_a, reader_b)
        r = repr(cr)
        assert "CombinedReader" in r
        assert "tesserae" in r
        assert "latinlibrary" in r

    def test_len(self, reader_a, reader_b):
        """len() returns total number of fileids."""
        cr = CombinedReader(reader_a, reader_b)
        assert len(cr) == 4
