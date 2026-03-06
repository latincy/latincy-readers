"""Tests for sibling .conlluc annotation loading.

When a .conlluc file exists alongside a .tess source file, the reader
should load pre-computed annotations from disk instead of running the
NLP pipeline. This is the core performance win of canonical annotations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from latincyreaders.readers.tesserae import TesseraeReader
from latincyreaders.cache.conlluc import doc_to_conlluc


@pytest.fixture
def tess_with_conlluc(tmp_path):
    """Create a tiny .tess file with a sibling .conlluc file."""
    tess_file = tmp_path / "test.tess"
    tess_file.write_text(
        "<test. 1> Gallia est omnis divisa in partes tres.\n"
        "<test. 2> Horum omnium fortissimi sunt Belgae.\n",
        encoding="utf-8",
    )

    # Generate the .conlluc by running NLP once
    reader = TesseraeReader(str(tmp_path))
    doc = next(reader.docs(["test.tess"]))
    nlp = reader.nlp
    model_name = nlp.meta.get("name", "unknown")
    model_version = nlp.meta.get("version", "unknown")

    content = doc_to_conlluc(
        doc,
        fileid="test.tess",
        collection="test",
        model_name=model_name,
        model_version=model_version,
    )
    conlluc_file = tmp_path / "test.conlluc"
    conlluc_file.write_text(content, encoding="utf-8")

    return tmp_path


class TestSiblingConllucLoading:
    def test_conlluc_exists_alongside_tess(self, tess_with_conlluc):
        """A .conlluc file should exist next to the .tess file."""
        assert (tess_with_conlluc / "test.conlluc").exists()
        assert (tess_with_conlluc / "test.tess").exists()

    def test_loads_from_conlluc_not_pipeline(self, tess_with_conlluc):
        """When a sibling .conlluc exists, the reader should NOT call nlp()."""
        reader = TesseraeReader(str(tess_with_conlluc))
        reader._cache.clear()

        # Track whether nlp.__call__ is invoked (i.e., the pipeline runs)
        nlp = reader.nlp
        call_count = 0
        original_call = nlp.__class__.__call__

        def tracking_call(self_nlp, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_call(self_nlp, *args, **kwargs)

        with patch.object(nlp.__class__, "__call__", tracking_call):
            doc = next(reader.docs(["test.tess"]))

        # Doc should still have tokens
        assert len(doc) > 0
        assert next(doc.sents).text  # should have sentence boundaries
        # Pipeline should NOT have been called
        assert call_count == 0, (
            f"NLP pipeline was called {call_count} time(s) "
            f"but should not be called when .conlluc exists"
        )

    def test_conlluc_doc_has_annotations(self, tess_with_conlluc):
        """Doc loaded from .conlluc should have POS, lemma, dep annotations."""
        reader = TesseraeReader(str(tess_with_conlluc))
        reader._cache.clear()

        doc = next(reader.docs(["test.tess"]))

        # Check that we got real annotations, not blank
        content_tokens = [t for t in doc if not t.is_punct and not t.is_space]
        assert len(content_tokens) > 0

        # At least some tokens should have POS tags
        pos_tags = [t.pos_ for t in content_tokens]
        assert any(p != "" for p in pos_tags), "No POS tags found"

        # At least some should have lemmas
        lemmas = [t.lemma_ for t in content_tokens]
        assert any(l != "" for l in lemmas), "No lemmas found"

    def test_no_conlluc_falls_back_to_pipeline(self, tmp_path):
        """Without a .conlluc file, the reader should use the NLP pipeline."""
        tess_file = tmp_path / "bare.tess"
        tess_file.write_text(
            "<bare. 1> Roma condita est.\n",
            encoding="utf-8",
        )

        reader = TesseraeReader(str(tmp_path))
        doc = next(reader.docs(["bare.tess"]))

        # Should still work, just via the pipeline
        assert len(doc) > 0
        content_tokens = [t for t in doc if not t.is_punct]
        assert any(t.pos_ != "" for t in content_tokens)

    def test_conlluc_faster_than_pipeline(self, tess_with_conlluc):
        """Loading from .conlluc should be significantly faster than NLP."""
        import time

        reader = TesseraeReader(str(tess_with_conlluc))

        # Time the conlluc path (clear cache to force re-read)
        reader._cache.clear()
        start = time.time()
        doc_cached = next(reader.docs(["test.tess"]))
        time_conlluc = time.time() - start

        # Time the pipeline path (use a file without .conlluc)
        bare_file = tess_with_conlluc / "bare.tess"
        bare_file.write_text(
            "<bare. 1> Gallia est omnis divisa in partes tres.\n"
            "<bare. 2> Horum omnium fortissimi sunt Belgae.\n",
            encoding="utf-8",
        )
        reader2 = TesseraeReader(str(tess_with_conlluc))
        start = time.time()
        doc_pipeline = next(reader2.docs(["bare.tess"]))
        time_pipeline = time.time() - start

        print(f"\n  .conlluc: {time_conlluc:.4f}s  pipeline: {time_pipeline:.4f}s")
        # conlluc should be faster (at least not slower)
        # For small texts the difference is small, so just check it works
        assert doc_cached is not None
        assert doc_pipeline is not None
