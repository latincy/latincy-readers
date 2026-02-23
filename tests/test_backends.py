"""Tests for NLP backend abstraction."""

import pytest

from latincyreaders.nlp.backends import (
    NLPBackend,
    SpaCyBackend,
    StanzaBackend,
    FlairBackend,
)
from latincyreaders.nlp.pipeline import AnnotationLevel


class TestSpaCyBackend:
    """Test the SpaCyBackend implementation."""

    @pytest.fixture
    def backend(self):
        """SpaCyBackend with TOKENIZE for fast tests."""
        return SpaCyBackend(annotation_level=AnnotationLevel.TOKENIZE)

    def test_lazy_loading(self):
        """NLP pipeline is None until first use."""
        backend = SpaCyBackend(annotation_level=AnnotationLevel.TOKENIZE)
        assert backend._nlp is None

    def test_process_returns_doc(self, backend):
        """process() returns a spaCy Doc."""
        from spacy.tokens import Doc

        doc = backend.process("Arma virumque cano.")
        assert isinstance(doc, Doc)
        assert len(doc) > 0

    def test_process_loads_nlp(self, backend):
        """process() triggers lazy loading."""
        assert backend._nlp is None
        backend.process("Arma virumque cano.")
        assert backend._nlp is not None

    def test_process_batch(self, backend):
        """process_batch() yields Docs for multiple texts."""
        from spacy.tokens import Doc

        texts = ["Arma virumque cano.", "Gallia est omnis divisa."]
        docs = list(backend.process_batch(texts))
        assert len(docs) == 2
        assert all(isinstance(d, Doc) for d in docs)

    def test_vocab_property(self, backend):
        """vocab property returns a spaCy Vocab."""
        from spacy.vocab import Vocab

        # Force pipeline load
        backend.process("test")
        assert isinstance(backend.vocab, Vocab)

    def test_nlp_property(self, backend):
        """nlp property returns spaCy Language after loading."""
        backend.process("test")
        assert backend.nlp is not None

    def test_annotation_level_none_blocks_process(self):
        """NONE annotation level prevents process()."""
        backend = SpaCyBackend(annotation_level=AnnotationLevel.NONE)
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            backend.process("test")

    def test_annotation_level_none_blocks_batch(self):
        """NONE annotation level prevents process_batch()."""
        backend = SpaCyBackend(annotation_level=AnnotationLevel.NONE)
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            list(backend.process_batch(["test"]))

    def test_annotation_level_none_blocks_vocab(self):
        """NONE annotation level prevents vocab access."""
        backend = SpaCyBackend(annotation_level=AnnotationLevel.NONE)
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            _ = backend.vocab


class TestStubBackends:
    """Test that stub backends raise NotImplementedError."""

    def test_stanza_backend_raises(self):
        """StanzaBackend raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="planned for a future release"):
            StanzaBackend()

    def test_flair_backend_raises(self):
        """FlairBackend raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="planned for a future release"):
            FlairBackend()


class TestProtocol:
    """Test the NLPBackend protocol."""

    def test_spacy_backend_is_nlp_backend(self):
        """SpaCyBackend satisfies the NLPBackend protocol."""
        backend = SpaCyBackend(annotation_level=AnnotationLevel.TOKENIZE)
        assert isinstance(backend, NLPBackend)


class TestBaseReaderWithBackend:
    """Test BaseCorpusReader integration with backend parameter."""

    def test_reader_with_backend(self, tesserae_dir):
        """Reader works when passed a backend explicitly."""
        from spacy.tokens import Doc
        from latincyreaders import TesseraeReader

        backend = SpaCyBackend(annotation_level=AnnotationLevel.TOKENIZE)
        reader = TesseraeReader(root=tesserae_dir, fileids="*.tess", backend=backend)
        docs = list(reader.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    def test_reader_without_backend(self, tesserae_dir):
        """Reader works normally without backend (backward compat)."""
        from spacy.tokens import Doc
        from latincyreaders import TesseraeReader

        reader = TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.TOKENIZE,
        )
        docs = list(reader.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    def test_backend_nlp_property_delegates(self, tesserae_dir):
        """Reader's nlp property delegates to backend when provided."""
        from latincyreaders import TesseraeReader

        backend = SpaCyBackend(annotation_level=AnnotationLevel.TOKENIZE)
        reader = TesseraeReader(root=tesserae_dir, fileids="*.tess", backend=backend)
        # Force load
        _ = reader.nlp
        assert reader.nlp is backend.nlp
