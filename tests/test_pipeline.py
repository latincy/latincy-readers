"""Tests for NLP pipeline module."""

import pytest

from latincyreaders.nlp.pipeline import (
    AnnotationLevel,
    BACKBONE_COMPONENTS,
    create_pipeline,
    get_nlp,
    load_model,
)


class TestAnnotationLevel:
    """Tests for AnnotationLevel enum."""

    def test_none_level(self):
        """NONE level creates no pipeline."""
        nlp = create_pipeline(AnnotationLevel.NONE)
        assert nlp is None

    def test_tokenize_level(self):
        """TOKENIZE level creates minimal pipeline."""
        nlp = create_pipeline(AnnotationLevel.TOKENIZE)
        assert nlp is not None
        assert "sentencizer" in nlp.pipe_names

    def test_basic_level(self):
        """BASIC level loads model without NER/parser."""
        nlp = create_pipeline(AnnotationLevel.BASIC)
        assert nlp is not None
        # NER and parser should be disabled
        assert "ner" not in nlp.pipe_names

    def test_full_level(self):
        """FULL level loads model with all components."""
        nlp = create_pipeline(AnnotationLevel.FULL)
        assert nlp is not None
        # Full model has all components
        assert nlp.max_length == 2_500_000


class TestGetNlp:
    """Tests for get_nlp function."""

    def test_get_nlp_basic(self):
        """get_nlp returns pipeline for BASIC level."""
        nlp = get_nlp(AnnotationLevel.BASIC)
        assert nlp is not None

    def test_get_nlp_none(self):
        """get_nlp returns None for NONE level."""
        nlp = get_nlp(AnnotationLevel.NONE)
        assert nlp is None


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_default(self):
        """load_model loads default Latin model."""
        nlp = load_model()
        assert nlp is not None
        assert nlp.max_length == 2_500_000

    def test_load_model_cached(self):
        """load_model returns cached model on repeat calls."""
        nlp1 = load_model()
        nlp2 = load_model()
        assert nlp1 is nlp2  # Same object from cache


class TestEnableDisable:
    """Tests for enable/disable pipeline customization."""

    def test_enable_additive(self):
        """enable keeps only backbone + requested components."""
        nlp = create_pipeline(enable=["tagger", "morphologizer"])
        assert nlp is not None
        active = set(nlp.pipe_names)
        # Requested components present
        assert "tagger" in active
        assert "morphologizer" in active
        # Backbone preserved
        assert "tok2vec" in active or "transformer" in active
        assert "senter" in active
        # Other components not present
        assert "ner" not in active
        assert "parser" not in active
        assert "trainable_lemmatizer" not in active

    def test_disable_subtractive(self):
        """disable removes requested components but keeps backbone."""
        nlp = create_pipeline(disable=["ner", "parser", "trainable_lemmatizer"])
        assert nlp is not None
        active = set(nlp.pipe_names)
        assert "ner" not in active
        assert "parser" not in active
        assert "trainable_lemmatizer" not in active
        # Backbone preserved
        assert "tok2vec" in active or "transformer" in active
        assert "senter" in active
        # Other components still active
        assert "tagger" in active
        assert "morphologizer" in active

    def test_disable_backbone_protected(self):
        """Backbone components are silently kept even if in disable list."""
        nlp = create_pipeline(disable=["tok2vec", "senter", "ner"])
        assert nlp is not None
        active = set(nlp.pipe_names)
        # tok2vec and senter should survive
        assert "tok2vec" in active or "transformer" in active
        assert "senter" in active
        # ner should be disabled
        assert "ner" not in active

    def test_enable_and_disable_raises(self):
        """Providing both enable and disable raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            create_pipeline(enable=["tagger"], disable=["ner"])

    def test_get_nlp_enable(self):
        """get_nlp forwards enable to create_pipeline."""
        nlp = get_nlp(enable=["morphologizer"])
        assert nlp is not None
        assert "morphologizer" in nlp.pipe_names
        assert "ner" not in nlp.pipe_names

    def test_get_nlp_disable(self):
        """get_nlp forwards disable to create_pipeline."""
        nlp = get_nlp(disable=["ner", "parser"])
        assert nlp is not None
        assert "ner" not in nlp.pipe_names
        assert "parser" not in nlp.pipe_names
        assert "tagger" in nlp.pipe_names

    def test_enable_overrides_annotation_level(self):
        """enable takes precedence over annotation_level."""
        # BASIC would normally keep lemmatizers; enable should override
        nlp = create_pipeline(
            level=AnnotationLevel.BASIC,
            enable=["tagger", "morphologizer"],
        )
        active = set(nlp.pipe_names)
        assert "tagger" in active
        assert "morphologizer" in active
        assert "trainable_lemmatizer" not in active
