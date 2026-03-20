"""NLP pipeline management for latincy-readers.

This module handles lazy loading and caching of spaCy models, as well as
registration of custom extensions for citation tracking.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

import spacy
from spacy.tokens import Doc, Span, Token

if TYPE_CHECKING:
    from spacy import Language


class AnnotationLevel(Enum):
    """Control how much NLP processing to apply.

    Use lower levels for better performance when you don't need full annotations.

    Levels:
        NONE: Raw text only, no spaCy processing
        TOKENIZE: Tokenization + sentence splitting only
        BASIC: + lemmatization, POS tagging (default)
        FULL: + NER, dependency parsing
    """

    NONE = "none"
    TOKENIZE = "tokenize"
    BASIC = "basic"
    FULL = "full"


# Components that are always kept when using enable/disable.
# tok2vec (or transformer) is the feature backbone; senter provides
# sentence boundaries used throughout the reader API.
BACKBONE_COMPONENTS = {"tok2vec", "transformer", "senter"}


# Register custom extensions once at module load
def _register_extensions() -> None:
    """Register spaCy custom extensions for citation tracking."""
    if not Doc.has_extension("metadata"):
        Doc.set_extension("metadata", default=None)
        Doc.set_extension("fileid", default=None)

    if not Span.has_extension("citation"):
        Span.set_extension("citation", default=None)
        Span.set_extension("metadata", default=None)

    if not Token.has_extension("citation"):
        Token.set_extension("citation", default=None)

    # UD treebank annotations (full CoNLL-U token data)
    if not Token.has_extension("ud"):
        Token.set_extension("ud", default=None)


# Register on import
_register_extensions()


@lru_cache(maxsize=4)
def load_model(model_name: str = "la_core_web_lg") -> Language:
    """Load and cache a spaCy model.

    Args:
        model_name: Name of the spaCy model to load.

    Returns:
        Loaded spaCy Language pipeline.
    """
    nlp = spacy.load(model_name)
    nlp.max_length = 2_500_000  # Handle large documents
    return nlp


def _resolve_backbone(pipe_names: list[str]) -> set[str]:
    """Return backbone component names present in a pipeline."""
    return {name for name in pipe_names if name in BACKBONE_COMPONENTS}


def create_pipeline(
    level: AnnotationLevel = AnnotationLevel.BASIC,
    model_name: str = "la_core_web_lg",
    lang: str = "la",
    enable: list[str] | None = None,
    disable: list[str] | None = None,
) -> Language | None:
    """Create a spaCy pipeline based on annotation level.

    Fine-grained control is available via ``enable`` and ``disable``.
    When either is provided, backbone components (tok2vec/transformer
    and senter) are always kept.

    * **enable** (additive): only the listed components plus backbone
      are active.  Everything else is disabled.
    * **disable** (subtractive): the listed components are disabled.
      Backbone components cannot be disabled and are silently kept.

    ``enable`` and ``disable`` are mutually exclusive; passing both
    raises :class:`ValueError`.

    Args:
        level: Desired annotation level (ignored when *enable* or
            *disable* is provided, except for NONE/TOKENIZE).
        model_name: Name of the spaCy model to load for BASIC/FULL levels.
        lang: Language code for blank model in TOKENIZE level.
        enable: Component names to enable (additive mode).
        disable: Component names to disable (subtractive mode).

    Returns:
        Configured spaCy pipeline, or None for NONE level.
    """
    if enable is not None and disable is not None:
        raise ValueError(
            "Cannot specify both 'enable' and 'disable'. "
            "Use 'enable' to list only the components you want (additive), "
            "or 'disable' to remove specific components (subtractive)."
        )

    if level == AnnotationLevel.NONE and enable is None and disable is None:
        return None

    if level == AnnotationLevel.TOKENIZE and enable is None and disable is None:
        # Minimal pipeline: just tokenization and sentence splitting
        nlp = spacy.blank(lang)
        if lang == "grc":
            # Greek uses ; as question mark and · (ano teleia) as a pause
            nlp.add_pipe("sentencizer", config={"punct_chars": [".", ";", "·", ":"]})
        else:
            nlp.add_pipe("sentencizer")
        nlp.max_length = 2_500_000
        return nlp

    # --- Custom enable/disable or AnnotationLevel-based ---

    if enable is not None:
        # Additive: load model, disable everything except backbone + requested
        nlp = spacy.load(model_name)
        backbone = _resolve_backbone(nlp.pipe_names)
        keep = backbone | set(enable)
        to_disable = [p for p in nlp.pipe_names if p not in keep]
        if to_disable:
            nlp.select_pipes(disable=to_disable)

    elif disable is not None:
        # Subtractive: load model, disable requested (protect backbone)
        nlp = spacy.load(model_name)
        backbone = _resolve_backbone(nlp.pipe_names)
        to_disable = [p for p in disable if p not in backbone]
        if to_disable:
            nlp.select_pipes(disable=to_disable)

    elif level == AnnotationLevel.BASIC:
        # Disable heavy components we don't need
        nlp = spacy.load(model_name, disable=["ner", "parser"])
        # If model has no sentence boundary component, add sentencizer
        has_sbd = any(
            "token.is_sent_start" in nlp.get_pipe_meta(name).assigns
            for name in nlp.pipe_names
        )
        if not has_sbd:
            nlp.add_pipe("sentencizer", first=True)
    else:
        # FULL: everything enabled
        nlp = spacy.load(model_name)

    nlp.max_length = 2_500_000
    return nlp


def get_nlp(
    level: AnnotationLevel = AnnotationLevel.BASIC,
    model_name: str = "la_core_web_lg",
    lang: str = "la",
    enable: list[str] | None = None,
    disable: list[str] | None = None,
) -> Language | None:
    """Get a spaCy pipeline for the given annotation level.

    This is the main entry point for getting an NLP pipeline.

    When ``enable`` or ``disable`` is provided, the pipeline is
    customized accordingly (see :func:`create_pipeline` for details).
    Backbone components (tok2vec/transformer and senter) are always
    preserved.

    Args:
        level: Desired annotation level.
        model_name: Name of the spaCy model to load for BASIC/FULL levels.
        lang: Language code for blank model in TOKENIZE level.
        enable: Component names to enable (additive). Backbone components
            are always included automatically.
        disable: Component names to disable (subtractive). Backbone
            components cannot be disabled.

    Returns:
        Configured spaCy pipeline, or None for NONE level.

    Examples:
        >>> nlp = get_nlp(AnnotationLevel.BASIC)
        >>> doc = nlp("Arma virumque cano.")
        >>> print([(t.text, t.lemma_) for t in doc])

        >>> # Only morphology — fastest for form counting
        >>> nlp = get_nlp(enable=["tagger", "morphologizer"])
        >>> nlp.pipe_names
        ['senter', 'tok2vec', 'tagger', 'morphologizer']

        >>> # Everything except NER
        >>> nlp = get_nlp(disable=["ner"])
    """
    return create_pipeline(
        level,
        model_name=model_name,
        lang=lang,
        enable=enable,
        disable=disable,
    )
