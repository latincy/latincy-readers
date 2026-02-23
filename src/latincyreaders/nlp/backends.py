"""NLP backend abstraction for latincy-readers.

Provides a protocol for swappable NLP backends. Currently only SpaCyBackend
is implemented; Stanza and Flair are planned for future releases.

All backends produce spaCy Doc objects, keeping the reader's public API
(docs(), sents(), tokens()) unchanged regardless of backend.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol, runtime_checkable, TYPE_CHECKING

from latincyreaders.nlp.pipeline import AnnotationLevel, get_nlp

if TYPE_CHECKING:
    from spacy import Language
    from spacy.tokens import Doc
    from spacy.vocab import Vocab


@runtime_checkable
class NLPBackend(Protocol):
    """Protocol for NLP processing backends.

    All backends must produce spaCy Doc objects regardless of the
    underlying NLP engine. This keeps the reader's public API
    (docs(), sents(), tokens()) unchanged.
    """

    def process(self, text: str) -> "Doc":
        """Process text and return a spaCy Doc."""
        ...

    def process_batch(self, texts: Iterable[str]) -> Iterator["Doc"]:
        """Process multiple texts efficiently."""
        ...

    @property
    def nlp(self) -> "Language | None":
        """The underlying spaCy Language pipeline, if any."""
        ...

    @property
    def vocab(self) -> "Vocab":
        """The spaCy Vocab used by this backend."""
        ...


class SpaCyBackend:
    """SpaCy-based NLP backend. The default and only production backend.

    Wraps the existing pipeline logic from latincyreaders.nlp.pipeline,
    providing lazy loading and configurable annotation levels.

    Args:
        model_name: Name of the spaCy model to load for BASIC/FULL levels.
        lang: Language code for blank model in TOKENIZE level.
        annotation_level: How much NLP annotation to apply.

    Example:
        >>> backend = SpaCyBackend(model_name="la_core_web_sm")
        >>> doc = backend.process("Arma virumque cano.")
        >>> print([(t.text, t.lemma_) for t in doc])
    """

    def __init__(
        self,
        model_name: str = "la_core_web_lg",
        lang: str = "la",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
    ):
        self._nlp: Language | None = None
        self._model_name = model_name
        self._lang = lang
        self._annotation_level = annotation_level

    @property
    def nlp(self) -> "Language | None":
        """Lazy-load the spaCy pipeline."""
        if self._nlp is None and self._annotation_level != AnnotationLevel.NONE:
            self._nlp = get_nlp(
                self._annotation_level,
                model_name=self._model_name,
                lang=self._lang,
            )
        return self._nlp

    def process(self, text: str) -> "Doc":
        """Process text and return a spaCy Doc.

        Args:
            text: Input text to process.

        Returns:
            Annotated spaCy Doc.

        Raises:
            ValueError: If annotation_level is NONE.
        """
        nlp = self.nlp
        if nlp is None:
            raise ValueError(
                "Cannot process text with annotation_level=NONE. "
                "Use a higher annotation level."
            )
        return nlp(text)

    def process_batch(self, texts: Iterable[str]) -> Iterator["Doc"]:
        """Process multiple texts efficiently using spaCy's pipe().

        Args:
            texts: Iterable of input texts.

        Yields:
            Annotated spaCy Doc objects.

        Raises:
            ValueError: If annotation_level is NONE.
        """
        nlp = self.nlp
        if nlp is None:
            raise ValueError(
                "Cannot process text with annotation_level=NONE. "
                "Use a higher annotation level."
            )
        yield from nlp.pipe(texts)

    @property
    def vocab(self) -> "Vocab":
        """The spaCy Vocab used by this backend.

        Returns:
            spaCy Vocab object.

        Raises:
            ValueError: If annotation_level is NONE.
        """
        nlp = self.nlp
        if nlp is None:
            raise ValueError(
                "Cannot access vocab with annotation_level=NONE. "
                "Use a higher annotation level."
            )
        return nlp.vocab


class StanzaBackend:
    """Stanza NLP backend (not yet implemented).

    Will convert Stanza output to spaCy Doc objects using
    spacy-stanza or manual conversion.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "StanzaBackend is planned for a future release. "
            "Use SpaCyBackend (the default) for now."
        )


class FlairBackend:
    """Flair NLP backend (not yet implemented).

    Will convert Flair output to spaCy Doc objects using
    manual conversion.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "FlairBackend is planned for a future release. "
            "Use SpaCyBackend (the default) for now."
        )
