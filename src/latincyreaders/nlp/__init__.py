"""NLP pipeline integration."""

from latincyreaders.nlp.pipeline import get_nlp, load_model
from latincyreaders.nlp.backends import (
    NLPBackend,
    SpaCyBackend,
    StanzaBackend,
    FlairBackend,
)

__all__ = [
    "get_nlp",
    "load_model",
    "NLPBackend",
    "SpaCyBackend",
    "StanzaBackend",
    "FlairBackend",
]
