"""Corpus reader implementations."""

from latincyreaders.readers.tesserae import TesseraeReader
from latincyreaders.readers.plaintext import PlaintextReader, LatinLibraryReader
from latincyreaders.readers.tei import TEIReader, PerseusReader
from latincyreaders.readers.ud import (
    UDReader,
    PROIELReader,
    PerseusUDReader,
    ITTBReader,
    LLCTReader,
    UDanteReader,
    CIRCSEReader,
    LatinUDReader,
    LATIN_TREEBANKS,
)

__all__ = [
    "TesseraeReader",
    "PlaintextReader",
    "LatinLibraryReader",
    "TEIReader",
    "PerseusReader",
    # Universal Dependencies readers
    "UDReader",
    "PROIELReader",
    "PerseusUDReader",
    "ITTBReader",
    "LLCTReader",
    "UDanteReader",
    "CIRCSEReader",
    "LatinUDReader",
    "LATIN_TREEBANKS",
]
