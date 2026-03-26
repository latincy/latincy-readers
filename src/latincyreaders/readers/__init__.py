"""Corpus reader implementations."""

from latincyreaders.readers.tesserae import TesseraeReader
from latincyreaders.readers.greek_tesserae import GreekTesseraeReader
from latincyreaders.readers.plaintext import PlaintextReader, LatinLibraryReader
from latincyreaders.readers.tei import TEIReader, PerseusReader
from latincyreaders.readers.digilibt import DigilibLTReader
from latincyreaders.readers.wikisource import WikiSourceReader
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
    "GreekTesseraeReader",
    "PlaintextReader",
    "LatinLibraryReader",
    "TEIReader",
    "PerseusReader",
    "DigilibLTReader",
    "WikiSourceReader",
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
