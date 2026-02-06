"""
latincy-readers: Corpus readers for Latin texts with LatinCy/spaCy integration.

This package provides readers for various Latin text corpora with spaCy integration:

Readers:
    - TesseraeReader: Tesserae citation format (.tess files)
    - PlaintextReader: Generic plaintext files
    - LatinLibraryReader: The Latin Library corpus
    - TEIReader: TEI/XML format base class
    - PerseusReader: Perseus Digital Library
    - CamenaReader: CAMENA Neo-Latin corpus
    - TxtdownReader: TXT-down format (Tesserae derivative)
    - UDReader: Universal Dependencies CoNLL-U format
    - LatinUDReader: All 6 Latin UD treebanks with auto-download
    - PROIELReader: Latin PROIEL treebank
    - PerseusUDReader: Latin Perseus UD treebank
    - ITTBReader: Index Thomisticus Treebank
    - LLCTReader: Late Latin Charter Treebank
    - UDanteReader: Dante's Latin works treebank
    - CIRCSEReader: CIRCSE Latin treebank

Core:
    - AnnotationLevel: Control NLP processing overhead
    - FileSelector: Fluent API for file filtering
    - MetadataManager: Schema-validated metadata handling

Example:
    >>> from latincyreaders import TesseraeReader, AnnotationLevel
    >>>
    >>> # Auto-download corpus on first use
    >>> reader = TesseraeReader()
    >>>
    >>> # Control annotation level
    >>> reader = TesseraeReader(annotation_level=AnnotationLevel.TOKENIZE)
    >>>
    >>> # Use FileSelector for complex queries
    >>> selection = reader.select().where(author="Vergil").date_range(-50, 50)
    >>> for doc in reader.docs(selection):
    ...     print(doc._.fileid)
"""

from latincyreaders.core.base import AnnotationLevel
from latincyreaders.core.selector import FileSelector
from latincyreaders.readers.tesserae import TesseraeReader
from latincyreaders.readers.plaintext import PlaintextReader, LatinLibraryReader
from latincyreaders.readers.tei import TEIReader, PerseusReader
from latincyreaders.readers.camena import CamenaReader
from latincyreaders.readers.txtdown import TxtdownReader
from latincyreaders.readers.ud import (
    UDReader,
    PROIELReader,
    PerseusUDReader,
    ITTBReader,
    LLCTReader,
    UDanteReader,
    CIRCSEReader,
    LatinUDReader,
)
from latincyreaders.utils.metadata import MetadataManager

__version__ = "1.1.0"
__all__ = [
    # Readers
    "TesseraeReader",
    "PlaintextReader",
    "LatinLibraryReader",
    "TEIReader",
    "PerseusReader",
    "CamenaReader",
    "TxtdownReader",
    # Universal Dependencies readers
    "UDReader",
    "PROIELReader",
    "PerseusUDReader",
    "ITTBReader",
    "LLCTReader",
    "UDanteReader",
    "CIRCSEReader",
    "LatinUDReader",
    # Core
    "AnnotationLevel",
    "FileSelector",
    "MetadataManager",
]
