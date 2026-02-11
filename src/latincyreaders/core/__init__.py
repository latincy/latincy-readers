"""Core abstractions for latincy-readers."""

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel
from latincyreaders.core.combined import CombinedReader, combine
from latincyreaders.core.download import DownloadableCorpusMixin
from latincyreaders.core.protocols import CorpusReaderProtocol
from latincyreaders.core.selector import FileSelector

__all__ = [
    "BaseCorpusReader",
    "AnnotationLevel",
    "CombinedReader",
    "combine",
    "CorpusReaderProtocol",
    "DownloadableCorpusMixin",
    "FileSelector",
]
