"""CombinedReader for merging outputs from multiple corpus readers.

This module provides a compositor that chains iterator-based methods
across multiple readers through a unified interface. It does NOT implement
CorpusReaderProtocol — it delegates to readers rather than being one.

Example:
    >>> from latincyreaders import TesseraeReader, LatinLibraryReader, combine
    >>>
    >>> combined = combine(TesseraeReader(), LatinLibraryReader())
    >>> for sent in combined.sents():
    ...     print(sent)
    >>>
    >>> # Explicit prefixes
    >>> combined = CombinedReader(
    ...     ("tess", TesseraeReader()),
    ...     ("ll", LatinLibraryReader()),
    ... )
    >>> combined.fileids()
    ['tess/vergil.aeneid.tess', 'll/cicero.txt', ...]
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from itertools import chain
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span, Token

    from latincyreaders.core.protocols import CorpusReaderProtocol


class CombinedReader:
    """Unified interface across multiple corpus readers.

    Combines outputs from any number of readers by chaining their
    iterator-based methods. File IDs are namespaced with a prefix
    derived from each reader's class name (or explicitly provided).

    Args:
        *readers: Readers or (prefix, reader) tuples.
        prefixes: Optional dict mapping reader instances to custom prefixes.
            Only used for readers not passed as tuples.

    Example:
        >>> # Auto-prefix from class name
        >>> CombinedReader(tess_reader, ll_reader)
        >>>
        >>> # Explicit prefix via tuple
        >>> CombinedReader(("tess", tess_reader), ("ll", ll_reader))
        >>>
        >>> # Mix both
        >>> CombinedReader(tess_reader, ("ll", ll_reader))
    """

    def __init__(
        self,
        *readers: CorpusReaderProtocol | tuple[str, CorpusReaderProtocol],
        prefixes: dict | None = None,
    ):
        self._readers: list[tuple[str, CorpusReaderProtocol]] = []

        for reader in readers:
            if isinstance(reader, tuple):
                prefix, r = reader
            else:
                r = reader
                name = type(r).__name__.lower()
                prefix = (prefixes or {}).get(
                    r,
                    name.removesuffix("reader") or name,
                )
            self._readers.append((prefix, r))

    @property
    def readers(self) -> dict[str, CorpusReaderProtocol]:
        """Access individual readers by prefix.

        Returns:
            Dict mapping prefix string to reader instance.
        """
        return dict(self._readers)

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{prefix}={type(r).__name__}" for prefix, r in self._readers
        )
        return f"CombinedReader({parts})"

    def __len__(self) -> int:
        return len(self.fileids())

    # -------------------------------------------------------------------------
    # File ID management
    # -------------------------------------------------------------------------

    def fileids(self, match: str | None = None) -> list[str]:
        """Return namespaced file IDs from all readers.

        Each file ID is prefixed with its reader's namespace:
        ``"tesserae/vergil.aeneid.tess"``.

        Args:
            match: Optional regex pattern to filter the namespaced file IDs.

        Returns:
            List of namespaced file identifiers.
        """
        result: list[str] = []
        for prefix, reader in self._readers:
            for fid in reader.fileids():
                result.append(f"{prefix}/{fid}")

        if match:
            regex = re.compile(match, re.IGNORECASE)
            result = [f for f in result if regex.search(f)]

        # Sort by local fileid (after prefix) so corpora interleave
        result.sort(key=lambda fid: fid.split("/", 1)[1])

        return result

    def _resolve_fileids(
        self,
        fileids: str | list[str] | None = None,
    ) -> list[tuple[str, CorpusReaderProtocol, list[str] | None]]:
        """Map namespaced fileids back to (prefix, reader, local_fileids).

        Args:
            fileids: Namespaced file ID(s), or None for all readers.

        Returns:
            List of (prefix, reader, local_fileids) triples. When fileids
            is None, local_fileids is None (meaning "all files").
        """
        if fileids is None:
            return [(p, r, None) for p, r in self._readers]

        if isinstance(fileids, str):
            fileids = [fileids]

        # Group by prefix
        by_prefix: dict[str, list[str]] = {}
        for fid in fileids:
            prefix, _, local = fid.partition("/")
            by_prefix.setdefault(prefix, []).append(local)

        return [
            (p, r, by_prefix.get(p))
            for p, r in self._readers
            if p in by_prefix
        ]

    # -------------------------------------------------------------------------
    # Core iteration methods
    # -------------------------------------------------------------------------

    def docs(self, fileids: str | list[str] | None = None) -> Iterator[Doc]:
        """Yield spaCy Doc objects from all readers.

        Args:
            fileids: Namespaced file IDs, or None for all.

        Yields:
            spaCy Doc objects.
        """
        return chain.from_iterable(
            r.docs(fileids=fids)
            for _, r, fids in self._resolve_fileids(fileids)
        )

    def sents(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator[Span | str]:
        """Yield sentences from all readers.

        Args:
            fileids: Namespaced file IDs, or None for all.
            as_text: If True, yield strings instead of Span objects.

        Yields:
            Sentence Spans (or strings if as_text=True).
        """
        return chain.from_iterable(
            r.sents(fileids=fids, as_text=as_text)
            for _, r, fids in self._resolve_fileids(fileids)
        )

    def tokens(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator[Token | str]:
        """Yield individual tokens from all readers.

        Args:
            fileids: Namespaced file IDs, or None for all.
            as_text: If True, yield strings instead of Token objects.

        Yields:
            Tokens (or strings if as_text=True).
        """
        return chain.from_iterable(
            r.tokens(fileids=fids, as_text=as_text)
            for _, r, fids in self._resolve_fileids(fileids)
        )

    def texts(self, fileids: str | list[str] | None = None) -> Iterator[str]:
        """Yield raw text strings from all readers. Zero NLP overhead.

        Args:
            fileids: Namespaced file IDs, or None for all.

        Yields:
            Raw text strings.
        """
        return chain.from_iterable(
            r.texts(fileids=fids)
            for _, r, fids in self._resolve_fileids(fileids)
        )

    def metadata(
        self,
        fileids: str | list[str] | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield (namespaced_fileid, metadata) pairs from all readers.

        Args:
            fileids: Namespaced file IDs, or None for all.

        Yields:
            Tuples of (namespaced_fileid, metadata_dict).
        """
        for prefix, reader, fids in self._resolve_fileids(fileids):
            for fileid, meta in reader.metadata(fileids=fids):
                yield (f"{prefix}/{fileid}", meta)

    # -------------------------------------------------------------------------
    # Search & analysis methods
    # -------------------------------------------------------------------------

    def search(
        self,
        pattern: str,
        fileids: str | list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[tuple[str, str, str, list[str]]]:
        """Fast regex search across all readers. No NLP models loaded.

        Delegates to each sub-reader's search() method, which operates
        on raw text. Only readers that implement search() are included.

        Args:
            pattern: Regex pattern to search for.
            fileids: Namespaced file IDs, or None for all.
            **kwargs: Passed through to each reader's search().

        Yields:
            Tuples of (namespaced_fileid, citation, text, matches).
        """
        for prefix, reader, fids in self._resolve_fileids(fileids):
            if not hasattr(reader, "search"):
                continue
            for fileid, citation, text, matches in reader.search(
                pattern, fileids=fids, **kwargs
            ):
                yield (f"{prefix}/{fileid}", citation, text, matches)

    def find_sents(
        self,
        fileids: str | list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[dict]:
        """Find sentences across all readers.

        All keyword arguments are forwarded to each reader's find_sents().

        Args:
            fileids: Namespaced file IDs, or None for all.
            **kwargs: Passed through to each reader's find_sents().

        Yields:
            Result dicts from each reader.
        """
        return chain.from_iterable(
            r.find_sents(fileids=fids, **kwargs)
            for _, r, fids in self._resolve_fileids(fileids)
        )

    def kwic(
        self,
        keyword: str,
        fileids: str | list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, str]]:
        """Keyword in context search across all readers.

        Args:
            keyword: Word to search for.
            fileids: Namespaced file IDs, or None for all.
            **kwargs: Passed through to each reader's kwic().

        Yields:
            KWIC result dicts.
        """
        return chain.from_iterable(
            r.kwic(keyword, fileids=fids, **kwargs)
            for _, r, fids in self._resolve_fileids(fileids)
        )

    def ngrams(
        self,
        n: int = 2,
        fileids: str | list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[str | tuple]:
        """Extract n-grams across all readers.

        Args:
            n: Size of n-grams.
            fileids: Namespaced file IDs, or None for all.
            **kwargs: Passed through to each reader's ngrams().

        Yields:
            N-gram strings or token tuples.
        """
        return chain.from_iterable(
            r.ngrams(n, fileids=fids, **kwargs)
            for _, r, fids in self._resolve_fileids(fileids)
        )

    def skipgrams(
        self,
        n: int = 2,
        k: int = 1,
        fileids: str | list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[str | tuple]:
        """Extract skipgrams across all readers.

        Args:
            n: Number of tokens in each skipgram.
            k: Maximum number of tokens to skip.
            fileids: Namespaced file IDs, or None for all.
            **kwargs: Passed through to each reader's skipgrams().

        Yields:
            Skipgram strings or token tuples.
        """
        return chain.from_iterable(
            r.skipgrams(n, k, fileids=fids, **kwargs)
            for _, r, fids in self._resolve_fileids(fileids)
        )

    def concordance(
        self,
        fileids: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[str]]:
        """Build a concordance merged across all readers.

        Citation lists are concatenated for matching keys.

        Args:
            fileids: Namespaced file IDs, or None for all.
            **kwargs: Passed through to each reader's concordance().

        Returns:
            Dict mapping word form -> list of citation strings.
        """
        merged: dict[str, list[str]] = {}
        for _, reader, fids in self._resolve_fileids(fileids):
            for word, citations in reader.concordance(
                fileids=fids, **kwargs
            ).items():
                merged.setdefault(word, []).extend(citations)
        return merged


def combine(
    *readers: CorpusReaderProtocol | tuple[str, CorpusReaderProtocol],
    **kwargs: Any,
) -> CombinedReader:
    """Shorthand for CombinedReader(*readers).

    Args:
        *readers: Readers or (prefix, reader) tuples.
        **kwargs: Passed through to CombinedReader.

    Returns:
        A new CombinedReader instance.

    Example:
        >>> combined = combine(TesseraeReader(), LatinLibraryReader())
        >>> for sent in combined.sents():
        ...     print(sent)
    """
    return CombinedReader(*readers, **kwargs)
