"""Base corpus reader class.

This module provides the abstract base class that all corpus readers inherit from.
It handles common functionality like file discovery, NLP pipeline management,
and the standard iteration interface.
"""

from __future__ import annotations

import json
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

from tqdm import tqdm

from latincyreaders.nlp.pipeline import AnnotationLevel, get_nlp

if TYPE_CHECKING:
    from spacy import Language
    from spacy.tokens import Doc, Span, Token
    from latincyreaders.core.selector import FileSelector

# Re-export for convenience
__all__ = ["BaseCorpusReader", "AnnotationLevel"]


class BaseCorpusReader(ABC):
    """Abstract base class for all Latin corpus readers.

    To create a new reader, subclass and implement:

    Required:
        - _parse_file(path) -> yields (text, metadata) tuples

    Optional overrides:
        - _normalize_text(text) -> cleaned text
        - _default_file_pattern() -> glob pattern for files

    Example:
        class MyReader(BaseCorpusReader):
            @classmethod
            def _default_file_pattern(cls) -> str:
                return "*.txt"

            def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
                yield path.read_text(), {"filename": path.name}
    """

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        metadata_pattern: str = "metadata/*.json",
        cache: bool = True,
        cache_maxsize: int = 128,
        model_name: str = "la_core_web_lg",
        lang: str = "la",
    ):
        """Initialize the corpus reader.

        Args:
            root: Root directory containing corpus files.
            fileids: Glob pattern for selecting files. If None, uses class default.
            encoding: Text encoding for reading files.
            annotation_level: How much NLP annotation to apply.
            metadata_pattern: Glob pattern for metadata JSON files. Set to None to disable.
            cache: If True (default), cache processed Doc objects for reuse.
            cache_maxsize: Maximum number of documents to cache (default 128).
            model_name: Name of the spaCy model to load for BASIC/FULL levels.
            lang: Language code for blank model in TOKENIZE level.
        """
        self._root = Path(root).resolve()
        self._fileids_pattern = fileids or self._default_file_pattern()
        self._encoding = encoding
        self._annotation_level = annotation_level
        self._model_name = model_name
        self._lang = lang
        self._nlp: Language | None = None  # Lazy loaded
        self._metadata_pattern = metadata_pattern
        self._metadata: dict[str, dict[str, Any]] | None = None  # Lazy loaded

        # Caching
        self._cache_enabled = cache
        self._cache_maxsize = cache_maxsize
        self._cache: OrderedDict[str, "Doc"] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def root(self) -> Path:
        """Root directory of the corpus."""
        return self._root

    @property
    def nlp(self) -> Language | None:
        """spaCy pipeline (lazy loaded on first access)."""
        if self._nlp is None and self._annotation_level != AnnotationLevel.NONE:
            self._nlp = get_nlp(
                self._annotation_level,
                model_name=self._model_name,
                lang=self._lang,
            )
        return self._nlp

    @property
    def annotation_level(self) -> AnnotationLevel:
        """Current annotation level."""
        return self._annotation_level

    @property
    def cache_enabled(self) -> bool:
        """Whether document caching is enabled."""
        return self._cache_enabled

    def cache_stats(self) -> dict[str, int]:
        """Return cache statistics.

        Returns:
            Dict with keys:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - size: Current number of cached documents
                - maxsize: Maximum cache size
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "maxsize": self._cache_maxsize,
        }

    def clear_cache(self) -> None:
        """Clear the document cache and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        """Load and aggregate metadata from JSON files.

        Searches for JSON files matching metadata_pattern and merges them
        by fileid key. Later files override earlier ones for duplicate keys.

        Returns:
            Dict mapping fileid -> metadata dict.
        """
        if self._metadata_pattern is None:
            return {}

        merged: dict[str, dict[str, Any]] = {}

        for json_file in sorted(self._root.glob(self._metadata_pattern)):
            try:
                data = json.loads(json_file.read_text(encoding=self._encoding))
                if isinstance(data, dict):
                    for fileid, meta in data.items():
                        if isinstance(meta, dict):
                            merged.setdefault(fileid, {}).update(meta)
            except (json.JSONDecodeError, OSError):
                # Skip malformed or unreadable files
                continue

        return merged

    def get_metadata(self, fileid: str) -> dict[str, Any]:
        """Get metadata for a specific file.

        Args:
            fileid: File identifier.

        Returns:
            Metadata dict for the file, or empty dict if not found.
        """
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata.get(fileid, {})

    def metadata(
        self,
        fileids: str | list[str] | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield (fileid, metadata) pairs.

        Args:
            fileids: Files to get metadata for, or None for all.

        Yields:
            Tuples of (fileid, metadata_dict).
        """
        for fileid in self._resolve_fileids(fileids):
            yield fileid, self.get_metadata(fileid)

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Default glob pattern for this corpus type. Override in subclasses."""
        return "*.*"

    def _normalize_text(self, text: str) -> str:
        """Normalize text. Override for corpus-specific cleaning.

        Args:
            text: Raw text from file.

        Returns:
            Normalized text.
        """
        return unicodedata.normalize("NFC", text)

    def fileids(self, match: str | None = None) -> list[str]:
        """Return list of file identifiers matching the pattern.

        Args:
            match: Optional regex pattern to filter filenames.

        Returns:
            Naturally sorted list of matching file identifiers (relative paths).
        """
        from natsort import natsorted

        pattern = self._fileids_pattern
        files = self._root.glob(pattern)

        # Convert to relative paths as strings
        result = [str(f.relative_to(self._root)) for f in files if f.is_file()]

        # Apply optional regex filter
        if match:
            regex = re.compile(match, re.IGNORECASE)
            result = [f for f in result if regex.search(f)]

        # Natural sort (handles numbers correctly: part.1, part.2, ..., part.10)
        return natsorted(result)

    def select(self) -> "FileSelector":
        """Create a FileSelector for fluent file filtering.

        Returns a FileSelector that allows chaining filters on filenames
        and metadata. The resulting selection can be passed to docs(),
        texts(), sents(), etc.

        Returns:
            A new FileSelector instance.

        Example:
            >>> # Select epic poetry by Vergil
            >>> selection = reader.select().where(author="Vergil", genre="epic")
            >>> for doc in reader.docs(selection):
            ...     print(doc._.fileid)

            >>> # Select files by date range
            >>> augustan = reader.select().date_range(-50, 50)
            >>> print(f"Found {len(augustan)} Augustan texts")
        """
        from latincyreaders.core.selector import FileSelector

        return FileSelector(self)

    def _resolve_fileids(
        self, fileids: str | list[str] | "FileSelector" | None
    ) -> list[str]:
        """Resolve fileids argument to a list of file identifiers.

        Args:
            fileids: Single fileid, list of fileids, FileSelector, or None for all files.

        Returns:
            List of file identifiers.
        """
        if fileids is None:
            return self.fileids()
        if isinstance(fileids, str):
            return [fileids]
        # Handle any iterable (including FileSelector)
        return list(fileids)

    def _iter_paths(self, fileids: str | list[str] | None = None) -> Iterator[Path]:
        """Iterate over file paths for the given fileids.

        Args:
            fileids: Files to iterate over.

        Yields:
            Path objects for each file.
        """
        for fid in self._resolve_fileids(fileids):
            yield self._root / fid

    @abstractmethod
    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a single file. Yield (text_chunk, metadata) pairs.

        This is the main extension point for subclasses. Implement this method
        to handle the specific file format of your corpus.

        Args:
            path: Path to the file to parse.

        Yields:
            Tuples of (text, metadata_dict) for each logical unit in the file.
        """
        ...

    # -------------------------------------------------------------------------
    # Core iteration methods
    # -------------------------------------------------------------------------

    def texts(self, fileids: str | list[str] | None = None) -> Iterator[str]:
        """Yield raw text strings. Zero NLP overhead.

        This is the fastest way to iterate over corpus content when you
        don't need any linguistic annotation.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Raw text strings.
        """
        for path in self._iter_paths(fileids):
            for text, _metadata in self._parse_file(path):
                yield self._normalize_text(text)

    def docs(self, fileids: str | list[str] | None = None) -> Iterator["Doc"]:
        """Yield spaCy Doc objects with annotations.

        The level of annotation depends on the reader's annotation_level setting.
        Metadata from JSON files is merged with any metadata from _parse_file().

        When caching is enabled (default), documents are stored after first access
        and returned from cache on subsequent requests for the same fileid.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            spaCy Doc objects.
        """
        nlp = self.nlp
        if nlp is None:
            raise ValueError(
                "Cannot create Docs with annotation_level=NONE. "
                "Use texts() for raw strings, or set a higher annotation level."
            )

        for path in self._iter_paths(fileids):
            fileid = str(path.relative_to(self._root))

            # Check cache first
            if self._cache_enabled and fileid in self._cache:
                self._cache_hits += 1
                # Move to end for LRU ordering
                self._cache.move_to_end(fileid)
                yield self._cache[fileid]
                continue

            # Cache miss - process the file
            if self._cache_enabled:
                self._cache_misses += 1

            # Get JSON metadata and merge with file-level metadata
            json_metadata = self.get_metadata(fileid)

            for text, file_metadata in self._parse_file(path):
                text = self._normalize_text(text)
                doc = nlp(text)
                doc._.fileid = fileid
                # Merge: JSON metadata as base, file metadata overrides
                doc._.metadata = {**json_metadata, **file_metadata}

                # Store in cache if enabled
                if self._cache_enabled:
                    # Evict oldest if at capacity
                    while len(self._cache) >= self._cache_maxsize:
                        self._cache.popitem(last=False)
                    self._cache[fileid] = doc

                yield doc

    def sents(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Span | str"]:
        """Yield sentences from documents.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Span objects.

        Yields:
            Sentence Spans (or strings if as_text=True).
        """
        for doc in self.docs(fileids):
            for sent in doc.sents:
                yield sent.text if as_text else sent

    def tokens(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Token | str"]:
        """Yield individual tokens from documents.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Token objects.

        Yields:
            Tokens (or strings if as_text=True).
        """
        for doc in self.docs(fileids):
            for token in doc:
                yield token.text if as_text else token

    # -------------------------------------------------------------------------
    # Text analysis methods
    # -------------------------------------------------------------------------

    def _get_token_citation(self, doc: "Doc", token: "Token", token_idx: int) -> str:
        """Get citation for a token, checking spans if not on token directly.

        Args:
            doc: The document containing the token.
            token: The token to get citation for.
            token_idx: Index of the token in the document.

        Returns:
            Citation string, or fileid:idx fallback.
        """
        # First check token-level citation
        citation = getattr(token._, "citation", None)
        if citation is not None:
            return citation

        # Check if token is within a span that has a citation (e.g., Tesserae lines)
        for span_key in doc.spans:
            for span in doc.spans[span_key]:
                if span.start <= token.i < span.end:
                    span_citation = getattr(span._, "citation", None)
                    if span_citation is not None:
                        return span_citation

        # Fallback to fileid:index
        fileid = doc._.fileid or "unknown"
        return f"{fileid}:{token_idx}"

    def concordance(
        self,
        fileids: str | list[str] | None = None,
        basis: str = "lemma",
        only_alpha: bool = True,
    ) -> dict[str, list[str]]:
        """Build a concordance mapping words to their citation locations.

        A concordance is a dictionary where keys are word forms and values
        are lists of citations/locations where that word appears.

        Args:
            fileids: Files to process, or None for all.
            basis: How to key the concordance:
                - "lemma": group by lemma (default, recommended)
                - "norm": group by normalized form (spaCy's norm_)
                - "text": group by exact surface form
            only_alpha: If True, skip non-alphabetic tokens (punctuation, numbers).

        Returns:
            Dict mapping word form -> list of citation strings.
            Citations are in format "<citation>" if available, else "fileid:token_idx".

        Example:
            >>> conc = reader.concordance(basis="lemma")
            >>> conc["amor"]
            ['<catull. 1.1>', '<catull. 1.3>', '<verg. aen. 4.1>']
        """
        from collections import defaultdict

        concordance_dict: defaultdict[str, list[str]] = defaultdict(list)

        for doc in self.docs(fileids):
            for i, token in enumerate(doc):
                # Skip non-alphabetic tokens if requested
                if only_alpha and not token.is_alpha:
                    continue

                # Determine the key based on basis
                if basis == "lemma":
                    key = token.lemma_
                elif basis == "norm":
                    key = token.norm_
                else:  # "text" or fallback
                    key = token.text

                citation = self._get_token_citation(doc, token, i)
                concordance_dict[key].append(citation)

        # Sort by key and return as regular dict
        return dict(sorted(concordance_dict.items()))

    def kwic(
        self,
        keyword: str,
        fileids: str | list[str] | None = None,
        window: int = 5,
        ignore_case: bool = True,
        by_lemma: bool = False,
        limit: int | None = None,
    ) -> Iterator[dict[str, str]]:
        """Find keyword in context (KWIC) across the corpus.

        Returns matches with surrounding context, useful for studying
        word usage patterns.

        Args:
            keyword: Word to search for.
            fileids: Files to search, or None for all.
            window: Number of tokens on each side for context.
            ignore_case: If True, match case-insensitively.
            by_lemma: If True, match against lemma instead of surface form.
            limit: Maximum number of results to return.

        Yields:
            Dicts with keys:
                - left: left context (string)
                - match: matched token (string)
                - right: right context (string)
                - citation: citation string if available
                - fileid: file identifier

        Example:
            >>> for hit in reader.kwic("amor", window=3, by_lemma=True):
            ...     print(f"{hit['left']} [{hit['match']}] {hit['right']}")
            ...     print(f"  -- {hit['citation']}")
        """
        target = keyword.lower() if ignore_case else keyword
        count = 0

        for doc in self.docs(fileids):
            fileid = doc._.fileid or "unknown"
            tokens = list(doc)

            for i, token in enumerate(tokens):
                # Determine what to match against
                if by_lemma:
                    token_value = token.lemma_.lower() if ignore_case else token.lemma_
                else:
                    token_value = token.text.lower() if ignore_case else token.text

                if token_value == target:
                    # Build context windows
                    left_start = max(0, i - window)
                    right_end = min(len(tokens), i + window + 1)

                    left_tokens = tokens[left_start:i]
                    right_tokens = tokens[i + 1:right_end]

                    left_text = " ".join(t.text for t in left_tokens)
                    right_text = " ".join(t.text for t in right_tokens)

                    citation = self._get_token_citation(doc, token, i)

                    yield {
                        "left": left_text,
                        "match": token.text,
                        "right": right_text,
                        "citation": citation,
                        "fileid": fileid,
                    }

                    count += 1
                    if limit is not None and count >= limit:
                        return

    def ngrams(
        self,
        n: int = 2,
        fileids: str | list[str] | None = None,
        filter_stops: bool = False,
        filter_punct: bool = True,
        filter_nums: bool = False,
        basis: str = "text",
        as_tuples: bool = False,
    ) -> Iterator[str | tuple["Token", ...]]:
        """Extract n-grams from the corpus.

        N-grams are contiguous sequences of n tokens. Useful for
        collocations, frequency analysis, and language modeling.

        Args:
            n: Size of n-grams (2 for bigrams, 3 for trigrams, etc.).
            fileids: Files to process, or None for all.
            filter_stops: If True, exclude n-grams containing stop words.
            filter_punct: If True, exclude n-grams containing punctuation.
            filter_nums: If True, exclude n-grams containing numbers.
            basis: How to represent tokens in output strings:
                - "text": surface form (default) - "amat te"
                - "lemma": lemmatized form - "amo tu"
                - "norm": normalized form (spaCy's norm_)
            as_tuples: If True, yield tuples of Token objects instead of strings.
                When True, basis is ignored.

        Yields:
            N-gram strings like "arma virumque" (default), or tuples of
            Token objects if as_tuples=True.

        Example:
            >>> # Get all bigrams from Catullus
            >>> bigrams = list(reader.ngrams(n=2, fileids="catullus.*"))
            >>> print(bigrams[:5])
            ['Cui dono', 'dono lepidum', 'lepidum novum', ...]

            >>> # Get bigrams by lemma for better frequency analysis
            >>> lemma_bigrams = list(reader.ngrams(n=2, basis="lemma"))
            >>> print(lemma_bigrams[:5])
            ['qui do', 'do lepidus', 'lepidus novus', ...]

            >>> # Get trigrams as token tuples for linguistic analysis
            >>> for gram in reader.ngrams(n=3, as_tuples=True, fileids="catullus.*"):
            ...     print([(t.text, t.pos_) for t in gram])
        """
        import textacy.extract

        for doc in self.docs(fileids):
            ngram_spans = textacy.extract.ngrams(
                doc,
                n=n,
                filter_stops=filter_stops,
                filter_punct=filter_punct,
                filter_nums=filter_nums,
            )

            for span in ngram_spans:
                if as_tuples:
                    yield tuple(token for token in span)
                else:
                    if basis == "lemma":
                        yield " ".join(t.lemma_ for t in span)
                    elif basis == "norm":
                        yield " ".join(t.norm_ for t in span)
                    else:  # "text" or fallback
                        yield span.text

    def skipgrams(
        self,
        n: int = 2,
        k: int = 1,
        fileids: str | list[str] | None = None,
        filter_stops: bool = False,
        filter_punct: bool = True,
        filter_nums: bool = False,
        basis: str = "text",
        as_tuples: bool = False,
    ) -> Iterator[str | tuple["Token", ...]]:
        """Extract skipgrams from the corpus.

        Skipgrams are like n-grams but allow gaps between tokens.
        For example, a (2,1)-skipgram from "the quick brown fox" includes
        both "the quick" and "the brown" (skipping "quick").

        Args:
            n: Number of tokens in each skipgram.
            k: Maximum number of tokens to skip between included tokens.
            fileids: Files to process, or None for all.
            filter_stops: If True, exclude skipgrams containing stop words.
            filter_punct: If True, exclude skipgrams containing punctuation.
            filter_nums: If True, exclude skipgrams containing numbers.
            basis: How to represent tokens in output strings:
                - "text": surface form (default)
                - "lemma": lemmatized form
                - "norm": normalized form (spaCy's norm_)
            as_tuples: If True, yield tuples of Token objects instead of strings.
                When True, basis is ignored.

        Yields:
            Skipgram strings (default), or tuples of Token objects if as_tuples=True.

        Example:
            >>> # Bigrams with 1 skip - captures non-adjacent word pairs
            >>> for sg in reader.skipgrams(n=2, k=1, fileids="catullus.*"):
            ...     print(sg)

            >>> # Skipgrams by lemma
            >>> for sg in reader.skipgrams(n=2, k=1, basis="lemma"):
            ...     print(sg)
        """
        for doc in self.docs(fileids):
            # Filter tokens first
            tokens = [t for t in doc if self._token_passes_filters(
                t, filter_stops, filter_punct, filter_nums
            )]

            for i in range(len(tokens)):
                for skip in range(k + 1):
                    # Build skipgram indices
                    indices = []
                    pos = i
                    for _ in range(n):
                        if pos >= len(tokens):
                            break
                        indices.append(pos)
                        pos += skip + 1

                    if len(indices) == n:
                        gram_tokens = tuple(tokens[idx] for idx in indices)
                        if as_tuples:
                            yield gram_tokens
                        else:
                            if basis == "lemma":
                                yield " ".join(t.lemma_ for t in gram_tokens)
                            elif basis == "norm":
                                yield " ".join(t.norm_ for t in gram_tokens)
                            else:  # "text" or fallback
                                yield " ".join(t.text for t in gram_tokens)

    def _token_passes_filters(
        self,
        token: "Token",
        filter_stops: bool,
        filter_punct: bool,
        filter_nums: bool,
    ) -> bool:
        """Check if a token passes the specified filters."""
        if filter_stops and token.is_stop:
            return False
        if filter_punct and token.is_punct:
            return False
        if filter_nums and token.like_num:
            return False
        return True

    # -------------------------------------------------------------------------
    # Sentence search methods
    # -------------------------------------------------------------------------

    def _get_citation_for_span(self, doc: "Doc", span: "Span") -> str:
        """Get citation for a span (sentence).

        Override in subclasses for format-specific citations.

        Args:
            doc: The document containing the span.
            span: The span to get citation for.

        Returns:
            Citation string.
        """
        # Check if span has a citation attribute
        citation = getattr(span._, "citation", None)
        if citation is not None:
            return citation

        # Check if span overlaps with any citation-bearing spans
        for span_key in doc.spans:
            for labeled_span in doc.spans[span_key]:
                if labeled_span.start <= span.start < labeled_span.end:
                    span_citation = getattr(labeled_span._, "citation", None)
                    if span_citation is not None:
                        return span_citation

        # Fallback to fileid:sent_index
        fileid = doc._.fileid or "unknown"
        sents = list(doc.sents)
        for i, s in enumerate(sents):
            if s.start == span.start:
                return f"{fileid}:sent{i}"
        return f"{fileid}:sent?"

    def find_sents(
        self,
        pattern: str | None = None,
        forms: list[str] | None = None,
        lemma: str | list[str] | None = None,
        matcher_pattern: list[dict] | None = None,
        fileids: str | list[str] | None = None,
        ignore_case: bool = True,
        context: bool = False,
        show_progress: bool = False,
    ) -> Iterator[dict]:
        """Find sentences containing specific words/patterns/lemmas.

        This is the main search method for extracting sentences for annotation.

        Args:
            pattern: Regex pattern to match.
            forms: List of exact word forms to match.
            lemma: Lemma or list of lemmas to match (requires NLP - slower).
            matcher_pattern: spaCy Matcher pattern for advanced queries.
            fileids: Files to search, or None for all.
            ignore_case: Whether to ignore case (default True for pattern/forms).
            context: If True, include surrounding sentences.
            show_progress: If True, show tqdm progress bar for file iteration.

        Yields:
            Dicts with keys: fileid, citation, sentence, matches, (prev_sent, next_sent).

        Example:
            >>> for hit in reader.find_sents(pattern=r"\\bTheb\\w+\\b"):
            ...     print(f"{hit['citation']}: {hit['sentence']}")

            >>> for hit in reader.find_sents(lemma=["bellum", "pax"]):
            ...     print(hit['sentence'])
        """
        if matcher_pattern is not None:
            yield from self._find_sents_by_matcher(matcher_pattern, fileids, context, show_progress)
        elif lemma is not None:
            lemmas = [lemma] if isinstance(lemma, str) else lemma
            yield from self._find_sents_by_lemma(lemmas, fileids, context, show_progress)
        else:
            yield from self._find_sents_by_pattern(pattern, forms, fileids, ignore_case, context, show_progress)

    @staticmethod
    def _normalize_sent_text(text: str) -> str:
        """Normalize sentence text by replacing newlines with spaces."""
        # Replace \r\n, \r, \n with space, then collapse multiple spaces
        return " ".join(text.split())

    def _find_sents_by_pattern(
        self,
        pattern: str | None,
        forms: list[str] | None,
        fileids: str | list[str] | None,
        ignore_case: bool,
        context: bool,
        show_progress: bool = False,
    ) -> Iterator[dict]:
        """Find sentences by regex pattern (fast path)."""
        if pattern is None and forms is None:
            raise ValueError("Must provide either pattern or forms")

        if forms is not None:
            escaped = [re.escape(f) for f in forms]
            pattern = r"\b(" + "|".join(escaped) + r")\b"

        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)

        # Get fileids list for progress bar
        if show_progress:
            fids = self._resolve_fileids(fileids)
            doc_iter = tqdm(self.docs(fids), total=len(fids), desc="Files", unit="file")
        else:
            doc_iter = self.docs(fileids)

        for doc in doc_iter:
            sents = list(doc.sents)
            for i, sent in enumerate(sents):
                matches = regex.findall(sent.text)
                if matches:
                    result = {
                        "fileid": doc._.fileid,
                        "citation": self._get_citation_for_span(doc, sent),
                        "sentence": self._normalize_sent_text(sent.text),
                        "matches": matches,
                    }
                    if context:
                        result["prev_sent"] = self._normalize_sent_text(sents[i - 1].text) if i > 0 else None
                        result["next_sent"] = self._normalize_sent_text(sents[i + 1].text) if i < len(sents) - 1 else None
                    yield result

    def _find_sents_by_lemma(
        self,
        lemmas: list[str],
        fileids: str | list[str] | None,
        context: bool,
        show_progress: bool = False,
    ) -> Iterator[dict]:
        """Find sentences by lemma(s) (uses NLP)."""
        target_lemmas = {lem.lower() for lem in lemmas}

        # Get fileids list for progress bar
        if show_progress:
            fids = self._resolve_fileids(fileids)
            doc_iter = tqdm(self.docs(fids), total=len(fids), desc="Files", unit="file")
        else:
            doc_iter = self.docs(fileids)

        for doc in doc_iter:
            sents = list(doc.sents)
            for i, sent in enumerate(sents):
                matches = [t.text for t in sent if t.lemma_.lower() in target_lemmas]
                if matches:
                    matched_lemmas = [
                        t.lemma_.lower() for t in sent if t.lemma_.lower() in target_lemmas
                    ]
                    result = {
                        "fileid": doc._.fileid,
                        "citation": self._get_citation_for_span(doc, sent),
                        "sentence": self._normalize_sent_text(sent.text),
                        "matches": matches,
                        "lemmas": list(set(matched_lemmas)),
                    }
                    if context:
                        result["prev_sent"] = self._normalize_sent_text(sents[i - 1].text) if i > 0 else None
                        result["next_sent"] = self._normalize_sent_text(sents[i + 1].text) if i < len(sents) - 1 else None
                    yield result

    def _find_sents_by_matcher(
        self,
        matcher_pattern: list[dict],
        fileids: str | list[str] | None,
        context: bool,
        show_progress: bool = False,
    ) -> Iterator[dict]:
        """Find sentences using spaCy Matcher patterns."""
        from spacy.matcher import Matcher

        nlp = self.nlp
        if nlp is None:
            raise ValueError("Matcher patterns require NLP pipeline")

        matcher = Matcher(nlp.vocab)
        matcher.add("PATTERN", [matcher_pattern])

        # Get fileids list for progress bar
        if show_progress:
            fids = self._resolve_fileids(fileids)
            doc_iter = tqdm(self.docs(fids), total=len(fids), desc="Files", unit="file")
        else:
            doc_iter = self.docs(fileids)

        for doc in doc_iter:
            sents = list(doc.sents)
            matches = matcher(doc)

            matched_sents: dict[int, list[str]] = {}
            for _, start, end in matches:
                match_span = doc[start:end]
                for i, sent in enumerate(sents):
                    if sent.start <= start < sent.end:
                        if i not in matched_sents:
                            matched_sents[i] = []
                        matched_sents[i].append(match_span.text)
                        break

            for i, match_texts in matched_sents.items():
                sent = sents[i]
                result = {
                    "fileid": doc._.fileid,
                    "citation": self._get_citation_for_span(doc, sent),
                    "sentence": self._normalize_sent_text(sent.text),
                    "matches": match_texts,
                }
                if context:
                    result["prev_sent"] = self._normalize_sent_text(sents[i - 1].text) if i > 0 else None
                    result["next_sent"] = self._normalize_sent_text(sents[i + 1].text) if i < len(sents) - 1 else None
                yield result
