"""Tesserae corpus reader.

Reads texts in the Tesserae format where each line contains a citation
followed by text:

    <cic. amicit. 1> Q. Mucius augur multa narrare...
    <cic. amicit. 2> cum saepe multa, tum memini...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel
from latincyreaders.core.download import DownloadableCorpusMixin

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span


@dataclass
class TesseraeLine:
    """A single citation-text pair from a Tesserae file."""

    citation: str
    text: str
    line_number: int


class TesseraeReader(DownloadableCorpusMixin, BaseCorpusReader):
    """Reader for Tesserae-format Latin texts with citation preservation.

    The Tesserae format uses angle-bracket citations at the start of each line:

        <author. work. section> Text content here...

    This reader preserves citation information through the NLP pipeline by
    storing it in spaCy custom extensions.

    If no root path is provided, looks for the corpus in:
    1. The path specified by TESSERAE_PATH environment variable
    2. ~/latincy_data/lat_text_tesserae/texts

    If the corpus is not found and auto_download=True (default), offers to
    download from GitHub.

    Example:
        >>> reader = TesseraeReader()  # Uses default location or downloads
        >>> reader = TesseraeReader("/custom/path/to/corpus")
        >>> for doc in reader.docs():
        ...     for line in doc.spans["lines"]:
        ...         print(f"{line._.citation}: {line.text[:50]}...")

    Attributes:
        CITATION_PATTERN: Regex for parsing citation-text pairs.
        CORPUS_URL: GitHub URL for downloading the corpus.
        ENV_VAR: Environment variable for custom corpus path.
    """

    CITATION_PATTERN = re.compile(r"<([^>]+)>\s*(.+)")
    CORPUS_URL = "https://github.com/cltk/lat_text_tesserae.git"
    ENV_VAR = "TESSERAE_PATH"
    DEFAULT_SUBDIR = "lat_text_tesserae/texts"
    _FILE_CHECK_PATTERN = "**/*.tess"

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        auto_download: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
        model_name: str = "la_core_web_lg",
        lang: str = "la",
    ):
        """Initialize the Tesserae reader.

        Args:
            root: Root directory containing .tess files. If None, uses default location.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: How much NLP annotation to apply.
            auto_download: If True and corpus not found, offer to download.
            cache: If True (default), cache processed Doc objects for reuse.
            cache_maxsize: Maximum number of documents to cache (default 128).
            model_name: Name of the spaCy model to load for BASIC/FULL levels.
            lang: Language code for blank model in TOKENIZE level.
        """
        if root is None:
            root = self._get_default_root(auto_download)
        super().__init__(
            root, fileids, encoding, annotation_level,
            cache=cache, cache_maxsize=cache_maxsize,
            model_name=model_name, lang=lang,
        )

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Tesserae files use .tess extension."""
        return "*.tess"

    def _parse_lines(self, text: str) -> Iterator[TesseraeLine]:
        """Parse citation-text pairs from Tesserae format.

        Handles line continuations (lines not starting with '<' are
        appended to the previous line).

        Args:
            text: Raw file content.

        Yields:
            TesseraeLine objects for each citation unit.
        """
        current_citation: str | None = None
        current_text: str | None = None
        line_num = 0

        for raw_line in text.strip().split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            match = self.CITATION_PATTERN.match(raw_line)
            if match:
                # New citation - yield previous if exists
                if current_citation is not None and current_text is not None:
                    yield TesseraeLine(
                        citation=f"<{current_citation}>",
                        text=current_text.strip(),
                        line_number=line_num,
                    )
                    line_num += 1

                current_citation = match.group(1)
                current_text = match.group(2)
            else:
                # Continuation line - append to current
                if current_text is not None:
                    current_text += " " + raw_line

        # Yield final line
        if current_citation is not None and current_text is not None:
            yield TesseraeLine(
                citation=f"<{current_citation}>",
                text=current_text.strip(),
                line_number=line_num,
            )

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a Tesserae file into text chunks with metadata.

        For Tesserae files, we yield the entire file as one text chunk
        but store line information for later span creation.

        Args:
            path: Path to .tess file.

        Yields:
            Single (text, metadata) tuple per file.
        """
        raw_text = path.read_text(encoding=self._encoding)
        lines = list(self._parse_lines(raw_text))

        if not lines:
            return

        # Combine all line texts into one document
        combined_text = " ".join(line.text for line in lines)

        # Store line info in metadata for span creation
        metadata = {
            "filename": path.name,
            "path": str(path),
            "_lines": lines,  # Private: used for span creation
        }

        yield combined_text, metadata

    def docs(self, fileids: str | list[str] | None = None) -> Iterator["Doc"]:
        """Yield spaCy Docs with citation spans.

        Each Doc has a "lines" span group containing citation-annotated spans.
        Metadata from JSON files is merged with file-level metadata.

        When caching is enabled (default), documents are stored after first access
        and returned from cache on subsequent requests for the same fileid.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            spaCy Doc objects with doc.spans["lines"] populated.
        """
        nlp = self.nlp
        if nlp is None:
            raise ValueError(
                "Cannot create Docs with annotation_level=NONE. "
                "Use texts() for raw strings."
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

            # Get JSON metadata
            json_metadata = self.get_metadata(fileid)

            for text, file_metadata in self._parse_file(path):
                text = self._normalize_text(text)
                doc = nlp(text)
                doc._.fileid = fileid
                # Merge: JSON metadata as base, file metadata overrides (excluding private keys)
                clean_file_meta = {k: v for k, v in file_metadata.items() if not k.startswith("_")}
                doc._.metadata = {**json_metadata, **clean_file_meta}

                # Create citation spans
                lines_data = file_metadata.get("_lines", [])
                doc.spans["lines"] = self._make_line_spans(doc, lines_data)

                # Store in cache if enabled
                if self._cache_enabled:
                    # Evict oldest if at capacity
                    while len(self._cache) >= self._cache_maxsize:
                        self._cache.popitem(last=False)
                    self._cache[fileid] = doc

                yield doc

    def _make_line_spans(self, doc: "Doc", lines: list[TesseraeLine]) -> list["Span"]:
        """Create citation-annotated spans from line data.

        Maps each TesseraeLine to a character span in the combined document,
        then creates spaCy Spans with citation metadata.

        Args:
            doc: The spaCy Doc containing the combined text.
            lines: List of TesseraeLine objects with citation info.

        Returns:
            List of Spans with _.citation set.
        """
        spans = []
        char_offset = 0

        for line in lines:
            start_char = char_offset
            end_char = char_offset + len(line.text)

            # Create span using character offsets
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is not None:
                span._.citation = line.citation
                spans.append(span)

            # Move offset (+1 for the space between lines)
            char_offset = end_char + 1

        return spans

    def lines(self, fileids: str | list[str] | None = None) -> Iterator["Span"]:
        """Yield individual citation lines as Spans.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Span objects with _.citation set.
        """
        for doc in self.docs(fileids):
            yield from doc.spans.get("lines", [])

    def doc_rows(self, fileids: str | list[str] | None = None) -> Iterator[dict[str, "Span"]]:
        """Yield citation -> Span mappings for each document.

        Useful for looking up text by citation.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Dict mapping citation strings to Spans.
        """
        for doc in self.docs(fileids):
            yield {span._.citation: span for span in doc.spans.get("lines", [])}

    def texts_by_line(self, fileids: str | list[str] | None = None) -> Iterator[tuple[str, str]]:
        """Yield (citation, text) pairs. Zero NLP overhead.

        This is the fastest way to iterate over Tesserae content when you
        just need citation-text pairs without linguistic annotation.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Tuples of (citation, text).
        """
        for path in self._iter_paths(fileids):
            raw_text = path.read_text(encoding=self._encoding)
            for line in self._parse_lines(raw_text):
                yield line.citation, self._normalize_text(line.text)

    # -------------------------------------------------------------------------
    # Search and filtering methods
    # -------------------------------------------------------------------------

    def search(
        self,
        pattern: str,
        fileids: str | list[str] | None = None,
        ignore_case: bool = True,
    ) -> Iterator[tuple[str, str, str, list[str]]]:
        """Search for pattern in texts. Fast, no NLP required.

        Args:
            pattern: Regex pattern to search for.
            fileids: Files to search, or None for all.
            ignore_case: Whether to ignore case (default True).

        Yields:
            Tuples of (fileid, citation, text, matches) for each matching line.

        Example:
            >>> for fid, cit, text, matches in reader.search(r"Theb\\w+"):
            ...     print(f"{cit}: found {matches}")
        """
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)

        for path in self._iter_paths(fileids):
            fileid = str(path.relative_to(self._root))
            raw_text = path.read_text(encoding=self._encoding)

            for line in self._parse_lines(raw_text):
                text = self._normalize_text(line.text)
                matches = regex.findall(text)
                if matches:
                    yield fileid, line.citation, text, matches

    def find_lines(
        self,
        pattern: str | None = None,
        forms: list[str] | None = None,
        fileids: str | list[str] | None = None,
        ignore_case: bool = True,
    ) -> Iterator[tuple[str, str, str]]:
        """Find citation lines containing specific words/patterns.

        Fast path using regex - no NLP required.

        Args:
            pattern: Regex pattern to match.
            forms: List of exact word forms to match (creates pattern from list).
            fileids: Files to search, or None for all.
            ignore_case: Whether to ignore case (default True).

        Yields:
            Tuples of (fileid, citation, text).

        Example:
            >>> # Find lines with any form of "Thebae"
            >>> for fid, cit, text in reader.find_lines(pattern=r"\\bTheb\\w*\\b"):
            ...     print(f"{cit}: {text[:60]}...")

            >>> # Find lines with specific forms
            >>> forms = ["Thebas", "Thebarum", "Thebis", "Thebanos"]
            >>> for fid, cit, text in reader.find_lines(forms=forms):
            ...     print(f"{cit}: {text[:60]}...")
        """
        if pattern is None and forms is None:
            raise ValueError("Must provide either pattern or forms")

        if forms is not None:
            # Build pattern from forms list with word boundaries
            escaped = [re.escape(f) for f in forms]
            pattern = r"\b(" + "|".join(escaped) + r")\b"

        for fileid, citation, text, _matches in self.search(pattern, fileids, ignore_case):
            yield fileid, citation, text

    def _get_citation_for_span(self, doc: "Doc", span: "Span") -> str:
        """Get Tesserae citation for a span.

        Tesserae files have citation spans in doc.spans["lines"].
        """
        # Check doc.spans["lines"] for overlapping citation
        for line_span in doc.spans.get("lines", []):
            if line_span.start <= span.start < line_span.end:
                citation = getattr(line_span._, "citation", None)
                if citation is not None:
                    return citation

        # Fallback to base implementation
        return super()._get_citation_for_span(doc, span)

    def _find_sents_by_matcher(
        self,
        matcher_pattern: list[dict],
        fileids: str | list[str] | None,
        context: bool,
        show_progress: bool = False,
    ) -> Iterator[dict]:
        """Find sentences using spaCy Matcher patterns.

        Override to use Tesserae-specific citation lookup.

        Args:
            matcher_pattern: List of token patterns for spaCy Matcher.
            fileids: Files to search.
            context: Include surrounding sentences.
            show_progress: If True, display progress bar.

        Yields:
            Result dicts with matched spans.
        """
        from spacy.matcher import Matcher
        from tqdm import tqdm

        nlp = self.nlp
        if nlp is None:
            raise ValueError("Matcher patterns require NLP. Set annotation_level > NONE.")

        # Create and configure matcher
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

            # Group matches by sentence
            sent_matches: dict[int, list[str]] = {}
            for match_id, start, end in matches:
                matched_span = doc[start:end]
                # Find which sentence this match belongs to
                for sent_idx, sent in enumerate(sents):
                    if start >= sent.start and end <= sent.end:
                        if sent_idx not in sent_matches:
                            sent_matches[sent_idx] = []
                        sent_matches[sent_idx].append(matched_span.text)
                        break

            # Yield results for sentences with matches
            for sent_idx, matched_texts in sent_matches.items():
                sent = sents[sent_idx]
                result = {
                    "fileid": doc._.fileid,
                    "citation": self._get_citation_for_span(doc, sent),
                    "sentence": sent.text,
                    "matches": matched_texts,
                    "pattern": matcher_pattern,
                }
                if context:
                    result["prev_sent"] = sents[sent_idx - 1].text if sent_idx > 0 else None
                    result["next_sent"] = sents[sent_idx + 1].text if sent_idx < len(sents) - 1 else None
                yield result

    def export_search_results(
        self,
        results: Iterator[dict],
        format: str = "tsv",
    ) -> str:
        """Export search results to a string in the specified format.

        Args:
            results: Iterator of result dicts from find_sents().
            format: Output format - "tsv", "csv", or "jsonl".

        Returns:
            Formatted string with results.

        Example:
            >>> results = reader.find_sents(forms=["Thebas", "Thebarum"])
            >>> print(reader.export_search_results(results, format="tsv"))
        """
        import json

        results_list = list(results)

        if format == "jsonl":
            return "\n".join(json.dumps(r, ensure_ascii=False) for r in results_list)

        elif format in ("tsv", "csv"):
            sep = "\t" if format == "tsv" else ","
            lines = []
            # Header
            lines.append(sep.join(["fileid", "citation", "matches", "sentence"]))
            # Data
            for r in results_list:
                matches_str = ";".join(r.get("matches", []))
                # Escape quotes and newlines in sentence
                sent = r.get("sentence", "").replace('"', '""').replace("\n", " ")
                if sep == ",":
                    sent = f'"{sent}"'
                lines.append(sep.join([
                    r.get("fileid", ""),
                    r.get("citation", ""),
                    matches_str,
                    sent,
                ]))
            return "\n".join(lines)

        else:
            raise ValueError(f"Unknown format: {format}")
