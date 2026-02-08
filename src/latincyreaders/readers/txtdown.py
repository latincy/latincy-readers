"""Txtdown format corpus reader.

Reader for txtdown (.txtd) files - a minimal markup format for Latin text
collections with YAML metadata and section-based organization.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span

# Import txtdown parser
try:
    from txtdown import parse as txtdown_parse
    from txtdown import Document as TxtdownDocument
    TXTDOWN_AVAILABLE = True
except ImportError:
    TXTDOWN_AVAILABLE = False

# Pattern to strip blockquote markers: leading whitespace, one or more >, optional space
_BLOCKQUOTE_PREFIX = re.compile(r"^\s*>+\s?")


class TxtdownReader(BaseCorpusReader):
    """Reader for txtdown format Latin texts.

    Txtdown is a minimal markup format designed for Latin text collections:
    - YAML front matter for document metadata
    - Section separators with optional IDs and titles (--- 99: Title)
    - Automatic line numbering within sections
    - Citation access via section.line notation
    - Blockquotes (> prefix) join with surrounding text for NLP

    Example:
        >>> reader = TxtdownReader("/path/to/texts")
        >>> for doc in reader.docs():
        ...     print(doc._.metadata)
        ...     for sent in doc.sents:
        ...         print(sent._.citation, sent.text)
    """

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        cache: bool = True,
        cache_maxsize: int = 128,
    ):
        """Initialize the txtdown reader.

        Args:
            root: Root directory containing .txtd files.
            fileids: Glob pattern for selecting files. Defaults to "**/*.txtd".
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            cache: If True (default), cache processed Doc objects for reuse.
            cache_maxsize: Maximum number of documents to cache (default 128).

        Raises:
            ImportError: If txtdown package is not installed.
        """
        if not TXTDOWN_AVAILABLE:
            raise ImportError(
                "txtdown package required. Install with: pip install txtdown"
            )
        super().__init__(
            root, fileids, encoding, annotation_level,
            cache=cache, cache_maxsize=cache_maxsize
        )

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Default glob pattern for txtdown files."""
        return "**/*.txtd"

    def _normalize_text(self, text: str) -> str:
        """Normalize text, stripping blockquote markers and joining continuations.

        Blockquotes (lines starting with >) are joined with the preceding
        text to form continuous sentences for NLP processing.

        Example:
            "Nonne uidit Aeneas Priamum per aras\\n\\n> Sanguine foedantem..."
            becomes:
            "Nonne uidit Aeneas Priamum per aras Sanguine foedantem..."

        Args:
            text: Raw text with possible blockquote markers.

        Returns:
            Text with blockquotes stripped and joined as continuations.
        """
        import unicodedata

        # First do unicode normalization (from base class)
        text = unicodedata.normalize("NFC", text)

        # Process lines to handle blockquotes
        lines = text.split("\n")
        result_lines: list[str] = []

        for line in lines:
            # Check if this is a blockquote line
            if line.lstrip().startswith(">"):
                # Strip the > prefix and leading whitespace after it
                stripped = _BLOCKQUOTE_PREFIX.sub("", line)
                if stripped:
                    # Join with previous line if there is one
                    if result_lines:
                        # Remove trailing whitespace from previous line and join
                        prev = result_lines[-1].rstrip()
                        result_lines[-1] = prev + " " + stripped
                    else:
                        result_lines.append(stripped)
            else:
                result_lines.append(line)

        return "\n".join(result_lines)

    @staticmethod
    def _strip_blockquote_marker(text: str) -> str:
        """Strip blockquote prefix (>) from a line of text.

        This is needed because the txtdown parser stores raw line text
        including blockquote markers, but _normalize_text strips them
        from the Doc text. When mapping lines back to Doc positions,
        we need to search for the text without the marker.

        Args:
            text: Raw line text, possibly with > prefix.

        Returns:
            Text with blockquote marker removed if present.
        """
        if text.lstrip().startswith(">"):
            return _BLOCKQUOTE_PREFIX.sub("", text).strip()
        return text

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a txtdown file into text with metadata.

        Yields a single (text, metadata) tuple per file, where metadata
        includes document-level info plus section/line citation data.
        """
        doc = txtdown_parse(path)

        # Build document metadata from front matter
        metadata = {
            "filename": path.name,
            "path": str(path),
            "author": doc.metadata.author,
            "work": doc.metadata.work,
            "source": doc.metadata.source,
            **doc.metadata.extras,
        }

        # Build section data for citation tracking
        sections_data = []
        for section in doc.sections:
            section_info = {
                "id": section.id,
                "title": section.title,
                "line_count": len(section.lines),
                "lines": [
                    {"number": line.number, "text": line.text}
                    for line in section.lines
                ],
            }
            sections_data.append(section_info)

        metadata["sections"] = sections_data

        # Combine all section text
        full_text = "\n\n".join(section.text for section in doc.sections)

        yield full_text, metadata

    def docs(
        self,
        fileids: str | list[str] | None = None,
        annotation_level: AnnotationLevel | None = None,
        cache: bool = True,
    ) -> Iterator["Doc"]:
        """Yield spaCy Doc objects with citation-aware spans.

        Each Doc has:
        - doc._.metadata: Full document metadata including sections
        - doc._.fileid: File identifier
        - doc.spans["sections"]: Section spans with citation info
        - doc.spans["lines"]: Line spans with citation info

        Args:
            fileids: Files to process, or None for all.
            annotation_level: Override default annotation level.
            cache: If True, cache docs for reuse.

        Yields:
            spaCy Doc objects.
        """
        nlp = self.nlp

        if nlp is None:
            raise ValueError(
                "Cannot create Docs with annotation_level=NONE. "
                "Use texts() for raw strings."
            )

        for path in self._iter_paths(fileids):
            fileid = str(path.relative_to(self._root))

            for text, metadata in self._parse_file(path):
                text = self._normalize_text(text)
                doc = nlp(text)
                doc._.fileid = fileid
                doc._.metadata = metadata

                # Add citation spans
                self._add_citation_spans(doc, metadata)

                yield doc

    def _add_citation_spans(self, doc: "Doc", metadata: dict) -> None:
        """Add section and line spans with citation info to the Doc."""
        from spacy.tokens import Span

        sections_data = metadata.get("sections", [])
        if not sections_data:
            return

        section_spans = []
        line_spans = []

        # Track character position through the document
        char_pos = 0

        for section_info in sections_data:
            section_id = section_info["id"]
            section_title = section_info.get("title")
            section_start_char = char_pos

            for line_info in section_info["lines"]:
                line_text = line_info["text"]
                line_num = line_info["number"]

                # Strip blockquote markers before searching in normalized doc text.
                # The txtdown parser stores raw line text (including > prefix),
                # but _normalize_text strips markers and joins blockquote content
                # with the preceding line. We must search for the cleaned text.
                line_text_stripped = self._strip_blockquote_marker(line_text)
                if not line_text_stripped:
                    line_text_stripped = line_text.strip()
                line_start = doc.text.find(line_text_stripped, char_pos)

                if line_start >= 0:
                    line_end = line_start + len(line_text_stripped)
                    span = doc.char_span(line_start, line_end, alignment_mode="expand")

                    if span:
                        citation = f"{section_id}.{line_num}"
                        span._.citation = citation
                        span._.metadata = {
                            "section_id": section_id,
                            "section_title": section_title,
                            "line_number": line_num,
                        }
                        line_spans.append(span)

                    char_pos = line_end

            # Create section span
            section_end_char = char_pos
            section_span = doc.char_span(
                section_start_char, section_end_char, alignment_mode="expand"
            )
            if section_span:
                section_span._.citation = section_id
                section_span._.metadata = {
                    "section_id": section_id,
                    "section_title": section_title,
                }
                section_spans.append(section_span)

            # Advance past whitespace between sections. The raw text uses
            # "\n\n" but spaCy may normalize this to a single space, so we
            # skip forward to the next non-whitespace character rather than
            # assuming a fixed offset.
            while char_pos < len(doc.text) and doc.text[char_pos] in ' \n\t\r':
                char_pos += 1

        doc.spans["sections"] = section_spans
        doc.spans["lines"] = line_spans

    def sents_with_citations(
        self,
        fileids: str | list[str] | None = None,
    ) -> Iterator[dict]:
        """Yield sentences with full citation metadata.

        Returns dicts with:
        - sentence: The sentence text
        - section_id: Section identifier
        - section_title: Section title (if any)
        - line_citations: List of line citations covered by this sentence
        - fileid: Source file
        - metadata: Document metadata

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Dicts with sentence text and citation info.
        """
        for doc in self.docs(fileids):
            metadata = doc._.metadata or {}
            fileid = doc._.fileid
            line_spans = doc.spans.get("lines", [])

            for sent in doc.sents:
                # Find which lines this sentence overlaps
                covered_lines = []
                section_id = None
                section_title = None

                for line_span in line_spans:
                    if (line_span.start < sent.end and line_span.end > sent.start):
                        covered_lines.append(line_span._.citation)
                        if line_span._.metadata:
                            section_id = line_span._.metadata.get("section_id")
                            section_title = line_span._.metadata.get("section_title")

                yield {
                    "sentence": sent.text,
                    "section_id": section_id,
                    "section_title": section_title,
                    "line_citations": covered_lines,
                    "fileid": fileid,
                    "metadata": {
                        "author": metadata.get("author"),
                        "work": metadata.get("work"),
                    },
                }


# Alias
TxtdownCorpusReader = TxtdownReader
