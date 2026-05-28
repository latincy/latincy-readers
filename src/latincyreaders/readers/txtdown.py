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
from latincyreaders.nlp.pipeline import mark_newlines_from_spans
from latincyreaders.utils.text_utils import find_line_in_doc_text as _find_line_in_doc_text

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
# Text-critical markup patterns (West 1973)
_CRUX_PATTERN = re.compile(r"†([^†]*)†")   # crux: keep text, strip daggers
_ADDITION_PATTERN = re.compile(r"<([^<>]*)>")              # addition: keep text, strip <>
_DELETION_PATTERN = re.compile(r"\{[^}]*\}")               # deletion {}: strip markers AND content
_EXPANSION_PATTERN = re.compile(r"(\w+)\((\w+)\)")         # expansion M(arcus) → Marcus


def _collect_markup(text: str) -> list[tuple]:
    """Find all text-critical markup occurrences in *text* before stripping.

    Returns a list of (source_start, source_end, type, original, replacement)
    sorted by position, where *replacement* is the string that will replace
    *original* after stripping (empty string for deletions).
    """
    spans: list[tuple] = []
    for m in _DELETION_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), "deletion", m.group(0), ""))
    for m in _EXPANSION_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), "expansion", m.group(0), m.group(1) + m.group(2)))
    for m in _CRUX_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), "crux", m.group(0), m.group(1)))
    for m in _ADDITION_PATTERN.finditer(text):
        spans.append((m.start(), m.end(), "addition", m.group(0), m.group(1)))
    return sorted(spans, key=lambda x: x[0])


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
        **kwargs,
    ):
        """Initialize the txtdown reader.

        Args:
            root: Root directory containing .txtd files.
            fileids: Glob pattern for selecting files. Defaults to "**/*.txtd".
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            cache: If True (default), cache processed Doc objects for reuse.
            cache_maxsize: Maximum number of documents to cache (default 128).
            **kwargs: Additional arguments passed to BaseCorpusReader (e.g., backend).

        Raises:
            ImportError: If txtdown package is not installed.
        """
        if not TXTDOWN_AVAILABLE:
            raise ImportError(
                "txtdown package required. Install with: pip install txtdown"
            )
        super().__init__(
            root, fileids, encoding, annotation_level,
            cache=cache, cache_maxsize=cache_maxsize,
            **kwargs,
        )

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Default glob pattern for txtdown files."""
        return "**/*.txtd"

    def _normalize_pre_markup(self, text: str) -> str:
        """Apply unicode normalization and blockquote joining, leaving markup in place.

        This is the first half of normalization. Call _strip_critical_markup()
        afterwards for fully clean NLP-ready text, or collect markup positions
        first via _collect_markup() before stripping.

        Args:
            text: Raw text with possible blockquote markers and critical markup.

        Returns:
            Text with blockquotes resolved but critical markup (†, <>, {}, ())
            still intact.
        """
        import unicodedata

        text = unicodedata.normalize("NFC", text)

        lines = text.split("\n")
        result_lines: list[str] = []

        for line in lines:
            if line.lstrip().startswith(">"):
                stripped = _BLOCKQUOTE_PREFIX.sub("", line)
                if stripped:
                    if result_lines:
                        prev = result_lines[-1].rstrip()
                        result_lines[-1] = prev + " " + stripped
                    else:
                        result_lines.append(stripped)
            else:
                result_lines.append(line)

        return "\n".join(result_lines)

    def _normalize_text(self, text: str) -> str:
        """Full normalization for texts(): blockquotes + all critical markup stripped.

        Args:
            text: Raw text.

        Returns:
            NLP-ready text with all markup removed.
        """
        return self._strip_critical_markup(self._normalize_pre_markup(text))

    @staticmethod
    def _strip_critical_markup(text: str) -> str:
        """Strip text-critical markup, preserving the enclosed text.

        Handles cruxes (†text†) and editorial additions (<text>).
        The enclosed text is kept; only the markers are removed.

        Args:
            text: Text possibly containing critical markup.

        Returns:
            Text with markup markers removed.
        """
        text = _DELETION_PATTERN.sub("", text)           # {spurious} → gone
        text = _EXPANSION_PATTERN.sub(r"\1\2", text)    # M(arcus) → Marcus
        text = _CRUX_PATTERN.sub(r"\1", text)            # †text† → text
        text = _ADDITION_PATTERN.sub(r"\1", text)        # <text> → text
        return text

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

    @staticmethod
    def _find_line_in_doc_text(
        doc_text: str, line_text: str, start_pos: int
    ) -> tuple[int, int]:
        """Locate ``line_text`` inside ``doc_text`` starting at ``start_pos``.

        The NLP pipeline (LatinCy ``normer`` plus tokenization) rewrites the
        text in ways a plain ``str.find`` cannot survive: J→I and V→U
        orthographic normalization, whitespace inserted around punctuation,
        and newline collapse. This helper tries an exact match first and
        falls back to a regex that tolerates those transformations.

        Args:
            doc_text: Text of the spaCy Doc (post-normalization).
            line_text: Original line text from the txtdown parser, with
                blockquote markers already stripped.
            start_pos: Char offset to begin the search at.

        Returns:
            ``(start, end)`` char positions in ``doc_text`` covering the
            matched text, or ``(-1, -1)`` if no match is found. Note that
            ``end - start`` may exceed ``len(line_text)`` because the doc
            text may contain extra whitespace inside the matched span.
        """
        line_text = line_text.strip()
        if not line_text:
            return -1, -1

        # Fast path: exact match works when no normalization applied.
        pos = doc_text.find(line_text, start_pos)
        if pos >= 0:
            return pos, pos + len(line_text)

        pattern_parts: list[str] = []
        prev_was_space = True
        for ch in line_text:
            if ch.isspace():
                if not prev_was_space:
                    pattern_parts.append(r"\s+")
                prev_was_space = True
                continue
            if ch in "ji":
                pattern_parts.append("[ji]")
            elif ch in "JI":
                pattern_parts.append("[JI]")
            elif ch in "vu":
                pattern_parts.append("[vu]")
            elif ch in "VU":
                pattern_parts.append("[VU]")
            elif ch.isalnum():
                pattern_parts.append(re.escape(ch))
            else:
                # Tokenization may insert whitespace before punctuation.
                pattern_parts.append(r"\s*" + re.escape(ch))
            prev_was_space = False

        pattern = "".join(pattern_parts)
        match = re.search(pattern, doc_text[start_pos:])
        if match:
            return start_pos + match.start(), start_pos + match.end()
        return -1, -1

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
                text = self._normalize_pre_markup(text)
                markup_data = _collect_markup(text)
                clean_text = self._strip_critical_markup(text)
                doc = nlp(clean_text)
                doc._.fileid = fileid
                doc._.metadata = metadata

                self._add_citation_spans(doc, metadata)
                self._apply_textcrit(doc, markup_data)

                yield doc

    def _apply_textcrit(self, doc: "Doc", markup_data: list[tuple]) -> None:
        """Populate doc._.textcrit and set per-token text-critical flags.

        Uses the markup positions collected before stripping to compute
        where each occurrence lands in the NLP-processed doc, then sets
        Token._.is_crux / is_addition / is_expansion accordingly.

        Deletions ({}) have no tokens in the Doc; they are recorded in
        doc._.textcrit["deletions"] without a span key.

        Args:
            doc: The spaCy Doc built from the stripped clean text.
            markup_data: Output of _collect_markup() — list of
                (source_start, source_end, type, original, replacement)
                sorted by source position.
        """
        textcrit: dict[str, list] = {
            "cruxes": [], "additions": [], "expansions": [], "deletions": [],
        }
        offset = 0

        for source_start, source_end, mtype, original, replacement in markup_data:
            clean_start = source_start - offset

            if mtype == "deletion":
                textcrit["deletions"].append({
                    "original": original,
                    "text": original[1:-1],  # strip enclosing { }
                })
            else:
                clean_end = clean_start + len(replacement)
                span = doc.char_span(clean_start, clean_end, alignment_mode="expand")
                token_span = (span.start, span.end) if span else None
                entry = {"original": original, "text": replacement, "span": token_span}

                if mtype == "crux":
                    textcrit["cruxes"].append(entry)
                    if span:
                        for token in span:
                            token._.is_crux = True
                elif mtype == "addition":
                    textcrit["additions"].append(entry)
                    if span:
                        for token in span:
                            token._.is_addition = True
                elif mtype == "expansion":
                    textcrit["expansions"].append(entry)
                    if span:
                        for token in span:
                            token._.is_expansion = True

            offset += len(original) - len(replacement)

        doc._.textcrit = textcrit

    def _add_citation_spans(self, doc: "Doc", metadata: dict) -> None:
        """Add section and line spans with citation info to the Doc."""
        from spacy.tokens import Span

        sections_data = metadata.get("sections", [])
        if not sections_data:
            return

        section_spans = []
        line_spans = []
        # Parallel list of (citation, metadata) for re-applying after section
        # spans are created. spaCy stores span extension values keyed by
        # (start, end) token range, not Python object identity. When a line
        # span and a section span share the same token range (single-line
        # sections), setting the section citation overwrites the line citation.
        # Re-applying line citations last ensures they win.
        line_span_annotations: list[tuple] = []

        # Track character position through the document
        char_pos = 0

        for section_info in sections_data:
            section_id = section_info["id"]
            section_title = section_info.get("title")
            section_start_char = char_pos

            for line_info in section_info["lines"]:
                line_text = line_info["text"]
                line_num = line_info["number"]

                # Strip blockquote markers and text-critical markup before
                # searching in normalized doc text, which has already had
                # these stripped by _normalize_text.
                line_text_stripped = self._strip_blockquote_marker(line_text)
                line_text_stripped = self._strip_critical_markup(line_text_stripped)
                if not line_text_stripped:
                    line_text_stripped = line_text.strip()
                line_start, line_end = self._find_line_in_doc_text(
                    doc.text, line_text_stripped, char_pos
                )

                if line_start >= 0:
                    span = doc.char_span(line_start, line_end, alignment_mode="expand")

                    if span:
                        citation = f"{section_id}.{line_num}"
                        line_meta = {
                            "section_id": section_id,
                            "section_title": section_title,
                            "line_number": line_num,
                        }
                        span._.citation = citation
                        span._.metadata = line_meta
                        line_spans.append(span)
                        line_span_annotations.append((span, citation, line_meta))

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

        # Re-apply line span citations after all section spans have been set,
        # so line citations win over any section citation that may share the
        # same token range (single-line sections).
        for span, citation, span_meta in line_span_annotations:
            span._.citation = citation
            span._.metadata = span_meta

        doc.spans["sections"] = section_spans
        doc.spans["lines"] = line_spans
        mark_newlines_from_spans(doc)

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
