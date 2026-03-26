"""DigilibLT corpus reader.

Reader for TEI-XML files from digilibLT (Digital Library of Late-Antique
Latin Texts). Handles the various structural patterns used across the
collection: flat paragraphs, chapter divs, nested book/chapter divs,
and verse line groups.

License: digilibLT texts are CC BY-NC-SA (3.0 IT on the site, 4.0 on GitHub).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from lxml import etree

from latincyreaders.core.base import AnnotationLevel
from latincyreaders.readers.tei import TEIReader

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span


@dataclass
class _Chapter:
    """Internal representation of a chapter division."""

    citation: str
    text: str
    head: str | None = None


# Patterns for text-critical symbols found in Latin critical editions.
_CRITICAL_MARK_PATTERNS = [
    (re.compile(r"(\w)\((\w+)\)"), r"\1\2"),   # M(arcus) → Marcus (requires preceding letter)
    (re.compile(r"<([^>]+)>"), r"\1"),          # <supplied> → supplied
    (re.compile(r"\[[^\]]+\]"), ""),            # [secluded text] → removed
    (re.compile(r"\{([^}]+)\}"), r"\1"),        # {corrected} → corrected
    (re.compile(r"†([^†]*)†"), r"\1"),          # †corrupt† → corrupt
    (re.compile(r"†"), ""),                     # stray daggers
    (re.compile(r"\*\s*\*\s*\*"), ""),          # *** lacuna markers
]


class DigilibLTReader(TEIReader):
    """Reader for digilibLT TEI-XML files.

    Handles the structural patterns found across the digilibLT collection:

    - Flat ``<div type="cap">`` chapters directly in body
    - Nested divs (e.g., ``<div type="lib">`` → ``<div type="cap">``)
    - Single ``<div type="section">`` with ``<head>`` elements
    - Flat ``<p>`` paragraphs with no div structure
    - Verse ``<lg>/<l>`` elements within chapters

    Chapter-level structure is preserved as named spans in
    ``doc.spans["chapters"]``, accessible via the :meth:`chapters` method.

    When ``use_symbols=True`` (default), text-critical marks like ``<est>``,
    ``[sic]``, ``{correction}``, and ``†crux†`` are stripped before NLP
    processing, preserving the enclosed word.

    Example:
        >>> reader = DigilibLTReader("/path/to/digilibt/xml")
        >>> for doc in reader.docs():
        ...     print(doc._.metadata.get("dlt_id"))
        ...     for ch in doc.spans.get("chapters", []):
        ...         print(f"  {ch._.citation}: {ch.text[:60]}...")
    """

    @classmethod
    def _default_file_pattern(cls) -> str:
        return "**/*.xml"

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        remove_notes: bool = True,
        use_symbols: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
        **kwargs,
    ):
        """Initialize the digilibLT reader.

        Args:
            root: Root directory containing digilibLT XML files.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            remove_notes: Whether to remove ``<note>`` elements from text.
            use_symbols: If True (default), strip text-critical marks
                (``<>``, ``[]``, ``{}``, ``†``) before NLP processing,
                preserving the enclosed word.
            cache: If True (default), cache processed Doc objects for reuse.
            cache_maxsize: Maximum number of documents to cache (default 128).
            **kwargs: Additional arguments passed to TEIReader.
        """
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            annotation_level=annotation_level,
            remove_notes=remove_notes,
            cache=cache,
            cache_maxsize=cache_maxsize,
            **kwargs,
        )
        self._use_symbols = use_symbols

    def _normalize_text(self, text: str) -> str:
        """Normalize text, optionally stripping text-critical marks.

        When ``use_symbols`` is enabled, strips ``<>``, ``[]``, ``{}``,
        ``†`` marks while preserving the enclosed words, and removes
        lacuna markers (``***``). Collapses resulting extra whitespace.

        Args:
            text: Raw text string.

        Returns:
            Normalized text.
        """
        text = super()._normalize_text(text)
        if self._use_symbols:
            for pattern, replacement in _CRITICAL_MARK_PATTERNS:
                text = pattern.sub(replacement, text)
            # Collapse any double spaces left by removals
            text = re.sub(r"  +", " ", text).strip()
        return text

    def _extract_author(self, header: etree._Element) -> str | None:
        """Extract author name, preferring persName[@type='usualname'].

        Args:
            header: teiHeader element.

        Returns:
            Author name string or None.
        """
        # Try persName with usualname type first
        for xpath in [
            ".//author/persName[@type='usualname']",
            ".//tei:author/tei:persName[@type='usualname']",
        ]:
            try:
                elems = header.xpath(xpath, namespaces=self.TEI_NS)
                if elems and elems[0].text:
                    return elems[0].text.strip()
            except Exception:
                pass

        # Fall back to plain <author> text
        author_elem = self._find_with_ns(header, ".//author")
        if author_elem is not None:
            text = "".join(author_elem.itertext()).strip()
            if text:
                return text

        return None

    def _extract_dlt_id(self, root: etree._Element) -> str | None:
        """Extract DLT ID from ``<idno>`` in publicationStmt.

        Args:
            root: XML root element.

        Returns:
            DLT ID string (e.g., 'DLT000405') or None.
        """
        for xpath in [
            ".//tei:publicationStmt/tei:idno",
            ".//publicationStmt/idno",
        ]:
            try:
                elems = root.xpath(xpath, namespaces=self.TEI_NS)
                if elems and elems[0].text:
                    return elems[0].text.strip()
            except Exception:
                pass
        return None

    def _extract_source(self, root: etree._Element) -> str | None:
        """Extract source bibliography from sourceDesc.

        Args:
            root: XML root element.

        Returns:
            Source description string or None.
        """
        for xpath in [".//sourceDesc//bibl", ".//tei:sourceDesc//tei:bibl"]:
            try:
                elems = root.xpath(xpath, namespaces=self.TEI_NS)
                if elems:
                    text = "".join(elems[0].itertext()).strip()
                    if text:
                        return text
            except Exception:
                pass
        return None

    def _extract_creation_date(self, root: etree._Element) -> str | None:
        """Extract creation date from profileDesc/creation.

        Args:
            root: XML root element.

        Returns:
            Date string or None.
        """
        for xpath in [
            ".//tei:profileDesc/tei:creation/tei:date",
            ".//profileDesc/creation/date",
        ]:
            try:
                elems = root.xpath(xpath, namespaces=self.TEI_NS)
                if elems:
                    # Prefer @when, then text content
                    when = elems[0].get("when")
                    if when:
                        return when
                    text = elems[0].text
                    if text:
                        return text.strip()
            except Exception:
                pass
        return None

    def _extract_text_from_element(self, elem: etree._Element) -> str:
        """Extract text from element, handling <p>, <lg>/<l>, and mixed content.

        Joins paragraph texts with spaces. Joins verse lines with spaces.

        Args:
            elem: XML element to extract text from.

        Returns:
            Extracted text string.
        """
        parts: list[str] = []

        for child in elem:
            tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ""

            if tag == "p":
                text = " ".join(child.itertext()).strip()
                if text:
                    parts.append(text)
            elif tag == "lg":
                # Line group: extract each <l> and join with spaces
                lines = []
                for line_elem in child:
                    ltag = etree.QName(line_elem.tag).localname if isinstance(line_elem.tag, str) else ""
                    if ltag == "l":
                        line_text = " ".join(line_elem.itertext()).strip()
                        if line_text:
                            lines.append(line_text)
                if lines:
                    parts.append(" ".join(lines))
            elif tag == "l":
                # Standalone verse line outside <lg>
                text = " ".join(child.itertext()).strip()
                if text:
                    parts.append(text)

        return " ".join(parts)

    def _find_leaf_divs(self, body: etree._Element) -> list[etree._Element]:
        """Find leaf-level div elements (divs with no child divs).

        Args:
            body: Body element to search.

        Returns:
            List of leaf div elements, in document order.
        """
        all_divs: list[etree._Element] = []

        # Collect all divs (with and without namespace)
        for xpath in [".//tei:div", ".//div"]:
            try:
                found = body.xpath(xpath, namespaces=self.TEI_NS)
                if found:
                    all_divs = found
                    break
            except Exception:
                pass

        if not all_divs:
            return []

        # Keep only leaf divs (no child divs)
        leaves = []
        for div in all_divs:
            child_divs = []
            for child in div:
                tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ""
                if tag == "div":
                    child_divs.append(child)
            if not child_divs:
                leaves.append(div)

        return leaves

    def _div_citation(self, div: etree._Element) -> str:
        """Build a citation string for a div element.

        Walks up the ancestor chain to build hierarchical citations like
        ``lib. V, cap. 1``.

        Args:
            div: A leaf div element.

        Returns:
            Citation string.
        """
        parts: list[str] = []
        elem = div
        while elem is not None:
            tag = etree.QName(elem.tag).localname if isinstance(elem.tag, str) else ""
            if tag == "div":
                div_type = elem.get("type", "div")
                div_n = elem.get("n")
                if div_n is not None:
                    parts.append(f"{div_type}. {div_n}")
                else:
                    # Use head text if no n attribute
                    head = None
                    for child in elem:
                        ctag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ""
                        if ctag == "head":
                            head = " ".join(child.itertext()).strip()
                            break
                    if head:
                        parts.append(head)
                    else:
                        parts.append(div_type)
            parent = elem.getparent()
            if parent is not None:
                ptag = etree.QName(parent.tag).localname if isinstance(parent.tag, str) else ""
                if ptag in ("body", "text"):
                    break
            elem = parent

        parts.reverse()
        return ", ".join(parts)

    def _parse_chapters(self, body: etree._Element) -> list[_Chapter]:
        """Parse body into chapter divisions.

        Args:
            body: Body element.

        Returns:
            List of _Chapter objects. Empty if no div structure found.
        """
        leaves = self._find_leaf_divs(body)
        if not leaves:
            return []

        chapters = []
        for div in leaves:
            citation = self._div_citation(div)
            text = self._extract_text_from_element(div)
            if not text.strip():
                continue

            # Extract head if present
            head = None
            for child in div:
                ctag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ""
                if ctag == "head":
                    head = " ".join(child.itertext()).strip()
                    break

            chapters.append(_Chapter(citation=citation, text=text, head=head))

        return chapters

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a digilibLT TEI file into text with metadata.

        Extracts rich metadata (DLT ID, author via persName, source, dates)
        and stores chapter data for span creation in :meth:`docs`.

        Args:
            path: Path to XML file.

        Yields:
            Single (text, metadata) tuple per file.
        """
        root = self._parse_xml(path)
        if root is None:
            return

        body = self._get_body(root)
        if body is None:
            return

        # Build metadata
        metadata: dict = {
            "filename": path.name,
            "path": str(path),
        }

        header = self._find_with_ns(root, ".//teiHeader")
        if header is not None:
            title_elem = self._find_with_ns(header, ".//title")
            if title_elem is not None and title_elem.text:
                metadata["title"] = title_elem.text.strip()

            author = self._extract_author(header)
            if author:
                metadata["author"] = author

        dlt_id = self._extract_dlt_id(root)
        if dlt_id:
            metadata["dlt_id"] = dlt_id

        source = self._extract_source(root)
        if source:
            metadata["source"] = source

        creation_date = self._extract_creation_date(root)
        if creation_date:
            metadata["creation_date"] = creation_date

        # Parse chapters
        chapters = self._parse_chapters(body)

        if chapters:
            # Normalize each chapter text individually so that character
            # offsets in _make_chapter_spans stay aligned with the Doc text.
            for ch in chapters:
                ch.text = self._normalize_text(ch.text)
            chapters = [ch for ch in chapters if ch.text.strip()]
            text = "\n\n".join(ch.text for ch in chapters)
            metadata["_chapters"] = chapters
        else:
            # No div structure — fall back to paragraph extraction
            paragraphs = self._extract_paragraphs(body)
            if not paragraphs:
                return
            text = "\n\n".join(
                self._normalize_text(p) for p in paragraphs
            )

        if not text.strip():
            return

        yield text, metadata

    def _make_chapter_spans(
        self, doc: "Doc", chapters: list[_Chapter]
    ) -> list["Span"]:
        """Create Span objects for each chapter, aligned to the tokenized Doc.

        Args:
            doc: Processed spaCy Doc.
            chapters: List of chapter data from parsing.

        Returns:
            List of Spans with citation extensions set.
        """
        spans = []
        char_offset = 0

        for ch in chapters:
            start_char = char_offset
            end_char = char_offset + len(ch.text)
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is not None:
                span._.citation = ch.citation
                if ch.head:
                    span._.metadata = {"head": ch.head}
                spans.append(span)
            char_offset = end_char + 2  # +2 for "\n\n" separator

        return spans

    def docs(
        self, fileids: str | list[str] | None = None
    ) -> Iterator["Doc"]:
        """Yield spaCy Docs with chapter spans.

        Each Doc has ``doc.spans["chapters"]`` containing Span objects for
        each chapter division, with ``span._.citation`` set to the chapter
        identifier.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            spaCy Doc objects with chapter spans and rich metadata.
        """
        for doc in super().docs(fileids):
            chapters_data = (doc._.metadata or {}).get("_chapters", [])
            if chapters_data:
                doc.spans["chapters"] = self._make_chapter_spans(doc, chapters_data)
                # Clean private key from metadata
                metadata = dict(doc._.metadata)
                del metadata["_chapters"]
                doc._.metadata = metadata
            yield doc

    def chapters(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Span | str"]:
        """Yield chapter divisions as Spans or text strings.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield ``(citation, text)`` tuples instead of Spans.

        Yields:
            Chapter Spans with ``_.citation`` set, or ``(citation, text)`` tuples
            if ``as_text=True``.
        """
        if as_text:
            for path in self._iter_paths(fileids):
                root = self._parse_xml(path)
                if root is None:
                    continue
                body = self._get_body(root)
                if body is None:
                    continue
                for ch in self._parse_chapters(body):
                    yield ch.citation, self._normalize_text(ch.text)
        else:
            for doc in self.docs(fileids):
                yield from doc.spans.get("chapters", [])

    def headers(
        self, fileids: str | list[str] | None = None
    ) -> Iterator[dict]:
        """Yield rich metadata dicts from TEI headers.

        Extracts title, author (via ``persName[@type='usualname']``), DLT ID,
        source bibliography, and creation date.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Metadata dictionaries.
        """
        for path in self._iter_paths(fileids):
            root = self._parse_xml(path)
            if root is None:
                continue

            metadata: dict = {"filename": path.name}

            header = self._find_with_ns(root, ".//teiHeader")
            if header is not None:
                title_elem = self._find_with_ns(header, ".//title")
                if title_elem is not None and title_elem.text:
                    metadata["title"] = title_elem.text.strip()

                author = self._extract_author(header)
                if author:
                    metadata["author"] = author

            dlt_id = self._extract_dlt_id(root)
            if dlt_id:
                metadata["dlt_id"] = dlt_id

            source = self._extract_source(root)
            if source:
                metadata["source"] = source

            creation_date = self._extract_creation_date(root)
            if creation_date:
                metadata["creation_date"] = creation_date

            yield metadata
