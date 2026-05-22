"""Corpus Scriptorum Ecclesiasticorum Latinorum (CSEL) corpus reader.

Reader for TEI-XML files from the CSEL digital edition published by the
Open Greek and Latin Project (https://github.com/OpenGreekAndLatin/csel-dev).

Licensed CC-BY-SA 4.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from lxml import etree

from latincyreaders.core.base import AnnotationLevel
from latincyreaders.readers.digilibt import DigilibLTReader, _Chapter

if TYPE_CHECKING:
    pass

# xml:lang namespace qualifier
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


class CSELReader(DigilibLTReader):
    """Reader for CSEL (Corpus Scriptorum Ecclesiasticorum Latinorum) TEI-XML files.

    Handles the two-level textpart hierarchy used by CSEL:
    ``<div type="edition">`` → ``<div subtype="book" type="textpart">``
    → ``<div subtype="section" type="textpart">`` → ``<p>``.

    Chapter-level structure is preserved as named spans in
    ``doc.spans["chapters"]``, accessible via :meth:`chapters`.
    Citations follow the form ``"book 1, section 3"``.

    Critical marks (``<supplied>``, ``<del>``, etc.) are normalized by
    default via the inherited ``use_symbols=True`` pipeline.

    Example:
        >>> reader = CSELReader("/path/to/csel-dev/data")
        >>> for doc in reader.docs():
        ...     print(doc._.metadata["author"], doc._.metadata["cts_urn"])
        ...     for ch in doc.spans["chapters"]:
        ...         print(f"  {ch._.citation}: {ch.text[:60]}...")
    """

    @classmethod
    def _default_file_pattern(cls) -> str:
        return "**/*.opp-lat1.xml"

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
        """Initialize the CSEL reader.

        Args:
            root: Root directory containing CSEL XML files.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            remove_notes: Whether to remove ``<note>`` elements from text.
            use_symbols: If True (default), strip text-critical marks.
            cache: If True (default), cache processed Doc objects for reuse.
            cache_maxsize: Maximum number of documents to cache.
            **kwargs: Additional arguments passed to TEIReader.
        """
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            annotation_level=annotation_level,
            remove_notes=remove_notes,
            use_symbols=use_symbols,
            cache=cache,
            cache_maxsize=cache_maxsize,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def _extract_author(self, header: etree._Element) -> str | None:
        """Extract author from plain ``<author>`` in titleStmt.

        Args:
            header: teiHeader element.

        Returns:
            Author name string or None.
        """
        author_elem = self._find_with_ns(header, ".//author")
        if author_elem is not None:
            text = "".join(author_elem.itertext()).strip()
            if text:
                return text
        return None

    def _extract_title(self, header: etree._Element) -> str | None:
        """Extract title, preferring ``<title xml:lang='lat'>``.

        Args:
            header: teiHeader element.

        Returns:
            Title string or None.
        """
        for xpath in [
            ".//tei:titleStmt/tei:title[@xml:lang='lat']",
            ".//titleStmt/title[@xml:lang='lat']",
            ".//tei:titleStmt/tei:title",
            ".//titleStmt/title",
        ]:
            try:
                elems = header.xpath(xpath, namespaces=self.TEI_NS)
                if elems:
                    text = "".join(elems[0].itertext()).strip()
                    if text:
                        return text
            except Exception:
                pass
        return None

    def _extract_cts_urn(self, root: etree._Element) -> str | None:
        """Extract CTS URN from ``<div type='edition' @n>``.

        Args:
            root: XML root element.

        Returns:
            CTS URN string (e.g. ``urn:cts:latinLit:stoa0040.stoa001.opp-lat1``)
            or None.
        """
        for xpath in [
            ".//tei:div[@type='edition']",
            ".//div[@type='edition']",
        ]:
            try:
                elems = root.xpath(xpath, namespaces=self.TEI_NS)
                if elems:
                    return elems[0].get("n")
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Citation building
    # ------------------------------------------------------------------

    def _div_citation(self, div: etree._Element) -> str:
        """Build a citation string using ``subtype`` labels (book/section).

        Walks the ancestor chain, collecting ``subtype + n`` pairs and
        skipping the ``<div type='edition'>`` wrapper.

        Args:
            div: A leaf div element.

        Returns:
            Citation string like ``"book 1, section 3"``.
        """
        parts: list[str] = []
        elem = div
        while elem is not None:
            tag = etree.QName(elem.tag).localname if isinstance(elem.tag, str) else ""
            if tag == "div":
                div_type = elem.get("type", "")
                if div_type == "edition":
                    pass  # skip the URN-bearing wrapper
                else:
                    subtype = elem.get("subtype") or div_type
                    div_n = elem.get("n")
                    if subtype and div_n:
                        parts.append(f"{subtype} {div_n}")
            parent = elem.getparent()
            if parent is not None:
                ptag = etree.QName(parent.tag).localname if isinstance(parent.tag, str) else ""
                if ptag in ("body", "text"):
                    break
            elem = parent
        parts.reverse()
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Core parsing
    # ------------------------------------------------------------------

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a CSEL TEI file into text with metadata.

        Yields one (text, metadata) tuple per file, with
        ``_chapters`` stashed in metadata for span creation in
        :meth:`docs`.

        Args:
            path: Path to XML file.

        Yields:
            Single ``(text, metadata)`` tuple per file.
        """
        root = self._parse_xml(path)
        if root is None:
            return

        body = self._get_body(root)
        if body is None:
            return

        metadata: dict = {
            "filename": path.name,
            "path": str(path),
        }

        header = self._find_with_ns(root, ".//teiHeader")
        if header is not None:
            title = self._extract_title(header)
            if title:
                metadata["title"] = title
            author = self._extract_author(header)
            if author:
                metadata["author"] = author

        cts_urn = self._extract_cts_urn(root)
        if cts_urn:
            metadata["cts_urn"] = cts_urn

        chapters = self._parse_chapters(body)

        if chapters:
            for ch in chapters:
                ch.text = self._normalize_text(ch.text)
            chapters = [ch for ch in chapters if ch.text.strip()]
            text = "\n\n".join(ch.text for ch in chapters)
            metadata["_chapters"] = chapters
        else:
            paragraphs = self._extract_paragraphs(body)
            if not paragraphs:
                return
            text = "\n\n".join(
                self._normalize_text(p) for p in paragraphs
            )

        if not text.strip():
            return

        yield text, metadata

    # ------------------------------------------------------------------
    # Header iteration
    # ------------------------------------------------------------------

    def headers(
        self, fileids: str | list[str] | None = None
    ) -> Iterator[dict]:
        """Yield metadata dicts from CSEL TEI headers.

        Extracts ``author``, ``title``, ``cts_urn``, and ``filename``.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Metadata dictionaries.
        """
        for path in self._iter_paths(fileids):
            root = self._parse_xml(path)
            if root is None:
                continue

            meta: dict = {"filename": path.name}

            header = self._find_with_ns(root, ".//teiHeader")
            if header is not None:
                title = self._extract_title(header)
                if title:
                    meta["title"] = title
                author = self._extract_author(header)
                if author:
                    meta["author"] = author

            cts_urn = self._extract_cts_urn(root)
            if cts_urn:
                meta["cts_urn"] = cts_urn

            yield meta
