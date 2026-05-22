"""Formulae-Litterae-Chartae corpus reader.

Reader for TEI-XML files from the Formulae-Litterae-Chartae project
(University of Hamburg), which provides open-access early medieval Latin
charters and formularies (500–1000 CE) under CC-BY 4.0.

Data repository: https://github.com/Formulae-Litterae-Chartae/formulae-open
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from lxml import etree

from latincyreaders.core.base import AnnotationLevel
from latincyreaders.readers.tei import TEIReader

if TYPE_CHECKING:
    pass


class FormulaeReader(TEIReader):
    """Reader for Formulae-Litterae-Chartae TEI-XML files.

    Handles the charter/formulary structure used across the corpus:
    ``<text><front>`` (French regest, excluded) and
    ``<body><div type='edition' xml:lang='lat'>`` (Latin text, extracted).

    Words are tokenized as ``<w>`` elements in the source; the reader joins
    them into running prose.  ``lemmaRef`` attributes are silently ignored.

    Each ``docs()`` / ``texts()`` call yields one item per file.  File-level
    metadata (CTS URN, collection, title, date) is attached to every Doc.

    Example:
        >>> reader = FormulaeReader("/path/to/formulae-open/data")
        >>> for doc in reader.docs():
        ...     print(doc._.metadata["cts_urn"], doc.text[:80])
    """

    @classmethod
    def _default_file_pattern(cls) -> str:
        # Match *.lat*.xml (e.g. .lat001.xml) and skip __capitains__.xml
        return "**/*.lat*.xml"

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        remove_notes: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
        **kwargs,
    ):
        """Initialize the Formulae reader.

        Args:
            root: Root directory containing Formulae XML files.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            remove_notes: Whether to remove ``<note>`` elements from text.
            cache: If True (default), cache processed Doc objects.
            cache_maxsize: Maximum number of documents to cache.
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

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _extract_title(self, root: etree._Element) -> str | None:
        """Extract title from teiHeader/titleStmt/title."""
        for xpath in [".//tei:titleStmt/tei:title", ".//titleStmt/title"]:
            try:
                elems = root.xpath(xpath, namespaces=self.TEI_NS)
                if elems:
                    text = "".join(elems[0].itertext()).strip()
                    if text:
                        return text
            except Exception:
                pass
        return None

    def _extract_cts_urn(self, root: etree._Element) -> str | None:
        """Extract CTS URN from ``<div type='edition' @n>`` in body."""
        for xpath in [".//tei:div[@type='edition']", ".//div[@type='edition']"]:
            try:
                elems = root.xpath(xpath, namespaces=self.TEI_NS)
                if elems:
                    return elems[0].get("n")
            except Exception:
                pass
        return None

    def _extract_collection(self, cts_urn: str | None) -> str | None:
        """Extract collection name from CTS URN.

        ``urn:cts:formulae:redon.courson0001.lat001`` → ``"redon"``
        """
        if not cts_urn:
            return None
        parts = cts_urn.split(":")
        if len(parts) >= 4:
            work_id = parts[3]          # e.g. "redon.courson0001.lat001"
            return work_id.split(".")[0]
        return None

    def _extract_date(self, root: etree._Element) -> str | None:
        """Extract date from ``<front>/<dateline>``.

        Prefers ``<date @when>``, then ``<date @notBefore>``, then the
        text content of ``<dateline>``.
        """
        for xpath in [".//tei:front//tei:dateline", ".//front//dateline"]:
            try:
                elems = root.xpath(xpath, namespaces=self.TEI_NS)
                if not elems:
                    continue
                dateline = elems[0]
                # Look for a <date> element with attributes
                for date_xpath in [".//tei:date", ".//date"]:
                    try:
                        dates = dateline.xpath(date_xpath, namespaces=self.TEI_NS)
                        if dates:
                            date_elem = dates[0]
                            for attr in ("when", "notBefore", "notAfter"):
                                val = date_elem.get(attr)
                                if val:
                                    # Strip leading zeros from year: "0832" → "832"
                                    return str(int(val.split("-")[0]))
                    except Exception:
                        pass
                # Fall back to text content
                text = "".join(dateline.itertext()).strip()
                if text:
                    return text
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_edition_text(self, root: etree._Element) -> str:
        """Extract Latin text from ``<div type='edition' xml:lang='lat'>``.

        Joins ``<w>`` element text nodes preserving original spacing.
        ``<seg>``, ``<hi>``, and other inline elements are traversed
        transparently via ``itertext()``.

        Args:
            root: XML root element.

        Returns:
            Joined text string, or empty string if no edition div found.
        """
        for xpath in [
            ".//tei:body//tei:div[@type='edition'][@xml:lang='lat']",
            ".//tei:body//tei:div[@type='edition']",
            ".//body//div[@type='edition']",
        ]:
            try:
                elems = root.xpath(xpath, namespaces=self.TEI_NS)
                if elems:
                    return "".join(elems[0].itertext())
            except Exception:
                pass
        return ""

    # ------------------------------------------------------------------
    # Core parsing
    # ------------------------------------------------------------------

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a Formulae TEI file, yielding one (text, metadata) per file.

        Extracts Latin text from the edition div, ignoring the French
        ``<front>`` regest.  Metadata includes CTS URN, collection,
        title, date, filename.

        Args:
            path: Path to XML file.

        Yields:
            Single ``(text, metadata)`` tuple per file.
        """
        root = self._parse_xml(path)
        if root is None:
            return

        text_raw = self._extract_edition_text(root)
        text = self._normalize_text(text_raw)
        if not text.strip():
            return

        cts_urn = self._extract_cts_urn(root)
        metadata: dict = {
            "filename": path.name,
            "path": str(path),
            "cts_urn": cts_urn,
            "collection": self._extract_collection(cts_urn),
            "title": self._extract_title(root),
            "date": self._extract_date(root),
        }

        yield text, metadata

    # ------------------------------------------------------------------
    # Header iteration
    # ------------------------------------------------------------------

    def headers(
        self, fileids: str | list[str] | None = None
    ) -> Iterator[dict]:
        """Yield metadata dicts from Formulae TEI headers.

        Extracts ``cts_urn``, ``collection``, ``title``, ``date``,
        and ``filename`` without NLP processing.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Metadata dictionaries.
        """
        for path in self._iter_paths(fileids):
            root = self._parse_xml(path)
            if root is None:
                continue

            cts_urn = self._extract_cts_urn(root)
            yield {
                "filename": path.name,
                "cts_urn": cts_urn,
                "collection": self._extract_collection(cts_urn),
                "title": self._extract_title(root),
                "date": self._extract_date(root),
            }
