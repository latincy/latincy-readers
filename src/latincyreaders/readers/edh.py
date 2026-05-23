"""Epigraphic Database Heidelberg (EDH) corpus reader.

Reader for EpiDoc TEI-XML files from the Epigraphic Database Heidelberg.
Extracts Latin inscription text with Leiden-convention markup normalized
to NLP-ready plaintext. Each inscription is one Doc; inscription lines
are tracked as named spans (doc.spans["lines"]) with citation keys of
the form "HD000001.N".

Clone the EDH data repository from
https://github.com/epigraphic-database-heidelberg/data
into ~/latincy_data/edh-data (or set $EDH_PATH to the repo root).

Usage:
    >>> reader = EDHReader(root="/path/to/edh-data")
    >>> for doc in reader.docs():
    ...     print(doc._.metadata["hd_nr"], doc.text[:80])
    ...     for span in doc.spans["lines"]:
    ...         print(span._.citation, span.text)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from lxml import etree

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel
from latincyreaders.core.download import DownloadableCorpusMixin
from latincyreaders.nlp.pipeline import mark_newlines_from_spans
from latincyreaders.readers.tei import TEIReader
from latincyreaders.utils.text_utils import find_line_in_doc_text

if TYPE_CHECKING:
    from spacy.tokens import Doc

# TEI namespace constant (re-used from TEIReader)
_TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": _TEI_NS}

# Tags whose content should be dropped from NLP text
_SKIP_TAGS = frozenset({"gap", "del", "note", "figure", "bibl", "ref"})

# Tags that expand abbreviations: abbr+ex → combined word
_EXPAN_TAG = "expan"
_ABBR_TAG = "abbr"
_EX_TAG = "ex"

# Tags whose text content is included verbatim
_PASSTHROUGH_TAGS = frozenset({
    "unclear", "supplied", "hi", "num", "name", "persName",
    "placeName", "geogName", "orgName", "add",
})


def _localname(elem: etree._Element) -> str:
    """Return local tag name without namespace."""
    if isinstance(elem.tag, str):
        return etree.QName(elem.tag).localname
    return ""


def _elem_to_text(elem: etree._Element) -> str:
    """Recursively extract NLP text from an EpiDoc element.

    Handles Leiden conventions:
    - <expan><abbr>D</abbr><ex>is</ex></expan>  → "Dis"
    - <gap/>                                     → "" (lost, not restored)
    - <del>text</del>                            → "" (erased)
    - <supplied reason="lost">text</supplied>    → "text" (editor restoration)
    - <unclear>text</unclear>                    → "text" (uncertain reading)
    - plain text nodes                           → included directly
    """
    tag = _localname(elem)

    if tag in _SKIP_TAGS:
        return ""

    if tag == _EXPAN_TAG:
        abbr_text = ""
        ex_text = ""
        for child in elem:
            ctag = _localname(child)
            if ctag == _ABBR_TAG:
                # abbr text may itself contain markup
                abbr_text = (child.text or "") + "".join(
                    _elem_to_text(c) + (c.tail or "") for c in child
                )
            elif ctag == _EX_TAG:
                ex_text = child.text or ""
        return abbr_text + ex_text

    # For all other elements (including unknown markup), recurse
    result = elem.text or ""
    for child in elem:
        ctag = _localname(child)
        if ctag == "lb":
            # Nested <lb> — unusual; skip element but keep tail
            result += child.tail or ""
        elif ctag in _SKIP_TAGS:
            result += child.tail or ""
        else:
            result += _elem_to_text(child)
            result += child.tail or ""
    return result


def _strip_leading_zeros(year_str: str) -> str:
    """Convert '0071' → '71', '0130' → '130', '-0050' → '-50'."""
    if not year_str:
        return year_str
    negative = year_str.startswith("-")
    digits = year_str.lstrip("-")
    try:
        val = int(digits)
        return ("-" if negative else "") + str(val)
    except ValueError:
        return year_str


class EDHReader(DownloadableCorpusMixin, TEIReader):
    """Reader for Epigraphic Database Heidelberg (EDH) EpiDoc TEI-XML files.

    Each Latin inscription file yields one Doc. Abbreviations are expanded
    (D(is) → Dis), editor restorations are included, erasures are dropped.
    Line spans with citation keys (HD000001.N) are stored in
    ``doc.spans["lines"]``.

    Only files containing a ``<div type="edition" xml:lang="la">`` are
    processed; Greek-only inscriptions are silently skipped.

    Example:
        >>> reader = EDHReader(root="~/latincy_data/edh-data")
        >>> for doc in reader.docs():
        ...     print(doc._.metadata["hd_nr"], doc.text[:80])
    """

    CORPUS_URL = "https://github.com/epigraphic-database-heidelberg/data"
    ENV_VAR = "EDH_PATH"
    DEFAULT_SUBDIR = "edh-data"
    _FILE_CHECK_PATTERN = "inscriptions/**/*.xml"

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        auto_download: bool = False,
        cache: bool = True,
        cache_maxsize: int = 128,
        **kwargs,
    ):
        """Initialize EDHReader.

        Args:
            root: Path to EDH data repo root. If None, uses $EDH_PATH or
                ~/latincy_data/edh-data (downloading if auto_download=True).
            fileids: Glob pattern for files. Defaults to
                ``inscriptions/**/*.xml``.
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            auto_download: If True, prompt to clone from GitHub when not found.
            cache: Cache processed Docs.
            cache_maxsize: LRU cache size.
            **kwargs: Passed to TEIReader.
        """
        if root is None:
            root = self._get_default_root(auto_download)
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            annotation_level=annotation_level,
            remove_notes=False,  # EDH notes handled via _SKIP_TAGS
            cache=cache,
            cache_maxsize=cache_maxsize,
            **kwargs,
        )

    @classmethod
    def _default_file_pattern(cls) -> str:
        return "inscriptions/**/*.xml"

    # ------------------------------------------------------------------
    # Metadata extraction helpers
    # ------------------------------------------------------------------

    def _extract_hd_nr(self, root: etree._Element) -> str | None:
        """Extract HD number from <idno type="localID">."""
        xpaths = [
            ".//tei:teiHeader//tei:idno[@type='localID']",
            ".//teiHeader//idno[@type='localID']",
        ]
        for xpath in xpaths:
            try:
                elems = root.xpath(xpath, namespaces=_NS)
                if elems:
                    text = (elems[0].text or "").strip()
                    if text:
                        return text
            except Exception:
                pass
        return None

    def _extract_dates(self, root: etree._Element) -> tuple[str | None, str | None]:
        """Extract not_before / not_after from <origDate>."""
        xpaths = [
            ".//tei:history//tei:origDate",
            ".//history//origDate",
        ]
        for xpath in xpaths:
            try:
                elems = root.xpath(xpath, namespaces=_NS)
                if elems:
                    elem = elems[0]
                    nb = elem.get("notBefore-custom") or elem.get("notBefore")
                    na = elem.get("notAfter-custom") or elem.get("notAfter")
                    return (
                        _strip_leading_zeros(nb) if nb else None,
                        _strip_leading_zeros(na) if na else None,
                    )
            except Exception:
                pass
        return None, None

    def _extract_province(self, root: etree._Element) -> str | None:
        """Extract province from <placeName type="provinceItalicRegion">."""
        xpaths = [
            ".//tei:history//tei:placeName[@type='provinceItalicRegion']",
            ".//history//placeName[@type='provinceItalicRegion']",
        ]
        for xpath in xpaths:
            try:
                elems = root.xpath(xpath, namespaces=_NS)
                if elems:
                    text = "".join(elems[0].itertext()).strip()
                    if text:
                        return text
            except Exception:
                pass
        return None

    def _extract_type_of_inscription(self, root: etree._Element) -> str | None:
        """Extract inscription type from <term> in profileDesc."""
        xpaths = [
            ".//tei:profileDesc//tei:keywords//tei:term",
            ".//profileDesc//keywords//term",
        ]
        for xpath in xpaths:
            try:
                elems = root.xpath(xpath, namespaces=_NS)
                if elems:
                    text = (elems[0].text or "").strip()
                    if text:
                        return text
            except Exception:
                pass
        return None

    def _find_edition_div(self, root: etree._Element) -> etree._Element | None:
        """Return the first <div type='edition' xml:lang='la'> element."""
        xpaths = [
            ".//tei:body//tei:div[@type='edition'][@xml:lang='la']",
            ".//body//div[@type='edition'][@xml:lang='la']",
        ]
        for xpath in xpaths:
            try:
                elems = root.xpath(xpath, namespaces=_NS)
                if elems:
                    return elems[0]
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Leiden text extraction
    # ------------------------------------------------------------------

    def _extract_edition_lines(
        self, root: etree._Element
    ) -> list[tuple[int, str]]:
        """Extract inscription lines as (line_num, text) pairs.

        Processes the ``<div type='edition' xml:lang='la'><ab>`` block,
        tracking ``<lb n="N"/>`` as line boundaries. Abbreviations are
        expanded via ``<expan>``; erasures (``<del>``) are dropped.

        Returns:
            List of (line_number, line_text) pairs for non-empty lines.
        """
        edition_div = self._find_edition_div(root)
        if edition_div is None:
            return []

        # Find <ab> in edition div (with or without namespace)
        ab = edition_div.find(f"{{{_TEI_NS}}}ab")
        if ab is None:
            ab = edition_div.find("ab")
        if ab is None:
            return []

        lines: list[tuple[int, str]] = []
        current_num: int | None = None
        current_parts: list[str] = []

        def flush() -> None:
            if current_num is not None and current_parts:
                text = re.sub(r"\s+", " ", " ".join(current_parts)).strip()
                if text:
                    lines.append((current_num, text))
            current_parts.clear()

        def add(text: str) -> None:
            stripped = text.strip()
            if stripped:
                current_parts.append(stripped)

        # Text before the first child (usually just whitespace)
        add(ab.text or "")

        for child in ab:
            tag = _localname(child)

            if tag == "lb":
                flush()
                try:
                    current_num = int(child.get("n", 0))
                except (ValueError, TypeError):
                    current_num = (lines[-1][0] + 1) if lines else 1
                # tail of <lb> is the text on that line before the first child
                add(child.tail or "")

            elif tag in _SKIP_TAGS:
                # Skip element content; tail belongs to the current line
                add(child.tail or "")

            else:
                add(_elem_to_text(child))
                add(child.tail or "")

        flush()
        return lines

    # ------------------------------------------------------------------
    # _parse_file and headers
    # ------------------------------------------------------------------

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse an EDH EpiDoc XML file.

        Yields a single ``(text, metadata)`` tuple for Latin inscriptions.
        Files without a Latin edition div are silently skipped.
        """
        root = self._parse_xml(path)
        if root is None:
            return

        # Skip non-Latin inscriptions
        if self._find_edition_div(root) is None:
            return

        # Extract line-level text
        lines = self._extract_edition_lines(root)
        if not lines:
            return

        hd_nr = self._extract_hd_nr(root)
        not_before, not_after = self._extract_dates(root)
        province = self._extract_province(root)
        type_of_inscription = self._extract_type_of_inscription(root)

        metadata = {
            "filename": path.name,
            "path": str(path),
            "hd_nr": hd_nr,
            "not_before": not_before,
            "not_after": not_after,
            "province": province,
            "type_of_inscription": type_of_inscription,
            "lines": lines,
        }

        full_text = "\n".join(text for _, text in lines)
        yield full_text, metadata

    def headers(self, fileids: str | list[str] | None = None) -> Iterator[dict]:
        """Yield metadata dicts for Latin inscriptions (zero NLP overhead).

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Metadata dicts with keys: hd_nr, not_before, not_after,
            province, type_of_inscription, filename, path.
        """
        for path in self._iter_paths(fileids):
            root = self._parse_xml(path)
            if root is None:
                continue
            if self._find_edition_div(root) is None:
                continue

            hd_nr = self._extract_hd_nr(root)
            not_before, not_after = self._extract_dates(root)
            province = self._extract_province(root)
            type_of_inscription = self._extract_type_of_inscription(root)

            yield {
                "filename": path.name,
                "path": str(path),
                "hd_nr": hd_nr,
                "not_before": not_before,
                "not_after": not_after,
                "province": province,
                "type_of_inscription": type_of_inscription,
            }

    # ------------------------------------------------------------------
    # docs() with line spans
    # ------------------------------------------------------------------

    def docs(
        self,
        fileids: str | list[str] | None = None,
        annotation_level: AnnotationLevel | None = None,
        cache: bool = True,
    ) -> Iterator["Doc"]:
        """Yield spaCy Docs with inscription line spans.

        Each Doc has:
        - ``doc._.fileid``: relative file path
        - ``doc._.metadata``: inscription metadata + raw line data
        - ``doc.spans["lines"]``: line spans with ``span._.citation``
          set to ``"HDNNNNNN.N"``

        Args:
            fileids: Files to process, or None for all.
            annotation_level: Override default annotation level.
            cache: Cache processed Docs.

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

                self._add_line_spans(doc, metadata)

                yield doc

    def _add_line_spans(self, doc: "Doc", metadata: dict) -> None:
        """Populate ``doc.spans["lines"]`` from parsed line data."""
        hd_nr = metadata.get("hd_nr") or ""
        lines = metadata.get("lines", [])
        if not lines:
            return

        line_spans = []
        char_pos = 0

        for line_num, line_text in lines:
            start, end = find_line_in_doc_text(doc.text, line_text, char_pos)
            if start >= 0:
                span = doc.char_span(start, end, alignment_mode="expand")
                if span:
                    citation = f"{hd_nr}.{line_num}" if hd_nr else str(line_num)
                    span._.citation = citation
                    span._.metadata = {
                        "hd_nr": hd_nr,
                        "line_number": line_num,
                    }
                    line_spans.append(span)
                char_pos = end

        doc.spans["lines"] = line_spans
        mark_newlines_from_spans(doc)
