"""Patristic Text Archive (PTA) corpus reader.

Reader for TEI-XML files from the Patristic Text Archive (https://pta.bbaw.de),
a Berlin-Brandenburg Academy project providing open-access ancient Christian
texts in Greek and Latin under CC-BY 4.0.

Data repository: https://github.com/PatristicTextArchive/pta_data
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from lxml import etree

from latincyreaders.core.base import AnnotationLevel
from latincyreaders.core.download import DownloadableCorpusMixin
from latincyreaders.readers.tei import TEIReader

if TYPE_CHECKING:
    pass

# XML namespace for xml:lang attribute
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


class PTAReader(DownloadableCorpusMixin, TEIReader):
    """Reader for Patristic Text Archive (PTA) TEI-XML files.

    Handles PTA-specific TEI structure: CTS URN identifiers, per-section
    chunking, language detection (Greek/Latin), and teiHeader metadata.

    Each ``docs()`` / ``texts()`` call yields one item per ``<div type='textpart'>``
    section. File-level metadata (CTS URN, language, author, title) is attached
    to every chunk; per-chunk metadata (div_type, div_n, citation) identifies
    the specific section.

    When ``root`` is omitted, the reader auto-downloads the PTA data repository
    from GitHub into ``~/latincy_data/pta_data`` (or the path in ``$PTA_PATH``).

    Example:
        >>> reader = PTAReader("/path/to/pta_data/data")
        >>> for doc in reader.docs(fileids="*lat*.xml"):
        ...     print(doc._.metadata["urn"], doc._.metadata["citation"])
    """

    CORPUS_URL = "https://github.com/PatristicTextArchive/pta_data.git"
    ENV_VAR = "PTA_PATH"
    DEFAULT_SUBDIR = "pta_data"       # git clone target under ~/latincy_data/
    _DATA_SUBDIR = "data"             # subdirectory inside the clone that has texts
    _FILE_CHECK_PATTERN = "**/*.xml"

    @classmethod
    def default_root(cls) -> Path:
        """Return the default reader root (the ``data/`` subdirectory of the clone).

        Checks ``$PTA_PATH`` first, then ``~/latincy_data/pta_data/data``.
        """
        import os
        from latincyreaders.core.download import LATINCY_DATA

        if env_path := os.environ.get(cls.ENV_VAR):
            return Path(env_path)
        return LATINCY_DATA / cls.DEFAULT_SUBDIR / cls._DATA_SUBDIR

    @classmethod
    def _get_default_root(cls, auto_download: bool = True) -> Path:
        """Return the reader root, cloning the repo first if necessary."""
        import os
        from latincyreaders.core.download import LATINCY_DATA
        import subprocess

        if env_path := os.environ.get(cls.ENV_VAR):
            data_root = Path(env_path)
        else:
            data_root = LATINCY_DATA / cls.DEFAULT_SUBDIR / cls._DATA_SUBDIR

        if data_root.exists() and any(data_root.glob(cls._FILE_CHECK_PATTERN)):
            return data_root

        if not auto_download:
            raise FileNotFoundError(
                f"PTA corpus not found at {data_root}. "
                f"Set $PTA_PATH to the pta_data/data/ directory, pass root= "
                f"explicitly, or set auto_download=True."
            )

        clone_target = data_root.parent  # ~/latincy_data/pta_data
        print(f"PTAReader corpus not found at {clone_target}")
        response = input("Download from GitHub (~400 MB)? [y/N]: ").strip().lower()

        if response in ("y", "yes"):
            clone_target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Cloning {cls.CORPUS_URL} to {clone_target}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", cls.CORPUS_URL, str(clone_target)],
                check=True,
            )
            return data_root
        else:
            raise FileNotFoundError(
                f"PTA corpus not found. Download manually:\n"
                f"  git clone --depth 1 {cls.CORPUS_URL} {clone_target}"
            )

    @classmethod
    def _default_file_pattern(cls) -> str:
        # Match pta-*.xml files (text files) and skip __cts__.xml metadata files
        return "**/*.pta-*.xml"

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        remove_notes: bool = True,
        cache: bool = False,
        cache_maxsize: int = 128,
        auto_download: bool = False,
        **kwargs,
    ):
        """Initialize the PTA reader.

        Args:
            root: Root directory containing PTA XML files. If None, uses the
                default location (``~/latincy_data/pta_data``), downloading
                automatically when ``auto_download=True``.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            remove_notes: Whether to remove ``<note>`` elements from text.
            cache: If False (default), disable LRU caching so that per-section
                chunking works correctly across repeated iterations.
            cache_maxsize: Maximum number of documents to cache.
            auto_download: If True and root is None, auto-download corpus.
            **kwargs: Additional arguments passed to TEIReader.
        """
        if root is None:
            root = self._get_default_root(auto_download)
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
    # Metadata extraction helpers
    # ------------------------------------------------------------------

    def _extract_pta_title(self, root: etree._Element) -> str | None:
        """Extract the work title from teiHeader/titleStmt/title."""
        header = self._find_with_ns(root, ".//teiHeader")
        if header is None:
            return None
        for xpath in [
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
        elem = self._find_with_ns(header, ".//title")
        if elem is not None:
            text = "".join(elem.itertext()).strip()
            if text:
                return text
        return None

    def _extract_pta_author(self, root: etree._Element) -> str | None:
        """Extract author name, preferring author/persName over plain author text."""
        header = self._find_with_ns(root, ".//teiHeader")
        if header is None:
            return None
        for xpath in [
            ".//tei:titleStmt/tei:author/tei:persName",
            ".//titleStmt/author/persName",
        ]:
            try:
                elems = header.xpath(xpath, namespaces=self.TEI_NS)
                if elems:
                    text = "".join(elems[0].itertext()).strip()
                    if text:
                        return text
            except Exception:
                pass
        author_elem = self._find_with_ns(header, ".//author")
        if author_elem is not None:
            text = "".join(author_elem.itertext()).strip()
            if text:
                return text
        return None

    # ------------------------------------------------------------------
    # Body structure helpers
    # ------------------------------------------------------------------

    def _find_body_div(self, body: etree._Element) -> etree._Element | None:
        """Return the first direct child <div> of <body> (the content wrapper)."""
        for child in body:
            if not isinstance(child.tag, str):
                continue
            if etree.QName(child.tag).localname == "div":
                return child
        return None

    def _find_textpart_divs(
        self, body_div: etree._Element
    ) -> list[etree._Element]:
        """Return all direct child <div type='textpart'> elements."""
        result = []
        for child in body_div:
            if not isinstance(child.tag, str):
                continue
            if etree.QName(child.tag).localname != "div":
                continue
            if child.get("type") == "textpart":
                result.append(child)
        return result

    def _extract_div_text(self, div: etree._Element) -> str:
        """Extract plain text from a div by joining all text nodes."""
        parts = [t.strip() for t in div.itertext() if t.strip()]
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Core parsing
    # ------------------------------------------------------------------

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a PTA TEI file, yielding one (text, metadata) per section.

        Each ``<div type='textpart'>`` becomes a separate chunk. When no
        textpart structure exists, the whole body div is one chunk.

        Args:
            path: Path to the XML file.

        Yields:
            ``(text, metadata)`` tuples with per-section metadata.
        """
        root = self._parse_xml(path)
        if root is None:
            return

        body = self._get_body(root)
        if body is None:
            return

        # Determine file-level metadata from the top-level body div
        urn: str | None = None
        language: str | None = None
        body_div = self._find_body_div(body)
        if body_div is not None:
            urn = body_div.get("n")
            language = body_div.get(_XML_LANG)

        # Fallback: infer language from filename (-lat1, -grc1, etc.)
        if not language:
            stem = path.stem
            if "-lat" in stem:
                language = "lat"
            elif "-grc" in stem:
                language = "grc"

        author = self._extract_pta_author(root)
        title = self._extract_pta_title(root)

        base_meta: dict = {
            "filename": path.name,
            "path": str(path),
            "urn": urn,
            "language": language,
            "author": author,
            "title": title,
        }

        if body_div is None:
            # No top-level div — fall back to paragraph extraction
            paragraphs = self._extract_paragraphs(body)
            text = self._normalize_text("\n\n".join(paragraphs))
            if text.strip():
                yield text, base_meta
            return

        textpart_divs = self._find_textpart_divs(body_div)

        if not textpart_divs:
            # No textpart structure: yield whole body div as one chunk
            text = self._normalize_text(self._extract_div_text(body_div))
            if text.strip():
                div_n = body_div.get("n") or ""
                yield text, {
                    **base_meta,
                    "div_type": body_div.get("type") or "",
                    "div_n": div_n,
                    "citation": div_n,
                }
            return

        for div in textpart_divs:
            text = self._normalize_text(self._extract_div_text(div))
            if not text.strip():
                continue
            div_n = div.get("n") or ""
            div_type = div.get("subtype") or div.get("type") or "section"
            yield text, {
                **base_meta,
                "div_type": div_type,
                "div_n": div_n,
                "citation": div_n,
            }
