"""CAMENA Neo-Latin corpus reader.

Reader for the CAMENA (Corpus Automatum Multiplex Electorum Neolatinitatis Auctorum)
collection of Neo-Latin texts from German-speaking countries.

Repository: https://github.com/nevenjovanovic/camena-neolatinlit
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from lxml import etree

from latincyreaders.core.base import AnnotationLevel
from latincyreaders.core.download import DownloadableCorpusMixin
from latincyreaders.readers.tei import TEIReader

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span


class CamenaCorpusReader(DownloadableCorpusMixin, TEIReader):
    """Reader for CAMENA Neo-Latin corpus.

    The CAMENA corpus contains Neo-Latin texts organized into collections:
    - POEMATA: Neo-Latin poetry by German authors
    - HISTORICA & POLITICA: Historical and political texts
    - THESAURUS ERUDITIONIS: Reference materials (dictionaries, handbooks)
    - CERA: Printed Latin letters from German scholars

    If no root path is provided, looks for the corpus in:
    1. The path specified by CAMENA_ROOT environment variable
    2. ~/latincy_data/camena-neolatinlit

    If the corpus is not found and auto_download=True (default), offers to
    download from GitHub.

    Example:
        >>> reader = CamenaCorpusReader()  # Uses default location
        >>> for doc in reader.docs():
        ...     print(f"{doc._.metadata.get('author')}: {doc._.metadata.get('title')}")

        >>> # Filter by collection
        >>> reader = CamenaCorpusReader(fileids="poemata/**/*.xml")
        >>> for doc in reader.docs():
        ...     print(doc.text[:100])

    Attributes:
        CORPUS_URL: GitHub URL for downloading the corpus.
        ENV_VAR: Environment variable for custom corpus path.
    """

    CORPUS_URL = "https://github.com/nevenjovanovic/camena-neolatinlit.git"
    ENV_VAR = "CAMENA_ROOT"
    DEFAULT_SUBDIR = "camena-neolatinlit"
    _FILE_CHECK_PATTERN = "**/*.xml"

    # Known collections in the corpus
    COLLECTIONS = ("poemata", "cera", "historica-et-politica", "thesaurus")

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        include_front: bool = True,
        remove_notes: bool = True,
        auto_download: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
        **kwargs,
    ):
        """Initialize the CAMENA reader.

        Args:
            root: Root directory. If None, uses CAMENA_ROOT env var or default.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            include_front: If True, include front matter (prefaces, dedications).
            remove_notes: Whether to remove <note> elements.
            auto_download: If True and corpus not found, prompt to download.
            cache: If True (default), cache processed Doc objects for reuse.
            cache_maxsize: Maximum number of documents to cache (default 128).
            **kwargs: Additional arguments passed to BaseCorpusReader (e.g., backend).
        """
        if root is None:
            root = self._get_default_root(auto_download)

        # CAMENA doesn't use namespaces consistently
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            annotation_level=annotation_level,
            namespaces=None,
            remove_notes=remove_notes,
            cache=cache,
            cache_maxsize=cache_maxsize,
            **kwargs,
        )
        self._include_front = include_front

    def _get_body(self, root_elem: etree._Element) -> etree._Element | None:
        """Extract TEI body and optionally front matter.

        Args:
            root_elem: XML root element.

        Returns:
            Body element (or combined front+body) with notes removed if configured.
        """
        text_elem = self._find_with_ns(root_elem, ".//text")
        if text_elem is None:
            return None

        if self._include_front:
            # Create a container for front + body content
            front = self._find_with_ns(text_elem, ".//front")
            body = self._find_with_ns(text_elem, ".//body")

            if front is not None or body is not None:
                # Return the text element which contains both
                if self._remove_notes:
                    self._remove_note_elements(text_elem)
                return text_elem
        else:
            # Just the body
            body = self._find_with_ns(text_elem, ".//body")
            if body is not None and self._remove_notes:
                self._remove_note_elements(body)
            return body

        return None

    def _remove_note_elements(self, element: etree._Element) -> None:
        """Remove note elements from the given element."""
        for notes_xpath in [".//note", ".//tei:note"]:
            try:
                for note in element.xpath(notes_xpath, namespaces=self.TEI_NS):
                    note.getparent().remove(note)
            except Exception:
                pass

    def _extract_text_units(self, body: etree._Element) -> list[str]:
        """Extract text units from body, handling both prose and poetry.

        CAMENA poetry uses <l> (line) and <lg> (line group) elements,
        while prose uses <p> (paragraph) elements.

        Args:
            body: TEI body/text element.

        Returns:
            List of text strings (paragraphs or stanzas).
        """
        units = []

        # First try paragraphs (prose)
        paras = self._findall_with_ns(body, ".//p")
        for p in paras:
            text = " ".join(p.itertext()).strip()
            if text:
                units.append(text)

        # Then try line groups (poetry stanzas)
        line_groups = self._findall_with_ns(body, ".//lg")
        for lg in line_groups:
            # Get all lines in this group
            lines = self._findall_with_ns(lg, ".//l")
            if lines:
                stanza_lines = []
                for line in lines:
                    line_text = " ".join(line.itertext()).strip()
                    if line_text:
                        stanza_lines.append(line_text)
                if stanza_lines:
                    # Join lines with space (they're verse lines, not sentences)
                    units.append(" ".join(stanza_lines))

        # If no structured content, try standalone lines
        if not units:
            lines = self._findall_with_ns(body, ".//l")
            for line in lines:
                text = " ".join(line.itertext()).strip()
                if text:
                    units.append(text)

        # Last resort: all text content
        if not units:
            text = " ".join(body.itertext()).strip()
            if text:
                units.append(text)

        return units

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a CAMENA TEI file into text with metadata.

        Args:
            path: Path to XML file.

        Yields:
            Single (text, metadata) tuple per file.
        """
        root_elem = self._parse_xml(path)
        if root_elem is None:
            return

        body = self._get_body(root_elem)
        if body is None:
            return

        # Use custom extraction that handles poetry <l> elements
        text_units = self._extract_text_units(body)
        if not text_units:
            return

        # Join units with double newlines
        text = "\n\n".join(text_units)
        text = self._normalize_text(text)

        if not text.strip():
            return

        # Extract metadata
        metadata = {
            "filename": path.name,
            "path": str(path),
        }

        # Determine collection from path
        for collection in self.COLLECTIONS:
            if collection in str(path).lower():
                metadata["collection"] = collection
                break

        # Extract from TEI header
        header = self._find_with_ns(root_elem, ".//teiHeader")
        if header is not None:
            self._extract_header_metadata(header, metadata)

        yield text, metadata

    def _extract_header_metadata(
        self, header: etree._Element, metadata: dict
    ) -> None:
        """Extract metadata from TEI header.

        Args:
            header: TEI header element.
            metadata: Dict to update with extracted metadata.
        """
        # Title
        title_elem = self._find_with_ns(header, ".//title")
        if title_elem is not None:
            title = " ".join(title_elem.itertext()).strip()
            if title:
                metadata["title"] = title

        # Author
        author_elem = self._find_with_ns(header, ".//author")
        if author_elem is not None:
            author = " ".join(author_elem.itertext()).strip()
            if author:
                metadata["author"] = author

        # Publication date (from sourceDesc or publicationStmt)
        date_elem = self._find_with_ns(header, ".//date")
        if date_elem is not None:
            date_text = date_elem.get("when") or " ".join(date_elem.itertext()).strip()
            if date_text:
                metadata["date"] = date_text

        # Publisher
        publisher_elem = self._find_with_ns(header, ".//publisher")
        if publisher_elem is not None:
            publisher = " ".join(publisher_elem.itertext()).strip()
            if publisher:
                metadata["publisher"] = publisher

    def collections(self) -> list[str]:
        """Return list of available collections.

        Returns:
            List of collection names found in the corpus.
        """
        found = set()
        for fileid in self.fileids():
            for collection in self.COLLECTIONS:
                if collection in fileid.lower():
                    found.add(collection)
                    break
        return sorted(found)

    def docs_by_collection(
        self,
        collection: str,
    ) -> Iterator["Doc"]:
        """Yield docs from a specific collection.

        Args:
            collection: Collection name (e.g., "poemata", "cera").

        Yields:
            spaCy Doc objects from the specified collection.
        """
        # Use regex pattern (fileids match uses re.compile)
        pattern = f".*{collection}.*"
        matching = self.fileids(match=pattern)
        yield from self.docs(matching)


# Alias for short form
CamenaReader = CamenaCorpusReader
