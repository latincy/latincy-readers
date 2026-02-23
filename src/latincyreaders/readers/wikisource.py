"""WikiSource corpus reader.

Reads Latin texts from local .wiki files (MediaWiki wikitext format)
downloaded from la.wikisource.org, with structured citation preservation.

Supports two content types:
- **Prose** (e.g., Seneca's De vita beata): section headers + numbered paragraphs
- **Verse** (e.g., Vergil's Aeneid): <poem> blocks + {{versus|N}} line numbers

Example:
    >>> reader = WikiSourceReader("path/to/wiki/files")
    >>> for doc in reader.docs():
    ...     print(doc._.fileid, doc._.metadata.get("author"))

    >>> # Download from la.wikisource.org
    >>> paths = WikiSourceReader.download("De_vita_beata", "~/latincy_data/wikisource")
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel

# Lazy import to avoid circular imports; used only in download()
def _get_version() -> str:
    from latincyreaders import __version__
    return __version__

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span


@dataclass
class WikiSection:
    """A section parsed from a WikiSource prose text.

    Attributes:
        header: Section header (e.g., "I.", "II.").
        text: Full cleaned section text.
        paragraphs: List of (paragraph_number, text) pairs.
    """

    header: str
    text: str
    paragraphs: list[tuple[int, str]] = field(default_factory=list)


class WikiSourceReader(BaseCorpusReader):
    """Reader for Latin texts from la.wikisource.org in MediaWiki wikitext format.

    Reads local .wiki files containing raw wikitext. Handles both prose
    (section headers + numbered paragraphs) and verse (<poem> blocks +
    {{versus|N}} templates).

    Citation information is preserved through spaCy custom extensions:
    - Prose: ``<section>.<paragraph>`` (e.g., ``I.1``, ``III.2``)
    - Verse: line numbers from ``{{versus|N}}`` (e.g., ``1``, ``15``)

    No auto-download from git — use the ``download()`` classmethod to fetch
    pages via the MediaWiki API.

    Args:
        root: Root directory containing .wiki files.
        fileids: Glob pattern for selecting files. Defaults to ``**/*.wiki``.
        encoding: Text encoding for reading files.
        annotation_level: How much NLP annotation to apply.
        cache: If True (default), cache processed Doc objects for reuse.
        cache_maxsize: Maximum number of documents to cache (default 128).
        model_name: Name of the spaCy model to load for BASIC/FULL levels.
        lang: Language code for blank model in TOKENIZE level.

    Example:
        >>> reader = WikiSourceReader("/path/to/wiki/files")
        >>> for doc in reader.docs():
        ...     for span in doc.spans.get("sections", []):
        ...         print(f"{span._.citation}: {span.text[:60]}...")
    """

    # Regex patterns for wikitext parsing
    _TITULUS_PATTERN = re.compile(
        r"\{\{titulus2\|([^}]+)\}\}", re.IGNORECASE
    )
    _SECTION_HEADER_PATTERN = re.compile(
        r"^==\s*([^=]+?)\s*==$", re.MULTILINE
    )
    _PARAGRAPH_NUM_PATTERN = re.compile(
        r"(?:^|\n)(\d+)\.\s+"
    )
    _VERSUS_PATTERN = re.compile(
        r"\{\{versus\|(\d+)\}\}"
    )
    _POEM_PATTERN = re.compile(
        r"<poem>(.*?)</poem>", re.DOTALL
    )
    _DIV_TEXT_PATTERN = re.compile(
        r'<div\s+class=["\']?text["\']?\s*>(.*?)</div>', re.DOTALL
    )

    # Patterns for markup to strip
    _FINIS_PATTERN = re.compile(r"\{\{finis\}\}", re.IGNORECASE)
    _TEXTQUALITY_PATTERN = re.compile(r"\{\{textquality\|[^}]*\}\}", re.IGNORECASE)
    _INTERWIKI_PATTERN = re.compile(r"\[\[[a-z]{2,3}:[^\]]+\]\]")
    _LIBER_PATTERN = re.compile(r"\{\{Liber\|[^}]*\}\}", re.IGNORECASE)
    _INTRAINCEPTI_PATTERN = re.compile(r"\{\{Intraincepti\|[^}]*\}\}", re.IGNORECASE)
    _IMAGO_PATTERN = re.compile(r"\[\[Imago:[^\]]*\]\]", re.IGNORECASE)
    _DIV_TAGS_PATTERN = re.compile(r"</?div[^>]*>", re.IGNORECASE)
    _BOLD_ITALIC_PATTERN = re.compile(r"'{2,5}")
    _WIKILINK_PATTERN = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
    _SIMPLE_WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]*)\]\]")

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        cache: bool = True,
        cache_maxsize: int = 128,
        model_name: str = "la_core_web_lg",
        lang: str = "la",
        **kwargs,
    ):
        """Initialize the WikiSource reader.

        Args:
            root: Root directory containing .wiki files.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: How much NLP annotation to apply.
            cache: If True (default), cache processed Doc objects for reuse.
            cache_maxsize: Maximum number of documents to cache (default 128).
            model_name: Name of the spaCy model to load for BASIC/FULL levels.
            lang: Language code for blank model in TOKENIZE level.
            **kwargs: Additional arguments passed to BaseCorpusReader (e.g., backend).
        """
        super().__init__(
            root, fileids, encoding, annotation_level,
            cache=cache, cache_maxsize=cache_maxsize,
            model_name=model_name, lang=lang,
            **kwargs,
        )

    @classmethod
    def _default_file_pattern(cls) -> str:
        """WikiSource files use .wiki extension."""
        return "**/*.wiki"

    # -------------------------------------------------------------------------
    # Metadata parsing
    # -------------------------------------------------------------------------

    def _parse_titulus(self, text: str) -> dict[str, Any]:
        """Extract metadata from ``{{titulus2}}`` template.

        Args:
            text: Raw wikitext content.

        Returns:
            Dict with keys like 'author', 'title', 'date', 'genre'.
        """
        match = self._TITULUS_PATTERN.search(text)
        if not match:
            return {}

        params_str = match.group(1)
        metadata: dict[str, Any] = {}

        for param in params_str.split("|"):
            if "=" in param:
                key, value = param.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Map WikiSource template keys to our standard keys
                key_map = {
                    "Scriptor": "author",
                    "OperaeTitulus": "title",
                    "Annus": "date",
                    "Genera": "genre",
                }
                mapped_key = key_map.get(key, key.lower())
                metadata[mapped_key] = value

        return metadata

    # -------------------------------------------------------------------------
    # Markup stripping
    # -------------------------------------------------------------------------

    def _strip_markup(self, text: str) -> str:
        """Remove wiki markup, return clean Latin text.

        Strips templates, interwiki links, HTML tags, and wiki formatting
        while preserving the actual Latin content.

        Args:
            text: Raw wikitext content.

        Returns:
            Cleaned Latin text.
        """
        # Remove templates
        text = self._FINIS_PATTERN.sub("", text)
        text = self._TEXTQUALITY_PATTERN.sub("", text)
        text = self._LIBER_PATTERN.sub("", text)
        text = self._INTRAINCEPTI_PATTERN.sub("", text)
        text = self._TITULUS_PATTERN.sub("", text)

        # Remove interwiki links and images
        text = self._INTERWIKI_PATTERN.sub("", text)
        text = self._IMAGO_PATTERN.sub("", text)

        # Strip HTML div tags (but not content inside)
        text = self._DIV_TAGS_PATTERN.sub("", text)

        # Remove poem tags (but not content inside)
        text = text.replace("<poem>", "").replace("</poem>", "")

        # Remove versus templates (but keep surrounding text)
        text = self._VERSUS_PATTERN.sub("", text)

        # Resolve wiki links: [[target|display]] -> display, [[target]] -> target
        text = self._WIKILINK_PATTERN.sub(r"\1", text)
        text = self._SIMPLE_WIKILINK_PATTERN.sub(r"\1", text)

        # Remove bold/italic wiki markup
        text = self._BOLD_ITALIC_PATTERN.sub("", text)

        # Remove section headers (== X. ==)
        text = self._SECTION_HEADER_PATTERN.sub("", text)

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        text = "\n".join(lines)

        return text.strip()

    # -------------------------------------------------------------------------
    # Content type detection
    # -------------------------------------------------------------------------

    def _is_verse(self, text: str) -> bool:
        """Check if text contains verse content (poem tags or versus templates).

        Args:
            text: Raw wikitext.

        Returns:
            True if the text is verse format.
        """
        return bool(self._POEM_PATTERN.search(text) or self._VERSUS_PATTERN.search(text))

    def _is_index_page(self, text: str) -> bool:
        """Check if text is an index/TOC page rather than actual content.

        Index pages typically contain lists of sub-page links but no
        actual Latin text content (no <poem>, no == section == headers
        with following paragraphs).

        Args:
            text: Raw wikitext.

        Returns:
            True if the text appears to be an index page.
        """
        # Index pages have wiki links to sub-pages but no actual content blocks
        has_subpage_links = bool(re.search(r"\[\[[^\]]+/[^\]]+\]\]", text))
        has_sections = bool(self._SECTION_HEADER_PATTERN.search(text))
        has_verse = self._is_verse(text)

        # It's an index page if it has sub-page links but no content
        # (no section headers with following paragraph text, no verse)
        if has_subpage_links and not has_verse:
            # Check if sections have paragraph content after them
            if has_sections:
                sections = self._parse_sections(text)
                has_para_content = any(s.paragraphs for s in sections)
                if not has_para_content:
                    return True
            else:
                return True

        return False

    # -------------------------------------------------------------------------
    # Section/verse parsing
    # -------------------------------------------------------------------------

    def _parse_sections(self, text: str) -> list[WikiSection]:
        """Parse ``== N. ==`` headers into WikiSection objects.

        Handles prose content organized into sections with numbered paragraphs.

        Args:
            text: Raw wikitext (may include markup).

        Returns:
            List of WikiSection objects.
        """
        # Extract content from <div class="text"> if present
        div_match = self._DIV_TEXT_PATTERN.search(text)
        if div_match:
            text = div_match.group(1)

        sections: list[WikiSection] = []
        # Split by section headers
        parts = self._SECTION_HEADER_PATTERN.split(text)

        # parts[0] is text before first header (usually empty or metadata)
        # Then alternating: header, content, header, content, ...
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i + 1] if i + 1 < len(parts) else ""

            # Strip markup from content
            clean_content = self._strip_markup(content)

            # Parse numbered paragraphs
            paragraphs: list[tuple[int, str]] = []
            # Split on paragraph numbers at start of line or after newline
            para_parts = self._PARAGRAPH_NUM_PATTERN.split(clean_content)

            # para_parts: [pre-text, num, text, num, text, ...]
            for j in range(1, len(para_parts), 2):
                para_num = int(para_parts[j])
                para_text = para_parts[j + 1].strip() if j + 1 < len(para_parts) else ""
                if para_text:
                    paragraphs.append((para_num, para_text))

            section_text = clean_content.strip()
            if section_text:
                sections.append(WikiSection(
                    header=header,
                    text=section_text,
                    paragraphs=paragraphs,
                ))

        return sections

    def _parse_verse_lines(self, text: str) -> list[tuple[int, str]]:
        """Parse verse content with ``{{versus|N}}`` line numbers.

        Args:
            text: Raw wikitext containing <poem> blocks.

        Returns:
            List of (line_number, text) pairs.
        """
        lines: list[tuple[int, str]] = []

        # Extract poem blocks
        poem_matches = self._POEM_PATTERN.findall(text)
        for poem_content in poem_matches:
            current_line_num = 0
            for raw_line in poem_content.split("\n"):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                # Check for versus template
                versus_match = self._VERSUS_PATTERN.search(raw_line)
                if versus_match:
                    current_line_num = int(versus_match.group(1))
                    # Remove the template from the line text
                    line_text = self._VERSUS_PATTERN.sub("", raw_line).strip()
                else:
                    # Continuation or unnumbered line
                    current_line_num += 1
                    line_text = raw_line.strip()

                # Strip any remaining markup
                line_text = self._BOLD_ITALIC_PATTERN.sub("", line_text)
                line_text = self._WIKILINK_PATTERN.sub(r"\1", line_text)
                line_text = self._SIMPLE_WIKILINK_PATTERN.sub(r"\1", line_text)

                if line_text:
                    lines.append((current_line_num, line_text))

        return lines

    # -------------------------------------------------------------------------
    # Core interface
    # -------------------------------------------------------------------------

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a WikiSource .wiki file into text + metadata.

        Handles both prose and verse formats. Index pages are skipped.

        Args:
            path: Path to .wiki file.

        Yields:
            (text, metadata) tuples. Metadata includes parsed titulus info
            and private ``_sections`` or ``_verse_lines`` for span creation.
        """
        raw_text = path.read_text(encoding=self._encoding)

        # Skip index pages
        if self._is_index_page(raw_text):
            return

        # Parse metadata
        metadata = self._parse_titulus(raw_text)
        metadata["filename"] = path.name

        if self._is_verse(raw_text):
            # Verse mode
            verse_lines = self._parse_verse_lines(raw_text)
            if not verse_lines:
                return
            combined_text = "\n".join(text for _, text in verse_lines)
            metadata["_verse_lines"] = verse_lines
            metadata["content_type"] = "verse"
        else:
            # Prose mode
            sections = self._parse_sections(raw_text)
            if not sections:
                return
            # Combine all section text
            combined_text = " ".join(s.text for s in sections)
            metadata["_sections"] = sections
            metadata["content_type"] = "prose"

        yield combined_text, metadata

    def docs(self, fileids: str | list[str] | None = None) -> Iterator["Doc"]:
        """Yield spaCy Docs with citation spans.

        Prose docs have ``doc.spans["sections"]`` with section-level citations.
        Verse docs have ``doc.spans["lines"]`` with line-number citations.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            spaCy Doc objects with citation spans.
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
                self._cache.move_to_end(fileid)
                yield self._cache[fileid]
                continue

            if self._cache_enabled:
                self._cache_misses += 1

            json_metadata = self.get_metadata(fileid)

            for text, file_metadata in self._parse_file(path):
                text = self._normalize_text(text)
                doc = nlp(text)
                doc._.fileid = fileid

                # Merge metadata (excluding private keys)
                clean_meta = {
                    k: v for k, v in file_metadata.items()
                    if not k.startswith("_")
                }
                doc._.metadata = {**json_metadata, **clean_meta}

                # Create citation spans based on content type
                if file_metadata.get("content_type") == "verse":
                    verse_lines = file_metadata.get("_verse_lines", [])
                    doc.spans["lines"] = self._make_verse_spans(doc, verse_lines)
                else:
                    sections = file_metadata.get("_sections", [])
                    doc.spans["sections"] = self._make_section_spans(doc, sections)

                # Cache
                if self._cache_enabled:
                    while len(self._cache) >= self._cache_maxsize:
                        self._cache.popitem(last=False)
                    self._cache[fileid] = doc

                yield doc

    def _make_section_spans(
        self, doc: "Doc", sections: list[WikiSection]
    ) -> list["Span"]:
        """Create citation-annotated spans from prose sections.

        Args:
            doc: The spaCy Doc.
            sections: Parsed WikiSection objects.

        Returns:
            List of Spans with ``_.citation`` set to section.paragraph format.
        """
        spans = []
        char_offset = 0

        for section in sections:
            section_start = char_offset
            section_end = char_offset + len(section.text)

            if section.paragraphs:
                # Create spans for individual paragraphs
                para_offset = section_start
                for para_num, para_text in section.paragraphs:
                    # Find this paragraph text within the section
                    idx = doc.text.find(para_text, para_offset)
                    if idx == -1:
                        continue
                    end_idx = idx + len(para_text)
                    span = doc.char_span(idx, end_idx, alignment_mode="expand")
                    if span is not None:
                        span._.citation = f"{section.header}{para_num}"
                        spans.append(span)
                    para_offset = end_idx
            else:
                # Section without numbered paragraphs
                span = doc.char_span(section_start, section_end, alignment_mode="expand")
                if span is not None:
                    span._.citation = section.header
                    spans.append(span)

            # +1 for the space between sections
            char_offset = section_end + 1

        return spans

    def _make_verse_spans(
        self, doc: "Doc", verse_lines: list[tuple[int, str]]
    ) -> list["Span"]:
        """Create citation-annotated spans from verse lines.

        Args:
            doc: The spaCy Doc.
            verse_lines: List of (line_number, text) pairs.

        Returns:
            List of Spans with ``_.citation`` set to line number.
        """
        spans = []
        char_offset = 0

        for line_num, line_text in verse_lines:
            start_char = char_offset
            end_char = char_offset + len(line_text)

            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is not None:
                span._.citation = str(line_num)
                spans.append(span)

            # +1 for the newline between verse lines
            char_offset = end_char + 1

        return spans

    # -------------------------------------------------------------------------
    # Additional iteration methods
    # -------------------------------------------------------------------------

    def sections(
        self, fileids: str | list[str] | None = None
    ) -> Iterator["Span"]:
        """Yield section-level Spans with citations (prose files).

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Span objects with ``_.citation`` set.
        """
        for doc in self.docs(fileids):
            yield from doc.spans.get("sections", [])

    def lines(
        self, fileids: str | list[str] | None = None
    ) -> Iterator["Span"]:
        """Yield verse line Spans with citations (verse files).

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Span objects with ``_.citation`` set to line number.
        """
        for doc in self.docs(fileids):
            yield from doc.spans.get("lines", [])

    def _get_citation_for_span(self, doc: "Doc", span: "Span") -> str:
        """Get WikiSource citation for a span.

        Checks both sections and lines span groups.
        """
        for span_key in ("sections", "lines"):
            for labeled_span in doc.spans.get(span_key, []):
                if labeled_span.start <= span.start < labeled_span.end:
                    citation = getattr(labeled_span._, "citation", None)
                    if citation is not None:
                        return citation

        return super()._get_citation_for_span(doc, span)

    # -------------------------------------------------------------------------
    # Download from la.wikisource.org
    # -------------------------------------------------------------------------

    @classmethod
    def download(
        cls,
        page: str,
        destination: str | Path,
        follow_subpages: bool = True,
    ) -> list[Path]:
        """Download wikitext from la.wikisource.org.

        Fetches the raw wikitext for a page via the MediaWiki API and saves
        it as a ``.wiki`` file. If the page contains sub-page links and
        ``follow_subpages=True``, recursively fetches each sub-page.

        Args:
            page: Page title (e.g., ``"De_vita_beata"`` or ``"Aeneis"``).
            destination: Local directory to save files into.
            follow_subpages: If True, recursively fetch sub-pages.

        Returns:
            List of saved file paths.

        Example:
            >>> paths = WikiSourceReader.download("De_vita_beata", "~/data/wikisource")
            >>> reader = WikiSourceReader("~/data/wikisource")
        """
        dest = Path(destination).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        cls._download_page(page, dest, follow_subpages, saved, set())
        return saved

    @classmethod
    def _download_page(
        cls,
        page: str,
        dest: Path,
        follow_subpages: bool,
        saved: list[Path],
        visited: set[str],
    ) -> None:
        """Recursively download a page and its sub-pages.

        Args:
            page: Page title to download.
            dest: Destination directory.
            follow_subpages: Whether to follow sub-page links.
            saved: Accumulator for saved paths.
            visited: Set of already-visited pages to prevent cycles.
        """
        if page in visited:
            return
        visited.add(page)

        # Fetch wikitext via MediaWiki API
        api_url = (
            "https://la.wikisource.org/w/api.php"
            f"?action=parse&page={urllib.parse.quote(page)}"
            "&prop=wikitext&format=json"
        )

        try:
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": f"latincy-readers/{_get_version()}"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise ConnectionError(
                f"Failed to fetch '{page}' from la.wikisource.org: {e}"
            ) from e

        if "error" in data:
            raise ValueError(
                f"MediaWiki API error for '{page}': {data['error'].get('info', 'unknown')}"
            )

        wikitext = data["parse"]["wikitext"]["*"]

        # Determine filename: normalize page title to filesystem-safe name
        safe_name = page.lower().replace(" ", "_").replace("/", "_")
        file_path = dest / f"{safe_name}.wiki"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(wikitext, encoding="utf-8")
        saved.append(file_path)

        # Follow sub-page links if requested
        if follow_subpages:
            # Find [[Page/Subpage|display]] links
            subpage_pattern = re.compile(
                r"\[\[(" + re.escape(page) + r"/[^\]|]+)(?:\|[^\]]*)?]]"
            )
            for match in subpage_pattern.finditer(wikitext):
                subpage = match.group(1)
                cls._download_page(subpage, dest, follow_subpages, saved, visited)
