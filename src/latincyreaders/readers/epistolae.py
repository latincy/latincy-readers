"""Epistolae corpus reader.

Reader for the Epistolae project (Columbia University / University of Siena):
medieval Latin letters written by and to women, 4th–13th century.

Source files are Hugo-flavored Markdown (.html.md) with YAML frontmatter.
Each file contains an English translation and a Latin original in separate
``<h2>``-delimited sections; this reader extracts only the Latin.

Data repository: https://github.com/ccnmtl/epistolae-hugo
License: CC BY-NC-SA 4.0 (Columbia University Libraries)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

import yaml

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel

if TYPE_CHECKING:
    pass

# Regex matching any <h2 ...> opening tag
_H2_OPEN = re.compile(r"<h2[^>]*>", re.IGNORECASE)
# Any HTML tag
_HTML_TAG = re.compile(r"<[^>]+>")


def _extract_section(content: str, heading: str) -> str:
    """Extract text between a named ``<h2>`` heading and the next ``<h2>``.

    Args:
        content: Full document body (after frontmatter).
        heading: Heading text to locate, e.g. ``"Original letter:"``.

    Returns:
        Stripped text of that section, with HTML tags removed.
        Empty string if heading not found.
    """
    # Find the h2 tag whose text content matches heading
    pattern = re.compile(
        r"<h2[^>]*>\s*" + re.escape(heading) + r"\s*</h2>(.*?)(?=<h2|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        return ""
    raw = match.group(1)
    # Strip HTML tags
    text = _HTML_TAG.sub(" ", raw)
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


class EpistolaeReader(BaseCorpusReader):
    """Reader for the Epistolae medieval Latin letters corpus.

    Parses Hugo Markdown files (``.html.md``) from the Epistolae project.
    Each file contains YAML frontmatter with metadata (sender, recipient,
    date) and HTML-structured body with English translation and Latin original
    in separate ``<h2>`` sections.

    Only the Latin ``"Original letter:"`` section is extracted as text.
    English translation, historical context, and scholarly notes are excluded.

    Example:
        >>> reader = EpistolaeReader("/path/to/epistolae-hugo/content/letter")
        >>> for doc in reader.docs():
        ...     meta = doc._.metadata
        ...     print(meta["senders"], meta["date"], doc.text[:80])
    """

    @classmethod
    def _default_file_pattern(cls) -> str:
        return "**/*.html.md"

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
        """Initialize the Epistolae reader.

        Args:
            root: Root directory containing ``.html.md`` letter files.
                Point at the ``content/letter/`` directory of the Hugo repo.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: NLP annotation level.
            cache: If True (default), cache processed Doc objects.
            cache_maxsize: Maximum number of documents to cache.
            **kwargs: Additional arguments passed to BaseCorpusReader.
        """
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            annotation_level=annotation_level,
            cache=cache,
            cache_maxsize=cache_maxsize,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _split_frontmatter(self, raw: str) -> tuple[dict, str]:
        """Split YAML frontmatter from body content.

        Args:
            raw: Full file content.

        Returns:
            ``(frontmatter_dict, body_content)`` tuple.
            Returns ``({}, raw)`` if no frontmatter found.
        """
        if not raw.startswith("---"):
            return {}, raw
        parts = raw.split("---", 2)
        if len(parts) < 3:
            return {}, raw
        try:
            fm = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            fm = {}
        return fm, parts[2]

    def _build_metadata(self, fm: dict, path: Path) -> dict:
        """Build metadata dict from parsed frontmatter.

        Args:
            fm: Parsed YAML frontmatter dict.
            path: Source file path.

        Returns:
            Metadata dict with letter_id, senders, receivers, date,
            title, and filename.
        """
        senders = [s["name"] for s in (fm.get("senders") or []) if "name" in s]
        receivers = [r["name"] for r in (fm.get("receivers") or []) if "name" in r]
        return {
            "filename": path.name,
            "path": str(path),
            "letter_id": str(fm.get("letter_id", "")),
            "title": fm.get("title", ""),
            "date": str(fm.get("ltr_date", "")) if fm.get("ltr_date") else None,
            "senders": senders,
            "receivers": receivers,
        }

    # ------------------------------------------------------------------
    # Core parsing
    # ------------------------------------------------------------------

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse an Epistolae ``.html.md`` file.

        Extracts the Latin text from the ``"Original letter:"`` section,
        discarding the English translation and all editorial apparatus.

        Args:
            path: Path to ``.html.md`` file.

        Yields:
            Single ``(text, metadata)`` tuple per file.
        """
        raw = path.read_text(encoding=self._encoding)
        fm, body = self._split_frontmatter(raw)

        latin = _extract_section(body, "Original letter:")
        latin = self._normalize_text(latin)
        if not latin.strip():
            return

        yield latin, self._build_metadata(fm, path)

    # ------------------------------------------------------------------
    # Header iteration
    # ------------------------------------------------------------------

    def headers(
        self, fileids: str | list[str] | None = None
    ) -> Iterator[dict]:
        """Yield metadata dicts from Epistolae frontmatter.

        Extracts ``letter_id``, ``title``, ``date``, ``senders``,
        ``receivers``, and ``filename`` without NLP processing.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Metadata dictionaries.
        """
        for path in self._iter_paths(fileids):
            raw = path.read_text(encoding=self._encoding)
            fm, _ = self._split_frontmatter(raw)
            yield self._build_metadata(fm, path)
