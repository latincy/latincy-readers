"""Project Gutenberg corpus reader.

Fetches plain-text files from Project Gutenberg by numeric ID, caches them
to disk, strips the standard PG boilerplate header/footer, and exposes the
usual corpus-reader interface.

URL pattern: https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt

Example:
    >>> reader = ProjectGutenbergReader(ids=[28233])
    >>> for text in reader.texts():
    ...     print(text[:200])
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Iterator

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel

_DEFAULT_CACHE = Path.home() / ".latincy_cache" / "gutenberg"

# Markers used in all standard PG plain-text files.
_START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


class ProjectGutenbergReader(BaseCorpusReader):
    """Reader for Project Gutenberg plain-text files.

    Fetches texts by numeric PG ID and caches them locally so subsequent
    reads are instant. PG boilerplate (license header and footer) is stripped
    automatically.

    Only ``texts()`` is guaranteed language-agnostic. ``docs()``, ``sents()``,
    and ``tokens()`` require a compatible spaCy model; for Latin texts use
    ``annotation_level=AnnotationLevel.TOKENIZE`` with ``la_core_web_sm`` or
    higher.

    Args:
        ids: One or more Project Gutenberg text IDs (e.g. ``28233``).
        cache_dir: Directory for cached ``.txt`` files. Defaults to
            ``~/.latincy_cache/gutenberg/``.
        annotation_level: How much NLP annotation to apply.
        encoding: Text encoding for reading cached files.
        cache: If True (default), cache processed Doc objects in memory.
        cache_maxsize: Maximum number of documents to keep in memory.
        model_name: spaCy model to load for BASIC/FULL annotation levels.
            Defaults to ``"la_core_web_lg"`` for Latin; pass e.g.
            ``"en_core_web_sm"`` for English texts.
        lang: Language code for the blank tokenizer used at TOKENIZE level.
            Defaults to ``"la"``; pass ``"en"`` for English.

    Example:
        >>> reader = ProjectGutenbergReader(ids=28233)          # single ID
        >>> reader = ProjectGutenbergReader(ids=[28233, 1727])  # multiple IDs
        >>> for text in reader.texts():
        ...     print(text[:120])
    """

    BASE_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

    def __init__(
        self,
        ids: int | str | list[int | str],
        cache_dir: str | Path | None = None,
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        encoding: str = "utf-8",
        cache: bool = True,
        cache_maxsize: int = 128,
        model_name: str = "la_core_web_lg",
        lang: str = "la",
        **kwargs,
    ):
        if isinstance(ids, (int, str)):
            ids = [ids]
        self._pg_ids = [str(i) for i in ids]

        pg_cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
        pg_cache.mkdir(parents=True, exist_ok=True)
        self._pg_cache_dir = pg_cache

        # Ensure each requested text is on disk before handing root to base.
        for pg_id in self._pg_ids:
            self._ensure_text(pg_id)

        super().__init__(
            root=pg_cache,
            fileids=None,
            encoding=encoding,
            annotation_level=annotation_level,
            cache=cache,
            cache_maxsize=cache_maxsize,
            model_name=model_name,
            lang=lang,
            **kwargs,
        )

    @classmethod
    def _default_file_pattern(cls) -> str:
        return "pg*.txt"

    # ------------------------------------------------------------------
    # Fetch / cache
    # ------------------------------------------------------------------

    def _ensure_text(self, pg_id: str) -> Path:
        dest = self._pg_cache_dir / f"pg{pg_id}.txt"
        if not dest.exists():
            self._fetch_text(pg_id, dest)
        return dest

    def _fetch_text(self, pg_id: str, dest: Path) -> None:
        url = self.BASE_URL.format(id=pg_id)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "latincy-readers"},
            )
            with urllib.request.urlopen(req) as resp:
                raw = resp.read()
        except Exception as exc:
            raise ConnectionError(
                f"Failed to fetch Project Gutenberg text {pg_id} from {url}: {exc}"
            ) from exc
        dest.write_bytes(raw)

    # ------------------------------------------------------------------
    # Boilerplate stripping
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_boilerplate(text: str) -> str:
        """Remove PG header (up to and including the START marker line) and footer."""
        start_idx = text.find(_START_MARKER)
        if start_idx != -1:
            newline = text.find("\n", start_idx)
            text = text[newline + 1 :] if newline != -1 else text[start_idx + len(_START_MARKER):]

        end_idx = text.find(_END_MARKER)
        if end_idx != -1:
            text = text[:end_idx]

        return text.strip()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        text = path.read_text(encoding=self._encoding)
        text = self._normalize_text(text)
        text = self._strip_boilerplate(text)

        if not text.strip():
            return

        # Filename is always "pg{id}.txt"; strip the "pg" prefix for the ID.
        pg_id = path.stem[2:]

        metadata = {
            "filename": path.name,
            "path": str(path),
            "pg_id": pg_id,
        }

        yield text, metadata
