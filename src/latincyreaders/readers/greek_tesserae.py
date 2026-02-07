"""Greek Tesserae corpus reader.

Reads Ancient Greek texts in the Tesserae format from the CLTK Greek
Tesserae corpus (grc_text_tesserae). Uses the same `.tess` citation
format as the Latin Tesserae corpus.

Requires OdyCy for BASIC/FULL annotation levels:
    pip install https://huggingface.co/chcaa/grc_odycy_joint_lg/resolve/main/grc_odycy_joint_lg-any-py3-none-any.whl
"""

from __future__ import annotations

import re
import subprocess
import unicodedata
from pathlib import Path

from latincyreaders.core.base import AnnotationLevel
from latincyreaders.core.download import LATINCY_DATA
from latincyreaders.readers.tesserae import TesseraeReader

# Regex to fix orphaned combining marks at word boundaries.
# The Greek Tesserae corpus has decomposed Unicode where breathing marks
# (U+0313, U+0314) sometimes appear before the capital letter instead
# of after it (e.g., ̓Α instead of Ἀ). This only happens at word
# boundaries -- after whitespace or at the start of the string.
_ORPHANED_COMBINING_RE = re.compile(r"(?<=\s)([\u0300-\u036f]+)(\w)")
_ORPHANED_COMBINING_START_RE = re.compile(r"^([\u0300-\u036f]+)(\w)")


class GreekTesseraeReader(TesseraeReader):
    """Reader for Tesserae-format Ancient Greek texts.

    Inherits all functionality from TesseraeReader (citation parsing,
    search, concordance, KWIC, ngrams, etc.) with Greek-specific defaults:

    - Corpus: CLTK Greek Tesserae (grc_text_tesserae)
    - NLP model: OdyCy (grc_odycy_joint_lg)
    - Language: Ancient Greek (grc)

    If no root path is provided, looks for the corpus in:
    1. The path specified by GRC_TESSERAE_PATH environment variable
    2. ~/latincy_data/grc_text_tesserae/texts/texts

    Note: The Greek Tesserae repo has .tess files inside a ``texts/``
    subdirectory, unlike the Latin repo where they are at the root.
    The reader handles this automatically.

    Example:
        >>> reader = GreekTesseraeReader()  # Uses default location or downloads
        >>> reader = GreekTesseraeReader("/path/to/greek/corpus")
        >>> for doc in reader.docs():
        ...     for line in doc.spans["lines"]:
        ...         print(f"{line._.citation}: {line.text[:50]}...")
    """

    CORPUS_URL = "https://github.com/cltk/grc_text_tesserae.git"
    ENV_VAR = "GRC_TESSERAE_PATH"
    # The repo is cloned to grc_text_tesserae/texts, but .tess files
    # are inside a texts/ subdirectory within the repo
    DEFAULT_SUBDIR = "grc_text_tesserae/texts/texts"
    _CLONE_SUBDIR = "grc_text_tesserae/texts"
    _FILE_CHECK_PATTERN = "**/*.tess"

    @classmethod
    def _clone_root(cls) -> Path:
        """Return the path where the repo should be cloned."""
        return LATINCY_DATA / cls._CLONE_SUBDIR

    @classmethod
    def _get_default_root(cls, auto_download: bool = True) -> Path:
        """Get the corpus root, downloading if necessary.

        Overrides the parent to handle the Greek repo's nested structure:
        the repo is cloned to ``_CLONE_SUBDIR`` but .tess files are
        read from ``DEFAULT_SUBDIR`` (one level deeper).
        """
        root = cls.default_root()

        if root.exists() and any(root.glob(cls._FILE_CHECK_PATTERN)):
            return root

        if not auto_download:
            raise FileNotFoundError(
                f"{cls.__name__} corpus not found at {root}. "
                f"Set {cls.ENV_VAR} environment variable or pass root= explicitly. "
                f"Or set auto_download=True to download automatically."
            )

        # Prompt for download
        print(f"{cls.__name__} corpus not found at {root}")
        response = input("Download from GitHub? [y/N]: ").strip().lower()

        if response in ("y", "yes"):
            cls.download()
            return root
        else:
            raise FileNotFoundError(
                f"{cls.__name__} corpus not found at {root}. "
                f"Download manually from {cls.CORPUS_URL}"
            )

    @classmethod
    def download(cls, destination: Path | None = None) -> Path:
        """Download the corpus from GitHub.

        Clones the repo to the clone root, then returns the texts
        subdirectory where .tess files live.

        Args:
            destination: Where to clone the repo. Defaults to clone root.

        Returns:
            Path to the downloaded corpus texts directory.
        """
        clone_dest = destination or cls._clone_root()
        clone_dest = Path(clone_dest)

        if clone_dest.exists():
            print(f"Corpus already exists at: {clone_dest}")
            # Return the texts subdirectory
            texts_dir = clone_dest / "texts"
            return texts_dir if texts_dir.exists() else clone_dest

        clone_dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"Cloning {cls.__name__} corpus to {clone_dest}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", cls.CORPUS_URL, str(clone_dest)],
                check=True,
            )
            print(f"Successfully downloaded to {clone_dest}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e}") from e
        except FileNotFoundError:
            raise RuntimeError(
                "git not found. Please install git or download manually from "
                f"{cls.CORPUS_URL}"
            )

        texts_dir = clone_dest / "texts"
        return texts_dir if texts_dir.exists() else clone_dest

    def _normalize_text(self, text: str) -> str:
        """Normalize Greek text, fixing decomposed Unicode issues.

        The Greek Tesserae corpus uses NFD (decomposed) Unicode with
        occasional misordered combining marks (breathing marks before
        their base capital letter). This method:

        1. Reorders orphaned combining marks to follow their base letter
        2. Applies NFC normalization to compose precomposed characters
        """
        # Fix combining marks that appear before their base character
        # (only at word boundaries, not mid-word where they're correct)
        text = _ORPHANED_COMBINING_RE.sub(r"\2\1", text)
        text = _ORPHANED_COMBINING_START_RE.sub(r"\2\1", text)
        # Replace ASCII apostrophe with modifier letter apostrophe (U+02BC)
        # so spaCy keeps elided forms as single tokens (e.g., ἄλγεʼ not ἄλγε + ')
        text = text.replace("'", "\u02bc")
        return unicodedata.normalize("NFC", text)

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        auto_download: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
        model_name: str = "grc_odycy_joint_lg",
    ):
        """Initialize the Greek Tesserae reader.

        Args:
            root: Root directory containing .tess files. If None, uses default location.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: How much NLP annotation to apply.
            auto_download: If True and corpus not found, offer to download.
            cache: If True (default), cache processed Doc objects for reuse.
            cache_maxsize: Maximum number of documents to cache (default 128).
            model_name: Name of the spaCy model to load. Defaults to OdyCy.
        """
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            annotation_level=annotation_level,
            auto_download=auto_download,
            cache=cache,
            cache_maxsize=cache_maxsize,
            model_name=model_name,
            lang="grc",
        )
