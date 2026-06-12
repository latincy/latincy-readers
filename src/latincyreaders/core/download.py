"""Mixin for downloadable corpus support.

Provides standardized auto-download functionality for corpus readers
that can be cloned from GitHub repositories.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


# Default data directory for latincy-readers
LATINCY_DATA = Path.home() / "latincy_data"


class DownloadableCorpusMixin:
    """Mixin providing auto-download functionality for corpus readers.

    Subclasses must define these class attributes:
        CORPUS_URL: GitHub URL for cloning the corpus.
        ENV_VAR: Environment variable name for custom corpus path.
        DEFAULT_SUBDIR: Subdirectory name under ~/latincy_data.
        _FILE_CHECK_PATTERN: Glob pattern to verify corpus exists (e.g., "**/*.tess").

    Optionally:
        CORPUS_VERSION: A git tag/branch to pin the clone to (e.g. "v0.5").
            When set, ``download()`` clones that ref instead of the default
            branch HEAD, making the corpus reproducible across time. Leave as
            None to track the default branch (legacy behaviour).

    Example:
        class MyCorpusReader(DownloadableCorpusMixin, BaseCorpusReader):
            CORPUS_URL = "https://github.com/org/corpus.git"
            ENV_VAR = "MY_CORPUS_PATH"
            DEFAULT_SUBDIR = "my_corpus"
            _FILE_CHECK_PATTERN = "**/*.txt"

            def __init__(self, root=None, auto_download=True, ...):
                if root is None:
                    root = self._get_default_root(auto_download)
                super().__init__(root, ...)
    """

    # Subclasses must override these
    CORPUS_URL: str
    ENV_VAR: str
    DEFAULT_SUBDIR: str
    _FILE_CHECK_PATTERN: str = "**/*"

    # Optional: pin the clone to a git tag/branch for reproducibility.
    # None = track the default branch HEAD (legacy behaviour).
    CORPUS_VERSION: str | None = None

    @classmethod
    def default_root(cls) -> Path:
        """Return the default corpus location.

        Checks in order:
        1. Environment variable specified by ENV_VAR
        2. ~/latincy_data/{DEFAULT_SUBDIR}

        Returns:
            Path to the default corpus location.
        """
        if env_path := os.environ.get(cls.ENV_VAR):
            return Path(env_path)
        return LATINCY_DATA / cls.DEFAULT_SUBDIR

    @classmethod
    def _get_default_root(
        cls, auto_download: bool = True, ref: str | None = None
    ) -> Path:
        """Get the corpus root, downloading if necessary.

        Args:
            auto_download: If True and corpus not found, offer to download.
            ref: git tag/branch to clone. Defaults to CORPUS_VERSION.

        Returns:
            Path to the corpus.

        Raises:
            FileNotFoundError: If corpus not found and auto_download is False.
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
        pin = ref or cls.CORPUS_VERSION
        pin_note = f" ({pin})" if pin else ""
        print(f"{cls.__name__} corpus not found at {root}")
        response = input(f"Download{pin_note} from GitHub? [y/N]: ").strip().lower()

        if response in ("y", "yes"):
            cls.download(root, ref=ref)
            return root
        else:
            raise FileNotFoundError(
                f"{cls.__name__} corpus not found at {root}. "
                f"Download manually from {cls.CORPUS_URL}"
            )

    @classmethod
    def download(
        cls, destination: Path | None = None, ref: str | None = None
    ) -> Path:
        """Download the corpus from GitHub.

        Args:
            destination: Where to clone the corpus. Defaults to default_root().
            ref: git tag/branch to clone. Defaults to CORPUS_VERSION. When set,
                the clone is pinned to that ref for reproducibility.

        Returns:
            Path to the downloaded corpus.

        Raises:
            RuntimeError: If git clone fails or git is not installed.
        """
        if destination is None:
            destination = cls.default_root()

        destination = Path(destination)

        if destination.exists():
            print(f"Corpus already exists at: {destination}")
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)

        pin = ref or cls.CORPUS_VERSION
        cmd = ["git", "clone", "--depth", "1"]
        if pin:
            cmd += ["--branch", pin]
        cmd += [cls.CORPUS_URL, str(destination)]

        pin_note = f" at {pin}" if pin else ""
        print(f"Cloning {cls.__name__} corpus{pin_note} to {destination}...")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully downloaded to {destination}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e}") from e
        except FileNotFoundError:
            raise RuntimeError(
                "git not found. Please install git or download manually from "
                f"{cls.CORPUS_URL}"
            )

        return destination

    @classmethod
    def installed_version(cls, root: Path | None = None) -> str | None:
        """Return the corpus version actually present on disk.

        Reads the git tag (or commit) of the cloned corpus via
        ``git describe``. Returns None if the corpus is not a git checkout
        (e.g. a manually unpacked tarball) or git is unavailable.

        Args:
            root: Corpus root. Defaults to default_root().

        Returns:
            The tag name (e.g. "v0.5"), a commit hash, or None.
        """
        root = Path(root) if root is not None else cls.default_root()
        if not (root / ".git").exists():
            return None
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "describe", "--tags", "--always"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return result.stdout.strip() or None
