"""Universal Dependencies treebank reader.

Reads CoNLL-U format files from UD treebanks, constructing spaCy Docs
directly from the gold-standard annotations.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

import conllu
from spacy.tokens import Doc
from spacy.vocab import Vocab

from latincyreaders.core.base import BaseCorpusReader
from latincyreaders.core.download import DownloadableCorpusMixin
from latincyreaders.nlp.pipeline import AnnotationLevel

if TYPE_CHECKING:
    from spacy.tokens import Span, Token


# Known Latin UD treebanks
LATIN_TREEBANKS = {
    "proiel": {
        "url": "https://github.com/UniversalDependencies/UD_Latin-PROIEL.git",
        "subdir": "ud_latin_proiel",
        "env_var": "UD_PROIEL_PATH",
        "description": "Vulgate, Caesar, Cicero, Palladius",
    },
    "perseus": {
        "url": "https://github.com/UniversalDependencies/UD_Latin-Perseus.git",
        "subdir": "ud_latin_perseus",
        "env_var": "UD_PERSEUS_PATH",
        "description": "Classical texts from Perseus Digital Library",
    },
    "ittb": {
        "url": "https://github.com/UniversalDependencies/UD_Latin-ITTB.git",
        "subdir": "ud_latin_ittb",
        "env_var": "UD_ITTB_PATH",
        "description": "Index Thomisticus (Thomas Aquinas)",
    },
    "llct": {
        "url": "https://github.com/UniversalDependencies/UD_Latin-LLCT.git",
        "subdir": "ud_latin_llct",
        "env_var": "UD_LLCT_PATH",
        "description": "Late Latin Charter Treebank",
    },
    "udante": {
        "url": "https://github.com/UniversalDependencies/UD_Latin-UDante.git",
        "subdir": "ud_latin_udante",
        "env_var": "UD_UDANTE_PATH",
        "description": "Dante's Latin works",
    },
    "circse": {
        "url": "https://github.com/UniversalDependencies/UD_Latin-CIRCSE.git",
        "subdir": "ud_latin_circse",
        "env_var": "UD_CIRCSE_PATH",
        "description": "CIRCSE Latin treebank",
    },
}


@dataclass
class UDSentence:
    """Parsed sentence from CoNLL-U file."""

    sent_id: str
    text: str
    tokens: list[dict[str, Any]]
    metadata: dict[str, str]


class UDReader(BaseCorpusReader):
    """Reader for Universal Dependencies CoNLL-U treebanks.

    Constructs spaCy Doc objects directly from CoNLL-U gold annotations,
    preserving all UD information in custom extensions.

    Unlike other readers, UDReader does NOT run the spaCy NLP pipeline.
    Instead, it constructs Docs from the gold-standard UD annotations,
    giving you access to curated linguistic data.

    Example:
        >>> reader = UDReader("/path/to/ud_treebank")
        >>> for doc in reader.docs():
        ...     for sent in doc.spans["ud_sents"]:
        ...         print(f"{sent._.citation}: {sent.text}")
        ...     for token in doc:
        ...         print(f"{token.text}: {token._.ud['upos']}")

    Attributes:
        TREEBANK: Name of the treebank (if using a known treebank).
    """

    TREEBANK: str | None = None

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        cache: bool = True,
        cache_maxsize: int = 128,
    ):
        """Initialize the UD reader.

        Args:
            root: Root directory containing .conllu files.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding (default UTF-8 per UD spec).
            annotation_level: Ignored - UD annotations are always used.
            cache: If True, cache processed Doc objects.
            cache_maxsize: Maximum docs to cache.

        Note:
            The annotation_level parameter is accepted for API consistency
            but is ignored. UDReader always uses the gold UD annotations
            rather than running the spaCy pipeline.
        """
        # Initialize base without parent's __init__ calling NLP
        self._root = Path(root).resolve()
        self._fileids_pattern = fileids or self._default_file_pattern()
        self._encoding = encoding
        self._annotation_level = AnnotationLevel.NONE  # Don't load spaCy model
        self._nlp = None
        self._metadata_pattern = None
        self._metadata = None

        # Caching (matches base class)
        self._cache_enabled = cache
        self._cache_maxsize = cache_maxsize
        self._cache: OrderedDict[str, Doc] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # Shared vocab for Doc construction
        self._vocab: Vocab | None = None

    @property
    def vocab(self) -> Vocab:
        """Shared spaCy Vocab for Doc construction."""
        if self._vocab is None:
            self._vocab = Vocab()
        return self._vocab

    @classmethod
    def _default_file_pattern(cls) -> str:
        """CoNLL-U files use .conllu extension."""
        return "**/*.conllu"

    def _iter_paths(self, fileids: str | list[str] | None = None) -> Iterator[Path]:
        """Iterate over paths matching fileids pattern.

        Args:
            fileids: Specific file(s) or pattern to match.

        Yields:
            Path objects for matching files.
        """
        if fileids is None:
            pattern = self._fileids_pattern
        elif isinstance(fileids, str):
            pattern = fileids
        else:
            # List of specific fileids
            for fid in fileids:
                path = self._root / fid
                if path.exists():
                    yield path
            return

        for path in sorted(self._root.glob(pattern)):
            if path.is_file():
                yield path

    def _parse_conllu(self, path: Path) -> Iterator[UDSentence]:
        """Parse CoNLL-U file into UDSentence objects.

        Args:
            path: Path to .conllu file.

        Yields:
            UDSentence objects for each sentence.
        """
        text = path.read_text(encoding=self._encoding)
        sentences = conllu.parse(text)

        for sent in sentences:
            # Extract metadata
            sent_id = sent.metadata.get("sent_id", "")
            sent_text = sent.metadata.get("text", "")

            # Convert tokens to dicts with parsed FEATS/MISC
            tokens = []
            for tok in sent:
                # Skip multi-word token lines (id like "1-2")
                if isinstance(tok["id"], tuple):
                    continue

                token_dict = {
                    "id": tok["id"],
                    "form": tok["form"],
                    "lemma": tok["lemma"],
                    "upos": tok["upos"],
                    "xpos": tok["xpos"],
                    "feats": tok["feats"] or {},
                    "head": tok["head"],
                    "deprel": tok["deprel"],
                    "deps": tok["deps"],
                    "misc": tok["misc"] or {},
                }
                tokens.append(token_dict)

            yield UDSentence(
                sent_id=sent_id,
                text=sent_text,
                tokens=tokens,
                metadata=dict(sent.metadata),
            )

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse CoNLL-U file.

        Note: This is implemented for API compatibility but docs()
        overrides this with custom Doc construction.
        """
        for sent in self._parse_conllu(path):
            yield sent.text, {"sent_id": sent.sent_id}

    def _build_doc(self, sentences: list[UDSentence], fileid: str) -> Doc:
        """Build a spaCy Doc from UD sentences.

        Constructs the Doc directly from gold annotations without
        running the spaCy pipeline.

        Args:
            sentences: List of parsed UD sentences.
            fileid: File identifier for metadata.

        Returns:
            spaCy Doc with UD annotations.
        """
        # Collect all tokens
        words: list[str] = []
        spaces: list[bool] = []
        sent_starts: list[bool] = []
        ud_data: list[dict] = []
        sent_offsets: list[tuple[int, int]] = []

        for sent in sentences:
            start_idx = len(words)
            for i, tok in enumerate(sent.tokens):
                words.append(tok["form"])

                # Determine spacing from MISC field
                misc = tok.get("misc", {})
                has_space = misc.get("SpaceAfter", "Yes") != "No"
                spaces.append(has_space)

                # Mark sentence starts
                sent_starts.append(i == 0)

                # Store UD data for later
                ud_data.append(tok)

            end_idx = len(words)
            sent_offsets.append((start_idx, end_idx))

        if not words:
            # Empty file - return empty doc
            doc = Doc(self.vocab)
            doc._.fileid = fileid
            doc._.metadata = {"source": "universal_dependencies"}
            doc.spans["ud_sents"] = []
            return doc

        # Create Doc with correct sentence boundaries
        doc = Doc(
            self.vocab,
            words=words,
            spaces=spaces,
            sent_starts=sent_starts,
        )

        # Populate token attributes and extensions
        for sent_idx, (start, end) in enumerate(sent_offsets):
            sent_tokens = sentences[sent_idx].tokens
            for local_idx, token in enumerate(doc[start:end]):
                ud = sent_tokens[local_idx]

                # Standard spaCy attributes
                token.lemma_ = ud["lemma"] or ""
                token.pos_ = ud["upos"] or ""
                token.tag_ = ud["xpos"] or ""
                token.dep_ = ud["deprel"] or ""

                # Handle head (UD uses 1-based, 0 means root)
                head_idx = ud["head"]
                if head_idx == 0 or head_idx is None:
                    token.head = token  # Root points to self
                else:
                    # Convert 1-based UD index to doc-level index
                    target_idx = start + head_idx - 1
                    if 0 <= target_idx < len(doc):
                        token.head = doc[target_idx]
                    else:
                        token.head = token  # Fallback

                # Store full UD data in extension
                token._.ud = ud

        # Set doc-level extensions
        doc._.fileid = fileid
        doc._.metadata = {"source": "universal_dependencies"}

        # Create sentence spans
        doc.spans["ud_sents"] = self._make_sentence_spans(doc, sentences, sent_offsets)

        return doc

    def _make_sentence_spans(
        self,
        doc: Doc,
        sentences: list[UDSentence],
        offsets: list[tuple[int, int]],
    ) -> list["Span"]:
        """Create citation-annotated spans for UD sentences.

        Args:
            doc: The constructed spaCy Doc.
            sentences: Original UD sentence data.
            offsets: List of (start, end) token indices for each sentence.

        Returns:
            List of Spans with _.citation set to sent_id.
        """
        spans = []

        for sent, (start, end) in zip(sentences, offsets):
            if end <= len(doc):
                span = doc[start:end]
                span._.citation = sent.sent_id
                span._.metadata = {"text": sent.text, **sent.metadata}
                spans.append(span)

        return spans

    def fileids(self, match: str | None = None) -> list[str]:
        """Return list of file identifiers in the corpus.

        Args:
            match: Optional regex pattern to filter fileids.

        Returns:
            List of file identifiers (relative paths).
        """
        import re
        from natsort import natsorted

        fileids = []
        for path in self._root.glob(self._fileids_pattern):
            if path.is_file():
                fid = str(path.relative_to(self._root))
                if match is None or re.search(match, fid):
                    fileids.append(fid)

        return natsorted(fileids)

    def docs(self, fileids: str | list[str] | None = None) -> Iterator[Doc]:
        """Yield spaCy Docs constructed from UD gold annotations.

        Unlike other readers, this does NOT run the spaCy NLP pipeline.
        Docs are constructed directly from CoNLL-U annotations.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            spaCy Doc objects with:
                - token._.ud containing all UD annotations
                - doc.spans["ud_sents"] with sentence boundaries
                - Standard spaCy attributes (lemma_, pos_, etc.) populated
        """
        for path in self._iter_paths(fileids):
            fileid = str(path.relative_to(self._root))

            # Check cache
            if self._cache_enabled and fileid in self._cache:
                self._cache_hits += 1
                self._cache.move_to_end(fileid)
                yield self._cache[fileid]
                continue

            if self._cache_enabled:
                self._cache_misses += 1

            # Parse and build Doc
            sentences = list(self._parse_conllu(path))
            if not sentences:
                continue

            doc = self._build_doc(sentences, fileid)

            # Cache
            if self._cache_enabled:
                while len(self._cache) >= self._cache_maxsize:
                    self._cache.popitem(last=False)
                self._cache[fileid] = doc

            yield doc

    def texts(
        self, fileids: str | list[str] | None = None
    ) -> Iterator[str]:
        """Yield raw text from each sentence.

        Zero NLP overhead - reads directly from # text = comments.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Raw text strings from each sentence.
        """
        for path in self._iter_paths(fileids):
            for sent in self._parse_conllu(path):
                if sent.text:
                    yield sent.text

    def sents(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Span | str"]:
        """Yield UD sentences as Spans or strings.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Spans.

        Yields:
            Sentence Spans with _.citation (or strings if as_text=True).
        """
        if as_text:
            yield from self.texts(fileids)
        else:
            for doc in self.docs(fileids):
                yield from doc.spans.get("ud_sents", [])

    def tokens(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Token | str"]:
        """Yield tokens from documents.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield token text strings instead of Token objects.

        Yields:
            Token objects or strings.
        """
        for doc in self.docs(fileids):
            for token in doc:
                yield token.text if as_text else token

    def ud_sents(
        self, fileids: str | list[str] | None = None
    ) -> Iterator["Span"]:
        """Yield UD sentence spans with full metadata.

        Convenience method that explicitly returns UD sentence spans.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Spans from doc.spans["ud_sents"] with _.citation and _.metadata.
        """
        for doc in self.docs(fileids):
            yield from doc.spans.get("ud_sents", [])

    def _get_citation_for_span(self, doc: Doc, span: "Span") -> str:
        """Get UD sentence ID for a span."""
        for ud_sent in doc.spans.get("ud_sents", []):
            if ud_sent.start <= span.start < ud_sent.end:
                citation = getattr(ud_sent._, "citation", None)
                if citation:
                    return citation
        fileid = doc._.fileid or "unknown"
        return f"{fileid}:{span.start}"


# Treebank-specific subclasses with auto-download support


class PROIELReader(DownloadableCorpusMixin, UDReader):
    """Reader for the Latin PROIEL treebank.

    Contains Vulgate New Testament, Caesar's Gallic War, Cicero's Letters,
    Palladius' Opus Agriculturae, and more.

    Example:
        >>> reader = PROIELReader()  # Auto-downloads if needed
        >>> for sent in reader.ud_sents():
        ...     print(f"{sent._.citation}: {sent.text}")
    """

    CORPUS_URL = "https://github.com/UniversalDependencies/UD_Latin-PROIEL.git"
    ENV_VAR = "UD_PROIEL_PATH"
    DEFAULT_SUBDIR = "ud_latin_proiel"
    _FILE_CHECK_PATTERN = "**/*.conllu"
    TREEBANK = "proiel"

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        auto_download: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
    ):
        """Initialize the PROIEL reader.

        Args:
            root: Root directory. If None, uses default location.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            auto_download: If True and corpus not found, offer to download.
            cache: If True, cache processed Doc objects.
            cache_maxsize: Maximum docs to cache.
        """
        if root is None:
            root = self._get_default_root(auto_download)
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            cache=cache,
            cache_maxsize=cache_maxsize,
        )


class PerseusUDReader(DownloadableCorpusMixin, UDReader):
    """Reader for the Latin Perseus treebank.

    Contains selections from the Ancient Greek and Latin Dependency Treebank.
    """

    CORPUS_URL = "https://github.com/UniversalDependencies/UD_Latin-Perseus.git"
    ENV_VAR = "UD_PERSEUS_PATH"
    DEFAULT_SUBDIR = "ud_latin_perseus"
    _FILE_CHECK_PATTERN = "**/*.conllu"
    TREEBANK = "perseus"

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        auto_download: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
    ):
        if root is None:
            root = self._get_default_root(auto_download)
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            cache=cache,
            cache_maxsize=cache_maxsize,
        )


class ITTBReader(DownloadableCorpusMixin, UDReader):
    """Reader for the Index Thomisticus Treebank.

    Contains works by Thomas Aquinas and related medieval authors.
    """

    CORPUS_URL = "https://github.com/UniversalDependencies/UD_Latin-ITTB.git"
    ENV_VAR = "UD_ITTB_PATH"
    DEFAULT_SUBDIR = "ud_latin_ittb"
    _FILE_CHECK_PATTERN = "**/*.conllu"
    TREEBANK = "ittb"

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        auto_download: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
    ):
        if root is None:
            root = self._get_default_root(auto_download)
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            cache=cache,
            cache_maxsize=cache_maxsize,
        )


class LLCTReader(DownloadableCorpusMixin, UDReader):
    """Reader for the Late Latin Charter Treebank.

    Contains late Latin legal documents and charters.
    """

    CORPUS_URL = "https://github.com/UniversalDependencies/UD_Latin-LLCT.git"
    ENV_VAR = "UD_LLCT_PATH"
    DEFAULT_SUBDIR = "ud_latin_llct"
    _FILE_CHECK_PATTERN = "**/*.conllu"
    TREEBANK = "llct"

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        auto_download: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
    ):
        if root is None:
            root = self._get_default_root(auto_download)
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            cache=cache,
            cache_maxsize=cache_maxsize,
        )


class UDanteReader(DownloadableCorpusMixin, UDReader):
    """Reader for Dante's Latin works treebank.

    Contains Dante Alighieri's Latin writings.
    """

    CORPUS_URL = "https://github.com/UniversalDependencies/UD_Latin-UDante.git"
    ENV_VAR = "UD_UDANTE_PATH"
    DEFAULT_SUBDIR = "ud_latin_udante"
    _FILE_CHECK_PATTERN = "**/*.conllu"
    TREEBANK = "udante"

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        auto_download: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
    ):
        if root is None:
            root = self._get_default_root(auto_download)
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            cache=cache,
            cache_maxsize=cache_maxsize,
        )


class CIRCSEReader(DownloadableCorpusMixin, UDReader):
    """Reader for the CIRCSE Latin treebank.

    Contains texts from the CIRCSE research center.
    """

    CORPUS_URL = "https://github.com/UniversalDependencies/UD_Latin-CIRCSE.git"
    ENV_VAR = "UD_CIRCSE_PATH"
    DEFAULT_SUBDIR = "ud_latin_circse"
    _FILE_CHECK_PATTERN = "**/*.conllu"
    TREEBANK = "circse"

    def __init__(
        self,
        root: str | Path | None = None,
        fileids: str | None = None,
        encoding: str = "utf-8",
        auto_download: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
    ):
        if root is None:
            root = self._get_default_root(auto_download)
        super().__init__(
            root=root,
            fileids=fileids,
            encoding=encoding,
            cache=cache,
            cache_maxsize=cache_maxsize,
        )


# Mapping of treebank names to reader classes
_TREEBANK_READERS = {
    "proiel": PROIELReader,
    "perseus": PerseusUDReader,
    "ittb": ITTBReader,
    "llct": LLCTReader,
    "udante": UDanteReader,
    "circse": CIRCSEReader,
}


class LatinUDReader:
    """Unified reader for all Latin Universal Dependencies treebanks.

    Provides access to all 6 Latin UD treebanks with auto-download support:
        - PROIEL: Vulgate, Caesar, Cicero, Palladius
        - Perseus: Classical texts from Perseus Digital Library
        - ITTB: Index Thomisticus (Thomas Aquinas)
        - LLCT: Late Latin Charter Treebank
        - UDante: Dante's Latin works
        - CIRCSE: CIRCSE Latin treebank

    Example:
        >>> reader = LatinUDReader()  # Downloads all treebanks if needed
        >>> for sent in reader.ud_sents():
        ...     print(f"{sent._.citation}: {sent.text}")

        >>> # Or select specific treebanks
        >>> reader = LatinUDReader(treebanks=["proiel", "perseus"])

        >>> # Access individual treebank readers
        >>> proiel = reader.readers["proiel"]
    """

    def __init__(
        self,
        treebanks: list[str] | None = None,
        auto_download: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
    ):
        """Initialize the unified Latin UD reader.

        Args:
            treebanks: List of treebank names to include. If None, includes all 6.
                Valid names: proiel, perseus, ittb, llct, udante, circse
            auto_download: If True and corpus not found, offer to download.
            cache: If True, cache processed Doc objects.
            cache_maxsize: Maximum docs to cache per treebank.
        """
        if treebanks is None:
            treebanks = list(LATIN_TREEBANKS.keys())

        # Validate treebank names
        invalid = set(treebanks) - set(LATIN_TREEBANKS.keys())
        if invalid:
            raise ValueError(
                f"Unknown treebank(s): {invalid}. "
                f"Valid names: {list(LATIN_TREEBANKS.keys())}"
            )

        self._treebank_names = treebanks
        self._auto_download = auto_download
        self._cache = cache
        self._cache_maxsize = cache_maxsize

        # Lazily initialize readers
        self._readers: dict[str, UDReader] | None = None

    @property
    def readers(self) -> dict[str, UDReader]:
        """Dict of treebank name -> reader instances (lazy loaded)."""
        if self._readers is None:
            self._readers = {}
            for name in self._treebank_names:
                reader_cls = _TREEBANK_READERS[name]
                self._readers[name] = reader_cls(
                    auto_download=self._auto_download,
                    cache=self._cache,
                    cache_maxsize=self._cache_maxsize,
                )
        return self._readers

    @property
    def treebanks(self) -> list[str]:
        """List of included treebank names."""
        return list(self._treebank_names)

    @classmethod
    def available_treebanks(cls) -> dict[str, str]:
        """Return dict of available treebanks with descriptions."""
        return {
            name: info["description"]
            for name, info in LATIN_TREEBANKS.items()
        }

    @classmethod
    def download_all(cls, treebanks: list[str] | None = None) -> None:
        """Download specified treebanks (or all if None).

        Args:
            treebanks: List of treebank names, or None for all.
        """
        if treebanks is None:
            treebanks = list(LATIN_TREEBANKS.keys())

        for name in treebanks:
            if name not in _TREEBANK_READERS:
                print(f"Unknown treebank: {name}, skipping")
                continue
            reader_cls = _TREEBANK_READERS[name]
            print(f"Downloading {name}...")
            reader_cls.download()

    def docs(
        self,
        treebanks: list[str] | None = None,
    ) -> Iterator[Doc]:
        """Yield spaCy Docs from specified treebanks.

        Args:
            treebanks: Treebank names to include, or None for all configured.

        Yields:
            spaCy Doc objects with UD annotations.
        """
        if treebanks is None:
            treebanks = self._treebank_names

        for name in treebanks:
            if name in self.readers:
                yield from self.readers[name].docs()

    def texts(
        self,
        treebanks: list[str] | None = None,
    ) -> Iterator[str]:
        """Yield raw text from sentences across treebanks.

        Args:
            treebanks: Treebank names to include, or None for all configured.

        Yields:
            Raw sentence text strings.
        """
        if treebanks is None:
            treebanks = self._treebank_names

        for name in treebanks:
            if name in self.readers:
                yield from self.readers[name].texts()

    def sents(
        self,
        treebanks: list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Span | str"]:
        """Yield sentences from specified treebanks.

        Args:
            treebanks: Treebank names to include, or None for all configured.
            as_text: If True, yield strings instead of Spans.

        Yields:
            Sentence Spans or strings.
        """
        if treebanks is None:
            treebanks = self._treebank_names

        for name in treebanks:
            if name in self.readers:
                yield from self.readers[name].sents(as_text=as_text)

    def ud_sents(
        self,
        treebanks: list[str] | None = None,
    ) -> Iterator["Span"]:
        """Yield UD sentence spans with citations from specified treebanks.

        Args:
            treebanks: Treebank names to include, or None for all configured.

        Yields:
            Spans with _.citation and _.metadata.
        """
        if treebanks is None:
            treebanks = self._treebank_names

        for name in treebanks:
            if name in self.readers:
                yield from self.readers[name].ud_sents()

    def tokens(
        self,
        treebanks: list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Token | str"]:
        """Yield tokens from specified treebanks.

        Args:
            treebanks: Treebank names to include, or None for all configured.
            as_text: If True, yield strings instead of Token objects.

        Yields:
            Token objects or strings.
        """
        if treebanks is None:
            treebanks = self._treebank_names

        for name in treebanks:
            if name in self.readers:
                yield from self.readers[name].tokens(as_text=as_text)
