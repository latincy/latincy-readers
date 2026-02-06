"""Tests for UDReader (Universal Dependencies treebank reader)."""

import pytest
from pathlib import Path

from latincyreaders.readers.ud import UDReader, UDSentence


class TestUDReader:
    """Test suite for UDReader."""

    @pytest.fixture
    def reader(self, ud_dir):
        """Create a UDReader with test fixtures."""
        return UDReader(root=ud_dir, fileids="*.conllu")

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of .conllu files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".conllu") for f in fileids)

    def test_fileids_contains_test_file(self, reader):
        """Test fixture file is discovered."""
        fileids = reader.fileids()
        assert "sample.conllu" in fileids

    def test_root_is_path(self, reader, ud_dir):
        """root property returns correct Path."""
        assert reader.root == ud_dir.resolve()

    # -------------------------------------------------------------------------
    # Raw text access (no NLP needed)
    # -------------------------------------------------------------------------

    def test_texts_yields_strings(self, reader):
        """texts() yields raw strings from # text = comments."""
        texts = list(reader.texts())
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_texts_contains_latin(self, reader):
        """Text content is Latin from test fixture."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        # Check for known content from Caesar's Gallic War
        assert "Gallia" in all_text
        assert "divisa" in all_text

    # -------------------------------------------------------------------------
    # spaCy Doc access
    # -------------------------------------------------------------------------

    def test_docs_yields_spacy_docs(self, reader):
        """docs() yields spaCy Doc objects."""
        from spacy.tokens import Doc

        docs = list(reader.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_have_fileid(self, reader):
        """Docs have fileid extension set."""
        doc = next(reader.docs())
        assert doc._.fileid is not None
        assert doc._.fileid.endswith(".conllu")

    def test_docs_have_metadata(self, reader):
        """Docs have metadata extension set."""
        doc = next(reader.docs())
        assert doc._.metadata is not None
        assert doc._.metadata.get("source") == "universal_dependencies"

    def test_docs_have_ud_sents_spans(self, reader):
        """Docs have 'ud_sents' span group."""
        doc = next(reader.docs())
        assert "ud_sents" in doc.spans
        assert len(doc.spans["ud_sents"]) > 0

    def test_ud_sents_have_citations(self, reader):
        """UD sentence spans have citation extensions (sent_id)."""
        doc = next(reader.docs())
        for span in doc.spans["ud_sents"]:
            assert span._.citation is not None
            # Citations should be sent_ids from the fixture
            assert span._.citation.startswith("test-s")

    def test_ud_sents_have_metadata(self, reader):
        """UD sentence spans have metadata with original text."""
        doc = next(reader.docs())
        span = doc.spans["ud_sents"][0]
        assert span._.metadata is not None
        assert "text" in span._.metadata

    # -------------------------------------------------------------------------
    # Token UD extensions
    # -------------------------------------------------------------------------

    def test_tokens_have_ud_extension(self, reader):
        """All tokens have the _.ud extension populated."""
        doc = next(reader.docs())
        for token in doc:
            assert token._.ud is not None

    def test_ud_extension_has_required_fields(self, reader):
        """The _.ud dict contains all CoNLL-U fields."""
        doc = next(reader.docs())
        token = doc[0]  # First token: "Gallia"
        ud = token._.ud

        expected_fields = [
            "id", "form", "lemma", "upos", "xpos",
            "feats", "head", "deprel", "deps", "misc"
        ]
        for field in expected_fields:
            assert field in ud, f"Missing field: {field}"

    def test_ud_feats_is_dict(self, reader):
        """The feats field is parsed into a dict."""
        doc = next(reader.docs())
        # Find a token with features
        for token in doc:
            if token._.ud["feats"]:
                assert isinstance(token._.ud["feats"], dict)
                break

    def test_ud_misc_is_dict(self, reader):
        """The misc field is parsed into a dict."""
        doc = next(reader.docs())
        # The misc field should be a dict (possibly empty)
        for token in doc:
            assert isinstance(token._.ud["misc"], dict)

    def test_ud_form_matches_token_text(self, reader):
        """The UD form field matches the spaCy token text."""
        doc = next(reader.docs())
        for token in doc:
            assert token.text == token._.ud["form"]

    # -------------------------------------------------------------------------
    # spaCy attributes populated from UD
    # -------------------------------------------------------------------------

    def test_tokens_have_lemma(self, reader):
        """Tokens have lemma_ populated from UD."""
        doc = next(reader.docs())
        # "Gallia" lemma should be "Gallia"
        assert doc[0].lemma_ == "Gallia"
        # "est" lemma should be "sum"
        assert doc[1].lemma_ == "sum"

    def test_tokens_have_pos(self, reader):
        """Tokens have pos_ populated from UD UPOS."""
        doc = next(reader.docs())
        # "Gallia" should be PROPN
        assert doc[0].pos_ == "PROPN"
        # "est" should be AUX
        assert doc[1].pos_ == "AUX"

    def test_tokens_have_tag(self, reader):
        """Tokens have tag_ populated from UD XPOS."""
        doc = next(reader.docs())
        # XPOS in our fixture is "_" (not specified)
        # spaCy normalizes "_" to empty string
        assert doc[0].tag_ == "" or doc[0].tag_ == "_"

    def test_tokens_have_dep(self, reader):
        """Tokens have dep_ populated from UD deprel."""
        doc = next(reader.docs())
        # "Gallia" should be nsubj
        assert doc[0].dep_ == "nsubj"
        # "divisa" should be root
        assert doc[3].dep_ == "root"

    def test_root_token_head_is_self(self, reader):
        """Root tokens point to themselves."""
        doc = next(reader.docs())
        # "divisa" is root (head=0 in UD)
        root_token = doc[3]  # divisa
        assert root_token.head == root_token

    def test_non_root_token_has_correct_head(self, reader):
        """Non-root tokens point to correct head."""
        doc = next(reader.docs())
        # "Gallia" (idx 0) has head=4 (divisa) in UD, which is idx 3 in 0-based
        gallia = doc[0]
        assert gallia.head == doc[3]  # Should point to "divisa"

    # -------------------------------------------------------------------------
    # Sentence and token iteration
    # -------------------------------------------------------------------------

    def test_sents_yields_spans(self, reader):
        """sents() yields Span objects."""
        from spacy.tokens import Span

        sents = list(reader.sents())
        assert len(sents) > 0
        assert all(isinstance(s, Span) for s in sents)

    def test_sents_count_matches_fixture(self, reader):
        """Number of sentences matches the fixture."""
        sents = list(reader.sents())
        # Our fixture has 3 sentences
        assert len(sents) == 3

    def test_sents_as_text_yields_strings(self, reader):
        """sents(as_text=True) yields strings."""
        sents = list(reader.sents(as_text=True))
        assert len(sents) > 0
        assert all(isinstance(s, str) for s in sents)

    def test_ud_sents_method(self, reader):
        """ud_sents() yields spans with citations."""
        sents = list(reader.ud_sents())
        assert len(sents) == 3
        citations = [s._.citation for s in sents]
        assert "test-s1" in citations
        assert "test-s2" in citations
        assert "test-s3" in citations

    def test_tokens_yields_tokens(self, reader):
        """tokens() yields Token objects."""
        from spacy.tokens import Token

        tokens = list(reader.tokens())
        assert len(tokens) > 0
        assert all(isinstance(t, Token) for t in tokens)

    def test_tokens_as_text_yields_strings(self, reader):
        """tokens(as_text=True) yields strings."""
        tokens = list(reader.tokens(as_text=True))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def test_cache_enabled_by_default(self, reader):
        """Cache is enabled by default."""
        assert reader.cache_enabled is True

    def test_cache_stats_initial(self, reader):
        """Initial cache stats are zero."""
        stats = reader.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

    def test_cache_hit_on_second_access(self, reader):
        """Second access to same doc hits cache."""
        # First access
        list(reader.docs())
        stats1 = reader.cache_stats()

        # Second access
        list(reader.docs())
        stats2 = reader.cache_stats()

        assert stats2["hits"] > stats1["hits"]

    def test_clear_cache(self, reader):
        """clear_cache() resets cache and stats."""
        list(reader.docs())  # Populate cache
        reader.clear_cache()
        stats = reader.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0


class TestUDParsing:
    """Test CoNLL-U parsing logic."""

    @pytest.fixture
    def reader(self, ud_dir):
        """Create a UDReader with test fixtures."""
        return UDReader(root=ud_dir)

    def test_parse_conllu_extracts_sentences(self, reader, ud_dir):
        """_parse_conllu extracts UDSentence objects."""
        path = ud_dir / "sample.conllu"
        sentences = list(reader._parse_conllu(path))
        assert len(sentences) == 3
        assert all(isinstance(s, UDSentence) for s in sentences)

    def test_parse_conllu_extracts_sent_id(self, reader, ud_dir):
        """Sentence IDs are extracted correctly."""
        path = ud_dir / "sample.conllu"
        sentences = list(reader._parse_conllu(path))
        assert sentences[0].sent_id == "test-s1"
        assert sentences[1].sent_id == "test-s2"
        assert sentences[2].sent_id == "test-s3"

    def test_parse_conllu_extracts_text(self, reader, ud_dir):
        """Sentence text is extracted from # text = comments."""
        path = ud_dir / "sample.conllu"
        sentences = list(reader._parse_conllu(path))
        assert "Gallia est omnis divisa" in sentences[0].text
        assert "Quarum unam incolunt Belgae" in sentences[1].text

    def test_parse_conllu_extracts_tokens(self, reader, ud_dir):
        """Tokens are extracted with all fields."""
        path = ud_dir / "sample.conllu"
        sent = next(reader._parse_conllu(path))
        assert len(sent.tokens) == 8  # First sentence has 8 tokens

        # Check first token
        tok = sent.tokens[0]
        assert tok["form"] == "Gallia"
        assert tok["lemma"] == "Gallia"
        assert tok["upos"] == "PROPN"

    def test_parse_conllu_parses_feats(self, reader, ud_dir):
        """Morphological features are parsed into dict."""
        path = ud_dir / "sample.conllu"
        sent = next(reader._parse_conllu(path))
        tok = sent.tokens[0]  # Gallia

        feats = tok["feats"]
        assert feats["Case"] == "Nom"
        assert feats["Gender"] == "Fem"
        assert feats["Number"] == "Sing"


class TestSpacingAndPunctuation:
    """Test that spacing and punctuation are handled correctly."""

    @pytest.fixture
    def reader(self, ud_dir):
        return UDReader(root=ud_dir)

    def test_space_after_no_handled(self, reader):
        """SpaceAfter=No in MISC removes trailing space."""
        doc = next(reader.docs())
        # "tres" (token 6, 0-indexed) has SpaceAfter=No
        tres_token = doc[6]
        assert tres_token.text == "tres"
        assert not tres_token.whitespace_

    def test_default_spacing(self, reader):
        """Tokens without SpaceAfter=No have space."""
        doc = next(reader.docs())
        # "Gallia" should have trailing space
        gallia_token = doc[0]
        assert gallia_token.whitespace_ == " "


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_directory(self, tmp_path):
        """Reader handles empty directory gracefully."""
        reader = UDReader(root=tmp_path)
        fileids = reader.fileids()
        assert fileids == []
        docs = list(reader.docs())
        assert docs == []

    def test_specific_fileid(self, ud_dir):
        """Can request specific file by fileid."""
        reader = UDReader(root=ud_dir)
        docs = list(reader.docs("sample.conllu"))
        assert len(docs) == 1

    def test_list_of_fileids(self, ud_dir):
        """Can request list of fileids."""
        reader = UDReader(root=ud_dir)
        docs = list(reader.docs(["sample.conllu"]))
        assert len(docs) == 1


class TestLatinUDReader:
    """Tests for the unified LatinUDReader."""

    def test_available_treebanks_returns_all_six(self):
        """available_treebanks() returns all 6 Latin treebanks."""
        from latincyreaders.readers.ud import LatinUDReader

        treebanks = LatinUDReader.available_treebanks()
        assert len(treebanks) == 6
        assert "proiel" in treebanks
        assert "perseus" in treebanks
        assert "ittb" in treebanks
        assert "llct" in treebanks
        assert "udante" in treebanks
        assert "circse" in treebanks

    def test_available_treebanks_has_descriptions(self):
        """Each treebank has a description."""
        from latincyreaders.readers.ud import LatinUDReader

        treebanks = LatinUDReader.available_treebanks()
        for name, desc in treebanks.items():
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_init_with_specific_treebanks(self):
        """Can initialize with subset of treebanks."""
        from latincyreaders.readers.ud import LatinUDReader

        reader = LatinUDReader(
            treebanks=["proiel", "perseus"],
            auto_download=False,
        )
        assert reader.treebanks == ["proiel", "perseus"]

    def test_init_with_invalid_treebank_raises(self):
        """Invalid treebank name raises ValueError."""
        from latincyreaders.readers.ud import LatinUDReader
        import pytest

        with pytest.raises(ValueError, match="Unknown treebank"):
            LatinUDReader(treebanks=["invalid_treebank"])

    def test_default_includes_all_treebanks(self):
        """Default initialization includes all 6 treebanks."""
        from latincyreaders.readers.ud import LatinUDReader

        reader = LatinUDReader(auto_download=False)
        assert len(reader.treebanks) == 6


class TestCIRCSEReader:
    """Tests for the CIRCSEReader class."""

    def test_circse_reader_attributes(self):
        """CIRCSEReader has correct class attributes."""
        from latincyreaders.readers.ud import CIRCSEReader

        assert CIRCSEReader.TREEBANK == "circse"
        assert "UD_Latin-CIRCSE" in CIRCSEReader.CORPUS_URL
        assert CIRCSEReader.ENV_VAR == "UD_CIRCSE_PATH"
        assert CIRCSEReader.DEFAULT_SUBDIR == "ud_latin_circse"
