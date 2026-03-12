"""Tests for AGLDT → UD converter.

Tests cover:
- POS mapping (AGLDT positional → UPOS)
- Morphological feature decoding
- Lemma normalization
- Tree restructuring (coordination, prepositions, subordinators, copula)
- Relation mapping
- Full sentence conversion against known gold data
"""

from __future__ import annotations

from pathlib import Path

import pytest

from latincyreaders.utils.agldt import (
    AGLDTToken,
    UDToken,
    convert_file,
    convert_sentence,
    decode_morph,
    normalize_lemma,
    parse_agldt_xml,
    restructure_coordination,
    restructure_copula,
    restructure_prepositions,
    restructure_subordinators,
    _initial_convert,
    _AGLDT_POS_TO_UPOS,
)


AGLDT_FILE = Path(__file__).parent / "fixtures" / "agldt" / "phi0959.phi006.perseus-lat1.tb.xml"


# ---------------------------------------------------------------------------
# POS mapping
# ---------------------------------------------------------------------------

class TestPOSMapping:
    def test_noun(self):
        assert _AGLDT_POS_TO_UPOS["n"] == "NOUN"

    def test_verb(self):
        assert _AGLDT_POS_TO_UPOS["v"] == "VERB"

    def test_adjective(self):
        assert _AGLDT_POS_TO_UPOS["a"] == "ADJ"

    def test_adverb(self):
        assert _AGLDT_POS_TO_UPOS["d"] == "ADV"

    def test_preposition(self):
        assert _AGLDT_POS_TO_UPOS["r"] == "ADP"

    def test_pronoun(self):
        assert _AGLDT_POS_TO_UPOS["p"] == "PRON"

    def test_conjunction(self):
        assert _AGLDT_POS_TO_UPOS["c"] == "CCONJ"

    def test_subordinator(self):
        assert _AGLDT_POS_TO_UPOS["g"] == "SCONJ"

    def test_determiner(self):
        assert _AGLDT_POS_TO_UPOS["l"] == "DET"

    def test_punctuation(self):
        assert _AGLDT_POS_TO_UPOS["u"] == "PUNCT"

    def test_numeral(self):
        assert _AGLDT_POS_TO_UPOS["m"] == "NUM"


# ---------------------------------------------------------------------------
# Morphological features
# ---------------------------------------------------------------------------

class TestMorphDecoding:
    def test_finite_verb(self):
        # v3spia--- = 3rd person singular present indicative active
        feats = decode_morph("v3spia---", "VERB")
        assert "Person=3" in feats
        assert "Number=Sing" in feats
        assert "Tense=Pres" in feats
        assert "Mood=Ind" in feats
        assert "VerbForm=Fin" in feats
        assert "Voice=Act" in feats

    def test_noun_nominative(self):
        # n-s---mn- = singular masculine nominative
        feats = decode_morph("n-s---mn-", "NOUN")
        assert "Number=Sing" in feats
        assert "Gender=Masc" in feats
        assert "Case=Nom" in feats
        assert "Person" not in feats
        assert "Tense" not in feats

    def test_adjective_accusative(self):
        # a-p---na- = plural neuter accusative
        feats = decode_morph("a-p---na-", "ADJ")
        assert "Number=Plur" in feats
        assert "Gender=Neut" in feats
        assert "Case=Acc" in feats

    def test_perfect_participle_uses_perf(self):
        # v-prppfa- = perfect participle passive feminine accusative
        feats = decode_morph("v-prppfa-", "VERB")
        assert "Tense=Perf" in feats
        assert "VerbForm=Part" in feats
        assert "Voice=Pass" in feats

    def test_finite_perfect_uses_past(self):
        # v3sria--- = 3rd singular perfect indicative active
        feats = decode_morph("v3sria---", "VERB")
        assert "Tense=Past" in feats
        assert "VerbForm=Fin" in feats
        assert "Mood=Ind" in feats

    def test_imperfect_uses_past(self):
        # v3siia--- = 3rd singular imperfect indicative active
        feats = decode_morph("v3siia---", "VERB")
        assert "Tense=Past" in feats

    def test_pluperfect(self):
        # v3slia--- = 3rd singular pluperfect indicative active
        feats = decode_morph("v3slia---", "VERB")
        assert "Tense=Pqp" in feats

    def test_future(self):
        # v3stia--- = 3rd singular future indicative active
        feats = decode_morph("v3stia---", "VERB")
        assert "Tense=Fut" in feats

    def test_subjunctive(self):
        # v3sssa--- = 3rd singular subjunctive active (some tense)
        feats = decode_morph("v3spsa---", "VERB")
        assert "Mood=Sub" in feats
        assert "VerbForm=Fin" in feats

    def test_infinitive(self):
        # v--pna--- = present infinitive active
        feats = decode_morph("v--pna---", "VERB")
        assert "VerbForm=Inf" in feats
        assert "Mood" not in feats

    def test_gerundive(self):
        # v--t-g--- = future gerundive (Gdv)
        feats = decode_morph("v-sfgpfa-", "VERB")
        assert "VerbForm=Part" in feats
        assert "Mood=Gdv" in feats

    def test_superlative_degree(self):
        feats = decode_morph("a-s---mns", "ADJ")
        assert "Degree=Sup" in feats

    def test_comparative_degree(self):
        feats = decode_morph("a-s---mnc", "ADJ")
        assert "Degree=Cmp" in feats

    def test_positive_degree_omitted(self):
        # Positive degree is unmarked in LatinCy
        feats = decode_morph("a-s---mnp", "ADJ")
        assert "Degree" not in feats

    def test_bare_preposition(self):
        feats = decode_morph("r--------", "ADP")
        assert feats == "_"

    def test_short_postag(self):
        feats = decode_morph("n", "NOUN")
        assert feats == "_"

    def test_imperative(self):
        feats = decode_morph("v2spma---", "VERB")
        assert "Mood=Imp" in feats
        assert "VerbForm=Fin" in feats


# ---------------------------------------------------------------------------
# Lemma normalization
# ---------------------------------------------------------------------------

class TestLemmaNormalization:
    def test_strip_digits(self):
        assert normalize_lemma("fero1") == "fero"

    def test_strip_multiple_digits(self):
        assert normalize_lemma("in12") == "in"

    def test_no_digits(self):
        assert normalize_lemma("corpus") == "corpus"

    def test_lowercase(self):
        assert normalize_lemma("Roma1") == "roma"

    def test_empty(self):
        assert normalize_lemma("") == "_"

    def test_punctuation_lemma(self):
        assert normalize_lemma("punc1") == "punc"

    def test_whitespace(self):
        assert normalize_lemma("  sum1  ") == "sum"


# ---------------------------------------------------------------------------
# Tree restructuring: Coordination
# ---------------------------------------------------------------------------

def _make_ud(id, form, head, rel, upos="NOUN", feats="_", postag="---------"):
    return UDToken(
        id=id, form=form, lemma=form.lower(), upos=upos,
        xpos="noun", feats=feats, head=head, deprel=rel,
        _prague_rel=rel, _prague_head=head, _prague_postag=postag,
    )


class TestCoordinationRestructuring:
    def test_simple_coordination(self):
        # Prague: et(COORD,→0) puer(SBJ_CO,→1) puella(SBJ_CO,→1)
        tokens = [
            _make_ud(1, "et", 0, "COORD", upos="CCONJ"),
            _make_ud(2, "puer", 1, "SBJ_CO"),
            _make_ud(3, "puella", 1, "SBJ_CO"),
        ]
        restructure_coordination(tokens)

        # puer should be first conjunct → head
        assert tokens[1].head == 0  # puer is now root
        # puella should attach to puer via conj
        assert tokens[2].head == 2
        assert tokens[2].deprel == "conj"
        # et should be cc
        assert tokens[0].deprel == "cc"

    def test_coordination_cc_attaches_to_following(self):
        # et between conjuncts should attach to the next conjunct
        tokens = [
            _make_ud(1, "puer", 3, "SBJ_CO"),
            _make_ud(2, "et", 3, "COORD", upos="CCONJ"),
            _make_ud(3, "puella", 0, "COORD", upos="CCONJ"),
            _make_ud(4, "vir", 3, "SBJ_CO"),
        ]
        # Actually this structure is: COORD(3) has children 1(SBJ_CO), 2(COORD?), 4(SBJ_CO)
        # Let me fix this to a proper Prague structure:
        tokens = [
            _make_ud(1, "puer", 2, "SBJ_CO"),
            _make_ud(2, "et", 0, "COORD", upos="CCONJ"),
            _make_ud(3, "puella", 2, "SBJ_CO"),
        ]
        restructure_coordination(tokens)

        assert tokens[0].head == 0  # puer promoted
        assert tokens[2].head == 1  # puella → puer
        assert tokens[2].deprel == "conj"
        assert tokens[1].head == 3  # et → puella (following conjunct)
        assert tokens[1].deprel == "cc"


# ---------------------------------------------------------------------------
# Tree restructuring: Prepositions
# ---------------------------------------------------------------------------

class TestPrepositionRestructuring:
    def test_simple_pp(self):
        # Prague: fert(PRED,→0) In(AuxP,→1) corpora(OBJ,→2)
        # But we need the real structure. In the example:
        # fert heads sentence, In is AuxP→fert(?), corpora is child of In
        tokens = [
            _make_ud(1, "fert", 0, "PRED", upos="VERB"),
            _make_ud(2, "In", 1, "AuxP", upos="ADP"),
            _make_ud(3, "corpora", 2, "OBJ"),
        ]
        restructure_prepositions(tokens)

        # corpora should be promoted to head→fert
        assert tokens[2].head == 1  # corpora → fert
        assert tokens[2].deprel == "obl"
        # In should be case dependent of corpora
        assert tokens[1].head == 3  # In → corpora
        assert tokens[1].deprel == "case"

    def test_pp_modifying_noun(self):
        # "urbis de muris" — PP modifying a noun
        tokens = [
            _make_ud(1, "urbis", 0, "PRED"),
            _make_ud(2, "de", 1, "AuxP", upos="ADP"),
            _make_ud(3, "muris", 2, "ADV"),
        ]
        restructure_prepositions(tokens)

        # When governor is a noun, relation should be nmod
        # But here governor (urbis) has UPOS=NOUN → nmod
        assert tokens[1].head == 3
        assert tokens[1].deprel == "case"

    def test_pp_with_adjective(self):
        # "In nova corpora" — PP with adjective modifier on noun
        tokens = [
            _make_ud(1, "fert", 0, "PRED", upos="VERB"),
            _make_ud(2, "In", 1, "AuxP", upos="ADP"),
            _make_ud(3, "nova", 4, "ATR", upos="ADJ"),
            _make_ud(4, "corpora", 2, "OBJ"),
        ]
        restructure_prepositions(tokens)

        assert tokens[1].head == 4  # In → corpora
        assert tokens[1].deprel == "case"
        assert tokens[3].head == 1  # corpora → fert
        assert tokens[2].head == 4  # nova stays on corpora


# ---------------------------------------------------------------------------
# Tree restructuring: Subordinators
# ---------------------------------------------------------------------------

class TestSubordinatorRestructuring:
    def test_simple_subord(self):
        # Prague: venit(PRED,→0) nam(AuxC,→1) vidit(ADV,→2)
        tokens = [
            _make_ud(1, "venit", 0, "PRED", upos="VERB"),
            _make_ud(2, "nam", 1, "AuxC", upos="SCONJ"),
            _make_ud(3, "vidit", 2, "ADV", upos="VERB"),
        ]
        restructure_subordinators(tokens)

        # vidit should be promoted
        assert tokens[2].head == 1  # vidit → venit
        assert tokens[2].deprel == "advcl"
        # nam should be mark
        assert tokens[1].head == 3  # nam → vidit
        assert tokens[1].deprel == "mark"


# ---------------------------------------------------------------------------
# Tree restructuring: Copula
# ---------------------------------------------------------------------------

class TestCopulaRestructuring:
    def test_sum_with_pnom(self):
        # Prague: erat(PRED,→0) vultus(SBJ,→1) unus(PNOM,→1)
        tokens = [
            _make_ud(1, "erat", 0, "PRED", upos="VERB"),
            _make_ud(2, "vultus", 1, "SBJ"),
            _make_ud(3, "unus", 1, "PNOM", upos="ADJ"),
        ]
        tokens[0].lemma = "sum"
        restructure_copula(tokens)

        # PNOM (unus) should be promoted to root
        assert tokens[2].head == 0  # unus → root
        # erat should be cop of unus
        assert tokens[0].head == 3  # erat → unus
        assert tokens[0].deprel == "cop"
        assert tokens[0].upos == "AUX"
        # SBJ should now point to unus
        assert tokens[1].head == 3


# ---------------------------------------------------------------------------
# Full sentence conversion
# ---------------------------------------------------------------------------

class TestFullConversion:
    def test_met1_sentence1(self):
        """In nova fert animus mutatas dicere formas corpora ;"""
        agldt = [
            AGLDTToken(1, "In", "in1", "r--------", "AuxP", 5),
            AGLDTToken(2, "nova", "novus1", "a-p---na-", "ATR", 8),
            AGLDTToken(3, "fert", "fero1", "v3spia---", "PRED", 0),
            AGLDTToken(4, "animus", "animus1", "n-s---mn-", "SBJ", 3),
            AGLDTToken(5, "mutatas", "muto1", "v-prppfa-", "ATR", 7),
            AGLDTToken(6, "dicere", "dico2", "v--pna---", "OBJ", 3),
            AGLDTToken(7, "formas", "forma1", "n-p---fa-", "OBJ", 6),
            AGLDTToken(8, "corpora", "corpus1", "n-p---na-", "OBJ", 1),
            AGLDTToken(9, ";", "punc1", "u--------", "AuxK", 0),
        ]
        ud = convert_sentence(agldt)

        # Check POS
        assert ud[0].upos == "ADP"   # In
        assert ud[1].upos == "ADJ"   # nova
        assert ud[2].upos == "VERB"  # fert
        assert ud[3].upos == "NOUN"  # animus
        assert ud[8].upos == "PUNCT" # ;

        # Check lemmas
        assert ud[0].lemma == "in"
        assert ud[2].lemma == "fero"
        assert ud[7].lemma == "corpus"

        # Check morph features
        assert "Tense=Pres" in ud[2].feats  # fert
        assert "Tense=Perf" in ud[4].feats  # mutatas (participle → Perf)
        assert "VerbForm=Part" in ud[4].feats
        assert "VerbForm=Inf" in ud[5].feats  # dicere

        # Structural checks after restructuring:
        # "In" should be case (preposition restructured)
        assert ud[0].deprel == "case"
        # fert should be ROOT
        assert ud[2].deprel == "ROOT"
        assert ud[2].head == 0
        # animus should be nsubj
        assert ud[3].deprel == "nsubj"
        # ; should be punct
        assert ud[8].deprel == "punct"

    def test_no_cycles(self):
        """Verify conversion produces no cycles."""
        agldt = [
            AGLDTToken(1, "In", "in1", "r--------", "AuxP", 5),
            AGLDTToken(2, "nova", "novus1", "a-p---na-", "ATR", 8),
            AGLDTToken(3, "fert", "fero1", "v3spia---", "PRED", 0),
            AGLDTToken(4, "animus", "animus1", "n-s---mn-", "SBJ", 3),
            AGLDTToken(5, "mutatas", "muto1", "v-prppfa-", "ATR", 7),
            AGLDTToken(6, "dicere", "dico2", "v--pna---", "OBJ", 3),
            AGLDTToken(7, "formas", "forma1", "n-p---fa-", "OBJ", 6),
            AGLDTToken(8, "corpora", "corpus1", "n-p---na-", "OBJ", 1),
            AGLDTToken(9, ";", "punc1", "u--------", "AuxK", 0),
        ]
        ud = convert_sentence(agldt)

        # Walk from each token to root — should terminate
        for tok in ud:
            visited = set()
            current = tok
            lookup = {t.id: t for t in ud}
            while current.head != 0:
                assert current.id not in visited, f"Cycle at token {current.id} ({current.form})"
                visited.add(current.id)
                current = lookup[current.head]

    def test_single_root(self):
        """Each converted sentence should have exactly one root."""
        agldt = [
            AGLDTToken(1, "fert", "fero1", "v3spia---", "PRED", 0),
            AGLDTToken(2, "animus", "animus1", "n-s---mn-", "SBJ", 1),
            AGLDTToken(3, ".", "punc1", "u--------", "AuxK", 0),
        ]
        ud = convert_sentence(agldt)
        roots = [t for t in ud if t.head == 0]
        assert len(roots) == 1


# ---------------------------------------------------------------------------
# Full file conversion
# ---------------------------------------------------------------------------

class TestFileConversion:
    @pytest.fixture(scope="class")
    def conlluc_output(self):
        if not AGLDT_FILE.exists():
            pytest.skip("AGLDT fixture not found")
        return convert_file(
            AGLDT_FILE,
            fileid="ovid.metamorphoses.part.1",
            collection="cltk-tesserae",
            model_name="agldt-gold",
        )

    def test_has_header(self, conlluc_output):
        assert "# generator = latincy-readers" in conlluc_output
        assert "# annotation_status = silver" in conlluc_output
        assert "# source = agldt" in conlluc_output
        assert "# tree_restructuring = prague-to-ud" in conlluc_output

    def test_has_sentences(self, conlluc_output):
        sent_ids = [l for l in conlluc_output.split("\n") if l.startswith("# sent_id")]
        assert len(sent_ids) > 300  # Met 1 has ~317 sentences

    def test_no_empty_deprels(self, conlluc_output):
        for line in conlluc_output.split("\n"):
            if line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) == 10:
                    assert parts[7] != "", f"Empty deprel in: {line}"

    def test_valid_upos(self, conlluc_output):
        valid_upos = {
            "NOUN", "VERB", "ADJ", "ADV", "ADP", "PRON", "DET",
            "NUM", "PUNCT", "CCONJ", "SCONJ", "INTJ", "AUX",
            "PROPN", "X", "PART",
        }
        for line in conlluc_output.split("\n"):
            if line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) == 10:
                    assert parts[3] in valid_upos, f"Invalid UPOS '{parts[3]}' in: {line}"

    def test_no_cycles_in_full_file(self, conlluc_output):
        """Check all sentences for dependency cycles."""
        sentences: list[list[dict]] = []
        current: list[dict] = []

        for line in conlluc_output.split("\n"):
            if not line or line.startswith("#"):
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) == 10:
                current.append({"id": int(parts[0]), "head": int(parts[6])})

        if current:
            sentences.append(current)

        for sent_idx, sent in enumerate(sentences):
            lookup = {t["id"]: t for t in sent}
            for tok in sent:
                visited: set[int] = set()
                current_id = tok["id"]
                while current_id != 0:
                    assert current_id not in visited, (
                        f"Cycle in sentence {sent_idx + 1} at token {current_id}"
                    )
                    visited.add(current_id)
                    head = lookup.get(current_id, {}).get("head", 0)
                    current_id = head

    def test_single_root_per_sentence(self, conlluc_output):
        """Each sentence should have exactly one ROOT."""
        sentences: list[list[dict]] = []
        current: list[dict] = []

        for line in conlluc_output.split("\n"):
            if not line or line.startswith("#"):
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) == 10:
                current.append({"id": int(parts[0]), "head": int(parts[6]), "deprel": parts[7]})

        if current:
            sentences.append(current)

        for sent_idx, sent in enumerate(sentences):
            roots = [t for t in sent if t["head"] == 0]
            assert len(roots) == 1, (
                f"Sentence {sent_idx + 1} has {len(roots)} roots "
                f"(expected 1): {roots}"
            )
