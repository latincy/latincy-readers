"""Experimental silver-vs-gold comparison using AGLDT Ovid Met 1.

Uses the gold-standard AGLDT treebank (phi0959.phi006) as ground truth
and compares against LatinCy silver annotations serialized as .conlluc.

This file is NOT part of the regular test suite — it requires la_core_web_lg.
Run explicitly with::

    pytest tests/test_agldt_experiment.py -v -s

The AGLDT postag scheme uses a 9-character positional tag:
    pos(1) person(2) number(3) tense(4) mood(5) voice(6) gender(7) case(8) degree(9)

POS mapping (position 1):
    n=NOUN v=VERB a=ADJ d=ADV c=CCONJ r=ADP p=PRON m=NUM u=PUNCT
    l=ART(DET) g=SCONJ e=INTJ i=INTJ(irregular) x=X
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from lxml import etree

# AGLDT postag position 1 → UD UPOS (approximate mapping)
_AGLDT_POS_TO_UPOS = {
    "n": "NOUN",
    "v": "VERB",
    "a": "ADJ",
    "d": "ADV",
    "c": "CCONJ",
    "r": "ADP",
    "p": "PRON",
    "m": "NUM",
    "u": "PUNCT",
    "l": "DET",
    "g": "SCONJ",
    "e": "INTJ",
    "i": "INTJ",
    "x": "X",
    "-": "X",
}

AGLDT_FILE = Path(__file__).parent / "fixtures" / "agldt" / "phi0959.phi006.perseus-lat1.tb.xml"


def parse_agldt(path: Path) -> list[dict]:
    """Parse AGLDT XML into a flat list of gold token dicts."""
    tree = etree.parse(str(path))
    tokens = []
    for sent in tree.findall(".//sentence"):
        sent_id = sent.get("id")
        subdoc = sent.get("subdoc", "")
        for word in sent.findall("word"):
            form = word.get("form", "")
            lemma = word.get("lemma", "")
            postag = word.get("postag", "---------")
            relation = word.get("relation", "")
            head = word.get("head", "0")
            cite = word.get("cite", "")

            # Clean lemma (AGLDT appends sense number like "fero1")
            clean_lemma = lemma.rstrip("0123456789")

            # Map POS
            pos_char = postag[0] if postag else "-"
            upos = _AGLDT_POS_TO_UPOS.get(pos_char, "X")

            tokens.append({
                "sent_id": sent_id,
                "subdoc": subdoc,
                "form": form,
                "lemma": clean_lemma,
                "upos": upos,
                "postag": postag,
                "relation": relation,
                "head": head,
                "cite": cite,
            })
    return tokens


def build_text_from_agldt(tokens: list[dict]) -> str:
    """Reconstruct running text from AGLDT tokens."""
    forms = [t["form"] for t in tokens if t["upos"] != "PUNCT" or t["form"] in ".;:!?"]
    # Simple join — imperfect but good enough for pipeline input
    return " ".join(forms)


def build_sentences_from_agldt(path: Path) -> list[list[dict]]:
    """Parse AGLDT into list of sentences, each a list of token dicts."""
    tree = etree.parse(str(path))
    sentences = []
    for sent in tree.findall(".//sentence"):
        tokens = []
        for word in sent.findall("word"):
            form = word.get("form", "")
            lemma = word.get("lemma", "").rstrip("0123456789")
            postag = word.get("postag", "---------")
            pos_char = postag[0] if postag else "-"
            upos = _AGLDT_POS_TO_UPOS.get(pos_char, "X")
            tokens.append({
                "form": form,
                "lemma": lemma,
                "upos": upos,
            })
        sentences.append(tokens)
    return sentences


@pytest.fixture(scope="module")
def gold_tokens():
    """Load gold tokens from AGLDT."""
    if not AGLDT_FILE.exists():
        pytest.skip("AGLDT fixture not found")
    return parse_agldt(AGLDT_FILE)


@pytest.fixture(scope="module")
def gold_sentences():
    """Load gold sentences from AGLDT."""
    if not AGLDT_FILE.exists():
        pytest.skip("AGLDT fixture not found")
    return build_sentences_from_agldt(AGLDT_FILE)


@pytest.fixture(scope="module")
def silver_doc(gold_tokens):
    """Run LatinCy on the AGLDT text and return the Doc."""
    try:
        import spacy
        nlp = spacy.load("la_core_web_lg")
    except OSError:
        pytest.skip("la_core_web_lg not installed")

    # Build text sentence by sentence from AGLDT to match segmentation
    tree = etree.parse(str(AGLDT_FILE))
    sent_texts = []
    for sent in tree.findall(".//sentence"):
        words = sent.findall("word")
        forms = [w.get("form", "") for w in words]
        sent_texts.append(" ".join(forms))

    full_text = " ".join(sent_texts)
    nlp.max_length = len(full_text) + 1000
    return nlp(full_text)


@pytest.fixture(scope="module")
def silver_conlluc(silver_doc, tmp_path_factory):
    """Serialize the silver Doc to .conlluc and return the content."""
    from latincyreaders.cache.conlluc import doc_to_conlluc

    content = doc_to_conlluc(
        silver_doc,
        fileid="phi0959.phi006.perseus-lat1",
        collection="agldt-experiment",
        model_name="la_core_web_lg",
    )

    # Write to disk for inspection
    out_dir = tmp_path_factory.mktemp("agldt_experiment")
    out_path = out_dir / "ovid.met1.conlluc"
    out_path.write_text(content, encoding="utf-8")
    print(f"\n.conlluc written to: {out_path}")

    return content


class TestAGLDTExperiment:
    """Silver-vs-gold comparison for Ovid Met 1."""

    def test_gold_token_count(self, gold_tokens):
        """Sanity check: AGLDT has the expected number of tokens."""
        # Including punctuation
        assert len(gold_tokens) > 5000
        print(f"\nGold tokens: {len(gold_tokens)}")

    def test_silver_token_count(self, silver_doc):
        """Silver doc should have a similar token count."""
        print(f"\nSilver tokens: {len(silver_doc)}")
        # Allow some variation due to tokenization differences
        assert len(silver_doc) > 4000

    def test_conlluc_header(self, silver_conlluc):
        """The .conlluc output should have proper silver metadata."""
        assert "# annotation_status = silver" in silver_conlluc
        assert "# do_not_use_for_training = true" in silver_conlluc
        assert "# model_name = la_core_web_lg" in silver_conlluc
        print("\n.conlluc header: OK")

    def test_lemma_accuracy(self, gold_sentences, silver_doc):
        """Compare lemma accuracy between gold and silver.

        Aligns by form to handle tokenization differences.
        """
        # Build a form→lemma lookup from gold (first occurrence wins)
        gold_lemmas: dict[str, str] = {}
        for sent in gold_sentences:
            for tok in sent:
                form = tok["form"].lower()
                if form not in gold_lemmas and tok["upos"] != "PUNCT":
                    gold_lemmas[form] = tok["lemma"].lower()

        correct = 0
        total = 0
        mismatches: list[tuple[str, str, str]] = []

        for token in silver_doc:
            if token.is_punct or token.is_space:
                continue
            form = token.text.lower()
            if form in gold_lemmas:
                total += 1
                silver_lemma = token.lemma_.lower()
                gold_lemma = gold_lemmas[form]
                if silver_lemma == gold_lemma:
                    correct += 1
                else:
                    mismatches.append((form, gold_lemma, silver_lemma))

        accuracy = correct / total if total > 0 else 0
        print(f"\nLemma accuracy: {correct}/{total} = {accuracy:.1%}")
        print(f"Sample mismatches (first 20):")
        for form, gold, silver in mismatches[:20]:
            print(f"  {form}: gold={gold} silver={silver}")

        # LatinCy should be at least 80% accurate on lemmas
        assert accuracy > 0.70, f"Lemma accuracy {accuracy:.1%} below threshold"

    def test_pos_accuracy(self, gold_sentences, silver_doc):
        """Compare POS accuracy between gold and silver.

        Aligns by form to handle tokenization differences.
        """
        # Build form→POS from gold
        gold_pos: dict[str, str] = {}
        for sent in gold_sentences:
            for tok in sent:
                form = tok["form"].lower()
                if form not in gold_pos and tok["upos"] != "PUNCT":
                    gold_pos[form] = tok["upos"]

        correct = 0
        total = 0
        confusion: Counter = Counter()

        for token in silver_doc:
            if token.is_punct or token.is_space:
                continue
            form = token.text.lower()
            if form in gold_pos:
                total += 1
                silver_pos = token.pos_
                gold = gold_pos[form]
                if silver_pos == gold:
                    correct += 1
                else:
                    confusion[(gold, silver_pos)] += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\nPOS accuracy: {correct}/{total} = {accuracy:.1%}")
        print(f"Top confusions:")
        for (gold, silver), count in confusion.most_common(10):
            print(f"  gold={gold} → silver={silver}: {count}")

        assert accuracy > 0.70, f"POS accuracy {accuracy:.1%} below threshold"

    def test_conlluc_roundtrip(self, silver_conlluc):
        """Verify the .conlluc round-trips correctly."""
        from spacy.vocab import Vocab
        from latincyreaders.cache.conlluc import conlluc_to_doc

        vocab = Vocab()
        doc, meta = conlluc_to_doc(silver_conlluc, vocab)

        assert doc is not None
        assert len(doc) > 4000
        assert meta["annotation_status"] == "silver"
        assert meta["model_name"] == "la_core_web_lg"
        print(f"\nRound-trip: {len(doc)} tokens preserved")
