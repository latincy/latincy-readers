"""Convert AGLDT treebank data to UD format with tree restructuring.

Handles the full Prague-to-UD conversion pipeline:

1. AGLDT 9-position postag → UPOS + UD morphological features
2. AGLDT POS → LatinCy XPOS convention (English word tags)
3. Prague dependency relations → UD dependency relations
4. Prague tree structure → UD tree structure (head restructuring)

The tree restructuring handles four major structural differences:

- **Coordination**: Prague uses a coordinator node as head; UD uses
  first-conjunct-as-head with ``conj`` for subsequent conjuncts.
- **Prepositions (AuxP)**: Prague makes the preposition an intermediate
  head; UD makes it a ``case`` dependent of the governed noun.
- **Subordinators (AuxC)**: Prague can make the subordinator an
  intermediate head; UD makes it a ``mark`` dependent.
- **Copula**: Prague makes the copula the predicate head; UD makes the
  predicate nominal the head with the copula as ``cop``.

Reference:
    Zeman et al. (2014). "HamleDT: Harmonized Multi-Language
    Dependency Treebank." *Language Resources and Evaluation*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from lxml import etree


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AGLDTToken:
    """Raw token from AGLDT XML."""

    id: int
    form: str
    lemma: str
    postag: str
    relation: str
    head: int
    cite: str = ""
    subdoc: str = ""


@dataclass
class UDToken:
    """Token in UD CoNLL-U format."""

    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str = "_"
    misc: str = "_"

    # Keep the original Prague relation for context during restructuring
    _prague_rel: str = field(default="", repr=False)
    _prague_head: int = field(default=0, repr=False)
    _prague_postag: str = field(default="", repr=False)

    def to_conllu_row(self) -> str:
        return "\t".join([
            str(self.id),
            self.form,
            self.lemma,
            self.upos,
            self.xpos,
            self.feats,
            str(self.head),
            self.deprel,
            self.deps,
            self.misc,
        ])


# ---------------------------------------------------------------------------
# POS mapping: AGLDT positional char → UPOS
# ---------------------------------------------------------------------------

_AGLDT_POS_TO_UPOS: dict[str, str] = {
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

# UPOS → LatinCy XPOS (English word tags)
_UPOS_TO_XPOS: dict[str, str] = {
    "NOUN": "noun",
    "VERB": "verb",
    "ADJ": "adjective",
    "ADV": "adverb",
    "ADP": "preposition",
    "PRON": "pronoun",
    "DET": "pronoun",
    "NUM": "number",
    "PUNCT": "punc",
    "CCONJ": "conjunction",
    "SCONJ": "conjunction",
    "INTJ": "noun",
    "AUX": "verb",
    "PROPN": "proper_noun",
    "X": "noun",
    "PART": "particle",
}


# ---------------------------------------------------------------------------
# Morphological feature decoding
# ---------------------------------------------------------------------------

# Positions in AGLDT 9-char postag:
# 0=pos 1=person 2=number 3=tense 4=mood 5=voice 6=gender 7=case 8=degree

_PERSON: dict[str, str] = {"1": "1", "2": "2", "3": "3"}
_NUMBER: dict[str, str] = {"s": "Sing", "p": "Plur", "d": "Dual"}
_TENSE: dict[str, str] = {
    "p": "Pres", "i": "Imp", "r": "Perf",
    "l": "Pqp", "t": "Fut", "f": "FutPerf",
}
_MOOD: dict[str, str] = {
    "i": "Ind", "s": "Sub", "n": "Inf",
    "p": "Part", "d": "Ger", "g": "Gdv", "m": "Imp",
}
_VOICE: dict[str, str] = {"a": "Act", "p": "Pass"}
_GENDER: dict[str, str] = {"m": "Masc", "f": "Fem", "n": "Neut"}
_CASE: dict[str, str] = {
    "n": "Nom", "g": "Gen", "d": "Dat",
    "a": "Acc", "v": "Voc", "b": "Abl", "l": "Loc",
}
_DEGREE: dict[str, str] = {"p": "Pos", "c": "Cmp", "s": "Sup"}


def decode_morph(postag: str, upos: str) -> str:
    """Decode AGLDT 9-position postag to UD morphological features.

    Handles the LatinCy convention where finite perfects use ``Tense=Past``
    but participial perfects use ``Tense=Perf``.
    """
    if len(postag) < 9:
        return "_"

    feats: dict[str, str] = {}

    # Person (pos 1)
    if postag[1] in _PERSON:
        feats["Person"] = _PERSON[postag[1]]

    # Number (pos 2)
    if postag[2] in _NUMBER:
        feats["Number"] = _NUMBER[postag[2]]

    # Mood → VerbForm + Mood (pos 4)
    mood_char = postag[4]
    verbform = None
    if mood_char in _MOOD:
        mood = _MOOD[mood_char]
        if mood == "Part":
            feats["VerbForm"] = "Part"
            verbform = "Part"
        elif mood == "Inf":
            feats["VerbForm"] = "Inf"
            verbform = "Inf"
        elif mood == "Ger":
            feats["VerbForm"] = "Ger"
            verbform = "Ger"
        elif mood == "Gdv":
            feats["VerbForm"] = "Part"
            feats["Mood"] = "Gdv"
            verbform = "Gdv"
        elif mood == "Imp":
            feats["VerbForm"] = "Fin"
            feats["Mood"] = "Imp"
            verbform = "Fin"
        else:
            feats["VerbForm"] = "Fin"
            feats["Mood"] = mood
            verbform = "Fin"

    # Tense (pos 3) — context-dependent on VerbForm
    tense_char = postag[3]
    if tense_char in _TENSE:
        raw_tense = _TENSE[tense_char]
        if raw_tense == "Perf":
            # LatinCy convention: finite perfect → Past, participle → Perf
            if verbform == "Part" or verbform == "Gdv":
                feats["Tense"] = "Perf"
            else:
                feats["Tense"] = "Past"
        elif raw_tense == "Imp":
            # Imperfect → Past in LatinCy
            feats["Tense"] = "Past"
        elif raw_tense == "FutPerf":
            feats["Tense"] = "Fut"
        else:
            feats["Tense"] = raw_tense

    # Voice (pos 5)
    if postag[5] in _VOICE:
        feats["Voice"] = _VOICE[postag[5]]

    # Gender (pos 6)
    if postag[6] in _GENDER:
        feats["Gender"] = _GENDER[postag[6]]

    # Case (pos 7)
    if postag[7] in _CASE:
        feats["Case"] = _CASE[postag[7]]

    # Degree (pos 8) — only Cmp/Sup, skip Pos (unmarked)
    if postag[8] in _DEGREE and postag[8] != "p":
        feats["Degree"] = _DEGREE[postag[8]]

    if not feats:
        return "_"

    return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))


# ---------------------------------------------------------------------------
# Lemma normalization
# ---------------------------------------------------------------------------

def normalize_lemma(lemma: str) -> str:
    """Normalize AGLDT lemma to LatinCy convention.

    - Strips trailing sense digits (``fero1`` → ``fero``)
    - Lowercases (except proper nouns, handled separately)
    - Strips leading/trailing whitespace
    """
    cleaned = lemma.strip()
    cleaned = re.sub(r"\d+$", "", cleaned)
    return cleaned.lower() if cleaned else "_"


# ---------------------------------------------------------------------------
# AGLDT XML parser
# ---------------------------------------------------------------------------

def parse_agldt_xml(path: str | Path) -> list[list[AGLDTToken]]:
    """Parse AGLDT XML into list of sentences, each a list of tokens."""
    tree = etree.parse(str(path))
    sentences: list[list[AGLDTToken]] = []

    for sent_el in tree.findall(".//sentence"):
        sent_id = sent_el.get("id", "")
        subdoc = sent_el.get("subdoc", "")
        tokens: list[AGLDTToken] = []

        for idx, word_el in enumerate(sent_el.findall("word"), start=1):
            tokens.append(AGLDTToken(
                id=idx,
                form=word_el.get("form", ""),
                lemma=word_el.get("lemma", ""),
                postag=word_el.get("postag", "---------"),
                relation=word_el.get("relation", ""),
                head=int(word_el.get("head") or "0"),
                cite=word_el.get("cite", ""),
                subdoc=subdoc,
            ))
        sentences.append(tokens)

    return sentences


# ---------------------------------------------------------------------------
# Initial conversion (before tree restructuring)
# ---------------------------------------------------------------------------

_COPULA_LEMMAS = frozenset({"sum", "sum1"})


def _initial_convert(agldt_tokens: list[AGLDTToken]) -> list[UDToken]:
    """Convert AGLDT tokens to UDToken with POS/morph but Prague deps."""
    ud_tokens: list[UDToken] = []

    for at in agldt_tokens:
        pos_char = at.postag[0] if at.postag else "-"
        upos = _AGLDT_POS_TO_UPOS.get(pos_char, "X")

        # Refine UPOS based on context
        clean_lemma = re.sub(r"\d+$", "", at.lemma).strip().lower()
        if upos == "VERB" and clean_lemma == "sum" and at.relation == "AuxV":
            upos = "AUX"
        if upos == "NOUN" and clean_lemma[0:1].isupper() and at.form[0:1].isupper():
            # Heuristic: capitalized noun with capitalized lemma → PROPN
            # But careful: sentence-initial words are also capitalized
            pass  # skip for now, too noisy

        xpos = _UPOS_TO_XPOS.get(upos, "noun")
        feats = decode_morph(at.postag, upos)
        lemma = normalize_lemma(at.lemma)

        # SpaceAfter heuristic
        misc = "_"
        # No space before punctuation, closing parens
        # (will be refined in post-processing)

        ud_tokens.append(UDToken(
            id=at.id,
            form=at.form,
            lemma=lemma,
            upos=upos,
            xpos=xpos,
            feats=feats,
            head=at.head,
            deprel=at.relation,  # Prague rel, will be remapped
            _prague_rel=at.relation,
            _prague_head=at.head,
            _prague_postag=at.postag,
        ))

    return ud_tokens


# ---------------------------------------------------------------------------
# Tree restructuring
# ---------------------------------------------------------------------------

def _tok_by_id(tokens: list[UDToken]) -> dict[int, UDToken]:
    """Build id → token lookup."""
    return {t.id: t for t in tokens}


def _children_of(tokens: list[UDToken], head_id: int) -> list[UDToken]:
    """Get children of a token."""
    return [t for t in tokens if t.head == head_id]


def _depth(tok: UDToken, lookup: dict[int, UDToken], seen: set[int] | None = None) -> int:
    """Compute depth of a token in the tree (0 for root)."""
    if seen is None:
        seen = set()
    if tok.head == 0 or tok.id in seen:
        return 0
    seen.add(tok.id)
    parent = lookup.get(tok.head)
    if parent is None:
        return 0
    return 1 + _depth(parent, lookup, seen)


def _reattach_children(
    tokens: list[UDToken],
    old_head_id: int,
    new_head_id: int,
    exclude: set[int] | None = None,
) -> None:
    """Move all children of old_head to new_head (except those in exclude)."""
    exclude = exclude or set()
    for t in tokens:
        if t.head == old_head_id and t.id not in exclude:
            t.head = new_head_id


def restructure_coordination(tokens: list[UDToken]) -> None:
    """Convert Prague coordination to UD first-conjunct-as-head.

    Prague: COORD node heads conjuncts (*_CO) and shared modifiers.
    UD: First conjunct is head; others attach via ``conj``; coordinator
    word attaches via ``cc`` to the following conjunct.
    """
    lookup = _tok_by_id(tokens)

    # Process COORD nodes bottom-up (deepest first) to handle nesting
    coord_nodes = [t for t in tokens if t._prague_rel == "COORD"]
    coord_nodes.sort(key=lambda t: -_depth(t, lookup))

    for coord_tok in coord_nodes:
        children = _children_of(tokens, coord_tok.id)

        # Separate conjuncts from shared dependents
        conjuncts = [c for c in children if c._prague_rel.endswith("_CO")]
        shared = [c for c in children if not c._prague_rel.endswith("_CO")]

        if not conjuncts:
            continue

        # Sort conjuncts by linear order
        conjuncts.sort(key=lambda t: t.id)
        first = conjuncts[0]

        # First conjunct inherits the COORD's position in the tree
        first.head = coord_tok.head
        # Map the base relation (strip _CO suffix)
        base_rel = first._prague_rel.rsplit("_CO", 1)[0]
        first._prague_rel = base_rel
        first.deprel = base_rel

        # Other conjuncts attach to the first via conj
        for conj in conjuncts[1:]:
            conj.head = first.id
            conj.deprel = "conj"
            conj._prague_rel = "conj"

        # The COORD word itself becomes cc
        # Attach it to the conjunct that follows it in linear order
        cc_target = first
        for conj in conjuncts[1:]:
            if conj.id > coord_tok.id:
                cc_target = conj
                break
        coord_tok.head = cc_target.id
        coord_tok.deprel = "cc"
        coord_tok._prague_rel = "cc"

        # Shared dependents (AuxY, AuxZ, etc.) attach to the first conjunct
        for dep in shared:
            dep.head = first.id

        # Anything else that was pointing at the COORD node now
        # points at the first conjunct
        _reattach_children(tokens, coord_tok.id, first.id, exclude={coord_tok.id})


def restructure_prepositions(tokens: list[UDToken]) -> None:
    """Convert Prague AuxP to UD case dependent.

    Prague: preposition heads its governed noun.
    UD: noun is the head; preposition attaches as ``case``.
    """
    lookup = _tok_by_id(tokens)

    # Process AuxP nodes
    auxp_nodes = [t for t in tokens if t._prague_rel == "AuxP"]

    for prep in auxp_nodes:
        children = _children_of(tokens, prep.id)

        # Find the content child (skip AuxX=commas, AuxG=brackets, AuxZ=particles)
        content = [
            c for c in children
            if c._prague_rel not in ("AuxX", "AuxG", "AuxZ", "AuxK")
        ]
        if not content:
            continue

        # Pick the first content child (usually the governed noun)
        # Prefer nouns/pronouns over other categories
        noun = None
        for c in content:
            if c.upos in ("NOUN", "PRON", "PROPN", "NUM", "DET", "ADJ"):
                noun = c
                break
        if noun is None:
            noun = content[0]

        # Promote the noun: it takes the preposition's head
        noun.head = prep.head
        # Determine relation based on the governor's POS
        governor = lookup.get(prep.head)
        if governor is not None and governor.upos in ("NOUN", "PROPN", "PRON", "NUM", "ADJ"):
            noun.deprel = "nmod"
            noun._prague_rel = "nmod"
        else:
            noun.deprel = "obl"
            noun._prague_rel = "obl"

        # Preposition becomes case dependent of the noun
        prep.head = noun.id
        prep.deprel = "case"
        prep._prague_rel = "case"

        # Other children of the preposition re-attach to the noun
        _reattach_children(tokens, prep.id, noun.id, exclude={prep.id})


def restructure_subordinators(tokens: list[UDToken]) -> None:
    """Convert Prague AuxC to UD mark dependent.

    Prague: subordinator can be an intermediate head.
    UD: subordinator is ``mark`` dependent of the clause verb.
    """
    lookup = _tok_by_id(tokens)

    auxc_nodes = [t for t in tokens if t._prague_rel == "AuxC"]

    for subord in auxc_nodes:
        children = _children_of(tokens, subord.id)

        # Find the verbal child (the clause head)
        verb = None
        for c in children:
            if c.upos in ("VERB", "AUX"):
                verb = c
                break
        # Fallback: any content child
        if verb is None:
            content = [
                c for c in children
                if c._prague_rel not in ("AuxX", "AuxG", "AuxZ", "AuxK")
            ]
            if content:
                verb = content[0]

        if verb is None:
            continue

        # Promote the verb: it takes the subordinator's head and relation
        verb.head = subord.head

        # Determine clause relation based on context
        governor = lookup.get(subord.head)
        if governor is not None and governor.upos in ("NOUN", "PROPN", "ADJ"):
            verb.deprel = "acl"
            verb._prague_rel = "acl"
        else:
            verb.deprel = "advcl"
            verb._prague_rel = "advcl"

        # Subordinator becomes mark
        subord.head = verb.id
        subord.deprel = "mark"
        subord._prague_rel = "mark"

        # Other children re-attach to the verb
        _reattach_children(tokens, subord.id, verb.id, exclude={subord.id})


def restructure_copula(tokens: list[UDToken]) -> None:
    """Convert Prague copular constructions to UD.

    Prague: copula (sum) is PRED or AuxV; PNOM is a sibling.
    UD: predicate nominal is the head; copula attaches as ``cop``.
    """
    lookup = _tok_by_id(tokens)

    # Find copular verbs: AuxV relation with sum lemma, or PRED with sum lemma
    # and a PNOM sibling
    for tok in list(tokens):
        if tok.lemma not in ("sum", "esse"):
            continue
        if tok.upos not in ("VERB", "AUX"):
            continue

        # Case 1: tok is AuxV — it's already a dependent, find PNOM sibling
        if tok._prague_rel == "AuxV":
            siblings = _children_of(tokens, tok.head)
            pnom = None
            for sib in siblings:
                if sib._prague_rel in ("PNOM", "PNOM_CO"):
                    pnom = sib
                    break
            if pnom is not None:
                tok.deprel = "cop"
                tok._prague_rel = "cop"
                tok.upos = "AUX"
                tok.xpos = "verb"
            continue

        # Case 2: tok is PRED with PNOM child
        if tok._prague_rel in ("PRED", "PRED_CO", "ROOT"):
            children = _children_of(tokens, tok.id)
            pnom = None
            for c in children:
                if c._prague_rel in ("PNOM", "PNOM_CO"):
                    pnom = c
                    break

            if pnom is None:
                continue

            # Promote PNOM to take the copula's position
            pnom.head = tok.head
            pnom.deprel = tok.deprel
            pnom._prague_rel = tok._prague_rel

            # Copula becomes cop dependent of PNOM
            tok.head = pnom.id
            tok.deprel = "cop"
            tok._prague_rel = "cop"
            tok.upos = "AUX"
            tok.xpos = "verb"

            # Re-attach copula's other children to PNOM
            _reattach_children(tokens, tok.id, pnom.id, exclude={tok.id})


# ---------------------------------------------------------------------------
# Relation mapping (after tree restructuring)
# ---------------------------------------------------------------------------

def _map_relation(tok: UDToken, lookup: dict[int, UDToken]) -> str:
    """Map a Prague relation to UD after tree restructuring.

    Many relations will already have been set during restructuring
    (case, cc, conj, cop, mark, nmod, obl). This handles the rest.
    """
    rel = tok._prague_rel
    head_tok = lookup.get(tok.head)

    # Already mapped during restructuring
    if rel in ("case", "cc", "conj", "cop", "mark", "nmod", "obl",
               "advcl", "acl", "acl:relcl"):
        return rel

    # Root
    if rel == "PRED" or tok.head == 0:
        return "ROOT"

    # Subject
    if rel == "SBJ":
        # Check if the verb is passive
        if tok.head != 0 and head_tok is not None:
            if "Voice=Pass" in (head_tok.feats or ""):
                return "nsubj:pass"
        return "nsubj"

    # Object
    if rel == "OBJ":
        # Clausal complement if the dependent is a verb
        if tok.upos == "VERB" and tok.feats and "VerbForm=Inf" in tok.feats:
            return "xcomp"
        if tok.upos == "VERB" and tok.feats and "VerbForm=Fin" in tok.feats:
            return "ccomp"
        return "obj"

    # Adverbial
    if rel == "ADV":
        if tok.upos in ("ADV",):
            return "advmod"
        if tok.upos == "VERB":
            if tok.feats and "VerbForm=Part" in tok.feats:
                return "advcl"
            if tok.feats and "VerbForm=Inf" in tok.feats:
                return "advcl"
            if tok.feats and "VerbForm=Fin" in tok.feats:
                return "advcl"
            if tok.feats and "VerbForm=Ger" in tok.feats:
                return "advcl"
            return "advcl"
        if tok.upos in ("NOUN", "PROPN", "PRON", "NUM"):
            return "obl"
        if tok.upos == "ADJ":
            return "advmod"
        return "obl"

    # Attribute
    if rel == "ATR":
        if head_tok is not None and head_tok.upos in ("NOUN", "PROPN", "PRON", "NUM"):
            if tok.upos in ("ADJ", "DET"):
                return "amod"
            if tok.upos in ("NOUN", "PROPN"):
                return "nmod"
            if tok.upos == "PRON":
                return "det"
            if tok.upos == "NUM":
                return "nummod"
            if tok.upos == "VERB":
                # Relative clause or participial modifier
                if tok.feats and "VerbForm=Part" in tok.feats:
                    return "amod"
                return "acl:relcl"
            return "nmod"
        else:
            # ATR on a verb — less common, usually participial or relative
            if tok.upos == "VERB":
                return "acl:relcl"
            if tok.upos in ("ADJ", "DET"):
                return "amod"
            if tok.upos == "PRON":
                return "det"
            return "amod"

    # Predicate nominal
    if rel == "PNOM":
        if tok.upos in ("ADJ", "NOUN", "PROPN", "PRON"):
            # In UD, PNOM after copula restructuring should already
            # be handled. If we get here, the copula wasn't restructured.
            if head_tok is not None and head_tok.upos in ("VERB", "AUX"):
                return "xcomp"
            return "nsubj"
        return "xcomp"

    # Secondary predication
    if rel in ("ATV", "AtvV"):
        return "xcomp"

    # Object complement
    if rel == "OCOMP":
        return "xcomp"

    # Apposition
    if "_AP" in rel or rel == "APOS":
        return "appos"

    # Extra-clausal
    if rel == "ExD":
        if tok.upos in ("NOUN", "PROPN") and tok.feats and "Case=Voc" in tok.feats:
            return "vocative"
        return "vocative"  # Most ExD in poetry is vocative

    # Auxiliary verb (non-copular)
    if rel == "AuxV":
        return "aux"

    # Punctuation
    if rel in ("AuxX", "AuxK", "AuxG"):
        return "punct"

    # Discourse particles, focus particles
    if rel == "AuxY":
        if tok.upos == "CCONJ":
            return "cc"
        return "discourse"

    if rel == "AuxZ":
        if tok.lemma in ("non", "ne", "haud", "nec"):
            return "advmod:neg"
        return "advmod"

    # Clausal subject
    if rel == "SBJ" and tok.upos == "VERB":
        return "csubj"

    # Fallback
    return "dep"


def _map_all_relations(tokens: list[UDToken]) -> None:
    """Apply relation mapping to all tokens."""
    lookup = _tok_by_id(tokens)
    for tok in tokens:
        tok.deprel = _map_relation(tok, lookup)


# ---------------------------------------------------------------------------
# SpaceAfter heuristic
# ---------------------------------------------------------------------------

def _set_space_after(tokens: list[UDToken]) -> None:
    """Set SpaceAfter=No in misc field based on punctuation heuristics."""
    for i, tok in enumerate(tokens):
        next_tok = tokens[i + 1] if i + 1 < len(tokens) else None

        no_space = False
        # No space before closing punctuation
        if next_tok and next_tok.form in (")", "]", "}", ",", ".", ";", ":", "!", "?"):
            if tok.form not in ("(", "[", "{"):
                no_space = False  # actually we want space before these usually
        # No space after opening punctuation
        if tok.form in ("(", "[", "{"):
            no_space = True
        # No space before closing punctuation from this token's perspective
        # is handled by the next token

        if next_tok and next_tok.form in (")", "]", "}"):
            no_space = True

        if no_space:
            tok.misc = "SpaceAfter=No"


# ---------------------------------------------------------------------------
# Validation: detect cycles and fix
# ---------------------------------------------------------------------------

def _fix_cycles(tokens: list[UDToken]) -> None:
    """Detect and fix dependency cycles by re-attaching to root."""
    lookup = _tok_by_id(tokens)

    for tok in tokens:
        visited: set[int] = set()
        current = tok
        while current.head != 0:
            if current.id in visited:
                # Cycle detected — break it by attaching tok to root
                tok.head = 0
                tok.deprel = "ROOT"
                break
            visited.add(current.id)
            parent = lookup.get(current.head)
            if parent is None:
                break
            current = parent


def _ensure_single_root(tokens: list[UDToken]) -> None:
    """Ensure exactly one root per sentence."""
    roots = [t for t in tokens if t.head == 0]

    if len(roots) == 0 and tokens:
        # No root — make the first verb the root
        for t in tokens:
            if t.upos in ("VERB", "AUX"):
                t.head = 0
                t.deprel = "ROOT"
                return
        # Fallback: first token
        tokens[0].head = 0
        tokens[0].deprel = "ROOT"

    elif len(roots) > 1:
        # Multiple roots — keep the first PRED, make others parataxis
        # or keep first content word
        main_root = roots[0]
        for r in roots:
            if r._prague_rel == "PRED":
                main_root = r
                break

        for r in roots:
            if r is not main_root:
                # Punctuation attaches to main root
                if r.upos == "PUNCT":
                    r.head = main_root.id
                    r.deprel = "punct"
                else:
                    r.head = main_root.id
                    r.deprel = "parataxis"


# ---------------------------------------------------------------------------
# High-level conversion
# ---------------------------------------------------------------------------

def convert_sentence(agldt_tokens: list[AGLDTToken]) -> list[UDToken]:
    """Convert a single sentence from AGLDT to UD format.

    Applies POS/morph conversion, tree restructuring, and relation mapping.
    """
    if not agldt_tokens:
        return []

    tokens = _initial_convert(agldt_tokens)

    # Tree restructuring in order:
    # 1. Coordination (bottom-up for nesting)
    restructure_coordination(tokens)
    # 2. Prepositions
    restructure_prepositions(tokens)
    # 3. Subordinating conjunctions
    restructure_subordinators(tokens)
    # 4. Copula
    restructure_copula(tokens)

    # Map remaining relations
    _map_all_relations(tokens)

    # Post-processing
    _fix_cycles(tokens)
    _ensure_single_root(tokens)
    _set_space_after(tokens)

    return tokens


def convert_file(
    agldt_path: str | Path,
    *,
    fileid: str = "",
    collection: str = "",
    model_name: str = "agldt-gold",
    model_version: str = "2.1",
) -> str:
    """Convert an AGLDT XML file to .conlluc format.

    Args:
        agldt_path: Path to AGLDT XML file.
        fileid: File identifier for the .conlluc header.
        collection: Collection name.
        model_name: Source annotation name (default ``"agldt-gold"``).
        model_version: Source version.

    Returns:
        Complete .conlluc file content as a string.
    """
    from datetime import datetime, timezone

    sentences = parse_agldt_xml(agldt_path)
    lines: list[str] = []

    # File-level metadata
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    file_meta = {
        "generator": "latincy-readers",
        "annotation_status": "silver",
        "do_not_use_for_training": "true",
        "source": "agldt",
        "source_file": Path(agldt_path).name,
        "model_name": model_name,
        "model_version": model_version,
        "generated": now,
        "collection": collection,
        "fileid": fileid,
        "corrections": "0",
        "tree_restructuring": "prague-to-ud",
        "pos_normalization": "agldt-to-latincy",
        "morph_normalization": "agldt-positional-to-ud-features",
    }
    for key, value in file_meta.items():
        lines.append(f"# {key} = {value}")
    lines.append("")

    # Sentences
    for sent_idx, agldt_sent in enumerate(sentences, start=1):
        ud_tokens = convert_sentence(agldt_sent)
        if not ud_tokens:
            continue

        # Sentence metadata
        sent_text = " ".join(t.form for t in ud_tokens)
        subdoc = agldt_sent[0].subdoc if agldt_sent else ""
        lines.append(f"# sent_id = {fileid}:{sent_idx}")
        if subdoc:
            lines.append(f"# subdoc = {subdoc}")
        lines.append(f"# text = {sent_text}")

        for tok in ud_tokens:
            lines.append(tok.to_conllu_row())

        lines.append("")

    return "\n".join(lines)
