"""UD tagset validation for Latin annotation editing.

Provides canonical inventories of UPOS tags, Latin XPOS tags,
morphological features, and NER labels, plus validation helpers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Universal POS tags (UD v2)
# ---------------------------------------------------------------------------

UPOS_TAGS: list[str] = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

UPOS_DESCRIPTIONS: dict[str, str] = {
    "ADJ": "Adjective",
    "ADP": "Adposition (preposition)",
    "ADV": "Adverb",
    "AUX": "Auxiliary verb",
    "CCONJ": "Coordinating conjunction",
    "DET": "Determiner",
    "INTJ": "Interjection",
    "NOUN": "Noun",
    "NUM": "Numeral",
    "PART": "Particle",
    "PRON": "Pronoun",
    "PROPN": "Proper noun",
    "PUNCT": "Punctuation",
    "SCONJ": "Subordinating conjunction",
    "SYM": "Symbol",
    "VERB": "Verb",
    "X": "Other",
}

_UPOS_SET = frozenset(UPOS_TAGS)

# ---------------------------------------------------------------------------
# Latin XPOS tags (Perseus / PROIEL / LatinCy conventions)
# ---------------------------------------------------------------------------

XPOS_TAGS: list[str] = [
    # LatinCy full-word tags (from la_core_web_lg / la_core_web_trf)
    "adjective",
    "adverb",
    "conjunction",
    "determiner",
    "noun",
    "number",
    "particle",
    "preposition",
    "pronoun",
    "proper_noun",
    "punc",
    "verb",
    "X",
    # Allow empty / underscore
    "_",
]

XPOS_DESCRIPTIONS: dict[str, str] = {
    "adjective": "Adjective",
    "adverb": "Adverb",
    "conjunction": "Conjunction",
    "determiner": "Determiner",
    "noun": "Noun",
    "number": "Numeral",
    "particle": "Particle",
    "preposition": "Preposition",
    "pronoun": "Pronoun",
    "proper_noun": "Proper noun",
    "punc": "Punctuation",
    "verb": "Verb",
    "X": "Other / Unknown",
    "_": "Unspecified",
}

# ---------------------------------------------------------------------------
# Morphological features (UD Latin inventory)
# ---------------------------------------------------------------------------

MORPH_FEATURES: dict[str, list[str]] = {
    "Case": ["Abl", "Acc", "Dat", "Gen", "Loc", "Nom", "Voc"],
    "Degree": ["Abs", "Cmp", "Pos", "Sup"],
    "Gender": ["Fem", "Masc", "Neut"],
    "Mood": ["Imp", "Ind", "Sub"],
    "Number": ["Plur", "Sing"],
    "NumType": ["Card", "Dist", "Mult", "Ord"],
    "Person": ["1", "2", "3"],
    "Polarity": ["Neg"],
    "Poss": ["Yes"],
    "PronType": ["Dem", "Ind", "Int", "Prs", "Rcp", "Rel", "Tot"],
    "Reflex": ["Yes"],
    "Tense": ["Fut", "Past", "Pqp", "Pres"],
    "VerbForm": ["Conv", "Fin", "Gdv", "Ger", "Inf", "Part", "Sup"],
    "Voice": ["Act", "Pass"],
}

# ---------------------------------------------------------------------------
# NER labels (LatinCy conventions + standard)
# ---------------------------------------------------------------------------

NER_LABELS: list[str] = [
    "O",       # Outside any entity
    "PER",     # Person
    "LOC",     # Location
    "GPE",     # Geo-political entity
    "NORP",    # National / religious / political group
    "EVENT",   # Named event
    "WORK",    # Work of art / literature
]

NER_DESCRIPTIONS: dict[str, str] = {
    "O": "Not an entity",
    "PER": "Person (e.g. Caesar, Cicero)",
    "LOC": "Location (e.g. Alpes, Tiberis)",
    "GPE": "Geo-political entity (e.g. Roma, Carthago)",
    "NORP": "Group (e.g. Romani, Stoici)",
    "EVENT": "Named event (e.g. Bellum Gallicum)",
    "WORK": "Work of art / literature (e.g. Aeneis)",
}

# For the editor, we present IOB2-style labels
NER_IOB_LABELS: list[str] = ["O"]
for _label in NER_LABELS[1:]:
    NER_IOB_LABELS.append(f"B-{_label}")
    NER_IOB_LABELS.append(f"I-{_label}")


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_upos(tag: str) -> str | None:
    """Return error message if *tag* is not a valid UPOS, else None."""
    if tag in _UPOS_SET or tag in ("", "_"):
        return None
    return f"Invalid UPOS tag: '{tag}'. Must be one of: {', '.join(UPOS_TAGS)}"


def validate_xpos(tag: str) -> str | None:
    """Return error message if *tag* looks problematic, else None.

    XPOS is model-specific so we only warn, not reject.
    """
    if tag in ("", "_") or tag in XPOS_DESCRIPTIONS:
        return None
    # Don't reject — just flag as unusual
    return f"Unusual XPOS tag: '{tag}' (not in standard Latin inventory)"


def validate_lemma(lemma: str) -> str | None:
    """Return error message if *lemma* is empty or suspicious."""
    if not lemma or lemma.isspace():
        return "Lemma cannot be empty"
    if lemma == "_":
        return None
    # Lemmas should be lowercase Latin text (with possible macrons)
    return None


def validate_morph(feats: dict[str, str]) -> list[str]:
    """Return list of warnings for morphological features."""
    warnings: list[str] = []
    for key, value in feats.items():
        if key not in MORPH_FEATURES:
            warnings.append(f"Unknown feature: '{key}'")
        elif value not in MORPH_FEATURES[key]:
            valid = ", ".join(MORPH_FEATURES[key])
            warnings.append(
                f"Invalid value '{value}' for {key}. Valid: {valid}"
            )
    return warnings


def validate_ner(label: str) -> str | None:
    """Return error message if *label* is not a valid NER IOB2 label."""
    if label in NER_IOB_LABELS or label in ("", "_"):
        return None
    return f"Invalid NER label: '{label}'. Must be one of: {', '.join(NER_IOB_LABELS)}"


def feats_to_str(feats: dict[str, str]) -> str:
    """Convert morph features dict to UD FEATS string."""
    if not feats:
        return "_"
    return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))


def feats_from_str(feats_str: str) -> dict[str, str]:
    """Parse UD FEATS string to dict."""
    if feats_str in ("_", ""):
        return {}
    result = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result
