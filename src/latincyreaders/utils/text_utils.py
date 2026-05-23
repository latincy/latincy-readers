"""Shared text utilities for corpus readers."""

from __future__ import annotations

import re


def find_line_in_doc_text(
    doc_text: str, line_text: str, start_pos: int
) -> tuple[int, int]:
    """Locate ``line_text`` inside ``doc_text`` starting at ``start_pos``.

    The NLP pipeline (LatinCy normer plus tokenization) rewrites text in
    ways a plain ``str.find`` cannot survive: J→I and V→U orthographic
    normalization, whitespace inserted around punctuation, and newline
    collapse. This function tries an exact match first and falls back to a
    regex that tolerates those transformations.

    Args:
        doc_text: Text of the spaCy Doc (post-normalization).
        line_text: Original line text (markup already stripped).
        start_pos: Char offset to begin the search at.

    Returns:
        ``(start, end)`` char positions in ``doc_text``, or ``(-1, -1)``
        if no match is found.
    """
    line_text = line_text.strip()
    if not line_text:
        return -1, -1

    # Fast path: exact match.
    pos = doc_text.find(line_text, start_pos)
    if pos >= 0:
        return pos, pos + len(line_text)

    # Regex fallback tolerating J/I, V/U, and whitespace variations.
    pattern_parts: list[str] = []
    prev_was_space = True
    for ch in line_text:
        if ch.isspace():
            if not prev_was_space:
                pattern_parts.append(r"\s+")
            prev_was_space = True
            continue
        if ch in "ji":
            pattern_parts.append("[ji]")
        elif ch in "JI":
            pattern_parts.append("[JI]")
        elif ch in "vu":
            pattern_parts.append("[vu]")
        elif ch in "VU":
            pattern_parts.append("[VU]")
        elif ch.isalnum():
            pattern_parts.append(re.escape(ch))
        else:
            pattern_parts.append(r"\s*" + re.escape(ch))
        prev_was_space = False

    if not pattern_parts:
        return -1, -1

    pattern = "".join(pattern_parts)
    match = re.search(pattern, doc_text[start_pos:])
    if match:
        return start_pos + match.start(), start_pos + match.end()
    return -1, -1
