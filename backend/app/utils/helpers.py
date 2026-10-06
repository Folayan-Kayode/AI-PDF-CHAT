"""Text helpers."""

import re

# Characters that are one token each rather than roughly a quarter -- CJK
# text has a very different characters-per-token ratio to Latin text.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # Japanese kana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility
    (0x20000, 0x2FA1F),  # CJK extensions B-F
)


def _is_cjk(character: str) -> bool:
    code = ord(character)

    return any(start <= code <= end for start, end in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """
    Approximate the token count of a piece of text.

    Deliberately heuristic: exact counting needs the provider's tokeniser, and
    the purpose here is to stop a context budget being silently blown by a
    document written in a language that packs many tokens per character, not
    to predict a bill.
    """
    if not text:
        return 0

    cjk = sum(1 for character in text if _is_cjk(character))

    other = len(text) - cjk

    return cjk + other // 4


def clean_text(text: str) -> str:
    """
    Normalise extracted text while preserving its shape.

    Line breaks and indentation are the only structure a PDF gives us for
    tables, forms, lists and code, so lines are preserved and only runs of
    spaces, tabs and blank lines are collapsed. Flattening a whole page to a
    single line destroys exactly the documents that are hardest to read.
    """
    if not text:
        return ""

    normalised = text.replace("\r\n", "\n").replace("\r", "\n")

    lines: list[str] = []

    for raw_line in normalised.split("\n"):
        line = re.sub(r"[ \t\u00a0\u2007\u202f]+", " ", raw_line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)
