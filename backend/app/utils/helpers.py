import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    """

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text
