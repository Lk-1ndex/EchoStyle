"""Shared text measurements used by generation and workflow state."""


def count_visible_chars(text: str) -> int:
    """Count non-whitespace characters without discarding punctuation."""
    return sum(1 for char in text if not char.isspace())
