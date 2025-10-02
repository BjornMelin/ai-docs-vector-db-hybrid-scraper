"""Text-processing helpers."""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOP_WORDS: set[str] = set(ENGLISH_STOP_WORDS)
"""Default stop-word set shared by query processing components."""
