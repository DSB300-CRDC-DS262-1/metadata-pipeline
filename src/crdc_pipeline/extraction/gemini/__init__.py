"""Statistics and context extraction with Gemini: one prompt over the whole document for each.

The local pipeline (extraction/statistics and extraction/contexts) works page by page instead; this
version is kept for comparison and for use where a Gemini API key is available.
"""

from .contexts import ContextEntry, ContextsExtractionResult, extract_contexts
from .statistics import StatisticEntry, StatisticsExtractionResult, extract_statistics

__all__ = [
    "ContextEntry",
    "ContextsExtractionResult",
    "extract_contexts",
    "StatisticEntry",
    "StatisticsExtractionResult",
    "extract_statistics",
]
