from .backends import Backend, GeminiBackend, OllamaBackend, make_backend
from .metadata import DocumentMetadata, MetadataExtractionResult, extract_metadata
from .statistics import StatisticEntry, StatisticsExtractionResult, extract_statistics
from .contexts import ContextEntry, ContextsExtractionResult, extract_contexts

__all__ = [
    "Backend",
    "GeminiBackend",
    "OllamaBackend",
    "make_backend",
    "DocumentMetadata",
    "MetadataExtractionResult",
    "extract_metadata",
    "StatisticEntry",
    "StatisticsExtractionResult",
    "extract_statistics",
    "ContextEntry",
    "ContextsExtractionResult",
    "extract_contexts",
]
