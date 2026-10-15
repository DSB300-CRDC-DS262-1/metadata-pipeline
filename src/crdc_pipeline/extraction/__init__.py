from .backends import Backend, GeminiBackend, OllamaBackend, make_backend
from .metadata.extract import DocumentMetadata, MetadataExtractionResult, extract_metadata
from .statistics import StatisticEntry, StatisticsExtractionResult, extract_statistics
from .contexts import ContextEntry, ContextsExtractionResult, extract_contexts
from .vision import make_page_transcriber, make_vision_backend, make_word_reader

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
    "make_page_transcriber",
    "make_vision_backend",
    "make_word_reader",
]
