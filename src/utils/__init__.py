"""
Utility functions and helpers for the ChatGPT Retrieval Plugin.

This module provides:
- Document preprocessing and text chunking
- Input validation utilities  
- Logging configuration
- Common helper functions
"""

from .preprocessing import DocumentProcessor, DocumentChunk
from .validation import (
    FileValidator, QueryValidator, ParameterValidator, MetadataValidator,
    validate_search_request, validate_upload_request, ValidationError
)
from .logging import (
    setup_logging, get_logger, PerformanceLogger, performance_logger,
    log_api_request, log_search_query, log_document_upload
)

__all__ = [
    # Preprocessing
    "DocumentProcessor", "DocumentChunk",
    # Validation
    "FileValidator", "QueryValidator", "ParameterValidator", "MetadataValidator",
    "validate_search_request", "validate_upload_request", "ValidationError",
    # Logging
    "setup_logging", "get_logger", "PerformanceLogger", "performance_logger",
    "log_api_request", "log_search_query", "log_document_upload"
]
