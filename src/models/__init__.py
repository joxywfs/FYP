"""
Data models and schemas for the ChatGPT Retrieval Plugin.

This module contains:
- Pydantic models for API requests and responses
- Data validation schemas
- Response formatting utilities
"""

from .schemas import (
    SearchRequest, SearchResponse, SearchResultResponse,
    UploadResponse, StatusResponse, DatabaseStatsResponse,
    HealthCheckResponse, MultiSearchRequest, HybridSearchRequest,
    ErrorResponse, DocumentType, SearchFormat
)
from .responses import (
    BaseResponse, SearchResultsResponse, UploadResultResponse,
    BulkUploadResponse, DocumentListResponse, SystemStatsResponse,
    EvaluationResponse, ConfigurationResponse, ErrorDetailResponse,
    ValidationResponse, AsyncTaskResponse,
    create_success_response, create_error_response, create_validation_error_response
)

__all__ = [
    # Schemas
    "SearchRequest", "SearchResponse", "SearchResultResponse",
    "UploadResponse", "StatusResponse", "DatabaseStatsResponse", 
    "HealthCheckResponse", "MultiSearchRequest", "HybridSearchRequest",
    "ErrorResponse", "DocumentType", "SearchFormat",
    # Responses
    "BaseResponse", "SearchResultsResponse", "UploadResultResponse",
    "BulkUploadResponse", "DocumentListResponse", "SystemStatsResponse",
    "EvaluationResponse", "ConfigurationResponse", "ErrorDetailResponse",
    "ValidationResponse", "AsyncTaskResponse",
    "create_success_response", "create_error_response", "create_validation_error_response"
]
