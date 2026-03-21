"""
Response models and utilities for the ChatGPT Retrieval Plugin API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

from .schemas import SearchResultResponse


class ResponseStatus(str, Enum):
    """Response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PROCESSING = "processing"


class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class PaginatedResponse(BaseModel):
    """Base paginated response model."""
    page: int = Field(default=1, description="Current page number", ge=1)
    per_page: int = Field(default=10, description="Items per page", ge=1, le=100)
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")


class SearchResultsResponse(BaseResponse):
    """Enhanced search results response."""
    results: List[SearchResultResponse] = Field(..., description="Search results")
    query: str = Field(..., description="Original search query")
    search_metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata")
    suggestions: List[str] = Field(default_factory=list, description="Query suggestions")
    
    
class PaginatedSearchResponse(SearchResultsResponse, PaginatedResponse):
    """Paginated search results response."""
    pass


class UploadResultResponse(BaseResponse):
    """File upload result response."""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Uploaded filename")
    file_size: int = Field(..., description="File size in bytes")
    chunks_created: int = Field(..., description="Number of text chunks created")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")


class BulkUploadResponse(BaseResponse):
    """Bulk file upload response."""
    uploaded_files: List[UploadResultResponse] = Field(..., description="Successfully uploaded files")
    failed_files: List[Dict[str, str]] = Field(default_factory=list, description="Failed uploads with errors")
    total_files: int = Field(..., description="Total number of files processed")
    successful_uploads: int = Field(..., description="Number of successful uploads")


class DocumentListResponse(BaseResponse):
    """Document listing response."""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents")
    total_documents: int = Field(..., description="Total number of documents")
    storage_stats: Dict[str, Any] = Field(default_factory=dict, description="Storage statistics")


class SystemStatsResponse(BaseResponse):
    """System statistics response."""
    database_stats: Dict[str, Any] = Field(..., description="Database statistics")
    system_health: Dict[str, str] = Field(..., description="System health indicators")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    uptime: float = Field(..., description="System uptime in seconds")


class EvaluationResponse(BaseResponse):
    """Evaluation results response."""
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics")
    test_queries: int = Field(..., description="Number of test queries")
    benchmark_results: Dict[str, Any] = Field(default_factory=dict, description="Benchmark results")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")


class ConfigurationResponse(BaseResponse):
    """Configuration information response."""
    current_config: Dict[str, Any] = Field(..., description="Current configuration")
    available_models: List[str] = Field(default_factory=list, description="Available embedding models")
    system_limits: Dict[str, int] = Field(default_factory=dict, description="System limits")


class ErrorDetailResponse(BaseResponse):
    """Detailed error response."""
    error_code: str = Field(..., description="Error code identifier")
    error_type: str = Field(..., description="Error type")
    details: Dict[str, Any] = Field(default_factory=dict, description="Error details")
    traceback: Optional[str] = Field(default=None, description="Error traceback (debug mode)")
    suggestions: List[str] = Field(default_factory=list, description="Error resolution suggestions")


class ValidationResponse(BaseResponse):
    """Validation result response."""
    is_valid: bool = Field(..., description="Whether validation passed")
    validation_errors: List[Dict[str, str]] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class AsyncTaskResponse(BaseResponse):
    """Asynchronous task response."""
    task_id: str = Field(..., description="Task identifier")
    status: ResponseStatus = Field(..., description="Task status")
    progress: float = Field(default=0.0, description="Task progress (0-1)", ge=0.0, le=1.0)
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result (when completed)")


# Utility functions for creating common responses

def create_success_response(
    message: str = "Operation completed successfully",
    data: Optional[Dict[str, Any]] = None
) -> BaseResponse:
    """Create a standard success response."""
    response = BaseResponse(success=True, message=message)
    if data:
        for key, value in data.items():
            setattr(response, key, value)
    return response


def create_error_response(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> ErrorDetailResponse:
    """Create a standard error response."""
    return ErrorDetailResponse(
        success=False,
        message=message,
        error_code=error_code or "UNKNOWN_ERROR",
        error_type="APIError",
        details=details or {}
    )


def create_validation_error_response(
    errors: List[Dict[str, str]],
    message: str = "Validation failed"
) -> ValidationResponse:
    """Create a validation error response."""
    return ValidationResponse(
        success=False,
        message=message,
        is_valid=False,
        validation_errors=errors
    )
