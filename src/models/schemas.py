from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Enums
class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "md"

class SearchFormat(str, Enum):
    DETAILED = "detailed"
    SIMPLE = "simple"
    NUMBERED = "numbered"

# Request Models
class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    max_results: Optional[int] = Field(5, ge=1, le=50, description="Maximum number of results")
    score_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    rerank: Optional[bool] = Field(False, description="Apply result reranking")
    
    @validator('query')
    def clean_query(cls, v):
        return v.strip()

class MultiSearchRequest(BaseModel):
    """Request for multi-query search."""
    queries: List[str] = Field(..., description="List of search queries", min_items=1, max_items=10)
    max_results_per_query: Optional[int] = Field(3, ge=1, le=10)
    deduplicate: Optional[bool] = Field(True, description="Remove duplicate results")

class HybridSearchRequest(BaseModel):
    """Request for hybrid semantic + keyword search."""
    query: str = Field(..., description="Semantic search query", min_length=1)
    keywords: Optional[List[str]] = Field(None, description="Keywords for filtering/boosting")
    max_results: Optional[int] = Field(5, ge=1, le=50)
    keyword_weight: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Weight for keyword matching")

class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    source: str = Field(..., description="Document source/filename")
    document_type: Optional[DocumentType] = Field(None, description="Document type")

class BatchDocumentRequest(BaseModel):
    """Request for batch document processing."""
    documents: List[DocumentUploadRequest] = Field(..., min_items=1, max_items=100)

# Response Models
class SearchResultResponse(BaseModel):
    """Individual search result."""
    content: str = Field(..., description="Text content of the result")
    score: float = Field(..., description="Similarity score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    chunk_id: str = Field(..., description="Unique chunk identifier")

class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: List[SearchResultResponse] = Field(default_factory=list, description="Search results")
    query: str = Field(..., description="Original query")
    total_results: int = Field(0, description="Total number of results")
    search_time_ms: Optional[float] = Field(None, description="Search execution time in milliseconds")
    
    class Config:
        json_example = {
            "results": [
                {
                    "content": "Sample text content...",
                    "score": 0.95,
                    "metadata": {"source": "document.pdf", "page": 1},
                    "chunk_id": "doc123_0001"
                }
            ],
            "query": "sample search query",
            "total_results": 1,
            "search_time_ms": 125.5
        }

class DocumentResponse(BaseModel):
    """Response for document operations."""
    id: str = Field(..., description="Document/chunk ID")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = Field(None)
    source: Optional[str] = Field(None)

class StatusResponse(BaseModel):
    """Generic status response."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")

class UploadResponse(BaseModel):
    """Response for document upload."""
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")
    document_ids: List[str] = Field(default_factory=list, description="IDs of uploaded documents")
    chunks_created: int = Field(0, description="Number of chunks created")
    processing_time_ms: Optional[float] = Field(None)

class DatabaseStatsResponse(BaseModel):
    """Database statistics response."""
    total_documents: int = Field(0, description="Total unique documents")
    total_chunks: int = Field(0, description="Total text chunks")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")
    storage_size_mb: Optional[float] = Field(None, description="Storage size in MB")

class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.now)
    components: Dict[str, str] = Field(default_factory=dict, description="Component health status")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)

# Additional Models
class QueryExpansionRequest(BaseModel):
    """Request for query expansion."""
    query: str = Field(..., min_length=1, max_length=500)
    expansion_method: Optional[str] = Field("synonyms", description="Expansion method")
    max_expansions: Optional[int] = Field(3, ge=1, le=10)

class SimilarityRequest(BaseModel):
    """Request for finding similar chunks."""
    chunk_id: str = Field(..., description="Source chunk ID")
    max_results: Optional[int] = Field(5, ge=1, le=20)

class ExplainSearchRequest(BaseModel):
    """Request for search result explanation."""
    query: str = Field(..., min_length=1)
    result_ids: List[str] = Field(..., min_items=1, max_items=10, description="Result chunk IDs to explain")