from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import os
import time
from datetime import datetime
import logging
from pathlib import Path
import tempfile
import asyncio

from ..core.config import settings
from ..core.database import VectorDatabase
from ..core.embeddings import EmbeddingService
from ..core.retrieval import RetrievalEngine
from ..utils.preprocessing import DocumentProcessor
from ..evaluation.complete_evaluation import ComprehensiveEvaluator
from ..models.schemas import (
    SearchRequest, SearchResponse, SearchResultResponse,
    DocumentUploadRequest, UploadResponse, DocumentResponse,
    StatusResponse, DatabaseStatsResponse, HealthCheckResponse,
    MultiSearchRequest, HybridSearchRequest, ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{settings.log_file}"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Retrieval Plugin API",
    description="Semantic search and document retrieval service for ChatGPT integration",
    version="1.0.0",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json"
)

# Frontend directory setup
BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

# CORS middleware for Custom GPT integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
vector_db = VectorDatabase()
embedding_service = EmbeddingService()
retrieval_engine = RetrievalEngine(vector_db, embedding_service)
document_processor = DocumentProcessor(
    chunk_size=settings.chunk_size,
    overlap=settings.chunk_overlap
)

# Dependency for rate limiting (simple in-memory implementation)
request_counts = {}

def check_rate_limit(client_id: str = "default"):
    """Simple rate limiting check."""
    current_minute = datetime.now().strftime("%Y%m%d%H%M")
    key = f"{client_id}:{current_minute}"
    
    if key not in request_counts:
        request_counts[key] = 0
    
    request_counts[key] += 1
    
    if request_counts[key] > settings.max_requests_per_minute:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Clean old entries
    if len(request_counts) > 100:
        keys_to_remove = list(request_counts.keys())[:50]
        for k in keys_to_remove:
            del request_counts[k]

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An internal error occurred",
            details={"type": type(exc).__name__}
        ).dict()
    )

# Health check endpoint
@app.get(
    f"{settings.api_prefix}/health",
    response_model=HealthCheckResponse,
    tags=["Health"]
)
async def health_check():
    """Check service health status."""
    components = {
        "database": "healthy",
        "embedding_service": "healthy",
        "retrieval_engine": "healthy"
    }
    
    # Check database
    try:
        stats = vector_db.get_statistics()
        if stats:
            components["database"] = "healthy"
    except:
        components["database"] = "unhealthy"
    
    # Overall status
    all_healthy = all(status == "healthy" for status in components.values())
    
    return HealthCheckResponse(
        status="healthy" if all_healthy else "degraded",
        service="retrieval-plugin",
        version="1.0.0",
        components=components
    )

# Main search endpoint
@app.post(
    f"{settings.api_prefix}/search",
    response_model=SearchResponse,
    tags=["Search"],
    dependencies=[Depends(check_rate_limit)]
)
async def search_documents(request: SearchRequest):
    """
    Perform semantic search across the knowledge base.
    
    This endpoint accepts a natural language query and returns
    the most relevant document chunks based on semantic similarity.
    """
    start_time = time.time()
    
    try:
        # Perform search
        results = await retrieval_engine.search_async(
            query=request.query,
            max_results=request.max_results,
            score_threshold=request.score_threshold,
            filter_dict=request.filter,
            rerank=request.rerank
        )
        
        # Format results
        formatted_results = [
            SearchResultResponse(
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                chunk_id=result.chunk_id
            )
            for result in results
        ]
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            total_results=len(formatted_results),
            search_time_ms=search_time_ms
        )
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Multi-query search
@app.post(
    f"{settings.api_prefix}/search/multi",
    response_model=SearchResponse,
    tags=["Search"]
)
async def multi_query_search(request: MultiSearchRequest):
    """Perform search with multiple query variations."""
    start_time = time.time()
    
    try:
        results = retrieval_engine.multi_query_search(
            queries=request.queries,
            max_results_per_query=request.max_results_per_query,
            deduplicate=request.deduplicate
        )
        
        formatted_results = [
            SearchResultResponse(
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                chunk_id=result.chunk_id
            )
            for result in results
        ]
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=formatted_results,
            query=" | ".join(request.queries),
            total_results=len(formatted_results),
            search_time_ms=search_time_ms
        )
    
    except Exception as e:
        logger.error(f"Multi-query search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Hybrid search
@app.post(
    f"{settings.api_prefix}/search/hybrid",
    response_model=SearchResponse,
    tags=["Search"]
)
async def hybrid_search(request: HybridSearchRequest):
    """Perform hybrid semantic + keyword search."""
    start_time = time.time()
    
    try:
        results = retrieval_engine.hybrid_search(
            query=request.query,
            keywords=request.keywords,
            max_results=request.max_results,
            keyword_weight=request.keyword_weight
        )
        
        formatted_results = [
            SearchResultResponse(
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                chunk_id=result.chunk_id
            )
            for result in results
        ]
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            total_results=len(formatted_results),
            search_time_ms=search_time_ms
        )
    
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document upload endpoint
@app.post(
    f"{settings.api_prefix}/documents/upload",
    response_model=UploadResponse,
    tags=["Documents"]
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document to the knowledge base.
    
    Supported formats: PDF, TXT, DOCX, HTML, Markdown
    """
    start_time = time.time()
    
    # Validate file type
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in document_processor.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {document_processor.supported_formats}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process document
        chunks = document_processor.process_document(tmp_file_path)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from document")
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_service.generate_embeddings_batch(texts, show_progress=False)
        
        # Add to database
        success, message = vector_db.add_chunks(chunks, embeddings)
        
        if not success:
            raise HTTPException(status_code=500, detail=message)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return UploadResponse(
            status="success",
            message=f"Successfully processed {file.filename}",
            document_ids=[chunk.chunk_id for chunk in chunks],
            chunks_created=len(chunks),
            processing_time_ms=processing_time_ms
        )
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

# Get document by ID
@app.get(
    f"{settings.api_prefix}/documents/{{document_id}}",
    response_model=DocumentResponse,
    tags=["Documents"]
)
async def get_document(document_id: str):
    """Retrieve a specific document by ID."""
    doc = vector_db.get_document_by_id(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=doc['id'],
        content=doc['content'],
        metadata=doc['metadata']
    )

# Delete document
@app.delete(
    f"{settings.api_prefix}/documents/{{document_id}}",
    response_model=StatusResponse,
    tags=["Documents"]
)
async def delete_document(document_id: str):
    """Delete a document from the knowledge base."""
    success, message = vector_db.delete_document(document_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return StatusResponse(
        status="success",
        message=message
    )

# Database statistics
@app.get(
    f"{settings.api_prefix}/stats",
    response_model=DatabaseStatsResponse,
    tags=["Database"]
)
async def get_database_stats():
    """Get database statistics."""
    try:
        stats = vector_db.get_statistics()

        last_updated = stats.get("last_updated")
        if isinstance(last_updated, datetime):
            last_updated = last_updated.isoformat()

        return DatabaseStatsResponse(
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            last_updated=last_updated
        )
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

# Clear database (use with caution!)
@app.post(
    f"{settings.api_prefix}/database/clear",
    response_model=StatusResponse,
    tags=["Database"]
)
async def clear_database(confirm: bool = False):
    """Clear all documents from the database."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Please set confirm=true to clear the database"
        )
    
    success, message = vector_db.clear_collection()
    
    if not success:
        raise HTTPException(status_code=500, detail=message)
    
    return StatusResponse(
        status="success",
        message=message
    )

# Root endpoint
@app.get("/")
async def root():
    demo_file = FRONTEND_DIR / "main.html"
    if demo_file.exists():
        return FileResponse(str(demo_file))

    """Root endpoint with API information."""
    return {
        "service": "Retrieval Plugin API",
        "version": "1.0.0",
        "docs": f"{settings.api_prefix}/docs",
        "health": f"{settings.api_prefix}/health"
    }

@app.get("/demo", include_in_schema=False)
async def demo_page():
    demo_file = FRONTEND_DIR / "main.html"
    if not demo_file.exists():
        raise HTTPException(status_code=404, detail="Demo frontend not found")
    return FileResponse(str(demo_file))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )