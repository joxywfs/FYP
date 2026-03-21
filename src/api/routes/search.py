"""
Search API routes.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
import logging
import time

from ...core.database import VectorDatabase
from ...core.retrieval import RetrievalEngine
from ...models.schemas import (
    SearchRequest, SearchResponse, SearchResultResponse, 
    MultiSearchRequest, HybridSearchRequest,
    ErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["search"])

# Dependencies
def get_database():
    return VectorDatabase()

def get_retrieval_engine():
    return RetrievalEngine()


@router.post("/", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    db: VectorDatabase = Depends(get_database),
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine)
):
    """Search for documents using semantic similarity."""
    start_time = time.time()
    
    try:
        logger.info(f"Searching for: '{request.query}' (top_k={request.top_k}, threshold={request.threshold})")
        
        # Perform search
        results = retrieval_engine.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            filters=request.filters
        )
        
        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(SearchResultResponse(
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                chunk_id=result.get("id", "")
            ))
        
        processing_time = time.time() - start_time
        
        logger.info(f"Search completed in {processing_time:.3f}s, found {len(search_results)} results")
        
        return SearchResponse(
            results=search_results,
            query=request.query,
            total_results=len(search_results),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/simple")
async def simple_search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=5, ge=1, le=100, description="Number of results"),
    threshold: float = Query(default=0.7, ge=0.0, le=1.0, description="Similarity threshold"),
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine)
):
    """Simple search endpoint with URL parameters."""
    start_time = time.time()
    
    try:
        logger.info(f"Simple search: '{q}' (limit={limit}, threshold={threshold})")
        
        results = retrieval_engine.search(
            query=q,
            top_k=limit,
            threshold=threshold
        )
        
        processing_time = time.time() - start_time
        
        return {
            "query": q,
            "results": results,
            "total_results": len(results),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Simple search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/multi", response_model=List[SearchResponse])
async def multi_search(
    request: MultiSearchRequest,
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine)
):
    """Search multiple queries simultaneously."""
    start_time = time.time()
    
    try:
        logger.info(f"Multi-search: {len(request.queries)} queries")
        
        responses = []
        for query in request.queries:
            query_start = time.time()
            
            results = retrieval_engine.search(
                query=query,
                top_k=request.top_k,
                threshold=request.threshold
            )
            
            search_results = [
                SearchResultResponse(
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {}),
                    chunk_id=result.get("id", "")
                ) for result in results
            ]
            
            query_time = time.time() - query_start
            
            responses.append(SearchResponse(
                results=search_results,
                query=query,
                total_results=len(search_results),
                processing_time=query_time
            ))
        
        total_time = time.time() - start_time
        logger.info(f"Multi-search completed in {total_time:.3f}s")
        
        return responses
        
    except Exception as e:
        logger.error(f"Multi-search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multi-search failed: {str(e)}")


@router.post("/hybrid")
async def hybrid_search(
    request: HybridSearchRequest,
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine)
):
    """Hybrid search combining semantic and keyword search."""
    start_time = time.time()
    
    try:
        logger.info(f"Hybrid search: '{request.query}' (semantic: {request.semantic_weight}, keyword: {request.keyword_weight})")
        
        # Perform semantic search
        semantic_results = retrieval_engine.search(
            query=request.query,
            top_k=request.top_k * 2,  # Get more results for combining
            threshold=0.5  # Lower threshold for hybrid
        )
        
        # Apply weights (simplified implementation)
        # In a full implementation, you'd also do keyword search and combine
        weighted_results = []
        for result in semantic_results[:request.top_k]:
            weighted_score = result.get("score", 0.0) * request.semantic_weight
            result["score"] = weighted_score
            weighted_results.append(result)
        
        # Sort by weighted score
        weighted_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        search_results = [
            SearchResultResponse(
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                chunk_id=result.get("id", "")
            ) for result in weighted_results
        ]
        
        processing_time = time.time() - start_time
        
        logger.info(f"Hybrid search completed in {processing_time:.3f}s")
        
        return SearchResponse(
            results=search_results,
            query=request.query,
            total_results=len(search_results),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Hybrid search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")
