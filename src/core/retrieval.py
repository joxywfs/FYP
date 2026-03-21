from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..core.database import VectorDatabase
from ..core.embeddings import EmbeddingService
from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a single search result."""
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id
        }

class RetrievalEngine:
    """Main retrieval engine that orchestrates search operations."""
    
    def __init__(
        self, 
        vector_db: VectorDatabase = None, 
        embedding_service: EmbeddingService = None
    ):
        self.vector_db = vector_db or VectorDatabase()
        self.embedding_service = embedding_service or EmbeddingService()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Search configuration
        self.default_max_results = settings.max_search_results
        self.default_score_threshold = settings.score_threshold
        
        # Cache for query embeddings
        self.query_cache = {}
        self.max_cache_size = 100
    
    def search(
        self, 
        query: str, 
        max_results: int = None,
        score_threshold: float = None,
        filter_dict: Optional[Dict] = None,
        rerank: bool = False
    ) -> List[SearchResult]:
        """
        Perform semantic search for the given query.
        
        Args:
            query: Search query text
            max_results: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_dict: Optional metadata filters
            rerank: Whether to apply reranking
            
        Returns:
            List of SearchResult objects
        """
        # Set defaults
        max_results = max_results or self.default_max_results
        score_threshold = score_threshold or self.default_score_threshold
        
        # Clean and validate query
        query = query.strip()
        if not query:
            logger.warning("Empty query provided")
            return []
        
        try:
            # Get query embedding (with caching)
            query_embedding = self._get_query_embedding(query)
            
            # Perform vector search
            raw_results = self.vector_db.search(
                query_embedding=query_embedding,
                n_results=max_results * 2 if rerank else max_results,  # Get more if reranking
                filter_dict=filter_dict,
                include_distance=True
            )
            print("RAW RESULTS:", raw_results)
            
            # Process results
            search_results = self._process_search_results(
                raw_results, 
                score_threshold
            )
            
            # Apply reranking if requested
            if rerank and search_results:
                search_results = self._rerank_results(
                    query, 
                    search_results, 
                    max_results
                )
            
            # Limit to max_results
            search_results = search_results[:max_results]
            
            # Log search statistics
            logger.info(
                f"Search completed - Query: '{query[:50]}...', "
                f"Results: {len(search_results)}, "
                f"Threshold: {score_threshold}"
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding with caching."""
        # Check cache
        if query in self.query_cache:
            logger.debug(f"Using cached embedding for query: {query[:50]}...")
            return self.query_cache[query]
        
        # Generate embedding
        embedding = self.embedding_service.generate_embedding(query)
        
        # Update cache (with size limit)
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[query] = embedding
        return embedding
    
    def _process_search_results(
        self, 
        raw_results: Dict[str, Any], 
        score_threshold: float
    ) -> List[SearchResult]:
        """Process raw search results into SearchResult objects."""
        search_results = []
        
        for i in range(len(raw_results['ids'])):
            # Calculate similarity score
            if raw_results['scores']:
                score = raw_results['scores'][i]
            else:
                # Calculate from distance if scores not provided
                distance = raw_results['distances'][i] if raw_results['distances'] else 0
                score = 1 - distance
            
            # Apply threshold filtering
            if score >= score_threshold:
                result = SearchResult(
                    content=raw_results['documents'][i],
                    score=score,
                    metadata=raw_results['metadatas'][i] or {},
                    chunk_id=raw_results['ids'][i]
                )
                search_results.append(result)
        
        # Sort by score (descending)
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        return search_results
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[SearchResult], 
        max_results: int
    ) -> List[SearchResult]:
        """
        Apply reranking to improve result relevance.
        Simple implementation - can be enhanced with cross-encoder models.
        """
        # Simple reranking based on keyword overlap
        query_terms = set(query.lower().split())
        
        for result in results:
            content_terms = set(result.content.lower().split())
            
            # Calculate overlap score
            overlap = len(query_terms & content_terms) / len(query_terms)
            
            # Combine with original score
            result.score = (result.score * 0.7) + (overlap * 0.3)
        
        # Re-sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:max_results]
    
    def search_with_context(
        self, 
        query: str, 
        max_results: int = None,
        context_format: str = "detailed"
    ) -> str:
        """
        Search and format results as context for LLM consumption.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            context_format: Format style ('detailed', 'simple', 'numbered')
            
        Returns:
            Formatted context string
        """
        results = self.search(query, max_results)
        
        if not results:
            return "No relevant information found for the query."
        
        if context_format == "simple":
            # Simple concatenation
            context_parts = [result.content for result in results]
            return "\n\n".join(context_parts)
        
        elif context_format == "numbered":
            # Numbered list
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(f"{i}. {result.content}")
            return "\n\n".join(context_parts)
        
        else:  # detailed
            # Detailed format with metadata
            context_parts = []
            for i, result in enumerate(results, 1):
                source = result.metadata.get('source', 'Unknown')
                score = result.score
                
                context_part = (
                    f"Source {i}: {source} (Relevance: {score:.3f})\n"
                    f"{result.content}"
                )
                context_parts.append(context_part)
            
            return "\n\n---\n\n".join(context_parts)
    
    def multi_query_search(
        self, 
        queries: List[str], 
        max_results_per_query: int = 3,
        deduplicate: bool = True
    ) -> List[SearchResult]:
        """
        Perform search with multiple query variations.
        Useful for improving recall.
        """
        all_results = []
        seen_chunks = set()
        
        for query in queries:
            results = self.search(query, max_results=max_results_per_query)
            
            for result in results:
                if deduplicate:
                    if result.chunk_id not in seen_chunks:
                        seen_chunks.add(result.chunk_id)
                        all_results.append(result)
                else:
                    all_results.append(result)
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results
    
    def hybrid_search(
        self, 
        query: str,
        keywords: List[str] = None,
        max_results: int = None,
        keyword_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword matching.
        
        Args:
            query: Semantic search query
            keywords: Optional keywords for filtering/boosting
            max_results: Maximum results
            keyword_weight: Weight for keyword matching (0-1)
            
        Returns:
            List of search results
        """
        # Perform semantic search
        semantic_results = self.search(query, max_results=max_results * 2)
        
        if not keywords:
            return semantic_results[:max_results]
        
        # Score adjustment based on keyword presence
        for result in semantic_results:
            content_lower = result.content.lower()
            keyword_score = 0
            
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    keyword_score += 1
            
            # Normalize keyword score
            if keywords:
                keyword_score = keyword_score / len(keywords)
            
            # Combine scores
            semantic_weight = 1 - keyword_weight
            result.score = (result.score * semantic_weight) + (keyword_score * keyword_weight)
        
        # Re-sort and return
        semantic_results.sort(key=lambda x: x.score, reverse=True)
        return semantic_results[:max_results]
    
    async def search_async(
        self,
        query: str,
        max_results: int = None,
        score_threshold: float = None,
        filter_dict: Optional[Dict] = None,
        rerank: bool = False
    ):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.search(
                query=query,
                max_results=max_results,
                score_threshold=score_threshold,
                filter_dict=filter_dict,
                rerank=rerank
            )
        )
    
    def get_similar_chunks(
        self, 
        chunk_id: str, 
        max_results: int = 5
    ) -> List[SearchResult]:
        """Find chunks similar to a given chunk."""
        # Get the chunk
        doc = self.vector_db.get_document_by_id(chunk_id)
        if not doc or not doc.get('embedding'):
            logger.warning(f"Chunk {chunk_id} not found or has no embedding")
            return []
        
        # Search using its embedding
        raw_results = self.vector_db.search(
            query_embedding=doc['embedding'],
            n_results=max_results + 1  # +1 to exclude self
        )
        print("RAW RESULTS:", raw_results)
        
        # Process results, excluding the original chunk
        search_results = []
        for i in range(len(raw_results['ids'])):
            if raw_results['ids'][i] != chunk_id:
                result = SearchResult(
                    content=raw_results['documents'][i],
                    score=raw_results['scores'][i] if raw_results['scores'] else 1 - raw_results['distances'][i],
                    metadata=raw_results['metadatas'][i] or {},
                    chunk_id=raw_results['ids'][i]
                )
                search_results.append(result)
        
        return search_results[:max_results]
    
    def explain_search_results(
        self, 
        query: str, 
        results: List[SearchResult]
    ) -> Dict[str, Any]:
        """
        Provide explanation for search results.
        Useful for debugging and transparency.
        """
        query_terms = set(query.lower().split())
        
        explanations = []
        for result in results:
            content_terms = set(result.content.lower().split())
            matching_terms = query_terms & content_terms
            
            explanation = {
                "chunk_id": result.chunk_id,
                "score": result.score,
                "matching_terms": list(matching_terms),
                "match_ratio": len(matching_terms) / len(query_terms) if query_terms else 0,
                "source": result.metadata.get('source', 'Unknown'),
                "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
            }
            explanations.append(explanation)
        
        return {
            "query": query,
            "query_terms": list(query_terms),
            "total_results": len(results),
            "explanations": explanations
        }