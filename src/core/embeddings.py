import openai
from typing import List, Union, Optional, Dict
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Simple file-based cache for embeddings."""
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model combination."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Retrieve embedding from cache."""
        cache_key = self.get_cache_key(text, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return data['embedding']
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        return None
    
    def set(self, text: str, model: str, embedding: List[float]):
        """Store embedding in cache."""
        cache_key = self.get_cache_key(text, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            data = {
                'text': text[:100],  # Store first 100 chars for reference
                'model': model,
                'embedding': embedding
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Error writing cache file {cache_file}: {e}")

class EmbeddingService:
    """Service for generating text embeddings using OpenAI API."""
    
    def __init__(self, api_key: str = None, model: str = None, use_cache: bool = True):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.client = openai.OpenAI(api_key=self.api_key)
        self.cache = EmbeddingCache() if use_cache else None
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Validate model and dimension
        self._validate_model()
    
    def _validate_model(self):
        """Validate the embedding model and dimension."""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        if self.model in model_dimensions:
            expected_dim = model_dimensions[self.model]
            if self.dimension != expected_dim:
                logger.warning(
                    f"Dimension mismatch for {self.model}. "
                    f"Expected {expected_dim}, got {self.dimension}"
                )
                self.dimension = expected_dim
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text with retry logic."""
        # Check cache first
        if self.cache:
            cached = self.cache.get(text, self.model)
            if cached:
                logger.debug(f"Retrieved embedding from cache for text: {text[:50]}...")
                return cached
        
        try:
            # Clean text
            text = text.strip()
            if not text:
                return [0.0] * self.dimension
            
            # Generate embedding
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            if self.cache:
                self.cache.set(text, self.model, embedding)
            
            return embedding
            
        except openai.RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        total_texts = len(texts)
        
        # Process in batches
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            
            if show_progress:
                progress = (i + len(batch)) / total_texts * 100
                logger.info(f"Processing embeddings: {progress:.1f}% complete")
            
            try:
                # Try batch processing
                batch_embeddings = self._process_batch(batch)
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error in batch {i//batch_size}: {e}")
                # Fallback to individual processing
                for text in batch:
                    try:
                        embedding = self.generate_embedding(text)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Error processing text: {e}")
                        embeddings.append([0.0] * self.dimension)
        
        return embeddings
    
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a batch of texts."""
        embeddings = []
        
        # Check cache for all texts
        uncached_texts = []
        uncached_indices = []
        
        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.model)
                if cached:
                    embeddings.append(cached)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    embeddings.append(None)  # Placeholder
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Process uncached texts
        if uncached_texts:
            try:
                response = self.client.embeddings.create(
                    input=uncached_texts,
                    model=self.model
                )
                
                # Fill in the embeddings
                for i, (idx, data) in enumerate(zip(uncached_indices, response.data)):
                    embedding = data.embedding
                    embeddings[idx] = embedding
                    
                    # Cache the result
                    if self.cache:
                        self.cache.set(uncached_texts[i], self.model, embedding)
                        
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Process individually as fallback
                for i, text in zip(uncached_indices, uncached_texts):
                    try:
                        embedding = self.generate_embedding(text)
                        embeddings[i] = embedding
                    except:
                        embeddings[i] = [0.0] * self.dimension
        
        return embeddings
    
    async def generate_embedding_async(self, text: str) -> List[float]:
        """Async wrapper for embedding generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate_embedding, text)
    
    async def generate_embeddings_batch_async(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """Async batch embedding generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.generate_embeddings_batch, 
            texts, 
            batch_size
        )
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)