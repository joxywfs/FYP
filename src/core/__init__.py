"""
Core module containing the main business logic components.

This module provides the core functionality for:
- Configuration management
- Vector database operations  
- Text embeddings
- Document retrieval
"""

from .config import settings
from .database import VectorDatabase
from .embeddings import EmbeddingService
from .retrieval import RetrievalEngine

__all__ = ["settings", "VectorDatabase", "EmbeddingService", "RetrievalEngine"]
