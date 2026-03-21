"""
ChatGPT Retrieval Plugin - A FastAPI-based document retrieval system.

This package provides a complete retrieval plugin implementation with:
- Vector database storage using ChromaDB
- OpenAI embeddings for semantic search
- FastAPI REST API endpoints
- Document processing and chunking
- Comprehensive evaluation tools
"""

__version__ = "1.0.0"
__author__ = "Joey Lim Hui Ming - U2120303G"
__description__ = "ChatGPT Retrieval Plugin Development with Vector Database"

# Main exports
from .core.config import settings
from .core.database import VectorDatabase
from .core.embeddings import EmbeddingService
from .core.retrieval import RetrievalEngine
from .utils.preprocessing import DocumentProcessor

__all__ = [
    "settings",
    "VectorDatabase", 
    "EmbeddingService",
    "RetrievalEngine",
    "DocumentProcessor"
]
