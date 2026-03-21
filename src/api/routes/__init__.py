"""
API routes for the ChatGPT Retrieval Plugin.

This module contains all API endpoint definitions organized by functionality.
"""

from .documents import router as documents_router
from .search import router as search_router

__all__ = ["documents_router", "search_router"]
