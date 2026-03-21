"""
API module for the ChatGPT Retrieval Plugin.

This module contains the FastAPI application setup and route definitions.
"""

from .main import app
from .middleware import setup_middleware

__all__ = ["app", "setup_middleware"]
