"""
FastAPI middleware for CORS and other cross-cutting concerns.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)

def add_cors_middleware(app: FastAPI) -> None:
    """Add CORS middleware to the FastAPI application."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS middleware added to application")

def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the application."""
    add_cors_middleware(app)
    logger.info("All middleware configured successfully")
