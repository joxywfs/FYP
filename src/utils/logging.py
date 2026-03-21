"""
Logging configuration and utilities for the ChatGPT Retrieval Plugin.
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager

from ..core.config import settings


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and structured logging."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors and additional info."""
        # Add color for console output
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.colored_levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # Add timestamp
        record.timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Add module info
        record.module_name = record.module
        record.func_name = record.funcName
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class PerformanceLogger:
    """Logger for tracking performance metrics."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
    
    @contextmanager
    def time_operation(self, operation_name: str, **kwargs):
        """Context manager to time operations."""
        start_time = datetime.now()
        try:
            yield
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Operation '{operation_name}' completed",
                extra={
                    'operation': operation_name,
                    'duration_seconds': duration,
                    'timestamp': start_time.isoformat(),
                    **kwargs
                }
            )


def setup_logging(
    log_level: str = None,
    log_file: str = None,
    enable_json_logging: bool = False,
    enable_console_colors: bool = True
) -> None:
    """Set up logging configuration for the application."""
    
    # Use settings or defaults
    log_level = log_level or getattr(settings, 'log_level', 'INFO')
    log_file = log_file or getattr(settings, 'log_file', 'app.log')
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if enable_console_colors:
        console_format = (
            "%(colored_levelname)s - %(timestamp)s - %(name)s - %(module_name)s:%(func_name)s:%(lineno)d - %(message)s"
        )
        console_handler.setFormatter(CustomFormatter(console_format))
    else:
        console_format = (
            "%(levelname)s - %(asctime)s - %(name)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        console_handler.setFormatter(logging.Formatter(console_format))
    
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    if enable_json_logging:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_format = (
            "%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(logging.Formatter(file_format))
    
    root_logger.addHandler(file_handler)
    
    # Set up specific loggers
    setup_specific_loggers()
    
    logging.info(f"Logging configured - Level: {log_level}, File: {log_file}")


def setup_specific_loggers():
    """Configure specific loggers for different components."""
    
    # Database operations logger
    db_logger = logging.getLogger("database")
    db_logger.setLevel(logging.INFO)
    
    # Embeddings logger
    embeddings_logger = logging.getLogger("embeddings")
    embeddings_logger.setLevel(logging.INFO)
    
    # API logger
    api_logger = logging.getLogger("api")
    api_logger.setLevel(logging.INFO)
    
    # Retrieval logger
    retrieval_logger = logging.getLogger("retrieval")
    retrieval_logger.setLevel(logging.INFO)
    
    # Silence noisy third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def log_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    duration: float,
    user_id: Optional[str] = None,
    **kwargs
):
    """Log API request details."""
    logger = logging.getLogger("api")
    logger.info(
        f"{method} {endpoint} - {status_code}",
        extra={
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'duration_seconds': duration,
            'user_id': user_id,
            **kwargs
        }
    )


def log_search_query(
    query: str,
    results_count: int,
    processing_time: float,
    user_id: Optional[str] = None,
    **kwargs
):
    """Log search query details."""
    logger = logging.getLogger("retrieval")
    logger.info(
        f"Search query processed: '{query}' -> {results_count} results",
        extra={
            'query': query,
            'results_count': results_count,
            'processing_time': processing_time,
            'user_id': user_id,
            **kwargs
        }
    )


def log_document_upload(
    filename: str,
    file_size: int,
    chunks_created: int,
    processing_time: float,
    user_id: Optional[str] = None,
    **kwargs
):
    """Log document upload details."""
    logger = logging.getLogger("database")
    logger.info(
        f"Document uploaded: {filename} -> {chunks_created} chunks",
        extra={
            'filename': filename,
            'file_size': file_size,
            'chunks_created': chunks_created,
            'processing_time': processing_time,
            'user_id': user_id,
            **kwargs
        }
    )


# Performance logger instance
performance_logger = PerformanceLogger()

# Initialize logging if this module is imported
if not logging.getLogger().handlers:
    setup_logging()
