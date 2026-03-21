"""
Input validation utilities for the ChatGPT Retrieval Plugin.
"""

import re
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, validator, ValidationError
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)


class FileValidator:
    """File validation utilities."""
    
    # Supported file types and their MIME types
    SUPPORTED_TYPES = {
        '.pdf': ['application/pdf'],
        '.txt': ['text/plain'],
        '.docx': [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ],
        '.doc': ['application/msword'],
        '.md': ['text/markdown', 'text/plain'],
        '.rtf': ['application/rtf', 'text/rtf']
    }
    
    # File size limits (in bytes)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MIN_FILE_SIZE = 1  # 1 byte
    
    @classmethod
    def validate_file_type(cls, filename: str) -> bool:
        """Validate if file type is supported."""
        file_ext = Path(filename).suffix.lower()
        return file_ext in cls.SUPPORTED_TYPES
    
    @classmethod
    def validate_file_size(cls, file_size: int) -> bool:
        """Validate file size is within limits."""
        return cls.MIN_FILE_SIZE <= file_size <= cls.MAX_FILE_SIZE
    
    @classmethod
    def validate_filename(cls, filename: str) -> bool:
        """Validate filename format."""
        if not filename or len(filename.strip()) == 0:
            return False
        
        # Check for invalid characters
        invalid_chars = r'[<>:"/\\|?*]'
        if re.search(invalid_chars, filename):
            return False
        
        # Check length
        if len(filename) > 255:
            return False
        
        return True
    
    @classmethod
    def validate_mime_type(cls, filename: str, content_type: str = None) -> bool:
        """Validate MIME type matches file extension."""
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in cls.SUPPORTED_TYPES:
            return False
        
        if content_type:
            expected_types = cls.SUPPORTED_TYPES[file_ext]
            return content_type in expected_types
        
        return True
    
    @classmethod
    def get_validation_errors(cls, filename: str, file_size: int, content_type: str = None) -> List[str]:
        """Get all validation errors for a file."""
        errors = []
        
        if not cls.validate_filename(filename):
            errors.append(f"Invalid filename: '{filename}'")
        
        if not cls.validate_file_type(filename):
            file_ext = Path(filename).suffix.lower()
            supported = ', '.join(cls.SUPPORTED_TYPES.keys())
            errors.append(f"Unsupported file type: '{file_ext}'. Supported types: {supported}")
        
        if not cls.validate_file_size(file_size):
            max_mb = cls.MAX_FILE_SIZE / (1024 * 1024)
            errors.append(f"File size {file_size} bytes is invalid. Must be between {cls.MIN_FILE_SIZE} and {max_mb}MB")
        
        if content_type and not cls.validate_mime_type(filename, content_type):
            errors.append(f"MIME type '{content_type}' doesn't match file extension")
        
        return errors


class QueryValidator:
    """Search query validation utilities."""
    
    MIN_QUERY_LENGTH = 1
    MAX_QUERY_LENGTH = 1000
    
    # Patterns for potentially problematic queries
    SUSPICIOUS_PATTERNS = [
        r'<script.*?</script>',  # XSS
        r'javascript:',
        r'on\w+\s*=',  # Event handlers
        r'SELECT.*FROM',  # SQL injection
        r'UNION.*SELECT',
        r'DROP\s+TABLE',
        r'INSERT\s+INTO',
        r'UPDATE.*SET',
        r'DELETE\s+FROM'
    ]
    
    @classmethod
    def validate_query_length(cls, query: str) -> bool:
        """Validate query length."""
        return cls.MIN_QUERY_LENGTH <= len(query.strip()) <= cls.MAX_QUERY_LENGTH
    
    @classmethod
    def validate_query_content(cls, query: str) -> bool:
        """Validate query doesn't contain suspicious patterns."""
        query_lower = query.lower()
        
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return False
        
        return True
    
    @classmethod
    def sanitize_query(cls, query: str) -> str:
        """Sanitize search query."""
        # Remove leading/trailing whitespace
        query = query.strip()
        
        # Remove control characters
        query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    @classmethod
    def validate_and_sanitize(cls, query: str) -> Tuple[str, List[str]]:
        """Validate and sanitize query, return sanitized query and errors."""
        errors = []
        
        if not query or not isinstance(query, str):
            errors.append("Query must be a non-empty string")
            return "", errors
        
        # Sanitize first
        sanitized = cls.sanitize_query(query)
        
        # Validate length
        if not cls.validate_query_length(sanitized):
            errors.append(f"Query length must be between {cls.MIN_QUERY_LENGTH} and {cls.MAX_QUERY_LENGTH} characters")
        
        # Validate content
        if not cls.validate_query_content(sanitized):
            errors.append("Query contains potentially harmful content")
        
        return sanitized, errors


class ParameterValidator:
    """Parameter validation utilities."""
    
    @staticmethod
    def validate_top_k(top_k: int) -> bool:
        """Validate top_k parameter."""
        return isinstance(top_k, int) and 1 <= top_k <= 100
    
    @staticmethod
    def validate_threshold(threshold: float) -> bool:
        """Validate similarity threshold."""
        return isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0
    
    @staticmethod
    def validate_page_params(page: int, per_page: int) -> bool:
        """Validate pagination parameters."""
        return (
            isinstance(page, int) and page >= 1 and
            isinstance(per_page, int) and 1 <= per_page <= 100
        )
    
    @staticmethod
    def validate_uuid(uuid_string: str) -> bool:
        """Validate UUID format."""
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(uuid_pattern, uuid_string.lower()))
    
    @staticmethod
    def validate_filters(filters: Dict[str, Any]) -> List[str]:
        """Validate search filters."""
        errors = []
        
        if not isinstance(filters, dict):
            errors.append("Filters must be a dictionary")
            return errors
        
        # Check for reasonable filter complexity
        if len(filters) > 10:
            errors.append("Too many filters specified (max 10)")
        
        # Validate filter values
        for key, value in filters.items():
            if not isinstance(key, str):
                errors.append(f"Filter key must be string, got {type(key)}")
            
            if len(key) > 100:
                errors.append(f"Filter key '{key}' is too long (max 100 chars)")
            
            # Basic type validation for values
            if isinstance(value, str) and len(value) > 1000:
                errors.append(f"Filter value for '{key}' is too long")
        
        return errors


class MetadataValidator:
    """Metadata validation utilities."""
    
    MAX_METADATA_SIZE = 10000  # 10KB
    MAX_METADATA_FIELDS = 50
    
    @classmethod
    def validate_metadata(cls, metadata: Dict[str, Any]) -> List[str]:
        """Validate document metadata."""
        errors = []
        
        if not isinstance(metadata, dict):
            errors.append("Metadata must be a dictionary")
            return errors
        
        # Check field count
        if len(metadata) > cls.MAX_METADATA_FIELDS:
            errors.append(f"Too many metadata fields (max {cls.MAX_METADATA_FIELDS})")
        
        # Check total size (approximate)
        metadata_str = str(metadata)
        if len(metadata_str) > cls.MAX_METADATA_SIZE:
            errors.append(f"Metadata too large (max {cls.MAX_METADATA_SIZE} bytes)")
        
        # Validate field names and values
        for key, value in metadata.items():
            if not isinstance(key, str):
                errors.append(f"Metadata key must be string, got {type(key)}")
                continue
            
            if len(key) > 100:
                errors.append(f"Metadata key '{key}' too long (max 100 chars)")
            
            # Check for reserved fields
            if key.startswith('_'):
                errors.append(f"Metadata key '{key}' is reserved (starts with underscore)")
            
            # Validate value types
            if not isinstance(value, (str, int, float, bool, type(None))):
                errors.append(f"Metadata value for '{key}' has unsupported type: {type(value)}")
        
        return errors


def validate_search_request(
    query: str,
    top_k: int = 5,
    threshold: float = 0.7,
    filters: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a complete search request."""
    errors = []
    
    # Validate and sanitize query
    sanitized_query, query_errors = QueryValidator.validate_and_sanitize(query)
    errors.extend(query_errors)
    
    # Validate parameters
    if not ParameterValidator.validate_top_k(top_k):
        errors.append(f"Invalid top_k: {top_k}. Must be integer between 1 and 100")
    
    if not ParameterValidator.validate_threshold(threshold):
        errors.append(f"Invalid threshold: {threshold}. Must be float between 0.0 and 1.0")
    
    # Validate filters
    if filters is not None:
        filter_errors = ParameterValidator.validate_filters(filters)
        errors.extend(filter_errors)
    
    # Return sanitized parameters
    sanitized_params = {
        'query': sanitized_query,
        'top_k': top_k,
        'threshold': threshold,
        'filters': filters
    }
    
    return sanitized_params, errors


def validate_upload_request(
    filename: str,
    file_size: int,
    content_type: str = None,
    metadata: Dict[str, Any] = None
) -> List[str]:
    """Validate a file upload request."""
    errors = []
    
    # Validate file
    file_errors = FileValidator.get_validation_errors(filename, file_size, content_type)
    errors.extend(file_errors)
    
    # Validate metadata
    if metadata is not None:
        metadata_errors = MetadataValidator.validate_metadata(metadata)
        errors.extend(metadata_errors)
    
    return errors
