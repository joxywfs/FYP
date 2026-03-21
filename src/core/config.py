import os
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(1536, env="EMBEDDING_DIMENSION")
    
    # Database Configuration
    chroma_persist_directory: str = Field("./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    collection_name: str = Field("knowledge_base", env="COLLECTION_NAME")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_prefix: str = Field("/api/v1", env="API_PREFIX")
    cors_origins: List[str] = Field(
        ["https://chat.openai.com", "https://chatgpt.com"],
        env="CORS_ORIGINS"
    )
    
    # Chunking Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(100, env="CHUNK_OVERLAP")
    
    # Search Configuration
    max_search_results: int = Field(5, env="MAX_SEARCH_RESULTS")
    score_threshold: float = Field(0.7, env="SCORE_THRESHOLD")
    
    # Rate Limiting
    max_requests_per_minute: int = Field(60, env="MAX_REQUESTS_PER_MINUTE")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("retrieval_plugin.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Create necessary directories
Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)