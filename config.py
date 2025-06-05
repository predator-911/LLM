import os
from typing import Optional

class Config:
    # Groq API Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
    # Embedding Configuration (using free sentence-transformers)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    
    # Database Configuration (SQLite for free deployment)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/documents.db")
    
    # Document Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_DOCUMENTS: int = int(os.getenv("MAX_DOCUMENTS", "20"))
    MAX_PAGES_PER_DOCUMENT: int = int(os.getenv("MAX_PAGES_PER_DOCUMENT", "1000"))
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    # Application Configuration
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Retrieval Configuration
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Data Directory
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Create data directory if it doesn't exist
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.VECTOR_STORE_PATH, exist_ok=True)
        
        return True

# Validate configuration on import
Config.validate_config()