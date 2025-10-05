"""
Configuration management for the RAG system.
Loads settings from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """
    Centralized configuration management.
    Supports YAML config files and environment variable overrides.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._override_with_env()
    
    def _get_default_config_path(self) -> str:
        """Get default config path based on environment"""
        env = os.getenv("APP_ENV", "development")
        base_path = Path(__file__).parent.parent.parent / "configs"
        
        config_file = f"{env}.yaml"
        config_path = base_path / config_file
        
        if not config_path.exists():
            config_path = base_path / "default.yaml"
        
        return str(config_path)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {self.config_path}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found"""
        return {
            "rag": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "top_k": 4,
                "temperature": 0.1
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu"
            },
            "vector_store": {
                "type": "faiss",
                "persist_directory": "./data/vector_db"
            },
            "llm": {
                "provider": "ollama",
                "model": "llama2",
                "temperature": 0.1,
                "max_tokens": 2000
            },
            "paths": {
                "documents": "./data/regulatory_documents",
                "vector_db": "./data/vector_db"
            }
        }
    
    def _override_with_env(self):
        """Override config with environment variables"""
        # LLM settings
        if os.getenv("LLM_PROVIDER"):
            self.config["llm"]["provider"] = os.getenv("LLM_PROVIDER")
        if os.getenv("MODEL_NAME"):
            self.config["llm"]["model"] = os.getenv("MODEL_NAME")
        if os.getenv("OPENAI_API_KEY"):
            self.config["llm"]["api_key"] = os.getenv("OPENAI_API_KEY")
        
        # Paths
        if os.getenv("DOCUMENTS_PATH"):
            self.config["paths"]["documents"] = os.getenv("DOCUMENTS_PATH")
        if os.getenv("VECTOR_DB_PATH"):
            self.config["paths"]["vector_db"] = os.getenv("VECTOR_DB_PATH")
        
        # RAG settings
        if os.getenv("CHUNK_SIZE"):
            self.config["rag"]["chunk_size"] = int(os.getenv("CHUNK_SIZE"))
        if os.getenv("CHUNK_OVERLAP"):
            self.config["rag"]["chunk_overlap"] = int(os.getenv("CHUNK_OVERLAP"))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "rag.chunk_size")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


# Global config instance
config = Config()