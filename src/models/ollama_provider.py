"""
Ollama LLM Provider for local models.
"""

from typing import Optional
from langchain_community.llms import Ollama

from .llm_factory import BaseLLMProvider
from ..utils.logger import app_logger
from ..utils.config import config


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM Provider.
    
    Supports local models via Ollama.
    """
    
    def __init__(
        self,
        model: str = config.get("llm.model", "llama2"),
        temperature: float = config.get("llm.temperature", 0.1),
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Ollama model name (llama2, mistral, etc.)
            temperature: Generation temperature
            base_url: Ollama API base URL
            **kwargs: Additional Ollama arguments
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url or config.get("llm.base_url", "http://localhost:11434")
        self.kwargs = kwargs
        
        self.llm = self._create_llm()
    
    def _create_llm(self):
        """Create Ollama LLM instance"""
        try:
            app_logger.info(f"Creating Ollama LLM: {self.model}")
            
            llm = Ollama(
                model=self.model,
                temperature=self.temperature,
                base_url=self.base_url,
                **self.kwargs
            )
            
            app_logger.info("Ollama LLM created successfully")
            return llm
            
        except Exception as e:
            app_logger.error(f"Error creating Ollama LLM: {e}")
            raise
    
    def get_llm(self):
        """Get the LLM instance"""
        return self.llm
