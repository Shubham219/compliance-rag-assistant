"""
LLM Factory - Factory pattern for creating LLM instances.
Supports multiple providers with a unified interface.
"""

from typing import Optional
from abc import ABC, abstractmethod

from ..utils.logger import app_logger
from ..utils.config import config


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def get_llm(self):
        """Get LLM instance"""
        pass


class LLMFactory:
    """
    Factory for creating LLM instances.
    
    Supports:
    - OpenAI (GPT models)
    - Ollama (Local models)
    - Hugging Face (Cloud models)
    - Can be extended for other providers
    """
    
    @staticmethod
    def create_llm(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Create an LLM instance based on provider.
        
        Args:
            provider: LLM provider ('openai', 'ollama')
            model_name: Model name
            temperature: Generation temperature
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLM instance
        """
        provider = provider or config.get("llm.provider", "ollama")
        model_name = model_name or config.get("llm.model", "llama2")
        temperature = temperature or config.get("llm.temperature", 0.1)
        
        app_logger.info(f"Creating LLM: provider={provider}, model={model_name}")
        
        if provider.lower() == "openai":
            from .openai_provider import OpenAIProvider
            return OpenAIProvider(
                model_name=model_name,
                temperature=temperature,
                **kwargs
            ).get_llm()
        
        elif provider.lower() == "ollama":
            from .ollama_provider import OllamaProvider
            return OllamaProvider(
                model=model_name,
                temperature=temperature,
                **kwargs
            ).get_llm()
        
        elif provider.lower() == "huggingface":
            from .huggingface_provider import HuggingFaceProvider
            return HuggingFaceProvider(
                model_name=model_name,
                temperature=temperature,
                **kwargs
            ).get_llm()
        
        else:
            app_logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported provider: {provider}")
