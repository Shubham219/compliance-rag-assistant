"""
OpenAI LLM Provider.
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI

from .llm_factory import BaseLLMProvider
from ..utils.logger import app_logger
from ..utils.config import config


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM Provider.
    
    Supports GPT models via OpenAI API.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model_name: OpenAI model name (gpt-3.5-turbo, gpt-4, etc.)
            temperature: Generation temperature
            api_key: OpenAI API key (or from environment)
            **kwargs: Additional ChatOpenAI arguments
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or config.get("llm.api_key")
        self.kwargs = kwargs
        
        if not self.api_key:
            app_logger.error("OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        self.llm = self._create_llm()
    
    def _create_llm(self):
        """Create OpenAI LLM instance"""
        try:
            app_logger.info(f"Creating OpenAI LLM: {self.model_name}")
            
            llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key,
                max_tokens=config.get("llm.max_tokens", 2000),
                **self.kwargs
            )
            
            app_logger.info("OpenAI LLM created successfully")
            return llm
            
        except Exception as e:
            app_logger.error(f"Error creating OpenAI LLM: {e}")
            raise
    
    def get_llm(self):
        """Get the LLM instance"""
        return self.llm

