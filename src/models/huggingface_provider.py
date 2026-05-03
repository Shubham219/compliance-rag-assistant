# Use Hugging Face Inference API for LLM calls
# Supports models like Llama 3.1, Mistral, CodeLlama, etc. via featherless-ai
# https://huggingface.co/featherless-ai/meta-llama-3.1-8b

from typing import Optional
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from getpass import getpass
from .llm_factory import BaseLLMProvider
from ..utils.logger import app_logger
from ..utils.config import config


# HUGGINGFACEHUB_API_TOKEN = getpass()

class HuggingFaceProvider(BaseLLMProvider):
    """
    Hugging Face LLM Provider.
    
    Supports models via Hugging Face Inference API.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Hugging Face provider.
        
        Args:
            model_name: Hugging Face model name
            temperature: Generation temperature
            api_key: Hugging Face API key (or from environment)
            **kwargs: Additional HuggingFaceEndpoint arguments
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.provider = kwargs.get("provider", "featherless-ai")
        self.kwargs = kwargs
        
        if not self.api_key:
            app_logger.error("Hugging Face API key not provided")
            raise ValueError("Hugging Face API key is required")
        
        self.llm = self._create_llm()
    
    def _create_llm(self):
        """Create Hugging Face LLM instance"""
        try:
            app_logger.info(f"Creating Hugging Face LLM: {self.model_name}")
            
            llm = HuggingFaceEndpoint(
                repo_id=self.model_name,
                temperature=self.temperature,
                huggingfacehub_api_token=self.api_key,
                max_new_tokens=config.get("llm.max_tokens", 2000),
                **self.kwargs
            )
            
            app_logger.info("Hugging Face LLM created successfully")
            return llm
            
        except Exception as e:
            app_logger.error(f"Error creating Hugging Face LLM: {e}")
            raise
    
    def get_llm(self):
        """Get the LLM instance"""
        return self.llm
    
if __name__ == "__main__":
    # Example usage
    provider = HuggingFaceProvider(
        model_name="meta-llama/Llama-3.1-8B",
        temperature=0.1,
        api_key=''
    )
    prompt_template = """You are a helpful assistant that provides accurate information. Answer the user query
        query:"{question}"
        answer:
        """
    prompt = PromptTemplate.from_template(prompt_template)
    llm = provider.get_llm()
    llm_chain = prompt | llm
    response = llm_chain.invoke({"question": "Why February has 28 days?"})
    print(response)



