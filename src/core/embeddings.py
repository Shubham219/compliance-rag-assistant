"""
Embedding model management.
Handles creation and configuration of embedding models.
"""

from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..utils.logger import app_logger
from ..utils.config import config


class EmbeddingManager:
    """
    Manages embedding models for the RAG system.
    
    Currently supports HuggingFace sentence-transformers models.
    Can be extended to support OpenAI embeddings or custom models.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name or config.get(
            "embeddings.model",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = device or config.get("embeddings.device", "cpu")
        
        self.embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        app_logger.info(f"Initializing embedding model: {self.model_name}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={
                    'normalize_embeddings': True,  # Normalize for cosine similarity
                    'batch_size': 32  # Process multiple texts at once
                }
            )
            
            app_logger.info(f"Embedding model loaded successfully on {self.device}")
            
        except Exception as e:
            app_logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def get_embeddings(self):
        """
        Get the embedding model instance.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        if self.embeddings is None:
            self._initialize_embeddings()
        
        return self.embeddings
    
    def embed_query(self, text: str) -> list:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            app_logger.error(f"Error embedding query: {e}")
            return []
    
    def embed_documents(self, texts: list) -> list:
        """
        Embed multiple document texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            app_logger.error(f"Error embedding documents: {e}")
            return []
        
if __name__ == "__main__":
    # Example usage
    manager = EmbeddingManager()
    embed_model = manager.get_embeddings()
    
    sample_text = "This is a sample document for embedding."
    query_embedding = manager.embed_query(sample_text)
    # doc_embeddings = manager.embed_documents([sample_text, "Another document."])
    
    print(f"Query Embedding: {query_embedding}")
    # print(f"Document Embeddings: {doc_embeddings}")