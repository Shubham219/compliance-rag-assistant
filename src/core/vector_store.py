"""
Vector store management for document storage and retrieval.
Handles FAISS vector database operations.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from ..utils.logger import app_logger
from ..utils.config import config


class VectorStoreManager:
    """
    Manages FAISS vector store for document storage and retrieval.
    
    Handles:
    - Creating new vector stores
    - Loading existing vector stores
    - Adding documents
    - Similarity search
    - Persistence
    """
    
    def __init__(
        self,
        embeddings,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize vector store manager.
        
        Args:
            embeddings: Embedding model instance
            persist_directory: Directory to save/load vector store
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory or config.get(
            "paths.vector_db",
            "./data/vector_db"
        )
        self.vector_store = None
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        app_logger.info(f"VectorStoreManager initialized with path: {self.persist_directory}")
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of document chunks to index
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            app_logger.warning("No documents provided for vector store creation")
            return False
        
        try:
            app_logger.info(f"Creating vector store from {len(documents)} documents...")
            
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            app_logger.info("Vector store created successfully")
            return True
            
        except Exception as e:
            app_logger.error(f"Error creating vector store: {e}")
            return False
    
    def load_vector_store(self) -> bool:
        """
        Load existing vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(self.persist_directory, "index.faiss")
        
        if not os.path.exists(index_path):
            app_logger.warning(f"Vector store not found at {self.persist_directory}")
            return False
        
        try:
            app_logger.info(f"Loading vector store from {self.persist_directory}")
            
            self.vector_store = FAISS.load_local(
                folder_path=self.persist_directory,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            app_logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            app_logger.error(f"Error loading vector store: {e}")
            return False
    
    def save_vector_store(self) -> bool:
        """
        Save vector store to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        if self.vector_store is None:
            app_logger.warning("No vector store to save")
            return False
        
        try:
            app_logger.info(f"Saving vector store to {self.persist_directory}")
            
            self.vector_store.save_local(self.persist_directory)
            
            app_logger.info("Vector store saved successfully")
            return True
            
        except Exception as e:
            app_logger.error(f"Error saving vector store: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add new documents to existing vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        if self.vector_store is None:
            app_logger.error("Vector store not initialized. Create or load first.")
            return False
        
        if not documents:
            app_logger.warning("No documents provided to add")
            return False
        
        try:
            app_logger.info(f"Adding {len(documents)} documents to vector store")
            
            self.vector_store.add_documents(documents)
            
            app_logger.info("Documents added successfully")
            return True
            
        except Exception as e:
            app_logger.error(f"Error adding documents: {e}")
            return False
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            app_logger.error("Vector store not initialized")
            return []
        
        try:
            app_logger.debug(f"Similarity search for: '{query[:50]}...' (k={k})")
            
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            app_logger.debug(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            app_logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            app_logger.error("Vector store not initialized")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            app_logger.debug(f"Found {len(results)} results with scores")
            return results
            
        except Exception as e:
            app_logger.error(f"Error in similarity search with score: {e}")
            return []
    
    def get_retriever(self, k: int = 4):
        """
        Get retriever interface for use with LangChain chains.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            VectorStoreRetriever instance
        """
        if self.vector_store is None:
            app_logger.error("Vector store not initialized")
            return None
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def get_stats(self) -> dict:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.vector_store is None:
            return {"status": "not_initialized", "num_vectors": 0}
        
        try:
            num_vectors = self.vector_store.index.ntotal
            return {
                "status": "initialized",
                "num_vectors": num_vectors,
                "persist_directory": self.persist_directory
            }
        except:
            return {"status": "error", "num_vectors": 0}
