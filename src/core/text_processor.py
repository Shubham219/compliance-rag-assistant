"""
Text processing and chunking functionality.
Splits documents into optimal chunks for RAG retrieval.
"""

from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..utils.logger import app_logger
from ..utils.config import config


class TextProcessor:
    """
    Handles text chunking and processing for RAG system.
    
    Uses RecursiveCharacterTextSplitter for intelligent document splitting
    that maintains semantic coherence.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize text processor.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size or config.get("rag.chunk_size", 1000)
        self.chunk_overlap = chunk_overlap or config.get("rag.chunk_overlap", 200)
        
        self.text_splitter = self._create_text_splitter()
        
        app_logger.info(
            f"TextProcessor initialized: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Create text splitter with optimal settings.
        
        Returns:
            Configured RecursiveCharacterTextSplitter
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentences
                " ",     # Words
                ""       # Characters
            ],
            keep_separator=True
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of document chunks
        """
        if not documents:
            app_logger.warning("No documents provided for chunking")
            return []
        
        app_logger.info(f"Chunking {len(documents)} documents...")
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            app_logger.info(
                f"Created {len(chunks)} chunks from {len(documents)} documents "
                f"(avg {len(chunks)//len(documents) if documents else 0} chunks per document)"
            )
            
            return chunks
            
        except Exception as e:
            app_logger.error(f"Error chunking documents: {e}")
            return []
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split a single text string into chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of document chunks
        """
        if not text:
            return []
        
        try:
            chunks = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata or {}]
            )
            
            app_logger.debug(f"Created {len(chunks)} chunks from text")
            return chunks
            
        except Exception as e:
            app_logger.error(f"Error chunking text: {e}")
            return []
    
    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """
        Calculate statistics about chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes)
        }
    
# Example usage:
if __name__ == "__main__":

    # Read text from document_loader example
    from .document_loader import DocumentLoader

    documents_dir = "./regulatory_documents"
    loader = DocumentLoader(documents_path=documents_dir)
    documents = loader.load_all_documents()

    processor = TextProcessor()
    chunks = processor.chunk_documents(documents)

    for chunk in chunks:
        print("--- Chunk Start ---")
        print(chunk.page_content)
        print("--- Chunk End ---\n" )