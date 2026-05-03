"""
Document loading functionality.
Handles different file formats and converts them to LangChain Documents.
"""

import os
from typing import List, Optional
from pathlib import Path
from langchain_classic.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    Docx2txtLoader
)

from ..utils.logger import app_logger
from ..utils.config import config


class DocumentLoader:
    """
    Handles loading of regulatory documents from various formats.
    
    Supports:
    - PDF files
    - Text files (.txt)
    - Word documents (.docx)
    - Directory recursive loading
    """
    
    def __init__(self, documents_path: Optional[str] = None):
        """
        Initialize document loader.
        
        Args:
            documents_path: Path to documents directory
        """
        self.documents_path = documents_path or config.get("paths.documents")
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """Create documents directory if it doesn't exist"""
        path = Path(self.documents_path)
        if not path.exists():
            app_logger.warning(f"Documents directory not found: {self.documents_path}")
            path.mkdir(parents=True, exist_ok=True)
            app_logger.info(f"Created documents directory: {self.documents_path}")
    
    def load_pdf_documents(self) -> List[Document]:
        """
        Load all PDF documents from the documents directory.
        
        Returns:
            List of Document objects from PDFs
        """
        app_logger.info("Loading PDF documents...")
        
        try:
            loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True
            )
            documents = loader.load()
            app_logger.info(f"Loaded {len(documents)} pages from PDF files")
            return documents
        except Exception as e:
            app_logger.error(f"Error loading PDF documents: {e}")
            return []
    
    def load_text_documents(self) -> List[Document]:
        """
        Load all text documents from the documents directory.
        
        Returns:
            List of Document objects from text files
        """
        app_logger.info("Loading text documents...")
        
        try:
            loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            documents = loader.load()
            app_logger.info(f"Loaded {len(documents)} text files")
            return documents
        except Exception as e:
            app_logger.error(f"Error loading text documents: {e}")
            return []
    
    def load_docx_documents(self) -> List[Document]:
        """
        Load all Word documents from the documents directory.
        
        Returns:
            List of Document objects from DOCX files
        """
        app_logger.info("Loading Word documents...")
        
        try:
            loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.docx",
                loader_cls=Docx2txtLoader,
                show_progress=True
            )
            documents = loader.load()
            app_logger.info(f"Loaded {len(documents)} Word documents")
            return documents
        except Exception as e:
            app_logger.error(f"Error loading Word documents: {e}")
            return []
    
    def load_all_documents(self) -> List[Document]:
        """
        Load all supported document types.
        
        Returns:
            Combined list of all documents
        """
        app_logger.info(f"Loading all documents from {self.documents_path}")
        
        all_documents = []
        
        # Load different file types
        all_documents.extend(self.load_pdf_documents())
        all_documents.extend(self.load_text_documents())
        all_documents.extend(self.load_docx_documents())
        
        app_logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def load_single_file(self, file_path: str) -> List[Document]:
        """
        Load a single document file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects (may be multiple pages)
        """
        app_logger.info(f"Loading single file: {file_path}")
        
        path = Path(file_path)
        
        if not path.exists():
            app_logger.error(f"File not found: {file_path}")
            return []
        
        try:
            # Determine loader based on file extension
            if path.suffix == ".pdf":
                loader = PyPDFLoader(file_path)
            elif path.suffix == ".txt":
                loader = TextLoader(file_path)
            elif path.suffix == ".docx":
                loader = Docx2txtLoader(file_path)
            else:
                app_logger.warning(f"Unsupported file type: {path.suffix}")
                return []
             
            documents = loader.load()
            app_logger.info(f"Loaded {len(documents)} pages from {path.name}")
            return documents
            
        except Exception as e:
            app_logger.error(f"Error loading file {file_path}: {e}")
            return []

if __name__ == "__main__":
    # Example usage: read a text file from ./regulatory_documents
    loader = DocumentLoader()
    documents = loader.load_all_documents()
    print("--- All Documents Loaded ---")
    print(documents)
    for doc in documents:
        print("--- Document Start ---")
        print(doc.page_content)
        print("--- Document End ---\n" )
