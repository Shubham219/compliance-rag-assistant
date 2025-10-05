"""
Main RAG Engine - Orchestrates all components.
This is the core class that brings everything together.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from .document_loader import DocumentLoader
from .text_processor import TextProcessor
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from ..models.llm_factory import LLMFactory
from ..utils.logger import app_logger
from ..utils.config import config


class RegulatoryComplianceRAG:
    """
    Main RAG Engine for Regulatory Compliance.
    
    Orchestrates all components:
    1. Document loading
    2. Text processing
    3. Embeddings
    4. Vector storage
    5. LLM interaction
    6. Query processing
    """
    
    def __init__(
        self,
        documents_path: Optional[str] = None,
        vector_db_path: Optional[str] = None,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        auto_load: bool = True
    ):
        """
        Initialize the RAG engine.
        
        Args:
            documents_path: Path to regulatory documents
            vector_db_path: Path to vector database
            llm_provider: LLM provider ('openai', 'ollama')
            model_name: Model name
            temperature: Generation temperature
            auto_load: Automatically load or create vector store
        """
        app_logger.info("Initializing Regulatory Compliance RAG Engine...")
        
        # Store configuration
        self.documents_path = documents_path or config.get("paths.documents")
        self.vector_db_path = vector_db_path or config.get("paths.vector_db")
        self.llm_provider = llm_provider or config.get("llm.provider")
        self.model_name = model_name or config.get("llm.model")
        self.temperature = temperature or config.get("llm.temperature")
        
        # Initialize components
        self.document_loader = DocumentLoader(self.documents_path)
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store_manager = VectorStoreManager(
            embeddings=self.embedding_manager.get_embeddings(),
            persist_directory=self.vector_db_path
        )
        self.llm = LLMFactory.create_llm(
            provider=self.llm_provider,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        # QA Chain (will be created after vector store is ready)
        self.qa_chain = None
        
        # Query history
        self.query_history = []
        
        # Setup vector store
        if auto_load:
            self._setup_vector_store()
            self._create_qa_chain()
        
        app_logger.info("RAG Engine initialization complete ✓")
    
    def _setup_vector_store(self):
        """Setup vector store - load existing or create new"""
        app_logger.info("Setting up vector store...")
        
        # Try to load existing vector store
        if self.vector_store_manager.load_vector_store():
            app_logger.info("Existing vector store loaded")
            return
        
        # Create new vector store if not exists
        app_logger.info("No existing vector store found. Creating new one...")
        
        # Load documents
        documents = self.document_loader.load_all_documents()
        
        if not documents:
            app_logger.warning("No documents found. Creating empty vector store.")
            # Create dummy document to initialize
            documents = [Document(
                page_content="Placeholder document for initialization",
                metadata={"source": "system"}
            )]
        
        # Process documents
        chunks = self.text_processor.chunk_documents(documents)
        
        # Create vector store
        self.vector_store_manager.create_vector_store(chunks)
        self.vector_store_manager.save_vector_store()
        
        app_logger.info("Vector store created and saved")
    
    def _create_qa_chain(self):
        """Create the question-answering chain"""
        app_logger.info("Creating QA chain...")
        
        # Create custom prompt for compliance queries
        prompt_template = """You are a Regulatory Compliance Assistant specialized in helping organizations understand and comply with regulations.

Use the following context from regulatory documents to answer the compliance question. If you cannot find the answer in the context, say so - never make up regulatory information.

Context:
{context}

Question: {question}

Instructions:
1. Provide accurate, detailed compliance guidance based on the context
2. Cite specific regulations or sections when possible
3. If the context doesn't contain enough information, acknowledge this
4. Use clear, professional language suitable for compliance officers
5. If relevant, mention potential risks of non-compliance

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Get retriever
        retriever = self.vector_store_manager.get_retriever(
            k=config.get("rag.top_k", 4)
        )
        
        if retriever is None:
            app_logger.error("Failed to create retriever")
            return
        
        # Create RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        app_logger.info("QA chain created successfully")
    
    def add_documents(self, file_paths: List[str]) -> bool:
        """
        Add new documents to the system.
        
        Args:
            file_paths: List of file paths to add
            
        Returns:
            True if successful
        """
        app_logger.info(f"Adding {len(file_paths)} new documents...")
        
        all_documents = []
        
        for file_path in file_paths:
            documents = self.document_loader.load_single_file(file_path)
            all_documents.extend(documents)
        
        if not all_documents:
            app_logger.warning("No documents loaded")
            return False
        
        # Process documents
        chunks = self.text_processor.chunk_documents(all_documents)
        
        # Add to vector store
        success = self.vector_store_manager.add_documents(chunks)
        
        if success:
            self.vector_store_manager.save_vector_store()
            app_logger.info(f"Successfully added {len(chunks)} chunks to vector store")
        
        return success
    
    def query(
        self,
        question: str,
        regulation_type: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: Compliance question
            regulation_type: Optional regulation filter
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.qa_chain:
            app_logger.error("QA chain not initialized")
            return {
                "error": "System not initialized",
                "answer": "Please initialize the system first"
            }
        
        app_logger.info(f"Processing query: {question[:100]}")
        
        # Add regulation type to query if specified
        if regulation_type:
            question = f"[{regulation_type}] {question}"
        
        try:
            # Execute query
            result = self.qa_chain.invoke({"query": question})
            
            # Format response
            response = {
                "query": question,
                "answer": result["result"],
                "timestamp": datetime.now().isoformat(),
                "regulation_type": regulation_type
            }
            
            # Add sources if requested
            if return_sources and "source_documents" in result:
                sources = []
                for i, doc in enumerate(result["source_documents"]):
                    sources.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_rank": i + 1
                    })
                response["sources"] = sources
                response["num_sources"] = len(sources)
            
            # Add to history
            self.query_history.append(response)
            
            app_logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            app_logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "query": question,
                "answer": f"Error processing query: {str(e)}"
            }
    
    def search_similar_documents(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents without generating an answer.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of similar documents with scores
        """
        app_logger.info(f"Searching similar documents: {query[:100]}")
        
        try:
            docs_and_scores = self.vector_store_manager.similarity_search_with_score(
                query, k=k
            )
            
            results = []
            for doc, score in docs_and_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })
            
            return results
            
        except Exception as e:
            app_logger.error(f"Error searching documents: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system stats
        """
        vector_stats = self.vector_store_manager.get_stats()
        
        return {
            "llm_provider": self.llm_provider,
            "model_name": self.model_name,
            "num_queries": len(self.query_history),
            "vector_store": vector_stats,
            "documents_path": self.documents_path,
            "vector_db_path": self.vector_db_path
        }
    
    def export_query_history(self, filename: str) -> bool:
        """
        Export query history to file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful
        """
        import json
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.query_history, f, indent=2)
            
            app_logger.info(f"Query history exported to {filename}")
            return True
            
        except Exception as e:
            app_logger.error(f"Error exporting history: {e}")
            return False
        

if __name__ == "__main__":
    # Example usage
    rag_engine = RegulatoryComplianceRAG()
    stats = rag_engine.get_system_stats()
    print(stats)

    # Example query
    response = rag_engine.query("What are the key requirements of AML regulations?")
    print(response)