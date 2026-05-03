"""
Main RAG Engine - Orchestrates all components.
This is the core class that brings everything together.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_classic.prompts import PromptTemplate
from langchain_classic.schema import Document

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
        """Create a manual QA chain with fine-grained control"""
        app_logger.info("Creating manual QA chain...")
        
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

        self.prompt_template = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Get retriever
        self.retriever = self.vector_store_manager.get_retriever(
            k=config.get("rag.top_k", 4)
        )
        
        if self.retriever is None:
            app_logger.error("Failed to create retriever")
            raise ValueError("Retriever initialization failed")
        
        # Mark chain as ready (we'll execute it manually in query method)
        self.qa_chain = True  # Flag indicating chain is ready
        
        app_logger.info("Manual QA chain created successfully")
    
    def _execute_manual_qa_chain(self, question: str) -> Dict[str, Any]:
        """
        Execute the QA chain manually for fine-grained control.
        
        Args:
            question: The compliance question
            
        Returns:
            Dictionary containing answer and context documents
        """
        app_logger.debug(f"Executing manual QA chain for: {question[:100]}")
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.vector_store_manager.similarity_search(
                query=question)
            
            if not retrieved_docs:
                app_logger.warning("No documents retrieved for query")
                return {
                    "answer": "No relevant documents found in the knowledge base for this query.",
                    "context_documents": [],
                    "num_sources": 0,
                    "execution_steps": ["retrieval"]
                }
            
            # Step 2: Format context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                doc_content = doc.page_content
                doc_source = doc.metadata.get("source", "Unknown source")
                context_parts.append(f"[Document {i} - {doc_source}]\n{doc_content}\n")
            
            formatted_context = "\n".join(context_parts)
            
            # Step 3: Format the prompt with context and question
            prompt_input = {
                "context": formatted_context,
                "question": question
            }
            
            formatted_prompt = self.prompt_template.format(**prompt_input)
            
            app_logger.debug(f"Formatted prompt length: {len(formatted_prompt)} characters")
            
            # Step 4: Call LLM to generate answer
            app_logger.debug("Calling LLM for answer generation...")
            llm_response = self.llm.invoke(formatted_prompt)
            
            # Extract text from response (handles both string and message objects)
            if hasattr(llm_response, 'content'):
                answer = llm_response.content
            else:
                answer = str(llm_response)
            
            # Step 5: Format source documents
            source_documents = []
            for i, doc in enumerate(retrieved_docs, 1):
                source_documents.append({
                    "document_id": i,
                    "content": doc.page_content[:500],  # Truncate for response
                    "metadata": doc.metadata,
                    "full_content": doc.page_content
                })
            
            result = {
                "answer": answer,
                "context_documents": source_documents,
                "num_sources": len(retrieved_docs),
                "execution_steps": ["retrieval", "context_formatting", "llm_call"],
                "raw_context": formatted_context[:1000]  # Include truncated context for debugging
            }
            
            app_logger.debug("Manual QA chain execution completed successfully")
            return result
            
        except Exception as e:
            app_logger.error(f"Error executing manual QA chain: {e}", exc_info=True)
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context_documents": [],
                "num_sources": 0,
                "error": str(e),
                "execution_steps": []
            }
    
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
        Query the RAG system using manual chain execution.
        
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
        full_question = question
        if regulation_type:
            full_question = f"[{regulation_type}] {question}"
        
        try:
            # Execute the manual QA chain
            chain_result = self._execute_manual_qa_chain(full_question)
            
            # Format response
            response = {
                "query": full_question,
                "answer": chain_result.get("answer", ""),
                "timestamp": datetime.now().isoformat(),
                "regulation_type": regulation_type,
                "execution_steps": chain_result.get("execution_steps", [])
            }
            
            # Add sources if requested
            if return_sources and chain_result.get("context_documents"):
                sources = []
                for i, doc in enumerate(chain_result["context_documents"], 1):
                    sources.append({
                        "rank": i,
                        "source": doc["metadata"].get("source", "Unknown"),
                        "content_preview": doc["content"],
                        "metadata": doc["metadata"]
                    })
                response["sources"] = sources
                response["num_sources"] = len(sources)
            
            # Add error if present
            if "error" in chain_result:
                response["error"] = chain_result["error"]
            
            # Add to history
            self.query_history.append(response)
            
            app_logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            app_logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "query": full_question,
                "answer": f"Error processing query: {str(e)}"
            }
    
    def _get_source_documents(
        self,
        query: str,
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieve source documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of source documents with metadata
        """
        try:
            docs_and_scores = self.vector_store_manager.similarity_search_with_score(
                query, k=k
            )
            
            sources = []
            for i, (doc, score) in enumerate(docs_and_scores):
                sources.append({
                    "content": doc.page_content[:500],  # Truncate for response
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                    "relevance_rank": i + 1
                })
            
            return sources
            
        except Exception as e:
            app_logger.debug(f"Error retrieving source documents: {e}")
            return []
    
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
        
        return self._get_source_documents(query, k=k)
    
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