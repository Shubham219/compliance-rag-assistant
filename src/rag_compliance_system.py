"""
Regulatory Compliance Assistant - RAG Application
================================================

This system helps organizations:
- Query regulatory documents
- Get compliance requirements and interpretations
- Track regulatory changes and updates

RAG Components Explained:
1. Document Loading: Import and parse regulatory documents
2. Text Chunking: Split documents into manageable pieces
3. Embeddings: Convert text to numerical vectors
4. Vector Store: Store and search embeddings efficiently
5. Retrieval: Find relevant context for queries
6. Generation: Use LLM to generate answers with retrieved context
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Core RAG libraries
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


@dataclass
class ComplianceQuery:
    """Data class to structure compliance queries and responses"""
    query: str
    regulation_type: Optional[str] = None  # e.g., "GDPR", "SOX", "HIPAA"
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class RegulatoryComplianceRAG:
    """
    Main RAG system for Regulatory Compliance
    
    This class orchestrates all components of the RAG pipeline:
    - Loading regulatory documents
    - Creating embeddings
    - Storing in vector database
    - Retrieving relevant context
    - Generating compliance-specific answers
    """
    
    def __init__(
        self,
        documents_path: str = "./regulatory_documents",
        vector_db_path: str = "./vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "ollama",  # Options: "openai", "ollama"
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1  # Low temperature for factual compliance answers
    ):
        """
        Initialize the RAG system
        
        Args:
            documents_path: Path to folder containing regulatory documents
            vector_db_path: Path to store/load the vector database
            embedding_model: HuggingFace model for embeddings
            llm_provider: Which LLM to use (OpenAI or local Ollama)
            model_name: Specific model name
            temperature: Controls randomness (lower = more deterministic)
        """
        self.documents_path = documents_path
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize components (will be set up later)
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        
        print("🚀 Initializing Regulatory Compliance Assistant...")
        self._setup_components()
    
    def _setup_components(self):
        """Set up all RAG components in sequence"""
        self._initialize_embeddings()
        self._initialize_llm()
        self._load_or_create_vector_store()
        self._create_qa_chain()
        print("✅ System ready for compliance queries!\n")
    
    def _initialize_embeddings(self):
        """
        Step 1: Initialize the embedding model
        
        Embeddings convert text into numerical vectors (arrays of numbers).
        Similar texts have similar vectors, enabling semantic search.

        """
        print("📊 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' for GPU
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
        print(f"   ✓ Loaded: {self.embedding_model_name}")
    
    def _initialize_llm(self):
        """
        Step 2: Initialize the Large Language Model
        
        The LLM generates human-readable answers based on retrieved context.
        We use low temperature for factual, consistent compliance answers.
        """
        print("🤖 Initializing Language Model...")
        
        if self.llm_provider == "openai":
            # OpenAI GPT models (requires API key)
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            print(f"   ✓ Using OpenAI: {self.model_name}")
        
        elif self.llm_provider == "ollama":
            # Local models via Ollama (free, runs on your machine)
            self.llm = Ollama(
                model=self.model_name,
                temperature=self.temperature
            )
            print(f"   ✓ Using Ollama: {self.model_name}")
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def load_documents(self, force_reload: bool = False) -> List[Document]:
        """
        Step 3: Load regulatory documents from disk
        
        Document loaders handle different file formats (PDF, TXT, DOCX, etc.)
        and convert them into a standard Document format with text and metadata.
        
        Args:
            force_reload: If True, reload even if vector store exists
            
        Returns:
            List of Document objects
        """
        print("📄 Loading regulatory documents...")
        
        # Check if documents directory exists
        if not os.path.exists(self.documents_path):
            print(f"   ⚠️  Creating documents directory: {self.documents_path}")
            os.makedirs(self.documents_path)
            print("   ⚠️  Please add regulatory documents to this folder!")
            return []
        
        documents = []
        
        # Load PDF documents
        try:
            pdf_loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            print(f"   ✓ Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            print(f"   ⚠️  Error loading PDFs: {e}")
        
        # Load text documents
        try:
            txt_loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            print(f"   ✓ Loaded {len(txt_docs)} text documents")
        except Exception as e:
            print(f"   ⚠️  Error loading text files: {e}")
        
        print(f"   📚 Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Step 4: Split documents into smaller chunks
        
        RecursiveCharacterTextSplitter:
        - Tries to keep paragraphs together
        - Falls back to sentences, then words if needed
        - Maintains context with overlap between chunks
        
        Args:
            documents: List of full documents
            
        Returns:
            List of document chunks
        """
        print("✂️  Chunking documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # Characters per chunk
            chunk_overlap=200,      # Overlap to maintain context across chunks
            length_function=len,    # How to measure chunk size
            separators=["\n\n", "\n", ". ", " ", ""]  # Split hierarchy
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"   ✓ Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
    
    def _load_or_create_vector_store(self):
        """
        Step 5: Create or load the vector database
        
        Vector Store (FAISS in this case)
        """
        # Check if vector store already exists
        if os.path.exists(self.vector_db_path) and os.path.exists(f"{self.vector_db_path}/index.faiss"):
            print("💾 Loading existing vector store...")
            self.vector_store = FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # Required for FAISS
            )
            print(f"   ✓ Vector store loaded from {self.vector_db_path}")
        else:
            print("🔨 Creating new vector store...")
            documents = self.load_documents()
            
            if not documents:
                print("   ⚠️  No documents found. Creating empty vector store.")
                # Create a dummy document to initialize the store
                documents = [Document(page_content="Placeholder document", metadata={})]
            
            chunks = self.chunk_documents(documents)
            
            # Create embeddings and store in FAISS
            print("   🔄 Creating embeddings (this may take a while)...")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            # Save for future use
            self.vector_store.save_local(self.vector_db_path)
            print(f"   ✓ Vector store created and saved to {self.vector_db_path}")
    
    def add_documents(self, new_documents: List[Document]):
        """
        Add new regulatory documents to existing vector store
        
        This allows incremental updates without rebuilding entire database.
        Useful when new regulations are published or updated.
        
        Args:
            new_documents: List of new Document objects to add
        """
        print(f"➕ Adding {len(new_documents)} new documents...")
        
        chunks = self.chunk_documents(new_documents)
        
        # Add to existing vector store
        self.vector_store.add_documents(chunks)
        
        # Save updated store
        self.vector_store.save_local(self.vector_db_path)
        print("   ✓ Documents added and vector store updated")
    
    def _create_qa_chain(self):
        """
        Step 6: Create the Question-Answering Chain
        
        The QA chain orchestrates the entire RAG process:
        1. Takes user query
        2. Retrieves relevant document chunks
        3. Constructs prompt with context
        4. Generates answer using LLM
        
        This is where retrieval and generation come together!
        """
        print("🔗 Creating QA chain...")
        
        # Create a custom prompt template for compliance queries
        # The template guides the LLM on how to use retrieved context
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
        
        # Create the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" means put all context into one prompt
            retriever=self.vector_store.as_retriever(
                search_type="similarity",  # Use semantic similarity search
                search_kwargs={"k": 4}     # Retrieve top 4 most relevant chunks
            ),
            return_source_documents=True,  # Return sources for transparency
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("   ✓ QA chain ready")
    
    def query(
        self,
        question: str,
        regulation_type: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Main query method - the heart of the RAG system
        
        This method:
        1. Takes a natural language question
        2. Retrieves relevant regulatory context
        3. Generates a compliance-specific answer
        4. Returns answer with sources for verification
        
        Args:
            question: Natural language compliance question
            regulation_type: Optional filter for specific regulation
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        print(f"\n🔍 Processing query: {question}")
        
        # Optionally filter by regulation type
        if regulation_type:
            question = f"[{regulation_type}] {question}"
        
        # Execute the RAG pipeline
        result = self.qa_chain.invoke({"query": question})
        
        # Format response
        response = {
            "query": question,
            "answer": result["result"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add source documents if requested
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
        
        return response
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple compliance queries in batch
        
        Useful for compliance audits or bulk policy questions.
        
        Args:
            questions: List of questions to process
            
        Returns:
            List of response dictionaries
        """
        print(f"\n📋 Processing {len(questions)} queries in batch...")
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"   Query {i}/{len(questions)}")
            result = self.query(question, return_sources=True)
            results.append(result)
        
        print("   ✅ Batch processing complete")
        return results
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents without LLM generation
        
        Useful for exploring what the system knows about a topic
        without generating an answer.
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of similar documents with scores
        """
        print(f"\n🔎 Finding similar documents for: {query}")
        
        # Perform similarity search with scores
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            })
        
        return results
    
    def save_query_log(self, queries: List[Dict[str, Any]], filename: str = "query_log.json"):
        """
        Save query history for audit trail
        
        Important for compliance: maintain records of what was queried and when.
        
        Args:
            queries: List of query results to save
            filename: Output file path
        """
        with open(filename, 'w') as f:
            json.dump(queries, f, indent=2)
        print(f"💾 Query log saved to {filename}")


def main():
    """
    Main execution function with example usage
    
    This demonstrates how to use the Regulatory Compliance RAG system.
    """
    print("="*70)
    print("REGULATORY COMPLIANCE ASSISTANT - RAG SYSTEM")
    print("="*70)
    
    # Step 2: Initialize the RAG system
    print("\n" + "="*70)
    rag_system = RegulatoryComplianceRAG(
        documents_path="./regulatory_documents",
        vector_db_path="./vector_db",
        llm_provider="ollama",  # Change to "openai" if using OpenAI
        model_name="llama2",    # Change to "gpt-3.5-turbo" for OpenAI
        temperature=0.1
    )
    
    # Step 3: Example queries
    print("\n" + "="*70)
    print("EXAMPLE COMPLIANCE QUERIES")
    print("="*70)
    
    example_queries = [
        "Write algorithm in plain text to detect money laundering as per FATF guidelines."]
    
    results = []
    for query in example_queries:
        result = rag_system.query(query, return_sources=True)
        results.append(result)
        
        print(f"\n{'='*70}")
        print(f"Q: {result['query']}")
        print(f"{'='*70}")
        print(f"A: {result['answer']}")
        
        if 'sources' in result:
            print(f"\n📚 Sources used: {result['num_sources']}")
            for i, source in enumerate(result['sources'][:2], 1):  # Show top 2 sources
                print(f"\nSource {i}:")
                print(f"  {source['content'][:200]}...")
    
    # Step 4: Save query log for audit trail
    print("\n" + "="*70)
    rag_system.save_query_log(results, "compliance_query_log.json")
    
    # Step 5: Demonstrate similarity search
    print("\n" + "="*70)
    print("SIMILARITY SEARCH EXAMPLE")
    print("="*70)
    similar_docs = rag_system.get_similar_documents("data encryption requirements", k=3)
    print(f"\nFound {len(similar_docs)} similar documents:")
    for i, doc in enumerate(similar_docs, 1):
        print(f"\n{i}. Similarity Score: {doc['similarity_score']:.4f}")
        print(f"   Content: {doc['content'][:150]}...")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
