"""
Script to rebuild the vector database from documents.
"""

from src.core.rag_engine import RegulatoryComplianceRAG
from src.utils.logger import app_logger


def rebuild_vector_database():
    """Rebuild the vector database"""
    
    app_logger.info("Starting vector database rebuild...")
    
    try:
        # Initialize without auto-loading
        rag = RegulatoryComplianceRAG(auto_load=False)
        
        # Setup vector store (will create new one)
        rag._setup_vector_store()
        
        # Create QA chain
        rag._create_qa_chain()
        
        app_logger.info("✅ Vector database rebuilt successfully")
        
        # Print stats
        stats = rag.get_system_stats()
        print(f"\nVector Store Stats:")
        print(f"  Vectors: {stats['vector_store'].get('num_vectors', 0)}")
        print(f"  Path: {stats['vector_db_path']}")
        
    except Exception as e:
        app_logger.error(f"Failed to rebuild vector database: {e}")
        raise


if __name__ == "__main__":
    rebuild_vector_database()
