"""
FastAPI REST API endpoints for Compliance RAG system.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import os

from ..core.rag_engine import RegulatoryComplianceRAG
from ..utils.logger import app_logger
from ..utils.config import config
from .schemas import (
    QueryRequest, QueryResponse,
    SearchRequest, SearchResponse,
    DocumentUploadResponse, SystemStatsResponse,
    ErrorResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Compliance RAG Assistant API",
    description="REST API for Regulatory Compliance Assistant using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_system: RegulatoryComplianceRAG = None


def get_rag_system() -> RegulatoryComplianceRAG:
    """Dependency to get RAG system instance"""
    global rag_system
    
    if rag_system is None:
        app_logger.info("Initializing RAG system...")
        rag_system = RegulatoryComplianceRAG(auto_load=True)
    
    return rag_system


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    app_logger.info("Starting Compliance RAG API...")
    global rag_system
    
    try:
        rag_system = RegulatoryComplianceRAG(auto_load=True)
        app_logger.info("RAG system initialized successfully")
    except Exception as e:
        app_logger.error(f"Failed to initialize RAG system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    app_logger.info("Shutting down Compliance RAG API...")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Compliance RAG Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None
    }


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_compliance(
    request: QueryRequest,
    rag: RegulatoryComplianceRAG = Depends(get_rag_system)
):
    """
    Query the compliance RAG system.
    
    Args:
        request: Query request with question and options
        
    Returns:
        Query response with answer and sources
    """
    try:
        app_logger.info(f"API Query: {request.question[:100]}")
        
        result = rag.query(
            question=request.question,
            regulation_type=request.regulation_type,
            return_sources=request.return_sources
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return QueryResponse(**result)
        
    except Exception as e:
        app_logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    rag: RegulatoryComplianceRAG = Depends(get_rag_system)
):
    """
    Search for similar documents.
    
    Args:
        request: Search request with query and limit
        
    Returns:
        List of similar documents
    """
    try:
        app_logger.info(f"API Search: {request.query[:100]}")
        
        results = rag.search_similar_documents(
            query=request.query,
            k=request.k
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            num_results=len(results)
        )
        
    except Exception as e:
        app_logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    rag: RegulatoryComplianceRAG = Depends(get_rag_system)
):
    """
    Upload and process regulatory documents.
    
    Args:
        files: List of files to upload
        
    Returns:
        Upload status and count
    """
    try:
        app_logger.info(f"API Upload: {len(files)} files")
        
        # Save files temporarily
        temp_paths = []
        for file in files:
            suffix = os.path.splitext(file.filename)[1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_paths.append(temp_file.name)
        
        # Add to RAG system
        success = rag.add_documents(temp_paths)
        
        # Cleanup temp files
        for path in temp_paths:
            try:
                os.unlink(path)
            except:
                pass
        
        if success:
            return DocumentUploadResponse(
                success=True,
                message="Documents processed successfully",
                num_documents=len(files)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to process documents")
            
    except Exception as e:
        app_logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    rag: RegulatoryComplianceRAG = Depends(get_rag_system)
):
    """
    Get system statistics.
    
    Returns:
        System statistics
    """
    try:
        stats = rag.get_system_stats()
        return SystemStatsResponse(**stats)
        
    except Exception as e:
        app_logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/history")
async def get_query_history(
    limit: int = 10,
    rag: RegulatoryComplianceRAG = Depends(get_rag_system)
):
    """
    Get query history.
    
    Args:
        limit: Maximum number of queries to return
        
    Returns:
        List of recent queries
    """
    try:
        history = rag.query_history[-limit:]
        return {
            "queries": history,
            "total": len(rag.query_history),
            "returned": len(history)
        }
        
    except Exception as e:
        app_logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the API server"""
    import uvicorn
    
    uvicorn.run(
        "src.api.routes:app",
        host=config.get("api.host", "0.0.0.0"),
        port=config.get("api.port", 8000),
        reload=config.get("app_env") == "development",
        log_level=config.get("logging.level", "info").lower()
    )


if __name__ == "__main__":
    main()