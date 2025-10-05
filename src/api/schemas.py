"""
Pydantic schemas for API request/response validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for compliance queries"""
    question: str = Field(..., description="Compliance question to ask")
    regulation_type: Optional[str] = Field(None, description="Filter by regulation type")
    return_sources: bool = Field(True, description="Include source documents")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are GDPR data retention requirements?",
                "regulation_type": "GDPR",
                "return_sources": True
            }
        }


class QueryResponse(BaseModel):
    """Response schema for compliance queries"""
    query: str
    answer: str
    timestamp: str
    regulation_type: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    num_sources: Optional[int] = None


class SearchRequest(BaseModel):
    """Request schema for similarity search"""
    query: str = Field(..., description="Search query")
    k: int = Field(5, ge=1, le=20, description="Number of results")


class SearchResponse(BaseModel):
    """Response schema for similarity search"""
    query: str
    results: List[Dict[str, Any]]
    num_results: int


class DocumentUploadResponse(BaseModel):
    """Response schema for document uploads"""
    success: bool
    message: str
    num_documents: int


class SystemStatsResponse(BaseModel):
    """Response schema for system statistics"""
    llm_provider: str
    model_name: str
    num_queries: int
    vector_store: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None