"""Pydantic models for request/response validation."""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class QAResponse(BaseModel):
    """Response model for QA records."""
    id: int
    question: str
    answer: str
    lastUpdated: Optional[datetime] = None
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score")
    
    class Config:
        from_attributes = True


class QAQueryResponse(BaseModel):
    """Response model for QA query endpoint."""
    found: bool
    answer: Optional[str] = None
    message: Optional[str] = None


class QACreateRequest(BaseModel):
    """Request model for creating a QA record."""
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=2000, 
        description="The question text",
        examples=["What is OSPF?"]
    )
    answer: str = Field(
        ..., 
        min_length=1, 
        max_length=10000, 
        description="The answer text",
        examples=["OSPF (Open Shortest Path First) is a routing protocol..."]
    )
    
    @field_validator('question', 'answer')
    @classmethod
    def validate_text_not_empty(cls, v):
        """Validate that text fields are not just whitespace."""
        if v and not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class QACreateResponse(BaseModel):
    """Response model for creating a QA record."""
    message: str = Field(..., description="Response message describing the operation result")
    status: str = Field(
        ..., 
        description="Status of the operation",
        examples=["created", "error"]
    )
    question: Optional[str] = Field(None, description="Echo of the question that was created")
    answer: Optional[str] = Field(None, description="Echo of the answer that was created")
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "message": "QA record created successfully",
                    "status": "created",
                    "question": "What is OSPF?",
                    "answer": "OSPF (Open Shortest Path First) is a routing protocol..."
                }
            ]
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


class TextChunkResponse(BaseModel):
    """Response model for text chunk records."""
    chunk_index: int
    text: str
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    
    class Config:
        from_attributes = True


class TextChunksQueryResponse(BaseModel):
    """Response model for text chunks query endpoint."""
    found: bool
    results: List[TextChunkResponse] = []
    count: int
    message: Optional[str] = None


class TextChunksQueryRequest(BaseModel):
    """Request model for querying text chunks."""
    query: str = Field(..., min_length=1, max_length=2000, description="Query text to search for")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return")

