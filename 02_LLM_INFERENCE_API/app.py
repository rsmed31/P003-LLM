"""
FastAPI application for LLM Inference API with RAG.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import sys
import os

# Add endpoints to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'endpoints'))

from endpoints.inference import generate
from retrieval.retrieval_orchestrator import RetrievalOrchestrator
from retrieval.postgres_client import PostgresClient
from retrieval.faiss_client import FAISSClient
from retrieval.embedding_service import EmbeddingService

# Initialize FastAPI app
app = FastAPI(
    title="LLM Inference API with RAG",
    description="Unified inference interface with retrieval triage",
    version="1.0.0"
)

# Initialize retrieval orchestrator with real clients
postgres_client = PostgresClient()
faiss_client = FAISSClient()
embedding_service = EmbeddingService()

RETRIEVAL_ORCHESTRATOR = RetrievalOrchestrator(
    postgres_client=postgres_client,
    faiss_client=faiss_client,
    embedding_service=embedding_service
)


class GenerateRequest(BaseModel):
    """Request model for /generate_config endpoint."""
    prompt: str
    model_name: str = "llama"
    use_triage: bool = True
    use_code_filter: bool = True


class GenerateResponse(BaseModel):
    """Response model for /generate_config endpoint."""
    model: str
    response: str
    route: Optional[str] = None
    confidence: Optional[float] = None


@app.post("/generate_config", response_model=GenerateResponse)
async def generate_config(request: GenerateRequest):
    """
    Main endpoint for LLM inference with RAG.
    
    Args:
        request: GenerateRequest with prompt and configuration
    
    Returns:
        GenerateResponse with model output
    """
    try:
        # Call generate function from inference.py
        result_json = generate(
            model_name=request.model_name,
            prompt=request.prompt,
            use_triage=request.use_triage,
            use_code_filter=request.use_code_filter
        )
        
        # Parse JSON response
        result = json.loads(result_json)
        
        return GenerateResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "postgres_connected": postgres_client.conn is not None,
        "faiss_loaded": faiss_client.index is not None,
        "embedding_loaded": embedding_service.model is not None
    }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("Initializing services...")
    
    # Load embedding model
    embedding_service.load_model()
    print("✓ Embedding service loaded")
    
    # Connect to PostgreSQL
    if postgres_client.connect():
        print("✓ PostgreSQL connected")
    else:
        print("✗ PostgreSQL connection failed")
    
    # Load FAISS index
    if faiss_client.load_index():
        print("✓ FAISS index loaded")
    else:
        print("✗ FAISS index loading failed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if postgres_client:
        postgres_client.close()
    print("Services shut down")
