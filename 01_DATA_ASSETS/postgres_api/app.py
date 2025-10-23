"""FastAPI application for Postgres QA API."""
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Path, status
from fastapi.responses import JSONResponse

from config import settings
from db import (
    DatabasePool,
    query_qa_by_id,
    qaembedding,
    readfromqa,
    readfromdoc
)
from models import (
    QAResponse,
    QAQueryResponse,
    QACreateRequest,
    QACreateResponse,
    ErrorResponse,
    TextChunkResponse,
    TextChunksQueryResponse,
    TextChunksQueryRequest
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting up Postgres QA API...")
    try:
        DatabasePool.initialize(minconn=2, maxconn=10)
        logger.info("Database pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Postgres QA API...")
    DatabasePool.close_all()
    logger.info("Database pool closed")


app = FastAPI(
    title="Postgres QA API",
    description="API for querying and managing QA records with similarity search",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Postgres QA API",
        "version": "1.0.0"
    }


@app.get(
    "/qa/query",
    response_model=QAQueryResponse,
    responses={
        200: {"description": "Query successful"},
        400: {"model": ErrorResponse, "description": "Invalid query parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["QA Query"]
)
async def query_qa(
    text: str = Query(..., min_length=1, max_length=2000, description="Query text to search for"),
    threshold: float = Query(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0 to 1.0)"
    )
):
    """
    Query QA records using vector similarity search.
    
    - **text**: The query text to search for (required)
    - **threshold**: Minimum similarity score (default: 0.8)
    
    Returns the best matching QA answer if similarity >= threshold.
    """
    try:
        logger.info(f"Querying QA with text: '{text[:50]}...' (threshold: {threshold})")
        
        answer = readfromqa(text, threshold)
        
        if answer:
            logger.info(f"Found QA match")
            return QAQueryResponse(
                found=True,
                answer=answer,
                message="Match found"
            )
        else:
            logger.info("No match found above threshold")
            return QAQueryResponse(
                found=False,
                answer=None,
                message=f"No match found with similarity >= {threshold}"
            )
    
    except Exception as e:
        logger.error(f"Error querying QA: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failed: {str(e)}"
        )


@app.get(
    "/qa/{id}",
    response_model=QAResponse,
    responses={
        200: {"description": "QA record found"},
        404: {"model": ErrorResponse, "description": "QA record not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["QA Query"]
)
async def get_qa_by_id(
    id: int = Path(..., gt=0, description="QA record ID")
):
    """
    Get a QA record by ID.
    
    - **id**: The ID of the QA record to retrieve
    
    Returns the QA record or 404 if not found.
    """
    try:
        logger.info(f"Fetching QA record with ID: {id}")
        
        result = query_qa_by_id(id)
        
        if result:
            logger.info(f"Found QA record {id}")
            return QAResponse(**result)
        else:
            logger.warning(f"QA record {id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"QA record with ID {id} not found"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching QA record {id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failed: {str(e)}"
        )


@app.post(
    "/qa",
    response_model=QACreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "QA record created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid payload"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["QA Management"]
)
async def create_qa(payload: QACreateRequest):
    """
    Create a new QA record with embedding.
    
    This endpoint creates a new QA record in the database with the provided
    question and answer. The embedding is automatically generated.
    
    - **question**: The question text (required)
    - **answer**: The answer text (required)
    
    Returns success status.
    """
    try:
        logger.info(f"Received QA create request: question='{payload.question[:50]}...'")
        
        # Insert QA record with automatic embedding generation
        success = qaembedding(payload.question, payload.answer)
        
        if success:
            logger.info(f"QA record created successfully")
            
            return QACreateResponse(
                message="QA record created successfully",
                status="created",
                question=payload.question,
                answer=payload.answer
            )
        else:
            raise Exception("Failed to create QA record")
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating QA record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create QA record: {str(e)}"
        )


@app.post(
    "/chunks/query",
    response_model=TextChunksQueryResponse,
    responses={
        200: {"description": "Query successful"},
        400: {"model": ErrorResponse, "description": "Invalid query parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Text Chunks"]
)
async def query_text_chunks(payload: TextChunksQueryRequest):
    """
    Query text chunks using semantic similarity search.
    
    Uses sentence-transformers (all-MiniLM-L6-v2) to encode the query text
    and searches for similar chunks using vector similarity.
    
    - **query**: The query text to search for (required)
    - **limit**: Maximum number of results to return (default: 5, max: 50)
    
    Returns a list of matching text chunks with chunk_index, text, and similarity score.
    """
    try:
        logger.info(f"Querying text chunks with: '{payload.query[:50]}...' (limit: {payload.limit})")
        
        # Query database using the new function
        results = readfromdoc(payload.query, payload.limit)
        
        if results:
            logger.info(f"Found {len(results)} matching chunks")
            return TextChunksQueryResponse(
                found=True,
                results=[TextChunkResponse(**r) for r in results],
                count=len(results),
                message=f"Found {len(results)} matching chunks"
            )
        else:
            logger.info("No chunks found")
            return TextChunksQueryResponse(
                found=False,
                results=[],
                count=0,
                message="No matching chunks found"
            )
    
    except Exception as e:
        logger.error(f"Error querying text chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query text chunks: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )
