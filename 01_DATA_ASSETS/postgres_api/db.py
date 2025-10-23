"""Database connection pool and query helpers."""
import logging
from contextlib import contextmanager
from typing import List, Tuple, Any, Optional
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from config import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DatabasePool:
    """Manages PostgreSQL connection pool."""
    
    _pool: Optional[pool.SimpleConnectionPool] = None
    
    @classmethod
    def initialize(cls, minconn: int = 1, maxconn: int = 10):
        """Initialize the connection pool."""
        if cls._pool is None:
            try:
                cls._pool = psycopg2.pool.SimpleConnectionPool(
                    minconn,
                    maxconn,
                    host=settings.DB_HOST,
                    port=settings.DB_PORT,
                    database=settings.DB_NAME,
                    user=settings.DB_USER,
                    password=settings.DB_PASSWORD
                )
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise
    
    @classmethod
    def get_connection(cls):
        """Get a connection from the pool."""
        if cls._pool is None:
            cls.initialize()
        return cls._pool.getconn()
    
    @classmethod
    def return_connection(cls, conn):
        """Return a connection to the pool."""
        if cls._pool:
            cls._pool.putconn(conn)
    
    @classmethod
    def close_all(cls):
        """Close all connections in the pool."""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            logger.info("Database connection pool closed")


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = None
    try:
        conn = DatabasePool.get_connection()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            DatabasePool.return_connection(conn)

model = SentenceTransformer("all-MiniLM-L6-v2")

@contextmanager
def get_db_cursor(dict_cursor: bool = False):
    """Context manager for database cursor."""
    with get_db_connection() as conn:
        cursor_factory = RealDictCursor if dict_cursor else None
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        finally:
            cursor.close()


def query_qa_by_id(qa_id: int) -> Optional[dict]:
    """Query QA record by ID."""
    try:
        with get_db_cursor(dict_cursor=True) as cur:
            cur.execute(
                "SELECT id, question, answer, lastUpdated FROM qa WHERE id = %s",
                (qa_id,)
            )
            result = cur.fetchone()
            return dict(result) if result else None
    except Exception as e:
        logger.error(f"Error querying QA by ID {qa_id}: {e}")
        raise


def insert_qa_record(question: str, answer: str, embedding: Optional[List[float]] = None) -> int:
    """
    Insert a new QA record (stub implementation).
    Returns the ID of the inserted record.
    """
    try:
        with get_db_cursor() as cur:
            if embedding:
                cur.execute("""
                    INSERT INTO qa (question, answer, embedding)
                    VALUES (%s, %s, %s::vector)
                    RETURNING id
                """, (question, answer, embedding))
            else:
                cur.execute("""
                    INSERT INTO qa (question, answer)
                    VALUES (%s, %s)
                    RETURNING id
                """, (question, answer))
            
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        logger.error(f"Error inserting QA record: {e}")
        raise


# ============================================================================
# Enhanced functions based on user requirements
# ============================================================================

def qaembedding(question: str, answer: str) -> bool:
    """
    Insert a question-answer pair with its embedding into the qa table.
    
    Args:
        question: The question text
        answer: The answer text
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate embedding from the question
        embedding = model.encode(question).tolist()
        
        with get_db_cursor() as cur:
            cur.execute("""
                INSERT INTO qa (question, answer, embedding)
                VALUES (%s, %s, %s::vector)
            """, (question, answer, embedding))
        
        logger.info(f"QA successfully saved to PostgreSQL: '{question[:50]}...'")
        return True
        
    except Exception as e:
        logger.error(f"Error saving QA to database: {e}")
        raise


def readfromqa(query: str, threshold: float = 0.8) -> Optional[str]:
    """
    Search for a similar question in the qa table and return the answer if similarity >= threshold.
    
    Args:
        query: The query text to search for
        threshold: Minimum similarity score (default: 0.8)
        
    Returns:
        The answer string if a match is found with similarity >= threshold, None otherwise
    """
    try:
        # Generate query embedding
        query_emb = model.encode(query).tolist()
        
        with get_db_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT question, answer,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM qa
                ORDER BY embedding <=> %s::vector
                LIMIT 1;
            """, (query_emb, query_emb))
            
            result = cur.fetchone()
            
            if result and result['similarity'] >= threshold:
                logger.info(f"QA match found with similarity {result['similarity']:.3f}")
                return result['answer']
            else:
                if result:
                    logger.info(f"QA match found but similarity too low: {result['similarity']:.3f} < {threshold}")
                else:
                    logger.info("No QA match found")
                return None
                
    except Exception as e:
        logger.error(f"Error querying QA by similarity: {e}")
        raise


def readfromdoc(query: str, amount: int = 5) -> List[dict]:
    """
    Search for similar text chunks in the text_chunks table.
    
    Args:
        query: The query text to search for
        amount: Number of results to return (default: 5)
        
    Returns:
        List of dictionaries containing chunk_index, text, and similarity score
    """
    try:
        # Generate query embedding
        query_emb = model.encode(query).tolist()
        
        with get_db_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT chunk_index, text,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM text_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_emb, query_emb, amount))
            
            results = cur.fetchall()
            
            logger.info(f"Found {len(results)} document chunks for query")
            return [dict(row) for row in results] if results else []
            
    except Exception as e:
        logger.error(f"Error querying document chunks: {e}")
        raise
