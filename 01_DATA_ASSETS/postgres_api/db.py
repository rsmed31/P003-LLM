"""Database connection pool and query helpers."""
import logging
from contextlib import contextmanager
from typing import List, Tuple, Any, Optional
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from config import settings

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


def query_qa_by_text_similarity(query_text: str, threshold: float = 0.75) -> Optional[dict]:
    """
    Query QA using text similarity (pg_trgm extension).
    Returns the best match if similarity >= threshold.
    """
    try:
        with get_db_cursor(dict_cursor=True) as cur:
            # Enable pg_trgm extension for text similarity
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            
            cur.execute("""
                SELECT id, question, answer, 
                       similarity(question, %s) AS score
                FROM qa
                WHERE similarity(question, %s) >= %s
                ORDER BY score DESC
                LIMIT 1
            """, (query_text, query_text, threshold))
            
            result = cur.fetchone()
            return dict(result) if result else None
    except Exception as e:
        logger.error(f"Error querying QA by text similarity: {e}")
        raise


def query_qa_by_vector_similarity(query_embedding: List[float], threshold: float = 0.75) -> Optional[dict]:
    """
    Query QA using vector similarity (pgvector).
    This assumes the qa table has an embedding column.
    Returns the best match if distance (converted to similarity) >= threshold.
    """
    try:
        with get_db_cursor(dict_cursor=True) as cur:
            # Using cosine distance, convert to similarity: 1 - distance
            cur.execute("""
                SELECT id, question, answer,
                       1 - (embedding <=> %s::vector) AS score
                FROM qa
                WHERE (1 - (embedding <=> %s::vector)) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT 1
            """, (query_embedding, query_embedding, threshold, query_embedding))
            
            result = cur.fetchone()
            return dict(result) if result else None
    except Exception as e:
        logger.error(f"Error querying QA by vector similarity: {e}")
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


def query_text_chunks_by_embedding(query_embedding: List[float], limit: int = 5) -> List[dict]:
    """
    Query text_chunks using vector similarity.
    Returns top N matches with source, chunk_index, text, and similarity score.
    """
    try:
        with get_db_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT source, chunk_index, text,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM text_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, query_embedding, limit))
            
            results = cur.fetchall()
            return [dict(row) for row in results] if results else []
    except Exception as e:
        logger.error(f"Error querying text chunks by embedding: {e}")
        raise
