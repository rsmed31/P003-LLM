"""
PostgreSQL Client with pgvector for semantic search on verified facts.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
import numpy as np


class PostgresClient:
    """
    Client for querying PostgreSQL verified facts with semantic similarity.
    """
    
    def __init__(self, connection_string: str = None):
        """
        Initialize PostgreSQL client.
        
        Args:
            connection_string: PostgreSQL connection string
                              Format: "postgresql://user:pass@host:port/dbname"
        """
        self.connection_string = connection_string or os.getenv(
            "POSTGRES_CONNECTION_STRING",
            "postgresql://postgres:password@localhost:5432/llm_kb"
        )
        self.conn = None
    
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            return True
        except Exception as e:
            print(f"PostgreSQL connection failed: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def semantic_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 1,
        table: str = "verified_facts"
    ) -> List[Dict]:
        """
        Perform semantic search using pgvector cosine similarity.
        
        Args:
            query: Original query text (for logging)
            query_embedding: Query vector from embedding model
            top_k: Number of results to return
            table: Table name to search
        
        Returns:
            List of dicts with 'text' and 'similarity' keys
        """
        if not self.conn:
            if not self.connect():
                return []
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Query using pgvector's <=> operator for cosine distance
                # Convert distance to similarity: similarity = 1 - distance
                sql = f"""
                    SELECT 
                        content as text,
                        metadata,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM {table}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                
                # Convert numpy array to list for PostgreSQL
                embedding_list = query_embedding.tolist()
                
                cur.execute(sql, (embedding_list, embedding_list, top_k))
                results = cur.fetchall()
                
                return [dict(row) for row in results]
        
        except Exception as e:
            print(f"PostgreSQL semantic search failed: {e}")
            return []
    
    def get_by_id(self, fact_id: int, table: str = "verified_facts") -> Optional[Dict]:
        """
        Retrieve a specific fact by ID.
        
        Args:
            fact_id: Fact ID
            table: Table name
        
        Returns:
            Dict with fact data or None
        """
        if not self.conn:
            if not self.connect():
                return None
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = f"SELECT * FROM {table} WHERE id = %s"
                cur.execute(sql, (fact_id,))
                result = cur.fetchone()
                return dict(result) if result else None
        
        except Exception as e:
            print(f"PostgreSQL get_by_id failed: {e}")
            return None
