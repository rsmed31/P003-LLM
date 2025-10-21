"""
LLM-007: Multi-Source Retrieval Orchestration with Triage Decision Gate

This module implements the retrieval triage logic that decides whether to:
1. Return a definitive answer directly from PostgreSQL (high confidence)
2. Proceed to full RAG with FAISS procedural context (lower confidence)
"""

from typing import Dict, Optional
from .postgres_client import PostgresClient
from .faiss_client import FAISSClient
from .embedding_service import EmbeddingService


class RetrievalOrchestrator:
    """
    Orchestrates retrieval from multiple sources and makes triage decisions.
    """
    
    DEFINITIVE_THRESHOLD = 0.98  # High confidence threshold for definitive answers
    
    def __init__(
        self,
        postgres_client: PostgresClient = None,
        faiss_client: FAISSClient = None,
        embedding_service: EmbeddingService = None
    ):
        """
        Initialize orchestrator with retrieval clients.
        
        Args:
            postgres_client: Client for PostgreSQL verified facts retrieval
            faiss_client: Client for FAISS procedural context retrieval
            embedding_service: Service for generating query embeddings
        """
        self.postgres_client = postgres_client or PostgresClient()
        self.faiss_client = faiss_client or FAISSClient()
        self.embedding_service = embedding_service or EmbeddingService()
    
    def retrieve_with_triage(
        self,
        user_query: str,
        enable_faiss: bool = True
    ) -> Dict[str, any]:
        """
        Execute retrieval triage decision gate.
        
        Decision Flow:
        1. Query PostgreSQL for verified facts with similarity scoring
        2. IF similarity > DEFINITIVE_THRESHOLD:
           - RETURN definitive answer immediately (fast path)
        3. ELSE:
           - Proceed to FAISS for procedural context (full RAG path)
        
        Args:
            user_query: User's natural language query
            enable_faiss: Whether to allow FAISS retrieval (for testing)
        
        Returns:
            Dict with keys:
                - route: "definitive" or "full_rag"
                - factual_data: str (definitive facts or supporting facts)
                - filtered_context: str (empty for definitive, populated for full RAG)
                - confidence: float (similarity score from postgres)
        """
        
        # Generate query embedding
        query_embedding = self.embedding_service.encode(user_query)
        
        # Check 1: Query PostgreSQL for verified facts
        postgres_result = self._query_postgres_facts(user_query, query_embedding)
        
        if postgres_result is None:
            # No postgres connection or no results
            return self._fallback_to_faiss(user_query, query_embedding, enable_faiss)
        
        confidence = postgres_result.get("confidence", 0.0)
        factual_data = postgres_result.get("content", "")
        
        # Decision Gate: Check confidence threshold
        if confidence >= self.DEFINITIVE_THRESHOLD:
            # HIGH CONFIDENCE: Return definitive answer (skip LLM generation)
            return {
                "route": "definitive",
                "factual_data": factual_data,
                "filtered_context": "",
                "confidence": confidence,
                "message": "Definitive match found. Returning verified configuration."
            }
        
        # LOW CONFIDENCE: Proceed to full RAG
        if enable_faiss:
            faiss_result = self._query_faiss_context(user_query, query_embedding)
            filtered_context = faiss_result.get("content", "")
        else:
            filtered_context = ""
        
        return {
            "route": "full_rag",
            "factual_data": factual_data if confidence > 0.7 else "",  # Include as supporting facts if moderate confidence
            "filtered_context": filtered_context,
            "confidence": confidence,
            "message": "Proceeding to full RAG with LLM generation."
        }
    
    def _query_postgres_facts(self, query: str, query_embedding) -> Optional[Dict]:
        """
        Query PostgreSQL for verified facts with semantic similarity.
        
        Returns:
            Dict with 'content' and 'confidence' or None if unavailable
        """
        try:
            # Execute semantic search against verified facts
            results = self.postgres_client.semantic_search(
                query=query,
                query_embedding=query_embedding,
                top_k=1,
                table="verified_facts"
            )
            
            if not results:
                return {"content": "", "confidence": 0.0}
            
            top_result = results[0]
            return {
                "content": top_result.get("text", ""),
                "confidence": top_result.get("similarity", 0.0)
            }
        
        except Exception as e:
            print(f"Warning: PostgreSQL query failed: {e}")
            return None
    
    def _query_faiss_context(self, query: str, query_embedding) -> Dict:
        """
        Query FAISS for procedural documentation context.
        
        Returns:
            Dict with 'content' key containing concatenated chunks
        """
        try:
            results = self.faiss_client.search(query_embedding=query_embedding, top_k=5)
            
            # Concatenate chunks with separator
            chunks = [r.get("text", "") for r in results if r.get("text")]
            return {"content": "\n---\n".join(chunks)}
        
        except Exception as e:
            print(f"Warning: FAISS query failed: {e}")
            return {"content": ""}
    
    def _fallback_to_faiss(self, query: str, query_embedding, enable_faiss: bool) -> Dict:
        """
        Fallback when PostgreSQL is unavailable.
        """
        if enable_faiss:
            faiss_result = self._query_faiss_context(query, query_embedding)
            return {
                "route": "full_rag",
                "factual_data": "",
                "filtered_context": faiss_result.get("content", ""),
                "confidence": 0.0,
                "message": "PostgreSQL unavailable. Using FAISS only."
            }
        
        return {
            "route": "no_context",
            "factual_data": "",
            "filtered_context": "",
            "confidence": 0.0,
            "message": "No retrieval sources available."
        }
