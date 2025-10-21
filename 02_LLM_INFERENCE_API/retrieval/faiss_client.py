"""
FAISS Client for procedural documentation retrieval.
"""

import os
import faiss
import numpy as np
import pickle
from typing import List, Dict


class FAISSClient:
    """
    Client for querying FAISS index of procedural documentation.
    """
    
    def __init__(self, index_path: str = None, metadata_path: str = None):
        """
        Initialize FAISS client.
        
        Args:
            index_path: Path to FAISS index file (.index)
            metadata_path: Path to metadata pickle file (.pkl)
        """
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.index_path = index_path or os.path.join(base_dir, 'faiss_procedural.index')
        self.metadata_path = metadata_path or os.path.join(base_dir, 'faiss_metadata.pkl')
        
        self.index = None
        self.metadata = None
    
    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            else:
                print(f"FAISS index not found: {self.index_path}")
                return False
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                print(f"FAISS metadata not found: {self.metadata_path}")
                return False
            
            return True
        
        except Exception as e:
            print(f"FAISS index loading failed: {e}")
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search FAISS index for similar procedural chunks.
        
        Args:
            query_embedding: Query vector from embedding model
            top_k: Number of results to return
        
        Returns:
            List of dicts with 'text', 'metadata', and 'score' keys
        """
        if self.index is None:
            if not self.load_index():
                return []
        
        try:
            # Ensure query is 2D array
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Build results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                
                meta = self.metadata[idx]
                results.append({
                    "text": meta.get("content", ""),
                    "metadata": meta.get("metadata", {}),
                    "score": float(dist),
                    "rank": i + 1
                })
            
            return results
        
        except Exception as e:
            print(f"FAISS search failed: {e}")
            return []
