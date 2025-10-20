"""
Runtime FAISS Retrieval Module for P003 - Team 2
Task: LLM-003 - First Layer RAG Retrieval

This module provides runtime vector similarity search capabilities using the
FAISS index and assets created by Team 1. It loads the serialized index once
and provides efficient retrieval of relevant chunks based on user queries.
"""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Constants for asset paths (Team 1 deliverables)
INDEX_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../01_DATA_ASSETS/chunks_faiss_data/faiss_index.bin"
)
MAP_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../01_DATA_ASSETS/chunks_faiss_data/id_map.json"
)


class FAISSAssetLoader:
    """
    Asset loader and retrieval class for FAISS-based vector search.
    
    This class loads the FAISS index and ID mapping once during initialization
    and provides efficient retrieval capabilities for runtime queries.
    """
    
    def __init__(self):
        """
        Initialize the FAISS asset loader.
        
        Loads:
        - FAISS index from serialized binary file
        - ID-to-chunk mapping from JSON file
        - Sentence Transformer embedding model (all-MiniLM-L6-v2)
        
        Raises:
            FileNotFoundError: If index or mapping files are not found
            RuntimeError: If assets fail to load properly
        """
        print("[FAISSAssetLoader] Initializing runtime retrieval assets...")
        
        # Validate paths exist
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at: {INDEX_PATH}")
        if not os.path.exists(MAP_PATH):
            raise FileNotFoundError(f"ID mapping not found at: {MAP_PATH}")
        
        # Load FAISS index
        try:
            self.index = faiss.read_index(INDEX_PATH)
            print(f"[FAISSAssetLoader] ✓ Loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")
        
        # Load ID-to-chunk mapping
        try:
            with open(MAP_PATH, 'r', encoding='utf-8') as f:
                self.id_map = json.load(f)
            print(f"[FAISSAssetLoader] ✓ Loaded ID mapping with {len(self.id_map)} entries")
        except Exception as e:
            raise RuntimeError(f"Failed to load ID mapping: {e}")
        
        # Initialize embedding model (MUST match Team 1's model)
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[FAISSAssetLoader] ✓ Initialized embedding model: all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {e}")
        
        print("[FAISSAssetLoader] Initialization complete. Ready for retrieval.")
    
    def retrieve_context(self, query_text: str, k: int = 5) -> list[str]:
        """
        Retrieve the top-k most relevant text chunks for a given query.
        
        This method performs vector similarity search using FAISS and returns
        the original text chunks corresponding to the nearest neighbors.
        
        Args:
            query_text (str): The user's natural language query
            k (int): Number of chunks to retrieve (default: 5)
        
        Returns:
            list[str]: List of the top-k most relevant text chunks
        
        Raises:
            ValueError: If query_text is empty or k is invalid
        """
        # Input validation
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if k > self.index.ntotal:
            print(f"[WARNING] k={k} exceeds index size ({self.index.ntotal}). Adjusting k.")
            k = self.index.ntotal
        
        # Step 1: Vectorize the query using the same embedding model
        query_vector = self.embedding_model.encode([query_text])
        query_vector = np.array(query_vector, dtype='float32')
        
        # Step 2: Perform FAISS search
        # Returns: distances (similarity scores) and indices (vector IDs)
        distances, indices = self.index.search(query_vector, k)
        
        # Step 3: Map FAISS indices to original text chunks
        retrieved_chunks = []
        for idx in indices[0]:  # indices[0] because we have a single query
            # Convert numpy int to Python int for JSON key lookup
            chunk_id = str(int(idx))
            
            # Retrieve the original chunk text from the mapping
            if chunk_id in self.id_map:
                retrieved_chunks.append(self.id_map[chunk_id])
            else:
                print(f"[WARNING] Chunk ID {chunk_id} not found in mapping. Skipping.")
        
        print(f"[FAISSAssetLoader] Retrieved {len(retrieved_chunks)} chunks for query: '{query_text[:50]}...'")
        
        return retrieved_chunks