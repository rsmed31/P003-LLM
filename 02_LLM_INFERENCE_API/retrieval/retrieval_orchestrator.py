"""
Enhanced TF-IDF retrieval with intelligent code-aware and theoretical chunk combination.
Provides confidence scoring based on chunk correlation and relevance.
Integrates with RetrievalStrategy for deterministic, configurable chunk selection.
"""

from typing import Dict, Optional, List, Tuple
import math
import re
from collections import Counter
import sys
import os

# Import retrieval strategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from rag_logic.retrieval_strategy import RetrievalStrategy, RetrievalPlan
except ImportError:
    RetrievalStrategy = None
    RetrievalPlan = None

class RetrievalOrchestrator:
    """
    Enhanced orchestrator:
    - In-memory chunk store with type classification (code/theoretical)
    - TF-IDF cosine similarity retrieval
    - Intelligent combination of code-aware and theoretical chunks
    - Confidence scoring based on correlation analysis
    """

    def __init__(self) -> None:
        self._chunks: List[Dict] = []  # [{ "text": str, "type": "code"|"theory", "meta": dict }]
        self._df: Counter = Counter()
        self._token_pattern = re.compile(r"[a-zA-Z0-9_]+")
        self._stop = set([
            "the","a","an","and","or","of","to","in","on","for","with","is","are","be","as","at","by","from"
        ])
        
        # CLI command patterns for chunk classification
        self._cli_patterns = [
            r'\brouter\s+', r'\binterface\s+', r'\bip\s+address\s+',
            r'\bnetwork\s+', r'\bswitchport\s+', r'\bvlan\s+',
            r'\bno\s+shutdown\b', r'\bexit\b', r'\bconfigure\s+terminal\b'
        ]
        self._cli_regex = re.compile('|'.join(self._cli_patterns), re.IGNORECASE)
        
        # Initialize retrieval strategy
        self._strategy = RetrievalStrategy() if RetrievalStrategy else None

    # ---------------------
    # Public API
    # ---------------------

    def add_chunks(self, chunks: List[str], meta: Optional[Dict] = None) -> None:
        """Ingest raw text chunks with automatic type classification."""
        for txt in chunks:
            if not txt or not txt.strip():
                continue
            
            # Classify chunk as code or theory
            chunk_type = self._classify_chunk(txt)
            
            entry = {
                "text": txt.strip(),
                "type": chunk_type,
                "meta": (meta or {}).copy()
            }
            self._chunks.append(entry)
        self._recompute_df()

    def retrieve_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """
        Return top-k chunks with intelligent code/theory combination using RetrievalStrategy.
        Returns: [{ chunk, score, type, confidence }]
        """
        if not query or not self._chunks:
            return []
        
        # Get raw scored chunks
        q_vec = self._tfidf(self._tokens(query), self._df, len(self._chunks))
        scored: List[Tuple[float, Dict]] = []
        
        for ch in self._chunks:
            d_vec = self._tfidf(self._tokens(ch["text"]), self._df, len(self._chunks))
            score = self._cosine(q_vec, d_vec)
            scored.append((score, ch))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Use retrieval strategy for selection if available
        if self._strategy:
            plan = self._strategy.get_plan_for_query(query, k)
            selection_result = self._strategy.select_chunks(scored, plan, query)
            
            # Convert strategy output to legacy format
            results = []
            for chunk_data in selection_result["code_chunks"]:
                results.append({
                    "chunk": chunk_data["text"],
                    "score": chunk_data["score"],
                    "type": "code",
                    "confidence": min(chunk_data["score"] * 1.2, 1.0),
                    "cli_density": chunk_data.get("cli_density", 0.0)
                })
            for chunk_data in selection_result["theory_chunks"]:
                results.append({
                    "chunk": chunk_data["text"],
                    "score": chunk_data["score"],
                    "type": "theory",
                    "confidence": min(chunk_data["score"] * 1.1, 1.0)
                })
            
            # Sort by score for consistency
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
        else:
            # Fallback to legacy selection
            selected = self._select_balanced_chunks(scored, k)
            results = self._calculate_confidence(selected, query)
            return results

    def retrieve_with_correlation(self, query: str, k: int = 5) -> Dict:
        """
        Advanced retrieval with correlation analysis and strategy-based selection.
        Returns structured output with confidence metrics and quality scoring.
        """
        results = self.retrieve_chunks(query, k)
        
        # Separate code and theory chunks
        code_chunks = [r for r in results if r["type"] == "code"]
        theory_chunks = [r for r in results if r["type"] == "theory"]
        
        # Calculate correlation strength
        correlation_score = self._measure_correlation(code_chunks, theory_chunks)
        
        # Determine overall confidence
        avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0.0
        
        # Calculate context quality using strategy (if available)
        if self._strategy and results:
            # Build mock scored chunks for quality calculation
            code_scored = [(c["score"], {"text": c["chunk"], "type": "code"}) for c in code_chunks]
            theory_scored = [(t["score"], {"text": t["chunk"], "type": "theory"}) for t in theory_chunks]
            quality_score = self._strategy._calculate_quality_score(code_scored, theory_scored, query)
        else:
            quality_score = (avg_confidence + correlation_score) / 2
        
        overall_confidence = quality_score
        
        return {
            "query": query,
            "total_chunks": len(results),
            "code_chunks": len(code_chunks),
            "theory_chunks": len(theory_chunks),
            "correlation_score": round(correlation_score, 4),
            "overall_confidence": round(overall_confidence, 4),
            "context_quality_score": round(quality_score, 4),
            "chunks": results
        }

    # ---------------------
    # Internals
    # ---------------------

    def _classify_chunk(self, text: str) -> str:
        """Classify chunk as 'code' or 'theory' based on CLI command density."""
        lines = text.split('\n')
        cli_lines = sum(1 for line in lines if self._cli_regex.search(line))
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines == 0:
            return "theory"
        
        cli_ratio = cli_lines / total_lines
        return "code" if cli_ratio >= 0.3 else "theory"

    def _select_balanced_chunks(self, scored: List[Tuple[float, Dict]], k: int) -> List[Tuple[float, Dict]]:
        """
        Select k chunks with intelligent balance between code and theory.
        Strategy: 60% code, 40% theory (adjustable based on top scores)
        """
        code_chunks = [(s, c) for s, c in scored if c["type"] == "code"]
        theory_chunks = [(s, c) for s, c in scored if c["type"] == "theory"]
        
        # Target distribution
        target_code = max(1, int(k * 0.6))
        target_theory = k - target_code
        
        # Select top chunks of each type
        selected_code = code_chunks[:target_code]
        selected_theory = theory_chunks[:target_theory]
        
        # If not enough of one type, fill with the other
        total_selected = len(selected_code) + len(selected_theory)
        if total_selected < k:
            remaining = k - total_selected
            if len(selected_code) < target_code:
                selected_theory.extend(theory_chunks[len(selected_theory):len(selected_theory)+remaining])
            else:
                selected_code.extend(code_chunks[len(selected_code):len(selected_code)+remaining])
        
        # Combine and re-sort by score
        combined = selected_code + selected_theory
        combined.sort(key=lambda x: x[0], reverse=True)
        
        return combined[:k]

    def _calculate_confidence(self, scored_chunks: List[Tuple[float, Dict]], query: str) -> List[Dict]:
        """
        Calculate confidence scores for each chunk based on:
        1. Similarity score (base relevance)
        2. Chunk type appropriateness
        3. Cross-correlation with other selected chunks
        """
        if not scored_chunks:
            return []
        
        results = []
        query_tokens = set(self._tokens(query))
        
        for score, chunk in scored_chunks:
            # Base confidence from similarity score
            base_confidence = min(score * 1.2, 1.0)  # Scale up slightly
            
            # Bonus for query keyword coverage
            chunk_tokens = set(self._tokens(chunk["text"]))
            keyword_overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
            
            # Bonus for appropriate chunk type
            type_bonus = 0.1 if chunk["type"] == "code" else 0.05
            
            # Final confidence
            confidence = min(base_confidence + (keyword_overlap * 0.2) + type_bonus, 1.0)
            
            results.append({
                "chunk": chunk["text"],
                "score": round(score, 6),
                "type": chunk["type"],
                "confidence": round(confidence, 4)
            })
        
        return results

    def _measure_correlation(self, code_chunks: List[Dict], theory_chunks: List[Dict]) -> float:
        """
        Measure correlation strength between code and theory chunks.
        High correlation = theory chunks explain the CLI commands in code chunks.
        """
        if not code_chunks or not theory_chunks:
            return 0.5  # Neutral if only one type present
        
        # Extract key terms from code chunks
        code_terms = set()
        for chunk in code_chunks:
            code_terms.update(self._tokens(chunk["chunk"]))
        
        # Check how many code terms are explained in theory chunks
        theory_terms = set()
        for chunk in theory_chunks:
            theory_terms.update(self._tokens(chunk["chunk"]))
        
        # Calculate overlap
        overlap = len(code_terms & theory_terms)
        correlation = overlap / max(len(code_terms), 1)
        
        return min(correlation * 1.5, 1.0)  # Scale up correlation importance

    def _tokens(self, text: str) -> List[str]:
        toks = [t.lower() for t in self._token_pattern.findall(text or "")]
        return [t for t in toks if t not in self._stop]

    def _recompute_df(self) -> None:
        df = Counter()
        for ch in self._chunks:
            terms = set(self._tokens(ch["text"]))
            for t in terms:
                df[t] += 1
        self._df = df

    def _tfidf(self, toks: List[str], df: Counter, N: int) -> Dict[str, float]:
        tf = Counter(toks)
        vec = {}
        for term, f in tf.items():
            dfi = df.get(term, 0)
            idf = math.log((N + 1) / (dfi + 1)) + 1.0
            vec[term] = (f / max(1, len(toks))) * idf
        return vec

    def _cosine(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0
        dot = 0.0
        (a, b) = (v1, v2) if len(v1) < len(v2) else (v2, v1)
        for t, w in a.items():
            dot += w * b.get(t, 0.0)
        n1 = math.sqrt(sum(w*w for w in v1.values()))
        n2 = math.sqrt(sum(w*w for w in v2.values()))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return dot / (n1 * n2)


