"""
Simple TF-IDF based chunk retrieval service.
No definitive facts - pure similarity-based retrieval.
"""

from typing import Dict, Optional, List, Tuple
import math
import re
from collections import Counter

class RetrievalOrchestrator:
    """
    Minimal orchestrator:
    - In-memory chunk store
    - Simple TF-IDF cosine similarity retrieval
    """

    def __init__(self) -> None:
        self._chunks: List[Dict] = []  # [{ "text": str, "meta": dict }]
        self._df: Counter = Counter()
        self._token_pattern = re.compile(r"[a-zA-Z0-9_]+")
        self._stop = set([
            "the","a","an","and","or","of","to","in","on","for","with","is","are","be","as","at","by","from"
        ])

    # ---------------------
    # Public API
    # ---------------------

    def add_chunks(self, chunks: List[str], meta: Optional[Dict] = None) -> None:
        """Ingest raw text chunks."""
        for txt in chunks:
            if not txt or not txt.strip():
                continue
            entry = {"text": txt.strip(), "meta": (meta or {}).copy()}
            self._chunks.append(entry)
        self._recompute_df()

    def retrieve_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Return top-k chunks: [{ chunk, score }]."""
        if not query or not self._chunks:
            return []
        q_vec = self._tfidf(self._tokens(query), self._df, len(self._chunks))
        scored: List[Tuple[float, Dict]] = []
        for ch in self._chunks:
            d_vec = self._tfidf(self._tokens(ch["text"]), self._df, len(self._chunks))
            score = self._cosine(q_vec, d_vec)
            scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, int(k))]
        return [{"chunk": c["text"], "score": round(s, 6)} for s, c in top]

    # ---------------------
    # Internals
    # ---------------------

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


