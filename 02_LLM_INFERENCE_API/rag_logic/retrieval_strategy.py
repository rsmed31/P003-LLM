"""
Retrieval Strategy Module

Implements deterministic, configurable chunk selection for RAG pipeline.
Handles code/theory balance, protocol-specific adjustments, and quality scoring.
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class RetrievalPlan:
    """Configuration for chunk retrieval strategy."""
    total_chunks: int
    code_ratio: float
    theory_ratio: float
    min_code: int
    min_theory: int
    code_chunk_priority: float = 1.2  # Boost for CLI-heavy chunks
    max_tokens_per_chunk: int = 400
    chunk_temperature: float = 0.0  # 0 = deterministic, >0 = randomness in near-ties
    
    @property
    def target_code_chunks(self) -> int:
        """Calculate target number of code chunks."""
        return max(self.min_code, int(self.total_chunks * self.code_ratio))
    
    @property
    def target_theory_chunks(self) -> int:
        """Calculate target number of theory chunks."""
        return max(self.min_theory, int(self.total_chunks * self.theory_ratio))
    
    def adjust_for_total(self) -> Tuple[int, int]:
        """
        Adjust code/theory targets to exactly match total_chunks.
        Returns: (final_code_count, final_theory_count)
        """
        target_code = self.target_code_chunks
        target_theory = self.target_theory_chunks
        
        # If sum exceeds total, scale proportionally
        if target_code + target_theory > self.total_chunks:
            ratio = self.total_chunks / (target_code + target_theory)
            target_code = max(self.min_code, int(target_code * ratio))
            target_theory = self.total_chunks - target_code
        
        # If sum is less than total, add to dominant type
        elif target_code + target_theory < self.total_chunks:
            remaining = self.total_chunks - (target_code + target_theory)
            if self.code_ratio >= self.theory_ratio:
                target_code += remaining
            else:
                target_theory += remaining
        
        return target_code, target_theory


class RetrievalStrategy:
    """
    Deterministic chunk selection strategy for RAG pipeline.
    
    Features:
    - Protocol-specific chunk distribution
    - Code/theory balancing with backfill
    - CLI command density scoring
    - Query-intent boosting
    - Correlation-based quality scoring
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize strategy with configuration."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'retrieval_plan.json'
            )
        
        self.config = self._load_config(config_path)
        self._cli_pattern = self._compile_cli_patterns()
        self._config_intent_keywords = [
            'configure', 'setup', 'enable', 'create', 'implement', 'build'
        ]
    
    def _load_config(self, path: str) -> Dict:
        """Load retrieval plan configuration."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return sensible defaults
            return {
                "default": {
                    "total_chunks": 15,
                    "code_ratio": 0.6,
                    "theory_ratio": 0.4,
                    "min_code": 3,
                    "min_theory": 2,
                    "code_chunk_priority": 1.2,
                    "max_tokens_per_chunk": 400,
                    "chunk_temperature": 0.0
                },
                "protocol_overrides": {}
            }
    
    def _compile_cli_patterns(self) -> re.Pattern:
        """Compile CLI command patterns for density scoring."""
        patterns = [
            r'\brouter\s+', r'\binterface\s+', r'\bip\s+address\s+',
            r'\bnetwork\s+', r'\bswitchport\s+', r'\bvlan\s+',
            r'\baccess-list\s+', r'\bno\s+shutdown\b', r'\bexit\b',
            r'\bconfigure\s+terminal\b', r'\bservice-policy\s+',
            r'\bcrypto\s+', r'\bspanning-tree\s+', r'\baaa\s+',
            r'\bsnmp-server\s+', r'\blogging\s+', r'\bntp\s+',
            r'\broute-map\s+', r'\bpolicy-map\s+', r'\bclass-map\s+'
        ]
        return re.compile('|'.join(patterns), re.IGNORECASE)
    
    def get_plan_for_query(self, query: str, base_total: Optional[int] = None) -> RetrievalPlan:
        """
        Get retrieval plan for specific query.
        Detects protocol and adjusts parameters accordingly.
        """
        # Detect protocol
        protocol = self._detect_protocol(query)
        
        # Get base config (protocol-specific or default)
        if protocol and protocol in self.config.get("protocol_overrides", {}):
            plan_config = self.config["protocol_overrides"][protocol].copy()
        else:
            plan_config = self.config["default"].copy()
        
        # Override total if provided
        if base_total is not None:
            plan_config["total_chunks"] = base_total
        
        # Detect config intent and boost code ratio
        if self._has_config_intent(query):
            plan_config["code_ratio"] = min(plan_config["code_ratio"] + 0.1, 0.8)
            plan_config["code_chunk_priority"] = 1.3
        
        return RetrievalPlan(**plan_config)
    
    def _detect_protocol(self, query: str) -> Optional[str]:
        """Detect network protocol mentioned in query."""
        query_lower = query.lower()
        protocols = ["ospf", "bgp", "eigrp", "rip", "isis", "vlan", "stp", "hsrp", "vrrp"]
        for protocol in protocols:
            if protocol in query_lower:
                return protocol
        return None
    
    def _has_config_intent(self, query: str) -> bool:
        """Check if query has configuration intent."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self._config_intent_keywords)
    
    def select_chunks(
        self,
        scored_chunks: List[Tuple[float, Dict]],
        plan: RetrievalPlan,
        query: str
    ) -> Dict:
        """
        Select chunks according to plan with deterministic logic.
        
        Args:
            scored_chunks: List of (score, chunk_dict) tuples
            plan: RetrievalPlan configuration
            query: Original query for boosting
        
        Returns:
            Dict with selected code_chunks, theory_chunks, and metadata
        """
        # Deduplicate chunks by text content (keep highest score)
        seen_texts = {}
        for score, chunk in scored_chunks:
            chunk_text = chunk["text"].strip()
            if chunk_text not in seen_texts or score > seen_texts[chunk_text][0]:
                seen_texts[chunk_text] = (score, chunk)
        
        # Rebuild deduplicated list
        scored_chunks = list(seen_texts.values())
        
        # Separate by type and apply boosts
        code_candidates = []
        theory_candidates = []
        
        for score, chunk in scored_chunks:
            chunk_type = chunk.get("type", "theory")
            
            # Apply CLI density boost for code chunks
            if chunk_type == "code":
                cli_density = self._calculate_cli_density(chunk["text"])
                boosted_score = score * (1.0 + cli_density * 0.5)
                
                # Apply config intent boost
                if self._has_config_intent(query):
                    boosted_score *= plan.code_chunk_priority
                
                code_candidates.append((boosted_score, chunk))
            else:
                theory_candidates.append((score, chunk))
        
        # Sort by score (deterministic)
        code_candidates.sort(key=lambda x: x[0], reverse=True)
        theory_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Get target counts
        target_code, target_theory = plan.adjust_for_total()
        
        # Select top N of each type
        selected_code = code_candidates[:target_code]
        selected_theory = theory_candidates[:target_theory]
        
        # Backfill if one type is insufficient
        total_selected = len(selected_code) + len(selected_theory)
        if total_selected < plan.total_chunks:
            remaining = plan.total_chunks - total_selected
            
            # Determine which pool to backfill from
            if len(selected_code) < target_code and code_candidates[len(selected_code):]:
                # Need more code, take from remaining code
                backfill = code_candidates[len(selected_code):len(selected_code)+remaining]
                selected_code.extend(backfill)
            elif len(selected_theory) < target_theory and theory_candidates[len(selected_theory):]:
                # Need more theory, take from remaining theory
                backfill = theory_candidates[len(selected_theory):len(selected_theory)+remaining]
                selected_theory.extend(backfill)
            else:
                # Backfill from whichever has more candidates
                if len(code_candidates) > len(selected_code):
                    backfill = code_candidates[len(selected_code):len(selected_code)+remaining]
                    selected_code.extend(backfill)
                elif len(theory_candidates) > len(selected_theory):
                    backfill = theory_candidates[len(selected_theory):len(selected_theory)+remaining]
                    selected_theory.extend(backfill)
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(
            selected_code, selected_theory, query
        )
        
        # Format results
        return {
            "code_chunks": [
                {
                    "text": chunk["text"],
                    "score": float(score),
                    "type": "code",
                    "cli_density": self._calculate_cli_density(chunk["text"])
                }
                for score, chunk in selected_code
            ],
            "theory_chunks": [
                {
                    "text": chunk["text"],
                    "score": float(score),
                    "type": "theory"
                }
                for score, chunk in selected_theory
            ],
            "metadata": {
                "total_chunks": len(selected_code) + len(selected_theory),
                "code_count": len(selected_code),
                "theory_count": len(selected_theory),
                "target_code": target_code,
                "target_theory": target_theory,
                "actual_code_ratio": len(selected_code) / max(1, len(selected_code) + len(selected_theory)),
                "quality_score": quality_score,
                "plan_config": {
                    "total_chunks": plan.total_chunks,
                    "code_ratio": plan.code_ratio,
                    "theory_ratio": plan.theory_ratio
                }
            }
        }
    
    def _calculate_cli_density(self, text: str) -> float:
        """Calculate CLI command density (0.0 to 1.0)."""
        lines = text.split('\n')
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines == 0:
            return 0.0
        
        cli_lines = sum(1 for line in lines if self._cli_pattern.search(line))
        return cli_lines / total_lines
    
    def _calculate_quality_score(
        self,
        code_chunks: List[Tuple[float, Dict]],
        theory_chunks: List[Tuple[float, Dict]],
        query: str
    ) -> float:
        """
        Calculate overall context quality score (0.0 to 1.0).
        
        Factors:
        - Average chunk relevance scores
        - Code-theory correlation
        - Query term coverage
        - Balance between code and theory
        """
        if not code_chunks and not theory_chunks:
            return 0.0
        
        # Average relevance scores
        all_scores = [s for s, _ in code_chunks] + [s for s, _ in theory_chunks]
        avg_relevance = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Correlation: check if theory explains code terms
        correlation = self._measure_code_theory_correlation(code_chunks, theory_chunks)
        
        # Query coverage
        query_tokens = set(query.lower().split())
        all_text = ' '.join([chunk["text"] for _, chunk in code_chunks + theory_chunks])
        all_tokens = set(all_text.lower().split())
        coverage = len(query_tokens & all_tokens) / max(len(query_tokens), 1)
        
        # Balance penalty if ratio is very skewed
        code_count = len(code_chunks)
        theory_count = len(theory_chunks)
        total = code_count + theory_count
        
        if total > 0:
            actual_code_ratio = code_count / total
            # Ideal is 0.6, penalize if too far
            balance_penalty = abs(actual_code_ratio - 0.6) * 0.3
        else:
            balance_penalty = 0.5
        
        # Combine factors
        quality = (
            avg_relevance * 0.35 +
            correlation * 0.25 +
            coverage * 0.20 +
            (1.0 - balance_penalty) * 0.20
        )
        
        return min(max(quality, 0.0), 1.0)
    
    def _measure_code_theory_correlation(
        self,
        code_chunks: List[Tuple[float, Dict]],
        theory_chunks: List[Tuple[float, Dict]]
    ) -> float:
        """
        Measure how well theory chunks explain code chunks.
        Returns 0.0 to 1.0.
        """
        if not code_chunks or not theory_chunks:
            return 0.5  # Neutral if only one type
        
        # Extract technical terms from code
        code_terms = set()
        for _, chunk in code_chunks:
            # Extract router/protocol/interface names
            text = chunk["text"].lower()
            code_terms.update(re.findall(r'\b(?:router|ospf|bgp|vlan|interface)\s+\w+', text))
            code_terms.update(re.findall(r'\b(?:network|area|ip)\s+[\d.]+', text))
        
        # Check coverage in theory
        theory_text = ' '.join([chunk["text"].lower() for _, chunk in theory_chunks])
        
        explained = sum(1 for term in code_terms if term in theory_text)
        correlation = explained / max(len(code_terms), 1)
        
        return min(correlation * 1.5, 1.0)  # Scale up importance
    
    def apply_query_boosting(
        self,
        chunks: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Apply query-specific boosting to chunk scores.
        Boosts chunks with high query term overlap.
        """
        query_tokens = set(query.lower().split())
        
        boosted_chunks = []
        for chunk in chunks:
            chunk_tokens = set(chunk["text"].lower().split())
            overlap = len(query_tokens & chunk_tokens)
            overlap_ratio = overlap / max(len(query_tokens), 1)
            
            # Apply boost (max 30% increase)
            boost_multiplier = 1.0 + (overlap_ratio * 0.3)
            boosted_score = chunk["score"] * boost_multiplier
            
            boosted_chunk = chunk.copy()
            boosted_chunk["score"] = boosted_score
            boosted_chunk["query_overlap"] = overlap_ratio
            boosted_chunks.append(boosted_chunk)
        
        return boosted_chunks


def create_strategy(config_path: Optional[str] = None) -> RetrievalStrategy:
    """Factory function to create retrieval strategy."""
    return RetrievalStrategy(config_path)


# Convenience function for backward compatibility
def get_chunk_distribution(query: str, total_chunks: int) -> Tuple[int, int]:
    """
    Get code/theory chunk distribution for query.
    Returns: (code_count, theory_count)
    """
    strategy = create_strategy()
    plan = strategy.get_plan_for_query(query, total_chunks)
    return plan.adjust_for_total()
