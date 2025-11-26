# RAG Retrieval Pipeline - Complete Redesign

## Overview

This document describes the redesigned RAG (Retrieval-Augmented Generation) pipeline for the network-automation LLM system. The new architecture provides **deterministic, configurable, and balanced** chunk retrieval with protocol-specific optimization.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  Protocol Detection         │
         │  Intent Classification      │
         │  (retrieval_strategy.py)    │
         └─────────┬───────────────────┘
                   │
                   ▼
         ┌─────────────────────────────┐
         │  Get Retrieval Plan         │
         │  (retrieval_plan.json)      │
         │  • Total chunks             │
         │  • Code/theory ratio        │
         │  • Protocol overrides       │
         └─────────┬───────────────────┘
                   │
                   ▼
         ┌─────────────────────────────┐
         │  External Retrieval Service │
         │  (Postgres FAISS)           │
         │  Returns raw chunks         │
         └─────────┬───────────────────┘
                   │
                   ▼
         ┌─────────────────────────────┐
         │  RetrievalOrchestrator      │
         │  • TF-IDF scoring           │
         │  • Type classification      │
         │  • Correlation analysis     │
         └─────────┬───────────────────┘
                   │
                   ▼
         ┌─────────────────────────────┐
         │  RetrievalStrategy          │
         │  • Select N code chunks     │
         │  • Select M theory chunks   │
         │  • Apply CLI density boost  │
         │  • Backfill if needed       │
         └─────────┬───────────────────┘
                   │
                   ▼
         ┌─────────────────────────────┐
         │  Code-Aware Filter          │
         │  • Filter CLI noise         │
         │  • Preserve command context │
         │  • Remove comments          │
         └─────────┬───────────────────┘
                   │
                   ▼
         ┌─────────────────────────────┐
         │  Structured Context Builder │
         │  ## CODE CONTEXT            │
         │  <code chunks>              │
         │  ## THEORY CONTEXT          │
         │  <theory chunks>            │
         └─────────┬───────────────────┘
                   │
                   ▼
         ┌─────────────────────────────┐
         │  LLM Generation             │
         │  (Gemini / Llama)           │
         └─────────────────────────────┘
```

## Key Components

### 1. retrieval_strategy.py

**Purpose**: Deterministic chunk selection strategy

**Features**:
- Protocol detection (OSPF, BGP, VLAN, etc.)
- Intent classification (configure, troubleshoot, explain)
- Configurable code/theory ratios
- CLI density scoring
- Quality metrics calculation

**Key Classes**:
```python
class RetrievalPlan:
    total_chunks: int
    code_ratio: float
    theory_ratio: float
    min_code: int
    min_theory: int
    code_chunk_priority: float
    
class RetrievalStrategy:
    def get_plan_for_query(query, base_total) -> RetrievalPlan
    def select_chunks(scored_chunks, plan, query) -> Dict
    def _calculate_quality_score(...) -> float
```

### 2. retrieval_plan.json

**Purpose**: Configuration for retrieval behavior

**Structure**:
```json
{
  "default": {
    "total_chunks": 15,
    "code_ratio": 0.6,
    "theory_ratio": 0.4,
    "min_code": 3,
    "min_theory": 2,
    "code_chunk_priority": 1.2
  },
  "protocol_overrides": {
    "ospf": {
      "total_chunks": 20,
      "code_ratio": 0.7,
      ...
    }
  }
}
```

**Protocol Overrides**:
- `ospf`: 70% code, 20 chunks
- `bgp`: 65% code, 25 chunks
- `vlan`: 75% code, 15 chunks
- `acl`: 80% code, 18 chunks
- `qos`: 65% code, 20 chunks

### 3. RetrievalOrchestrator (Enhanced)

**Purpose**: Chunk scoring and correlation analysis

**New Features**:
- Integrates with RetrievalStrategy
- Provides context quality scoring
- Measures code-theory correlation
- CLI density calculation

**API**:
```python
orchestrator = RetrievalOrchestrator()
orchestrator.add_chunks(chunks)
result = orchestrator.retrieve_with_correlation(query, k)

# Returns:
{
    "total_chunks": 15,
    "code_chunks": 9,
    "theory_chunks": 6,
    "correlation_score": 0.75,
    "context_quality_score": 0.82,
    "overall_confidence": 0.79,
    "chunks": [...]
}
```

### 4. code_aware_filter.py (Enhanced)

**Purpose**: Filter and refine CLI-heavy content

**New Functions**:
```python
# Filter chunks, remove noise
filter_and_refine_context(chunks, aggressive=False) -> str

# Extract only CLI-heavy chunks
extract_cli_heavy_chunks(chunks) -> List[Dict]

# Aggressive filtering for code chunks
refine_code_chunks(code_chunks) -> List[str]
```

**Filtering Rules**:
- Minimum 2-3 CLI commands per chunk
- Removes comments and filler text
- Preserves indentation
- Supports Cisco/Huawei/H3C syntax

### 5. inference.py (Updated)

**Purpose**: Main inference orchestration

**New Features**:
- Strategy-based chunk selection
- Structured context assembly
- Comprehensive metrics logging
- Plan metadata in context.json

**Context Structure**:
```
## CODE CONTEXT (9 chunks, ranked by relevance)

router ospf 1
 network 10.0.0.0 0.255.255.255 area 0
quit

---

interface GigabitEthernet0/0
 ip address 192.168.1.1 255.255.255.0
 no shutdown

## THEORY CONTEXT (6 chunks, ranked by relevance)

OSPF (Open Shortest Path First) is a link-state routing protocol...

---

Area 0 is the backbone area in OSPF...
```

## Workflow

### Example Query: "Configure OSPF for 3 routers"

1. **Protocol Detection**: Detects "OSPF" → applies OSPF plan
   - Total chunks: 20
   - Code ratio: 70%
   - Theory ratio: 30%

2. **External Retrieval**: Fetches 20 chunks from Postgres/FAISS

3. **Orchestrator Processing**:
   - Classifies chunks as code (CLI-heavy) or theory
   - Scores with TF-IDF
   - Calculates correlation

4. **Strategy Selection**:
   - Target: 14 code chunks, 6 theory chunks
   - Applies CLI density boost to code chunks
   - Selects top 14 code + top 6 theory

5. **Code Filtering**:
   - Removes comment lines
   - Preserves command structure
   - Filters noise

6. **Context Assembly**:
   ```
   ## CODE CONTEXT (14 chunks)
   [filtered CLI commands]
   
   ## THEORY CONTEXT (6 chunks)
   [explanatory text]
   ```

7. **Quality Metrics**:
   - Correlation score: 0.78
   - Quality score: 0.85
   - Confidence: 0.82

8. **Saved to context.json**:
   ```json
   {
     "retrieval_plan": {
       "total_chunks": 20,
       "code_chunks": 14,
       "theory_chunks": 6,
       "correlation_score": 0.78,
       "quality_score": 0.85
     }
   }
   ```

## Configuration Guide

### Adjusting Code/Theory Balance

Edit `models/retrieval_plan.json`:

```json
{
  "default": {
    "code_ratio": 0.7,  // Increase for more code
    "theory_ratio": 0.3  // Decrease for less theory
  }
}
```

### Adding New Protocol

```json
{
  "protocol_overrides": {
    "mpls": {
      "total_chunks": 22,
      "code_ratio": 0.65,
      "theory_ratio": 0.35,
      "min_code": 5,
      "min_theory": 3,
      "code_chunk_priority": 1.25
    }
  }
}
```

### Adjusting CLI Filter Aggressiveness

In code:
```python
# Standard filtering
filtered = filter_and_refine_context(chunks, aggressive=False)

# Aggressive filtering (min 3 CLI commands)
filtered = filter_and_refine_context(chunks, aggressive=True)
```

## Quality Metrics

### Correlation Score (0.0 - 1.0)
Measures how well theory chunks explain terms in code chunks.
- < 0.3: Poor correlation (theory doesn't explain code)
- 0.3 - 0.6: Moderate correlation
- > 0.6: Good correlation

### Quality Score (0.0 - 1.0)
Overall context quality based on:
- Average relevance (35%)
- Code-theory correlation (25%)
- Query term coverage (20%)
- Balance penalty (20%)

### Confidence Score (0.0 - 1.0)
Weighted average of correlation and quality.

## API Compatibility

### External Interface (UNCHANGED)

```python
# inference.py
generate(
    query="Configure OSPF...",
    model_name="gemini",
    loopback=False
) -> str  # JSON response
```

### Input/Output (UNCHANGED)

- Input: Query string + model name
- Output: JSON with configuration objects
- External retrieval service calls remain the same

## Testing

### Unit Tests

Test retrieval strategy:
```python
from rag_logic.retrieval_strategy import RetrievalStrategy

strategy = RetrievalStrategy()
plan = strategy.get_plan_for_query("Configure OSPF", 20)
assert plan.code_ratio == 0.7  # OSPF override
```

### Integration Test

```python
from endpoints.inference import generate

response = generate(
    query="Configure OSPF area 0 for R1 and R2",
    model_name="gemini"
)

# Check context.json for metrics
# Verify CODE CONTEXT and THEORY CONTEXT sections
```

## Performance

### Determinism
- Same query + same chunks = same selection (temperature=0.0)
- No randomness in chunk selection
- Reproducible results

### Efficiency
- Strategy selection: O(n log n) for sorting
- Filtering: O(n × m) where m = lines per chunk
- Overall: ~100-200ms for 50 chunks

### Memory
- Minimal overhead (~1-2 MB for strategy)
- Chunks processed in-place
- No large intermediate structures

## Migration

### From Old System

1. **No changes to external calls** - retrieval service unchanged
2. **Context structure enhanced** - but backward compatible
3. **New metrics added** - existing metrics preserved
4. **Configuration externalized** - hardcoded values moved to JSON

### Rollback

If needed, set in `retrieval_plan.json`:
```json
{
  "default": {
    "code_ratio": 0.5,
    "theory_ratio": 0.5
  }
}
```

## Troubleshooting

### Too Many Code Chunks

Lower `code_ratio` in retrieval_plan.json:
```json
{"code_ratio": 0.5}  // Instead of 0.6
```

### Low Quality Score

Check:
1. Correlation score - theory should explain code terms
2. Query coverage - chunks should contain query keywords
3. Balance - avoid extreme ratios (e.g., 90/10)

### Filter Too Aggressive

```python
# Use less aggressive filtering
filter_and_refine_context(chunks, aggressive=False)
```

## Future Enhancements

- [ ] Dynamic ratio adjustment based on query complexity
- [ ] Multi-language protocol support (Chinese, Japanese)
- [ ] Chunk caching for repeated queries
- [ ] A/B testing framework for ratio optimization
- [ ] Real-time quality feedback loop
- [ ] Semantic deduplication for similar chunks

## Summary

The redesigned RAG pipeline provides:

✅ **Deterministic** - Same input → same output  
✅ **Configurable** - JSON-based configuration  
✅ **Balanced** - 60/40 code/theory (adjustable)  
✅ **Protocol-aware** - OSPF gets 70% code  
✅ **Filtered** - CLI noise removed  
✅ **Structured** - Clear CODE/THEORY sections  
✅ **Measurable** - Quality, correlation, confidence scores  
✅ **Compatible** - No API changes  

## Authors

- RAG Pipeline Redesign: AI Assistant
- Integration: Team 2
- Configuration: Network Automation Team

## Support

For issues:
1. Check `models/context.json` for retrieval metrics
2. Review `retrieval_plan.json` configuration
3. Examine console logs for RAG METRICS output
4. Verify external retrieval service connectivity
