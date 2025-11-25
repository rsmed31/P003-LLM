# BENCHMARK ANALYSIS SUMMARY

Run Directory: C:\Users\enmoh\Desktop\P003-LLM\03_AGENT_VALIDATION\benchmarks\runs\2025-11-25_19-40-10
Analysis Date: 2025-11-25_19-40-10


## RAG & CHUNKING CONFIGURATION

Chunk Size: unknown
Chunk Overlap: unknown
Retriever Top-K: unknown
Retriever Type: unknown


## OVERALL PERFORMANCE METRICS



## GEMINI (RAG OFF):

  Commands F1:       0.3803 ± 0.2399
  Exact Match:       0.0000
  Precision:         0.3940
  Recall:            0.3862
  Batfish Pass Rate: 1.0000
  Number of Tests:   17


## GEMINI (RAG ON):

  Commands F1:       0.3575 ± 0.2119
  Exact Match:       0.0000
  Precision:         0.3855
  Recall:            0.3474
  Batfish Pass Rate: 1.0000
  Number of Tests:   17


## LLAMA (RAG OFF):

  Commands F1:       0.0000 ± 0.0000
  Exact Match:       0.0000
  Precision:         0.0000
  Recall:            0.0000
  Batfish Pass Rate: 1.0000
  Number of Tests:   7


## LLAMA (RAG ON):

  Commands F1:       0.0000 ± 0.0000
  Exact Match:       0.0000
  Precision:         0.0000
  Recall:            0.0000
  Batfish Pass Rate: 1.0000
  Number of Tests:   17



## RAG IMPACT ANALYSIS



## GEMINI:

  F1 Delta (RAG On - RAG Off):          -0.0228
  Exact Match Delta:                    +0.0000
  Batfish Pass Delta:                   +0.0000
  F1 with RAG:    0.3575
  F1 without RAG: 0.3803


## LLAMA:

  F1 Delta (RAG On - RAG Off):          +0.0000
  Exact Match Delta:                    +0.0000
  Batfish Pass Delta:                   +0.0000
  F1 with RAG:    0.0000
  F1 without RAG: 0.0000


**BEST MODEL FOR RAG: LLAMA with F1 improvement of +0.0000**



5 HARDEST TESTS (Lowest Avg F1)


## SEC-8021X-017:

  Avg F1 Score:      0.0000
  Avg Batfish Pass:  1.0000
  Best Combination:  gemini_ragon (F1: 0.0000)


## SEC-8021X-011:

  Avg F1 Score:      0.0833
  Avg Batfish Pass:  1.0000
  Best Combination:  gemini_ragoff (F1: 0.3333)


## SEC-8021X-013:

  Avg F1 Score:      0.0893
  Avg Batfish Pass:  1.0000
  Best Combination:  gemini_ragon (F1: 0.1429)


## SEC-8021X-014:

  Avg F1 Score:      0.1247
  Avg Batfish Pass:  1.0000
  Best Combination:  gemini_ragoff (F1: 0.3158)


## SEC-8021X-010:

  Avg F1 Score:      0.1333
  Avg Batfish Pass:  1.0000
  Best Combination:  gemini_ragon (F1: 0.5333)


5 EASIEST TESTS (Highest Avg F1)


## SEC-8021X-006:

  Avg F1 Score:      0.4486
  Avg Batfish Pass:  1.0000
  Best Combination:  gemini_ragoff (F1: 0.4762)


## SEC-8021X-007:

  Avg F1 Score:      0.4152
  Avg Batfish Pass:  1.0000
  Best Combination:  gemini_ragoff (F1: 0.4667)


## SEC-8021X-009:

  Avg F1 Score:      0.3750
  Avg Batfish Pass:  1.0000
  Best Combination:  gemini_ragoff (F1: 0.6250)


## SEC-AAA-RADIUS-004:

  Avg F1 Score:      0.3692
  Avg Batfish Pass:  1.0000
  Best Combination:  gemini_ragoff (F1: 0.8000)


## SEC-ACL-DATA-003:

  Avg F1 Score:      0.3667
  Avg Batfish Pass:  1.0000
  Best Combination:  gemini_ragon (F1: 0.8000)



## KEY INSIGHTS

1. Overall RAG Impact: Negative
   - Average F1 with RAG:    0.1787
   - Average F1 without RAG: 0.1901
   - Improvement:            -0.0114 (-1.14%)

2. Chunking Configuration Impact:
   - Current chunk size of unknown with overlap of
     unknown appears to be suboptimal
   - Retrieving top-unknown chunks per query

3. Model Performance:
   - GEMINI shows the best absolute performance with RAG

## (F1: 0.3575)

   - LLAMA benefits most from RAG (improvement: +0.0000)
