# Benchmarking Pipeline

Automated benchmarking system for evaluating LLM models (Gemini, Llama) on network configuration generation tasks with comprehensive metrics.

## Overview

The benchmarking pipeline automatically evaluates models across multiple configurations:

1. **Gemini with RAG** - Gemini model with retrieval-augmented generation
2. **Gemini without RAG** - Gemini model direct inference only
3. **Llama with RAG (no loopback)** - Llama with RAG, no fallback retry
4. **Llama without RAG (no loopback)** - Llama direct inference, no fallback
5. **Llama with RAG + loopback** - Llama with RAG and automatic retry on failure

## Quick Start

### Prerequisites

1. Ensure all services are running:
   - **T1 (Data Assets)**: `http://localhost:8000`
   - **T2 (LLM Inference)**: `http://localhost:8001`
   - **T3 (Batfish Validation)**: `http://localhost:5000`

2. Virtual environment is activated with all dependencies installed

### Run Benchmarks

```bash
# From the benchmarks directory
cd 03_AGENT_VALIDATION/benchmarks

# Run with default test suite
python benchmarking.py

# Run with custom test suite
python benchmarking.py --test-suite custom_tests.json

# Run with verbose output
python benchmarking.py --verbose

# Custom output directory
python benchmarking.py --output-dir ./my_results
```

## Test Suite Format

The script expects `test_suite.json` in the same directory. Each test has:

```json
{
  "test_suite_name": "cisco_ios_security_acl_aaa_ssh",
  "description": "Test suite description",
  "tests": [
    {
      "id": "SEC-SSH-001",
      "query": "On router R1, configure SSHv2-only remote management...",
      "expected": {
        "response": [
          {
            "device_name": "R1",
            "configuration_mode_commands": ["hostname R1", "..."],
            "protocol": "SECURITY",
            "intent": [...]
          }
        ]
      }
    }
  ]
}
```

## Output Structure

Each run creates a timestamped directory under `runs/`:

```
runs/2025-11-25_14-32-10/
├── config.json              # Run metadata and configuration
├── raw_responses.jsonl      # Complete model outputs (one per line)
├── metrics.csv              # Metrics in CSV format
└── metrics.json             # Metrics in JSON format
```

### config.json

Contains run metadata:
```json
{
  "test_suite_name": "cisco_ios_security_acl_aaa_ssh",
  "benchmark_configs": [
    {"model": "gemini", "rag_enabled": true, "loopback_enabled": false, "description": "..."},
    ...
  ],
  "timestamp": "2025-11-25T14:32:10",
  "total_tests": 5
}
```

### raw_responses.jsonl

One JSON object per line with complete details:
```json
{
  "test_id": "SEC-SSH-001",
  "model": "gemini",
  "rag_enabled": true,
  "loopback_enabled": false,
  "loopback_attempted": false,
  "query": "...",
  "model_output": {
    "response": [...],
    "config": "...",
    "error": null
  },
  "validation": {
    "result": "OK",
    "summary": {...}
  },
  "verdict": {
    "text": "PASS: Configuration correctly implements...",
    "status": "PASS"
  },
  "expected": {...}
}
```

### metrics.csv / metrics.json

Computed metrics for each test run:

| Column | Description |
|--------|-------------|
| `test_id` | Test case identifier |
| `query` | Input query |
| `model` | Model name (gemini/llama) |
| `rag_enabled` | RAG enabled (True/False) |
| `loopback_enabled` | Loopback enabled (True/False) |
| `loopback_attempted` | Whether loopback was triggered |
| `exact_match` | Exact match with expected (0/1) |
| `commands_precision` | Precision of config commands (0.0-1.0) |
| `commands_recall` | Recall of config commands (0.0-1.0) |
| `commands_f1` | F1 score for commands (0.0-1.0) |
| `intent_match` | Intent structure match (0/1) |
| `batfish_pass` | Batfish validation passed (0/1) |
| `batfish_violations` | Number of Batfish violations |
| `ai_verdict` | AI verdict (PASS/FAIL/ERROR) |
| `has_error` | Whether an error occurred (0/1) |
| `timestamp` | ISO8601 timestamp |

## Metrics Explained

### 1. Exact Match
- Binary (0 or 1)
- Compares entire response structure after normalization
- Strict equality check

### 2. Command Metrics
- **Precision**: `TP / (TP + FP)` - How many predicted commands are correct
- **Recall**: `TP / (TP + FN)` - How many expected commands were generated
- **F1**: Harmonic mean of precision and recall

### 3. Intent Match
- Binary (0 or 1)
- Compares intent blocks (type, service, requirements)
- All expected intents must be present, no extra ones

### 4. Batfish Validation
- **batfish_pass**: Whether configuration passes all Batfish checks
- **batfish_violations**: Count of validation failures
- Checks:
  - Control Plane (CP): Interface configurations
  - Topology (TP): Network topology correctness
  - Reachability (REACH): Connectivity requirements

### 5. AI Verdict
- LLM-generated assessment: PASS/FAIL/ERROR
- Based on validation results and configuration quality
- Includes explanation and suggestions

## Analyzing Results

### View Summary in Terminal

The script outputs summary statistics grouped by configuration:

```
SUMMARY STATISTICS BY CONFIGURATION
================================================================================

Gemini with RAG:
  Runs completed: 5
  Avg exact match: 60.00%
  Avg commands F1: 85.50%
  Avg intent match: 80.00%
  Avg Batfish pass: 100.00%
  AI verdict PASS: 5/5

Llama with RAG + loopback:
  Runs completed: 5
  Avg exact match: 40.00%
  Avg commands F1: 75.20%
  Avg intent match: 60.00%
  Avg Batfish pass: 80.00%
  AI verdict PASS: 4/5
  Loopback attempts: 2
```

### Analyze with benchmark_analysis.py

For deeper analysis:

```bash
python benchmark_analysis.py runs/2025-11-25_14-32-10
```

This generates:
- Comparative charts by model and configuration
- Per-test breakdowns
- Error analysis
- Statistical significance tests

## Troubleshooting

### Service Connection Errors

If you see connection errors:

```bash
# Check service status
curl http://localhost:8000/health  # T1
curl http://localhost:8001/health  # T2
curl http://localhost:5000/health  # T3
```

### Import Errors

Ensure agent service is in Python path:

```bash
# From benchmarks directory
export PYTHONPATH="${PYTHONPATH}:../langchain_agent"  # Linux/Mac
$env:PYTHONPATH += ";..\langchain_agent"  # Windows PowerShell
```

### Timeout Issues

For long-running tests, adjust timeout in agent config:

```json
{
  "HTTP_TIMEOUT": 180
}
```

### Memory Issues

For large test suites, process in batches by editing `test_suite.json` to include only a subset of tests.

## Advanced Usage

### Custom Test Suite

Create your own `test_suite.json`:

1. Define test cases with queries and expected outputs
2. Run benchmarks: `python benchmarking.py --test-suite my_tests.json`

### Extending Metrics

To add custom metrics, modify `compute_metrics()` in `benchmarking.py`:

```python
def compute_metrics(test_case, model_output, batfish_result):
    # ... existing metrics ...
    
    # Add custom metric
    custom_score = compute_custom_metric(model_output)
    metrics["custom_score"] = custom_score
    
    return metrics
```

### Parallel Execution

For faster benchmarking with multiple workers:

```python
# TODO: Add multiprocessing support
# Use ProcessPoolExecutor to run test cases in parallel
```

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Run Benchmarks
  run: |
    cd 03_AGENT_VALIDATION/benchmarks
    python benchmarking.py --output-dir ./ci_results
    
- name: Upload Results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: 03_AGENT_VALIDATION/benchmarks/ci_results
```

## Notes

- **Loopback**: Automatic retry mechanism that disables RAG on failure
- **Test Isolation**: Each test is independent; failures don't affect others
- **Deterministic**: Same test suite should produce comparable results
- **Production Ready**: Handles errors gracefully, continues on failures
