# Quick Start Guide - Benchmarking Pipeline

## Prerequisites Check

```bash
# 1. Check all services are running
curl http://localhost:8000/health  # T1 - Data Assets
curl http://localhost:8001/health  # T2 - LLM Inference  
curl http://localhost:5000/health  # T3 - Batfish Validation

# 2. Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac

# 3. Navigate to benchmarks
cd 03_AGENT_VALIDATION\benchmarks
```

## Run Benchmarks

```bash
# Basic run (uses test_suite.json in current directory)
python benchmarking.py

# With verbose output (recommended for first run)
python benchmarking.py --verbose

# Custom test suite
python benchmarking.py --test-suite path/to/custom_tests.json

# Custom output directory
python benchmarking.py --output-dir ./my_results
```

## What Gets Tested

5 configurations automatically run for each test in `test_suite.json`:

1. **Gemini + RAG** → Best accuracy expected
2. **Gemini (no RAG)** → Direct inference baseline
3. **Llama + RAG** → Alternative model with retrieval
4. **Llama (no RAG)** → Llama baseline
5. **Llama + RAG + Loopback** → Automatic retry on failure

## Output Location

```
runs/2025-11-25_14-32-10/
├── config.json              # What was run
├── raw_responses.jsonl      # All model outputs
├── metrics.csv              # Metrics spreadsheet
└── metrics.json             # Metrics for analysis
```

## Key Metrics to Watch

- **commands_f1**: Overall configuration quality (0.0-1.0, higher better)
- **batfish_pass**: Network validation (0 = fail, 1 = pass)
- **ai_verdict**: PASS/FAIL from AI assessment
- **loopback_attempted**: Retry was needed (only for config #5)

## Troubleshooting

### "Connection refused"
→ Start the missing service (T1/T2/T3)

### "Import agent_service error"  
→ Check you're in the correct directory and venv is activated

### "Test suite not found"
→ Use `--test-suite` with full path, or ensure `test_suite.json` exists

### Timeout errors
→ Services may be slow, increase timeout in `langchain_agent/config.json`

## Example Run Time

- 5 tests × 5 configurations = 25 runs
- ~30-60 seconds per run  
- **Total: ~12-25 minutes**

## Quick Results Check

```bash
# View latest run
cd runs
ls -t | head -1  # Linux/Mac
dir | sort -desc | select -first 1  # PowerShell

# Quick metrics summary
python -c "import pandas as pd; df = pd.read_csv('runs/LATEST_RUN/metrics.csv'); print(df.groupby(['model', 'rag_enabled', 'loopback_enabled'])['commands_f1', 'batfish_pass'].mean())"
```

## Next Steps After Benchmarking

1. **Analyze results**: `python benchmark_analysis.py runs/TIMESTAMP`
2. **Compare configs**: Look at summary statistics in terminal output
3. **Investigate failures**: Check `raw_responses.jsonl` for failed tests
4. **Iterate**: Adjust models/prompts and re-run

## Need Help?

- Full documentation: `README.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Setup test: `python test_setup.py`
