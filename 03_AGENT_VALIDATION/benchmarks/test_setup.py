#!/usr/bin/env python3
"""
Quick test script to verify benchmarking.py setup
"""
import sys
from pathlib import Path

# Add agent service to path
SCRIPT_DIR = Path(__file__).parent
AGENT_DIR = SCRIPT_DIR.parent / "langchain_agent"
sys.path.insert(0, str(AGENT_DIR))

print("Testing imports...")

try:
    from agent_service import call_t2_generate, call_t3_validate, verdict_chain, _parse_verdict_status
    print("✓ Agent service imports successful")
except ImportError as e:
    print(f"✗ Failed to import agent service: {e}")
    sys.exit(1)

try:
    import json
    import csv
    import logging
    from datetime import datetime
    print("✓ Standard library imports successful")
except ImportError as e:
    print(f"✗ Failed to import standard libraries: {e}")
    sys.exit(1)

# Test test_suite.json exists
test_suite_path = SCRIPT_DIR / "test_suite.json"
if test_suite_path.exists():
    print(f"✓ Test suite found: {test_suite_path}")
    with open(test_suite_path) as f:
        data = json.load(f)
    print(f"  - Test suite name: {data.get('test_suite_name')}")
    print(f"  - Number of tests: {len(data.get('tests', []))}")
else:
    print(f"✗ Test suite not found: {test_suite_path}")
    sys.exit(1)

# Test output directory creation
output_dir = SCRIPT_DIR / "runs"
output_dir.mkdir(exist_ok=True)
print(f"✓ Output directory ready: {output_dir}")

print("\n" + "="*60)
print("All checks passed! Ready to run benchmarks.")
print("="*60)
print("\nRun benchmarks with:")
print("  python benchmarking.py")
print("  python benchmarking.py --verbose")
