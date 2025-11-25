#!/usr/bin/env python3
"""
Benchmarking Pipeline for LLM Network Configuration Generation

This script evaluates multiple LLM models with and without RAG against a test suite
of network configuration queries. It computes various metrics including exact match,
command similarity, intent matching, and Batfish validation.

Usage:
    python benchmarking.py
    python benchmarking.py --test-suite custom_tests.json
    python benchmarking.py --output-dir ./custom_runs
"""

import argparse
import csv
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
AGENT_DIR = SCRIPT_DIR.parent / "langchain_agent"
sys.path.insert(0, str(AGENT_DIR))

# Import agent service functions
from agent_service import (
    call_t2_generate,
    call_t3_validate,
    verdict_chain,
    _parse_verdict_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def call_model(model_name: str, query: str, rag_enabled: bool, loopback_enabled: bool = False) -> Dict[str, Any]:
    """
    Call an LLM model with a given query and RAG mode using the agent service.
    
    Args:
        model_name: Name of the model to call (e.g., "gemini", "llama")
        query: The network configuration query to send to the model
        rag_enabled: Whether to enable RAG (Retrieval-Augmented Generation)
        loopback_enabled: Whether to enable loopback fallback on failure
    
    Returns:
        Dict containing model response, validation, and verdict:
        {
            "model": str,
            "response": List[Dict],
            "rag_enabled": bool,
            "loopback_enabled": bool,
            "validation": Dict,
            "verdict": Dict,
            "config": str,
            "error": Optional[str]
        }
    """
    logger.info(f"Calling model '{model_name}' with RAG={'ON' if rag_enabled else 'OFF'}, Loopback={'ON' if loopback_enabled else 'OFF'}")
    logger.debug(f"Query: {query}")
    
    try:
        # Step 1: Generate configuration using T2
        gen_result = call_t2_generate(query, model=model_name, rag_enabled=rag_enabled)
        
        # Step 2: Validate with T3
        validation = call_t3_validate(gen_result["evaluate_payload"])
        
        # Step 3: Get AI verdict
        verdict_text = verdict_chain.invoke({
            "query": query,
            "config": gen_result["joined_config"],
            "validation_json": json.dumps(validation, ensure_ascii=False)
        })
        verdict_status = _parse_verdict_status(verdict_text)
        
        # Step 4: Loopback retry if enabled and failed
        loopback_attempted = False
        if loopback_enabled and verdict_status == "FAIL":
            logger.info(f"Loopback triggered for {model_name} - retrying without RAG")
            loopback_attempted = True
            
            # Retry with RAG disabled
            gen_result = call_t2_generate(query, model=model_name, rag_enabled=False)
            validation = call_t3_validate(gen_result["evaluate_payload"])
            verdict_text = verdict_chain.invoke({
                "query": query,
                "config": gen_result["joined_config"],
                "validation_json": json.dumps(validation, ensure_ascii=False)
            })
            verdict_status = _parse_verdict_status(verdict_text)
        
        # Parse response from T2 result
        response = gen_result.get("evaluate_payload", {}).get("changes", {})
        response_list = []
        for device_name, commands in response.items():
            response_list.append({
                "device_name": device_name,
                "configuration_mode_commands": commands,
                "protocol": "GENERATED",
                "intent": []
            })
        
        return {
            "model": model_name,
            "response": response_list,
            "rag_enabled": rag_enabled,
            "loopback_enabled": loopback_enabled,
            "loopback_attempted": loopback_attempted,
            "validation": validation,
            "verdict": {
                "text": verdict_text,
                "status": verdict_status
            },
            "config": gen_result["joined_config"],
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error calling model {model_name}: {e}")
        return {
            "model": model_name,
            "response": [],
            "rag_enabled": rag_enabled,
            "loopback_enabled": loopback_enabled,
            "loopback_attempted": False,
            "validation": {"result": "ERROR", "error": str(e)},
            "verdict": {
                "text": f"Error: {str(e)}",
                "status": "FAIL"
            },
            "config": "",
            "error": str(e)
        }


def normalize_commands(commands: List[str]) -> List[str]:
    """
    Normalize command strings for comparison by stripping whitespace.
    
    Args:
        commands: List of configuration commands
    
    Returns:
        List of normalized commands
    """
    return [cmd.strip() for cmd in commands if cmd.strip()]


def compute_command_metrics(expected_commands: List[str], 
                           predicted_commands: List[str]) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for configuration commands.
    
    Args:
        expected_commands: Ground truth commands
        predicted_commands: Commands generated by the model
    
    Returns:
        Dict with precision, recall, and f1 scores
    """
    expected_set = set(normalize_commands(expected_commands))
    predicted_set = set(normalize_commands(predicted_commands))
    
    if not predicted_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if not expected_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    true_positives = len(expected_set & predicted_set)
    
    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def extract_all_commands(response: List[Dict]) -> List[str]:
    """
    Extract all configuration commands from a response.
    
    Args:
        response: List of device configuration dictionaries
    
    Returns:
        Flattened list of all commands
    """
    commands = []
    for device in response:
        if "configuration_mode_commands" in device:
            commands.extend(device["configuration_mode_commands"])
    return commands


def compare_intents(expected_intents: List[Dict], 
                   predicted_intents: List[Dict]) -> int:
    """
    Compare intent blocks between expected and predicted responses.
    
    This is a simplified comparison. For production, you may want more
    sophisticated matching logic.
    
    Args:
        expected_intents: Ground truth intent blocks
        predicted_intents: Predicted intent blocks
    
    Returns:
        1 if intents match, 0 otherwise
    """
    # Normalize intents for comparison
    def normalize_intent(intent: Dict) -> str:
        # Create a comparable string representation
        return json.dumps({
            "type": intent.get("type", ""),
            "service": intent.get("service", ""),
            "requirements": sorted(intent.get("requirements", []))
        }, sort_keys=True)
    
    expected_normalized = {normalize_intent(i) for i in expected_intents}
    predicted_normalized = {normalize_intent(i) for i in predicted_intents}
    
    # Check if all expected intents are present and no extra ones
    return 1 if expected_normalized == predicted_normalized else 0


def extract_all_intents(response: List[Dict]) -> List[Dict]:
    """
    Extract all intent blocks from a response.
    
    Args:
        response: List of device configuration dictionaries
    
    Returns:
        Flattened list of all intent blocks
    """
    intents = []
    for device in response:
        if "intent" in device:
            intents.extend(device["intent"])
    return intents


def compute_exact_match(expected: Dict, predicted: Dict) -> int:
    """
    Compute exact match between expected and predicted responses.
    
    Args:
        expected: Expected response dict
        predicted: Predicted response dict
    
    Returns:
        1 if exact match (after normalization), 0 otherwise
    """
    # Normalize both responses for comparison
    def normalize_response(resp: Dict) -> str:
        # Remove model-specific fields and normalize
        normalized = []
        for device in resp.get("response", []):
            normalized_device = {
                "device_name": device.get("device_name", ""),
                "configuration_mode_commands": normalize_commands(
                    device.get("configuration_mode_commands", [])
                ),
                "protocol": device.get("protocol", ""),
                "intent": device.get("intent", [])
            }
            normalized.append(normalized_device)
        return json.dumps(normalized, sort_keys=True)
    
    try:
        expected_normalized = normalize_response(expected)
        predicted_normalized = normalize_response(predicted)
        return 1 if expected_normalized == predicted_normalized else 0
    except Exception as e:
        logger.warning(f"Error during exact match comparison: {e}")
        return 0


def validate_with_batfish(test_id: str, 
                         model_output: Dict, 
                         expected: Dict) -> Dict[str, Any]:
    """
    Extract Batfish validation results from model output.
    
    Since we're calling the full pipeline (T2+T3), the validation is already
    performed and stored in model_output["validation"].
    
    Args:
        test_id: Test case identifier
        model_output: Model's generated output including validation
        expected: Expected configuration from test suite
    
    Returns:
        Dict with Batfish validation results:
        {
            "batfish_pass": bool,
            "violations": List,
            "details": Dict,
            "summary": Dict
        }
    """
    logger.info(f"Extracting Batfish validation for test {test_id}")
    
    validation = model_output.get("validation", {})
    
    # Determine pass/fail from validation result
    result = validation.get("result", "UNKNOWN")
    batfish_pass = (result == "OK")
    
    # Extract violations and details
    summary = validation.get("summary", {})
    details = {
        "result": result,
        "summary": summary,
        "control_plane": summary.get("CP", {}),
        "topology": summary.get("TP", {}),
        "reachability": summary.get("REACH", [])
    }
    
    # Collect violations
    violations = []
    if not batfish_pass:
        # Check control plane
        cp = summary.get("CP", {})
        if cp.get("status") != "PASS":
            violations.append({"type": "control_plane", "message": cp.get("error", "Failed")})
        
        # Check topology
        tp = summary.get("TP", {})
        if tp.get("status") != "PASS":
            violations.append({"type": "topology", "message": tp.get("error", "Failed")})
        
        # Check reachability
        for reach in summary.get("REACH", []):
            if reach.get("status") != "PASS":
                violations.append({
                    "type": "reachability",
                    "message": f"{reach.get('src')} -> {reach.get('dst')}: {reach.get('error', 'Failed')}"
                })
    
    return {
        "batfish_pass": batfish_pass,
        "violations": violations,
        "details": details,
        "summary": summary
    }


def compute_metrics(test_case: Dict, 
                   model_output: Dict, 
                   batfish_result: Dict) -> Dict[str, Any]:
    """
    Compute all metrics for a single test case and model output.
    
    Args:
        test_case: Test case from test suite
        model_output: Model's generated output
        batfish_result: Results from Batfish validation
    
    Returns:
        Dict containing all computed metrics
    """
    expected = test_case["expected"]
    
    # Extract commands
    expected_commands = extract_all_commands(expected.get("response", []))
    predicted_commands = extract_all_commands(model_output.get("response", []))
    
    # Extract intents
    expected_intents = extract_all_intents(expected.get("response", []))
    predicted_intents = extract_all_intents(model_output.get("response", []))
    
    # Compute metrics
    exact_match = compute_exact_match(expected, model_output)
    command_metrics = compute_command_metrics(expected_commands, predicted_commands)
    intent_match = compare_intents(expected_intents, predicted_intents)
    
    # Extract verdict information
    verdict = model_output.get("verdict", {})
    verdict_status = verdict.get("status", "UNKNOWN")
    verdict_text = verdict.get("text", "")
    
    # Extract validation details
    validation = model_output.get("validation", {})
    validation_result = validation.get("result", "UNKNOWN")
    validation_status = validation.get("status", "UNKNOWN")
    
    metrics = {
        "test_id": test_case["id"],
        "query": test_case["query"],
        "model": model_output.get("model", "unknown"),
        "rag_enabled": model_output.get("rag_enabled", False),
        "loopback_enabled": model_output.get("loopback_enabled", False),
        "loopback_attempted": model_output.get("loopback_attempted", False),
        "exact_match": exact_match,
        "commands_precision": command_metrics["precision"],
        "commands_recall": command_metrics["recall"],
        "commands_f1": command_metrics["f1"],
        "intent_match": intent_match,
        "batfish_pass": 1 if batfish_result.get("batfish_pass", False) else 0,
        "batfish_violations": len(batfish_result.get("violations", [])),
        "batfish_result": validation_result,
        "batfish_status": validation_status,
        "ai_verdict": verdict_status,
        "ai_verdict_text": verdict_text,
        "has_error": 1 if model_output.get("error") else 0,
        "error_message": model_output.get("error", ""),
        "timestamp": datetime.now().isoformat()
    }
    
    return metrics


def create_output_directory(base_dir: str) -> Path:
    """
    Create a timestamped output directory.
    
    Args:
        base_dir: Base directory for output
    
    Returns:
        Path to the created output directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def write_config(output_dir: Path, 
                config_data: Dict[str, Any]) -> None:
    """
    Write run configuration metadata to config.json.
    
    Args:
        output_dir: Output directory path
        config_data: Configuration data to write
    """
    config_path = output_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Wrote configuration to {config_path}")


def append_raw_response(output_dir: Path, 
                       test_id: str,
                       model: str,
                       rag_enabled: bool,
                       loopback_enabled: bool,
                       query: str,
                       model_output: Dict,
                       expected: Dict) -> None:
    """
    Append a raw response entry to raw_responses.jsonl.
    
    Args:
        output_dir: Output directory path
        test_id: Test case ID
        model: Model name
        rag_enabled: Whether RAG was enabled
        loopback_enabled: Whether loopback was enabled
        query: Input query
        model_output: Model's output
        expected: Expected output
    """
    jsonl_path = output_dir / "raw_responses.jsonl"
    entry = {
        "test_id": test_id,
        "model": model,
        "rag_enabled": rag_enabled,
        "loopback_enabled": loopback_enabled,
        "loopback_attempted": model_output.get("loopback_attempted", False),
        "query": query,
        "model_output": {
            "response": model_output.get("response", []),
            "config": model_output.get("config", ""),
            "error": model_output.get("error")
        },
        "validation": model_output.get("validation", {}),
        "verdict": model_output.get("verdict", {}),
        "expected": expected
    }
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def write_metrics(output_dir: Path, 
                 metrics_list: List[Dict[str, Any]]) -> None:
    """
    Write metrics to both CSV and JSON files.
    
    Args:
        output_dir: Output directory path
        metrics_list: List of metric dictionaries
    """
    if not metrics_list:
        logger.warning("No metrics to write")
        return
    
    # Write CSV
    csv_path = output_dir / "metrics.csv"
    fieldnames = list(metrics_list[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)
    logger.info(f"Wrote metrics CSV to {csv_path}")
    
    # Write JSON
    json_path = output_dir / "metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_list, f, indent=2)
    logger.info(f"Wrote metrics JSON to {json_path}")


def load_test_suite(test_suite_path: str) -> Dict[str, Any]:
    """
    Load test suite from JSON file.
    
    Args:
        test_suite_path: Path to test suite JSON file
    
    Returns:
        Parsed test suite dictionary
    
    Raises:
        FileNotFoundError: If test suite file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not os.path.exists(test_suite_path):
        raise FileNotFoundError(f"Test suite not found: {test_suite_path}")
    
    with open(test_suite_path, 'r', encoding='utf-8') as f:
        test_suite = json.load(f)
    
    logger.info(f"Loaded test suite: {test_suite.get('test_suite_name', 'unknown')}")
    logger.info(f"Number of tests: {len(test_suite.get('tests', []))}")
    
    return test_suite


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Benchmark LLM models for network configuration generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarking.py
  python benchmarking.py --test-suite custom_tests.json
  python benchmarking.py --output-dir ./results
  
Test Configurations:
  The script automatically runs the following benchmarks:
  1. Gemini with RAG
  2. Gemini without RAG
  3. Llama with RAG
  4. Llama without RAG (no loopback)
  5. Llama with RAG + loopback enabled
        """
    )
    
    parser.add_argument(
        '--test-suite',
        type=str,
        default=str(SCRIPT_DIR / 'test_suite.json'),
        help='Path to test suite JSON file (default: ./test_suite.json in benchmarks folder)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(SCRIPT_DIR / 'runs'),
        help='Base output directory for results (default: ./runs)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main execution function for the benchmarking pipeline.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("LLM Network Configuration Benchmarking Pipeline")
    logger.info("=" * 80)
    
    # Global variables for signal handler
    global output_dir, all_metrics
    output_dir = None
    all_metrics = []
    
    # Define benchmark configurations
    # Format: (model_name, rag_enabled, loopback_enabled, description)
    benchmark_configs = [
        ("gemini", True, False, "Gemini with RAG"),
        ("gemini", False, False, "Gemini without RAG"),
        ("llama", True, False, "Llama with RAG (no loopback)"),
        ("llama", False, False, "Llama without RAG (no loopback)"),
        ("llama", True, True, "Llama with RAG + loopback"),
    ]
    
    logger.info("Benchmark configurations:")
    for idx, (model, rag, loopback, desc) in enumerate(benchmark_configs, 1):
        logger.info(f"  {idx}. {desc}")
    
    logger.info(f"Test suite: {args.test_suite}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load test suite
    try:
        test_suite = load_test_suite(args.test_suite)
    except Exception as e:
        logger.error(f"Failed to load test suite: {e}")
        return 1
    
    # Create output directory
    try:
        output_dir = create_output_directory(args.output_dir)
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return 1
    
    # Write run configuration
    config_data = {
        "test_suite_name": test_suite.get("test_suite_name", "unknown"),
        "description": test_suite.get("description", ""),
        "benchmark_configs": [
            {"model": m, "rag_enabled": r, "loopback_enabled": l, "description": d}
            for m, r, l, d in benchmark_configs
        ],
        "test_suite_path": os.path.abspath(args.test_suite),
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(test_suite.get("tests", []))
    }
    write_config(output_dir, config_data)
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.warning("\n\nInterrupt received! Saving metrics...")
        if all_metrics:
            try:
                write_metrics(output_dir, all_metrics)
                logger.info(f"Saved {len(all_metrics)} metrics before exit")
            except Exception as e:
                logger.error(f"Failed to save metrics on interrupt: {e}")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run benchmarks
    tests = test_suite.get("tests", [])
    total_runs = len(tests) * len(benchmark_configs)
    current_run = 0
    
    logger.info(f"Starting benchmark with {total_runs} total runs")
    logger.info("=" * 80)
    
    for test_case in tests:
        test_id = test_case["id"]
        query = test_case["query"]
        expected = test_case["expected"]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing test: {test_id}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*80}")
        
        for model, rag_enabled, loopback_enabled, description in benchmark_configs:
            current_run += 1
            logger.info(f"\n[{current_run}/{total_runs}] {description}")
            logger.info(f"  Model: {model} | RAG: {rag_enabled} | Loopback: {loopback_enabled}")
            
            try:
                # Call the model with full pipeline
                model_output = call_model(model, query, rag_enabled, loopback_enabled)
                
                # Extract Batfish validation
                batfish_result = validate_with_batfish(test_id, model_output, expected)
                
                # Compute metrics
                metrics = compute_metrics(test_case, model_output, batfish_result)
                all_metrics.append(metrics)
                
                # Log summary
                logger.info(f"  ✓ Exact match: {metrics['exact_match']}")
                logger.info(f"  ✓ Commands F1: {metrics['commands_f1']:.3f}")
                logger.info(f"  ✓ Intent match: {metrics['intent_match']}")
                logger.info(f"  ✓ Batfish pass: {metrics['batfish_pass']}")
                logger.info(f"  ✓ AI verdict: {metrics['ai_verdict']}")
                if metrics.get('loopback_attempted'):
                    logger.info(f"  ⚠ Loopback was attempted")
                if metrics.get('has_error'):
                    logger.info(f"  ✗ Error occurred during processing")
                
                # Save raw response
                append_raw_response(
                    output_dir,
                    test_id,
                    model,
                    rag_enabled,
                    loopback_enabled,
                    query,
                    model_output,
                    expected
                )
                
            except Exception as e:
                logger.error(f"Error processing {test_id} with {description}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Create error metrics entry
                error_metrics = {
                    "test_id": test_id,
                    "query": query,
                    "model": model,
                    "rag_enabled": rag_enabled,
                    "loopback_enabled": loopback_enabled,
                    "loopback_attempted": False,
                    "exact_match": 0,
                    "commands_precision": 0.0,
                    "commands_recall": 0.0,
                    "commands_f1": 0.0,
                    "intent_match": 0,
                    "batfish_pass": 0,
                    "batfish_violations": 0,
                    "batfish_result": "ERROR",
                    "batfish_status": "ERROR",
                    "ai_verdict": "ERROR",
                    "ai_verdict_text": f"Error: {str(e)}",
                    "has_error": 1,
                    "error_message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                all_metrics.append(error_metrics)
                continue
        
        # Write incremental metrics after each test (all configs for this test)
        logger.info(f"Saving incremental metrics ({len(all_metrics)} total so far)...")
        try:
            write_metrics(output_dir, all_metrics)
        except Exception as e:
            logger.warning(f"Failed to write incremental metrics: {e}")
    
    # Write final metrics
    logger.info("=" * 80)
    logger.info("Writing final metrics")
    try:
        write_metrics(output_dir, all_metrics)
    except Exception as e:
        logger.error(f"Failed to write metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Try to save what we can
        try:
            import json
            emergency_file = output_dir / "metrics_emergency.json"
            with open(emergency_file, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"Saved emergency metrics to {emergency_file}")
        except Exception as e2:
            logger.error(f"Emergency save also failed: {e2}")
    
    # Compute and log summary statistics by configuration
    if all_metrics:
        logger.info("=" * 80)
        logger.info("SUMMARY STATISTICS BY CONFIGURATION")
        logger.info("=" * 80)
        
        for model, rag, loopback, desc in benchmark_configs:
            config_metrics = [
                m for m in all_metrics 
                if m['model'] == model 
                and m['rag_enabled'] == rag 
                and m['loopback_enabled'] == loopback
            ]
            
            if config_metrics:
                avg_exact = sum(m['exact_match'] for m in config_metrics) / len(config_metrics)
                avg_f1 = sum(m['commands_f1'] for m in config_metrics) / len(config_metrics)
                avg_intent = sum(m['intent_match'] for m in config_metrics) / len(config_metrics)
                avg_batfish = sum(m['batfish_pass'] for m in config_metrics) / len(config_metrics)
                pass_verdicts = sum(1 for m in config_metrics if m['ai_verdict'] == 'PASS')
                loopback_attempts = sum(1 for m in config_metrics if m.get('loopback_attempted', False))
                errors = sum(1 for m in config_metrics if m.get('has_error', 0))
                
                logger.info(f"\n{desc}:")
                logger.info(f"  Runs completed: {len(config_metrics)}")
                logger.info(f"  Avg exact match: {avg_exact:.2%}")
                logger.info(f"  Avg commands F1: {avg_f1:.2%}")
                logger.info(f"  Avg intent match: {avg_intent:.2%}")
                logger.info(f"  Avg Batfish pass: {avg_batfish:.2%}")
                logger.info(f"  AI verdict PASS: {pass_verdicts}/{len(config_metrics)}")
                if loopback_attempts > 0:
                    logger.info(f"  Loopback attempts: {loopback_attempts}")
                if errors > 0:
                    logger.info(f"  Errors: {errors}")
        
        logger.info("=" * 80)
        logger.info(f"Total runs completed: {len(all_metrics)}/{total_runs}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
    
    logger.info("Benchmarking complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())