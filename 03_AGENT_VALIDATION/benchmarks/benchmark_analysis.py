#!/usr/bin/env python3
"""
Benchmark Analysis Script - RAG Performance Analysis

Analyzes RAG and chunking performance using:
- Global context: 02_LLM_INFERENCE_API/models/context.json
- Run metrics: 03_AGENT_VALIDATION/benchmarks/runs/<run-id>/metrics.json

Generates aggregations, comparisons, plots, and summary reports.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import statistics

# Try to import pandas, fall back to manual processing if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using fallback aggregation methods")

# Try to import sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("Warning: sentence-transformers not available, using Levenshtein distance fallback")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze RAG/chunking performance from benchmark runs"
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to run directory containing metrics.json'
    )
    parser.add_argument(
        '--context-path',
        type=str,
        default=None,
        help='Optional path to context.json (default: auto-detect from repo root)'
    )
    return parser.parse_args()


def find_repo_root(start_path: Path) -> Optional[Path]:
    """Find repository root by looking for common markers."""
    current = start_path.resolve()
    while current != current.parent:
        # Check for common repo markers
        if (current / '.git').exists() or \
           (current / 'README.md').exists() and \
           (current / '02_LLM_INFERENCE_API').exists():
            return current
        current = current.parent
    return None


def load_context(context_path: Optional[str], run_dir: Path) -> Dict[str, Any]:
    """Load global context.json with robust path resolution."""
    if context_path:
        ctx_path = Path(context_path)
    else:
        # Try to find repo root
        repo_root = find_repo_root(run_dir)
        if repo_root:
            ctx_path = repo_root / '02_LLM_INFERENCE_API' / 'models' / 'context.json'
        else:
            # Fallback: assume run_dir is under 03_AGENT_VALIDATION/benchmarks/runs/
            ctx_path = run_dir.parent.parent.parent.parent / '02_LLM_INFERENCE_API' / 'models' / 'context.json'
    
    if not ctx_path.exists():
        print(f"Warning: context.json not found at {ctx_path}")
        return {}
    
    with open(ctx_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    """Load metrics.json from run directory and filter out 502 errors."""
    metrics_path = run_dir / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {run_dir}")
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        all_metrics = json.load(f)
    
    # Filter out 502 errors and other error blocks
    filtered_metrics = []
    error_count = 0
    for m in all_metrics:
        # Check for 502 errors in error_message or ai_verdict_text
        has_502_error = False
        if m.get('error_message') and '502' in str(m.get('error_message')):
            has_502_error = True
        if m.get('ai_verdict_text') and 'Error: 502' in str(m.get('ai_verdict_text')):
            has_502_error = True
        
        if has_502_error:
            error_count += 1
            continue
        
        # Add ai_verdict_pass field based on ai_verdict (global failure metric)
        m['ai_verdict_pass'] = 1 if m.get('ai_verdict') == 'PASS' else 0
        
        # Fix batfish_pass to use batfish_status instead of batfish_result
        m['batfish_pass'] = 1 if m.get('batfish_status') == 'PASS' else 0
        
        filtered_metrics.append(m)
    
    if error_count > 0:
        print(f"  ⚠ Filtered out {error_count} entries with 502 errors")
    
    return filtered_metrics


def extract_chunking_info(context: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Extract chunking and RAG settings from context.json and retrieval_config.json."""
    info = {
        'chunk_size': 'unknown',
        'chunk_overlap': 'unknown',
        'retriever_top_k': 'unknown',
        'retriever_type': 'unknown',
        'avg_chunks_retrieved': 0,
        'avg_code_chunks': 0,
        'avg_theory_chunks': 0,
        'avg_context_length': 0,
        'avg_quality_score': 0.0
    }
    
    # Try to load retrieval_config.json from 02_LLM_INFERENCE_API
    repo_root = find_repo_root(run_dir)
    if repo_root:
        config_path = repo_root / '02_LLM_INFERENCE_API' / 'models' / 'retrieval_config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                info['chunk_size'] = config.get('max_chunk_length', 'unknown')
                info['retriever_top_k'] = config.get('default_chunks', 'unknown')
    
    # Extract stats from context.json history
    if 'history' in context and len(context['history']) > 0:
        chunks_retrieved = []
        code_chunks = []
        theory_chunks = []
        context_lengths = []
        quality_scores = []
        
        for entry in context['history']:
            if 'retrieval_plan' in entry:
                plan = entry['retrieval_plan']
                if 'total_chunks' in plan:
                    chunks_retrieved.append(plan['total_chunks'])
                if 'code_chunks' in plan:
                    code_chunks.append(plan['code_chunks'])
                if 'theory_chunks' in plan:
                    theory_chunks.append(plan['theory_chunks'])
                if 'quality_score' in plan:
                    quality_scores.append(plan['quality_score'])
            
            if 'context_length' in entry:
                context_lengths.append(entry['context_length'])
        
        if chunks_retrieved:
            info['avg_chunks_retrieved'] = statistics.mean(chunks_retrieved)
        if code_chunks:
            info['avg_code_chunks'] = statistics.mean(code_chunks)
        if theory_chunks:
            info['avg_theory_chunks'] = statistics.mean(theory_chunks)
        if context_lengths:
            info['avg_context_length'] = statistics.mean(context_lengths)
        if quality_scores:
            info['avg_quality_score'] = statistics.mean(quality_scores)
        
        info['retriever_type'] = 'hybrid_code_theory'
    
    return info


def aggregate_metrics_pandas(metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate metrics using pandas."""
    df = pd.DataFrame(metrics)
    
    # Filter out error rows
    df = df[df['has_error'] == 0].copy()
    
    # Group by model and rag_enabled
    group_cols = ['model', 'rag_enabled']
    
    agg_dict = {
        'exact_match': ['mean', 'std', 'count'],
        'commands_precision': ['mean', 'std'],
        'commands_recall': ['mean', 'std'],
        'commands_f1': ['mean', 'std'],
        'intent_match': ['mean', 'std'],
        'batfish_pass': ['mean', 'std'],
        'ai_verdict_pass': ['mean', 'std'],
    }
    
    # Check for optional fields
    if 'latency_ms' in df.columns:
        agg_dict['latency_ms'] = ['mean', 'std']
    if 'tokens_prompt' in df.columns:
        agg_dict['tokens_prompt'] = ['mean', 'std']
    if 'tokens_completion' in df.columns:
        agg_dict['tokens_completion'] = ['mean', 'std']
    
    aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                          for col in aggregated.columns.values]
    
    return aggregated


def aggregate_metrics_manual(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate metrics without pandas (fallback)."""
    # Filter out errors
    clean_metrics = [m for m in metrics if m.get('has_error', 0) == 0]
    
    # Group by (model, rag_enabled)
    groups = defaultdict(list)
    for m in clean_metrics:
        key = (m['model'], m['rag_enabled'])
        groups[key].append(m)
    
    results = []
    for (model, rag_enabled), group in groups.items():
        def safe_mean(field):
            values = [m[field] for m in group if field in m and m[field] is not None]
            return statistics.mean(values) if values else 0.0
        
        def safe_std(field):
            values = [m[field] for m in group if field in m and m[field] is not None]
            return statistics.stdev(values) if len(values) > 1 else 0.0
        
        result = {
            'model': model,
            'rag_enabled': rag_enabled,
            'exact_match_mean': safe_mean('exact_match'),
            'commands_precision_mean': safe_mean('commands_precision'),
            'commands_recall_mean': safe_mean('commands_recall'),
            'commands_f1_mean': safe_mean('commands_f1'),
            'commands_f1_std': safe_std('commands_f1'),
            'intent_match_mean': safe_mean('intent_match'),
            'batfish_pass_mean': safe_mean('batfish_pass'),
            'ai_verdict_pass_mean': safe_mean('ai_verdict_pass'),
            'exact_match_count': len(group),
        }
        results.append(result)
    
    return results


def analyze_per_test(metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyze performance per test_id."""
    # Group by test_id
    test_groups = defaultdict(list)
    for m in metrics:
        if m.get('has_error', 0) == 0:
            test_groups[m['test_id']].append(m)
    
    test_results = {}
    for test_id, group in test_groups.items():
        def safe_mean(field):
            values = [m[field] for m in group if field in m and m[field] is not None]
            return statistics.mean(values) if values else 0.0
        
        # Find best model/rag combination
        best_f1 = -1
        best_combo = None
        for m in group:
            if m['commands_f1'] > best_f1:
                best_f1 = m['commands_f1']
                best_combo = f"{m['model']}_rag{'on' if m['rag_enabled'] else 'off'}"
        
        test_results[test_id] = {
            'avg_commands_f1': safe_mean('commands_f1'),
            'avg_batfish_pass': safe_mean('batfish_pass'),
            'avg_ai_verdict_pass': safe_mean('ai_verdict_pass'),
            'avg_exact_match': safe_mean('exact_match'),
            'best_combo': best_combo,
            'best_f1': best_f1,
            'num_runs': len(group)
        }
    
    return test_results


def find_hardest_easiest_tests(test_results: Dict[str, Dict[str, Any]], n: int = 5) -> Tuple[List, List]:
    """Find n hardest and easiest tests based on avg_commands_f1."""
    sorted_tests = sorted(test_results.items(), key=lambda x: x[1]['avg_commands_f1'])
    hardest = sorted_tests[:n]
    easiest = sorted_tests[-n:][::-1]  # Reverse to show best first
    return hardest, easiest


def compute_rag_comparison(aggregated: Any) -> Dict[str, Any]:
    """Compare RAG on vs off for each model."""
    comparisons = {}
    
    if HAS_PANDAS:
        models = aggregated['model'].unique()
        for model in models:
            rag_on = aggregated[(aggregated['model'] == model) & (aggregated['rag_enabled'] == True)]
            rag_off = aggregated[(aggregated['model'] == model) & (aggregated['rag_enabled'] == False)]
            
            if len(rag_on) > 0 and len(rag_off) > 0:
                comparisons[model] = {
                    'f1_delta': float(rag_on['commands_f1_mean'].iloc[0] - rag_off['commands_f1_mean'].iloc[0]),
                    'exact_match_delta': float(rag_on['exact_match_mean'].iloc[0] - rag_off['exact_match_mean'].iloc[0]),
                    'batfish_pass_delta': float(rag_on['batfish_pass_mean'].iloc[0] - rag_off['batfish_pass_mean'].iloc[0]),
                    'ai_verdict_pass_delta': float(rag_on['ai_verdict_pass_mean'].iloc[0] - rag_off['ai_verdict_pass_mean'].iloc[0]),
                    'f1_rag_on': float(rag_on['commands_f1_mean'].iloc[0]),
                    'f1_rag_off': float(rag_off['commands_f1_mean'].iloc[0]),
                }
    else:
        # Manual grouping
        model_data = defaultdict(dict)
        for row in aggregated:
            model = row['model']
            rag_key = 'rag_on' if row['rag_enabled'] else 'rag_off'
            model_data[model][rag_key] = row
        
        for model, data in model_data.items():
            if 'rag_on' in data and 'rag_off' in data:
                comparisons[model] = {
                    'f1_delta': data['rag_on']['commands_f1_mean'] - data['rag_off']['commands_f1_mean'],
                    'exact_match_delta': data['rag_on']['exact_match_mean'] - data['rag_off']['exact_match_mean'],
                    'batfish_pass_delta': data['rag_on']['batfish_pass_mean'] - data['rag_off']['batfish_pass_mean'],
                    'ai_verdict_pass_delta': data['rag_on']['ai_verdict_pass_mean'] - data['rag_off']['ai_verdict_pass_mean'],
                    'f1_rag_on': data['rag_on']['commands_f1_mean'],
                    'f1_rag_off': data['rag_off']['commands_f1_mean'],
                }
    
    return comparisons


def compute_semantic_distance(text1: str, text2: str, model=None) -> float:
    """Compute semantic distance between two command strings using embeddings."""
    if not text1 or not text2:
        return 1.0
    
    if HAS_EMBEDDINGS and model is not None:
        try:
            # Generate embeddings
            embeddings = model.encode([text1, text2])
            # Compute cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            # Convert to distance (0=identical, 1=completely different)
            distance = 1.0 - float(similarity)
            return max(0.0, min(1.0, distance))
        except Exception as e:
            print(f"Warning: Embedding computation failed: {e}")
    
    # Fallback: Levenshtein-based distance
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 0.0
    
    # Simple character-level distance
    distance = sum(c1 != c2 for c1, c2 in zip(text1, text2))
    distance += abs(len(text1) - len(text2))
    return min(1.0, distance / max_len)


def load_raw_responses(run_dir: Path) -> List[Dict[str, Any]]:
    """Load raw_responses.jsonl with full model outputs."""
    raw_path = run_dir / 'raw_responses.jsonl'
    if not raw_path.exists():
        return []
    
    responses = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses


def create_output_dirs(run_dir: Path):
    """Create output directories."""
    analysis_dir = run_dir / 'analysis'
    plots_dir = analysis_dir / 'plots'
    analysis_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    return analysis_dir, plots_dir


def plot_commands_f1_by_model_rag(aggregated: Any, plots_dir: Path, chunking_info: Dict):
    """Grouped bar chart: mean_commands_f1 per model, rag_on vs rag_off."""
    plt.figure(figsize=(10, 6))
    
    if HAS_PANDAS:
        models = sorted(aggregated['model'].unique())
        rag_on_values = []
        rag_off_values = []
        
        for model in models:
            rag_on = aggregated[(aggregated['model'] == model) & (aggregated['rag_enabled'] == True)]
            rag_off = aggregated[(aggregated['model'] == model) & (aggregated['rag_enabled'] == False)]
            
            rag_on_values.append(float(rag_on['commands_f1_mean'].iloc[0]) if len(rag_on) > 0 else 0)
            rag_off_values.append(float(rag_off['commands_f1_mean'].iloc[0]) if len(rag_off) > 0 else 0)
    else:
        # Manual grouping
        model_groups = defaultdict(dict)
        for row in aggregated:
            model = row['model']
            rag_key = 'rag_on' if row['rag_enabled'] else 'rag_off'
            model_groups[model][rag_key] = row['commands_f1_mean']
        
        models = sorted(model_groups.keys())
        rag_on_values = [model_groups[m].get('rag_on', 0) for m in models]
        rag_off_values = [model_groups[m].get('rag_off', 0) for m in models]
    
    x = range(len(models))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], rag_on_values, width, label='RAG On', alpha=0.8)
    plt.bar([i + width/2 for i in x], rag_off_values, width, label='RAG Off', alpha=0.8)
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Commands F1 Score', fontsize=12, fontweight='bold')
    plt.title(f'Commands F1 Score by Model and RAG Status\nChunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=0)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'commands_f1_by_model_rag.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_batfish_pass_by_model_rag(aggregated: Any, plots_dir: Path, chunking_info: Dict):
    """Grouped bar chart: mean_batfish_pass per model, rag_on vs rag_off."""
    plt.figure(figsize=(10, 6))
    
    if HAS_PANDAS:
        models = sorted(aggregated['model'].unique())
        rag_on_values = []
        rag_off_values = []
        
        for model in models:
            rag_on = aggregated[(aggregated['model'] == model) & (aggregated['rag_enabled'] == True)]
            rag_off = aggregated[(aggregated['model'] == model) & (aggregated['rag_enabled'] == False)]
            
            rag_on_values.append(float(rag_on['batfish_pass_mean'].iloc[0]) if len(rag_on) > 0 else 0)
            rag_off_values.append(float(rag_off['batfish_pass_mean'].iloc[0]) if len(rag_off) > 0 else 0)
    else:
        model_groups = defaultdict(dict)
        for row in aggregated:
            model = row['model']
            rag_key = 'rag_on' if row['rag_enabled'] else 'rag_off'
            model_groups[model][rag_key] = row['batfish_pass_mean']
        
        models = sorted(model_groups.keys())
        rag_on_values = [model_groups[m].get('rag_on', 0) for m in models]
        rag_off_values = [model_groups[m].get('rag_off', 0) for m in models]
    
    x = range(len(models))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], rag_on_values, width, label='RAG On', alpha=0.8, color='#2ecc71')
    plt.bar([i + width/2 for i in x], rag_off_values, width, label='RAG Off', alpha=0.8, color='#e74c3c')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Batfish Pass Rate', fontsize=12, fontweight='bold')
    plt.title(f'Batfish Validation Pass Rate by Model and RAG Status\nChunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=0)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'batfish_pass_by_model_rag.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_ai_verdict_pass_by_model_rag(aggregated: Any, plots_dir: Path, chunking_info: Dict):
    """Grouped bar chart: mean_ai_verdict_pass per model, rag_on vs rag_off (GLOBAL FAILURE METRIC)."""
    plt.figure(figsize=(10, 6))
    
    if HAS_PANDAS:
        models = sorted(aggregated['model'].unique())
        rag_on_values = []
        rag_off_values = []
        
        for model in models:
            rag_on = aggregated[(aggregated['model'] == model) & (aggregated['rag_enabled'] == True)]
            rag_off = aggregated[(aggregated['model'] == model) & (aggregated['rag_enabled'] == False)]
            
            rag_on_values.append(float(rag_on['ai_verdict_pass_mean'].iloc[0]) if len(rag_on) > 0 else 0)
            rag_off_values.append(float(rag_off['ai_verdict_pass_mean'].iloc[0]) if len(rag_off) > 0 else 0)
    else:
        model_groups = defaultdict(dict)
        for row in aggregated:
            model = row['model']
            rag_key = 'rag_on' if row['rag_enabled'] else 'rag_off'
            model_groups[model][rag_key] = row['ai_verdict_pass_mean']
        
        models = sorted(model_groups.keys())
        rag_on_values = [model_groups[m].get('rag_on', 0) for m in models]
        rag_off_values = [model_groups[m].get('rag_off', 0) for m in models]
    
    x = range(len(models))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], rag_on_values, width, label='RAG On', alpha=0.8, color='#2ecc71')
    plt.bar([i + width/2 for i in x], rag_off_values, width, label='RAG Off', alpha=0.8, color='#e74c3c')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Mean AI Verdict Pass Rate', fontsize=12, fontweight='bold')
    plt.title(f'AI Verdict Pass Rate by Model and RAG Status (GLOBAL FAILURE METRIC)\nChunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=0)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'ai_verdict_pass_by_model_rag.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_exact_match_by_model_rag(aggregated: Any, plots_dir: Path, chunking_info: Dict):
    """Grouped bar chart: mean_exact_match per model, rag_on vs rag_off."""
    plt.figure(figsize=(10, 6))
    
    if HAS_PANDAS:
        models = sorted(aggregated['model'].unique())
        rag_on_values = []
        rag_off_values = []
        
        for model in models:
            rag_on = aggregated[(aggregated['model'] == model) & (aggregated['rag_enabled'] == True)]
            rag_off = aggregated[(aggregated['model'] == model) & (aggregated['rag_enabled'] == False)]
            
            rag_on_values.append(float(rag_on['exact_match_mean'].iloc[0]) if len(rag_on) > 0 else 0)
            rag_off_values.append(float(rag_off['exact_match_mean'].iloc[0]) if len(rag_off) > 0 else 0)
    else:
        model_groups = defaultdict(dict)
        for row in aggregated:
            model = row['model']
            rag_key = 'rag_on' if row['rag_enabled'] else 'rag_off'
            model_groups[model][rag_key] = row['exact_match_mean']
        
        models = sorted(model_groups.keys())
        rag_on_values = [model_groups[m].get('rag_on', 0) for m in models]
        rag_off_values = [model_groups[m].get('rag_off', 0) for m in models]
    
    x = range(len(models))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], rag_on_values, width, label='RAG On', alpha=0.8, color='#3498db')
    plt.bar([i + width/2 for i in x], rag_off_values, width, label='RAG Off', alpha=0.8, color='#95a5a6')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Exact Match Rate', fontsize=12, fontweight='bold')
    plt.title(f'Exact Match Rate by Model and RAG Status\nChunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=0)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'exact_match_by_model_rag.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_f1_vs_batfish_scatter(aggregated: Any, plots_dir: Path, chunking_info: Dict):
    """Scatter plot: commands_f1 vs batfish_pass comparing RAG on vs off."""
    plt.figure(figsize=(12, 8))
    
    if HAS_PANDAS:
        rag_on = aggregated[aggregated['rag_enabled'] == True]
        rag_off = aggregated[aggregated['rag_enabled'] == False]
        
        # RAG On points
        x_on = rag_on['commands_f1_mean'].values
        y_on = rag_on['batfish_pass_mean'].values
        labels_on = rag_on['model'].values
        
        # RAG Off points
        x_off = rag_off['commands_f1_mean'].values
        y_off = rag_off['batfish_pass_mean'].values
        labels_off = rag_off['model'].values
    else:
        rag_on_rows = [r for r in aggregated if r['rag_enabled']]
        rag_off_rows = [r for r in aggregated if not r['rag_enabled']]
        
        x_on = [r['commands_f1_mean'] for r in rag_on_rows]
        y_on = [r['batfish_pass_mean'] for r in rag_on_rows]
        labels_on = [r['model'] for r in rag_on_rows]
        
        x_off = [r['commands_f1_mean'] for r in rag_off_rows]
        y_off = [r['batfish_pass_mean'] for r in rag_off_rows]
        labels_off = [r['model'] for r in rag_off_rows]
    
    # Plot RAG On (large, filled)
    plt.scatter(x_on, y_on, s=400, alpha=0.7, c='#2ecc71', 
                marker='o', edgecolors='black', linewidths=2, label='RAG On')
    
    # Plot RAG Off (smaller, hollow)
    plt.scatter(x_off, y_off, s=250, alpha=0.7, c='#e74c3c', 
                marker='s', edgecolors='black', linewidths=2, label='RAG Off')
    
    # Add labels for RAG On
    for i, label in enumerate(labels_on):
        plt.annotate(label.upper(), (x_on[i], y_on[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#2ecc71', alpha=0.3))
    
    # Add labels for RAG Off
    for i, label in enumerate(labels_off):
        plt.annotate(label.upper(), (x_off[i], y_off[i]), 
                    xytext=(10, -15), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.3))
    
    # Draw diagonal line (F1 = Batfish)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='F1 = Batfish')
    
    plt.xlabel('Mean Commands F1 Score', fontsize=13, fontweight='bold')
    plt.ylabel('Mean Batfish Validation Pass Rate', fontsize=13, fontweight='bold')
    plt.title(f'Commands F1 vs Batfish Pass Rate\nRAG On (Green) vs RAG Off (Red)\nChunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}', 
              fontsize=14, fontweight='bold')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'commands_f1_vs_batfish_pass_ragon.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_f1_distribution(metrics: List[Dict[str, Any]], plots_dir: Path, chunking_info: Dict):
    """Boxplot: commands_f1 distribution per (model, rag_enabled)."""
    plt.figure(figsize=(12, 6))
    
    # Clean metrics
    clean = [m for m in metrics if m.get('has_error', 0) == 0]
    
    if HAS_PANDAS:
        df = pd.DataFrame(clean)
        df['model_rag'] = df['model'] + '_' + df['rag_enabled'].map({True: 'RAG_On', False: 'RAG_Off'})
        
        groups = df.groupby('model_rag')['commands_f1'].apply(list).to_dict()
        labels = sorted(groups.keys())
        data = [groups[label] for label in labels]
    else:
        # Manual grouping
        groups = defaultdict(list)
        for m in clean:
            key = f"{m['model']}_{'RAG_On' if m['rag_enabled'] else 'RAG_Off'}"
            groups[key].append(m['commands_f1'])
        
        labels = sorted(groups.keys())
        data = [groups[label] for label in labels]
    
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.xlabel('Model + RAG Status', fontsize=12, fontweight='bold')
    plt.ylabel('Commands F1 Score', fontsize=12, fontweight='bold')
    plt.title(f'Commands F1 Distribution\nChunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'commands_f1_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hardest_tests(hardest: List[Tuple[str, Dict]], plots_dir: Path):
    """Bar chart: 5 hardest tests by avg_commands_f1."""
    plt.figure(figsize=(12, 6))
    
    test_ids = [t[0] for t in hardest]
    f1_scores = [t[1]['avg_commands_f1'] for t in hardest]
    
    plt.barh(range(len(test_ids)), f1_scores, alpha=0.8, color='#e74c3c')
    plt.yticks(range(len(test_ids)), test_ids)
    plt.xlabel('Average Commands F1 Score', fontsize=12, fontweight='bold')
    plt.ylabel('Test ID', fontsize=12, fontweight='bold')
    plt.title('5 Hardest Tests (Lowest Avg F1 Score)', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'hardest_tests_commands_f1.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_rag_impact_comparison(aggregated: Any, plots_dir: Path, chunking_info: Dict, rag_comparison: Dict):
    """Visualization showing the impact of RAG on each model."""
    if not rag_comparison:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    models = sorted(rag_comparison.keys())
    f1_deltas = [rag_comparison[m]['f1_delta'] for m in models]
    batfish_deltas = [rag_comparison[m]['batfish_pass_delta'] for m in models]
    
    # Plot 1: F1 Delta
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in f1_deltas]
    bars1 = ax1.barh(range(len(models)), f1_deltas, alpha=0.8, color=colors)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_xlabel('F1 Score Change (RAG On - RAG Off)', fontsize=12, fontweight='bold')
    ax1.set_title('RAG Impact on F1 Score by Model', fontsize=14, fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, f1_deltas)):
        label_x = val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax1.text(label_x, i, f'{val:+.3f}', va='center', ha=ha, fontweight='bold')
    
    # Plot 2: Batfish Pass Delta
    colors2 = ['#2ecc71' if d > 0 else '#e74c3c' for d in batfish_deltas]
    bars2 = ax2.barh(range(len(models)), batfish_deltas, alpha=0.8, color=colors2)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models)
    ax2.set_xlabel('Batfish Pass Rate Change (RAG On - RAG Off)', fontsize=12, fontweight='bold')
    ax2.set_title('RAG Impact on Validation Pass Rate', fontsize=14, fontweight='bold')
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, batfish_deltas)):
        label_x = val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax2.text(label_x, i, f'{val:+.3f}', va='center', ha=ha, fontweight='bold')
    
    plt.suptitle(f'RAG Impact Analysis\nChunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / 'rag_impact_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_average_f1_by_model(aggregated: Any, plots_dir: Path, chunking_info: Dict):
    """Bar chart: Average F1 score across all configurations per model."""
    plt.figure(figsize=(10, 6))
    
    if HAS_PANDAS:
        model_avg_f1 = aggregated.groupby('model')['commands_f1_mean'].mean().sort_values(ascending=False)
        models = model_avg_f1.index.tolist()
        f1_values = model_avg_f1.values.tolist()
    else:
        model_groups = defaultdict(list)
        for row in aggregated:
            model_groups[row['model']].append(row['commands_f1_mean'])
        
        model_avg = {m: statistics.mean(vals) for m, vals in model_groups.items()}
        sorted_models = sorted(model_avg.items(), key=lambda x: x[1], reverse=True)
        models = [m[0] for m in sorted_models]
        f1_values = [m[1] for m in sorted_models]
    
    colors = ['#3498db' if m == 'gemini' else '#e74c3c' for m in models]
    plt.bar(range(len(models)), f1_values, alpha=0.8, color=colors)
    plt.xticks(range(len(models)), models, rotation=0)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Average F1 Score (All Configs)', fontsize=12, fontweight='bold')
    plt.title(f'Average F1 Score by Model\nChunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}',
              fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'average_f1_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_comparison(aggregated: Any, plots_dir: Path, chunking_info: Dict):
    """Grouped bar chart: Precision vs Recall per model with RAG."""
    plt.figure(figsize=(12, 6))
    
    if HAS_PANDAS:
        rag_on = aggregated[aggregated['rag_enabled'] == True].sort_values('model')
        models = rag_on['model'].tolist()
        precision = rag_on['commands_precision_mean'].tolist()
        recall = rag_on['commands_recall_mean'].tolist()
    else:
        rag_on_rows = sorted([r for r in aggregated if r['rag_enabled']], key=lambda x: x['model'])
        models = [r['model'] for r in rag_on_rows]
        precision = [r['commands_precision_mean'] for r in rag_on_rows]
        recall = [r['commands_recall_mean'] for r in rag_on_rows]
    
    x = range(len(models))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], precision, width, label='Precision', alpha=0.8, color='#3498db')
    plt.bar([i + width/2 for i in x], recall, width, label='Recall', alpha=0.8, color='#2ecc71')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(f'Precision vs Recall by Model (RAG Enabled)\nChunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}',
              fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=0)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'precision_recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_gemini_comparison(metrics: List[Dict[str, Any]], plots_dir: Path, chunking_info: Dict):
    """Scatter plot: Compare other models' F1 vs Gemini's F1 for each test."""
    clean = [m for m in metrics if m.get('has_error', 0) == 0]
    
    # Group by test_id and model
    test_model_f1 = defaultdict(dict)
    for m in clean:
        test_id = m['test_id']
        model = m['model']
        # Use RAG-enabled scores for comparison
        if m['rag_enabled']:
            if model not in test_model_f1[test_id]:
                test_model_f1[test_id][model] = []
            test_model_f1[test_id][model].append(m['commands_f1'])
    
    # Average F1 per test per model
    test_model_avg = {}
    for test_id, models in test_model_f1.items():
        test_model_avg[test_id] = {m: statistics.mean(scores) for m, scores in models.items()}
    
    # Prepare data for scatter plot
    plt.figure(figsize=(10, 10))
    
    colors_map = {'llama': '#e74c3c', 'gemini': '#3498db'}
    
    for test_id, models_dict in test_model_avg.items():
        if 'gemini' in models_dict:
            gemini_f1 = models_dict['gemini']
            for model, f1 in models_dict.items():
                if model != 'gemini':
                    color = colors_map.get(model, '#95a5a6')
                    plt.scatter(gemini_f1, f1, s=100, alpha=0.6, color=color, label=model if test_id == list(test_model_avg.keys())[0] else '')
    
    # Add diagonal line (perfect match)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Match')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.xlabel('Gemini F1 Score', fontsize=12, fontweight='bold')
    plt.ylabel('Other Model F1 Score', fontsize=12, fontweight='bold')
    plt.title(f'Model Performance vs Gemini Baseline (RAG Enabled)\nChunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}',
              fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'gemini_comparison_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_chunk_retrieval_analysis(context: Dict[str, Any], plots_dir: Path):
    """Multi-panel analysis of chunk retrieval patterns."""
    if 'history' not in context or not context['history']:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    total_chunks = []
    code_chunks = []
    theory_chunks = []
    quality_scores = []
    models = []
    
    for entry in context['history']:
        if 'retrieval_plan' in entry and 'model' in entry:
            plan = entry['retrieval_plan']
            total_chunks.append(plan.get('total_chunks', 0))
            code_chunks.append(plan.get('code_chunks', 0))
            theory_chunks.append(plan.get('theory_chunks', 0))
            quality_scores.append(plan.get('quality_score', 0))
            models.append(entry['model'])
    
    if not total_chunks:
        plt.close(fig)
        return
    
    # 1. Chunks retrieved distribution
    axes[0, 0].hist(total_chunks, bins=max(total_chunks) if total_chunks else 10, 
                    alpha=0.7, color='#3498db', edgecolor='black')
    axes[0, 0].set_xlabel('Total Chunks Retrieved', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Distribution of Chunks Retrieved per Query', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Code vs Theory chunks
    model_types = sorted(set(models))
    code_by_model = {m: [] for m in model_types}
    theory_by_model = {m: [] for m in model_types}
    
    for m, c, t in zip(models, code_chunks, theory_chunks):
        code_by_model[m].append(c)
        theory_by_model[m].append(t)
    
    x = range(len(model_types))
    width = 0.35
    code_means = [statistics.mean(code_by_model[m]) if code_by_model[m] else 0 for m in model_types]
    theory_means = [statistics.mean(theory_by_model[m]) if theory_by_model[m] else 0 for m in model_types]
    
    axes[0, 1].bar([i - width/2 for i in x], code_means, width, label='Code Chunks', 
                   alpha=0.8, color='#e74c3c')
    axes[0, 1].bar([i + width/2 for i in x], theory_means, width, label='Theory Chunks', 
                   alpha=0.8, color='#2ecc71')
    axes[0, 1].set_xlabel('Model', fontweight='bold')
    axes[0, 1].set_ylabel('Avg Chunks Retrieved', fontweight='bold')
    axes[0, 1].set_title('Code vs Theory Chunks by Model', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_types)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Quality score distribution
    axes[1, 0].hist(quality_scores, bins=20, alpha=0.7, color='#9b59b6', edgecolor='black')
    axes[1, 0].axvline(statistics.mean(quality_scores), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {statistics.mean(quality_scores):.3f}')
    axes[1, 0].set_xlabel('Quality Score', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Chunk Retrieval Quality Score Distribution', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Quality score by model
    quality_by_model = {m: [] for m in model_types}
    for m, q in zip(models, quality_scores):
        quality_by_model[m].append(q)
    
    bp = axes[1, 1].boxplot([quality_by_model[m] for m in model_types], 
                            labels=model_types, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#f39c12')
    axes[1, 1].set_xlabel('Model', fontweight='bold')
    axes[1, 1].set_ylabel('Quality Score', fontweight='bold')
    axes[1, 1].set_title('Retrieval Quality Score by Model', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'chunk_retrieval_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_semantic_distance_comparison(raw_responses: List[Dict[str, Any]], plots_dir: Path, 
                                      chunking_info: Dict, embedding_model=None):
    """Compare semantic distance of model outputs vs ground truth using embeddings."""
    if not raw_responses or not embedding_model:
        print("  ⚠ Skipping semantic distance (no embeddings or responses)")
        return
    
    print("  Computing semantic distance vs ground truth...")
    
    # Group responses by model configuration and extract ground truth
    model_distances = defaultdict(list)
    
    for resp in raw_responses:
        if not resp.get('model_output') or 'config' not in resp['model_output']:
            continue
        
        model = resp['model']
        rag_label = 'RAG_On' if resp.get('rag_enabled', False) else 'RAG_Off'
        generated_config = resp['model_output']['config']
        
        # Extract ground truth from expected response
        expected = resp.get('expected', {})
        expected_responses = expected.get('response', [])
        
        # Build ground truth config string from expected commands
        ground_truth_commands = []
        for device_resp in expected_responses:
            device_name = device_resp.get('device_name', '')
            commands = device_resp.get('configuration_mode_commands', [])
            if device_name and commands:
                ground_truth_commands.append(f"! Device: {device_name}")
                ground_truth_commands.extend(commands)
        
        if not ground_truth_commands or not generated_config.strip():
            continue
        
        ground_truth_config = '\n'.join(ground_truth_commands)
        
        # Compute semantic distance between generated and ground truth
        distance = compute_semantic_distance(generated_config, ground_truth_config, embedding_model)
        model_distances[f"{model}_{rag_label}"].append(distance)
    
    if not model_distances:
        print("  ⚠ Could not compute distance to ground truth")
        return
    
    # Calculate average distance for each model configuration
    model_avg_distances = {}
    for model_key, distances in model_distances.items():
        if distances:
            model_avg_distances[model_key] = statistics.mean(distances)
    
    if not model_avg_distances:
        print("  ⚠ No valid distances computed")
        return
    
    # Plot distance comparison
    plt.figure(figsize=(12, 6))
    
    models = sorted(model_avg_distances.keys())
    distances = [model_avg_distances[m] for m in models]
    colors = ['#2ecc71' if 'RAG_On' in m else '#e74c3c' for m in models]
    
    bars = plt.bar(range(len(models)), distances, alpha=0.8, color=colors)
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel('Average Semantic Distance to Ground Truth', fontsize=12, fontweight='bold')
    plt.xlabel('Model Configuration', fontsize=12, fontweight='bold')
    plt.title(f'Semantic Distance vs Ground Truth (Lower = Better)\n'
              f'Embedding-Based Similarity Comparison\n'
              f'Chunk Size: {chunking_info["chunk_size"]}, Top-K: {chunking_info["retriever_top_k"]}',
              fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, distances)):
        plt.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, max(distances) * 1.15 if distances else 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / 'semantic_distance_ground_truth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Semantic distance to ground truth plot saved")
    
    # Print summary statistics
    print("\n  Semantic Distance Summary (lower = closer to ground truth):")
    for model in models:
        print(f"    {model}: {model_avg_distances[model]:.4f}")


def generate_summary_text(
    aggregated: Any,
    rag_comparison: Dict,
    test_results: Dict,
    hardest: List,
    easiest: List,
    chunking_info: Dict,
    run_dir: Path
) -> str:
    """Generate human-readable summary text."""
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append(f"\nRun Directory: {run_dir}")
    lines.append(f"Analysis Date: {Path(run_dir).name}\n")
    
    # Chunking info
    lines.append("RAG & CHUNKING CONFIGURATION")
    lines.append("-" * 80)
    lines.append(f"Chunk Size: {chunking_info['chunk_size']}")
    lines.append(f"Chunk Overlap: {chunking_info['chunk_overlap']}")
    lines.append(f"Retriever Top-K: {chunking_info['retriever_top_k']}")
    lines.append(f"Retriever Type: {chunking_info['retriever_type']}\n")
    
    # Overall metrics
    lines.append("OVERALL PERFORMANCE METRICS")
    lines.append("-" * 80)
    
    if HAS_PANDAS:
        for _, row in aggregated.iterrows():
            rag_status = "RAG ON" if row['rag_enabled'] else "RAG OFF"
            lines.append(f"\n{row['model'].upper()} ({rag_status}):")
            lines.append(f"  Commands F1:       {row['commands_f1_mean']:.4f} ± {row['commands_f1_std']:.4f}")
            lines.append(f"  Exact Match:       {row['exact_match_mean']:.4f}")
            lines.append(f"  Precision:         {row['commands_precision_mean']:.4f}")
            lines.append(f"  Recall:            {row['commands_recall_mean']:.4f}")
            lines.append(f"  Batfish Pass Rate: {row['batfish_pass_mean']:.4f}")
            lines.append(f"  AI Verdict Pass:   {row['ai_verdict_pass_mean']:.4f} (GLOBAL FAILURE METRIC)")
            lines.append(f"  Number of Tests:   {int(row['exact_match_count'])}")
    else:
        for row in aggregated:
            rag_status = "RAG ON" if row['rag_enabled'] else "RAG OFF"
            lines.append(f"\n{row['model'].upper()} ({rag_status}):")
            lines.append(f"  Commands F1:       {row['commands_f1_mean']:.4f} ± {row['commands_f1_std']:.4f}")
            lines.append(f"  Exact Match:       {row['exact_match_mean']:.4f}")
            lines.append(f"  Precision:         {row['commands_precision_mean']:.4f}")
            lines.append(f"  Recall:            {row['commands_recall_mean']:.4f}")
            lines.append(f"  Batfish Pass Rate: {row['batfish_pass_mean']:.4f}")
            lines.append(f"  AI Verdict Pass:   {row['ai_verdict_pass_mean']:.4f} (GLOBAL FAILURE METRIC)")
            lines.append(f"  Number of Tests:   {row['exact_match_count']}")
    
    # RAG comparison
    lines.append("\n\nRAG IMPACT ANALYSIS")
    lines.append("-" * 80)
    
    best_model = None
    best_improvement = -999
    
    for model, comp in rag_comparison.items():
        lines.append(f"\n{model.upper()}:")
        lines.append(f"  F1 Delta (RAG On - RAG Off):          {comp['f1_delta']:+.4f}")
        lines.append(f"  Exact Match Delta:                    {comp['exact_match_delta']:+.4f}")
        lines.append(f"  Batfish Pass Delta:                   {comp['batfish_pass_delta']:+.4f}")
        lines.append(f"  AI Verdict Pass Delta:                {comp['ai_verdict_pass_delta']:+.4f} (GLOBAL METRIC)")
        lines.append(f"  F1 with RAG:    {comp['f1_rag_on']:.4f}")
        lines.append(f"  F1 without RAG: {comp['f1_rag_off']:.4f}")
        
        if comp['f1_delta'] > best_improvement:
            best_improvement = comp['f1_delta']
            best_model = model
    
    lines.append(f"\n>> BEST MODEL FOR RAG: {best_model.upper()} with F1 improvement of {best_improvement:+.4f}")
    
    # Hardest tests
    lines.append("\n\n5 HARDEST TESTS (Lowest Avg F1)")
    lines.append("-" * 80)
    for test_id, info in hardest:
        lines.append(f"\n{test_id}:")
        lines.append(f"  Avg F1 Score:         {info['avg_commands_f1']:.4f}")
        lines.append(f"  Avg Batfish Pass:     {info['avg_batfish_pass']:.4f}")
        lines.append(f"  Avg AI Verdict Pass:  {info['avg_ai_verdict_pass']:.4f}")
        lines.append(f"  Best Combination:     {info['best_combo']} (F1: {info['best_f1']:.4f})")
    
    # Easiest tests
    lines.append("\n\n5 EASIEST TESTS (Highest Avg F1)")
    lines.append("-" * 80)
    for test_id, info in easiest:
        lines.append(f"\n{test_id}:")
        lines.append(f"  Avg F1 Score:         {info['avg_commands_f1']:.4f}")
        lines.append(f"  Avg Batfish Pass:     {info['avg_batfish_pass']:.4f}")
        lines.append(f"  Avg AI Verdict Pass:  {info['avg_ai_verdict_pass']:.4f}")
        lines.append(f"  Best Combination:     {info['best_combo']} (F1: {info['best_f1']:.4f})")
    
    # Insights
    lines.append("\n\nKEY INSIGHTS")
    lines.append("-" * 80)
    
    # Overall RAG benefit
    all_rag_on_f1 = []
    all_rag_off_f1 = []
    
    if HAS_PANDAS:
        all_rag_on_f1 = aggregated[aggregated['rag_enabled'] == True]['commands_f1_mean'].tolist()
        all_rag_off_f1 = aggregated[aggregated['rag_enabled'] == False]['commands_f1_mean'].tolist()
    else:
        for row in aggregated:
            if row['rag_enabled']:
                all_rag_on_f1.append(row['commands_f1_mean'])
            else:
                all_rag_off_f1.append(row['commands_f1_mean'])
    
    if all_rag_on_f1 and all_rag_off_f1:
        avg_rag_on = statistics.mean(all_rag_on_f1)
        avg_rag_off = statistics.mean(all_rag_off_f1)
        improvement = avg_rag_on - avg_rag_off
        
        lines.append(f"1. Overall RAG Impact: {'Positive' if improvement > 0 else 'Negative'}")
        lines.append(f"   - Average F1 with RAG:    {avg_rag_on:.4f}")
        lines.append(f"   - Average F1 without RAG: {avg_rag_off:.4f}")
        lines.append(f"   - Improvement:            {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    lines.append(f"\n2. Chunking Configuration Impact:")
    lines.append(f"   - Current chunk size of {chunking_info['chunk_size']} with overlap of")
    lines.append(f"     {chunking_info['chunk_overlap']} appears to be {'effective' if improvement > 0 else 'suboptimal'}")
    lines.append(f"   - Retrieving top-{chunking_info['retriever_top_k']} chunks per query")
    
    lines.append(f"\n3. Model Performance:")
    if rag_comparison:
        best_overall = max(rag_comparison.items(), key=lambda x: x[1]['f1_rag_on'])
        lines.append(f"   - {best_overall[0].upper()} shows the best absolute performance with RAG")
        lines.append(f"     (F1: {best_overall[1]['f1_rag_on']:.4f})")
        lines.append(f"   - {best_model.upper()} benefits most from RAG (improvement: {best_improvement:+.4f})")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def generate_summary_markdown(summary_text: str) -> str:
    """Convert summary text to markdown format."""
    # Simple conversion: replace = with # headers, - with ##
    lines = summary_text.split('\n')
    md_lines = []
    
    for line in lines:
        if line.startswith('=' * 80):
            continue  # Skip separator lines
        elif line and all(c in '=-' for c in line.strip()) and len(line.strip()) > 20:
            continue  # Skip other separator lines
        elif 'BENCHMARK ANALYSIS SUMMARY' in line:
            md_lines.append(f"# {line.strip()}")
        elif line.strip() and line.strip().isupper() and not line.strip().startswith('>>'):
            # Section headers
            md_lines.append(f"\n## {line.strip()}\n")
        elif line.strip().startswith('>>'):
            # Highlighted insights
            md_lines.append(f"\n**{line.strip()[3:]}**\n")
        else:
            md_lines.append(line)
    
    return '\n'.join(md_lines)


def save_aggregated_csv(aggregated: Any, analysis_dir: Path):
    """Save aggregated results to CSV."""
    csv_path = analysis_dir / 'aggregated_results.csv'
    
    if HAS_PANDAS:
        aggregated.to_csv(csv_path, index=False)
    else:
        # Manual CSV writing
        if not aggregated:
            return
        
        headers = list(aggregated[0].keys())
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(headers) + '\n')
            for row in aggregated:
                values = [str(row.get(h, '')) for h in headers]
                f.write(','.join(values) + '\n')
    
    print(f"✓ Saved aggregated results to {csv_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate run directory
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    print(f"Analyzing benchmark run: {run_dir.name}")
    print("=" * 80)
    
    # Load data
    print("\n[1/8] Loading data...")
    context = load_context(args.context_path, run_dir)
    metrics = load_metrics(run_dir)
    raw_responses = load_raw_responses(run_dir)
    chunking_info = extract_chunking_info(context, run_dir)
    print(f"  ✓ Loaded {len(metrics)} metric entries")
    print(f"  ✓ Loaded {len(raw_responses)} raw responses")
    print(f"  ✓ Chunking config: size={chunking_info['chunk_size']}, top_k={chunking_info['retriever_top_k']}")
    print(f"  ✓ Avg chunks retrieved: {chunking_info['avg_chunks_retrieved']:.1f}, quality: {chunking_info['avg_quality_score']:.3f}")
    
    # Load embedding model for semantic distance
    embedding_model = None
    if HAS_EMBEDDINGS:
        print("\n[2/8] Loading embedding model for semantic distance...")
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("  ✓ Loaded sentence transformer model")
        except Exception as e:
            print(f"  ⚠ Failed to load embeddings: {e}")
            embedding_model = None
    else:
        print("\n[2/8] Skipping embedding model (not available)...")
    
    # Create output directories
    print("\n[3/8] Creating output directories...")
    analysis_dir, plots_dir = create_output_dirs(run_dir)
    print(f"  ✓ Analysis directory: {analysis_dir}")
    print(f"  ✓ Plots directory: {plots_dir}")
    
    # Aggregate metrics
    print("\n[4/8] Aggregating metrics...")
    if HAS_PANDAS:
        aggregated = aggregate_metrics_pandas(metrics)
        print(f"  ✓ Aggregated using pandas")
    else:
        aggregated = aggregate_metrics_manual(metrics)
        print(f"  ✓ Aggregated using manual methods")
    
    # Per-test analysis
    print("\n[5/8] Analyzing per-test performance...")
    test_results = analyze_per_test(metrics)
    hardest, easiest = find_hardest_easiest_tests(test_results)
    print(f"  ✓ Analyzed {len(test_results)} unique tests")
    print(f"  ✓ Hardest test: {hardest[0][0]} (F1: {hardest[0][1]['avg_commands_f1']:.4f})")
    print(f"  ✓ Easiest test: {easiest[0][0]} (F1: {easiest[0][1]['avg_commands_f1']:.4f})")
    
    # RAG comparison
    print("\n[6/8] Computing RAG vs non-RAG comparison...")
    rag_comparison = compute_rag_comparison(aggregated)
    print(f"  ✓ Compared {len(rag_comparison)} models")
    for model, comp in rag_comparison.items():
        print(f"    - {model}: F1 delta = {comp['f1_delta']:+.4f}")
    
    # Generate plots
    print("\n[7/8] Generating plots...")
    plot_commands_f1_by_model_rag(aggregated, plots_dir, chunking_info)
    print("  ✓ commands_f1_by_model_rag.png")
    
    plot_batfish_pass_by_model_rag(aggregated, plots_dir, chunking_info)
    print("  ✓ batfish_pass_by_model_rag.png")
    
    plot_ai_verdict_pass_by_model_rag(aggregated, plots_dir, chunking_info)
    print("  ✓ ai_verdict_pass_by_model_rag.png (GLOBAL FAILURE METRIC)")
    
    plot_f1_vs_batfish_scatter(aggregated, plots_dir, chunking_info)
    print("  ✓ commands_f1_vs_batfish_pass_ragon.png")
    
    # RAG comparison plots
    plot_rag_impact_comparison(aggregated, plots_dir, chunking_info, rag_comparison)
    print("  ✓ rag_impact_comparison.png")
    
    plot_average_f1_by_model(aggregated, plots_dir, chunking_info)
    print("  ✓ average_f1_by_model.png")
    
    plot_precision_recall_comparison(aggregated, plots_dir, chunking_info)
    print("  ✓ precision_recall_comparison.png")
    
    # Chunk analysis
    plot_chunk_retrieval_analysis(context, plots_dir)
    print("  ✓ chunk_retrieval_analysis.png")
    
    # Semantic distance variance (replaces heatmap)
    plot_semantic_distance_comparison(raw_responses, plots_dir, chunking_info, embedding_model)
    print("  ✓ semantic_distance_ground_truth.png")
    
    # Generate summaries
    print("\n[8/8] Generating summary reports...")
    summary_text = generate_summary_text(
        aggregated, rag_comparison, test_results,
        hardest, easiest, chunking_info, run_dir
    )
    
    # Save outputs
    save_aggregated_csv(aggregated, analysis_dir)
    
    summary_txt_path = analysis_dir / 'summary.txt'
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  ✓ summary.txt")
    
    summary_md = generate_summary_markdown(summary_text)
    summary_md_path = analysis_dir / 'summary.md'
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    print(f"  ✓ summary.md")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {analysis_dir}")
    print(f"Plots saved to:   {plots_dir}")
    print("\nGenerated files:")
    print("  - aggregated_results.csv")
    print("  - summary.txt")
    print("  - summary.md")
    print("  - plots/commands_f1_by_model_rag.png (RAG on vs off)")
    print("  - plots/batfish_pass_by_model_rag.png (RAG on vs off)")
    print("  - plots/ai_verdict_pass_by_model_rag.png (GLOBAL FAILURE METRIC - RAG on vs off)")
    print("  - plots/commands_f1_vs_batfish_pass_ragon.png (F1 vs Batfish, RAG comparison)")
    print("  - plots/rag_impact_comparison.png (Delta visualization)")
    print("  - plots/average_f1_by_model.png")
    print("  - plots/precision_recall_comparison.png (RAG enabled)")
    print("  - plots/chunk_retrieval_analysis.png (RAG behavior)")
    print("  - plots/semantic_distance_ground_truth.png (distance to ground truth)")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
