"""
Graph Evaluation Module

Main evaluation entry point for comparing generated and real instance graphs
using the metrics from ig_metrics.py:
- Accuracy (Acc): Count of exactly matching graphs
- Matching Cost (MC): Graph Edit Distance
- Average Generalization (AG): Number of occurrence sequences
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from glob import glob
from ig_metrics import evaluate_instance_graphs
from graph_conversion import (
    batch_matrices_to_networkx,
    traces_to_networkx,
    txt_files_to_networkx
)


def evaluate_generated_graphs(
    pred_adj: np.ndarray,
    pred_nodes: np.ndarray,
    idx_to_activity: Dict[int, str],
    true_traces: Optional[List[Dict]] = None,
    pad_activity: str = 'PAD',
    compute_ag: bool = True,
    ag_max_count: int = 10000,
    mc_timeout: float = 5.0,
    verbose: bool = True
) -> dict:
    """
    Evaluate generated graphs using Instance Graph metrics.
    
    Args:
        pred_adj: Generated adjacency matrices
                 Shape: (batch, max_nodes, max_nodes) or 
                        (batch, max_nodes, max_nodes, num_edge_types)
        pred_nodes: Generated node matrices
                   Shape: (batch, max_nodes, num_activities)
        idx_to_activity: Mapping from activity index to activity name
        true_traces: Optional list of real traces from ig_loader
                    If provided, computes Accuracy and Matching Cost
        pad_activity: Name of PAD activity to filter out
        compute_ag: Whether to compute Average Generalization
        ag_max_count: Maximum count for AG computation per graph
        mc_timeout: Timeout in seconds for Matching Cost computation per pair
        verbose: Whether to print progress
    
    Returns:
        Dictionary with evaluation results:
        {
            'num_predicted': int,
            'num_true': int (if true_traces provided),
            'accuracy': {'count': int, 'percentage': float},
            'matching_cost': {'mean': float, 'median': float, ...},
            'avg_generalization': {'mean': float, ...}
        }
    """
    if verbose:
        print("=" * 60)
        print("INSTANCE GRAPH EVALUATION")
        print("=" * 60)
    
    if verbose:
        print(f"\nConverting {len(pred_adj)} generated graphs to NetworkX format...")
    
    pred_graphs = batch_matrices_to_networkx(
        adj_matrices=pred_adj,
        node_matrices=pred_nodes,
        idx_to_activity=idx_to_activity,
        pad_activity=pad_activity
    )
    
    if verbose:
        node_counts = [len(g.nodes()) for g in pred_graphs]
        edge_counts = [len(g.edges()) for g in pred_graphs]
        print(f"  Avg nodes: {np.mean(node_counts):.1f} (min={min(node_counts)}, max={max(node_counts)})")
        print(f"  Avg edges: {np.mean(edge_counts):.1f} (min={min(edge_counts)}, max={max(edge_counts)})")
    
    true_graphs = None
    if true_traces is not None:
        if verbose:
            print(f"\nConverting {len(true_traces)} real traces to NetworkX format...")
        true_graphs = traces_to_networkx(true_traces)
        
        if verbose:
            node_counts = [len(g.nodes()) for g in true_graphs]
            edge_counts = [len(g.edges()) for g in true_graphs]
            print(f"  Avg nodes: {np.mean(node_counts):.1f} (min={min(node_counts)}, max={max(node_counts)})")
            print(f"  Avg edges: {np.mean(edge_counts):.1f} (min={min(edge_counts)}, max={max(edge_counts)})")
    
    if verbose:
        print("\n" + "-" * 40)
        print("Running IG Metrics Evaluation...")
        print("-" * 40)
    
    results = evaluate_instance_graphs(
        pred_graphs=pred_graphs,
        true_graphs=true_graphs,
        compute_ag=compute_ag,
        ag_max_count=ag_max_count,
        mc_timeout=mc_timeout
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
    
    return results


def evaluate_from_txt_files(
    generated_dir: str,
    true_traces: Optional[List[Dict]] = None,
    pattern: str = '*.txt',
    compute_ag: bool = True,
    ag_max_count: int = 10000,
    mc_timeout: float = 5.0,
    verbose: bool = True
) -> dict:
    """
    Evaluate generated graphs from saved .txt files.
    
    Args:
        generated_dir: Directory containing .txt graph files
        true_traces: Optional list of real traces from ig_loader
        pattern: Glob pattern for finding graph files
        compute_ag: Whether to compute Average Generalization
        ag_max_count: Maximum count for AG computation
        mc_timeout: Timeout for MC computation per pair
        verbose: Whether to print progress
    
    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print("=" * 60)
        print("INSTANCE GRAPH EVALUATION (from files)")
        print("=" * 60)
    
    generated_path = Path(generated_dir)
    txt_files = sorted(glob(str(generated_path / pattern)))
    
    if not txt_files:
        raise FileNotFoundError(f"No {pattern} files found in {generated_dir}")
    
    if verbose:
        print(f"\nFound {len(txt_files)} graph files in {generated_dir}")
    
    if verbose:
        print(f"Loading generated graphs...")
    pred_graphs = txt_files_to_networkx(txt_files)
    
    if verbose:
        node_counts = [len(g.nodes()) for g in pred_graphs]
        edge_counts = [len(g.edges()) for g in pred_graphs]
        print(f"  Loaded {len(pred_graphs)} graphs")
        print(f"  Avg nodes: {np.mean(node_counts):.1f} (min={min(node_counts)}, max={max(node_counts)})")
        print(f"  Avg edges: {np.mean(edge_counts):.1f} (min={min(edge_counts)}, max={max(edge_counts)})")
    
    true_graphs = None
    if true_traces is not None:
        if verbose:
            print(f"\nConverting {len(true_traces)} real traces to NetworkX format...")
        true_graphs = traces_to_networkx(true_traces)
    
    if verbose:
        print("\n" + "-" * 40)
        print("Running IG Metrics Evaluation...")
        print("-" * 40)
    
    results = evaluate_instance_graphs(
        pred_graphs=pred_graphs,
        true_graphs=true_graphs,
        compute_ag=compute_ag,
        ag_max_count=ag_max_count,
        mc_timeout=mc_timeout
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
    
    return results


def print_evaluation_results(results: dict):
    """
    Pretty-print evaluation results.
    
    Args:
        results: Dictionary from evaluate_generated_graphs or evaluate_from_txt_files
    """
    print("\n" + "=" * 50)
    print("INSTANCE GRAPH METRICS SUMMARY")
    print("=" * 50)
    
    print(f"\nDataset: {results.get('num_predicted', 'N/A')} predicted graphs")
    if 'num_true' in results:
        print(f"         {results['num_true']} ground truth graphs")
    
    # Accuracy
    if 'accuracy' in results:
        acc = results['accuracy']
        print(f"\n📊 ACCURACY:")
        print(f"   Correct matches: {acc['count']}")
        print(f"   Percentage: {acc['percentage']:.2f}%")
    
    # Matching Cost
    if 'matching_cost' in results:
        mc = results['matching_cost']
        print(f"\n📏 MATCHING COST (Graph Edit Distance):")
        print(f"   Mean: {mc['mean']:.2f}")
        print(f"   Median: {mc['median']:.2f}")
        print(f"   Std: {mc['std']:.2f}")
        print(f"   Computed: {mc['computed']}, Failed: {mc['failed']}")
    
    # Average Generalization
    if 'avg_generalization' in results:
        ag = results['avg_generalization']
        print(f"\n🔄 AVERAGE GENERALIZATION:")
        print(f"   DAGs: {ag['num_dags']}, Cyclic: {ag['num_cyclic']}")
        if ag['num_dags'] > 0:
            print(f"   Mean (DAGs only): {ag['dag_mean']:.2f}")
            print(f"   Median (DAGs only): {ag['dag_median']:.2f}")
            print(f"   Min/Max: {ag['min']}/{ag['max']}")
            print(f"   Tailored (AG=1): {ag['ag_1']}")
            print(f"   Capped (AG≥max): {ag['ag_capped']}")
    
    print("\n" + "=" * 50)

