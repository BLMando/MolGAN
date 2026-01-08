"""
Run Instance Graph Evaluation

Evaluate generated graphs against ground truth using IG metrics:
- Accuracy (Acc): Count of exactly matching graphs
- Matching Cost (MC): Graph Edit Distance
- Average Generalization (AG): Number of occurrence sequences

Usage:
    python run_ig_evaluation.py --data ../../data/Helpdesk_igs_complete.g --generated-dir ... --max-nodes 5
    python run_ig_evaluation.py --generated-dir ../../output_graph_gan/samples/20251201_105053 --max-nodes 5 --min-nodes 5
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ig_loader import load_ig_for_gan
from graph_evaluation import (
    evaluate_generated_graphs,
    evaluate_from_txt_files,
    print_evaluation_results
)
from graph_conversion import batch_matrices_to_networkx, traces_to_networkx


def main():
    parser = argparse.ArgumentParser(description='Evaluate generated graphs using IG metrics')
    
    # Data source options
    parser.add_argument('--data', type=str, required=True,
                       help='Path to ground truth .g file')
    
    parser.add_argument('--generated-dir', type=str,
                       help='Path to directory with generated .txt graph files')
    
    # Evaluation options
    parser.add_argument('--num-real', type=int, default=None,
                       help='Number of real graphs to use (default: all)')
    parser.add_argument('--max-nodes', type=int, default=None,
                       help='Filter ground truth to graphs with at most this many nodes (matches fixed GAN output size)')
    parser.add_argument('--min-nodes', type=int, default=None,
                       help='Filter ground truth to graphs with at least this many nodes')
    parser.add_argument('--compute-ag', action='store_true', default=True,
                       help='Compute Average Generalization (default: True)')
    parser.add_argument('--no-ag', action='store_true',
                       help='Skip Average Generalization computation')
    parser.add_argument('--mc-timeout', type=float, default=5.0,
                       help='Timeout per Matching Cost computation (seconds)')
    parser.add_argument('--ag-max-count', type=int, default=10000,
                       help='Maximum AG count before capping')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # ==================================================================
    # Load ground truth data
    # ==================================================================
    print("=" * 60)
    print("INSTANCE GRAPH EVALUATION")
    print("=" * 60)
    print(f"\nLoading ground truth from: {args.data}")
    
    data = load_ig_for_gan(args.data)
    all_traces = data['traces_test']
    idx_to_activity = data['idx_to_activity']
    
    if args.max_nodes is not None or args.min_nodes is not None:
        min_n = args.min_nodes if args.min_nodes is not None else 0
        max_n = args.max_nodes if args.max_nodes is not None else float('inf')
        filtered_traces = [
            t for t in all_traces 
            if min_n <= len(t['vertices']) <= max_n
        ]
        print(f"Filtered traces by node count [{min_n}, {max_n}]: {len(all_traces)} -> {len(filtered_traces)}")
        all_traces = filtered_traces
    
    if args.num_real is not None:
        all_traces = all_traces[:args.num_real]
    
    print(f"Loaded {len(all_traces)} ground truth traces")
    
    # ==================================================================
    # Load or generate predicted graphs
    # ==================================================================
        
    if args.generated_dir:
        print(f"\nLoading generated graphs from: {args.generated_dir}")
        
        compute_ag = args.compute_ag and not args.no_ag
        results = evaluate_from_txt_files(
            generated_dir=args.generated_dir,
            true_traces=all_traces,
            compute_ag=compute_ag,
            ag_max_count=args.ag_max_count,
            mc_timeout=args.mc_timeout,
            verbose=not args.quiet
        )
        
    # ==================================================================
    # Print and save results
    # ==================================================================
    
    if not args.quiet:
        print_evaluation_results(results)
    
    if args.output:
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            return obj
        
        results_json = convert_for_json(results)
        results_json['timestamp'] = datetime.now().isoformat()
        results_json['args'] = vars(args)
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    if 'accuracy' in results:
        acc = results['accuracy']['percentage']
        mc = results['matching_cost']['mean'] if 'matching_cost' in results else 'N/A'
        ag = results['avg_generalization']['dag_mean'] if 'avg_generalization' in results else 'N/A'
        print(f"\n📊 Summary: Accuracy={acc:.1f}%, MC_mean={mc}, AG_mean={ag}")
    
    return results


if __name__ == '__main__':
    main()
