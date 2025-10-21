#!/usr/bin/env python3
"""
Visualize ProcessGAN results

Usage:
    python visualize_results.py --data data/sample_event_log.csv --model results/model_epoch_100.h5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.process_dataset import ProcessDataset
from models.process_gan import ProcessGAN
from utils.visualization import ProcessGraphVisualizer, plot_training_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize ProcessGAN generated graphs')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to event log CSV file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model checkpoint (optional)')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of graphs to generate and visualize')
    parser.add_argument('--output', type=str, default='visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--z-dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print(f"[INFO] Loading dataset from {args.data}...")
    dataset = ProcessDataset()
    dataset.load_from_csv(args.data, case_id_col='case_id',
                          activity_col='activity')
    dataset.split_train_val_test(train_ratio=0.8, val_ratio=0.1)

    print(f"[INFO] Dataset loaded: {len(dataset.traces)} traces")
    print(
        f"[INFO]   Activities: {dataset.max_activities}, Flow types: {len(dataset.flow_encoder)}")

    # Create visualizer
    activity_decoder = {v: k for k, v in dataset.activity_encoder.items()}
    flow_decoder = {v: k for k, v in dataset.flow_encoder.items()}
    visualizer = ProcessGraphVisualizer(activity_decoder, flow_decoder)

    # ═══════════════════════════════════════════════════════════
    # Visualize Original Traces
    # ═══════════════════════════════════════════════════════════

    print(f"\n[INFO] Visualizing original traces...")

    # Visualize first few real traces
    for i in range(min(3, len(dataset.traces))):
        trace = dataset.traces[i]
        fig = visualizer.visualize_trace(
            trace,
            title=f"Real Trace {i+1}",
            figsize=(14, 2)
        )
        save_path = output_dir / f"real_trace_{i+1}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

    # Visualize real graphs
    for i in range(min(3, len(dataset.adjacency_matrices))):
        adj = dataset.adjacency_matrices[i]
        nodes = dataset.node_features[i]

        # Get node labels
        node_indices = np.argmax(nodes, axis=-1)
        node_labels = [activity_decoder.get(
            idx, 'PAD') for idx in node_indices]

        # Graph visualization
        fig = visualizer.visualize_graph(
            adj, nodes,
            title=f"Real Process Graph {i+1}",
            figsize=(12, 8)
        )
        save_path = output_dir / f"real_graph_{i+1}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

        # Adjacency matrix visualization
        fig = visualizer.visualize_adjacency_matrix(
            adj,
            node_labels=node_labels[:10],  # First 10 nodes
            title=f"Real Adjacency Matrix {i+1}",
            figsize=(10, 8)
        )
        save_path = output_dir / f"real_adjacency_{i+1}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

    # ═══════════════════════════════════════════════════════════
    # Generate and Visualize Synthetic Traces
    # ═══════════════════════════════════════════════════════════

    if args.model:
        print(f"\n[INFO] Loading model from {args.model}...")
        # TODO: Load trained model checkpoint
        print("  [WARNING] Model loading not yet implemented")
        print("  [INFO] Generating random samples instead...")

    print(f"\n[INFO] Generating {args.n_samples} synthetic graphs...")

    # Create model
    model = ProcessGAN(
        max_activities=dataset.max_activities,
        flow_types=len(dataset.flow_encoder),
        activity_types=len(dataset.activity_encoder),
        embedding_dim=args.z_dim,
        decoder_units=[128, 256, 512],
        discriminator_units=[128, 64],
        mlp_units=[64],
        dropout_rate=0.3
    )

    # Generate random samples
    z = np.random.normal(0, 1, (args.n_samples, args.z_dim))
    adjacency_batch, nodes_batch = model.generator(z, training=False)

    # Convert to numpy
    adjacency_batch = adjacency_batch.numpy()
    nodes_batch = nodes_batch.numpy()

    # Decode traces
    traces = dataset.decode_batch(adjacency_batch, nodes_batch)

    # Visualize each generated graph
    for i in range(args.n_samples):
        trace = traces[i]
        adj = adjacency_batch[i]
        nodes = nodes_batch[i]

        # Trace visualization
        fig = visualizer.visualize_trace(
            trace,
            title=f"Generated Trace {i+1}",
            figsize=(14, 2)
        )
        save_path = output_dir / f"generated_trace_{i+1}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

        # Graph visualization
        fig = visualizer.visualize_graph(
            adj, nodes,
            title=f"Generated Process Graph {i+1}",
            figsize=(12, 8)
        )
        save_path = output_dir / f"generated_graph_{i+1}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

        # Print trace
        print(f"  Trace {i+1}: {' → '.join(trace)}")

    # ═══════════════════════════════════════════════════════════
    # Compare Real vs Generated
    # ═══════════════════════════════════════════════════════════

    print(f"\n[INFO] Creating comparison visualizations...")

    # Side-by-side comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Real vs Generated Process Graphs',
                 fontsize=16, fontweight='bold')

    for idx in range(2):
        # Real graph
        if idx < len(dataset.adjacency_matrices):
            real_adj = dataset.adjacency_matrices[idx]
            real_nodes = dataset.node_features[idx]

            temp_fig = visualizer.visualize_graph(
                real_adj, real_nodes,
                title=f"Real Graph {idx+1}",
                figsize=(8, 6)
            )
            plt.close(temp_fig)

        # Generated graph
        gen_adj = adjacency_batch[idx]
        gen_nodes = nodes_batch[idx]

        temp_fig = visualizer.visualize_graph(
            gen_adj, gen_nodes,
            title=f"Generated Graph {idx+1}",
            figsize=(8, 6)
        )
        plt.close(temp_fig)

    save_path = output_dir / "comparison.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # ═══════════════════════════════════════════════════════════
    # Statistics
    # ═══════════════════════════════════════════════════════════

    print(f"\n[INFO] Statistics:")
    print(f"  Real traces:")
    real_lengths = [len(t) for t in dataset.traces]
    print(
        f"    Average length: {np.mean(real_lengths):.2f} ± {np.std(real_lengths):.2f}")
    print(f"    Min/Max length: {min(real_lengths)} / {max(real_lengths)}")

    print(f"\n  Generated traces:")
    gen_lengths = [len(t) for t in traces]
    print(
        f"    Average length: {np.mean(gen_lengths):.2f} ± {np.std(gen_lengths):.2f}")
    print(f"    Min/Max length: {min(gen_lengths)} / {max(gen_lengths)}")

    print(f"\n[INFO] All visualizations saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
