#!/usr/bin/env python3
"""
Simple visualization test without model dependencies

This demonstrates the visualization capabilities on real data only.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Simple dataset loader (no TF dependencies)
import pandas as pd


def load_simple_dataset(csv_path):
    """Load event log without full ProcessDataset"""
    df = pd.read_csv(csv_path)

    # Extract traces
    traces = []
    for case_id, group in df.groupby('case_id'):
        trace = group['activity'].tolist()
        traces.append(trace)

    return traces


def create_simple_visualizer():
    """Create basic visualizer with common process activities"""
    activity_decoder = {
        0: 'PAD',
        1: 'Start',
        2: 'Submit',
        3: 'Review',
        4: 'Approve',
        5: 'Reject',
        6: 'Rework',
        7: 'End'
    }

    flow_decoder = {
        0: 'NONE',
        1: 'SEQUENCE',
        2: 'PARALLEL',
        3: 'LOOP',
        4: 'CHOICE'
    }

    from utils.visualization import ProcessGraphVisualizer
    return ProcessGraphVisualizer(activity_decoder, flow_decoder)


def visualize_trace_simple(trace, output_path):
    """Simple trace visualization without full dependencies"""
    fig, ax = plt.subplots(figsize=(14, 2))

    colors = {
        'Start': '#90EE90',
        'End': '#FFB6C6',
        'default': '#87CEEB'
    }

    for i, activity in enumerate(trace):
        # Determine color
        if 'Start' in activity:
            color = colors['Start']
        elif 'End' in activity:
            color = colors['End']
        else:
            color = colors['default']

        # Draw box
        rect = plt.Rectangle((i*1.5, 0), 1.2, 0.6,
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(rect)

        # Add text
        ax.text(i*1.5 + 0.6, 0.3, activity,
               ha='center', va='center',
               fontsize=9, fontweight='bold',
               wrap=True)

        # Draw arrow
        if i < len(trace) - 1:
            ax.arrow(i*1.5 + 1.2, 0.3, 0.2, 0,
                    head_width=0.15, head_length=0.1,
                    fc='black', ec='black')

    ax.set_xlim(-0.3, len(trace)*1.5)
    ax.set_ylim(-0.2, 0.8)
    ax.axis('off')
    ax.set_title(f'Process Trace ({len(trace)} activities)',
                fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("[INFO] Simple Visualization Test")
    print("=" * 60)

    # Create output directory
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    # Load data
    csv_path = 'data/sample_event_log.csv'
    print(f"\n[INFO] Loading traces from {csv_path}...")

    try:
        traces = load_simple_dataset(csv_path)
        print(f"[INFO] Loaded {len(traces)} traces")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        print("[INFO] Creating example traces instead...")
        traces = [
            ['Start', 'Submit', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Review', 'Reject', 'Rework', 'Submit', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Review', 'Approve', 'End']
        ]

    # Visualize traces
    print(f"\n[INFO] Creating visualizations...")

    for i, trace in enumerate(traces[:5]):  # First 5 traces
        print(f"  Trace {i+1}: {' → '.join(trace)}")

        output_path = output_dir / f'trace_{i+1}.png'
        visualize_trace_simple(trace, output_path)
        print(f"    Saved: {output_path}")

    # Statistics
    print(f"\n[INFO] Statistics:")
    lengths = [len(t) for t in traces]
    print(f"  Total traces: {len(traces)}")
    print(f"  Average length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(f"  Min/Max length: {min(lengths)} / {max(lengths)}")

    # Activity distribution
    all_activities = [act for trace in traces for act in trace]
    unique_activities = set(all_activities)
    print(f"  Unique activities: {len(unique_activities)}")
    print(f"    {', '.join(sorted(unique_activities))}")

    print(f"\n[SUCCESS] Visualizations saved to: {output_dir.absolute()}")
    print("\nTo use full visualization features with graphs:")
    print("  1. Activate conda environment: conda activate ProcessGAN")
    print("  2. Run: python visualize_results.py --data data/sample_event_log.csv")


if __name__ == '__main__':
    main()
