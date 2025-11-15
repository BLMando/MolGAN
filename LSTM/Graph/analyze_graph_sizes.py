"""
Analyze graph size distribution in dataset

Usage:
    python analyze_graph_sizes.py
"""

import sys
sys.path.append('..')

from ig_loader import load_instance_graph
from graph_dataset import GraphDataset
import numpy as np
import matplotlib.pyplot as plt


def analyze_sizes(data_path='../../data/Helpdesk_igs_complete.g'):
    """Analyze and visualize graph size distribution"""
    
    print('Loading data...')
    graphs, vocab = load_instance_graph(data_path)
    
    # Get size distribution without filtering
    print('\nAnalyzing size distribution...')
    sizes = [len(g['vertices']) for g in graphs]
    sizes_array = np.array(sizes)
    
    print(f'\nGraph Size Statistics:')
    print(f'  Total graphs: {len(sizes)}')
    print(f'  Min size: {np.min(sizes_array)}')
    print(f'  Max size: {np.max(sizes_array)}')
    print(f'  Mean size: {np.mean(sizes_array):.2f}')
    print(f'  Median size: {np.median(sizes_array):.1f}')
    print(f'  Std dev: {np.std(sizes_array):.2f}')
    
    # Count by size
    unique_sizes, counts = np.unique(sizes_array, return_counts=True)
    print(f'\nSize distribution:')
    for size, count in zip(unique_sizes, counts):
        percentage = (count / len(sizes)) * 100
        print(f'  Size {size:2d}: {count:4d} graphs ({percentage:5.1f}%)')
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(sizes_array, bins=range(int(np.min(sizes_array)), int(np.max(sizes_array)) + 2), 
             alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency')
    plt.title('Graph Size Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_size_distribution.png', dpi=150)
    print(f'\nSaved histogram to: graph_size_distribution.png')
    
    # Test filtering
    print('\n' + '='*80)
    print('Testing size filtering:')
    print('='*80)
    
    # Example 1: Small graphs only (5-8 nodes)
    print('\nExample 1: Small graphs (5-8 nodes)')
    dataset_small = GraphDataset(
        traces=graphs,
        activity_to_idx=vocab['activity_to_idx'],
        max_nodes=20,
        include_features=False,
        verbose=True,
        min_graph_size=5,
        max_graph_size=8
    )
    dataset_small.print_statistics()
    
    # Example 2: Medium graphs only (9-12 nodes)
    print('\nExample 2: Medium graphs (9-12 nodes)')
    dataset_medium = GraphDataset(
        traces=graphs,
        activity_to_idx=vocab['activity_to_idx'],
        max_nodes=20,
        include_features=False,
        verbose=True,
        min_graph_size=9,
        max_graph_size=12
    )
    dataset_medium.print_statistics()
    
    # Example 3: Large graphs only (13+ nodes)
    print('\nExample 3: Large graphs (13+ nodes)')
    dataset_large = GraphDataset(
        traces=graphs,
        activity_to_idx=vocab['activity_to_idx'],
        max_nodes=20,
        include_features=False,
        verbose=True,
        min_graph_size=13,
        max_graph_size=None
    )
    dataset_large.print_statistics()


if __name__ == '__main__':
    analyze_sizes()
