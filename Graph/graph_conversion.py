"""
Graph Conversion Utilities

Convert between different graph data formats:
- Matrix format (adjacency + node matrices) from GAN generator
- Trace dictionary format from ig_loader
- NetworkX DiGraph format for ig_metrics evaluation
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple


def matrix_to_networkx(
    adj_matrix: np.ndarray,
    node_matrix: np.ndarray,
    idx_to_activity: Dict[int, str],
    pad_activity: str = 'PAD'
) -> nx.DiGraph:
    """
    Convert generated graph matrices to NetworkX DiGraph.
    
    Args:
        adj_matrix: Adjacency matrix, shape (max_nodes, max_nodes) for binary
                   or (max_nodes, max_nodes, num_edge_types) for multi-channel
        node_matrix: Node feature matrix, shape (max_nodes, num_activities)
                    One-hot encoded activity for each node
        idx_to_activity: Mapping from activity index to activity name
        pad_activity: Name of the PAD activity to filter out
    
    Returns:
        nx.DiGraph with 'label' attribute on each node
    """
    G = nx.DiGraph()
    max_nodes = node_matrix.shape[0]
    
    # Find PAD index
    pad_idx = None
    for idx, name in idx_to_activity.items():
        if name.upper() == pad_activity.upper():
            pad_idx = idx
            break
    
    # Track valid (non-PAD) nodes
    valid_nodes = []
    
    # Add nodes
    for node_id in range(max_nodes):
        node_probs = node_matrix[node_id]
        
        # Skip nodes with all zeros (empty slots)
        if np.sum(node_probs) < 0.01:
            continue
        
        activity_idx = int(np.argmax(node_probs))
        
        # Skip PAD nodes
        if pad_idx is not None and activity_idx == pad_idx:
            continue
        
        activity_name = idx_to_activity.get(activity_idx, f'UNK_{activity_idx}')
        G.add_node(node_id, label=activity_name)
        valid_nodes.append(node_id)
    
    # Add edges (only between valid nodes)
    valid_set = set(valid_nodes)
    
    # Handle both 2D binary and 3D multi-channel adjacency
    if adj_matrix.ndim == 3:
        # Multi-channel: sum across edge types
        binary_adj = np.sum(adj_matrix, axis=-1)
    else:
        binary_adj = adj_matrix
    
    for source in valid_nodes:
        for target in valid_nodes:
            if binary_adj[source, target] > 0.5:
                G.add_edge(source, target)
    
    return G


def trace_to_networkx(trace: Dict) -> nx.DiGraph:
    """
    Convert trace dictionary from ig_loader to NetworkX DiGraph.
    
    Args:
        trace: Dictionary with 'vertices' and 'edges' lists from ig_loader
               vertices: [{'node_id': float, 'activity': str, ...}, ...]
               edges: [{'source': float, 'target': float, ...}, ...]
    
    Returns:
        nx.DiGraph with 'label' attribute on each node
    """
    G = nx.DiGraph()
    
    # Node ID mapping (original float IDs to integer indices)
    node_id_map = {}
    
    # Add vertices
    for vertex in trace.get('vertices', []):
        original_id = vertex['node_id']
        # Convert to int for consistent node IDs
        node_id = int(original_id) if isinstance(original_id, (int, float)) else original_id
        activity = vertex.get('activity', 'UNKNOWN')
        
        G.add_node(node_id, label=activity)
        node_id_map[original_id] = node_id
    
    # Add edges
    for edge in trace.get('edges', []):
        source_orig = edge['source']
        target_orig = edge['target']
        
        source = node_id_map.get(source_orig, int(source_orig))
        target = node_id_map.get(target_orig, int(target_orig))
        
        # Only add edge if both nodes exist
        if G.has_node(source) and G.has_node(target):
            G.add_edge(source, target)
    
    return G


def batch_matrices_to_networkx(
    adj_matrices: np.ndarray,
    node_matrices: np.ndarray,
    idx_to_activity: Dict[int, str],
    pad_activity: str = 'PAD'
) -> List[nx.DiGraph]:
    """
    Convert batch of generated matrices to list of NetworkX graphs.
    
    Args:
        adj_matrices: Shape (batch, max_nodes, max_nodes) or 
                     (batch, max_nodes, max_nodes, num_edge_types)
        node_matrices: Shape (batch, max_nodes, num_activities)
        idx_to_activity: Mapping from activity index to activity name
        pad_activity: Name of the PAD activity to filter out
    
    Returns:
        List of nx.DiGraph objects
    """
    batch_size = adj_matrices.shape[0]
    graphs = []
    
    for i in range(batch_size):
        G = matrix_to_networkx(
            adj_matrix=adj_matrices[i],
            node_matrix=node_matrices[i],
            idx_to_activity=idx_to_activity,
            pad_activity=pad_activity
        )
        graphs.append(G)
    
    return graphs


def traces_to_networkx(traces: List[Dict]) -> List[nx.DiGraph]:
    """
    Convert list of traces to list of NetworkX graphs.
    
    Args:
        traces: List of trace dictionaries from ig_loader
    
    Returns:
        List of nx.DiGraph objects
    """
    return [trace_to_networkx(trace) for trace in traces]


def txt_file_to_networkx(filename: str) -> nx.DiGraph:
    """
    Load graph from .txt file (saved by evaluate_graph_gan.py) to NetworkX.
    
    The file format is:
        Node <id>: <activity>
        Edge <source> <target>
    
    Args:
        filename: Path to .txt graph file
    
    Returns:
        nx.DiGraph with 'label' attribute on each node
    """
    G = nx.DiGraph()
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Node'):
                # Parse: "Node <id>: <activity>"
                parts = line.split(': ', 1)
                node_id = int(parts[0].split()[1])
                activity = parts[1] if len(parts) > 1 else 'UNKNOWN'
                G.add_node(node_id, label=activity)
            elif line.startswith('Edge'):
                # Parse: "Edge <source> <target>"
                parts = line.split()
                source = int(parts[1])
                target = int(parts[2])
                G.add_edge(source, target)
    
    return G


def txt_files_to_networkx(filenames: List[str]) -> List[nx.DiGraph]:
    """
    Load multiple graph files to NetworkX graphs.
    
    Args:
        filenames: List of paths to .txt graph files
    
    Returns:
        List of nx.DiGraph objects
    """
    return [txt_file_to_networkx(f) for f in filenames]
