"""
Evaluate Graph Process GAN: Load latest model and generate sample graphs.

Saves graphs in readable .txt format for NetworkX loading (adjacency matrix as edges).
"""

import os
import json
import numpy as np
import tensorflow as tf
from graph_process_gan import GraphProcessGAN
from pathlib import Path

def load_latest_model(output_dir='../../output_graph_gan'):
    """
    Load the latest trained model from output directory.
    
    Returns:
        gan: Loaded GraphProcessGAN model
        metadata: Model metadata dict
    """

    checkpoint_base = Path(output_dir) / 'checkpoints'
    if not checkpoint_base.exists():
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_base}")
    
    subdirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint subdirs in {checkpoint_base}")
    
    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    final_path = latest_dir / 'final'
    
    if not final_path.exists():
        raise FileNotFoundError(f"Final model not found in {final_path}")
    
    metadata_path = final_path / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
 
    gan = GraphProcessGAN(
        max_nodes=metadata['max_nodes'],
        num_activities=metadata['num_activities'],
        start_idx=metadata['vocab']['activity_to_idx']['START'],
        end_idx=metadata['vocab']['activity_to_idx']['END'],
        activity_frequencies=np.array(metadata['activity_frequencies']),
        noise_dim=128,
        generator_hidden_dims=(256, 512, 1024),
        generator_dropout=0.0,
        rgcn_hidden_dims=(128, 64),
        mlp_hidden_dims=(128, 64),
        discriminator_dropout=0.3,
        n_critic=5,
        lambda_gp=10.0,
        lambda_constraint=0.1,
        temp_start=5.0,
        temp_min=0.5,
        temp_decay=0.99995
    )
    
    gan.load_weights(str(final_path / 'model'))
    
    print(f"Loaded model from {final_path}")
    print(f"Final epoch: {metadata['final_epoch']}")
    print(f"Final losses - D: {metadata['final_d_loss']:.4f}, G: {metadata['final_g_loss']:.4f}")
    
    return gan, metadata

def ensure_connectivity(adj_matrix, node_matrix, start_activity='START', end_activity='END', idx_to_activity=None):
    """    
    Args:
        adj_matrix: adjacency matrix. Can be either
            - (max_nodes, max_nodes) binary adjacency, or
            - (max_nodes, max_nodes, num_edge_types) multi-channel adjacency
        node_matrix: (max_nodes, num_activities)
        start_activity, end_activity: Activity names
        idx_to_activity: Dict to get activity names
    
    Returns:
        Modified adj_matrix and node_matrix
    """

    max_nodes, num_activities = node_matrix.shape
    
    if adj_matrix.ndim == 2:
        adj_2d = (adj_matrix > 0).astype(int)
    elif adj_matrix.ndim == 3:
        adj_2d = (np.sum(adj_matrix, axis=-1) > 0).astype(int)
    else:
        adj_2d = (np.sum(adj_matrix, axis=-1) > 0).astype(int)
    
 
    start_idx = None
    end_idx = None
    for node_id in range(max_nodes):
        probs = node_matrix[node_id]
        if np.sum(probs) > 0:
            activity_idx = np.argmax(probs)
            activity = idx_to_activity.get(activity_idx, '')
            if activity == start_activity:
                start_idx = node_id
            elif activity == end_activity:
                end_idx = node_id
    
    if start_idx is None or end_idx is None:
        return adj_matrix, node_matrix
    
    in_degrees = np.sum(adj_2d, axis=0)  
    out_degrees = np.sum(adj_2d, axis=1)
    

    kept_nodes = set()
    for node_id in range(max_nodes):
        if np.sum(node_matrix[node_id]) == 0:
            continue  # Skip padding
        if node_id == start_idx:
            if out_degrees[node_id] >= 1:
                kept_nodes.add(node_id)
        elif node_id == end_idx:
            if in_degrees[node_id] >= 1:
                kept_nodes.add(node_id)
        else:
            if in_degrees[node_id] >= 1 and out_degrees[node_id] >= 1:
                kept_nodes.add(node_id)
    
    reachable = set()
    queue = [start_idx] if start_idx in kept_nodes else []
    visited = set()
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        reachable.add(current)
        for neighbor in range(max_nodes):
            if adj_2d[current, neighbor] > 0 and neighbor in kept_nodes:
                queue.append(neighbor)
    
    kept_nodes = kept_nodes & reachable
    
    if end_idx not in reachable:
        kept_nodes = {start_idx, end_idx} if start_idx in kept_nodes and end_idx in kept_nodes else set()
    
    new_node_matrix = np.zeros_like(node_matrix)
    
    if adj_matrix.ndim == 2:
        new_adj_matrix = np.zeros_like(adj_matrix)
    elif adj_matrix.ndim == 3:
        new_adj_matrix = np.zeros_like(adj_matrix)
    else:
        new_adj_matrix = np.zeros_like(adj_matrix)
    
    old_to_new = {}
    new_node_id = 0
    for old_node in sorted(kept_nodes):
        old_to_new[old_node] = new_node_id
        new_node_matrix[new_node_id] = node_matrix[old_node]
        new_node_id += 1
    
    for old_src in kept_nodes:
        for old_tgt in kept_nodes:
                if old_src in old_to_new and old_tgt in old_to_new:
                    new_src = old_to_new[old_src]
                    new_tgt = old_to_new[old_tgt]
                    new_adj_matrix[new_src, new_tgt] = adj_matrix[old_src, old_tgt]
    
    return new_adj_matrix, new_node_matrix

def deduplicate_activities(adj_matrix, node_matrix):
    """
    Ensure each activity appears at most once in the graph.
    For duplicate activities, keep only the node with highest probability.
    
    Args:
        adj_matrix: (max_nodes, max_nodes, num_edge_types)
        node_matrix: (max_nodes, num_activities)
    
    Returns:
        Modified adj_matrix and node_matrix with unique activities
    """
    max_nodes, num_activities = node_matrix.shape
    if adj_matrix.ndim == 2:
        pass
    elif adj_matrix.ndim == 3:
        pass
    else:
        pass
    
    node_activities = []
    activity_to_nodes = {}
    
    for node_id in range(max_nodes):
        probs = node_matrix[node_id]
        if np.sum(probs) > 0:
            activity_idx = np.argmax(probs)
            prob = probs[activity_idx]
            node_activities.append((node_id, activity_idx, prob))
            if activity_idx not in activity_to_nodes:
                activity_to_nodes[activity_idx] = []
            activity_to_nodes[activity_idx].append((node_id, prob))
    
    kept_nodes = set()
    for activity_idx, nodes_probs in activity_to_nodes.items():
        nodes_probs.sort(key=lambda x: x[1], reverse=True)
        kept_node = nodes_probs[0][0]
        kept_nodes.add(kept_node)
    
    new_node_matrix = np.zeros_like(node_matrix)
    new_adj_matrix = np.zeros_like(adj_matrix)
    
    old_to_new = {}
    new_node_id = 0
    for old_node in sorted(kept_nodes):
        old_to_new[old_node] = new_node_id
        new_node_matrix[new_node_id] = node_matrix[old_node]
        new_node_id += 1
    
    for old_src in kept_nodes:
        for old_tgt in kept_nodes:
                if old_src in old_to_new and old_tgt in old_to_new:
                    new_src = old_to_new[old_src]
                    new_tgt = old_to_new[old_tgt]
                    new_adj_matrix[new_src, new_tgt] = adj_matrix[old_src, old_tgt]
    
    return new_adj_matrix, new_node_matrix

def save_graph_to_txt(adj_matrix, node_matrix, idx_to_activity, filename):
    """
    Save a single graph to .txt in NetworkX-readable format (adjacency as edges).
    
    Args:
        adj_matrix: (max_nodes, max_nodes, num_edge_types) adjacency
        node_matrix: (max_nodes, num_activities) node labels
        idx_to_activity: Dict for activity names
        filename: Output .txt file path
    """
    max_nodes = adj_matrix.shape[0]
    num_activities = node_matrix.shape[1]
    
    with open(filename, 'w') as f:
        f.write("# Graph in NetworkX DiGraph format (adjacency matrix as edges)\n")
        f.write("# Nodes: node_id: activity\n")
        f.write("# Edges: Edge source target\n\n")
        
        for node_id in range(max_nodes):
            activity_probs = node_matrix[node_id]
            if np.sum(activity_probs) > 0:
                activity_idx = np.argmax(activity_probs)
                activity = idx_to_activity.get(activity_idx, f"UNK_{activity_idx}")
                f.write(f"Node {node_id}: {activity}\n")
        
        f.write("\n")
        
        for source in range(max_nodes):
            for target in range(max_nodes):
                if adj_matrix.ndim == 2:
                    has_edge = adj_matrix[source, target] > 0.5
                elif adj_matrix.ndim == 3:
                    has_edge = np.sum(adj_matrix[source, target, :]) > 0.5
                else:
                    has_edge = np.any(adj_matrix[source, target] > 0)

                if has_edge:
                    f.write(f"Edge {source} {target}\n")

def generate_and_save_samples(gan, metadata, num_samples=10, output_dir='../../output_graph_gan/samples'):
    """
    Generate sample graphs and save to .txt files.
    
    Args:
        gan: Trained model
        metadata: Model metadata
        num_samples: Number of graphs to generate
        output_dir: Directory to save samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_adj, sample_nodes, _ = gan.generate_graphs(
        num_samples=num_samples,
        temperature=0.5,
        hard=True
    )
    
    idx_to_activity = {v: k for k, v in metadata['vocab']['activity_to_idx'].items()}
    
    print(f"Generated {num_samples} graphs. Deduplicating activities and saving to {output_dir}")
    
    for i in range(num_samples):
        adj = sample_adj[i]
        nodes = sample_nodes[i]
        
        adj, nodes = deduplicate_activities(adj, nodes)
        
        adj, nodes = ensure_connectivity(adj, nodes, idx_to_activity=idx_to_activity)
        
        filename = os.path.join(output_dir, f'sample_graph_{i}.txt')
        save_graph_to_txt(adj, nodes, idx_to_activity, filename)
        print(f"Saved graph {i} to {filename}")

if __name__ == "__main__":
    gan, metadata = load_latest_model()
    
    generate_and_save_samples(gan, metadata)
    
    print("Evaluation complete! Load .txt files in NetworkX for visualization.")