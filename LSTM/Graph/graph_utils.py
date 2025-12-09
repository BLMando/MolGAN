"""
Utility functions for graph-based GAN

Includes:
- Gumbel-Softmax for differentiable sampling
- Wasserstein loss
- Gradient penalty
- Graph pooling operations
"""

import tensorflow as tf
import numpy as np


def gumbel_softmax(logits, temperature=1.0, hard=False, axis=-1):
    """
    Gumbel-Softmax sampling for differentiable discrete sampling

    Args:
        logits: Logits tensor
        temperature: Temperature parameter (lower = more discrete)
        hard: If True, return one-hot (straight-through estimator)
        axis: Axis to apply softmax

    Returns:
        Sampled tensor (differentiable)
    """
    # Sample from Gumbel(0, 1)
    uniform_samples = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
    gumbel_noise = -tf.math.log(-tf.math.log(uniform_samples + 1e-20) + 1e-20)

    # Add Gumbel noise to logits
    noisy_logits = logits + gumbel_noise

    # Apply softmax with temperature
    y_soft = tf.nn.softmax(noisy_logits / temperature, axis=axis)

    if hard:
        # Straight-through estimator: forward pass uses one-hot, backward uses soft
        y_hard = tf.one_hot(
            tf.argmax(noisy_logits, axis=axis),
            depth=tf.shape(logits)[axis],
            dtype=logits.dtype
        )
        # Stop gradient on difference
        y = tf.stop_gradient(y_hard - y_soft) + y_soft
    else:
        y = y_soft

    return y


def wasserstein_loss(real_scores, fake_scores):
    """
    Wasserstein loss for WGAN

    Args:
        real_scores: Discriminator scores for real data
        fake_scores: Discriminator scores for fake data

    Returns:
        Discriminator loss (minimize this)
    """
    return tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)


def gradient_penalty_graph(discriminator, real_adj, real_nodes, fake_adj, fake_nodes, real_features=None, fake_features=None, lambda_gp=10.0):
    """
    Gradient penalty for WGAN-GP on graph data

    Args:
        discriminator: Discriminator model
        real_adj: Real adjacency matrices (batch, nodes, nodes) - binary
        real_nodes: Real node matrices (batch, nodes, activities)
        fake_adj: Fake adjacency matrices
        fake_nodes: Fake node matrices
        real_features: Real feature matrices (optional)
        fake_features: Fake feature matrices (optional)
        lambda_gp: Gradient penalty coefficient

    Returns:
        Gradient penalty term
    """
    batch_size = tf.shape(real_adj)[0]

    # Random interpolation coefficients (3D for binary adjacency)
    alpha_adj = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    alpha_nodes = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)

    # Interpolated samples
    interpolated_adj = alpha_adj * real_adj + (1 - alpha_adj) * fake_adj
    interpolated_nodes = alpha_nodes * \
        real_nodes + (1 - alpha_nodes) * fake_nodes

    interpolated_features = None
    if real_features is not None and fake_features is not None:
        alpha_features = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated_features = alpha_features * \
            real_features + (1 - alpha_features) * fake_features

    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch([interpolated_adj, interpolated_nodes])
        if interpolated_features is not None:
            tape.watch(interpolated_features)
            scores = discriminator(
                interpolated_adj, interpolated_nodes, interpolated_features, training=True)
        else:
            scores = discriminator(
                interpolated_adj, interpolated_nodes, training=True)

    if interpolated_features is not None:
        gradients = tape.gradient(scores, [interpolated_adj, interpolated_nodes, interpolated_features])
    else:
        gradients = tape.gradient(scores, [interpolated_adj, interpolated_nodes])

    # Compute gradient norm
    gradients_adj = tf.reshape(gradients[0], [batch_size, -1])
    gradients_nodes = tf.reshape(gradients[1], [batch_size, -1])
    
    if interpolated_features is not None:
        gradients_features = tf.reshape(gradients[2], [batch_size, -1])
        gradients_combined = tf.concat([gradients_adj, gradients_nodes, gradients_features], axis=1)
    else:
        gradients_combined = tf.concat([gradients_adj, gradients_nodes], axis=1)

    gradient_norm = tf.sqrt(tf.reduce_sum(
        tf.square(gradients_combined), axis=1) + 1e-12)

    # Penalty term: (||gradient|| - 1)^2
    penalty = tf.reduce_mean(tf.square(gradient_norm - 1.0))

    return penalty


def global_mean_pool(node_features):
    """
    Global mean pooling over graph nodes

    Args:
        node_features: Node feature tensor (batch, num_nodes, features)

    Returns:
        Pooled features (batch, features)
    """
    return tf.reduce_mean(node_features, axis=1)


def global_sum_pool(node_features):
    """
    Global sum pooling over graph nodes

    Args:
        node_features: Node feature tensor (batch, num_nodes, features)

    Returns:
        Pooled features (batch, features)
    """
    return tf.reduce_sum(node_features, axis=1)


def global_max_pool(node_features):
    """
    Global max pooling over graph nodes

    Args:
        node_features: Node feature tensor (batch, num_nodes, features)

    Returns:
        Pooled features (batch, features)
    """
    return tf.reduce_max(node_features, axis=1)


def attention_pool(node_features, num_hidden=128):
    """
    Attention-based global pooling

    Args:
        node_features: Node feature tensor (batch, num_nodes, features)
        num_hidden: Hidden dimension for attention

    Returns:
        Pooled features (batch, features)
    """
    # Simple attention: MLP -> softmax weights
    batch_size = tf.shape(node_features)[0]
    num_nodes = tf.shape(node_features)[1]
    feature_dim = tf.shape(node_features)[2]

    # Attention weights
    attention_logits = tf.keras.layers.Dense(
        1)(node_features)  # (batch, nodes, 1)
    attention_weights = tf.nn.softmax(
        attention_logits, axis=1)  # (batch, nodes, 1)

    # Weighted sum
    pooled = tf.reduce_sum(
        node_features * attention_weights, axis=1)  # (batch, features)

    return pooled


def compute_graph_statistics(adj_matrices, node_matrices):
    """
    Compute statistics from generated graphs

    Args:
        adj_matrices: Adjacency matrices. Can be either
            - (batch, nodes, nodes) binary adjacency, or
            - (batch, nodes, nodes, edge_types) multi-channel adjacency
        node_matrices: Node matrices (batch, nodes, activities)

    Returns:
        Dictionary with statistics
    """
    # Number of edges per graph
    if adj_matrices.ndim == 4:
        num_edges = tf.reduce_sum(adj_matrices, axis=[1, 2, 3])
        edge_type_counts = tf.reduce_sum(adj_matrices, axis=[0, 1, 2])
    elif adj_matrices.ndim == 3:
        num_edges = tf.reduce_sum(adj_matrices, axis=[1, 2])
        edge_type_counts = None
    else:
        # Fallback: flatten
        num_edges = tf.reduce_sum(adj_matrices, axis=list(range(1, tf.rank(adj_matrices))))
        edge_type_counts = None

    # Number of nodes per graph (non-padding)
    # Assuming first activity index (0) is <PAD>
    node_mask = 1.0 - node_matrices[:, :, 0]  # (batch, nodes)
    num_nodes = tf.reduce_sum(node_mask, axis=1)

    stats = {
        'avg_num_edges': tf.reduce_mean(num_edges),
        'avg_num_nodes': tf.reduce_mean(num_nodes),
        'edge_type_distribution': edge_type_counts
    }

    return stats


def build_edge_index(adj_matrix):
    """
    Convert adjacency matrix to edge index format (for debugging/visualization)

    Args:
        adj_matrix: Adjacency matrix (nodes, nodes, edge_types)

    Returns:
        edge_index: Array of shape (num_edges, 2) with [source, target]
        edge_types: Array of shape (num_edges,) with edge type indices
    """
    # Find non-zero entries. Support 2D binary adjacency or 3D multi-channel.
    shape = adj_matrix.shape
    edge_list = []
    edge_type_list = []

    if adj_matrix.ndim == 2:
        nodes = shape[0]
        for i in range(nodes):
            for j in range(nodes):
                if adj_matrix[i, j] > 0:
                    edge_list.append([i, j])
                    edge_type_list.append(None)
    elif adj_matrix.ndim == 3:
        nodes = shape[0]
        edge_types_dim = shape[2]
        for i in range(nodes):
            for j in range(nodes):
                for k in range(edge_types_dim):
                    if adj_matrix[i, j, k] > 0:
                        edge_list.append([i, j])
                        edge_type_list.append(k)
    else:
        # Fallback: try to flatten last axis
        nodes = shape[0]
        for i in range(nodes):
            for j in range(nodes):
                if np.any(adj_matrix[i, j] > 0):
                    edge_list.append([i, j])
                    edge_type_list.append(None)

    if len(edge_list) == 0:
        return np.array([]).reshape(0, 2), np.array([])

    return np.array(edge_list), np.array(edge_type_list)


def visualize_graph(adj_matrix, node_matrix, idx_to_activity, edge_type_names):
    """
    Print graph structure (for debugging)

    Args:
        adj_matrix: Adjacency matrix (nodes, nodes, edge_types)
        node_matrix: Node matrix (nodes, activities)
        idx_to_activity: Activity index to name mapping
        edge_type_names: List of edge type names
    """
    num_nodes = adj_matrix.shape[0]

    print('\nGraph Structure:')
    print('-' * 60)

    # Print nodes
    print('Nodes:')
    for i in range(num_nodes):
        activity_idx = np.argmax(node_matrix[i])
        activity = idx_to_activity.get(activity_idx, f'UNK_{activity_idx}')
        if activity != '<PAD>':
            print(f'  Node {i}: {activity}')

    # Print edges
    print('\nEdges:')
    edge_index, edge_types = build_edge_index(adj_matrix)
    for (src, tgt), edge_type in zip(edge_index, edge_types):
        src_activity = idx_to_activity.get(np.argmax(node_matrix[src]), 'UNK')
        tgt_activity = idx_to_activity.get(np.argmax(node_matrix[tgt]), 'UNK')
        if edge_type is None:
            edge_name = 'EDGE'
        else:
            edge_name = edge_type_names[edge_type] if edge_type < len(
                edge_type_names) else f'TYPE_{edge_type}'
        print(
            f'  {src} ({src_activity}) --[{edge_name}]--> {tgt} ({tgt_activity})')

    print('-' * 60)
