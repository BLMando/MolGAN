"""
Graph Process Constraints - Structural and semantic constraints for process graphs

Implements constraints similar to process_constraints.py but adapted for graph structure:
- Start/End node constraints
- Activity frequency matching
- Connectivity constraints
- Structural validity
"""

import tensorflow as tf
import numpy as np


class GraphProcessConstraints:
    """
    Process mining constraints for graph-based generation

    Enforces:
    1. First node should be START
    2. Last active node should be END
    3. Activity frequency distribution matching
    4. Graph connectivity (paths exist)
    5. Structural validity (no isolated nodes)
    """

    def __init__(self,
                 start_idx,
                 end_idx,
                 activity_frequencies,
                 pad_idx=0,
                 lambda_start=1.0,
                 lambda_end=3.0,
                 lambda_frequency=1.0,
                 lambda_connectivity=0.5,
                 lambda_structure=20.0,
                 lambda_unique=10.0,
                 lambda_degree=10.0):
        """
        Args:
            start_idx: Index of START activity
            end_idx: Index of END activity
            activity_frequencies: Target activity frequency distribution
            pad_idx: Index of PAD token
            lambda_*: Weights for different constraint losses
        """
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.activity_frequencies = tf.constant(
            activity_frequencies, dtype=tf.float32)
        self.pad_idx = pad_idx

        # Loss weights
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.lambda_frequency = lambda_frequency
        self.lambda_connectivity = lambda_connectivity
        self.lambda_structure = lambda_structure
        self.lambda_unique = lambda_unique
        self.lambda_degree = lambda_degree

    def start_node_loss(self, nodes):
        """
        Constraint: First node should be START

        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            Loss encouraging first node to be START
        """
        # First node probabilities
        first_node = nodes[:, 0, :]  # (batch, num_activities)

        # Cross-entropy loss: encourage START at position 0
        start_probs = first_node[:, self.start_idx]
        loss = -tf.reduce_mean(tf.math.log(start_probs + 1e-10))

        return loss

    def end_node_loss(self, nodes, adjacency):
        """
        Constraint: Last active node should be END

        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)

        Returns:
            Loss encouraging last active node to be END
        """
        batch_size = tf.shape(nodes)[0]
        max_nodes = tf.shape(nodes)[1]

        # Find last active node (node with no outgoing edges)
        # Sum over all edge types to get total outgoing edges
        # (batch, max_nodes, max_nodes)
        total_adj = tf.reduce_sum(adjacency, axis=-1)
        outgoing_edges = tf.reduce_sum(
            total_adj, axis=-1)  # (batch, max_nodes)

        # Mask for active nodes (not PAD)
        active_mask = 1.0 - nodes[:, :, self.pad_idx]  # (batch, max_nodes)

        # Last active node: active and no outgoing edges
        is_last = active_mask * (1.0 - tf.minimum(outgoing_edges, 1.0))

        # Soft attention: weight each position by likelihood of being last
        attention_weights = tf.nn.softmax(
            is_last, axis=1)  # (batch, max_nodes)
        attention_weights = tf.expand_dims(
            attention_weights, axis=-1)  # (batch, max_nodes, 1)

        # Weighted sum of node probabilities
        weighted_nodes = tf.reduce_sum(
            nodes * attention_weights, axis=1)  # (batch, num_activities)

        # Encourage END activity at last position
        end_probs = weighted_nodes[:, self.end_idx]
        loss = -tf.reduce_mean(tf.math.log(end_probs + 1e-10))

        return loss

    def activity_frequency_loss(self, nodes):
        """
        Constraint: Match target activity frequency distribution

        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            KL divergence between generated and target frequencies
        """
        # Compute generated activity frequencies
        # Average over batch and nodes
        generated_freq = tf.reduce_mean(
            nodes, axis=[0, 1])  # (num_activities,)

        # Normalize
        generated_freq = generated_freq / \
            (tf.reduce_sum(generated_freq) + 1e-10)

        # KL divergence: D_KL(target || generated)
        kl_loss = tf.reduce_sum(
            self.activity_frequencies * tf.math.log(
                (self.activity_frequencies + 1e-10) / (generated_freq + 1e-10)
            )
        )

        return kl_loss

    def degree_constraint_loss(self, adjacency, nodes):
        """
        Strict degree constraints for process graphs:
        - START: exactly 0 in-degree, at least 1 out-degree
        - END: at least 1 in-degree, exactly 0 out-degree
        - Other nodes: at least 1 in-degree and 1 out-degree
        
        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss penalizing degree violations
        """
        # Sum over edge types
        total_adj = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes, max_nodes)
        
        # Compute degrees
        in_degrees = tf.reduce_sum(total_adj, axis=-1)  # (batch, max_nodes)
        out_degrees = tf.reduce_sum(total_adj, axis=-2)  # (batch, max_nodes)
        
        # Identify START and END nodes
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # For each graph, penalize violations
        loss = 0.0
        
        # START constraints: 0 in-degree, >=1 out-degree
        start_in_penalty = tf.maximum(in_degrees * start_probs, 0.0)  # Penalize any in-degree for START
        start_out_penalty = tf.maximum((1.0 - out_degrees) * start_probs, 0.0)  # Penalize 0 out-degree for START
        loss += tf.reduce_mean(start_in_penalty + start_out_penalty)
        
        # END constraints: >=1 in-degree, 0 out-degree
        end_in_penalty = tf.maximum((1.0 - in_degrees) * end_probs, 0.0)  # Penalize 0 in-degree for END
        end_out_penalty = tf.maximum(out_degrees * end_probs, 0.0)  # Penalize any out-degree for END
        loss += tf.reduce_mean(end_in_penalty + end_out_penalty)
        
        # Other nodes: >=1 in-degree and >=1 out-degree
        other_mask = 1.0 - start_probs - end_probs - nodes[:, :, self.pad_idx]  # (batch, max_nodes)
        other_in_penalty = tf.maximum((1.0 - in_degrees) * other_mask, 0.0)
        other_out_penalty = tf.maximum((1.0 - out_degrees) * other_mask, 0.0)
        loss += tf.reduce_mean(other_in_penalty + other_out_penalty)
        
        return loss

    def connectivity_loss(self, adjacency, nodes):
        """
        Constraint: Ensure graph connectivity (approximate)

        Encourages that most active nodes have at least one incoming or outgoing edge

        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            Loss penalizing isolated nodes
        """
        # Sum over edge types
        # (batch, max_nodes, max_nodes)
        total_adj = tf.reduce_sum(adjacency, axis=-1)

        # Outgoing edges per node
        outgoing = tf.reduce_sum(total_adj, axis=-1)  # (batch, max_nodes)

        # Incoming edges per node
        incoming = tf.reduce_sum(total_adj, axis=-2)  # (batch, max_nodes)

        # Node is connected if it has incoming OR outgoing edges
        is_connected = tf.minimum(
            outgoing + incoming, 1.0)  # (batch, max_nodes)

        # Mask for active nodes (not PAD)
        active_mask = 1.0 - nodes[:, :, self.pad_idx]  # (batch, max_nodes)

        # Penalize active nodes that are not connected
        isolated_nodes = active_mask * (1.0 - is_connected)
        loss = tf.reduce_mean(isolated_nodes)

        return loss

    def structural_validity_loss(self, adjacency):
        """
        Constraint: Structural validity of process graphs

        Encourages:
        - No self-loops (except maybe for specific patterns)
        - Reasonable number of edges

        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)

        Returns:
            Loss penalizing invalid structures
        """
        batch_size = tf.shape(adjacency)[0]
        max_nodes = tf.shape(adjacency)[1]

        # Penalize self-loops (diagonal elements)
        # Sum over edge types
        # (batch, max_nodes, max_nodes)
        total_adj = tf.reduce_sum(adjacency, axis=-1)

        # Extract diagonal (self-loops)
        diagonal = tf.linalg.diag_part(total_adj)  # (batch, max_nodes)
        self_loop_penalty = tf.reduce_mean(diagonal)

        return self_loop_penalty

    def unique_start_end_loss(self, nodes):
        """
        Constraint: Ensure only one START and one END per graph
        
        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss penalizing multiple START/END nodes
        """
        # Count START occurrences
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        start_count = tf.reduce_sum(start_probs, axis=1)  # (batch,)
        
        # Count END occurrences
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        end_count = tf.reduce_sum(end_probs, axis=1)  # (batch,)
        
        # Penalize deviation from 1
        start_loss = tf.reduce_mean(tf.square(start_count - 1.0))
        end_loss = tf.reduce_mean(tf.square(end_count - 1.0))
        
        return start_loss + end_loss

    def total_constraint_loss(self, adjacency, nodes):
        """
        Compute total constraint loss

        Args:
            adjacency: Adjacency matrices
            nodes: Node matrices

        Returns:
            Weighted sum of all constraint losses
        """
        # Individual losses
        start_loss = self.start_node_loss(nodes)
        end_loss = self.end_node_loss(nodes, adjacency)
        freq_loss = self.activity_frequency_loss(nodes)
        connect_loss = self.connectivity_loss(adjacency, nodes)
        struct_loss = self.structural_validity_loss(adjacency)
        unique_loss = self.unique_start_end_loss(nodes)
        degree_loss = self.degree_constraint_loss(adjacency, nodes)

        # Weighted sum
        total_loss = (
            self.lambda_start * start_loss +
            self.lambda_end * end_loss +
            self.lambda_frequency * freq_loss +
            self.lambda_connectivity * connect_loss +
            self.lambda_structure * struct_loss +
            self.lambda_unique * unique_loss +
            self.lambda_degree * degree_loss
        )

        return total_loss

    def get_individual_losses(self, adjacency, nodes):
        """
        Get dictionary of individual constraint losses for logging

        Args:
            adjacency: Adjacency matrices
            nodes: Node matrices

        Returns:
            Dictionary with individual losses
        """
        return {
            'start_loss': self.start_node_loss(nodes),
            'end_loss': self.end_node_loss(nodes, adjacency),
            'frequency_loss': self.activity_frequency_loss(nodes),
            'connectivity_loss': self.connectivity_loss(adjacency, nodes),
            'structural_loss': self.structural_validity_loss(adjacency),
            'density_loss': self.edge_density_loss(adjacency),
            'unique_loss': self.unique_start_end_loss(nodes),
            'degree_loss': self.degree_constraint_loss(adjacency, nodes)
        }
