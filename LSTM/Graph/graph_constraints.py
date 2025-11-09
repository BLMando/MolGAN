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
                 lambda_degree=10.0,
                 lambda_path=5.0,
                 lambda_start_connect=15.0,
                 lambda_end_connect=15.0,
                 lambda_node_on_path=10.0,
                 lambda_end_no_out=25.0,
                 lambda_end_unique=25.0):
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
        self.lambda_path = lambda_path
        self.lambda_start_connect = lambda_start_connect
        self.lambda_end_connect = lambda_end_connect
        self.lambda_node_on_path = lambda_node_on_path
        self.lambda_end_no_out = lambda_end_no_out
        self.lambda_end_unique = lambda_end_unique

    def start_node_loss(self, nodes):
        """
        Constraint: First node should be START
        
        Enhanced with:
        1. Cross-entropy for first node being START
        2. Penalty for START appearing elsewhere
        
        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            Loss encouraging first node to be START (and only first node)
        """
        # First node probabilities
        first_node = nodes[:, 0, :]  # (batch, num_activities)

        # Cross-entropy loss: encourage START at position 0
        start_probs_first = first_node[:, self.start_idx]
        loss_first = -tf.reduce_mean(tf.math.log(start_probs_first + 1e-10))
        
        # Also penalize START appearing at other positions (not first)
        other_nodes = nodes[:, 1:, :]  # (batch, max_nodes-1, num_activities)
        start_probs_others = other_nodes[:, :, self.start_idx]  # (batch, max_nodes-1)
        loss_others = tf.reduce_mean(start_probs_others)  # Penalize any START elsewhere
        
        return loss_first + 2.0 * loss_others

    def end_node_loss(self, nodes, adjacency):
        """
        Constraint: Last active node should be END
        
        Enhanced version:
        1. Identifies nodes with zero out-degree
        2. Among those, encourages END label
        3. Penalizes END appearing at nodes WITH out-degree

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
        loss_encourage_end = -tf.reduce_mean(tf.math.log(end_probs + 1e-10))
        
        # NEW: Heavily penalize END nodes that HAVE outgoing edges
        end_node_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        has_outgoing = tf.minimum(outgoing_edges, 1.0)  # Binary: 1 if has out-edges
        end_with_outgoing = end_node_probs * has_outgoing  # END nodes with out-edges
        loss_end_with_edges = tf.reduce_mean(end_with_outgoing)
        
        return loss_encourage_end + 5.0 * loss_end_with_edges

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
        end_out_penalty = out_degrees * end_probs  # Penalize ANY out-degree for END (no tf.maximum needed - we want to penalize all)
        # Use squared penalty for out-degree to make it more severe
        end_out_penalty_squared = tf.square(end_out_penalty)
        loss += tf.reduce_mean(end_in_penalty + 3.0 * end_out_penalty_squared)
        
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
        
        Uses a more stringent approach:
        1. Penalize squared deviation from count=1
        2. Add entropy penalty to encourage concentration
        
        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss penalizing multiple START/END nodes
        """
        # Count START occurrences (sum of probabilities across all nodes)
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        start_count = tf.reduce_sum(start_probs, axis=1)  # (batch,)
        
        # Count END occurrences
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        end_count = tf.reduce_sum(end_probs, axis=1)  # (batch,)
        
        # Penalize deviation from 1 (squared loss for stronger penalty)
        count_loss = tf.reduce_mean(tf.square(start_count - 1.0)) + \
                     tf.reduce_mean(tf.square(end_count - 1.0))
        
        # Add concentration penalty: encourage one node to have high probability
        # rather than spreading probability across multiple nodes
        # Entropy is high when probability is spread, low when concentrated
        # We want low entropy, so we penalize high entropy
        
        # For START: we want one node to have prob ~1, others ~0
        # Compute negative max probability (we want max to be close to 1)
        max_start_prob = tf.reduce_max(start_probs, axis=1)  # (batch,)
        concentration_loss_start = tf.reduce_mean(1.0 - max_start_prob)
        
        # For END
        max_end_prob = tf.reduce_max(end_probs, axis=1)  # (batch,)
        concentration_loss_end = tf.reduce_mean(1.0 - max_end_prob)
        
        # Total loss combines count constraint and concentration
        total_loss = count_loss + 2.0 * (concentration_loss_start + concentration_loss_end)
        
        return total_loss

    def end_no_outgoing_loss(self, adjacency, nodes):
        """
        CRITICAL constraint: END nodes must have ZERO outgoing edges
        
        This is a hard constraint for process graphs - END is terminal.
        Uses very strong penalty with multiple approaches:
        1. Direct penalty on out-degree of END nodes
        2. Penalty on edges FROM any node labeled as END
        3. Squared penalty for severity
        
        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss heavily penalizing END nodes with outgoing edges
        """
        # Identify END nodes
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # Sum over edge types to get total adjacency
        total_adj = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes, max_nodes)
        
        # Compute out-degree for each node
        out_degrees = tf.reduce_sum(total_adj, axis=-1)  # (batch, max_nodes)
        
        # For END nodes: penalize ANY out-degree
        # Use exponential penalty: penalty grows exponentially with out-degree
        end_out_penalty = end_probs * out_degrees  # (batch, max_nodes)
        
        # Apply multiple penalty formulations for stronger effect:
        # 1. Linear penalty
        loss_linear = tf.reduce_mean(end_out_penalty)
        
        # 2. Squared penalty (more severe for high out-degrees)
        loss_squared = tf.reduce_mean(tf.square(end_out_penalty))
        
        # 3. Per-edge penalty: penalize each edge from END nodes
        # For each END node, sum ALL its outgoing edges
        end_probs_expanded = tf.expand_dims(end_probs, axis=-1)  # (batch, max_nodes, 1)
        edges_from_end = total_adj * end_probs_expanded  # (batch, max_nodes, max_nodes)
        loss_per_edge = tf.reduce_mean(edges_from_end)
        
        # Combine all penalties
        total_loss = loss_linear + 2.0 * loss_squared + 3.0 * loss_per_edge
        
        return total_loss

    def end_uniqueness_loss(self, nodes):
        """
        CRITICAL constraint: Only ONE END node per graph
        
        Uses multiple approaches:
        1. Count constraint (sum should equal 1)
        2. Concentration constraint (max probability should be ~1)
        3. Sparsity constraint (penalize spread of END across nodes)
        
        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss encouraging exactly one END node
        """
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # 1. Count constraint: sum of END probabilities should be 1
        end_count = tf.reduce_sum(end_probs, axis=1)  # (batch,)
        count_loss = tf.reduce_mean(tf.square(end_count - 1.0))
        
        # 2. Concentration: one node should have high probability
        max_end_prob = tf.reduce_max(end_probs, axis=1)  # (batch,)
        concentration_loss = tf.reduce_mean(1.0 - max_end_prob)
        
        # 3. Sparsity: penalize having many nodes with non-zero END probability
        # Use entropy-like measure: we want low entropy (concentrated)
        # Clipped to avoid log(0)
        end_probs_safe = tf.clip_by_value(end_probs, 1e-10, 1.0)
        entropy = -tf.reduce_sum(end_probs * tf.math.log(end_probs_safe), axis=1)
        entropy_loss = tf.reduce_mean(entropy)
        
        total_loss = count_loss + 3.0 * concentration_loss + 2.0 * entropy_loss
        
        return total_loss

    def start_connectivity_loss(self, adjacency, nodes):
        """
        Strict constraint: START must have outgoing edges to non-PAD nodes
        
        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss heavily penalizing START with no outgoing connections
        """
        # Identify START node (should be at position 0, but let's be flexible)
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        
        # Sum over edge types to get total adjacency
        total_adj = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes, max_nodes)
        
        # For each potential START node, get its outgoing edges
        # (batch, max_nodes, max_nodes) - outgoing from each node
        outgoing_adj = total_adj
        
        # Mask for non-PAD target nodes
        non_pad_mask = 1.0 - nodes[:, :, self.pad_idx]  # (batch, max_nodes)
        non_pad_mask = tf.expand_dims(non_pad_mask, axis=1)  # (batch, 1, max_nodes)
        
        # Outgoing edges to non-PAD nodes
        valid_outgoing = outgoing_adj * non_pad_mask  # (batch, max_nodes, max_nodes)
        
        # Sum outgoing edges per source node
        total_valid_out = tf.reduce_sum(valid_outgoing, axis=-1)  # (batch, max_nodes)
        
        # For START nodes, heavily penalize if no outgoing edges
        # Use sigmoid to create smooth penalty: 1 when total_valid_out=0, ~0 when total_valid_out>=1
        start_no_out_penalty = start_probs * tf.nn.sigmoid(-(total_valid_out - 0.5) * 10.0)
        
        loss = tf.reduce_mean(start_no_out_penalty)
        
        return loss

    def end_connectivity_loss(self, adjacency, nodes):
        """
        Strict constraint: END must have incoming edges from non-PAD/non-END nodes
        
        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss heavily penalizing END with no incoming connections
        """
        # Identify END node
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # Sum over edge types
        total_adj = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes, max_nodes)
        
        # Incoming edges: transpose to get (batch, target, source)
        incoming_adj = tf.transpose(total_adj, perm=[0, 2, 1])
        
        # Mask for valid source nodes (non-PAD and non-END)
        valid_source_mask = 1.0 - nodes[:, :, self.pad_idx] - end_probs  # (batch, max_nodes)
        valid_source_mask = tf.expand_dims(valid_source_mask, axis=1)  # (batch, 1, max_nodes)
        
        # Incoming edges from valid sources
        valid_incoming = incoming_adj * valid_source_mask  # (batch, max_nodes, max_nodes)
        
        # Sum incoming edges per target node
        total_valid_in = tf.reduce_sum(valid_incoming, axis=-1)  # (batch, max_nodes)
        
        # For END nodes, heavily penalize if no incoming edges
        end_no_in_penalty = end_probs * tf.nn.sigmoid(-(total_valid_in - 0.5) * 10.0)
        
        loss = tf.reduce_mean(end_no_in_penalty)
        
        return loss

    def path_existence_loss(self, adjacency, nodes):
        """
        Constraint: Ensure path exists from START to END
        
        Uses approximate reachability via adjacency matrix powers.
        A^k tells us which nodes are reachable in k steps.
        
        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss penalizing disconnected START-END
        """
        # Binary adjacency (any edge type)
        total_adj = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes, max_nodes)
        binary_adj = tf.minimum(total_adj, 1.0)  # Clip to binary
        
        # Compute reachability matrix using transitive closure approximation
        # A + A^2 + A^3 + ... captures multi-hop connections
        reachable = binary_adj
        current = binary_adj
        
        # Compute up to 8 hops (sufficient for most process graphs)
        # Using fixed number to avoid symbolic tensor issues
        for _ in range(8):
            current = tf.matmul(current, binary_adj)
            current = tf.minimum(current, 1.0)  # Keep binary
            reachable = tf.minimum(reachable + current, 1.0)
        
        # Get START and END positions
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # For each batch, compute weighted reachability from START to END
        # Expand dims for matrix multiplication
        start_probs_exp = tf.expand_dims(start_probs, axis=1)  # (batch, 1, max_nodes)
        end_probs_exp = tf.expand_dims(end_probs, axis=-1)  # (batch, max_nodes, 1)
        
        # Weighted reachability: start -> reachable -> end
        # (batch, 1, max_nodes) @ (batch, max_nodes, max_nodes) @ (batch, max_nodes, 1)
        path_strength = tf.matmul(start_probs_exp, reachable)  # (batch, 1, max_nodes)
        path_strength = tf.matmul(path_strength, end_probs_exp)  # (batch, 1, 1)
        path_strength = tf.squeeze(path_strength, axis=[1, 2])  # (batch,)
        
        # Penalize weak paths: loss is high when path_strength is low
        loss = tf.reduce_mean(1.0 - path_strength)
        
        return loss

    def nodes_on_path_loss(self, adjacency, nodes):
        """
        Constraint: Every active node must be on at least one path from START to END
        
        This ensures no "dead" nodes that don't contribute to any valid process flow.
        A node is on a path if it's reachable from START AND can reach END.
        
        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss penalizing nodes not on any START-END path
        """
        # Binary adjacency (any edge type)
        total_adj = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes, max_nodes)
        binary_adj = tf.minimum(total_adj, 1.0)
        
        # 1. Compute forward reachability from START (nodes reachable from START)
        forward_reach = binary_adj
        current_forward = binary_adj
        for _ in range(8):
            current_forward = tf.matmul(current_forward, binary_adj)
            current_forward = tf.minimum(current_forward, 1.0)
            forward_reach = tf.minimum(forward_reach + current_forward, 1.0)
        
        # 2. Compute backward reachability to END (nodes that can reach END)
        # Transpose adjacency for backward propagation
        binary_adj_T = tf.transpose(binary_adj, perm=[0, 2, 1])
        backward_reach = binary_adj_T
        current_backward = binary_adj_T
        for _ in range(8):
            current_backward = tf.matmul(current_backward, binary_adj_T)
            current_backward = tf.minimum(current_backward, 1.0)
            backward_reach = tf.minimum(backward_reach + current_backward, 1.0)
        # Transpose back to (batch, from, to)
        backward_reach = tf.transpose(backward_reach, perm=[0, 2, 1])
        
        # Get START and END node positions
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # Expand for matrix operations
        start_probs_exp = tf.expand_dims(start_probs, axis=1)  # (batch, 1, max_nodes)
        end_probs_exp = tf.expand_dims(end_probs, axis=-1)  # (batch, max_nodes, 1)
        
        # For each node: can it be reached from START?
        # (batch, 1, max_nodes) @ (batch, max_nodes, max_nodes) -> (batch, 1, max_nodes)
        reachable_from_start = tf.matmul(start_probs_exp, forward_reach)
        reachable_from_start = tf.squeeze(reachable_from_start, axis=1)  # (batch, max_nodes)
        
        # For each node: can it reach END?
        # (batch, max_nodes, max_nodes) @ (batch, max_nodes, 1) -> (batch, max_nodes, 1)
        can_reach_end = tf.matmul(backward_reach, end_probs_exp)
        can_reach_end = tf.squeeze(can_reach_end, axis=-1)  # (batch, max_nodes)
        
        # Node is on a START-END path if BOTH:
        # 1. Reachable from START, AND
        # 2. Can reach END
        on_path = reachable_from_start * can_reach_end  # (batch, max_nodes)
        
        # Identify active nodes (not PAD, not START, not END)
        # These are the intermediate nodes that should be on paths
        active_mask = 1.0 - nodes[:, :, self.pad_idx]  # All non-PAD
        intermediate_mask = active_mask - start_probs - end_probs  # Remove START and END
        intermediate_mask = tf.maximum(intermediate_mask, 0.0)  # Ensure non-negative
        
        # Penalize intermediate nodes NOT on any path
        nodes_not_on_path = intermediate_mask * (1.0 - on_path)
        loss = tf.reduce_mean(nodes_not_on_path)
        
        return loss

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
        path_loss = self.path_existence_loss(adjacency, nodes)
        start_connect_loss = self.start_connectivity_loss(adjacency, nodes)
        end_connect_loss = self.end_connectivity_loss(adjacency, nodes)
        node_on_path_loss = self.nodes_on_path_loss(adjacency, nodes)
        end_no_out_loss = self.end_no_outgoing_loss(adjacency, nodes)
        end_unique_loss = self.end_uniqueness_loss(nodes)

        # Weighted sum
        total_loss = (
            self.lambda_start * start_loss +
            self.lambda_end * end_loss +
            self.lambda_frequency * freq_loss +
            self.lambda_connectivity * connect_loss +
            self.lambda_structure * struct_loss +
            self.lambda_unique * unique_loss +
            self.lambda_degree * degree_loss +
            self.lambda_path * path_loss +
            self.lambda_start_connect * start_connect_loss +
            self.lambda_end_connect * end_connect_loss +
            self.lambda_node_on_path * node_on_path_loss +
            self.lambda_end_no_out * end_no_out_loss +
            self.lambda_end_unique * end_unique_loss
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
            'unique_loss': self.unique_start_end_loss(nodes),
            'degree_loss': self.degree_constraint_loss(adjacency, nodes),
            'path_loss': self.path_existence_loss(adjacency, nodes),
            'start_connect_loss': self.start_connectivity_loss(adjacency, nodes),
            'end_connect_loss': self.end_connectivity_loss(adjacency, nodes),
            'node_on_path_loss': self.nodes_on_path_loss(adjacency, nodes),
            'end_no_out_loss': self.end_no_outgoing_loss(adjacency, nodes),
            'end_unique_loss': self.end_uniqueness_loss(nodes)
        }
