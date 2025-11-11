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
                 lambda_end=2.0,
                 lambda_frequency=1.0,
                 lambda_connectivity=0.5,
                 lambda_structure=5.0,
                 lambda_unique=5.0,
                 lambda_degree=5.0,
                 lambda_path=3.0,
                 lambda_start_connect=5.0,
                 lambda_end_connect=8.0,
                 lambda_node_on_path=5.0,
                 lambda_end_no_out=8.0,
                 lambda_end_unique=8.0,
                 lambda_edge_continuity=12.0):
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
        self.lambda_edge_continuity = lambda_edge_continuity

    def start_node_loss(self, nodes):
        """
        Constraint: First node should be START and ONLY first node
        
        Strong enforcement with:
        1. Cross-entropy for first node being START
        2. Heavy penalty for START appearing elsewhere
        3. Concentration penalty to ensure single START
        
        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            Loss encouraging first node to be START (and only first node)
        """
        # Extract START probabilities for all positions
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        
        # 1. First node must be START - use strong cross-entropy
        start_probs_first = start_probs[:, 0]  # (batch,)
        loss_first = -tf.reduce_mean(tf.math.log(start_probs_first + 1e-10))
        
        # 2. Heavily penalize START appearing at other positions
        start_probs_others = start_probs[:, 1:]  # (batch, max_nodes-1)
        # Use squared penalty for stronger effect
        loss_others = tf.reduce_mean(tf.square(start_probs_others))
        
        # 3. Ensure START probability sums to ~1 (count constraint)
        start_count = tf.reduce_sum(start_probs, axis=1)  # (batch,)
        loss_count = tf.reduce_mean(tf.square(start_count - 1.0))
        
        # 4. Concentration: max START probability should be at position 0
        max_start_elsewhere = tf.reduce_max(start_probs_others, axis=1)  # (batch,)
        loss_concentration = tf.reduce_mean(max_start_elsewhere)
        
        # Combine all penalties
        return loss_first + 3.0 * loss_others + 2.0 * loss_count + 2.0 * loss_concentration

    def end_node_loss(self, nodes, adjacency):
        """
        Constraint: Last active node should be END
        
        Soft version that encourages but doesn't force:
        1. Identifies nodes with low/zero out-degree
        2. Among those, encourages END label
        3. Gently discourages (not forbids) END at nodes with out-degree

        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)

        Returns:
            Loss encouraging last active node to be END
        """
        # Find nodes with low out-degree (terminal candidates)
        total_adj = tf.reduce_sum(adjacency, axis=-1)
        outgoing_edges = tf.reduce_sum(total_adj, axis=2)  # (batch, max_nodes)

        # Mask for active nodes (not PAD)
        active_mask = 1.0 - nodes[:, :, self.pad_idx]  # (batch, max_nodes)

        # Terminal score: active nodes with low out-degree
        # Use soft weighting instead of hard threshold
        terminal_score = active_mask * tf.nn.sigmoid(-(outgoing_edges - 0.5) * 5.0)

        # Soft attention: weight each position by likelihood of being terminal
        attention_weights = tf.nn.softmax(terminal_score + 1e-10, axis=1)  # (batch, max_nodes)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)  # (batch, max_nodes, 1)

        # Weighted sum of node probabilities
        weighted_nodes = tf.reduce_sum(nodes * attention_weights, axis=1)  # (batch, num_activities)

        # Encourage END activity at terminal positions
        end_probs = weighted_nodes[:, self.end_idx]
        loss_encourage_end = -tf.reduce_mean(tf.math.log(end_probs + 1e-10))
        
        # Penalty for END nodes with any outgoing edges (zero tolerance)
        end_node_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        # Penalize if out_degree > 0 (no flexibility)
        end_with_any_outgoing = end_node_probs * tf.nn.relu(outgoing_edges - 0.0)
        loss_end_with_edges = tf.reduce_mean(end_with_any_outgoing)
        
        return loss_encourage_end + 2.0 * loss_end_with_edges

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
        Soft degree constraints for process graphs:
        - START: exactly 0 in-degree, at least 1 out-degree
        - END: exactly 1 in-degree, exactly 0 out-degree
        - Other nodes: at least 1 in-degree and 1 out-degree
        
        Uses SOFT penalties that allow gradual learning
        
        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss penalizing degree violations
        """
        # Sum over edge types
        total_adj = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes, max_nodes)
        
        # Compute degrees
        in_degrees = tf.reduce_sum(total_adj, axis=1)  # (batch, max_nodes) - sum over sources
        out_degrees = tf.reduce_sum(total_adj, axis=2)  # (batch, max_nodes) - sum over targets
        
        # Identify START and END nodes
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # For each graph, penalize violations
        loss = 0.0
        
        # START constraints: 0 in-degree, >=1 out-degree
        # Use smooth penalty: sigmoid-based instead of hard threshold
        start_in_penalty = start_probs * in_degrees
        start_out_penalty = start_probs * tf.nn.relu(1.0 - out_degrees)  # Only penalize if < 1
        loss += tf.reduce_mean(start_in_penalty + start_out_penalty)
        
        # END constraints: EXACTLY 1 in-degree, EXACTLY 0 out-degree
        # Stronger penalties for END to ensure uniqueness and finality
        # Penalize deviation from exactly 1 in-degree
        end_in_penalty = end_probs * tf.square(in_degrees - 1.0)  # Square penalty for stronger effect
        # Penalize any out-degree (must be exactly 0)
        end_out_penalty = end_probs * tf.square(out_degrees)  # Square penalty for stronger effect
        loss += tf.reduce_mean(end_in_penalty + end_out_penalty)
        
        # Other nodes: >=1 in-degree and >=1 out-degree
        other_mask = 1.0 - start_probs - end_probs - nodes[:, :, self.pad_idx]  # (batch, max_nodes)
        other_mask = tf.maximum(other_mask, 0.0)  # Ensure non-negative
        other_in_penalty = other_mask * tf.nn.relu(1.0 - in_degrees)
        other_out_penalty = other_mask * tf.nn.relu(1.0 - out_degrees)
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
        
        Soft version that allows gradual learning:
        1. Gentle count constraint (L1 instead of L2)
        2. Concentration encouragement
        
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
        
        # Soft count constraint using L1 (less aggressive than L2)
        count_loss = tf.reduce_mean(tf.abs(start_count - 1.0)) + \
                     tf.reduce_mean(tf.abs(end_count - 1.0))
        
        # Concentration: encourage one node to dominate
        # For START
        max_start_prob = tf.reduce_max(start_probs, axis=1)  # (batch,)
        concentration_loss_start = tf.reduce_mean(tf.nn.relu(0.5 - max_start_prob))
        
        # For END
        max_end_prob = tf.reduce_max(end_probs, axis=1)  # (batch,)
        concentration_loss_end = tf.reduce_mean(tf.nn.relu(0.5 - max_end_prob))
        
        # Softer combination
        total_loss = count_loss + concentration_loss_start + concentration_loss_end
        
        return total_loss

    def end_no_outgoing_loss(self, adjacency, nodes):
        """
        Moderate constraint: END nodes should have minimal outgoing edges
        
        Uses soft penalty that allows learning without being too restrictive
        
        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss encouraging END nodes to have few/no outgoing edges
        """
        # Identify END nodes
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # Sum over edge types to get total adjacency
        total_adj = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes, max_nodes)
        
        # Compute out-degree for each node
        out_degrees = tf.reduce_sum(total_adj, axis=2)  # (batch, max_nodes)
        
        # For END nodes: soft penalty on out-degree
        # Use ReLU to only penalize when out_degree > threshold
        threshold = 0  # No flexibility
        end_out_penalty = end_probs * tf.nn.relu(out_degrees - threshold)
        
        loss = tf.reduce_mean(end_out_penalty)
        return loss

    def edge_continuity_loss(self, adjacency, nodes):
        """
        Constraint: ensure edges continue from previously reached nodes.

        Penalize soft cases where a node has outgoing edges but no incoming
        edges (i.e., the edge would "start in the void") unless the node
        is START. This encourages edges to chain: new arcs should originate
        from nodes that were previously reached (or from START).

        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            Loss penalizing outgoing-from-unreached-node cases
        """
        # Sum over edge types
        total_adj = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes, max_nodes)

        # Degrees
        in_degrees = tf.reduce_sum(total_adj, axis=1)  # (batch, max_nodes)
        out_degrees = tf.reduce_sum(total_adj, axis=2)  # (batch, max_nodes)

        # Soft indicators for having any in/out degree (gradient-friendly)
        has_in = tf.nn.sigmoid((in_degrees - 0.1) * 10.0)
        has_out = tf.nn.sigmoid((out_degrees - 0.1) * 10.0)

        # START probabilities
        start_probs = nodes[:, :, self.start_idx]

        # Active nodes mask (not PAD)
        active_mask = 1.0 - nodes[:, :, self.pad_idx]

        # Penalize nodes that have outgoing edges but NO incoming edges and are not START
        continuity_penalty = has_out * (1.0 - has_in) * (1.0 - start_probs) * active_mask

        loss = tf.reduce_mean(continuity_penalty)

        return loss
    
    def end_uniqueness_loss(self, nodes):
        """
        Moderate constraint: Encourage one dominant END node per graph
        
        Uses soft concentration penalty without being too restrictive
        
        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss encouraging concentration of END probability
        """
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # 1. Soft count constraint: gently push sum towards 1
        end_count = tf.reduce_sum(end_probs, axis=1)  # (batch,)
        count_loss = tf.reduce_mean(tf.abs(end_count - 1.0))  # Use L1 instead of L2 for softer penalty
        
        # 2. Concentration: encourage one node to dominate
        max_end_prob = tf.reduce_max(end_probs, axis=1)  # (batch,)
        concentration_loss = tf.reduce_mean(tf.nn.relu(0.5 - max_end_prob))  # Only penalize if max < 0.5
        
        total_loss = count_loss + concentration_loss
        
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
        Soft constraint: END should have incoming edges from valid nodes
        
        Uses gradient-friendly formulation
        
        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss gently encouraging END to have incoming connections
        """
        # Identify END node probabilities
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # Sum over edge types
        total_adj = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes, max_nodes)
        
        # Incoming edges per node (sum over source dimension)
        incoming = tf.reduce_sum(total_adj, axis=1)  # (batch, max_nodes)
        
        # Mask for valid source nodes (non-PAD, non-END)
        valid_source_mask = 1.0 - nodes[:, :, self.pad_idx] - end_probs
        valid_source_mask = tf.maximum(valid_source_mask, 0.0)
        
        # Weight incoming edges by source validity
        # This approximates "incoming from valid sources"
        weighted_incoming = incoming * tf.reduce_mean(valid_source_mask, axis=1, keepdims=True)
        
        # For END nodes, gently encourage having some incoming connections
        # Use soft threshold: only penalize if incoming < 0.5
        end_needs_incoming = end_probs * tf.nn.relu(0.5 - weighted_incoming)
        
        loss = tf.reduce_mean(end_needs_incoming)
        
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
        edge_continuity = self.edge_continuity_loss(adjacency, nodes)

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
            self.lambda_end_unique * end_unique_loss +
            self.lambda_edge_continuity * edge_continuity
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
            'end_unique_loss': self.end_uniqueness_loss(nodes),
            'edge_continuity_loss': self.edge_continuity_loss(adjacency, nodes),
        }
