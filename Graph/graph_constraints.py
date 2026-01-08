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
                 lambda_start=3.0,
                 lambda_end=10.0,
                 lambda_frequency=1.0,
                 lambda_connectivity=5.0,
                 lambda_structure=5.0,
                 lambda_degree=5.0,
                 lambda_path=3.0,
                 lambda_node_on_path=5.0,
                 lambda_sparsity=0.0,
                 lambda_time_monotonic=10.0):
        """
        Args:
            start_idx: Index of START activity
            end_idx: Index of END activity
            activity_frequencies: Target activity frequency distribution
            pad_idx: Index of PAD token
            lambda_*: Weights for different constraint losses
            lambda_sparsity: Weight for sparsity loss (encourages varying node counts)
            lambda_no_start_end_direct: Weight for preventing direct START→END connections
        """
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.activity_frequencies = tf.constant(
            activity_frequencies, dtype=tf.float32)
        self.pad_idx = pad_idx

        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.lambda_frequency = lambda_frequency
        self.lambda_connectivity = lambda_connectivity
        self.lambda_structure = lambda_structure
        self.lambda_degree = lambda_degree
        self.lambda_path = lambda_path
        self.lambda_node_on_path = lambda_node_on_path
        self.lambda_node_on_path = lambda_node_on_path
        self.lambda_sparsity = lambda_sparsity
        self.lambda_time_monotonic = lambda_time_monotonic
        self.lambda_time_start_end = 2.0
        self.lambda_time_parallel = 2.0

    def start_node_loss(self, nodes, return_components=False):
        """
        Constraint: First node should be START and ONLY first node
        
        Strong enforcement with:
        1. Cross-entropy for first node being START
        2. Heavy penalty for START appearing elsewhere
        3. Concentration penalty to ensure single START
        
        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            return_components: If True, return (total_loss, components_dict)

        Returns:
            Loss encouraging first node to be START (and only first node)
        """
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        
        start_probs_first = start_probs[:, 0]  # (batch,)
        loss_first = -tf.reduce_mean(tf.math.log(start_probs_first + 1e-10))
        
        start_probs_others = start_probs[:, 1:]  # (batch, max_nodes-1)
        loss_others = tf.reduce_mean(tf.square(start_probs_others))
        
        start_count = tf.reduce_sum(start_probs, axis=1)  # (batch,)
        loss_count = tf.reduce_mean(tf.square(start_count - 1.0))
        
        max_start_elsewhere = tf.reduce_max(start_probs_others, axis=1)  # (batch,)
        loss_concentration = tf.reduce_mean(max_start_elsewhere)
        
        total_loss = loss_first + 3.0 * loss_others + 2.0 * loss_count + 2.0 * loss_concentration
        
        if return_components:
            return total_loss, {
                'start_loss_first': loss_first,
                'start_loss_others': loss_others,
                'start_loss_count': loss_count,
                'start_loss_concentration': loss_concentration
            }
        return total_loss

    def end_node_loss(self, nodes, adjacency, return_components=False):
        """
        END Constraint: Strong enforcement like START node

        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            adjacency: Binary adjacency matrix (batch, max_nodes, max_nodes)
            return_components: If True, return (total_loss, components_dict)

        Returns:
            Combined loss for all END-related constraints
        """
        out_degrees = tf.reduce_sum(adjacency, axis=2)  # (batch, max_nodes)
        
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        active_mask = 1.0 - nodes[:, :, self.pad_idx]  # (batch, max_nodes)

        terminal_logits = -(out_degrees - 0.1) * 20.0
        
        terminal_logits = terminal_logits - (1.0 - active_mask) * 1e9
        terminal_logits = terminal_logits - (1.0 - active_mask) * 1e9
        
        attention_weights = tf.nn.softmax(terminal_logits, axis=1)
        attention_weights_exp = tf.expand_dims(attention_weights, axis=-1)
        
        weighted_nodes = tf.reduce_sum(nodes * attention_weights_exp, axis=1)
        end_at_best_terminal = weighted_nodes[:, self.end_idx]
        loss_positioning = -tf.reduce_mean(tf.math.log(end_at_best_terminal + 1e-10))
        
        non_terminal_mask = tf.nn.sigmoid((out_degrees - 0.1) * 10.0)
        end_at_non_terminal = end_probs * non_terminal_mask
        loss_non_terminal = tf.reduce_mean(tf.square(end_at_non_terminal))
        
        end_count = tf.reduce_sum(end_probs, axis=1)
        loss_count = tf.reduce_mean(tf.square(end_count - 1.0))
        
        max_end_prob = tf.reduce_max(end_probs, axis=1)
        loss_concentration = tf.reduce_mean(tf.nn.relu(0.7 - max_end_prob))
        
        sorted_end_probs = tf.sort(end_probs, axis=1, direction='DESCENDING')
        second_highest = sorted_end_probs[:, 1]  
        loss_second = tf.reduce_mean(tf.square(second_highest))
        
        end_out_penalty = end_probs * out_degrees
        loss_zero_out = tf.reduce_mean(end_out_penalty)
        
        total_loss = (
            1.0 * loss_positioning +      
            2.0 * loss_non_terminal +     
            1.5 * loss_count +            
            1.5 * loss_concentration +    
            2.0 * loss_second +           
            2.0 * loss_zero_out           
        )
        
        if return_components:
            return total_loss, {
                'end_loss_positioning': loss_positioning,
                'end_loss_non_terminal': loss_non_terminal,
                'end_loss_count': loss_count,
                'end_loss_concentration': loss_concentration,
                'end_loss_second': loss_second,
                'end_loss_zero_out': loss_zero_out
            }
        return total_loss

    def activity_frequency_loss(self, nodes): 
        """
        Constraint: Match target activity frequency distribution

        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            KL divergence between generated and target frequencies
        """
        generated_freq = tf.reduce_mean(
            nodes, axis=[0, 1])  # (num_activities,)

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
        - END: at least 1 in-degree, exactly 0 out-degree
        - Other nodes: at least 1 in-degree and 1 out-degree
        
        Uses SOFT penalties that allow gradual learning
        
        Args:
            adjacency: Binary adjacency matrix (batch, max_nodes, max_nodes)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss penalizing degree violations
        """
        in_degrees = tf.reduce_sum(adjacency, axis=1)  # (batch, max_nodes) - sum over sources
        out_degrees = tf.reduce_sum(adjacency, axis=2)  # (batch, max_nodes) - sum over targets
        
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        loss = 0.0
        
        start_in_penalty = start_probs * in_degrees
        start_out_penalty = start_probs * tf.nn.relu(1.0 - out_degrees)
        loss += tf.reduce_mean(start_in_penalty + start_out_penalty)
        
        end_in_penalty = end_probs * tf.nn.relu(1.0 - in_degrees)
        end_out_penalty = end_probs * out_degrees
        loss += tf.reduce_mean(end_in_penalty + end_out_penalty)
        
        other_mask = 1.0 - start_probs - end_probs - nodes[:, :, self.pad_idx]
        other_mask = tf.maximum(other_mask, 0.0)
        other_in_penalty = other_mask * tf.nn.relu(1.0 - in_degrees)
        other_out_penalty = other_mask * tf.nn.relu(1.0 - out_degrees)
        loss += tf.reduce_mean(other_in_penalty + other_out_penalty)
        
        pad_probs = nodes[:, :, self.pad_idx]
        pad_degree_penalty = pad_probs * (in_degrees + out_degrees)
        loss += tf.reduce_mean(pad_degree_penalty)
        
        return loss

    def connectivity_loss(self, adjacency, nodes):
        """
        Constraint: Ensure graph connectivity (approximate)

        Encourages that most active nodes have at least one incoming or outgoing edge

        Args:
            adjacency: Binary adjacency matrix (batch, max_nodes, max_nodes)
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            Loss penalizing isolated nodes
        """
        outgoing = tf.reduce_sum(adjacency, axis=-1)

        incoming = tf.reduce_sum(adjacency, axis=-2)
        incoming = tf.reduce_sum(adjacency, axis=-2)  # (batch, max_nodes)

        is_connected = tf.minimum(
            outgoing + incoming, 1.0)  # (batch, max_nodes)

        active_mask = 1.0 - nodes[:, :, self.pad_idx]  # (batch, max_nodes)

        isolated_nodes = active_mask * (1.0 - is_connected)
        loss = tf.reduce_mean(isolated_nodes)

        return loss

    def structural_validity_loss(self, adjacency):
        """
        Constraint: Structural validity of process graphs - NO LOOPS ALLOWED
        
        Process graphs should be DAGs (Directed Acyclic Graphs).
        This function penalizes loops with STRONG enforcement:
        1. Self-loops (direct cycles) - VERY high priority
        2. Bidirectional edges (A→B and B→A) - high priority
        3. General cycles via transitive closure - medium priority
        
        Args:
            adjacency: Binary adjacency matrix (batch, max_nodes, max_nodes)

        Returns:
            Loss penalizing loops/cycles in the graph
        """
        diagonal = tf.linalg.diag_part(adjacency)
        loss_self_loops = tf.reduce_mean(tf.square(diagonal))
        
        adjacency_T = tf.transpose(adjacency, perm=[0, 2, 1])
        bidirectional = adjacency * adjacency_T
        indices = tf.range(tf.shape(adjacency)[1])
        i_indices = tf.expand_dims(indices, axis=1)  # (max_nodes, 1)
        j_indices = tf.expand_dims(indices, axis=0)  # (1, max_nodes)
        upper_mask = tf.cast(i_indices < j_indices, tf.float32)  # (max_nodes, max_nodes)
        upper_mask = tf.expand_dims(upper_mask, axis=0)  # (1, max_nodes, max_nodes)
        
        bidirectional_upper = bidirectional * upper_mask
        loss_2cycles = tf.reduce_mean(tf.square(bidirectional_upper))       
        
        reachable = adjacency
        for _ in range(6):
            reachable = tf.minimum(tf.matmul(reachable, adjacency) + reachable, 1.0)
        
        reachable_diag = tf.linalg.diag_part(reachable)
        reachable_diag = tf.linalg.diag_part(reachable)  # (batch, max_nodes)
        loss_cycles = tf.reduce_mean(tf.square(reachable_diag))
        
        total_loss = (
            1.0 * loss_self_loops +    
            1.0 * loss_2cycles +       
            1.0 * loss_cycles          
        )
        return total_loss


    def sparsity_loss(self, nodes, adjacency):
        """
        Encourage varying graph sizes by penalizing using all available nodes
        
        This creates pressure to generate graphs with different node counts,
        making PAD tokens more prominent when graphs should be smaller.
        
        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            adjacency: Binary adjacency matrix (batch, max_nodes, max_nodes)
            
        Returns:
            Loss encouraging sparse graphs with varying sizes
        """
        active_mask = 1.0 - nodes[:, :, self.pad_idx]  # (batch, max_nodes)
        num_active_nodes = tf.reduce_sum(active_mask, axis=1)  # (batch,)
        
        max_nodes = tf.cast(tf.shape(nodes)[1], tf.float32)
        usage_ratio = num_active_nodes / max_nodes  # (batch,) - ratio of nodes used
        
        target_ratio = 0.6  
        loss_overuse = tf.reduce_mean(tf.nn.relu(usage_ratio - target_ratio) ** 2)
        
        mean_usage = tf.reduce_mean(usage_ratio)
        variance_usage = tf.reduce_mean(tf.square(usage_ratio - mean_usage))
        min_variance = 0.02  
        loss_variance = tf.nn.relu(min_variance - variance_usage)
        
        pad_probs = nodes[:, :, self.pad_idx]  
        ambiguous = tf.nn.relu(0.8 - pad_probs) * tf.nn.relu(pad_probs - 0.2)
        loss_ambiguity = tf.reduce_mean(ambiguous)
        
        total_loss = (
            2.0 * loss_overuse +      
            1.0 * loss_variance +      
            0.5 * loss_ambiguity       
        )
        
        return total_loss

    def start_end_time_loss(self, nodes, features):
        """
        Constraint: START and END nodes should have 0 duration/time
        
        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            features: Feature matrix (batch, max_nodes, num_features)
                      0: norm_time, 1: trace_time, 2: prev_event_time
            
        Returns:
            Loss penalizing non-zero times for START/END
        """
        if features is None:
            return 0.0
            

        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        end_probs = nodes[:, :, self.end_idx]      # (batch, max_nodes)
        
        start_time_penalty = start_probs * tf.reduce_sum(tf.square(features), axis=2)
        
        end_time_penalty = end_probs * tf.reduce_sum(tf.square(features[:, :, 1:]), axis=2)
        
        loss = tf.reduce_mean(start_time_penalty + end_time_penalty)
        return loss

    def parallel_time_loss(self, adjacency, features):
        """
        Constraint: Parallel nodes (same source) should have similar start times
        
        Args:
            adjacency: Binary adjacency matrix (batch, max_nodes, max_nodes)
            features: Feature matrix (batch, max_nodes, num_features)
            
        Returns:
            Loss penalizing time differences between parallel nodes
        """
        if features is None:
            return 0.0
            
        features_exp = tf.expand_dims(features, axis=1)
        
        diff_norm = tf.abs(features_exp[:, :, :, 0] - tf.transpose(features_exp[:, :, :, 0], perm=[0, 2, 1]))
        
        diff_trace = tf.abs(features_exp[:, :, :, 1] - tf.transpose(features_exp[:, :, :, 1], perm=[0, 2, 1]))
        
        time_diffs = diff_norm + diff_trace
        
        siblings = tf.matmul(tf.transpose(adjacency, perm=[0, 2, 1]), adjacency)
        
        mask = 1.0 - tf.eye(tf.shape(adjacency)[1])
        siblings = siblings * tf.expand_dims(mask, axis=0)
        
        loss = tf.reduce_mean(siblings * time_diffs)
        return loss


    def monotonic_time_loss(self, adjacency, features):
        """
        Constraint: Time must be monotonic along edges
        
        For every edge u -> v:
        1. trace_time(v) > trace_time(u)  (strictly increasing)
        2. prev_event_time(v) ≈ trace_time(v) - trace_time(u) (consistency)

        
        Args:
            adjacency: Binary adjacency matrix (batch, max_nodes, max_nodes)
            features: Feature matrix (batch, max_nodes, num_features)
                      0: norm_time, 1: trace_time, 2: prev_event_time
            
        Returns:
            Loss penalizing temporal violations
        """
        if features is None:
            return 0.0
            
        features_source = tf.expand_dims(features, axis=2)  # (batch, max_nodes, 1, feats)
        features_target = tf.expand_dims(features, axis=1)  # (batch, 1, max_nodes, feats)
        
        trace_time_s = features_source[:, :, :, 1]
        trace_time_t = features_target[:, :, :, 1]
        
        prev_time_t = features_target[:, :, :, 2]
        
        edge_mask = adjacency
        
        margin = 0.01
        time_violation = tf.nn.relu(trace_time_s - trace_time_t + margin)
        loss_monotonic = tf.reduce_mean(edge_mask * time_violation)
        
        delta_trace = trace_time_t - trace_time_s
        consistency_error = tf.abs(prev_time_t - delta_trace)
        loss_consistency = tf.reduce_mean(edge_mask * consistency_error)
        
        return 2.0 * loss_monotonic + 1.5 * loss_consistency


    def connectivity_loss(self, adjacency, nodes):
        """
        Connectivity Constraint: General + START/END specific + edge continuity
 
        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            Combined connectivity loss
        """
        in_degrees = tf.reduce_sum(adjacency, axis=1)   # (batch, max_nodes)
        out_degrees = tf.reduce_sum(adjacency, axis=2)  # (batch, max_nodes)
        
        start_probs = nodes[:, :, self.start_idx]
        end_probs = nodes[:, :, self.end_idx]
        active_mask = 1.0 - nodes[:, :, self.pad_idx]
        
        is_connected = tf.minimum(in_degrees + out_degrees, 1.0)
        isolated_nodes = active_mask * (1.0 - is_connected)
        loss_general = tf.reduce_mean(isolated_nodes)
        
        non_pad_mask = tf.expand_dims(active_mask, axis=1)  # (batch, 1, max_nodes)
        valid_outgoing = adjacency * non_pad_mask
        total_valid_out = tf.reduce_sum(valid_outgoing, axis=2)  # (batch, max_nodes)
        
        start_no_out = start_probs * tf.nn.sigmoid(-(total_valid_out - 0.5) * 10.0)
        loss_start = tf.reduce_mean(start_no_out)
        
        valid_source_mask = 1.0 - nodes[:, :, self.pad_idx] - end_probs
        valid_source_mask = tf.maximum(valid_source_mask, 0.0)
        weighted_incoming = in_degrees * tf.reduce_mean(valid_source_mask, axis=1, keepdims=True)
        
        end_no_in = end_probs * tf.nn.relu(0.5 - weighted_incoming)
        loss_end = tf.reduce_mean(end_no_in)
        
        has_in = tf.nn.sigmoid((in_degrees - 0.1) * 10.0)
        has_out = tf.nn.sigmoid((out_degrees - 0.1) * 10.0)
        
        continuity_penalty = has_out * (1.0 - has_in) * (1.0 - start_probs) * active_mask
        loss_continuity = tf.reduce_mean(continuity_penalty)
        
        total_loss = (
            1.0 * loss_general +      # General isolation
            2.0 * loss_start +        # START must connect
            2.0 * loss_end +          # END must be reachable
            1.5 * loss_continuity     # Edges must chain
        )
        
        return total_loss

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
        reachable = adjacency
        current = adjacency
        
        for _ in range(8):
            current = tf.matmul(current, adjacency)
            current = tf.minimum(current, 1.0)  # Keep binary
            reachable = tf.minimum(reachable + current, 1.0)
        
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        start_probs_exp = tf.expand_dims(start_probs, axis=1)  # (batch, 1, max_nodes)
        end_probs_exp = tf.expand_dims(end_probs, axis=-1)  # (batch, max_nodes, 1)
        
        path_strength = tf.matmul(start_probs_exp, reachable)  # (batch, 1, max_nodes)
        path_strength = tf.matmul(path_strength, end_probs_exp)  # (batch, 1, 1)
        path_strength = tf.squeeze(path_strength, axis=[1, 2])  # (batch,)
        
        path_strength = tf.clip_by_value(path_strength, 0.0, 1.0)
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
        
        forward_reach = adjacency
        current_forward = adjacency
        for _ in range(10):
            current_forward = tf.matmul(current_forward, adjacency)
            current_forward = tf.minimum(current_forward, 1.0)
            forward_reach = tf.minimum(forward_reach + current_forward, 1.0)
        
        adjacency_T = tf.transpose(adjacency, perm=[0, 2, 1])
        backward_reach = adjacency_T
        current_backward = adjacency_T
        for _ in range(8):
            current_backward = tf.matmul(current_backward, adjacency_T)
            current_backward = tf.minimum(current_backward, 1.0)
            backward_reach = tf.minimum(backward_reach + current_backward, 1.0)

        backward_reach = tf.transpose(backward_reach, perm=[0, 2, 1])
        
        start_probs = nodes[:, :, self.start_idx]  # (batch, max_nodes)
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        start_probs_exp = tf.expand_dims(start_probs, axis=1)  # (batch, 1, max_nodes)
        end_probs_exp = tf.expand_dims(end_probs, axis=-1)  # (batch, max_nodes, 1)
        
        reachable_from_start = tf.matmul(start_probs_exp, forward_reach)
        reachable_from_start = tf.squeeze(reachable_from_start, axis=1)  # (batch, max_nodes)
        
        can_reach_end = tf.matmul(backward_reach, end_probs_exp)
        can_reach_end = tf.squeeze(can_reach_end, axis=-1)  # (batch, max_nodes)
        
        on_path = reachable_from_start * can_reach_end  # (batch, max_nodes)
        on_path = tf.clip_by_value(on_path, 0.0, 1.0)
        
        active_mask = 1.0 - nodes[:, :, self.pad_idx]
        intermediate_mask = active_mask - start_probs - end_probs
        intermediate_mask = tf.maximum(intermediate_mask, 0.0)  

        nodes_not_on_path = intermediate_mask * (1.0 - on_path)
        loss = tf.reduce_mean(nodes_not_on_path)
        
        return loss

    def total_constraint_loss(self, adjacency, nodes, features=None):
        """
        Compute total constraint loss

        Args:
            adjacency: Adjacency matrices
            nodes: Node matrices

        Returns:
            Weighted sum of all constraint losses
        """
        start_loss = self.start_node_loss(nodes)
        end_loss = self.end_node_loss(nodes, adjacency)
        freq_loss = self.activity_frequency_loss(nodes)
        connect_loss = self.connectivity_loss(adjacency, nodes)
        struct_loss = self.structural_validity_loss(adjacency)
        degree_loss = self.degree_constraint_loss(adjacency, nodes)
        path_loss = self.path_existence_loss(adjacency, nodes)
        node_on_path_loss = self.nodes_on_path_loss(adjacency, nodes)
        sparsity_loss = self.sparsity_loss(nodes, adjacency)
        
        time_start_end_loss = self.start_end_time_loss(nodes, features)
        time_parallel_loss = self.parallel_time_loss(adjacency, features)
        time_monotonic_loss = self.monotonic_time_loss(adjacency, features)

        total_loss = (
            self.lambda_start * start_loss +
            self.lambda_end * end_loss +
            self.lambda_frequency * freq_loss +
            self.lambda_connectivity * connect_loss +
            self.lambda_structure * struct_loss +
            self.lambda_degree * degree_loss +
            self.lambda_path * path_loss +
            self.lambda_node_on_path * node_on_path_loss +
            self.lambda_sparsity * sparsity_loss +
            self.lambda_time_start_end * time_start_end_loss +
            self.lambda_time_parallel * time_parallel_loss +
            self.lambda_time_monotonic * time_monotonic_loss
        )

        return total_loss

    def get_individual_losses(self, adjacency, nodes, features=None):
        """
        Get dictionary of individual constraint losses for logging

        Args:
            adjacency: Adjacency matrices
            nodes: Node matrices

        Returns:
            Dictionary with individual losses
        """
        start_loss, start_components = self.start_node_loss(nodes, return_components=True)
        end_loss, end_components = self.end_node_loss(nodes, adjacency, return_components=True)
        
        losses = {
            'start_loss': start_loss,
            'end_loss': end_loss,
            'frequency_loss': self.activity_frequency_loss(nodes),
            'connectivity_loss': self.connectivity_loss(adjacency, nodes),
            'structural_loss': self.structural_validity_loss(adjacency),
            'degree_loss': self.degree_constraint_loss(adjacency, nodes),
            'path_loss': self.path_existence_loss(adjacency, nodes),
            'node_on_path_loss': self.nodes_on_path_loss(adjacency, nodes),
            'sparsity_loss': self.sparsity_loss(nodes, adjacency),
            'time_start_end_loss': self.start_end_time_loss(nodes, features),
            'time_parallel_loss': self.parallel_time_loss(adjacency, features),
            'time_monotonic_loss': self.monotonic_time_loss(adjacency, features),
        }
     
        losses.update(start_components)
        losses.update(end_components)
        
        return losses

    