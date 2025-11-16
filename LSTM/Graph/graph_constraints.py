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
                 lambda_sparsity=0.0):
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

        # Loss weights (simplified after merging)
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.lambda_frequency = lambda_frequency
        self.lambda_connectivity = lambda_connectivity
        self.lambda_structure = lambda_structure
        self.lambda_degree = lambda_degree
        self.lambda_path = lambda_path
        self.lambda_node_on_path = lambda_node_on_path
        self.lambda_sparsity = lambda_sparsity
        self.lambda_no_start_end_direct = lambda_no_start_end_direct

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
        UNIFIED END Constraint: Strong enforcement like START node
        
        Strong enforcement with:
        1. Terminal positioning via attention (soft guidance to terminals)
        2. Penalize END appearing at NON-terminal positions
        3. Count constraint (sum should be ~1)
        4. Concentration penalty (max END probability should be high)
        5. Zero out-degree enforcement (END must have no outgoing edges)

        Args:
            nodes: Node matrix (batch, max_nodes, num_activities)
            adjacency: Binary adjacency matrix (batch, max_nodes, max_nodes)

        Returns:
            Combined loss for all END-related constraints
        """
        # Compute out-degrees (adjacency already binary)
        out_degrees = tf.reduce_sum(adjacency, axis=2)  # (batch, max_nodes)
        
        # Get END node probabilities for all positions
        end_probs = nodes[:, :, self.end_idx]  # (batch, max_nodes)
        
        # Mask for active nodes (not PAD)
        active_mask = 1.0 - nodes[:, :, self.pad_idx]  # (batch, max_nodes)

        # === COMPONENT 1: Terminal Positioning (identify best END position) ===
        # Find the terminal node (lowest out-degree among active nodes)
        terminal_score = active_mask * tf.nn.sigmoid(-(out_degrees - 0.1) * 20.0)
        attention_weights = tf.nn.softmax(terminal_score + 1e-10, axis=1)
        attention_weights_exp = tf.expand_dims(attention_weights, axis=-1)
        
        # Weighted sum - encourage END at most terminal position
        weighted_nodes = tf.reduce_sum(nodes * attention_weights_exp, axis=1)
        end_at_best_terminal = weighted_nodes[:, self.end_idx]
        loss_positioning = -tf.reduce_mean(tf.math.log(end_at_best_terminal + 1e-10))
        
        # === COMPONENT 2: Penalize END at NON-terminal positions ===
        # Instead of penalizing all END probs, only penalize END where out_degree > 0
        # This allows END to appear at terminal nodes but not elsewhere
        non_terminal_mask = tf.nn.sigmoid((out_degrees - 0.1) * 10.0)  # 1 if out_degree > 0
        end_at_non_terminal = end_probs * non_terminal_mask
        loss_non_terminal = tf.reduce_mean(tf.square(end_at_non_terminal))
        
        # === COMPONENT 3: Count constraint (sum should be ~1) ===
        end_count = tf.reduce_sum(end_probs, axis=1)  # (batch,)
        loss_count = tf.reduce_mean(tf.square(end_count - 1.0))
        
        # === COMPONENT 4: Concentration (one node should dominate) ===
        # Find max END probability and penalize if it's not high enough
        max_end_prob = tf.reduce_max(end_probs, axis=1)  # (batch,)
        # Penalize if max END prob is less than 0.7 (strong concentration but not too aggressive)
        loss_concentration = tf.reduce_mean(tf.nn.relu(0.7 - max_end_prob))
        
        # Also penalize the second-highest END probability to ensure uniqueness
        sorted_end_probs = tf.sort(end_probs, axis=1, direction='DESCENDING')
        second_highest = sorted_end_probs[:, 1]  # Get second highest value
        loss_second = tf.reduce_mean(tf.square(second_highest))  # Should be close to 0
        
        # === COMPONENT 5: Zero Out-Degree Enforcement ===
        # Heavily penalize ANY out-degree for END nodes
        end_out_penalty = end_probs * out_degrees
        loss_zero_out = tf.reduce_mean(end_out_penalty)
        
        # === COMBINED LOSS ===
        # Reduced weights to avoid over-constraining
        total_loss = (
            1.0 * loss_positioning +      # Soft guidance to terminals
            2.0 * loss_non_terminal +     # Penalize END at non-terminals
            1.5 * loss_count +            # Count constraint (reduced from 2.0)
            1.5 * loss_concentration +    # High concentration (reduced from 2.0)
            2.0 * loss_second +           # Penalize second-highest END prob (reduced from 3.0)
            2.0 * loss_zero_out           # No out-edges (reduced from 3.0)
        )
        
        return total_loss

    def activity_frequency_loss(self, nodes): #serve a mantenere la stessa distribuzione delle attività --> verificare che non crea problemi con i PAD
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
        - END: at least 1 in-degree, exactly 0 out-degree
        - Other nodes: at least 1 in-degree and 1 out-degree
        
        Uses SOFT penalties that allow gradual learning
        
        Args:
            adjacency: Binary adjacency matrix (batch, max_nodes, max_nodes)
            nodes: Node matrix (batch, max_nodes, num_activities)
            
        Returns:
            Loss penalizing degree violations
        """
        # Compute degrees (adjacency already binary)
        in_degrees = tf.reduce_sum(adjacency, axis=1)  # (batch, max_nodes) - sum over sources
        out_degrees = tf.reduce_sum(adjacency, axis=2)  # (batch, max_nodes) - sum over targets
        
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
        
        # END constraints: >=1 in-degree, 0 out-degree
        # More lenient: allow some flexibility during training
        end_in_penalty = end_probs * tf.nn.relu(1.0 - in_degrees)  # Only penalize if < 1
        end_out_penalty = end_probs * out_degrees  # Penalize any out-degree but softly
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
            adjacency: Binary adjacency matrix (batch, max_nodes, max_nodes)
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            Loss penalizing isolated nodes
        """
        # Adjacency is already binary (batch, max_nodes, max_nodes)

        # Outgoing edges per node
        outgoing = tf.reduce_sum(adjacency, axis=-1)  # (batch, max_nodes)

        # Incoming edges per node
        incoming = tf.reduce_sum(adjacency, axis=-2)  # (batch, max_nodes)

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
        # Adjacency is already binary (batch, max_nodes, max_nodes)
        
        # === COMPONENT 1: Self-loops (diagonal elements) ===
        # Highest priority - these are never valid in process graphs
        diagonal = tf.linalg.diag_part(adjacency)  # (batch, max_nodes)
        loss_self_loops = tf.reduce_mean(tf.square(diagonal))  # Squared for stronger penalty
        
        # === COMPONENT 2: 2-cycles (A→B and B→A) ===
        # Check if edge exists in both directions
        adjacency_T = tf.transpose(adjacency, perm=[0, 2, 1])  # Transpose
        bidirectional = adjacency * adjacency_T  # Element-wise product
        # Only count upper triangle to avoid double counting
        indices = tf.range(tf.shape(adjacency)[1])
        i_indices = tf.expand_dims(indices, axis=1)  # (max_nodes, 1)
        j_indices = tf.expand_dims(indices, axis=0)  # (1, max_nodes)
        upper_mask = tf.cast(i_indices < j_indices, tf.float32)  # (max_nodes, max_nodes)
        upper_mask = tf.expand_dims(upper_mask, axis=0)  # (1, max_nodes, max_nodes)
        
        bidirectional_upper = bidirectional * upper_mask
        loss_2cycles = tf.reduce_mean(tf.square(bidirectional_upper))  # Squared for stronger penalty
        
        # === COMPONENT 3: Longer cycles via transitive closure ===
        # Compute up to 6 hops to catch longer cycles
        reachable = adjacency
        for _ in range(6):  # Increased from 4 to 6 for better cycle detection
            reachable = tf.minimum(tf.matmul(reachable, adjacency) + reachable, 1.0)
        
        # Check diagonal of reachability matrix
        # If reachable[i,i] > 0, there's a path from i to i (cycle)
        reachable_diag = tf.linalg.diag_part(reachable)  # (batch, max_nodes)
        loss_cycles = tf.reduce_mean(tf.square(reachable_diag))  # Squared for stronger penalty
        
        # === COMBINED LOSS ===
        # VERY HIGH weights for aggressive loop prevention
        total_loss = (
            20.0 * loss_self_loops +    # Self-loops - MASSIVELY increased
            25.0 * loss_2cycles +       # Bidirectional edges - MASSIVELY increased
            15.0 * loss_cycles          # General cycles - MASSIVELY increased
        )
        
        return total_loss

    # START part kept in start_node_loss (already has count constraint)

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
        # Count active (non-PAD) nodes per graph
        active_mask = 1.0 - nodes[:, :, self.pad_idx]  # (batch, max_nodes)
        num_active_nodes = tf.reduce_sum(active_mask, axis=1)  # (batch,)
        
        # Penalize using too many nodes (push towards smaller graphs)
        # Use a soft quadratic penalty that increases with node count
        max_nodes = tf.cast(tf.shape(nodes)[1], tf.float32)
        usage_ratio = num_active_nodes / max_nodes  # (batch,) - ratio of nodes used
        
        # Penalize high usage (> 60% of max_nodes)
        # This encourages the generator to use fewer nodes when possible
        target_ratio = 0.6  # Target using around 60% of available nodes
        loss_overuse = tf.reduce_mean(tf.nn.relu(usage_ratio - target_ratio) ** 2)
        
        # Encourage variance in graph sizes across the batch
        # Low variance means all graphs have similar size (bad)
        mean_usage = tf.reduce_mean(usage_ratio)
        variance_usage = tf.reduce_mean(tf.square(usage_ratio - mean_usage))
        # Penalize low variance (want diversity)
        min_variance = 0.02  # Minimum desired variance in usage ratios
        loss_variance = tf.nn.relu(min_variance - variance_usage)
        
        # Also penalize nodes that are "barely active" (probabilistic PAD)
        # Encourage nodes to be either clearly active or clearly PAD
        pad_probs = nodes[:, :, self.pad_idx]  # (batch, max_nodes)
        # Penalize values in middle range [0.2, 0.8] - want clear decisions
        ambiguous = tf.nn.relu(0.8 - pad_probs) * tf.nn.relu(pad_probs - 0.2)
        loss_ambiguity = tf.reduce_mean(ambiguous)
        
        # Combine components
        total_loss = (
            2.0 * loss_overuse +      # Penalize using too many nodes
            1.0 * loss_variance +      # Encourage size diversity
            0.5 * loss_ambiguity       # Encourage clear PAD decisions
        )
        
        return total_loss

    def connectivity_loss(self, adjacency, nodes):
        """
        UNIFIED Connectivity Constraint: General + START/END specific + edge continuity
        
        Combines:
        - connectivity_loss (no isolated nodes)
        - start_connectivity_loss (START has out-edges)
        - end_connectivity_loss (END has in-edges)
        - edge_continuity_loss (edges chain properly)
        
        Multi-level enforcement:
        1. General: all active nodes must be connected
        2. START-specific: must have valid outgoing edges
        3. END-specific: must have valid incoming edges
        4. Edge chaining: no edges "from nowhere"

        Args:
            adjacency: Adjacency matrix (batch, max_nodes, max_nodes, edge_types)
            nodes: Node matrix (batch, max_nodes, num_activities)

        Returns:
            Combined connectivity loss
        """
        # Compute degrees (adjacency already binary)
        in_degrees = tf.reduce_sum(adjacency, axis=1)   # (batch, max_nodes)
        out_degrees = tf.reduce_sum(adjacency, axis=2)  # (batch, max_nodes)
        
        # Node identifiers
        start_probs = nodes[:, :, self.start_idx]
        end_probs = nodes[:, :, self.end_idx]
        active_mask = 1.0 - nodes[:, :, self.pad_idx]
        
        # === COMPONENT 1: General Connectivity (no isolated nodes) ===
        is_connected = tf.minimum(in_degrees + out_degrees, 1.0)
        isolated_nodes = active_mask * (1.0 - is_connected)
        loss_general = tf.reduce_mean(isolated_nodes)
        
        # === COMPONENT 2: START Connectivity (must have out-edges) ===
        # Mask for valid targets (non-PAD)
        non_pad_mask = tf.expand_dims(active_mask, axis=1)  # (batch, 1, max_nodes)
        valid_outgoing = adjacency * non_pad_mask
        total_valid_out = tf.reduce_sum(valid_outgoing, axis=2)  # (batch, max_nodes)
        
        # Penalize START with no outgoing edges
        start_no_out = start_probs * tf.nn.sigmoid(-(total_valid_out - 0.5) * 10.0)
        loss_start = tf.reduce_mean(start_no_out)
        
        # === COMPONENT 3: END Connectivity (must have in-edges) ===
        # Weight incoming by valid sources
        valid_source_mask = 1.0 - nodes[:, :, self.pad_idx] - end_probs
        valid_source_mask = tf.maximum(valid_source_mask, 0.0)
        weighted_incoming = in_degrees * tf.reduce_mean(valid_source_mask, axis=1, keepdims=True)
        
        # Penalize END with insufficient incoming
        end_no_in = end_probs * tf.nn.relu(0.5 - weighted_incoming)
        loss_end = tf.reduce_mean(end_no_in)
        
        # === COMPONENT 4: Edge Continuity (no edges from nowhere) ===
        # Soft indicators
        has_in = tf.nn.sigmoid((in_degrees - 0.1) * 10.0)
        has_out = tf.nn.sigmoid((out_degrees - 0.1) * 10.0)
        
        # Penalize: out-edges but no in-edges (except START)
        continuity_penalty = has_out * (1.0 - has_in) * (1.0 - start_probs) * active_mask
        loss_continuity = tf.reduce_mean(continuity_penalty)
        
        # === COMBINED LOSS ===
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
        # Adjacency is already binary (batch, max_nodes, max_nodes)
        
        # Compute reachability matrix using transitive closure approximation
        # A + A^2 + A^3 + ... captures multi-hop connections
        reachable = adjacency
        current = adjacency
        
        # Compute up to 8 hops (sufficient for most process graphs)
        # Using fixed number to avoid symbolic tensor issues
        for _ in range(8):
            current = tf.matmul(current, adjacency)
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
        # Adjacency is already binary (batch, max_nodes, max_nodes)
        
        # 1. Compute forward reachability from START (nodes reachable from START)
        forward_reach = adjacency
        current_forward = adjacency
        for _ in range(10):
            current_forward = tf.matmul(current_forward, adjacency)
            current_forward = tf.minimum(current_forward, 1.0)
            forward_reach = tf.minimum(forward_reach + current_forward, 1.0)
        
        # 2. Compute backward reachability to END (nodes that can reach END)
        # Transpose adjacency for backward propagation
        adjacency_T = tf.transpose(adjacency, perm=[0, 2, 1])
        backward_reach = adjacency_T
        current_backward = adjacency_T
        for _ in range(8):
            current_backward = tf.matmul(current_backward, adjacency_T)
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
        # Individual losses (simplified after merging)
        start_loss = self.start_node_loss(nodes)
        end_loss = self.end_node_loss(nodes, adjacency)
        freq_loss = self.activity_frequency_loss(nodes)
        connect_loss = self.connectivity_loss(adjacency, nodes)
        struct_loss = self.structural_validity_loss(adjacency)
        degree_loss = self.degree_constraint_loss(adjacency, nodes)
        path_loss = self.path_existence_loss(adjacency, nodes)
        node_on_path_loss = self.nodes_on_path_loss(adjacency, nodes)
        sparsity_loss = self.sparsity_loss(nodes, adjacency)

        # Weighted sum
        total_loss = (
            self.lambda_start * start_loss +
            self.lambda_end * end_loss +
            self.lambda_frequency * freq_loss +
            self.lambda_connectivity * connect_loss +
            self.lambda_structure * struct_loss +
            self.lambda_degree * degree_loss +
            self.lambda_path * path_loss +
            self.lambda_node_on_path * node_on_path_loss +
            self.lambda_sparsity * sparsity_loss
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
            'degree_loss': self.degree_constraint_loss(adjacency, nodes),
            'path_loss': self.path_existence_loss(adjacency, nodes),
            'node_on_path_loss': self.nodes_on_path_loss(adjacency, nodes),
            'sparsity_loss': self.sparsity_loss(nodes, adjacency),
        }
