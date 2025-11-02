import tensorflow as tf

# ============================================================================
# PROCESS CONSTRAINTS
# ============================================================================

class ProcessConstraints:
    """
    Enforce process mining-specific constraints
    """
    def __init__(self, start_idx, end_idx, activity_frequencies):
        """
        Parameters:
        -----------
        start_idx: int
            Index of START activity
        end_idx: int
            Index of END activity
        activity_frequencies: np.array [num_activities]
            Normalized frequency of each activity in real data
        """
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.activity_frequencies = tf.constant(activity_frequencies, dtype=tf.float32)
    
    def start_constraint_loss(self, generated_traces):
        """
        Penalize traces that don't start with START token
        
        Parameters:
        -----------
        generated_traces: tf.Tensor [batch_size, max_length, num_activities]
        
        Returns:
        --------
        loss: tf.Tensor scalar
        """
        # Get first activity distribution
        first_activity = generated_traces[:, 0, :]  # [batch_size, num_activities]
        
        # Loss: negative log probability of START token
        start_prob = first_activity[:, self.start_idx]
        loss = tf.reduce_mean(-tf.math.log(start_prob + 1e-10))
        
        return loss
    
    def end_constraint_loss(self, generated_traces):
        """
        Encourage traces to have END token
        
        Parameters:
        -----------
        generated_traces: tf.Tensor [batch_size, max_length, num_activities]
        
        Returns:
        --------
        loss: tf.Tensor scalar
        """
        # Get END token probabilities across all positions
        end_probs = generated_traces[:, :, self.end_idx]  # [batch_size, max_length]
        
        # Get maximum END probability for each trace
        max_end_prob = tf.reduce_max(end_probs, axis=1)  # [batch_size]
        
        # Loss: encourage END token to appear
        loss = tf.reduce_mean(1.0 - max_end_prob)
        
        return loss
    
    def frequency_matching_loss(self, generated_traces):
        """
        Match activity frequency distribution
        
        Parameters:
        -----------
        generated_traces: tf.Tensor [batch_size, max_length, num_activities]
        
        Returns:
        --------
        loss: tf.Tensor scalar
        """
        # Compute generated frequency distribution
        gen_freq = tf.reduce_sum(generated_traces, axis=[0, 1])  # [num_activities]
        gen_freq = gen_freq / tf.reduce_sum(gen_freq)
        
        # L1 distance between distributions
        loss = tf.reduce_mean(tf.abs(gen_freq - self.activity_frequencies))
        
        return loss
    
    def total_constraint_loss(self, generated_traces):
        """
        Compute total constraint loss
        """
        start_loss = self.start_constraint_loss(generated_traces)
        end_loss = self.end_constraint_loss(generated_traces)
        freq_loss = self.frequency_matching_loss(generated_traces)
        
        return start_loss + end_loss + 0.5 * freq_loss