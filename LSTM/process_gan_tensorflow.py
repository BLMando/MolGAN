"""
Process Mining GAN - TensorFlow 2.x Implementation
Trace-Based GAN with Process Constraints

Compatible with TensorFlow 2.x and Keras API
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import os
from utils import gumbel_softmax, wasserstein_loss, gradient_penalty
from process_constraints import ProcessConstraints


# ============================================================================
# GENERATOR MODEL
# ============================================================================

class ProcessTraceGenerator(keras.Model):
    """
    LSTM-based generator for process traces
    """
    def __init__(self,
                 num_activities,
                 noise_dim=128,
                 embedding_dim=64,
                 lstm_units=256,
                 lstm_layers=2,
                 max_trace_length=50,
                 name='generator'):
        """
        Parameters:
        -----------
        num_activities: int
            Total number of unique activities (including special tokens)
        noise_dim: int
            Dimension of input noise vector
        embedding_dim: int
            Dimension of activity embeddings
        lstm_units: int
            Number of units in LSTM layers
        lstm_layers: int
            Number of LSTM layers
        max_trace_length: int
            Maximum length of generated traces
        """
        super(ProcessTraceGenerator, self).__init__(name=name)

        self.num_activities = num_activities
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.max_trace_length = max_trace_length
        
        # Noise processing layers
        self.noise_processor = keras.Sequential([
            layers.Dense(lstm_units, activation='relu', name='noise_fc1'),
            layers.Dense(lstm_units, activation='relu', name='noise_fc2')
        ], name='noise_processor')
        
        # Activity embedding
        self.embedding = layers.Embedding(
            num_activities, 
            embedding_dim,
            name='activity_embedding'
        )
        
        # LSTM layers
        self.lstm_cells = []
        for i in range(lstm_layers):
            self.lstm_cells.append(
                layers.LSTMCell(lstm_units, name=f'lstm_cell_{i}')
            )
        self.stacked_lstm = layers.StackedRNNCells(self.lstm_cells)
        
        # Output projection
        self.output_projection = layers.Dense(
            num_activities,
            name='output_projection'
        )
    
    def initialize_state(self, batch_size, noise):
        """
        Initialize LSTM states from noise
        
        Returns:
        --------
        states: list of tuples
            [(h_0, c_0), (h_1, c_1), ...] for each LSTM layer
        """
        # Process noise
        processed = self.noise_processor(noise)  # [batch_size, lstm_units]
        
        # Create initial states for each LSTM layer
        states = []
        for _ in range(self.lstm_layers):
            h = processed
            c = processed
            states.append((h, c))
        
        return states
    
    def call(self, batch_size, temperature=1.0, noise=None, hard=True, training=True):
        """
        Generate a batch of traces
        
        Parameters:
        -----------
        batch_size: int
            Number of traces to generate
        temperature: float
            Temperature for Gumbel-Softmax
        noise: tf.Tensor, optional
            Input noise [batch_size, noise_dim]
            If None, random noise will be generated
        hard: bool
            If True, use hard (one-hot) Gumbel-Softmax
            If False, use soft probabilities
        training: bool
            Training mode flag
        
        Returns:
        --------
        traces: tf.Tensor [batch_size, max_trace_length, num_activities]
            Generated traces (one-hot or soft probabilities)
        """
        # Generate noise if not provided
        if noise is None:
            noise = tf.random.normal([batch_size, self.noise_dim])
        
        # Initialize LSTM states from noise
        states = self.initialize_state(batch_size, noise)

        # Start with zero input for the first timestep
        # The first output (position 0) will be generated from the LSTM state,
        # which is initialized from noise. The model will learn to output START
        # at position 0 via constraint losses, not via hardcoding.
        current_input = tf.zeros([batch_size, self.embedding_dim])

        outputs = []
        
        for t in range(self.max_trace_length):
            # LSTM step
            lstm_output, states = self.stacked_lstm(current_input, states)
            
            # Project to activity space
            logits = self.output_projection(lstm_output)
            
            # Sample next activity using Gumbel-Softmax
            activity_probs = gumbel_softmax(logits, temperature, hard=hard)
            
            outputs.append(activity_probs)
            
            # Prepare next input
            if hard:
                # Get index of selected activity
                next_activity_idx = tf.argmax(activity_probs, axis=-1)
                current_input = self.embedding(next_activity_idx)
            else:
                # For soft sampling, use weighted embedding
                # [batch_size, num_activities, 1] * [1, num_activities, embedding_dim]
                embeddings = self.embedding.embeddings  # [num_activities, embedding_dim]
                current_input = tf.matmul(
                    tf.expand_dims(activity_probs, 1),
                    tf.expand_dims(embeddings, 0)
                )
                current_input = tf.squeeze(current_input, axis=1)
        
        # Stack outputs: [batch_size, max_trace_length, num_activities]
        traces = tf.stack(outputs, axis=1)
        
        return traces

# ============================================================================
# DISCRIMINATOR MODEL
# ============================================================================

class ProcessTraceDiscriminator(keras.Model):
    """
    Bidirectional LSTM-based discriminator for process traces
    """
    def __init__(self,
                 num_activities,
                 embedding_dim=64,
                 lstm_units=256,
                 lstm_layers=2,
                 dropout_rate=0.3,
                 name='discriminator'):
        """
        Parameters:
        -----------
        num_activities: int
            Total number of unique activities
        embedding_dim: int
            Dimension of activity embeddings
        lstm_units: int
            Number of units in LSTM layers
        lstm_layers: int
            Number of LSTM layers
        dropout_rate: float
            Dropout probability
        """
        super(ProcessTraceDiscriminator, self).__init__(name=name)
        
        self.num_activities = num_activities
        self.embedding_dim = embedding_dim
        
        # Embedding for discrete inputs
        self.embedding = layers.Embedding(
            num_activities,
            embedding_dim,
            name='activity_embedding'
        )
        
        # Projection for continuous inputs (from generator)
        self.continuous_projection = layers.Dense(
            embedding_dim,
            name='continuous_projection'
        )
        
        # Bidirectional LSTM layers
        self.bilstm = keras.Sequential(name='bilstm_stack')
        for i in range(lstm_layers):
            self.bilstm.add(
                layers.Bidirectional(
                    layers.LSTM(
                        lstm_units,
                        return_sequences=(i < lstm_layers - 1),
                        dropout=dropout_rate if lstm_layers > 1 else 0,
                        name=f'lstm_{i}'
                    ),
                    name=f'bidirectional_{i}'
                )
            )
        
        # Classification head
        self.classifier = keras.Sequential([
            layers.Dense(lstm_units, name='fc1'),
            layers.LeakyReLU(0.2),
            layers.Dropout(dropout_rate),
            layers.Dense(lstm_units // 2, name='fc2'),
            layers.LeakyReLU(0.2),
            layers.Dropout(dropout_rate),
            layers.Dense(1, name='output')
        ], name='classifier')
    
    def call(self, traces, is_discrete=False, training=True):
        """
        Discriminate real from fake traces
        
        Parameters:
        -----------
        traces: tf.Tensor
            If is_discrete=True: [batch_size, max_trace_length] (indices)
            If is_discrete=False: [batch_size, max_trace_length, num_activities] (one-hot/soft)
        is_discrete: bool
            Whether input is discrete indices or continuous probabilities
        training: bool
            Training mode flag
        
        Returns:
        --------
        scores: tf.Tensor [batch_size, 1]
            Realness scores (Wasserstein distance)
        """
        if is_discrete:
            # Embed discrete traces
            embedded = self.embedding(traces)
        else:
            # Project continuous traces
            embedded = self.continuous_projection(traces)
        
        # Process with bidirectional LSTM
        lstm_output = self.bilstm(embedded, training=training)
        
        # For sequences, take last output (already done by return_sequences=False in last layer)
        # For stacked LSTMs with return_sequences, we'd need to take [:, -1, :]
        
        # Classify
        scores = self.classifier(lstm_output, training=training)
        
        return scores

# ============================================================================
# PROCESS GAN CLASS
# ============================================================================

class ProcessGAN:
    """
    Main GAN class for process trace generation
    """
    def __init__(self,
                 num_activities,
                 max_trace_length,
                 start_idx,
                 end_idx,
                 activity_frequencies,
                 noise_dim=128,
                 embedding_dim=64,
                 generator_lstm_units=256,
                 discriminator_lstm_units=256,
                 lstm_layers=2,
                 learning_rate=0.0001,
                 n_critic=5,
                 lambda_gp=10.0,
                 lambda_constraint=0.1,
                 temp_start=5.0,
                 temp_min=0.5,
                 temp_decay=0.99995):
        """
        Initialize Process GAN

        Note: start_idx and end_idx are kept for constraint losses only.
        The generator no longer hardcodes the START token - it learns it.
        """
        self.num_activities = num_activities
        self.max_trace_length = max_trace_length
        
        # Training hyperparameters
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lambda_constraint = lambda_constraint
        self.temperature = temp_start
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        
        # Build models
        self.generator = ProcessTraceGenerator(
            num_activities=num_activities,
            noise_dim=noise_dim,
            embedding_dim=embedding_dim,
            lstm_units=generator_lstm_units,
            lstm_layers=lstm_layers,
            max_trace_length=max_trace_length
        )
        
        self.discriminator = ProcessTraceDiscriminator(
            num_activities=num_activities,
            embedding_dim=embedding_dim,
            lstm_units=discriminator_lstm_units,
            lstm_layers=lstm_layers
        )
        
        # Optimizers
        self.g_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)
        
        # Constraints
        self.constraints = ProcessConstraints(start_idx, end_idx, activity_frequencies)
        
        # Metrics
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')
        self.d_real_score_metric = keras.metrics.Mean(name='d_real_score')
        self.d_fake_score_metric = keras.metrics.Mean(name='d_fake_score')
        self.constraint_loss_metric = keras.metrics.Mean(name='constraint_loss')
    
    @tf.function
    def train_discriminator_step(self, real_traces, batch_size):
        """
        Single discriminator training step
        """
        with tf.GradientTape() as tape:
            # Real data scores
            real_scores = self.discriminator(real_traces, is_discrete=True, training=True)

            # Generate fake data
            fake_traces = self.generator(batch_size, temperature=self.temperature,
                                        hard=False, training=True)
            fake_scores = self.discriminator(fake_traces, is_discrete=False, training=True)

            # Wasserstein loss
            d_loss = wasserstein_loss(real_scores, fake_scores)

            # Gradient penalty
            real_one_hot = tf.one_hot(real_traces, self.num_activities)
            gp = gradient_penalty(self.discriminator, real_one_hot, fake_traces)

            # Total discriminator loss
            total_d_loss = d_loss + self.lambda_gp * gp

        # Update discriminator
        d_gradients = tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        return d_loss, real_scores, fake_scores
    
    @tf.function
    def train_generator_step(self, batch_size):
        """
        Single generator training step
        """
        with tf.GradientTape() as tape:
            # Generate fake data
            fake_traces = self.generator(batch_size, temperature=self.temperature,
                                        hard=False, training=True)
            
            # Discriminator scores for fake data
            fake_scores = self.discriminator(fake_traces, is_discrete=False, training=True)
            
            # Generator loss (fool discriminator)
            g_loss = -tf.reduce_mean(fake_scores)
            
            # Process constraints
            constraint_loss = self.constraints.total_constraint_loss(fake_traces)
            
            # Total generator loss
            total_g_loss = g_loss + self.lambda_constraint * constraint_loss
        
        # Update generator
        g_gradients = tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return g_loss, constraint_loss
    
    def train(self, dataset, epochs=100, steps_per_epoch=None,
              eval_every=500, save_dir='./checkpoints', verbose=True,
              idx_to_activity=None, num_preview_traces=5, resume=True):
        """
        Train the GAN

        Parameters:
        -----------
        dataset: tf.data.Dataset
            Dataset of real traces
        epochs: int
            Number of training epochs (total, not additional)
        steps_per_epoch: int, optional
            Steps per epoch (if None, determined from dataset)
        eval_every: int
            Evaluate and print metrics every N steps
        save_dir: str
            Directory to save checkpoints
        verbose: bool
            Print training progress
        idx_to_activity: dict, optional
            Mapping from activity index to name (for preview)
        num_preview_traces: int
            Number of example traces to show after each epoch
        resume: bool
            If True, automatically resume from latest checkpoint
        """
        # Create checkpoint directory
        os.makedirs(save_dir, exist_ok=True)

        # Build models by doing a forward pass (required before loading weights)
        # This creates the model variables
        dummy_batch_size = 1
        _ = self.generator(dummy_batch_size, temperature=1.0, training=False)
        dummy_trace = tf.zeros((dummy_batch_size, self.max_trace_length), dtype=tf.int32)
        _ = self.discriminator(dummy_trace, is_discrete=True, training=False)

        # Try to load latest checkpoint
        start_epoch = 0
        if resume:
            checkpoint_path, checkpoint_epoch = self.find_latest_checkpoint(save_dir)
            if checkpoint_path is not None:
                try:
                    if verbose:
                        print(f"\n{'='*70}")
                        print(f"Found checkpoint at epoch {checkpoint_epoch}")
                        print(f"Loading weights from: {checkpoint_path}")
                    self.load_weights(checkpoint_path)
                    start_epoch = checkpoint_epoch
                    if verbose:
                        print(f"✓ Resumed from epoch {checkpoint_epoch}")
                        print(f"Will train from epoch {start_epoch + 1} to {epochs}")
                        print(f"{'='*70}\n")
                except Exception as e:
                    if verbose:
                        print(f"⚠ Warning: Failed to load checkpoint: {e}")
                        error_msg = str(e)
                        if "Shape mismatch" in error_msg or "Layer count mismatch" in error_msg:
                            print(f"   ⚠ Checkpoint is incompatible with current model architecture.")
                            print(f"   Possible reasons:")
                            print(f"     - Different vocabulary size (number of activities changed)")
                            print(f"     - Different model parameters (lstm_units, layers, etc.)")
                            print(f"   ")
                            print(f"   To start fresh, delete old checkpoints:")
                            print(f"   PowerShell: Remove-Item -Recurse -Force {save_dir}")
                            print(f"   Bash/Linux: rm -rf {save_dir}")
                        print(f"\n   Starting training from scratch...")
                    start_epoch = 0
            elif verbose:
                print(f"\nNo checkpoint found. Starting training from scratch.\n")

        global_step = 0
        start_time = time.time()

        for epoch in range(start_epoch, epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
            
            for step, real_traces in enumerate(dataset):
                batch_size = tf.shape(real_traces)[0]
                
                # Train discriminator multiple times
                for _ in range(self.n_critic):
                    d_loss, real_scores, fake_scores = self.train_discriminator_step(real_traces, batch_size)
                    self.d_loss_metric.update_state(d_loss)
                    self.d_real_score_metric.update_state(tf.reduce_mean(real_scores))
                    self.d_fake_score_metric.update_state(tf.reduce_mean(fake_scores))

                # Train generator once
                g_loss, constraint_loss = self.train_generator_step(batch_size)
                self.g_loss_metric.update_state(g_loss)
                self.constraint_loss_metric.update_state(constraint_loss)
                
                # Update temperature
                self.temperature = max(self.temperature * self.temp_decay, self.temp_min)
                
                global_step += 1
                
                # Logging
                if verbose and global_step % eval_every == 0:
                    elapsed = time.time() - start_time
                    print(f"Step {global_step}: "
                          f"D_loss={self.d_loss_metric.result():.4f}, "
                          f"D_real={self.d_real_score_metric.result():.4f}, "
                          f"D_fake={self.d_fake_score_metric.result():.4f}, "
                          f"G_loss={self.g_loss_metric.result():.4f}, "
                          f"Constraint={self.constraint_loss_metric.result():.4f}, "
                          f"Temp={self.temperature:.3f}, "
                          f"Time={elapsed:.1f}s")

                    # Reset metrics
                    self.d_loss_metric.reset_states()
                    self.d_real_score_metric.reset_states()
                    self.d_fake_score_metric.reset_states()
                    self.g_loss_metric.reset_states()
                    self.constraint_loss_metric.reset_states()
                    start_time = time.time()
                
                if steps_per_epoch and step >= steps_per_epoch:
                    break

            # Generate preview traces at end of epoch
            if verbose and idx_to_activity is not None and num_preview_traces > 0:
                print(f"\n--- Epoch {epoch + 1} Generated Samples (Temp={self.temperature:.3f}) ---")
                preview_traces = self.generate_traces(
                    num_samples=num_preview_traces,
                    temperature=self.temperature,
                    return_indices=True
                )

                for i, trace_indices in enumerate(preview_traces, 1):
                    # Convert to activity names and filter out special tokens
                    activities = []
                    for idx in trace_indices:
                        if idx in idx_to_activity:
                            activity = idx_to_activity[idx]
                            # Skip special tokens
                            if activity not in ['<PAD>','<UNK>']:
                                activities.append(activity)

                    if activities:
                        trace_str = ' → '.join(activities)
                    else:
                        trace_str = "(empty trace)"

                    print(f"  {i}. {trace_str}")
                print()

            # Save checkpoint after each epoch (if save_dir provided)
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_weights(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}'))
                if verbose:
                    print(f"Saved checkpoint at epoch {epoch + 1}")

        # Save final checkpoint if not already saved
        if save_dir and epochs % 10 != 0:
            final_checkpoint = os.path.join(save_dir, f'checkpoint_epoch_{epochs}')
            self.save_weights(final_checkpoint)
            if verbose:
                print(f"\n✓ Saved final checkpoint at epoch {epochs}")
    
    def generate_traces(self, num_samples, temperature=0.5, return_indices=True):
        """
        Generate synthetic traces
        
        Parameters:
        -----------
        num_samples: int
            Number of traces to generate
        temperature: float
            Temperature for generation (lower = more confident)
        return_indices: bool
            If True, return activity indices
            If False, return one-hot encoded traces
        
        Returns:
        --------
        traces: np.array
            Generated traces
        """
        # Generate in batches to avoid memory issues
        batch_size = 128
        all_traces = []
        
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Generate traces
            traces = self.generator(
                current_batch_size,
                temperature=temperature,
                hard=True,
                training=False
            )
            
            if return_indices:
                traces = tf.argmax(traces, axis=-1)
            
            all_traces.append(traces.numpy())
        
        return np.concatenate(all_traces, axis=0)
    
    def save_weights(self, filepath):
        """Save model weights and configuration"""
        import json
        
        # Save weights
        gen_path = filepath + '_generator.h5'
        disc_path = filepath + '_discriminator.h5'
        config_path = filepath + '_config.json'
        
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)
        
        # Save model configuration for compatibility checking
        config = {
            'num_activities': self.num_activities,
            'max_trace_length': self.max_trace_length,
            'generator_lstm_units': self.generator.lstm_units,
            'discriminator_lstm_units': self.discriminator.num_activities,  # This should be lstm_units but not accessible
            'temperature': float(self.temperature),
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_weights(self, filepath):
        """Load model weights with compatibility check"""
        import json
        
        gen_path = filepath + '_generator.h5'
        disc_path = filepath + '_discriminator.h5'
        config_path = filepath + '_config.json'
        
        # Check if files exist
        if not os.path.exists(gen_path):
            raise FileNotFoundError(f"Generator weights not found: {gen_path}")
        if not os.path.exists(disc_path):
            raise FileNotFoundError(f"Discriminator weights not found: {disc_path}")
        
        # Check compatibility if config exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            
            # Verify critical parameters match
            if saved_config.get('num_activities') != self.num_activities:
                raise ValueError(
                    f"Vocabulary size mismatch: "
                    f"checkpoint has {saved_config.get('num_activities')} activities, "
                    f"current model has {self.num_activities} activities"
                )
            
            if saved_config.get('max_trace_length') != self.max_trace_length:
                raise ValueError(
                    f"Max trace length mismatch: "
                    f"checkpoint has {saved_config.get('max_trace_length')}, "
                    f"current model has {self.max_trace_length}"
                )
        
        # Load weights
        self.generator.load_weights(gen_path)
        self.discriminator.load_weights(disc_path)

    def find_latest_checkpoint(self, save_dir):
        """
        Find the latest checkpoint in the save directory

        Returns:
        --------
        checkpoint_path: str or None
            Path to latest checkpoint (without suffix), or None if not found
        epoch: int
            Epoch number of the checkpoint
        """
        if not os.path.exists(save_dir):
            return None, 0

        # Find all generator checkpoint files
        import glob
        import re
        
        checkpoints = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*_generator.h5'))

        if not checkpoints:
            return None, 0

        # Extract epoch numbers using regex (more robust)
        epochs = []
        pattern = r'checkpoint_epoch_(\d+)_generator\.h5'
        
        for ckpt in checkpoints:
            try:
                basename = os.path.basename(ckpt)
                match = re.search(pattern, basename)
                if match:
                    epoch_num = int(match.group(1))
                    epochs.append(epoch_num)
            except Exception as e:
                # Skip this checkpoint if we can't parse it
                continue

        if not epochs:
            return None, 0

        # Get latest epoch
        latest_epoch = max(epochs)
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{latest_epoch}')

        return checkpoint_path, latest_epoch

