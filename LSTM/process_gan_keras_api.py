"""
Process Mining GAN - TensorFlow/Keras API Implementation

This version provides better integration with Keras features:
- Automatic callbacks support (ModelCheckpoint, EarlyStopping, TensorBoard, etc.)
- Built-in metrics tracking
- Progress bars
- Validation support
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
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

        # Build embedding layer to ensure weights are initialized
        self.embedding.build((None,))

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
        processed = self.noise_processor(noise)
        states = []
        for _ in range(self.lstm_layers):
            h = processed
            c = processed
            states.append((h, c))
        return states

    def call(self, batch_size, temperature=1.0, noise=None, hard=True, training=True):
        if noise is None:
            noise = tf.random.normal([batch_size, self.noise_dim])

        states = self.initialize_state(batch_size, noise)
        current_input = tf.zeros([batch_size, self.embedding_dim])
        outputs = []

        for t in range(self.max_trace_length):
            lstm_output, states = self.stacked_lstm(current_input, states)
            logits = self.output_projection(lstm_output)
            activity_probs = gumbel_softmax(logits, temperature, hard=hard)
            outputs.append(activity_probs)

            if hard:
                next_activity_idx = tf.argmax(activity_probs, axis=-1)
                current_input = self.embedding(next_activity_idx)
            else:
                # Soft embedding: weighted sum of embeddings using activity_probs
                # Get embedding matrix (now safe since we built the layer)
                embedding_matrix = self.embedding.embeddings
                # activity_probs shape: [batch_size, num_activities]
                # embedding_matrix shape: [num_activities, embedding_dim]
                # Result shape: [batch_size, embedding_dim]
                current_input = tf.matmul(activity_probs, embedding_matrix)

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
        super(ProcessTraceDiscriminator, self).__init__(name=name)

        self.num_activities = num_activities
        self.embedding_dim = embedding_dim

        self.embedding = layers.Embedding(
            num_activities,
            embedding_dim,
            name='activity_embedding'
        )

        self.continuous_projection = layers.Dense(
            embedding_dim,
            name='continuous_projection'
        )

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

        self.classifier = keras.Sequential([
            layers.Dense(lstm_units, name='fc1',
                         kernel_constraint=tf.keras.constraints.MaxNorm(1.0)),
            layers.LeakyReLU(0.2),
            layers.Dropout(dropout_rate),
            layers.Dense(lstm_units // 2, name='fc2',
                         kernel_constraint=tf.keras.constraints.MaxNorm(1.0)),
            layers.LeakyReLU(0.2),
            layers.Dropout(dropout_rate),
            layers.Dense(1, name='output',
                         kernel_constraint=tf.keras.constraints.MaxNorm(1.0))
        ], name='classifier')

    def call(self, traces, is_discrete=False, training=True):
        if is_discrete:
            embedded = self.embedding(traces)
        else:
            embedded = self.continuous_projection(traces)

        lstm_output = self.bilstm(embedded, training=training)
        scores = self.classifier(lstm_output, training=training)

        return scores


# ============================================================================
# PROCESS GAN FULL MODEL
# ============================================================================

class ProcessGAN(keras.Model):
    """
    Main GAN class
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
                 n_critic=5,
                 lambda_gp=10.0,
                 lambda_constraint=0.1,
                 temp_start=5.0,
                 temp_min=0.5,
                 temp_decay=0.99995):
        super(ProcessGAN, self).__init__()

        self.num_activities = num_activities
        self.max_trace_length = max_trace_length
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lambda_constraint = lambda_constraint
        self.temperature = tf.Variable(
            temp_start, trainable=False, dtype=tf.float32)
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

        # Constraints
        self.constraints = ProcessConstraints(
            start_idx, end_idx, activity_frequencies)

        # Metrics (will be tracked automatically)
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.d_real_score_tracker = keras.metrics.Mean(name='d_real_score')
        self.d_fake_score_tracker = keras.metrics.Mean(name='d_fake_score')
        self.constraint_loss_tracker = keras.metrics.Mean(
            name='constraint_loss')
        self.temperature_tracker = keras.metrics.Mean(name='temperature')

    def compile(self, d_optimizer, g_optimizer):
        """
        Custom compile method for GAN

        Parameters:
        -----------
        d_optimizer: keras.optimizers.Optimizer
            Optimizer for discriminator
        g_optimizer: keras.optimizers.Optimizer
            Optimizer for generator
        """
        super().compile()

        # Add gradient clipping to prevent exploding gradients
        # Set clipnorm directly on the optimizer
        d_optimizer.clipnorm = 1.0
        g_optimizer.clipnorm = 1.0

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @property
    def metrics(self):
        """
        List metrics that should be tracked and reset automatically
        """
        return [
            self.d_loss_tracker,
            self.g_loss_tracker,
            self.d_real_score_tracker,
            self.d_fake_score_tracker,
            self.constraint_loss_tracker,
            self.temperature_tracker
        ]

    def train_step(self, real_traces):
        """
        Custom training step - called by fit() for each batch

        """
        # Handle tuple input (in case labels are provided)
        if isinstance(real_traces, tuple):
            real_traces = real_traces[0]

        batch_size = tf.shape(real_traces)[0]

        # ====================================================================
        # TRAIN DISCRIMINATOR (n_critic times)
        # ====================================================================
        for _ in range(self.n_critic):
            with tf.GradientTape() as tape:
                # Real data scores
                real_scores = self.discriminator(
                    real_traces, is_discrete=True, training=True)

                # Generate fake data
                fake_traces = self.generator(batch_size, temperature=self.temperature,
                                             hard=False, training=True)
                fake_scores = self.discriminator(
                    fake_traces, is_discrete=False, training=True)

                # Wasserstein loss
                d_loss = wasserstein_loss(real_scores, fake_scores)

                # Gradient penalty
                real_one_hot = tf.one_hot(real_traces, self.num_activities)
                gp = gradient_penalty(self.discriminator,
                                      real_one_hot, fake_traces)

                # Total discriminator loss
                total_d_loss = d_loss + self.lambda_gp * gp

            # Update discriminator
            d_gradients = tape.gradient(
                total_d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables))

        # ====================================================================
        # TRAIN GENERATOR (once)
        # ====================================================================
        with tf.GradientTape() as tape:
            # Generate fake data
            fake_traces = self.generator(batch_size, temperature=self.temperature,
                                         hard=False, training=True)

            # Discriminator scores for fake data
            fake_scores = self.discriminator(
                fake_traces, is_discrete=False, training=True)

            # Generator loss (fool discriminator)
            g_loss = -tf.reduce_mean(fake_scores)

            # Process constraints
            constraint_loss = self.constraints.total_constraint_loss(
                fake_traces)

            # Total generator loss
            total_g_loss = g_loss + self.lambda_constraint * constraint_loss

        # Update generator
        g_gradients = tape.gradient(
            total_g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables))

        # ====================================================================
        # UPDATE TEMPERATURE
        # ====================================================================
        new_temp = tf.maximum(
            self.temperature * self.temp_decay, self.temp_min)
        self.temperature.assign(new_temp)

        # ====================================================================
        # UPDATE METRICS
        # ====================================================================
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.d_real_score_tracker.update_state(tf.reduce_mean(real_scores))
        self.d_fake_score_tracker.update_state(tf.reduce_mean(fake_scores))
        self.constraint_loss_tracker.update_state(constraint_loss)
        self.temperature_tracker.update_state(self.temperature)

        # Return metrics (will be displayed in progress bar)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, real_traces):
        """
        Custom test/validation step - called by evaluate()

        This allows you to run validation during training!
        """
        if isinstance(real_traces, tuple):
            real_traces = real_traces[0]

        batch_size = tf.shape(real_traces)[0]

        # Generate fake traces (no gradient tracking)
        fake_traces = self.generator(batch_size, temperature=self.temperature,
                                     hard=False, training=False)

        # Compute discriminator scores
        real_scores = self.discriminator(
            real_traces, is_discrete=True, training=False)
        fake_scores = self.discriminator(
            fake_traces, is_discrete=False, training=False)

        # Compute losses (for validation metrics)
        d_loss = wasserstein_loss(real_scores, fake_scores)
        g_loss = -tf.reduce_mean(fake_scores)
        constraint_loss = self.constraints.total_constraint_loss(fake_traces)

        # Update metrics
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.d_real_score_tracker.update_state(tf.reduce_mean(real_scores))
        self.d_fake_score_tracker.update_state(tf.reduce_mean(fake_scores))
        self.constraint_loss_tracker.update_state(constraint_loss)

        return {m.name: m.result() for m in self.metrics}

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
        batch_size = 128
        all_traces = []

        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)

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
