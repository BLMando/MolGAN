"""
ProcessGAN LSTM-based Architecture

Modern LSTM-based implementation for process trace generation.
Replaces matrix-based MolGAN with sequential LSTM autoregressive generation.

Key improvements:
- LSTM Generator: Autoregressive trace generation with hard START/END constraints
- Bi-LSTM Discriminator: Captures temporal dependencies in both directions
- Natural sequence modeling instead of graph representation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ProcessGeneratorLSTM(keras.Model):
    """
    LSTM-based Generator: z → sequential trace generation

    Generates traces autoregressively:
    - Forced START token at position 0
    - LSTM generates activities step-by-step
    - Stops at END token or max_length
    - Natural handling of variable-length traces
    """

    def __init__(self, max_activities, activity_types, embedding_dim,
                 lstm_hidden_dim=256, lstm_num_layers=2, dropout_rate=0.0, **kwargs):
        """
        Initialize LSTM Generator

        Args:
            max_activities: Maximum trace length
            activity_types: Number of activity types (vocabulary size)
            embedding_dim: Dimension of latent space z
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
        """
        super(ProcessGeneratorLSTM, self).__init__(**kwargs)

        self.max_activities = max_activities
        self.activity_types = activity_types
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.dropout_rate = dropout_rate

        # Latent vector projection
        self.z_projection = layers.Dense(lstm_hidden_dim, activation='tanh', name='z_projection')

        # Activity embedding for teacher forcing / conditioning
        self.activity_embedding = layers.Embedding(
            activity_types,
            lstm_hidden_dim,
            mask_zero=True,
            name='activity_embedding'
        )

        # LSTM layers
        self.lstm_layers = []
        for i in range(lstm_num_layers):
            self.lstm_layers.append(
                layers.LSTM(
                    lstm_hidden_dim,
                    return_sequences=True,
                    return_state=True,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    name=f'lstm_{i}'
                )
            )

        # Output head: activity prediction
        self.activity_output = layers.Dense(
            activity_types,
            activation=None,
            name='activity_logits'
        )

        # Dropout
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, embeddings, training=False, temperature=1.0, start_token_id=0, end_token_id=None):
        """
        Forward pass: Generate traces autoregressively

        Args:
            embeddings: Latent vectors z (batch, embedding_dim)
            training: Training mode flag
            temperature: Gumbel-Softmax temperature
            start_token_id: ID of START activity (default: 0)
            end_token_id: ID of END activity (optional, for early stopping)

        Returns:
            nodes: Generated activity sequences (batch, max_activities, activity_types)
        """
        batch_size = tf.shape(embeddings)[0]

        # Project latent vector to LSTM initial state
        initial_hidden = self.z_projection(embeddings)  # (batch, lstm_hidden_dim)

        # Initialize LSTM states for all layers
        states = []
        for _ in range(self.lstm_num_layers):
            h = initial_hidden
            c = tf.zeros_like(h)
            states.append([h, c])

        # Force START token as first input (one-hot)
        current_token = tf.one_hot(
            tf.fill([batch_size], start_token_id),
            depth=self.activity_types,
            dtype=tf.float32
        )  # (batch, activity_types)

        # Store generated tokens
        generated_tokens = []
        generated_tokens.append(current_token)

        # Autoregressive generation
        for t in range(1, self.max_activities):
            # Embed current token (soft embedding via matrix multiplication)
            # Convert one-hot/soft to embedding: (batch, activity_types) @ (activity_types, hidden_dim)
            embedded = tf.matmul(
                current_token,
                self.activity_embedding.embeddings
            )  # (batch, lstm_hidden_dim)

            # Expand dims for sequence: (batch, 1, hidden_dim)
            lstm_input = tf.expand_dims(embedded, axis=1)

            # Apply dropout
            lstm_input = self.dropout(lstm_input, training=training)

            # Pass through LSTM layers
            lstm_output = lstm_input
            new_states = []
            for i, lstm_layer in enumerate(self.lstm_layers):
                lstm_output, h, c = lstm_layer(
                    lstm_output,
                    initial_state=states[i],
                    training=training
                )
                new_states.append([h, c])
            states = new_states

            # Remove sequence dimension: (batch, 1, hidden) → (batch, hidden)
            lstm_output = tf.squeeze(lstm_output, axis=1)

            # Predict next activity
            logits = self.activity_output(lstm_output)  # (batch, activity_types)

            # Apply Gumbel-Softmax
            next_token = self.gumbel_softmax(logits, temperature, training)

            generated_tokens.append(next_token)
            current_token = next_token

        # Stack all tokens: (max_activities, batch, activity_types) → (batch, max_activities, activity_types)
        nodes = tf.stack(generated_tokens, axis=1)

        return nodes

    @staticmethod
    def gumbel_softmax(logits, temperature=1.0, training=True, hard=False):
        """
        Gumbel-Softmax sampling

        Args:
            logits: Input logits (batch, num_classes)
            temperature: Softmax temperature
            training: If True, add Gumbel noise
            hard: If True, use straight-through estimator

        Returns:
            Differentiable samples (batch, num_classes)
        """
        if training:
            # Add Gumbel noise
            uniform = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
            gumbel_noise = -tf.math.log(-tf.math.log(uniform + 1e-20) + 1e-20)
            logits = logits + gumbel_noise

        # Softmax
        y_soft = tf.nn.softmax(logits / temperature)

        if hard:
            # Straight-through estimator
            y_hard = tf.one_hot(
                tf.argmax(logits, axis=-1),
                depth=tf.shape(logits)[-1],
                dtype=logits.dtype
            )
            y = tf.stop_gradient(y_hard - y_soft) + y_soft
        else:
            y = y_soft

        return y

    def sample_z(self, batch_size):
        """Sample latent vectors from standard normal"""
        return np.random.normal(0, 1, size=(batch_size, self.embedding_dim)).astype(np.float32)

    def generate(self, batch_size, temperature=0.5, start_token_id=0, end_token_id=None):
        """
        Generate traces (inference mode)

        Args:
            batch_size: Number of traces to generate
            temperature: Sampling temperature
            start_token_id: START activity ID
            end_token_id: END activity ID (optional)

        Returns:
            Generated activity sequences (batch, max_activities, activity_types)
        """
        z = self.sample_z(batch_size)
        return self.call(z, training=False, temperature=temperature,
                        start_token_id=start_token_id, end_token_id=end_token_id)


class ProcessDiscriminatorLSTM(keras.Model):
    """
    Bi-LSTM Discriminator: Sequence → real/fake score

    Processes activity sequences using bidirectional LSTM to capture
    temporal dependencies in both directions.
    """

    def __init__(self, activity_types, lstm_hidden_dim=256, lstm_num_layers=2,
                 mlp_units=128, dropout_rate=0.0, **kwargs):
        """
        Initialize Bi-LSTM Discriminator

        Args:
            activity_types: Number of activity types
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of Bi-LSTM layers
            mlp_units: MLP classifier units
            dropout_rate: Dropout rate
        """
        super(ProcessDiscriminatorLSTM, self).__init__(**kwargs)

        self.activity_types = activity_types
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate

        # Activity embedding
        self.activity_embedding = layers.Embedding(
            activity_types,
            lstm_hidden_dim,
            mask_zero=True,
            name='disc_activity_embedding'
        )

        # Bi-LSTM layers
        self.bilstm_layers = []
        for i in range(lstm_num_layers):
            self.bilstm_layers.append(
                layers.Bidirectional(
                    layers.LSTM(
                        lstm_hidden_dim // 2,  # Divide by 2 because Bi-LSTM concatenates
                        return_sequences=True,
                        dropout=dropout_rate,
                        recurrent_dropout=dropout_rate,
                        name=f'bilstm_{i}_inner'
                    ),
                    name=f'bilstm_{i}'
                )
            )

        # Attention pooling
        self.attention_dense = layers.Dense(1, activation='sigmoid', name='attention')

        # Pooling projection
        self.pooling_dense = layers.Dense(mlp_units, activation='tanh', name='pooling')

        # MLP classifier
        self.mlp = keras.Sequential([
            layers.Dense(128, activation='tanh'),
            layers.Dropout(dropout_rate),
            layers.Dense(64, activation='tanh'),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation=None)  # Logits (not sigmoid, for WGAN)
        ], name='mlp_classifier')

        # Dropout
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, nodes, training=False):
        """
        Forward pass

        Args:
            nodes: Activity sequences (batch, max_activities, activity_types) - soft or one-hot
            training: Training mode flag

        Returns:
            score: Discriminator score (batch, 1)
            features: Intermediate features for feature matching (batch, mlp_units)
        """
        # Convert soft probabilities to hard indices for embedding lookup
        # For training with Gumbel-Softmax, we need to handle soft assignments
        # Use weighted embedding: soft_probs @ embedding_matrix

        # nodes shape: (batch, max_activities, activity_types)
        # embedding matrix: (activity_types, lstm_hidden_dim)
        embedded = tf.matmul(nodes, self.activity_embedding.embeddings)  # (batch, max_activities, hidden_dim)

        # Apply dropout
        h = self.dropout(embedded, training=training)

        # Pass through Bi-LSTM layers
        for bilstm_layer in self.bilstm_layers:
            h = bilstm_layer(h, training=training)  # (batch, max_activities, lstm_hidden_dim)

        # Attention-based pooling
        attention_weights = self.attention_dense(h)  # (batch, max_activities, 1)
        pooled_features = self.pooling_dense(h)      # (batch, max_activities, mlp_units)

        # Weighted sum
        graph_embedding = tf.reduce_sum(attention_weights * pooled_features, axis=1)  # (batch, mlp_units)

        # MLP classifier
        score = self.mlp(graph_embedding, training=training)  # (batch, 1)

        return score, graph_embedding


class ProcessValueNetworkLSTM(keras.Model):
    """
    Value Network for Reinforcement Learning (LSTM-based)

    Estimates expected reward for a given trace.
    Reuses Bi-LSTM discriminator architecture.
    """

    def __init__(self, activity_types, lstm_hidden_dim=256, lstm_num_layers=2,
                 mlp_units=128, dropout_rate=0.0, **kwargs):
        """Initialize Value Network (same architecture as Discriminator)"""
        super(ProcessValueNetworkLSTM, self).__init__(**kwargs)

        # Reuse discriminator architecture
        self.discriminator = ProcessDiscriminatorLSTM(
            activity_types=activity_types,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate
        )

        # Final value layer (sigmoid for [0,1] range like reward)
        self.value_layer = layers.Dense(1, activation='sigmoid', name='value_output')

    def call(self, nodes, training=False):
        """
        Forward pass

        Args:
            nodes: Activity sequences (batch, max_activities, activity_types)
            training: Training mode flag

        Returns:
            value: Estimated reward (batch, 1) ∈ [0,1]
        """
        _, features = self.discriminator(nodes, training=training)
        value = self.value_layer(features)
        return value


class ProcessGANLSTM:
    """
    Complete ProcessGAN model wrapper (LSTM-based)

    Combines LSTM Generator, Bi-LSTM Discriminator, and Value Network.
    """

    def __init__(self, max_activities, activity_types, embedding_dim,
                 lstm_hidden_dim=256, lstm_num_layers=2, mlp_units=128,
                 dropout_rate=0.0):
        """
        Initialize ProcessGAN LSTM

        Args:
            max_activities: Maximum trace length
            activity_types: Number of activity types
            embedding_dim: Dimension of latent space
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            mlp_units: MLP classifier units
            dropout_rate: Dropout rate
        """
        self.max_activities = max_activities
        self.activity_types = activity_types
        self.embedding_dim = embedding_dim

        # Build models
        self.generator = ProcessGeneratorLSTM(
            max_activities=max_activities,
            activity_types=activity_types,
            embedding_dim=embedding_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            dropout_rate=dropout_rate
        )

        self.discriminator = ProcessDiscriminatorLSTM(
            activity_types=activity_types,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate
        )

        self.value_network = ProcessValueNetworkLSTM(
            activity_types=activity_types,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate
        )

    def sample_z(self, batch_size):
        """Sample latent vectors"""
        return self.generator.sample_z(batch_size)

    def generate(self, batch_size, temperature=0.5):
        """
        Generate traces

        Args:
            batch_size: Number of traces
            temperature: Sampling temperature

        Returns:
            Generated activity sequences (batch, max_activities, activity_types)
        """
        return self.generator.generate(batch_size, temperature=temperature)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def sequences_to_traces(node_sequences, dataset):
    """
    Convert batch of sequences to traces

    Args:
        node_sequences: (batch, max_activities, activity_types) - soft or one-hot
        dataset: ProcessDataset instance for decoding

    Returns:
        List of traces (each trace is list of activity names)
    """
    batch_size = node_sequences.shape[0]
    traces = []

    for i in range(batch_size):
        # Get node indices (argmax over activity types)
        node_indices = np.argmax(node_sequences[i], axis=-1)

        # Decode to trace
        trace = []
        for node_idx in node_indices:
            activity = dataset.activity_decoder.get(int(node_idx), 'UNKNOWN')

            # Stop at PAD token
            if activity == 'PAD':
                break

            trace.append(activity)

        traces.append(trace)

    return traces
