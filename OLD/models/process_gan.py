"""
ProcessGAN Model for TensorFlow 2.x

Modern implementation using tf.keras.Model and eager execution.
Compatible with TensorFlow 2.15+ and Python 3.10+.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ProcessGenerator(keras.Model):
    """
    Generator network: z → (Adjacency, Nodes)

    Transforms latent vector z into process trace represented as graph.
    """

    def __init__(self, max_activities, flow_types, activity_types, embedding_dim,
                 decoder_units=(128, 256, 512), dropout_rate=0.0, enforce_start=False, **kwargs):
        """
        Initialize Generator

        Args:
            max_activities: Maximum number of activities per trace
            flow_types: Number of flow control types
            activity_types: Number of activity types
            embedding_dim: Dimension of latent space z
            decoder_units: Tuple of hidden units for dense layers
            dropout_rate: Dropout rate
        """
        super(ProcessGenerator, self).__init__(**kwargs)

        self.max_activities = max_activities
        self.flow_types = flow_types
        self.activity_types = activity_types
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.enforce_start = enforce_start

        # Dense layers
        self.dense_layers = []
        for units in decoder_units:
            self.dense_layers.append(layers.Dense(units, activation='tanh'))
            self.dense_layers.append(layers.Dropout(dropout_rate))

        # Adjacency matrix branch
        self.edges_dense = layers.Dense(
            flow_types * max_activities * max_activities,
            activation=None,
            name='edges_logits'
        )

        # Node vector branch
        self.nodes_dense = layers.Dense(
            max_activities * activity_types,
            activation=None,
            name='nodes_logits'
        )

    def call(self, embeddings, training=False, temperature=1.0):
        """
        Forward pass

        Args:
            embeddings: Latent vectors z (batch, embedding_dim)
            training: Training mode flag
            temperature: Gumbel-Softmax temperature

        Returns:
            edges: Adjacency matrices (batch, max_act, max_act, flow_types)
            nodes: Node vectors (batch, max_act, activity_types)
        """
        # batch_size = tf.shape(embeddings)[0]
        
        # Pass through dense layers
        h = embeddings
        for layer in self.dense_layers:
            if isinstance(layer, layers.Dropout):
                h = layer(h, training=training)
            else:
                h = layer(h)

        # Adjacency matrix logits
        edges_logits = self.edges_dense(h)
        edges_logits = tf.reshape(
            edges_logits,
            (-1, self.flow_types, self.max_activities, self.max_activities)
        )
        # Transpose to (batch, max_act, max_act, flow_types)
        edges_logits = tf.transpose(edges_logits, (0, 2, 3, 1))

        # Node vector logits
        nodes_logits = self.nodes_dense(h)
        nodes_logits = tf.reshape(
            nodes_logits,
            (-1, self.max_activities, self.activity_types)
        )

        # Apply Gumbel-Softmax
        edges = self.gumbel_softmax(edges_logits, temperature, training)
        nodes = self.gumbel_softmax(nodes_logits, temperature, training)
        
        # Hard constraint: Force first node to be 'Start' (index 0) if enabled
        # if self.enforce_start:
        #     start_onehot = tf.one_hot(
        #         tf.zeros(batch_size, dtype=tf.int32),
        #         depth=self.activity_types,
        #         dtype=tf.float32
        #     )
        #     nodes = tf.concat([
        #         tf.expand_dims(start_onehot, axis=1),
        #         nodes[:, 1:, :]
        #     ], axis=1)

        return edges, nodes

    @staticmethod
    def gumbel_softmax(logits, temperature=1.0, training=True, hard=False):
        """
        Gumbel-Softmax sampling

        Args:
            logits: Input logits
            temperature: Softmax temperature
            training: If True, add Gumbel noise
            hard: If True, use straight-through estimator

        Returns:
            Differentiable samples
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


class ProcessDiscriminator(keras.Model):
    """
    Discriminator network: (Adjacency, Nodes) → Score [0,1]

    Uses Relational Graph Convolutional Network (R-GCN) to process graphs.
    """

    def __init__(self, flow_types, activity_types, rgcn_units=(128, 64),
                 mlp_units=128, dropout_rate=0.0, **kwargs):
        """
        Initialize Discriminator

        Args:
            flow_types: Number of flow control types
            activity_types: Number of activity types
            rgcn_units: Tuple of R-GCN hidden dimensions
            mlp_units: MLP classifier units
            dropout_rate: Dropout rate
        """
        super(ProcessDiscriminator, self).__init__(**kwargs)

        self.flow_types = flow_types
        self.activity_types = activity_types
        self.rgcn_units = rgcn_units
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate

        # R-GCN layers
        self.rgcn_layers = []
        for units in rgcn_units:
            # One dense layer per flow type
            flow_dense_layers = [
                layers.Dense(units, activation=None)
                for _ in range(flow_types)
            ]
            self.rgcn_layers.append({
                'flow_layers': flow_dense_layers,
                'self_layer': layers.Dense(units, activation=None),
                'dropout': layers.Dropout(dropout_rate)
            })

        # Graph pooling (attention-based)
        self.attention_dense = layers.Dense(1, activation='sigmoid', name='attention')
        self.pooling_dense = layers.Dense(mlp_units, activation='tanh', name='pooling')

        # MLP classifier
        self.mlp = keras.Sequential([
            layers.Dense(128, activation='tanh'),
            layers.Dropout(dropout_rate),
            layers.Dense(64, activation='tanh'),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation=None)  # Logits (not sigmoid, for WGAN)
        ], name='mlp_classifier')

    def call(self, adjacency, nodes, training=False):
        """
        Forward pass

        Args:
            adjacency: Adjacency matrices (batch, max_act, max_act, flow_types)
            nodes: Node vectors (batch, max_act, activity_types)
            training: Training mode flag

        Returns:
            score: Discriminator score (batch, 1)
            features: Intermediate features for feature matching (batch, mlp_units)
        """
        # Initial node features
        h = nodes  # (batch, max_act, activity_types)

        # R-GCN layers
        for rgcn_layer in self.rgcn_layers:
            # Aggregate messages by flow type
            messages = []
            for flow_type in range(self.flow_types):
                # Extract adjacency for this flow type
                adj_slice = adjacency[:, :, :, flow_type]  # (batch, max_act, max_act)

                # Transform node features
                transformed = rgcn_layer['flow_layers'][flow_type](h)

                # Message passing: multiply by adjacency
                message = tf.matmul(adj_slice, transformed)
                messages.append(message)

            # Aggregate all flow types
            aggregated = tf.reduce_sum(tf.stack(messages, axis=1), axis=1)

            # Self-connection
            self_msg = rgcn_layer['self_layer'](h)

            # Update with activation
            h = tf.nn.tanh(aggregated + self_msg)
            h = rgcn_layer['dropout'](h, training=training)

        # Graph pooling (attention-based)
        attention_weights = self.attention_dense(h)  # (batch, max_act, 1)
        pooled_features = self.pooling_dense(h)      # (batch, max_act, mlp_units)
        graph_embedding = tf.reduce_sum(attention_weights * pooled_features, axis=1)

        # MLP classifier
        score = self.mlp(graph_embedding, training=training)

        return score, graph_embedding  # Return features for feature matching


class ProcessValueNetwork(keras.Model):
    """
    Value network for Reinforcement Learning

    Estimates expected reward for a given trace.
    """

    def __init__(self, flow_types, activity_types, rgcn_units=(128, 64),
                 mlp_units=128, dropout_rate=0.0, **kwargs):
        """Initialize Value Network (same architecture as Discriminator)"""
        super(ProcessValueNetwork, self).__init__(**kwargs)

        # Reuse discriminator architecture
        self.discriminator = ProcessDiscriminator(
            flow_types=flow_types,
            activity_types=activity_types,
            rgcn_units=rgcn_units,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate
        )

        # Final value layer (sigmoid for [0,1] range like reward)
        self.value_layer = layers.Dense(1, activation='sigmoid', name='value_output')

    def call(self, adjacency, nodes, training=False):
        """
        Forward pass

        Returns:
            value: Estimated reward (batch, 1) ∈ [0,1]
        """
        _, features = self.discriminator(adjacency, nodes, training=training)
        value = self.value_layer(features)
        return value


class ProcessGAN:
    """
    Complete ProcessGAN model wrapper

    Combines Generator, Discriminator, and Value Network.
    """

    def __init__(self, max_activities, flow_types, activity_types, embedding_dim,
                 decoder_units=(128, 256, 512), discriminator_units=(128, 64),
                 mlp_units=128, dropout_rate=0.0, enforce_start=False):
        """
        Initialize ProcessGAN

        Args:
            max_activities: Maximum number of activities per trace
            flow_types: Number of flow control types
            activity_types: Number of activity types
            embedding_dim: Dimension of latent space
            decoder_units: Generator dense layer units
            discriminator_units: R-GCN layer units
            mlp_units: MLP classifier units
            dropout_rate: Dropout rate
        """
        self.max_activities = max_activities
        self.flow_types = flow_types
        self.activity_types = activity_types
        self.embedding_dim = embedding_dim
        self.enforce_start = enforce_start

        # Build models
        self.generator = ProcessGenerator(
            max_activities=max_activities,
            flow_types=flow_types,
            activity_types=activity_types,
            embedding_dim=embedding_dim,
            decoder_units=decoder_units,
            dropout_rate=dropout_rate,
            enforce_start=enforce_start
        )

        self.discriminator = ProcessDiscriminator(
            flow_types=flow_types,
            activity_types=activity_types,
            rgcn_units=discriminator_units,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate
        )

        self.value_network = ProcessValueNetwork(
            flow_types=flow_types,
            activity_types=activity_types,
            rgcn_units=discriminator_units,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate
        )

    def sample_z(self, batch_size):
        """Sample latent vectors"""
        return self.generator.sample_z(batch_size)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def matrices_to_traces(adjacency_matrices, node_vectors, dataset):
    """
    Convert batch of matrices to traces

    Args:
        adjacency_matrices: (batch, max_act, max_act, flow_types)
        node_vectors: (batch, max_act, activity_types)
        dataset: ProcessDataset instance for decoding

    Returns:
        List of traces (each trace is list of activity names)
    """
    batch_size = node_vectors.shape[0]
    traces = []

    for i in range(batch_size):
        # Get node indices (argmax over activity types)
        node_indices = np.argmax(node_vectors[i], axis=-1)

        # Convert to trace
        trace = dataset.matrices_to_trace(node_indices, adjacency_matrices[i])

        traces.append(trace)

    return traces
