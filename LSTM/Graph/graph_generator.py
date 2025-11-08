"""
Graph Generator - Generate process graphs from latent noise

Architecture:
    z (noise) → Dense layers → Split into two heads:
        1. Adjacency head → (max_nodes, max_nodes, num_edge_types)
        2. Node head → (max_nodes, num_activities)
    
Uses Gumbel-Softmax for differentiable discrete sampling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from graph_utils import gumbel_softmax


class GraphGenerator(keras.Model):
    """
    Generator network: z → (Adjacency, Nodes)

    Transforms latent vector z into process graph represented as:
    - Adjacency matrix with multiple edge types
    - Node feature matrix with activities
    """

    def __init__(self,
                 max_nodes,
                 num_edge_types,
                 num_activities,
                 noise_dim=128,
                 hidden_dims=(256, 512, 1024),
                 dropout_rate=0.1,
                 name='graph_generator'):
        """
        Args:
            max_nodes: Maximum number of nodes in graph
            num_edge_types: Number of edge types (SEQUENCE, AND_SPLIT, etc.)
            num_activities: Number of activity types
            noise_dim: Dimension of latent noise vector z
            hidden_dims: Tuple of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            name: Model name
        """
        super(GraphGenerator, self).__init__(name=name)

        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        self.num_activities = num_activities
        self.noise_dim = noise_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # +1 for NO_EDGE channel (allows sparsity)
        self.num_edge_types_with_no_edge = num_edge_types + 1

        # Dense layers for processing noise
        self.dense_layers = []
        for i, units in enumerate(hidden_dims):
            self.dense_layers.append(
                layers.Dense(units, activation='tanh', name=f'dense_{i}')
            )
            if dropout_rate > 0:
                self.dense_layers.append(
                    layers.Dropout(dropout_rate, name=f'dropout_{i}')
                )

        # Adjacency matrix branch (output has +1 channel for NO_EDGE)
        adj_output_dim = max_nodes * max_nodes * (num_edge_types + 1)
        self.adjacency_head = keras.Sequential([
            layers.Dense(adj_output_dim, activation=None, name='adj_logits')
        ], name='adjacency_head')

        # Node matrix branch
        node_output_dim = max_nodes * num_activities
        self.node_head = keras.Sequential([
            layers.Dense(node_output_dim, activation=None, name='node_logits')
        ], name='node_head')

    def call(self, z, temperature=1.0, hard=False, training=True):
        """
        Forward pass

        Args:
            z: Latent noise vectors (batch_size, noise_dim)
            temperature: Gumbel-Softmax temperature (lower = more discrete)
            hard: If True, use straight-through estimator (one-hot in forward pass)
            training: Training mode flag

        Returns:
            adjacency: Adjacency matrices (batch, max_nodes, max_nodes, num_edge_types)
            nodes: Node matrices (batch, max_nodes, num_activities)
        """
        batch_size = tf.shape(z)[0]

        # Process through dense layers
        h = z
        for layer in self.dense_layers:
            if isinstance(layer, layers.Dropout):
                h = layer(h, training=training)
            else:
                h = layer(h)

        # Generate adjacency matrix logits (with NO_EDGE channel)
        adj_logits = self.adjacency_head(h)
        adj_logits = tf.reshape(
            adj_logits,
            (batch_size, self.max_nodes, self.max_nodes,
             self.num_edge_types_with_no_edge)
        )

        # Generate node matrix logits
        node_logits = self.node_head(h)
        node_logits = tf.reshape(
            node_logits,
            (batch_size, self.max_nodes, self.num_activities)
        )

        # Apply Gumbel-Softmax for differentiable sampling
        # For adjacency: sample over [edge_type_0, ..., edge_type_n, NO_EDGE]
        adjacency_with_no_edge = gumbel_softmax(
            adj_logits, temperature=temperature, hard=hard, axis=-1)

        # Remove NO_EDGE channel to get only real edges (shape: batch, nodes, nodes, num_edge_types)
        adjacency = adjacency_with_no_edge[:, :, :, :-1]

        nodes = gumbel_softmax(
            node_logits, temperature=temperature, hard=hard, axis=-1)

        return adjacency, nodes

    def sample_noise(self, batch_size):
        """
        Sample latent noise vectors from standard normal distribution

        Args:
            batch_size: Number of samples

        Returns:
            z: Noise vectors (batch_size, noise_dim)
        """
        return tf.random.normal([batch_size, self.noise_dim])

    def generate(self, num_samples, temperature=0.5, hard=True):
        """
        Generate graphs (convenience method for inference)

        Args:
            num_samples: Number of graphs to generate
            temperature: Sampling temperature
            hard: Use hard samples (one-hot)

        Returns:
            adjacency: Generated adjacency matrices
            nodes: Generated node matrices
        """
        z = self.sample_noise(num_samples)
        adjacency, nodes = self(
            z, temperature=temperature, hard=hard, training=False)
        return adjacency, nodes


class GraphGeneratorWithFeatures(GraphGenerator):
    """
    Extended generator that also produces temporal features
    """

    def __init__(self,
                 max_nodes,
                 num_edge_types,
                 num_activities,
                 num_features=3,
                 noise_dim=128,
                 hidden_dims=(256, 512, 1024),
                 dropout_rate=0.0,
                 name='graph_generator_features'):
        """
        Args:
            num_features: Number of temporal features (default 3: norm_time, trace_time, prev_time)
            Other args same as GraphGenerator
        """
        super().__init__(
            max_nodes=max_nodes,
            num_edge_types=num_edge_types,
            num_activities=num_activities,
            noise_dim=noise_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            name=name
        )

        self.num_features = num_features

        # Feature matrix branch
        feature_output_dim = max_nodes * num_features
        self.feature_head = keras.Sequential([
            layers.Dense(feature_output_dim, activation='sigmoid',
                         name='feature_output')
        ], name='feature_head')

    def call(self, z, temperature=1.0, hard=False, training=True):
        """
        Forward pass with features

        Returns:
            adjacency: Adjacency matrices
            nodes: Node matrices
            features: Feature matrices (batch, max_nodes, num_features)
        """
        batch_size = tf.shape(z)[0]

        # Get base outputs
        adjacency, nodes = super().call(z, temperature, hard, training)

        # Process through dense layers again for features
        h = z
        for layer in self.dense_layers:
            if isinstance(layer, layers.Dropout):
                h = layer(h, training=training)
            else:
                h = layer(h)

        # Generate features (normalized between 0 and 1)
        features = self.feature_head(h)
        features = tf.reshape(
            features,
            (batch_size, self.max_nodes, self.num_features)
        )

        return adjacency, nodes, features

    def generate(self, num_samples, temperature=0.5, hard=True):
        """Generate graphs with features"""
        z = self.sample_noise(num_samples)
        adjacency, nodes, features = self(
            z, temperature=temperature, hard=hard, training=False)
        return adjacency, nodes, features
