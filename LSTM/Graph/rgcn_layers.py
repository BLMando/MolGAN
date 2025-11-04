"""
Relational Graph Convolutional Network (R-GCN) Layers

R-GCN processes graphs with multiple edge types (relations).
Each edge type has its own weight matrix for message passing.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RGCNLayer(keras.layers.Layer):
    """
    Single R-GCN layer for multi-relational graphs

    For each edge type r:
        h_i^(l+1) = σ(Σ_r Σ_{j∈N_i^r} (1/c_{i,r}) * W_r^(l) * h_j^(l) + W_0^(l) * h_i^(l))

    where:
        - h_i^(l): node i features at layer l
        - N_i^r: neighbors of node i via relation r
        - c_{i,r}: normalization constant
        - W_r^(l): weight matrix for relation r
        - W_0^(l): self-loop weight matrix
    """

    def __init__(self,
                 num_edge_types,
                 output_dim,
                 activation='relu',
                 use_bias=True,
                 dropout_rate=0.0,
                 use_layer_norm=False,
                 **kwargs):
        """
        Args:
            num_edge_types: Number of edge/relation types
            output_dim: Output dimension
            activation: Activation function ('relu', 'tanh', etc.)
            use_bias: Whether to use bias
            dropout_rate: Dropout rate
            use_layer_norm: Whether to apply layer normalization
        """
        super(RGCNLayer, self).__init__(**kwargs)

        self.num_edge_types = num_edge_types
        self.output_dim = output_dim
        self.activation_fn = keras.activations.get(activation)
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        # Create weight matrices for each edge type
        self.edge_type_weights = []
        for i in range(num_edge_types):
            self.edge_type_weights.append(
                layers.Dense(output_dim, use_bias=False, name=f'edge_type_{i}')
            )

        # Self-loop weight
        self.self_weight = layers.Dense(
            output_dim, use_bias=False, name='self_loop')

        # Bias (shared across all relations)
        if use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(output_dim,),
                initializer='zeros',
                trainable=True
            )

        # Dropout
        if dropout_rate > 0:
            self.dropout = layers.Dropout(dropout_rate)

        # Layer normalization
        if use_layer_norm:
            self.layer_norm = layers.LayerNormalization()

    def call(self, node_features, adjacency, training=False):
        """
        Forward pass

        Args:
            node_features: Node feature matrix (batch, num_nodes, feature_dim)
            adjacency: Adjacency tensor (batch, num_nodes, num_nodes, num_edge_types)
            training: Training mode flag

        Returns:
            Updated node features (batch, num_nodes, output_dim)
        """
        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]

        # Message passing for each edge type
        messages = []
        for edge_type_idx in range(self.num_edge_types):
            # Extract adjacency for this edge type
            # (batch, nodes, nodes)
            adj_slice = adjacency[:, :, :, edge_type_idx]

            # Transform features
            transformed = self.edge_type_weights[edge_type_idx](node_features)

            # Aggregate messages via matrix multiplication
            # adj_slice @ transformed = sum over neighbors
            # (batch, nodes, output_dim)
            message = tf.matmul(adj_slice, transformed)

            # Normalization: divide by in-degree for this edge type
            # in_degree should be (batch, nodes) - sum over source nodes (axis=1)
            in_degree = tf.reduce_sum(adj_slice, axis=1)  # (batch, nodes)
            in_degree = tf.maximum(in_degree, 1.0)  # Avoid division by zero
            # Expand to (batch, nodes, 1) for broadcasting
            in_degree = tf.expand_dims(in_degree, axis=-1)  # (batch, nodes, 1)
            # (batch, nodes, output_dim) / (batch, nodes, 1)
            message = message / in_degree

            messages.append(message)

        # Aggregate all edge types
        aggregated_messages = tf.add_n(messages)  # Sum over edge types

        # Self-loop (update with node's own features)
        self_message = self.self_weight(node_features)

        # Combine aggregated messages and self-loop
        output = aggregated_messages + self_message

        # Add bias
        if self.use_bias:
            output = output + self.bias

        # Apply activation
        output = self.activation_fn(output)

        # Layer normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)

        # Dropout
        if self.dropout_rate > 0:
            output = self.dropout(output, training=training)

        return output


class RGCNStack(keras.Model):
    """
    Stack of R-GCN layers
    """

    def __init__(self,
                 num_edge_types,
                 hidden_dims=(128, 64),
                 dropout_rate=0.3,
                 use_layer_norm=True,
                 name='rgcn_stack'):
        """
        Args:
            num_edge_types: Number of edge types
            hidden_dims: Tuple of hidden dimensions for each layer
            dropout_rate: Dropout rate
            use_layer_norm: Use layer normalization
            name: Model name
        """
        super(RGCNStack, self).__init__(name=name)

        self.num_edge_types = num_edge_types
        self.hidden_dims = hidden_dims

        # Create R-GCN layers
        self.rgcn_layers = []
        for i, dim in enumerate(hidden_dims):
            layer = RGCNLayer(
                num_edge_types=num_edge_types,
                output_dim=dim,
                activation='relu',
                dropout_rate=dropout_rate,
                use_layer_norm=use_layer_norm,
                name=f'rgcn_layer_{i}'
            )
            self.rgcn_layers.append(layer)

    def call(self, node_features, adjacency, training=False):
        """
        Forward pass through all layers

        Args:
            node_features: Initial node features (batch, nodes, features)
            adjacency: Adjacency tensor (batch, nodes, nodes, edge_types)
            training: Training mode

        Returns:
            Final node features (batch, nodes, hidden_dims[-1])
        """
        h = node_features

        for layer in self.rgcn_layers:
            h = layer(h, adjacency, training=training)

        return h
