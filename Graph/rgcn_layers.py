import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RGCNLayer(keras.layers.Layer):
    """
    Single GCN layer for binary graphs
    """

    def __init__(self,
                 output_dim,
                 activation='relu',
                 use_bias=True,
                 dropout_rate=0.0,
                 use_layer_norm=False,
                 **kwargs):
        """
        Args:
            output_dim: Output dimension
            activation: Activation function ('relu', 'tanh', etc.)
            use_bias: Whether to use bias
            dropout_rate: Dropout rate
            use_layer_norm: Whether to apply layer normalization
        """
        super(RGCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation_fn = keras.activations.get(activation)
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        self.edge_weight = layers.Dense(output_dim, use_bias=False, name='edge_transform')
        
        self.self_weight = layers.Dense(
            output_dim, use_bias=False, name='self_loop')

        if use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(output_dim,),
                initializer='zeros',
                trainable=True
            )

        if dropout_rate > 0:
            self.dropout = layers.Dropout(dropout_rate)

        if use_layer_norm:
            self.layer_norm = layers.LayerNormalization()

    def call(self, node_features, adjacency, training=False):
        """
        Forward pass

        Args:
            node_features: Node feature matrix (batch, num_nodes, feature_dim)
            adjacency: Binary adjacency matrix (batch, num_nodes, num_nodes)
            training: Training mode flag

        Returns:
            Updated node features (batch, num_nodes, output_dim)
        """

        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]

        transformed = self.edge_weight(node_features)  # (batch, nodes, output_dim)

        message = tf.matmul(adjacency, transformed)  # (batch, nodes, output_dim)

        # Normalization: divide by in-degree
        in_degree = tf.reduce_sum(adjacency, axis=1)  # (batch, nodes)
        in_degree = tf.maximum(in_degree, 1.0)
        in_degree = tf.expand_dims(in_degree, axis=-1)  # (batch, nodes, 1)
        aggregated_messages = message / in_degree

        self_message = self.self_weight(node_features)

        output = aggregated_messages + self_message

        if self.use_bias:
            output = output + self.bias

        output = self.activation_fn(output)

        if self.use_layer_norm:
            output = self.layer_norm(output)

        if self.dropout_rate > 0:
            output = self.dropout(output, training=training)

        return output


class RGCNStack(keras.Model):
    """
    Stack of R-GCN layers
    """

    def __init__(self,
                 hidden_dims=(128, 64),
                 dropout_rate=0.3,
                 use_layer_norm=True,
                 name='rgcn_stack'):
        """
        Args:
            hidden_dims: Tuple of hidden dimensions for each layer
            dropout_rate: Dropout rate
            use_layer_norm: Use layer normalization
            name: Model name
        """
        super(RGCNStack, self).__init__(name=name)

        self.hidden_dims = hidden_dims

        self.rgcn_layers = []
        for i, dim in enumerate(hidden_dims):
            layer = RGCNLayer(
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
            adjacency: Binary adjacency matrix (batch, nodes, nodes)
            training: Training mode

        Returns:
            Final node features (batch, nodes, hidden_dims[-1])
        """
        h = node_features

        for layer in self.rgcn_layers:
            h = layer(h, adjacency, training=training)

        return h
