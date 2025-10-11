"""
ProcessGAN: GAN model for process mining event log generation

Adapts GraphGANModel architecture for process mining domain:
- Generator: z → (adjacency_matrix, node_vector)
- Discriminator: (adjacency_matrix, node_vector) → real/fake score
- Uses Gumbel-Softmax for differentiability
- R-GCN discriminator for graph processing
"""

import numpy as np
import tensorflow as tf

from models import postprocess_logits
from utils.layers import multi_dense_layers, multi_graph_convolution_layers, graph_aggregation_layer


class ProcessGANModel(object):
    """
    GAN model for generating process mining traces

    Architecture:
    - Generator: Dense layers → (Adjacency, Nodes)
    - Discriminator: R-GCN → Score [0,1]
    - Value Network: For reinforcement learning (optional)
    """

    def __init__(self, max_activities, flow_types, activity_types, embedding_dim,
                 decoder_units, discriminator_units,
                 soft_gumbel_softmax=False, hard_gumbel_softmax=False,
                 batch_discriminator=False):
        """
        Initialize ProcessGAN model

        Args:
            max_activities: Maximum number of activities per trace
            flow_types: Number of flow control types (SEQUENCE, XOR, AND, LOOP, SKIP)
            activity_types: Number of activity types (Start, Submit, Review, etc.)
            embedding_dim: Dimension of latent space z
            decoder_units: Tuple of units for generator dense layers (e.g., (128, 256, 512))
            discriminator_units: Tuple of units for discriminator (e.g., ((128, 64), 128))
            soft_gumbel_softmax: Use soft Gumbel-Softmax
            hard_gumbel_softmax: Use hard Gumbel-Softmax (straight-through estimator)
            batch_discriminator: Use batch discrimination layer
        """
        self.vertexes = max_activities
        self.edges = flow_types
        self.nodes = activity_types
        self.embedding_dim = embedding_dim
        self.decoder_units = decoder_units
        self.discriminator_units = discriminator_units
        self.batch_discriminator = batch_discriminator

        # Placeholders
        self.training = tf.placeholder_with_default(False, shape=())
        self.dropout_rate = tf.placeholder_with_default(0., shape=())
        self.soft_gumbel_softmax = tf.placeholder_with_default(soft_gumbel_softmax, shape=())
        self.hard_gumbel_softmax = tf.placeholder_with_default(hard_gumbel_softmax, shape=())
        self.temperature = tf.placeholder_with_default(1., shape=())

        # Input placeholders
        self.edges_labels = tf.placeholder(dtype=tf.int64, shape=(None, max_activities, max_activities))
        self.nodes_labels = tf.placeholder(dtype=tf.int64, shape=(None, max_activities))
        self.embeddings = tf.placeholder(dtype=tf.float32, shape=(None, embedding_dim))

        # Reward placeholders (for RL)
        self.rewardR = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.rewardF = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        # Convert labels to one-hot
        self.adjacency_tensor = tf.one_hot(self.edges_labels, depth=flow_types, dtype=tf.float32)
        self.node_tensor = tf.one_hot(self.nodes_labels, depth=activity_types, dtype=tf.float32)

        # Build generator
        with tf.variable_scope('generator'):
            self.edges_logits, self.nodes_logits = self.generator(
                self.embeddings,
                decoder_units,
                max_activities,
                flow_types,
                activity_types,
                training=self.training,
                dropout_rate=self.dropout_rate
            )

        # Process generator outputs
        with tf.name_scope('outputs'):
            (self.edges_softmax, self.nodes_softmax), \
            (self.edges_argmax, self.nodes_argmax), \
            (self.edges_gumbel_logits, self.nodes_gumbel_logits), \
            (self.edges_gumbel_softmax, self.nodes_gumbel_softmax), \
            (self.edges_gumbel_argmax, self.nodes_gumbel_argmax) = postprocess_logits(
                (self.edges_logits, self.nodes_logits),
                temperature=self.temperature
            )

            # Select output based on Gumbel-Softmax mode
            self.edges_hat = tf.case(
                {
                    self.soft_gumbel_softmax: lambda: self.edges_gumbel_softmax,
                    self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                        self.edges_gumbel_argmax - self.edges_gumbel_softmax
                    ) + self.edges_gumbel_softmax
                },
                default=lambda: self.edges_softmax,
                exclusive=True
            )

            self.nodes_hat = tf.case(
                {
                    self.soft_gumbel_softmax: lambda: self.nodes_gumbel_softmax,
                    self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                        self.nodes_gumbel_argmax - self.nodes_gumbel_softmax
                    ) + self.nodes_gumbel_softmax
                },
                default=lambda: self.nodes_softmax,
                exclusive=True
            )

        # Build discriminator
        with tf.name_scope('D_x_real'):
            self.logits_real, self.features_real = self.discriminator(
                (self.adjacency_tensor, None, self.node_tensor),
                units=discriminator_units
            )

        with tf.name_scope('D_x_fake'):
            self.logits_fake, self.features_fake = self.discriminator(
                (self.edges_hat, None, self.nodes_hat),
                units=discriminator_units
            )

        # Build value network (for RL)
        with tf.name_scope('V_x_real'):
            self.value_logits_real = self.value_network(
                (self.adjacency_tensor, None, self.node_tensor),
                units=discriminator_units
            )

        with tf.name_scope('V_x_fake'):
            self.value_logits_fake = self.value_network(
                (self.edges_hat, None, self.nodes_hat),
                units=discriminator_units
            )

    def generator(self, embeddings, units, vertexes, edges, nodes, training, dropout_rate):
        """
        Generator network: z → (Adjacency, Nodes)

        Args:
            embeddings: Latent vector z (batch, embedding_dim)
            units: Tuple of hidden units (e.g., (128, 256, 512))
            vertexes: Max number of activities
            edges: Number of flow types
            nodes: Number of activity types
            training: Training mode flag
            dropout_rate: Dropout rate

        Returns:
            edges_logits: (batch, vertexes, vertexes, edges)
            nodes_logits: (batch, vertexes, nodes)
        """
        # Multi-layer dense network
        output = multi_dense_layers(
            embeddings,
            units=units,
            activation=tf.nn.tanh,
            dropout_rate=dropout_rate,
            training=training
        )

        # Branch A: Adjacency matrix logits
        with tf.variable_scope('edges_logits'):
            edges_logits = tf.layers.dense(
                inputs=output,
                units=edges * vertexes * vertexes,
                activation=None
            )
            edges_logits = tf.reshape(edges_logits, (-1, edges, vertexes, vertexes))

            # Transpose to (batch, vertexes, vertexes, edges)
            edges_logits = tf.transpose(edges_logits, (0, 2, 3, 1))

            # Optional: Make symmetric for undirected graphs
            # For process mining, typically directed, so comment out
            # edges_logits = (edges_logits + tf.matrix_transpose(edges_logits)) / 2

            edges_logits = tf.layers.dropout(edges_logits, dropout_rate, training=training)

        # Branch B: Node vector logits
        with tf.variable_scope('nodes_logits'):
            nodes_logits = tf.layers.dense(
                inputs=output,
                units=vertexes * nodes,
                activation=None
            )
            nodes_logits = tf.reshape(nodes_logits, (-1, vertexes, nodes))
            nodes_logits = tf.layers.dropout(nodes_logits, dropout_rate, training=training)

        return edges_logits, nodes_logits

    def discriminator(self, inputs, units):
        """
        Discriminator network: (Adjacency, Nodes) → Score

        Uses Relational Graph Convolutional Network (R-GCN)

        Args:
            inputs: Tuple (adjacency, hidden, nodes)
            units: Tuple of units ((rgcn_units,), mlp_units, (mlp_units2,))

        Returns:
            logits: Score (batch, 1)
            features: Intermediate features (batch, features_dim)
        """
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            # R-GCN layers
            outputs0 = self.rgcn_encoder(
                inputs,
                units=units[:-1],
                training=self.training,
                dropout_rate=self.dropout_rate
            )

            # MLP classifier
            outputs1 = multi_dense_layers(
                outputs0,
                units=units[-1],
                activation=tf.nn.tanh,
                training=self.training,
                dropout_rate=self.dropout_rate
            )

            # Optional: Batch discrimination
            if self.batch_discriminator:
                outputs_batch = tf.layers.dense(outputs0, units[-2] // 8, activation=tf.tanh)
                outputs_batch = tf.layers.dense(
                    tf.reduce_mean(outputs_batch, 0, keep_dims=True),
                    units[-2] // 8,
                    activation=tf.nn.tanh
                )
                outputs_batch = tf.tile(outputs_batch, (tf.shape(outputs0)[0], 1))
                outputs1 = tf.concat((outputs1, outputs_batch), -1)

            # Final score
            logits = tf.layers.dense(outputs1, units=1)

        return logits, outputs1

    def rgcn_encoder(self, inputs, units, training, dropout_rate):
        """
        R-GCN encoder for graph processing

        Args:
            inputs: Tuple (adjacency, hidden, nodes)
            units: Tuple of RGCN hidden dimensions

        Returns:
            Graph embedding (batch, features)
        """
        adjacency_tensor, hidden_tensor, node_tensor = inputs

        # R-GCN convolution layers
        with tf.variable_scope('graph_convolutions'):
            output = multi_graph_convolution_layers(
                inputs,
                units=units[0],
                activation=tf.nn.tanh,
                dropout_rate=dropout_rate,
                training=training
            )

        # Graph aggregation (pooling)
        with tf.variable_scope('graph_aggregation'):
            annotations = tf.concat(
                (output, hidden_tensor, node_tensor) if hidden_tensor is not None else (output, node_tensor),
                -1
            )

            output = graph_aggregation_layer(
                annotations,
                units=units[1],
                activation=tf.nn.tanh,
                dropout_rate=dropout_rate,
                training=training
            )

        return output

    def value_network(self, inputs, units):
        """
        Value network for reinforcement learning

        Estimates expected reward for a given trace

        Args:
            inputs: Tuple (adjacency, hidden, nodes)
            units: Network units

        Returns:
            value: Estimated reward (batch, 1)
        """
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            outputs = self.rgcn_encoder(
                inputs,
                units=units[:-1],
                training=self.training,
                dropout_rate=self.dropout_rate
            )

            outputs = multi_dense_layers(
                outputs,
                units=units[-1],
                activation=tf.nn.tanh,
                training=self.training,
                dropout_rate=self.dropout_rate
            )

            # Value output (sigmoid for [0,1] range like reward)
            outputs = tf.layers.dense(outputs, units=1, activation=tf.nn.sigmoid)

        return outputs

    def sample_z(self, batch_dim):
        """
        Sample latent vectors from standard normal distribution

        Args:
            batch_dim: Batch size

        Returns:
            Latent vectors (batch_dim, embedding_dim)
        """
        return np.random.normal(0, 1, size=(batch_dim, self.embedding_dim))


# ─────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────

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


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("ProcessGAN Model Test")
    print("=" * 60)

    # Model configuration
    config = {
        'max_activities': 10,
        'flow_types': 5,
        'activity_types': 8,
        'embedding_dim': 16,
        'decoder_units': (128, 256, 512),
        'discriminator_units': ((128, 64), 128, (128, 64)),
    }

    print("\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create model
    print("\nBuilding ProcessGAN model...")
    model = ProcessGANModel(
        max_activities=config['max_activities'],
        flow_types=config['flow_types'],
        activity_types=config['activity_types'],
        embedding_dim=config['embedding_dim'],
        decoder_units=config['decoder_units'],
        discriminator_units=config['discriminator_units'],
        soft_gumbel_softmax=True,
        hard_gumbel_softmax=False,
        batch_discriminator=False
    )

    print("✓ Generator built")
    print("✓ Discriminator built")
    print("✓ Value network built")

    # Test forward pass
    print("\nTesting forward pass...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Sample latent vectors
        batch_size = 4
        z = model.sample_z(batch_size)
        print(f"  Latent vectors shape: {z.shape}")

        # Generate traces
        edges, nodes = sess.run(
            [model.edges_softmax, model.nodes_softmax],
            feed_dict={model.embeddings: z, model.training: False}
        )

        print(f"  Generated edges shape: {edges.shape}")
        print(f"  Generated nodes shape: {nodes.shape}")

        # Test discriminator
        adj_labels = np.zeros((batch_size, config['max_activities'], config['max_activities']), dtype=np.int64)
        node_labels = np.zeros((batch_size, config['max_activities']), dtype=np.int64)

        D_score = sess.run(
            model.logits_real,
            feed_dict={
                model.edges_labels: adj_labels,
                model.nodes_labels: node_labels,
                model.training: False
            }
        )

        print(f"  Discriminator scores shape: {D_score.shape}")

        # Count parameters
        total_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        print(f"\n  Total trainable parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("ProcessGAN model test completed successfully!")
    print("=" * 60)
