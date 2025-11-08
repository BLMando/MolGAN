"""
Graph Process GAN - Main GAN model with Keras API

Combines generator and discriminator for graph-based process mining.
Fully compatible with Keras .fit() API, callbacks, and metrics.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

from graph_generator import GraphGenerator
from graph_discriminator import GraphDiscriminator
from graph_constraints import GraphProcessConstraints
from graph_utils import wasserstein_loss, gradient_penalty_graph


class GraphProcessGAN(keras.Model):
    """
    Main GAN model for process graph generation

    Compatible with Keras API:
    - Custom train_step and test_step
    - Automatic metrics tracking
    - Callback support (ModelCheckpoint, EarlyStopping, TensorBoard, etc.)
    - Validation support
    """

    def __init__(self,
                 max_nodes,
                 num_edge_types,
                 num_activities,
                 start_idx,
                 end_idx,
                 activity_frequencies,
                 # Generator params
                 noise_dim=128,
                 generator_hidden_dims=(256, 512, 1024),
                 generator_dropout=0.1,
                 # Discriminator params
                 rgcn_hidden_dims=(128, 64),
                 mlp_hidden_dims=(128, 64),
                 discriminator_dropout=0.3,
                 # Training params
                 n_critic=5,
                 lambda_gp=10.0,
                 lambda_constraint=0.1,
                 # Temperature scheduling
                 temp_start=5.0,
                 temp_min=0.5,
                 temp_decay=0.99995,
                 name='graph_process_gan'):
        """
        Args:
            max_nodes: Maximum number of nodes in graph
            num_edge_types: Number of edge types
            num_activities: Number of activity types
            start_idx: Index of START activity
            end_idx: Index of END activity
            activity_frequencies: Target activity frequency distribution
            noise_dim: Latent noise dimension
            generator_hidden_dims: Hidden dimensions for generator
            generator_dropout: Dropout rate for generator
            rgcn_hidden_dims: Hidden dimensions for R-GCN in discriminator
            mlp_hidden_dims: Hidden dimensions for MLP in discriminator
            discriminator_dropout: Dropout rate for discriminator
            n_critic: Number of discriminator updates per generator update
            lambda_gp: Gradient penalty coefficient
            lambda_constraint: Constraint loss coefficient
            temp_start: Initial Gumbel-Softmax temperature
            temp_min: Minimum temperature
            temp_decay: Temperature decay rate
            name: Model name
        """
        super(GraphProcessGAN, self).__init__(name=name)

        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        self.num_activities = num_activities
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lambda_constraint = lambda_constraint

        # Temperature for Gumbel-Softmax
        self.temperature = tf.Variable(
            temp_start, trainable=False, dtype=tf.float32, name='temperature')
        self.temp_min = temp_min
        self.temp_decay = temp_decay

        # Build generator
        self.generator = GraphGenerator(
            max_nodes=max_nodes,
            num_edge_types=num_edge_types,
            num_activities=num_activities,
            noise_dim=noise_dim,
            hidden_dims=generator_hidden_dims,
            dropout_rate=generator_dropout
        )

        # Build discriminator
        self.discriminator = GraphDiscriminator(
            num_edge_types=num_edge_types,
            num_activities=num_activities,
            rgcn_hidden_dims=rgcn_hidden_dims,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout_rate=discriminator_dropout
        )

        # Constraints
        self.constraints = GraphProcessConstraints(
            start_idx=start_idx,
            end_idx=end_idx,
            activity_frequencies=activity_frequencies
        )

        # Metrics (tracked automatically by Keras)
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.gp_tracker = keras.metrics.Mean(name='gradient_penalty')
        self.constraint_loss_tracker = keras.metrics.Mean(
            name='constraint_loss')
        self.d_real_score_tracker = keras.metrics.Mean(name='d_real_score')
        self.d_fake_score_tracker = keras.metrics.Mean(name='d_fake_score')
        self.temperature_tracker = keras.metrics.Mean(name='temperature')

    def compile(self, d_optimizer, g_optimizer):
        """
        Custom compile method

        Args:
            d_optimizer: Optimizer for discriminator
            g_optimizer: Optimizer for generator
        """
        super().compile()

        # Add gradient clipping
        d_optimizer.clipnorm = 1.0
        g_optimizer.clipnorm = 1.0

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @property
    def metrics(self):
        """List of metrics tracked by the model"""
        return [
            self.d_loss_tracker,
            self.g_loss_tracker,
            self.gp_tracker,
            self.constraint_loss_tracker,
            self.d_real_score_tracker,
            self.d_fake_score_tracker,
            self.temperature_tracker
        ]

    def train_step(self, real_data):
        """
        Custom training step - called by fit() for each batch

        Args:
            real_data: Real graph data (adjacency, nodes) or (adjacency, nodes, features)

        Returns:
            Dictionary of metric values
        """
        # Unpack data
        if isinstance(real_data, tuple) and len(real_data) == 2:
            real_adj, real_nodes = real_data
        elif isinstance(real_data, tuple) and len(real_data) == 3:
            real_adj, real_nodes, _ = real_data  # Ignore features for now
        else:
            real_adj, real_nodes = real_data, real_data

        batch_size = tf.shape(real_adj)[0]

        # ====================================================================
        # TRAIN DISCRIMINATOR (n_critic times)
        # ====================================================================
        for _ in range(self.n_critic):
            with tf.GradientTape() as tape:
                # Real data scores
                real_scores = self.discriminator(
                    real_adj, real_nodes, training=True)

                # Generate fake data
                z = self.generator.sample_noise(batch_size)
                fake_adj, fake_nodes = self.generator(z, temperature=self.temperature,
                                                      hard=False, training=True)
                fake_scores = self.discriminator(
                    fake_adj, fake_nodes, training=True)

                # Wasserstein loss
                d_loss = wasserstein_loss(real_scores, fake_scores)

                # Gradient penalty
                gp = gradient_penalty_graph(self.discriminator, real_adj, real_nodes,
                                            fake_adj, fake_nodes)

                # Total discriminator loss
                total_d_loss = d_loss + self.lambda_gp * gp

            # Update discriminator
            d_gradients = tape.gradient(
                total_d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables)
            )

        # ====================================================================
        # TRAIN GENERATOR (once)
        # ====================================================================
        with tf.GradientTape() as tape:
            # Generate fake data
            z = self.generator.sample_noise(batch_size)
            fake_adj, fake_nodes = self.generator(z, temperature=self.temperature,
                                                  hard=False, training=True)

            # Discriminator scores for fake data
            fake_scores = self.discriminator(
                fake_adj, fake_nodes, training=True)

            # Generator loss (fool discriminator)
            g_loss = -tf.reduce_mean(fake_scores)

            # Process constraints
            constraint_loss = self.constraints.total_constraint_loss(
                fake_adj, fake_nodes)

            # Total generator loss
            total_g_loss = g_loss + self.lambda_constraint * constraint_loss

        # Update generator
        g_gradients = tape.gradient(
            total_g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        # ====================================================================
        # UPDATE TEMPERATURE (anneal for more discrete outputs)
        # ====================================================================
        new_temp = tf.maximum(
            self.temperature * self.temp_decay, self.temp_min)
        self.temperature.assign(new_temp)

        # ====================================================================
        # UPDATE METRICS
        # ====================================================================
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.gp_tracker.update_state(gp)
        self.constraint_loss_tracker.update_state(constraint_loss)
        self.d_real_score_tracker.update_state(tf.reduce_mean(real_scores))
        self.d_fake_score_tracker.update_state(tf.reduce_mean(fake_scores))
        self.temperature_tracker.update_state(self.temperature)

        # Return metrics (displayed in progress bar)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, real_data):
        """
        Custom validation step - called by evaluate()

        Args:
            real_data: Real graph data

        Returns:
            Dictionary of metric values
        """
        # Unpack data
        if isinstance(real_data, tuple) and len(real_data) == 2:
            real_adj, real_nodes = real_data
        elif isinstance(real_data, tuple) and len(real_data) == 3:
            real_adj, real_nodes, _ = real_data
        else:
            real_adj, real_nodes = real_data, real_data

        batch_size = tf.shape(real_adj)[0]

        # Generate fake data (no gradient tracking)
        z = self.generator.sample_noise(batch_size)
        fake_adj, fake_nodes = self.generator(z, temperature=self.temperature,
                                              hard=False, training=False)

        # Compute scores
        real_scores = self.discriminator(real_adj, real_nodes, training=False)
        fake_scores = self.discriminator(fake_adj, fake_nodes, training=False)

        # Compute losses (for validation metrics)
        d_loss = wasserstein_loss(real_scores, fake_scores)
        g_loss = -tf.reduce_mean(fake_scores)
        constraint_loss = self.constraints.total_constraint_loss(
            fake_adj, fake_nodes)

        # Update metrics
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.constraint_loss_tracker.update_state(constraint_loss)
        self.d_real_score_tracker.update_state(tf.reduce_mean(real_scores))
        self.d_fake_score_tracker.update_state(tf.reduce_mean(fake_scores))

        return {m.name: m.result() for m in self.metrics}

    def generate_graphs(self, num_samples, temperature=0.5, hard=True):
        """
        Generate synthetic process graphs

        Args:
            num_samples: Number of graphs to generate
            temperature: Sampling temperature (lower = more discrete)
            hard: Use hard samples (one-hot)

        Returns:
            adjacency: Generated adjacency matrices (num_samples, max_nodes, max_nodes, edge_types)
            nodes: Generated node matrices (num_samples, max_nodes, num_activities)
        """
        batch_size = 128
        all_adj = []
        all_nodes = []

        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)

            adj, nodes = self.generator.generate(
                num_samples=current_batch_size,
                temperature=temperature,
                hard=hard
            )

            all_adj.append(adj.numpy())
            all_nodes.append(nodes.numpy())

        return np.concatenate(all_adj, axis=0), np.concatenate(all_nodes, axis=0)
