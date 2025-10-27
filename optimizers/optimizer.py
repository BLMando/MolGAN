"""
ProcessGAN Optimizer for TensorFlow 2.x

Modern training implementation using tf.GradientTape and eager execution.
Implements WGAN-GP with gradient penalty and RL integration.
"""

import tensorflow as tf
import numpy as np
import warnings
import logging

# Suppress TensorFlow gradient warnings for Value Network
# (Value Network is trained only when lambda_mix < 1.0, causing warnings during @tf.function compilation)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*Gradients do not exist.*')


class ProcessGANTrainer:
    """
    Trainer for ProcessGAN with WGAN-GP + Reinforcement Learning

    Uses modern TF 2.x APIs:
    - tf.GradientTape for gradients
    - Eager execution (no session)
    - tf.keras.optimizers
    """

    def __init__(self, model, dataset=None, learning_rate=1e-4, learning_rate_D=2e-4, learning_rate_V=1e-4, gradient_penalty_weight=10.0,
                 lambda_adv=0.6, lambda_reward=0.4):
        """
        Initialize trainer

        Args:
            model: ProcessGAN instance
            dataset: ProcessDataset instance (needed for matrix→trace decoding)
            learning_rate: Learning rate for AdamW optimizers
            gradient_penalty_weight: Weight for gradient penalty (WGAN-GP)
            lambda_adv: Weight for adversarial loss in generator
            lambda_reward: Weight for reward loss in generator
            
        Note:
            Uses AdamW optimizer with decoupled weight decay for better generalization.
            Weight decay values:
            - Generator: 0.01 (higher to prevent mode collapse)
            - Discriminator: 0.005 (moderate)
            - Value Network: 0.001 (lower for RL)
        """
        self.model = model
        self.dataset = dataset
        self.gradient_penalty_weight = gradient_penalty_weight
        self.lambda_adv = lambda_adv
        self.lambda_reward = lambda_reward

        # Cache for real trace rewards (real traces don't change during training)
        self.real_rewards_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Create optimizers using AdamW (better generalization for small datasets)
        # Generator: higher weight decay to prevent mode collapse
        self.optimizer_G = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,  # 1% weight decay
            beta_1=0.5,
            beta_2=0.9
        )
        # Discriminator: moderate weight decay
        self.optimizer_D = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate_D,
            weight_decay=0.005,  # 0.5% weight decay
            beta_1=0.5,
            beta_2=0.9
        )
        # Value Network: lower weight decay for RL reward prediction
        self.optimizer_V = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate_V,
            weight_decay=0.001,  # 0.1% weight decay
            beta_1=0.5,
            beta_2=0.9
        )

        # Metrics for monitoring
        self.metrics = {
            'loss_D': tf.keras.metrics.Mean(),
            'loss_G': tf.keras.metrics.Mean(),
            'loss_V': tf.keras.metrics.Mean(),
            'loss_RL': tf.keras.metrics.Mean(),
            'grad_penalty': tf.keras.metrics.Mean(),
            'reward_mean': tf.keras.metrics.Mean(),
        }

    @tf.function
    def train_discriminator_step(self, real_adj, real_nodes, z, training=True):
        """
        Train discriminator for one step

        Args:
            real_adj: Real adjacency matrices (batch, max_act, max_act)
            real_nodes: Real node vectors (batch, max_act)
            z: Latent vectors (batch, z_dim)
            training: Training mode

        Returns:
            loss_D: Discriminator loss
            grad_penalty: Gradient penalty
        """
        with tf.GradientTape() as tape:
            # Convert labels to one-hot
            real_adj_onehot = tf.one_hot(
                real_adj, depth=self.model.flow_types, dtype=tf.float32)
            real_nodes_onehot = tf.one_hot(
                real_nodes, depth=self.model.activity_types, dtype=tf.float32)

            # Generate fake samples
            fake_adj, fake_nodes = self.model.generator(z, training=training)

            # Discriminator scores
            D_real, _ = self.model.discriminator(
                real_adj_onehot, real_nodes_onehot, training=training)
            D_fake, _ = self.model.discriminator(
                fake_adj, fake_nodes, training=training)

            # Wasserstein loss
            loss_D = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)

            # Gradient penalty
            grad_penalty = self._gradient_penalty(real_adj_onehot, real_nodes_onehot,
                                                  fake_adj, fake_nodes, training)

            # Total discriminator loss
            total_loss_D = loss_D + self.gradient_penalty_weight * grad_penalty

        # Compute gradients and update
        gradients = tape.gradient(
            total_loss_D, self.model.discriminator.trainable_variables)
        self.optimizer_D.apply_gradients(
            zip(gradients, self.model.discriminator.trainable_variables))

        return loss_D, grad_penalty

    # Removed @tf.function to allow dynamic control flow based on lambda_mix
    def train_generator_step(self, real_adj, real_nodes, z, rewards_real, rewards_fake,
                             lambda_mix=1.0, training=True):
        """
        Train generator for one step

        Args:
            real_adj: Real adjacency matrices
            real_nodes: Real node vectors
            z: Latent vectors
            rewards_real: Rewards for real traces (batch, 1)
            rewards_fake: Rewards for fake traces (batch, 1)
            lambda_mix: Mixing parameter (1.0=pure GAN, 0.0=pure RL)
            training: Training mode

        Returns:
            loss_G: Generator loss
            loss_RL: RL loss
        """
        with tf.GradientTape() as tape:
            # Convert labels to one-hot
            real_adj_onehot = tf.one_hot(
                real_adj, depth=self.model.flow_types, dtype=tf.float32)
            real_nodes_onehot = tf.one_hot(
                real_nodes, depth=self.model.activity_types, dtype=tf.float32)

            # Generate fake samples
            fake_adj, fake_nodes = self.model.generator(z, training=training)

            # Discriminator scores and features
            D_fake, features_fake = self.model.discriminator(
                fake_adj, fake_nodes, training=training)
            _, features_real = self.model.discriminator(
                real_adj_onehot, real_nodes_onehot, training=training)

            # Adversarial loss (feature matching)
            loss_F = tf.reduce_mean((tf.reduce_mean(features_real, axis=0) -
                                    tf.reduce_mean(features_fake, axis=0)) ** 2)

            # RL loss (only if lambda_mix < 1.0)
            if lambda_mix < 1.0:
                # Value network predictions
                V_fake = self.model.value_network(
                    fake_adj, fake_nodes, training=training)
                # RL loss (maximize predicted reward)
                loss_RL = -tf.reduce_mean(V_fake)
            else:
                # Pure GAN mode - no RL loss
                loss_RL = tf.constant(0.0)

            # Combined generator loss
            loss_G_adv = loss_F  # Feature matching
            loss_G_total = lambda_mix * loss_G_adv + \
                (1 - lambda_mix) * tf.abs(loss_G_adv / (loss_RL + 1e-8)) * loss_RL

        # Compute gradients and update
        gradients = tape.gradient(
            loss_G_total, self.model.generator.trainable_variables)
        self.optimizer_G.apply_gradients(
            zip(gradients, self.model.generator.trainable_variables))

        return loss_G_adv, loss_RL

    @tf.function
    def train_value_network_step(self, real_adj, real_nodes, z, rewards_real, rewards_fake, training=True):
        """
        Train value network for one step

        Args:
            real_adj: Real adjacency matrices
            real_nodes: Real node vectors
            z: Latent vectors
            rewards_real: Rewards for real traces (batch, 1)
            rewards_fake: Rewards for fake traces (batch, 1)
            training: Training mode

        Returns:
            loss_V: Value network loss
        """
        with tf.GradientTape() as tape:
            # Convert labels to one-hot
            real_adj_onehot = tf.one_hot(
                real_adj, depth=self.model.flow_types, dtype=tf.float32)
            real_nodes_onehot = tf.one_hot(
                real_nodes, depth=self.model.activity_types, dtype=tf.float32)

            # Generate fake samples
            fake_adj, fake_nodes = self.model.generator(z, training=training)

            # Value predictions
            V_real = self.model.value_network(
                real_adj_onehot, real_nodes_onehot, training=training)
            V_fake = self.model.value_network(
                fake_adj, fake_nodes, training=training)

            # MSE loss
            loss_V_real = tf.reduce_mean((V_real - rewards_real) ** 2)
            loss_V_fake = tf.reduce_mean((V_fake - rewards_fake) ** 2)
            loss_V = loss_V_real + loss_V_fake

        # Compute gradients and update
        gradients = tape.gradient(
            loss_V, self.model.value_network.trainable_variables)
        self.optimizer_V.apply_gradients(
            zip(gradients, self.model.value_network.trainable_variables))

        return loss_V

    def _gradient_penalty(self, real_adj, real_nodes, fake_adj, fake_nodes, training):
        """
        Compute gradient penalty for WGAN-GP

        Args:
            real_adj: Real adjacency (one-hot)
            real_nodes: Real nodes (one-hot)
            fake_adj: Fake adjacency (soft)
            fake_nodes: Fake nodes (soft)
            training: Training mode

        Returns:
            Gradient penalty
        """
        batch_size = tf.shape(real_adj)[0]

        # Random interpolation
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated_adj = alpha * real_adj + (1 - alpha) * fake_adj

        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated_nodes = alpha * real_nodes + (1 - alpha) * fake_nodes

        with tf.GradientTape() as gp_tape:
            gp_tape.watch([interpolated_adj, interpolated_nodes])
            D_interpolated, _ = self.model.discriminator(
                interpolated_adj, interpolated_nodes, training=training)

        # Compute gradients
        gradients = gp_tape.gradient(
            D_interpolated, [interpolated_adj, interpolated_nodes])

        # Gradient norm
        grad_norm = tf.sqrt(
            tf.reduce_sum(gradients[0] ** 2, axis=[1, 2, 3]) +
            tf.reduce_sum(gradients[1] ** 2, axis=[1, 2])
        )

        # Penalty
        gradient_penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)

        return gradient_penalty

    def train_step(self, real_adj, real_nodes, batch_size, n_critic,
                   reward_function, lambda_mix=1.0, temperature=1.0):
        """
        Complete training step (discriminator + generator + value network)

        Args:
            real_adj: Real adjacency matrices (batch, max_act, max_act)
            real_nodes: Real node vectors (batch, max_act)
            batch_size: Batch size
            n_critic: Number of discriminator updates per generator update
            reward_function: ProcessRewardFunction instance
            lambda_mix: GAN/RL mixing parameter
            temperature: Gumbel-Softmax temperature

        Returns:
            Dict with loss values
        """
        # Sample latent vectors
        z = self.model.sample_z(batch_size)

        # Train discriminator (n_critic steps)
        for _ in range(n_critic):
            loss_D, grad_penalty = self.train_discriminator_step(
                real_adj, real_nodes, z, training=True
            )

        # Compute rewards (for RL)
        rl_enabled = lambda_mix < 1.0 and reward_function is not None and self.dataset is not None
        
        # Log RL status once
        if not hasattr(self, '_rl_status_logged'):
            if rl_enabled:
                print(f"RL Training enabled (lambda_mix={lambda_mix:.2f})")
            else:
                print(f"⚠ RL disabled: lambda_mix={lambda_mix}, reward_fn={reward_function is not None}, dataset={self.dataset is not None}")
            self._rl_status_logged = True
        
        if rl_enabled:
            # Generate traces for reward computation
            fake_adj, fake_nodes = self.model.generator(
                z, training=False, temperature=temperature)

            # Convert to traces using matrices_to_traces
            from models.process_gan import matrices_to_traces
            
            # Convert to numpy and get discrete indices
            fake_adj_np = fake_adj.numpy()
            fake_nodes_np = fake_nodes.numpy()
            
            # Argmax to get hard assignments (batch, max_activities, max_activities)
            fake_edges_indices = np.argmax(fake_adj_np, axis=-1)
            
            # Convert matrices to traces using dataset
            fake_traces = matrices_to_traces(
                fake_edges_indices, fake_nodes_np, self.dataset
            )
            
            # Compute rewards using actual reward function
            rewards_fake_np = reward_function.compute_reward(fake_traces)
            rewards_fake = tf.constant(rewards_fake_np, dtype=tf.float32)
            
            # For real traces, we need to decode from real_adj and real_nodes
            # real_adj and real_nodes are already numpy arrays from data.next_train_batch
            
            # Create cache key from real_adj (unique identifier for this batch)
            cache_key = tuple(map(lambda x: tuple(x.flatten()), real_adj))
            
            # Check cache for real rewards (real traces don't change during training)
            if cache_key in self.real_rewards_cache:
                rewards_real_np = self.real_rewards_cache[cache_key]
                self.cache_hits += 1
            else:
                # Cache miss - compute rewards and store
                real_nodes_onehot = np.eye(self.dataset.activity_num_types)[real_nodes]
                real_traces = matrices_to_traces(
                    real_adj, real_nodes_onehot, self.dataset
                )
                rewards_real_np = reward_function.compute_reward(real_traces)
                self.real_rewards_cache[cache_key] = rewards_real_np
                self.cache_misses += 1
            
            rewards_real = tf.constant(rewards_real_np, dtype=tf.float32)

            # Train generator with RL
            loss_G, loss_RL = self.train_generator_step(
                real_adj, real_nodes, z, rewards_real, rewards_fake, lambda_mix, training=True
            )

            # Train value network
            loss_V = self.train_value_network_step(
                real_adj, real_nodes, z, rewards_real, rewards_fake, training=True
            )

            # Update metrics
            self.metrics['reward_mean'].update_state(
                tf.reduce_mean(rewards_fake))
        else:
            # Pure GAN training (no RL)
            rewards_real = tf.zeros((batch_size, 1))
            rewards_fake = tf.zeros((batch_size, 1))

            loss_G, loss_RL = self.train_generator_step(
                real_adj, real_nodes, z, rewards_real, rewards_fake, lambda_mix, training=True
            )
            loss_V = tf.constant(0.0)

        # Update metrics
        self.metrics['loss_D'].update_state(loss_D)
        self.metrics['loss_G'].update_state(loss_G)
        self.metrics['loss_V'].update_state(loss_V)
        self.metrics['loss_RL'].update_state(loss_RL)
        self.metrics['grad_penalty'].update_state(grad_penalty)

        # Return current values
        return {
            'loss_D': loss_D.numpy(),
            'loss_G': loss_G.numpy(),
            'loss_V': loss_V.numpy(),
            'loss_RL': loss_RL.numpy(),
            'grad_penalty': grad_penalty.numpy(),
            'reward_mean': self.metrics['reward_mean'].result().numpy()
        }

    def reset_metrics(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset_states()

    def get_metrics(self):
        """Get current metric values"""
        return {name: metric.result().numpy() for name, metric in self.metrics.items()}
    
    
    def clear_cache(self):
        """Clear reward cache (useful between epochs or datasets)"""
        self.real_rewards_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
