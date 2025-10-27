"""
ProcessGAN LSTM Optimizer for TensorFlow 2.x

Training implementation for LSTM-based architecture using tf.GradientTape.
Implements WGAN-GP with gradient penalty and RL integration for sequential generation.
"""

import tensorflow as tf
import numpy as np
import warnings
import logging

# Suppress TensorFlow gradient warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*Gradients do not exist.*')


class ProcessGANLSTMTrainer:
    """
    Trainer for ProcessGAN LSTM with WGAN-GP + Reinforcement Learning

    Similar to ProcessGANTrainer but adapted for sequence-based generation.
    """

    def __init__(self, model, dataset=None, learning_rate=5e-5, learning_rate_D=2e-4,
                 learning_rate_V=5e-5, gradient_penalty_weight=10.0,
                 lambda_adv=0.6, lambda_reward=0.4):
        """
        Initialize LSTM trainer

        Args:
            model: ProcessGANLSTM instance
            dataset: ProcessDataset instance (needed for trace decoding)
            learning_rate: Learning rate for AdamW optimizers
            learning_rate_D: Learning rate for discriminator
            learning_rate_V: Learning rate for value network
            gradient_penalty_weight: Weight for gradient penalty (WGAN-GP)
            lambda_adv: Weight for adversarial loss in generator
            lambda_reward: Weight for reward loss in generator
        """
        self.model = model
        self.dataset = dataset
        self.gradient_penalty_weight = gradient_penalty_weight
        self.lambda_adv = lambda_adv
        self.lambda_reward = lambda_reward

        # Cache for real trace rewards
        self.real_rewards_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Create optimizers using AdamW
        self.optimizer_G = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.5,
            beta_2=0.9
        )
        self.optimizer_D = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate_D,
            weight_decay=0.005,
            beta_1=0.5,
            beta_2=0.9
        )
        self.optimizer_V = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate_V,
            weight_decay=0.001,
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
    def train_discriminator_step(self, real_nodes, z, training=True, temperature=1.0):
        """
        Train discriminator for one step

        Args:
            real_nodes: Real node sequences (batch, max_activities) - activity indices
            z: Latent vectors (batch, z_dim)
            training: Training mode
            temperature: Gumbel-Softmax temperature

        Returns:
            loss_D: Discriminator loss
            grad_penalty: Gradient penalty
        """
        with tf.GradientTape() as tape:
            # Convert real nodes to one-hot
            real_nodes_onehot = tf.one_hot(
                real_nodes, depth=self.model.activity_types, dtype=tf.float32)

            # Generate fake samples
            fake_nodes = self.model.generator(
                z, training=training, temperature=temperature)

            # Discriminator scores
            D_real, _ = self.model.discriminator(real_nodes_onehot, training=training)
            D_fake, _ = self.model.discriminator(fake_nodes, training=training)

            # Wasserstein loss
            loss_D = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)

            # Gradient penalty
            grad_penalty = self._gradient_penalty(
                real_nodes_onehot, fake_nodes, training)

            # Total discriminator loss
            total_loss_D = loss_D + self.gradient_penalty_weight * grad_penalty

        # Compute gradients and update
        gradients = tape.gradient(
            total_loss_D, self.model.discriminator.trainable_variables)
        self.optimizer_D.apply_gradients(
            zip(gradients, self.model.discriminator.trainable_variables))

        return loss_D, grad_penalty

    def train_generator_step(self, real_nodes, z, rewards_real, rewards_fake,
                            lambda_mix=1.0, training=True, temperature=1.0):
        """
        Train generator for one step

        Args:
            real_nodes: Real node sequences
            z: Latent vectors
            rewards_real: Rewards for real traces (batch, 1)
            rewards_fake: Rewards for fake traces (batch, 1)
            lambda_mix: Mixing parameter (1.0=pure GAN, 0.0=pure RL)
            training: Training mode
            temperature: Gumbel-Softmax temperature

        Returns:
            loss_G: Generator loss
            loss_RL: RL loss
        """
        with tf.GradientTape() as tape:
            # Convert real nodes to one-hot
            real_nodes_onehot = tf.one_hot(
                real_nodes, depth=self.model.activity_types, dtype=tf.float32)

            # Generate fake samples
            fake_nodes = self.model.generator(
                z, training=training, temperature=temperature)

            # Discriminator scores and features
            D_fake, features_fake = self.model.discriminator(
                fake_nodes, training=training)
            _, features_real = self.model.discriminator(
                real_nodes_onehot, training=training)

            # Adversarial loss (feature matching)
            loss_F = tf.reduce_mean((tf.reduce_mean(features_real, axis=0) -
                                    tf.reduce_mean(features_fake, axis=0)) ** 2)

            # RL loss (only if lambda_mix < 1.0)
            if lambda_mix < 1.0:
                # Value network predictions
                V_fake = self.model.value_network(fake_nodes, training=training)
                # RL loss (maximize predicted reward)
                loss_RL = -tf.reduce_mean(V_fake)
            else:
                # Pure GAN mode - no RL loss
                loss_RL = tf.constant(0.0)

            # Combined generator loss (FIXED: simple weighted sum)
            loss_G_adv = loss_F  # Feature matching
            loss_G_total = lambda_mix * loss_G_adv + (1 - lambda_mix) * loss_RL

        # Compute gradients and update
        gradients = tape.gradient(
            loss_G_total, self.model.generator.trainable_variables)
        self.optimizer_G.apply_gradients(
            zip(gradients, self.model.generator.trainable_variables))

        return loss_G_adv, loss_RL

    @tf.function
    def train_value_network_step(self, real_nodes, z, rewards_real, rewards_fake,
                                 training=True, temperature=1.0):
        """
        Train value network for one step

        Args:
            real_nodes: Real node sequences
            z: Latent vectors
            rewards_real: Rewards for real traces (batch, 1)
            rewards_fake: Rewards for fake traces (batch, 1)
            training: Training mode
            temperature: Gumbel-Softmax temperature

        Returns:
            loss_V: Value network loss
        """
        with tf.GradientTape() as tape:
            # Convert real nodes to one-hot
            real_nodes_onehot = tf.one_hot(
                real_nodes, depth=self.model.activity_types, dtype=tf.float32)

            # Generate fake samples
            fake_nodes = self.model.generator(
                z, training=training, temperature=temperature)

            # Value predictions
            V_real = self.model.value_network(real_nodes_onehot, training=training)
            V_fake = self.model.value_network(fake_nodes, training=training)

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

    def _gradient_penalty(self, real_nodes, fake_nodes, training):
        """
        Compute gradient penalty for WGAN-GP

        Args:
            real_nodes: Real nodes (one-hot)
            fake_nodes: Fake nodes (soft)
            training: Training mode

        Returns:
            Gradient penalty
        """
        batch_size = tf.shape(real_nodes)[0]

        # Random interpolation
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated_nodes = alpha * real_nodes + (1 - alpha) * fake_nodes

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_nodes)
            D_interpolated, _ = self.model.discriminator(
                interpolated_nodes, training=training)

        # Compute gradients
        gradients = gp_tape.gradient(D_interpolated, interpolated_nodes)

        # Gradient norm
        grad_norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))

        # Penalty
        gradient_penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)

        return gradient_penalty

    def train_step(self, real_nodes, batch_size, n_critic,
                   reward_function, lambda_mix=1.0, temperature=1.0):
        """
        Complete training step (discriminator + generator + value network)

        Args:
            real_nodes: Real node sequences (batch, max_activities)
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
                real_nodes, z, training=True, temperature=temperature
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
            fake_nodes = self.model.generator(
                z, training=False, temperature=temperature)

            # Convert to traces
            from models.process_gan_lstm import sequences_to_traces

            fake_nodes_np = fake_nodes.numpy()
            fake_traces = sequences_to_traces(fake_nodes_np, self.dataset)

            # Compute rewards
            rewards_fake_np = reward_function.compute_reward(fake_traces)
            rewards_fake = tf.constant(rewards_fake_np, dtype=tf.float32)

            # For real traces
            cache_key = tuple(map(lambda x: tuple(x.flatten()), real_nodes))

            if cache_key in self.real_rewards_cache:
                rewards_real_np = self.real_rewards_cache[cache_key]
                self.cache_hits += 1
            else:
                # Decode real traces
                real_nodes_onehot = np.eye(self.dataset.activity_num_types)[real_nodes]
                real_traces = sequences_to_traces(real_nodes_onehot, self.dataset)
                rewards_real_np = reward_function.compute_reward(real_traces)
                self.real_rewards_cache[cache_key] = rewards_real_np
                self.cache_misses += 1

            rewards_real = tf.constant(rewards_real_np, dtype=tf.float32)

            # Train generator with RL
            loss_G, loss_RL = self.train_generator_step(
                real_nodes, z, rewards_real, rewards_fake, lambda_mix,
                training=True, temperature=temperature
            )

            # Train value network
            loss_V = self.train_value_network_step(
                real_nodes, z, rewards_real, rewards_fake,
                training=True, temperature=temperature
            )

            # Update metrics
            self.metrics['reward_mean'].update_state(tf.reduce_mean(rewards_fake))
        else:
            # Pure GAN training (no RL)
            rewards_real = tf.zeros((batch_size, 1))
            rewards_fake = tf.zeros((batch_size, 1))

            loss_G, loss_RL = self.train_generator_step(
                real_nodes, z, rewards_real, rewards_fake, lambda_mix,
                training=True, temperature=temperature
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
        """Clear reward cache"""
        self.real_rewards_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
