"""
ProcessGAN Optimizer for TensorFlow 2.x

Modern training implementation using tf.GradientTape and eager execution.
Implements WGAN-GP with gradient penalty and RL integration.
"""

import tensorflow as tf
import numpy as np


class ProcessGANTrainer:
    """
    Trainer for ProcessGAN with WGAN-GP + Reinforcement Learning

    Uses modern TF 2.x APIs:
    - tf.GradientTape for gradients
    - Eager execution (no session)
    - tf.keras.optimizers
    """

    def __init__(self, model, learning_rate=1e-4, gradient_penalty_weight=10.0,
                 lambda_adv=0.6, lambda_reward=0.4):
        """
        Initialize trainer

        Args:
            model: ProcessGAN instance
            learning_rate: Learning rate for Adam optimizers
            gradient_penalty_weight: Weight for gradient penalty (WGAN-GP)
            lambda_adv: Weight for adversarial loss in generator
            lambda_reward: Weight for reward loss in generator
        """
        self.model = model
        self.gradient_penalty_weight = gradient_penalty_weight
        self.lambda_adv = lambda_adv
        self.lambda_reward = lambda_reward

        # Create optimizers
        self.optimizer_G = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)
        self.optimizer_D = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)
        self.optimizer_V = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)

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
            real_adj_onehot = tf.one_hot(real_adj, depth=self.model.flow_types, dtype=tf.float32)
            real_nodes_onehot = tf.one_hot(real_nodes, depth=self.model.activity_types, dtype=tf.float32)

            # Generate fake samples
            fake_adj, fake_nodes = self.model.generator(z, training=training)

            # Discriminator scores
            D_real, _ = self.model.discriminator(real_adj_onehot, real_nodes_onehot, training=training)
            D_fake, _ = self.model.discriminator(fake_adj, fake_nodes, training=training)

            # Wasserstein loss
            loss_D = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)

            # Gradient penalty
            grad_penalty = self._gradient_penalty(real_adj_onehot, real_nodes_onehot,
                                                 fake_adj, fake_nodes, training)

            # Total discriminator loss
            total_loss_D = loss_D + self.gradient_penalty_weight * grad_penalty

        # Compute gradients and update
        gradients = tape.gradient(total_loss_D, self.model.discriminator.trainable_variables)
        self.optimizer_D.apply_gradients(zip(gradients, self.model.discriminator.trainable_variables))

        return loss_D, grad_penalty

    @tf.function
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
            real_adj_onehot = tf.one_hot(real_adj, depth=self.model.flow_types, dtype=tf.float32)
            real_nodes_onehot = tf.one_hot(real_nodes, depth=self.model.activity_types, dtype=tf.float32)

            # Generate fake samples
            fake_adj, fake_nodes = self.model.generator(z, training=training)

            # Discriminator scores and features
            D_fake, features_fake = self.model.discriminator(fake_adj, fake_nodes, training=training)
            _, features_real = self.model.discriminator(real_adj_onehot, real_nodes_onehot, training=training)

            # Value network predictions
            V_fake = self.model.value_network(fake_adj, fake_nodes, training=training)

            # Adversarial loss (feature matching)
            loss_F = tf.reduce_mean((tf.reduce_mean(features_real, axis=0) -
                                    tf.reduce_mean(features_fake, axis=0)) ** 2)

            # RL loss (maximize predicted reward)
            loss_RL = -tf.reduce_mean(V_fake)

            # Combined generator loss
            loss_G_adv = loss_F  # Feature matching
            loss_G_total = lambda_mix * loss_G_adv + (1 - lambda_mix) * tf.abs(loss_G_adv / (loss_RL + 1e-8)) * loss_RL

        # Compute gradients and update
        gradients = tape.gradient(loss_G_total, self.model.generator.trainable_variables)
        self.optimizer_G.apply_gradients(zip(gradients, self.model.generator.trainable_variables))

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
            real_adj_onehot = tf.one_hot(real_adj, depth=self.model.flow_types, dtype=tf.float32)
            real_nodes_onehot = tf.one_hot(real_nodes, depth=self.model.activity_types, dtype=tf.float32)

            # Generate fake samples
            fake_adj, fake_nodes = self.model.generator(z, training=training)

            # Value predictions
            V_real = self.model.value_network(real_adj_onehot, real_nodes_onehot, training=training)
            V_fake = self.model.value_network(fake_adj, fake_nodes, training=training)

            # MSE loss
            loss_V_real = tf.reduce_mean((V_real - rewards_real) ** 2)
            loss_V_fake = tf.reduce_mean((V_fake - rewards_fake) ** 2)
            loss_V = loss_V_real + loss_V_fake

        # Compute gradients and update
        gradients = tape.gradient(loss_V, self.model.value_network.trainable_variables)
        self.optimizer_V.apply_gradients(zip(gradients, self.model.value_network.trainable_variables))

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
            D_interpolated, _ = self.model.discriminator(interpolated_adj, interpolated_nodes, training=training)

        # Compute gradients
        gradients = gp_tape.gradient(D_interpolated, [interpolated_adj, interpolated_nodes])

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
        if lambda_mix < 1.0:
            # Generate traces for reward computation
            fake_adj, fake_nodes = self.model.generator(z, training=False, temperature=temperature)

            # Convert to traces (requires dataset for decoding)
            # This is done outside @tf.function for compatibility
            fake_adj_np = fake_adj.numpy()
            fake_nodes_np = fake_nodes.numpy()

            # Compute rewards (external function, not in graph)
            # Placeholder: in real implementation, use reward_function
            rewards_fake = tf.constant(np.random.uniform(0.5, 0.9, (batch_size, 1)), dtype=tf.float32)
            rewards_real = tf.constant(np.random.uniform(0.6, 1.0, (batch_size, 1)), dtype=tf.float32)

            # Train generator with RL
            loss_G, loss_RL = self.train_generator_step(
                real_adj, real_nodes, z, rewards_real, rewards_fake, lambda_mix, training=True
            )

            # Train value network
            loss_V = self.train_value_network_step(
                real_adj, real_nodes, z, rewards_real, rewards_fake, training=True
            )

            # Update metrics
            self.metrics['reward_mean'].update_state(tf.reduce_mean(rewards_fake))
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


if __name__ == '__main__':
    # Test ProcessGAN Trainer TF2
    print("=" * 70)
    print("ProcessGAN Trainer TensorFlow 2.x Test")
    print("=" * 70)

    from models.process_gan_tf2 import ProcessGAN

    # Configuration
    config = {
        'max_activities': 10,
        'flow_types': 5,
        'activity_types': 8,
        'embedding_dim': 16,
        'decoder_units': (128, 256, 512),
        'discriminator_units': (128, 64),
        'mlp_units': 128,
    }

    print("\nCreating model and trainer...")
    model = ProcessGAN(**config)
    trainer = ProcessGANTrainer(model, learning_rate=1e-4)

    print("✓ Model created")
    print("✓ Trainer created")

    # Dummy data
    print("\nTesting training step...")
    batch_size = 4
    real_adj = np.zeros((batch_size, config['max_activities'], config['max_activities']), dtype=np.int32)
    real_nodes = np.zeros((batch_size, config['max_activities']), dtype=np.int32)

    # Training step
    losses = trainer.train_step(
        real_adj=real_adj,
        real_nodes=real_nodes,
        batch_size=batch_size,
        n_critic=5,
        reward_function=None,
        lambda_mix=1.0,
        temperature=1.0
    )

    print("\nLoss values:")
    for key, value in losses.items():
        print(f"  {key}: {value:.4f}")

    print("\n✓ Training step successful")

    print("\n" + "=" * 70)
    print("ProcessGAN Trainer TF2 test completed successfully!")
    print("=" * 70)
