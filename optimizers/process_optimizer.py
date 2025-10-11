"""
ProcessGANOptimizer: Optimizer for training ProcessGAN

Implements:
- Wasserstein GAN loss with gradient penalty
- Adversarial training (discriminator vs generator)
- Reinforcement learning integration (reward maximization)
- Feature matching (optional)
"""

import tensorflow as tf


class ProcessGANOptimizer(object):
    """
    Optimizer for ProcessGAN training

    Loss components:
    - Discriminator: Wasserstein loss + gradient penalty
    - Generator: Adversarial loss + RL reward + feature matching
    - Value network: Mean squared error for reward prediction
    """

    def __init__(self, model, learning_rate=1e-3, feature_matching=True,
                 gradient_penalty_weight=10.0, lambda_adv=0.6, lambda_reward=0.4):
        """
        Initialize optimizer

        Args:
            model: ProcessGANModel instance
            learning_rate: Learning rate for Adam optimizer
            feature_matching: Use feature matching loss
            gradient_penalty_weight: Weight for gradient penalty (default: 10.0)
            lambda_adv: Weight for adversarial loss in generator (default: 0.6)
            lambda_reward: Weight for reward loss in generator (default: 0.4)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.feature_matching = feature_matching
        self.gradient_penalty_weight = gradient_penalty_weight
        self.lambda_adv = lambda_adv
        self.lambda_reward = lambda_reward

        # Lambda placeholder for mixing GAN and RL losses
        self.la = tf.placeholder_with_default(1., shape=())

        # Build loss functions
        self._build_losses()

        # Build training ops
        self._build_training_ops()

    def _build_losses(self):
        """Build all loss functions"""
        with tf.name_scope('losses'):
            # ─────────────────────────────────────────────────────────
            # GRADIENT PENALTY (for WGAN-GP stability)
            # ─────────────────────────────────────────────────────────

            # Random interpolation
            eps = tf.random_uniform(tf.shape(self.model.logits_real)[:1], dtype=self.model.logits_real.dtype)

            # Interpolate adjacency
            x_int0 = self.model.adjacency_tensor * tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1), -1) + \
                     self.model.edges_softmax * (1 - tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1), -1))

            # Interpolate nodes
            x_int1 = self.model.node_tensor * tf.expand_dims(tf.expand_dims(eps, -1), -1) + \
                     self.model.nodes_softmax * (1 - tf.expand_dims(tf.expand_dims(eps, -1), -1))

            # Compute gradients
            grad0, grad1 = tf.gradients(
                self.model.discriminator((x_int0, None, x_int1), self.model.discriminator_units)[0],
                (x_int0, x_int1)
            )

            # Gradient penalty
            self.grad_penalty = tf.reduce_mean(((1 - tf.norm(grad0, axis=-1)) ** 2), (-2, -1)) + \
                               tf.reduce_mean(((1 - tf.norm(grad1, axis=-1)) ** 2), -1, keep_dims=True)

            # ─────────────────────────────────────────────────────────
            # DISCRIMINATOR LOSS
            # ─────────────────────────────────────────────────────────

            # Wasserstein loss
            self.loss_D = -self.model.logits_real + self.model.logits_fake

            # ─────────────────────────────────────────────────────────
            # GENERATOR LOSS
            # ─────────────────────────────────────────────────────────

            # Adversarial loss
            self.loss_G = -self.model.logits_fake

            # Feature matching loss
            self.loss_F = (tf.reduce_mean(self.model.features_real, 0) -
                          tf.reduce_mean(self.model.features_fake, 0)) ** 2

            # ─────────────────────────────────────────────────────────
            # VALUE NETWORK LOSS (for RL)
            # ─────────────────────────────────────────────────────────

            self.loss_V = (self.model.value_logits_real - self.model.rewardR) ** 2 + \
                         (self.model.value_logits_fake - self.model.rewardF) ** 2

            # ─────────────────────────────────────────────────────────
            # RL LOSS (maximize reward)
            # ─────────────────────────────────────────────────────────

            self.loss_RL = -self.model.value_logits_fake

        # Aggregate losses
        self.loss_D = tf.reduce_mean(self.loss_D)
        self.loss_G = tf.reduce_sum(self.loss_F) if self.feature_matching else tf.reduce_mean(self.loss_G)
        self.loss_V = tf.reduce_mean(self.loss_V)
        self.loss_RL = tf.reduce_mean(self.loss_RL)
        self.grad_penalty = tf.reduce_mean(self.grad_penalty)

        # Compute alpha for balancing G and RL losses
        self.alpha = tf.abs(tf.stop_gradient(self.loss_G / (self.loss_RL + 1e-8)))

    def _build_training_ops(self):
        """Build training operations"""
        with tf.name_scope('train_step'):
            # ─────────────────────────────────────────────────────────
            # DISCRIMINATOR TRAINING
            # ─────────────────────────────────────────────────────────

            self.train_step_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                loss=self.loss_D + self.gradient_penalty_weight * self.grad_penalty,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            )

            # ─────────────────────────────────────────────────────────
            # GENERATOR TRAINING (with RL)
            # ─────────────────────────────────────────────────────────

            # Conditional loss based on lambda
            loss_G_combined = tf.cond(
                tf.greater(self.la, 0),
                lambda: self.la * self.loss_G,
                lambda: 0.
            ) + tf.cond(
                tf.less(self.la, 1),
                lambda: (1 - self.la) * self.alpha * self.loss_RL,
                lambda: 0.
            )

            self.train_step_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                loss=loss_G_combined,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            )

            # ─────────────────────────────────────────────────────────
            # VALUE NETWORK TRAINING
            # ─────────────────────────────────────────────────────────

            self.train_step_V = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                loss=self.loss_V,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value')
            )

    def get_loss_dict(self):
        """
        Get dictionary of loss tensors for monitoring

        Returns:
            Dict with loss tensors
        """
        return {
            'loss_D': self.loss_D,
            'loss_G': self.loss_G,
            'loss_V': self.loss_V,
            'loss_RL': self.loss_RL,
            'loss_F': self.loss_F,
            'grad_penalty': self.grad_penalty,
            'alpha': self.alpha,
            'la': self.la
        }

    def get_training_ops(self):
        """
        Get training operations

        Returns:
            Dict with training ops
        """
        return {
            'train_D': self.train_step_D,
            'train_G': self.train_step_G,
            'train_V': self.train_step_V
        }


# ─────────────────────────────────────────────────────────
# TRAINING HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────

def compute_reward_batch(traces, reward_function):
    """
    Compute rewards for batch of traces

    Args:
        traces: List of traces
        reward_function: ProcessRewardFunction instance

    Returns:
        Rewards array (batch, 1)
    """
    return reward_function.compute_reward(traces)


def training_step(session, model, optimizer, data, batch_size, n_critic, la,
                 reward_function, epoch, use_rl=True):
    """
    Perform one training step

    Args:
        session: TensorFlow session
        model: ProcessGANModel instance
        optimizer: ProcessGANOptimizer instance
        data: ProcessDataset instance
        batch_size: Batch size
        n_critic: Number of discriminator updates per generator update
        la: Lambda for mixing GAN/RL losses
        reward_function: ProcessRewardFunction instance
        epoch: Current epoch number
        use_rl: Use reinforcement learning

    Returns:
        Dict with loss values
    """
    # Get real traces from dataset
    traces_real, adj_real, nodes_real, features_real = data.next_train_batch(batch_size)

    # Sample latent vectors
    embeddings = model.sample_z(batch_size)

    # ─────────────────────────────────────────────────────────
    # TRAIN DISCRIMINATOR (n_critic steps)
    # ─────────────────────────────────────────────────────────

    for _ in range(n_critic):
        _ = session.run(
            optimizer.train_step_D,
            feed_dict={
                model.edges_labels: adj_real,
                model.nodes_labels: nodes_real,
                model.embeddings: embeddings,
                model.training: True,
                model.dropout_rate: 0.0
            }
        )

    # ─────────────────────────────────────────────────────────
    # TRAIN GENERATOR (with optional RL)
    # ─────────────────────────────────────────────────────────

    # Compute rewards if using RL
    if use_rl and la < 1.0:
        # Generate fake traces
        n, e = session.run(
            [model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
            feed_dict={model.training: False, model.embeddings: embeddings}
        )
        n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

        # Convert to traces
        from models.process_gan import matrices_to_traces
        traces_fake = matrices_to_traces(e, n, data)

        # Compute rewards
        rewardR = compute_reward_batch(traces_real, reward_function)
        rewardF = compute_reward_batch(traces_fake, reward_function)

        # Train generator with RL
        ops_to_run = [optimizer.train_step_G, optimizer.train_step_V]
        feed_dict = {
            model.edges_labels: adj_real,
            model.nodes_labels: nodes_real,
            model.embeddings: embeddings,
            model.rewardR: rewardR,
            model.rewardF: rewardF,
            model.training: True,
            model.dropout_rate: 0.0,
            optimizer.la: la if epoch > 0 else 1.0
        }
    else:
        # Train generator without RL (pure GAN)
        ops_to_run = [optimizer.train_step_G]
        feed_dict = {
            model.edges_labels: adj_real,
            model.nodes_labels: nodes_real,
            model.embeddings: embeddings,
            model.training: True,
            model.dropout_rate: 0.0,
            optimizer.la: 1.0
        }

    _ = session.run(ops_to_run, feed_dict=feed_dict)

    # ─────────────────────────────────────────────────────────
    # COMPUTE LOSSES FOR MONITORING
    # ─────────────────────────────────────────────────────────

    loss_dict = optimizer.get_loss_dict()
    losses = session.run(loss_dict, feed_dict=feed_dict)

    return losses


if __name__ == '__main__':
    # Example usage
    import numpy as np
    from models.process_gan import ProcessGANModel

    print("=" * 60)
    print("ProcessGANOptimizer Test")
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

    print("\nCreating model and optimizer...")

    # Create model
    model = ProcessGANModel(
        max_activities=config['max_activities'],
        flow_types=config['flow_types'],
        activity_types=config['activity_types'],
        embedding_dim=config['embedding_dim'],
        decoder_units=config['decoder_units'],
        discriminator_units=config['discriminator_units']
    )

    # Create optimizer
    optimizer = ProcessGANOptimizer(
        model,
        learning_rate=1e-4,
        feature_matching=True,
        gradient_penalty_weight=10.0
    )

    print("✓ Model created")
    print("✓ Optimizer created")

    # Test with TensorFlow session
    print("\nTesting optimizer operations...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Dummy data
        batch_size = 4
        adj = np.zeros((batch_size, config['max_activities'], config['max_activities']), dtype=np.int64)
        nodes = np.zeros((batch_size, config['max_activities']), dtype=np.int64)
        z = model.sample_z(batch_size)
        rewards = np.ones((batch_size, 1), dtype=np.float32) * 0.8

        # Feed dict
        feed_dict = {
            model.edges_labels: adj,
            model.nodes_labels: nodes,
            model.embeddings: z,
            model.rewardR: rewards,
            model.rewardF: rewards,
            model.training: True,
            optimizer.la: 1.0
        }

        # Run training ops
        print("  Running discriminator update...")
        sess.run(optimizer.train_step_D, feed_dict=feed_dict)
        print("  ✓ Discriminator updated")

        print("  Running generator update...")
        sess.run(optimizer.train_step_G, feed_dict=feed_dict)
        print("  ✓ Generator updated")

        print("  Running value network update...")
        sess.run(optimizer.train_step_V, feed_dict=feed_dict)
        print("  ✓ Value network updated")

        # Get losses
        print("\n  Computing losses...")
        loss_dict = optimizer.get_loss_dict()
        losses = sess.run(loss_dict, feed_dict=feed_dict)

        print("  Loss values:")
        for key, value in losses.items():
            if isinstance(value, np.ndarray):
                value = value.mean()
            print(f"    {key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("ProcessGANOptimizer test completed successfully!")
    print("=" * 60)
