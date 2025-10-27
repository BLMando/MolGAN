"""
Model Evaluator
Sample generation, metric computation, validation
"""

import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any

from models.process_gan import matrices_to_traces
from metrics.static_metrics import ProcessMetrics


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {msg}')


class ModelEvaluator:
    """
    Evaluate ProcessGAN model

    Handles:
    - Sample generation
    - Metric computation (reward, validity, fitness, etc.)
    - Validation loss computation
    - Epoch summary printing
    """

    def __init__(self, model, dataset, reward_function, config):
        """
        Initialize model evaluator

        Args:
            model: ProcessGAN model
            dataset: ProcessDataset instance
            reward_function: ProcessRewardFunction instance
            config: Configuration dictionary
        """
        self.model = model
        self.dataset = dataset
        self.reward_function = reward_function
        self.config = config

    def generate_samples(self, n_samples: int, temperature: float = 0.5):
        """
        Generate synthetic traces from model

        Args:
            n_samples: Number of traces to generate
            temperature: Gumbel-Softmax temperature

        Returns:
            List of generated traces (as activity sequences)
        """
        z = self.model.sample_z(n_samples)

        edges, nodes = self.model.generator(
            z, training=False, temperature=temperature)

        # Convert to numpy and argmax
        edges_np = edges.numpy()
        nodes_np = nodes.numpy()

        nodes_indices = np.argmax(nodes_np, axis=-1)
        edges_indices = np.argmax(edges_np, axis=-1)

        traces = matrices_to_traces(edges_indices, nodes_np, self.dataset)

        return traces

    def evaluate_samples(self, n_samples: int, temperature: float = 0.5):
        """
        Generate and evaluate synthetic traces

        Args:
            n_samples: Number of traces to generate
            temperature: Gumbel-Softmax temperature

        Returns:
            Dictionary of metrics
        """
        log('Generating and evaluating samples...')

        # Generate traces
        traces = self.generate_samples(n_samples, temperature)

        # Compute reward metrics
        metrics = self.reward_function.evaluate_batch(traces)

        # Compute process metrics
        valid_traces = ProcessMetrics.valid_traces(traces)
        unique_traces = ProcessMetrics.unique_traces(traces)
        novel_traces = ProcessMetrics.novel_traces(traces, self.dataset.data)

        metrics['valid_rate'] = len(valid_traces) / len(traces)
        metrics['unique_rate'] = len(unique_traces) / len(traces)
        metrics['novel_rate'] = len(novel_traces) / len(traces)

        # Sample traces for inspection
        metrics['sample_traces'] = traces[:5]

        return metrics

    def compute_validation_losses(self, trainer, temperature: float):
        """
        Compute validation losses on validation set

        Args:
            trainer: ProcessGANTrainer instance
            temperature: Current Gumbel-Softmax temperature

        Returns:
            Dictionary of validation losses
        """
        log('Computing validation losses...')

        # Reset validation counter
        self.dataset.validation_counter = 0

        # Reset metrics for validation
        trainer.reset_metrics()

        # Compute validation losses
        val_steps = self.dataset.validation_count // self.config['batch_size']
        if val_steps == 0:
            val_steps = 1  # At least one step

        for step in range(val_steps):
            # Get validation batch
            _, adj_batch, nodes_batch, _ = self.dataset.next_validation_batch(
                self.config['batch_size'])

            # Convert adjacency matrices to 4D tensors (add flow_types dimension)
            adj_batch_4d = tf.one_hot(
                adj_batch, depth=self.dataset.flow_num_types, dtype=tf.float32)

            # Convert nodes to one-hot if needed
            if len(nodes_batch.shape) == 2:  # (batch, max_act) with activity indices
                nodes_batch_onehot = tf.one_hot(
                    nodes_batch, depth=self.dataset.activity_num_types, dtype=tf.float32)
            else:  # Already one-hot
                nodes_batch_onehot = nodes_batch

            # Forward pass without training (no gradients)
            with tf.GradientTape() as tape:
                # Generate fake samples
                z = self.model.sample_z(self.config['batch_size'])
                fake_adj, fake_nodes = self.model.generator(
                    z, training=False, temperature=temperature)

                # Discriminator forward pass
                real_scores, real_features = self.model.discriminator(
                    adj_batch_4d, nodes_batch_onehot, training=False)
                fake_scores, fake_features = self.model.discriminator(
                    fake_adj, fake_nodes, training=False)

                # Compute discriminator loss (WGAN)
                loss_D = tf.reduce_mean(fake_scores) - \
                    tf.reduce_mean(real_scores)

                # Compute generator loss
                loss_G = -tf.reduce_mean(fake_scores)

                # Compute gradient penalty
                grad_penalty = trainer._gradient_penalty(
                    adj_batch_4d, nodes_batch_onehot, fake_adj, fake_nodes, training=False)

                # Add gradient penalty to discriminator loss
                loss_D += 10.0 * grad_penalty

                # RL losses (if enabled)
                if self.config['use_rl']:
                    # Convert to traces for reward computation
                    fake_nodes_onehot = np.eye(self.dataset.activity_num_types)[
                        np.argmax(fake_nodes.numpy(), axis=-1)]
                    fake_traces = matrices_to_traces(
                        np.argmax(fake_adj.numpy(), axis=-1), fake_nodes_onehot, self.dataset)

                    # Compute rewards
                    rewards_fake = self.reward_function.compute_reward(
                        fake_traces)
                    rewards_fake = tf.constant(rewards_fake, dtype=tf.float32)

                    # Value network prediction
                    value_pred = self.model.value_network(
                        fake_adj, fake_nodes, training=False)
                    loss_V = tf.reduce_mean(
                        tf.square(value_pred - rewards_fake))

                    # RL loss (policy gradient)
                    loss_RL = -tf.reduce_mean(
                        rewards_fake *
                        tf.math.log(tf.reduce_sum(fake_nodes, axis=-1) + 1e-8)
                    )
                else:
                    loss_V = tf.constant(0.0)
                    loss_RL = tf.constant(0.0)
                    rewards_fake = tf.zeros((self.config['batch_size'], 1))

            # Update metrics
            trainer.metrics['loss_D'].update_state(loss_D)
            trainer.metrics['loss_G'].update_state(loss_G)
            trainer.metrics['loss_V'].update_state(loss_V)
            trainer.metrics['loss_RL'].update_state(loss_RL)
            trainer.metrics['grad_penalty'].update_state(grad_penalty)
            trainer.metrics['reward_mean'].update_state(
                tf.reduce_mean(rewards_fake))

        # Get validation metrics
        val_metrics = trainer.get_metrics()

        return val_metrics

    def print_epoch_summary(self, epoch: int, total_epochs: int,
                            eval_metrics: Dict, train_losses: Dict,
                            val_losses: Dict, use_rl: bool):
        """
        Print formatted epoch summary

        Args:
            epoch: Current epoch number (0-indexed)
            total_epochs: Total number of epochs
            eval_metrics: Evaluation metrics dictionary
            train_losses: Training losses dictionary
            val_losses: Validation losses dictionary
            use_rl: Whether RL is enabled
        """
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch+1}/{total_epochs}")
        print("=" * 80)

        print("\nTraining Losses:")
        print(f"  D Loss:       {train_losses['loss_D']:.4f}")
        print(f"  G Loss:       {train_losses['loss_G']:.4f}")
        if use_rl:
            print(f"  RL Loss:      {train_losses['loss_RL']:.4f}")
            print(f"  V Loss:       {train_losses['loss_V']:.4f}")
            print(f"  Reward Mean:  {train_losses['reward_mean']:.4f}")
        print(f"  Grad Penalty: {train_losses['grad_penalty']:.4f}")

        print("\nValidation Losses:")
        print(f"  D Loss:       {val_losses['loss_D']:.4f}")
        print(f"  G Loss:       {val_losses['loss_G']:.4f}")
        if use_rl:
            print(f"  RL Loss:      {val_losses['loss_RL']:.4f}")
            print(f"  V Loss:       {val_losses['loss_V']:.4f}")
            print(f"  Reward Mean:  {val_losses['reward_mean']:.4f}")
        print(f"  Grad Penalty: {val_losses['grad_penalty']:.4f}")

        print("\nGeneration Metrics:")
        print(
            f"  Reward:       {eval_metrics['reward_mean']:.3f} ± {eval_metrics['reward_std']:.3f}")
        print(
            f"  Validity:     {eval_metrics['validity_mean']:.3f} ± {eval_metrics['validity_std']:.3f}")
        print(
            f"  Fitness:      {eval_metrics['fitness_mean']:.3f} ± {eval_metrics['fitness_std']:.3f}")
        print(
            f"  Conformance:  {eval_metrics['conformance_mean']:.3f} ± {eval_metrics['conformance_std']:.3f}")
        print(
            f"  Diversity:    {eval_metrics['diversity_mean']:.3f} ± {eval_metrics['diversity_std']:.3f}")

        print("\nQuality Metrics:")
        print(f"  Valid Rate:   {eval_metrics['valid_rate']:.1%}")
        print(f"  Unique Rate:  {eval_metrics['unique_rate']:.1%}")
        print(f"  Novel Rate:   {eval_metrics['novel_rate']:.1%}")
        print(
            f"  High Quality: {eval_metrics['high_quality_rate']:.1%} (reward >= 0.7)")

        print("\nSample Generated Traces:")
        for i, trace in enumerate(eval_metrics['sample_traces'][:3]):
            print(f"  {i+1}. {' → '.join(trace)}")

        print("=" * 80 + "\n")
