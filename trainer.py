"""
ProcessGAN Training Script

Modern training implementation using TensorFlow 2.x eager execution.
Compatible with TensorFlow 2.15+ and Python 3.10+.

Usage:
    # Train on event log (XES)
    python trainer.py --data data/helpdesk_parsed.xes --input-format xes
"""

import numpy as np
import tensorflow as tf
import argparse
import os
from datetime import datetime
from pathlib import Path

# Import ProcessGAN components
from utils.process_dataset import ProcessDataset
from utils.process_metrics import ProcessRewardFunction, ProcessMetrics
from models.process_gan import ProcessGAN, matrices_to_traces
from optimizers.process_optimizer import ProcessGANTrainer


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE OTTIMIZZATA PER CPU (3804 tracce, no GPU)
# ═════════════════════════════════════════════════════════════════════════════

config = {
    # Data
    'data_file': 'data/helpdesk_parsed.xes',
    'max_activities': 10,  # Ottimale per helpdesk (95° percentile)
    # Aumentato per dataset piccolo (validazione più robusta)
    'validation_split': 0.15,
    'test_split': 0.1,

    # Model architecture
    'z_dim': 32,  # 64→32: Meno parametri, meno overfitting, più veloce
    'decoder_units': (64, 128, 128),  # Dimezzato: ~75% parametri in meno
    'discriminator_units': (64, 64),  # Dimezzato: bilanciato con generator
    'mlp_units': 64,  # 128→64: Value network più leggero
    'dropout_rate': 0.2,  # 0.1→0.2: Più regolarizzazione (dataset piccolo)

    # Training
    'batch_size': 16,  # 32→16: Meno memoria, più step (migliore per CPU)
    'epochs': 60,  # 50→60: Compensa modello più piccolo
    'learning_rate': 2e-4,  # 3e-4→2e-4: Più stabile per modello piccolo
    'n_critic': 2,  # 3→2: ~33% più veloce, accettabile per CPU

    # GAN/RL mixing
    'lambda_start': 0.7,  # 0.8→0.7: RL attivo prima (30% da subito)
    'lambda_end': 0.4,    # 0.5→0.4: Più RL alla fine (60%)
    'lambda_decay_start': 3,  # 5→3: Decay prima (meno warm-up)

    # Hard constraints
    'enforce_start': True,   # Forza START come prima attività

    # Reward function
    'use_rl': True,
    'reward_weights': {
        'validity': 0.50,      # Aumentato da 0.40 → 0.50 (vincolo START/END)
        'fitness': 0.25,       # Ridotto da 0.30 → 0.25
        'conformance': 0.15,   # Ridotto da 0.20 → 0.15
        'diversity': 0.10      # Invariato
    },

    # 'reward_weights': {
    #    'validity': 0.20,    # Ridotto da 0.40 (START già garantito)
    #    'fitness': 0.40,     # Aumentato (focus su pattern)
    #    'conformance': 0.25,
    #    'diversity': 0.15    # Aumentato (più esplorazione)
    # },


    # Generation - RIDOTTO per velocità
    'n_samples_eval': 300,  # 500→300: Valutazione più veloce (~40% speedup)

    # Checkpointing
    'save_dir': 'checkpoints/process_gan',
    'save_every': 10,

    # Gumbel-Softmax - DECAY più veloce
    'temperature_start': 5.0,
    'temperature_end': 0.5,
    'temperature_decay': 0.93,  # 0.95→0.93: Discretizzazione più rapida

    # Logging
    'log_every': 1,
}


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE ARGUMENTS
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='ProcessGAN Training')

    # Data arguments
    parser.add_argument('--data', type=str, default=None,
                        help='Path to input data file (default: from config)')
    parser.add_argument('--input-format', type=str, default='xes',
                        choices=['xes', 'pnml'],
                        help='Input data format (xes or pnml)')
    parser.add_argument('--reference-model', type=str, default=None,
                        help='Path to reference Petri net (PNML) for fitness/conformance checking (optional)')

    # Model arguments
    parser.add_argument('--max-activities', type=int, default=None,
                        help='Maximum number of activities (default: from config)')
    parser.add_argument('--z-dim', type=int, default=None,
                        help='Latent dimension (default: from config)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (default: from config)')

    # Output arguments
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save checkpoints (default: from config)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for logs (default: results/logs)')

    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {msg}')


def merge_config_with_args(config, args):
    """
    Smart merge: CLI args override config only if explicitly provided

    Priority:
    1. CLI argument (if not None) - highest priority
    2. Config dict value - fallback

    Args:
        config: Base configuration dict
        args: Parsed command-line arguments

    Returns:
        Updated config dict with clear logging
    """
    log("Configuration merge:")

    # Data file
    if args.data is not None:
        config['data_file'] = args.data
        log(f"  data_file: {args.data} (from CLI)")
    else:
        log(f"  data_file: {config['data_file']} (from config)")

    # Model parameters
    if args.max_activities is not None:
        config['max_activities'] = args.max_activities
        log(f"  max_activities: {args.max_activities} (from CLI)")
    else:
        log(f"  max_activities: {config['max_activities']} (from config)")

    if args.z_dim is not None:
        config['z_dim'] = args.z_dim
        log(f"  z_dim: {args.z_dim} (from CLI)")
    else:
        log(f"  z_dim: {config['z_dim']} (from config)")

    # Training parameters
    if args.epochs is not None:
        config['epochs'] = args.epochs
        log(f"  epochs: {args.epochs} (from CLI)")
    else:
        log(f"  epochs: {config['epochs']} (from config)")

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        log(f"  batch_size: {args.batch_size} (from CLI)")
    else:
        log(f"  batch_size: {config['batch_size']} (from config)")

    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
        log(f"  learning_rate: {args.learning_rate} (from CLI)")
    else:
        log(f"  learning_rate: {config['learning_rate']} (from config)")

    # Output directories
    if args.save_dir is not None:
        config['save_dir'] = args.save_dir
        log(f"  save_dir: {args.save_dir} (from CLI)")
    else:
        log(f"  save_dir: {config['save_dir']} (from config)")

    if args.log_dir is not None:
        config['log_dir'] = args.log_dir
        log(f"  log_dir: {args.log_dir} (from CLI)")
    else:
        # Default log_dir if not in config
        if 'log_dir' not in config:
            config['log_dir'] = 'results/logs'
        log(f"  log_dir: {config['log_dir']} (default)")

    return config


def create_checkpoint_dir(save_dir):
    """Create checkpoint directory if it doesn't exist"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log(f'Checkpoint directory ready: {save_dir}')


def compute_lambda(epoch, config):
    """Compute lambda for mixing GAN and RL losses"""
    if epoch < config['lambda_decay_start']:
        return config['lambda_start']
    else:
        # Linear decay from lambda_start to lambda_end
        progress = (epoch - config['lambda_decay_start']) / \
            (config['epochs'] - config['lambda_decay_start'])
        return config['lambda_start'] + progress * (config['lambda_end'] - config['lambda_start'])


def compute_temperature(epoch, config):
    """Compute Gumbel-Softmax temperature"""
    temp = config['temperature_start'] * (config['temperature_decay'] ** epoch)
    return max(temp, config['temperature_end'])


def plot_training_history(history, save_dir):
    """
    Plot and save training history curves

    Args:
        history: Dict with training metrics over epochs
        save_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(save_dir, 'plots')
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    epochs = history['epoch']

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # ─────────────────────────────────────────────────────────
    # 1. Loss Curves (Training vs Validation)
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training vs Validation Losses',
                 fontsize=16, fontweight='bold')

    # Discriminator Loss
    axes[0, 0].plot(epochs, history['loss_D'], 'b-',
                    linewidth=2, label='Training')
    axes[0, 0].plot(epochs, history['val_loss_D'], 'r--',
                    linewidth=2, label='Validation')
    axes[0, 0].set_title('Discriminator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Generator Loss
    axes[0, 1].plot(epochs, history['loss_G'], 'b-',
                    linewidth=2, label='Training')
    axes[0, 1].plot(epochs, history['val_loss_G'], 'r--',
                    linewidth=2, label='Validation')
    axes[0, 1].set_title('Generator Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # RL Loss
    axes[1, 0].plot(epochs, history['loss_RL'], 'b-',
                    linewidth=2, label='Training')
    axes[1, 0].plot(epochs, history['val_loss_RL'], 'r--',
                    linewidth=2, label='Validation')
    axes[1, 0].set_title('Reinforcement Learning Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Value Network Loss
    axes[1, 1].plot(epochs, history['loss_V'], 'b-',
                    linewidth=2, label='Training')
    axes[1, 1].plot(epochs, history['val_loss_V'], 'r--',
                    linewidth=2, label='Validation')
    axes[1, 1].set_title('Value Network Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_validation_losses.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ─────────────────────────────────────────────────────────
    # 2. Reward Curves
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reward Metrics', fontsize=16, fontweight='bold')

    # Total Reward
    axes[0, 0].plot(epochs, history['eval_reward'], 'b-',
                    linewidth=2, label='Evaluation')
    axes[0, 0].plot(epochs, history['reward_mean'], 'r--',
                    linewidth=2, alpha=0.7, label='Training')
    axes[0, 0].set_title('Total Reward')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Validity
    axes[0, 1].plot(epochs, history['eval_validity'], 'g-', linewidth=2)
    axes[0, 1].set_title('Validity Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Fitness
    axes[1, 0].plot(epochs, history['eval_fitness'], 'orange', linewidth=2)
    axes[1, 0].set_title('Fitness Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Fitness')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # Diversity
    axes[1, 1].plot(epochs, history['eval_diversity'], 'purple', linewidth=2)
    axes[1, 1].set_title('Diversity Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Diversity')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rewards.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ─────────────────────────────────────────────────────────
    # 3. Quality Metrics
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Generation Quality Metrics', fontsize=16, fontweight='bold')

    # Valid Rate
    axes[0].plot(epochs, [r * 100 for r in history['valid_rate']],
                 'b-', linewidth=2)
    axes[0].set_title('Valid Traces (%)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Percentage')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 100])

    # Unique Rate
    axes[1].plot(epochs, [r * 100 for r in history['unique_rate']],
                 'g-', linewidth=2)
    axes[1].set_title('Unique Traces (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Percentage')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])

    # Novel Rate
    axes[2].plot(epochs, [r * 100 for r in history['novel_rate']],
                 'r-', linewidth=2)
    axes[2].set_title('Novel Traces (%)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Percentage')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'quality_metrics.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ─────────────────────────────────────────────────────────
    # 4. Combined Overview
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Overview', fontsize=16, fontweight='bold')

    # D + G Loss
    axes[0, 0].plot(epochs, history['loss_D'], 'b-',
                    linewidth=2, label='Discriminator')
    axes[0, 0].plot(epochs, history['loss_G'], 'r-',
                    linewidth=2, label='Generator')
    axes[0, 0].set_title('GAN Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Gradient Penalty
    axes[0, 1].plot(epochs, history['grad_penalty'], 'orange', linewidth=2)
    axes[0, 1].set_title('Gradient Penalty (WGAN-GP)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Penalty')
    axes[0, 1].grid(True, alpha=0.3)

    # Reward Components
    axes[0, 2].plot(epochs, history['eval_validity'],
                    label='Validity', linewidth=2)
    axes[0, 2].plot(epochs, history['eval_fitness'],
                    label='Fitness', linewidth=2)
    axes[0, 2].plot(epochs, history['eval_conformance'],
                    label='Conformance', linewidth=2)
    axes[0, 2].plot(epochs, history['eval_diversity'],
                    label='Diversity', linewidth=2)
    axes[0, 2].set_title('Reward Components')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    axes[0, 2].set_ylim([0, 1])

    # RL Losses
    axes[1, 0].plot(epochs, history['loss_RL'], 'g-',
                    linewidth=2, label='RL Loss')
    axes[1, 0].plot(epochs, history['loss_V'], 'm-',
                    linewidth=2, label='V Loss')
    axes[1, 0].set_title('RL Losses')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Total Reward
    axes[1, 1].plot(epochs, history['eval_reward'], 'b-',
                    linewidth=2, marker='o', markersize=4)
    axes[1, 1].set_title('Total Reward (Evaluation)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].grid(True, alpha=0.3)

    # Quality Summary
    axes[1, 2].plot(
        epochs, [r * 100 for r in history['valid_rate']], label='Valid', linewidth=2)
    axes[1, 2].plot(
        epochs, [r * 100 for r in history['unique_rate']], label='Unique', linewidth=2)
    axes[1, 2].plot(
        epochs, [r * 100 for r in history['novel_rate']], label='Novel', linewidth=2)
    axes[1, 2].set_title('Quality Rates (%)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Percentage')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    axes[1, 2].set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'overview.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    log(f"📊 Training plots saved in: {plots_dir}")
    log(f"  - training_validation_losses.png")
    log(f"  - rewards.png")
    log(f"  - quality_metrics.png")
    log(f"  - overview.png")


def generate_samples(model, dataset, n_samples, temperature=0.5):
    """Generate samples from model"""
    z = model.sample_z(n_samples)

    edges, nodes = model.generator(z, training=False, temperature=temperature)

    # Convert to numpy and argmax
    edges_np = edges.numpy()
    nodes_np = nodes.numpy()

    nodes_indices = np.argmax(nodes_np, axis=-1)
    edges_indices = np.argmax(edges_np, axis=-1)

    traces = matrices_to_traces(edges_indices, nodes_np, dataset)

    return traces


def evaluate_model(model, dataset, reward_function, config, epoch):
    """Evaluate model by generating samples and computing metrics"""
    log('Generating samples for evaluation...')

    # Generate samples
    temperature = compute_temperature(epoch, config)
    traces = generate_samples(
        model,
        dataset,
        n_samples=config['n_samples_eval'],
        temperature=temperature
    )

    # Compute reward metrics
    metrics = reward_function.evaluate_batch(traces)

    # Compute process metrics
    valid_traces = ProcessMetrics.valid_traces(traces)
    unique_traces = ProcessMetrics.unique_traces(traces)
    novel_traces = ProcessMetrics.novel_traces(traces, dataset.data)

    metrics['valid_rate'] = len(valid_traces) / len(traces)
    metrics['unique_rate'] = len(unique_traces) / len(traces)
    metrics['novel_rate'] = len(novel_traces) / len(traces)

    # Sample traces for inspection
    metrics['sample_traces'] = traces[:5]

    return metrics


def evaluate_validation_losses(model, dataset, trainer, reward_function, config, epoch):
    """Evaluate model losses on validation set"""
    log('Computing validation losses...')

    # Reset validation counter
    dataset.validation_counter = 0

    # Compute dynamic parameters
    lambda_mix = compute_lambda(epoch, config)
    temperature = compute_temperature(epoch, config)

    # Reset metrics for validation
    trainer.reset_metrics()

    # Compute validation losses
    val_steps = dataset.validation_count // config['batch_size']
    if val_steps == 0:
        val_steps = 1  # At least one step

    for step in range(val_steps):
        # Get validation batch
        _, adj_batch, nodes_batch, _ = dataset.next_validation_batch(
            config['batch_size'])

        # Convert adjacency matrices to 4D tensors (add flow_types dimension)
        # Dataset provides: (batch, max_act, max_act) with flow type indices
        # Discriminator expects: (batch, max_act, max_act, flow_types) as one-hot
        adj_batch_4d = tf.one_hot(
            adj_batch, depth=dataset.flow_num_types, dtype=tf.float32)

        # Convert nodes to one-hot if needed
        if len(nodes_batch.shape) == 2:  # (batch, max_act) with activity indices
            nodes_batch_onehot = tf.one_hot(
                nodes_batch, depth=dataset.activity_num_types, dtype=tf.float32)
        else:  # Already one-hot
            nodes_batch_onehot = nodes_batch

        # Forward pass without training (no gradients)
        with tf.GradientTape() as tape:
            # Generate fake samples
            z = model.sample_z(config['batch_size'])
            fake_adj, fake_nodes = model.generator(
                z, training=False, temperature=temperature)

            # Discriminator forward pass
            real_scores, real_features = model.discriminator(
                adj_batch_4d, nodes_batch_onehot, training=False)
            fake_scores, fake_features = model.discriminator(
                fake_adj, fake_nodes, training=False)

            # Compute discriminator loss (WGAN)
            loss_D = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)

            # Compute generator loss
            loss_G = -tf.reduce_mean(fake_scores)

            # Compute gradient penalty
            grad_penalty = trainer._gradient_penalty(
                adj_batch_4d, nodes_batch_onehot, fake_adj, fake_nodes, training=False)

            # Add gradient penalty to discriminator loss
            loss_D += 10.0 * grad_penalty

            # RL losses (if enabled)
            if config['use_rl']:
                # Convert to traces for reward computation
                fake_nodes_onehot = np.eye(dataset.activity_num_types)[
                    np.argmax(fake_nodes.numpy(), axis=-1)]
                fake_traces = matrices_to_traces(
                    np.argmax(fake_adj.numpy(), axis=-1), fake_nodes_onehot, dataset)

                # Compute rewards
                rewards_fake = reward_function.compute_reward(fake_traces)
                rewards_fake = tf.constant(rewards_fake, dtype=tf.float32)

                # Value network prediction
                value_pred = model.value_network(
                    fake_adj, fake_nodes, training=False)
                loss_V = tf.reduce_mean(tf.square(value_pred - rewards_fake))

                # RL loss (policy gradient)
                # Use tf.math.log for TF2 compatibility
                loss_RL = -tf.reduce_mean(
                    rewards_fake *
                    tf.math.log(tf.reduce_sum(fake_nodes, axis=-1) + 1e-8)
                )
            else:
                loss_V = tf.constant(0.0)
                loss_RL = tf.constant(0.0)
                rewards_fake = tf.zeros((config['batch_size'], 1))

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


def print_epoch_summary(epoch, epochs, metrics, losses, val_losses, config):
    """Print summary of epoch results"""
    print("\n" + "=" * 80)
    print(f"EPOCH {epoch+1}/{epochs}")
    print("=" * 80)

    print("\nTraining Losses:")
    print(f"  D Loss:       {losses['loss_D']:.4f}")
    print(f"  G Loss:       {losses['loss_G']:.4f}")
    if config['use_rl']:
        print(f"  RL Loss:      {losses['loss_RL']:.4f}")
        print(f"  V Loss:       {losses['loss_V']:.4f}")
        print(f"  Reward Mean:  {losses['reward_mean']:.4f}")
    print(f"  Grad Penalty: {losses['grad_penalty']:.4f}")

    print("\nValidation Losses:")
    print(f"  D Loss:       {val_losses['loss_D']:.4f}")
    print(f"  G Loss:       {val_losses['loss_G']:.4f}")
    if config['use_rl']:
        print(f"  RL Loss:      {val_losses['loss_RL']:.4f}")
        print(f"  V Loss:       {val_losses['loss_V']:.4f}")
        print(f"  Reward Mean:  {val_losses['reward_mean']:.4f}")
    print(f"  Grad Penalty: {val_losses['grad_penalty']:.4f}")

    print("\nGeneration Metrics:")
    print(
        f"  Reward:       {metrics['reward_mean']:.3f} ± {metrics['reward_std']:.3f}")
    print(
        f"  Validity:     {metrics['validity_mean']:.3f} ± {metrics['validity_std']:.3f}")
    print(
        f"  Fitness:      {metrics['fitness_mean']:.3f} ± {metrics['fitness_std']:.3f}")
    print(
        f"  Conformance:  {metrics['conformance_mean']:.3f} ± {metrics['conformance_std']:.3f}")
    print(
        f"  Diversity:    {metrics['diversity_mean']:.3f} ± {metrics['diversity_std']:.3f}")

    print("\nQuality Metrics:")
    print(f"  Valid Rate:   {metrics['valid_rate']:.1%}")
    print(f"  Unique Rate:  {metrics['unique_rate']:.1%}")
    print(f"  Novel Rate:   {metrics['novel_rate']:.1%}")
    print(
        f"  High Quality: {metrics['high_quality_rate']:.1%} (reward >= 0.7)")

    print("\nSample Generated Traces:")
    for i, trace in enumerate(metrics['sample_traces'][:3]):
        print(f"  {i+1}. {' → '.join(trace)}")

    print("=" * 80 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Main training function"""

    log("=" * 80)
    log("ProcessGAN Training")
    log("=" * 80)

    # Check TensorFlow version
    log(f"TensorFlow version: {tf.__version__}")
    log(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    # ─────────────────────────────────────────────────────────
    # 1. PARSE ARGUMENTS AND MERGE WITH CONFIG
    # ─────────────────────────────────────────────────────────

    # Parse command-line arguments
    args = parse_args()

    # Smart merge: CLI overrides config only if explicitly provided
    config_merged = merge_config_with_args(config.copy(), args)

    # ─────────────────────────────────────────────────────────
    # 2. LOAD DATASET
    # ─────────────────────────────────────────────────────────

    log(f"Input format: {args.input_format}")
    log("Loading dataset...")
    data = ProcessDataset(max_activities=config_merged['max_activities'])

    # Load based on format
    if os.path.exists(config_merged['data_file']):
        if args.input_format == 'xes':
            data.load_from_xes(
                config_merged['data_file'],
                validation=config_merged['validation_split'],
                test=config_merged['test_split']
            )
        else:
            log(f"Unsupported format: {args.input_format}. Use 'xes' or 'pnml'", level='ERROR')
            return
    else:
        log(f"Data file not found: {config_merged['data_file']}", level='ERROR')
        return

    stats = data.get_stats()
    log(f"Dataset loaded: {stats['total_traces']} traces")
    log(f"  Train: {stats['train_size']}, Val: {stats['val_size']}, Test: {stats['test_size']}")
    log(f"  Activities: {stats['num_activities']}, Flow types: {stats['num_flow_types']}")

    # ─────────────────────────────────────────────────────────
    # 3. CREATE REWARD FUNCTION
    # ─────────────────────────────────────────────────────────

    log("Creating reward function...")

    # Try to load reference Petri net model for fitness/conformance
    reference_model = None
    reference_model_path = None

    # Priority 1: Explicit --reference-model argument
    if args.reference_model and os.path.exists(args.reference_model):
        reference_model_path = args.reference_model
        log(f"Using explicit reference model: {reference_model_path}")

    # Priority 2: If input is PNML, use it as reference model
    elif args.input_format == 'pnml' and os.path.exists(config_merged['data_file']):
        reference_model_path = config_merged['data_file']
        log(f"Using input PNML as reference model: {reference_model_path}")

    # Priority 3: Try to find companion PNML file for XES
    elif args.input_format == 'xes':
        # Try common naming patterns:
        # helpdesk_parsed.xes → Helpdesk_parsed_net.pnml
        # helpdesk_parsed.xes → helpdesk_parsed.pnml
        base_path = os.path.splitext(config_merged['data_file'])[0]

        candidate_paths = [
            base_path.replace('_parsed', '_parsed_net') +
            '.pnml',  # _parsed → _parsed_net
            base_path.capitalize().replace('_parsed', '_parsed_net') +
            '.pnml',  # Capitalize first letter
            base_path + '_net.pnml',  # Add _net suffix
            base_path + '.pnml',  # Simple replacement
        ]

        for candidate in candidate_paths:
            if os.path.exists(candidate):
                reference_model_path = candidate
                log(f"✓ Found companion Petri net: {reference_model_path}")
                break

    # Load reference model if path found
    if reference_model_path:
        log("Loading Petri net as reference model for fitness/conformance...")
        reference_model = ProcessRewardFunction.load_petri_net_model(
            reference_model_path)
        if reference_model:
            log("Reference model loaded")
        else:
            log("⚠ Failed to load reference model - using simplified metrics")
    else:
        log("⚠ No reference model available - using simplified fitness/conformance metrics")

    # Auto-detect START/END activities from dataset
    start_activity = None
    end_activity = None

    if data.data:
        from collections import Counter
        first_activities = [t[0] for t in data.data if len(t) > 0]
        last_activities = [t[-1] for t in data.data if len(t) > 0]

        if first_activities:
            start_activity = Counter(first_activities).most_common(1)[0][0]
        if last_activities:
            end_activity = Counter(last_activities).most_common(1)[0][0]

    log(f"Auto-detected START activity: {start_activity}")
    log(f"Auto-detected END activity: {end_activity}")

    reward_function = ProcessRewardFunction(
        reference_model=reference_model,
        training_traces=data.data,
        weights=config_merged['reward_weights'],
        start_activity=start_activity,  # NEW: Pass auto-detected START
        end_activity=end_activity        # NEW: Pass auto-detected END
    )
    log(f"Reward weights: {config_merged['reward_weights']}")

    # ─────────────────────────────────────────────────────────
    # 4. BUILD MODEL
    # ─────────────────────────────────────────────────────────

    log("Building ProcessGAN model...")
    model = ProcessGAN(
        max_activities=data.max_activities,
        flow_types=data.flow_num_types,
        activity_types=data.activity_num_types,
        embedding_dim=config_merged['z_dim'],
        decoder_units=config_merged['decoder_units'],
        discriminator_units=config_merged['discriminator_units'],
        mlp_units=config_merged['mlp_units'],
        dropout_rate=config_merged['dropout_rate'],
        enforce_start=config_merged['enforce_start']
    )
    log("Model built successfully")

    # ─────────────────────────────────────────────────────────
    # 5. CREATE TRAINER
    # ─────────────────────────────────────────────────────────

    log("Creating trainer...")
    trainer = ProcessGANTrainer(
        model,
        dataset=data,  # Pass dataset for trace decoding
        learning_rate=config_merged['learning_rate'],
        gradient_penalty_weight=10.0,
        lambda_adv=0.6,
        lambda_reward=0.4
    )
    log("Trainer created")

    # Build models by calling them with sample input to create weights
    sample_z = tf.random.normal((1, config_merged['z_dim']))
    sample_adj, sample_nodes = model.generator(sample_z, training=False)
    _ = model.discriminator(sample_adj, sample_nodes, training=False)
    _ = model.value_network(sample_adj, sample_nodes, training=False)

    # Count parameters
    total_g = sum([np.prod(v.shape)
                  for v in model.generator.trainable_variables])
    total_d = sum([np.prod(v.shape)
                  for v in model.discriminator.trainable_variables])
    total_v = sum([np.prod(v.shape)
                  for v in model.value_network.trainable_variables])
    log(f"Total trainable parameters: {total_g + total_d + total_v:,}")

    # Create checkpoint directory
    create_checkpoint_dir(config_merged['save_dir'])

    # ─────────────────────────────────────────────────────────
    # 6. TRAINING LOOP
    # ─────────────────────────────────────────────────────────

    log("\n" + "=" * 80)
    log("Starting training...")
    log("=" * 80 + "\n")

    steps_per_epoch = data.train_count // config_merged['batch_size']

    # Best model tracking
    best_reward = -float('inf')
    best_epoch = 0

    # Training history for plotting
    history = {
        'epoch': [],
        'loss_D': [],
        'loss_G': [],
        'loss_RL': [],
        'loss_V': [],
        'grad_penalty': [],
        'reward_mean': [],
        'val_loss_D': [],
        'val_loss_G': [],
        'val_loss_RL': [],
        'val_loss_V': [],
        'val_grad_penalty': [],
        'val_reward_mean': [],
        'eval_reward': [],
        'eval_validity': [],
        'eval_fitness': [],
        'eval_conformance': [],
        'eval_diversity': [],
        'valid_rate': [],
        'unique_rate': [],
        'novel_rate': []
    }

    for epoch in range(config_merged['epochs']):

        # Compute dynamic parameters
        lambda_mix = compute_lambda(epoch, config_merged)
        temperature = compute_temperature(epoch, config_merged)

        # Reset metrics
        trainer.reset_metrics()

        # Training epoch
        for step in range(steps_per_epoch):
            # Get batch (ignore traces_batch as we only need matrix representation)
            _, adj_batch, nodes_batch, _ = data.next_train_batch(
                config_merged['batch_size'])

            # Training step (losses are accumulated in trainer.metrics)
            _ = trainer.train_step(
                real_adj=adj_batch,
                real_nodes=nodes_batch,
                batch_size=config_merged['batch_size'],
                n_critic=config_merged['n_critic'],
                reward_function=reward_function,
                lambda_mix=lambda_mix,
                temperature=temperature
            )

        # Get average losses over all steps
        avg_losses = trainer.get_metrics()

        # Compute validation losses
        val_losses = evaluate_validation_losses(
            model, data, trainer, reward_function, config_merged, epoch)


        # Evaluation
        if (epoch + 1) % config_merged['log_every'] == 0:
            metrics = evaluate_model(
                model, data, reward_function, config_merged, epoch)
            print_epoch_summary(
                epoch, config_merged['epochs'], metrics, avg_losses, val_losses, config_merged)
        else:
            # Still compute metrics for best model tracking
            metrics = evaluate_model(
                model, data, reward_function, config_merged, epoch)

        # Save metrics to history
        history['epoch'].append(epoch + 1)
        history['loss_D'].append(avg_losses['loss_D'])
        history['loss_G'].append(avg_losses['loss_G'])
        history['loss_RL'].append(avg_losses['loss_RL'])
        history['loss_V'].append(avg_losses['loss_V'])
        history['grad_penalty'].append(avg_losses['grad_penalty'])
        history['reward_mean'].append(avg_losses['reward_mean'])
        history['val_loss_D'].append(val_losses['loss_D'])
        history['val_loss_G'].append(val_losses['loss_G'])
        history['val_loss_RL'].append(val_losses['loss_RL'])
        history['val_loss_V'].append(val_losses['loss_V'])
        history['val_grad_penalty'].append(val_losses['grad_penalty'])
        history['val_reward_mean'].append(val_losses['reward_mean'])
        history['eval_reward'].append(metrics['reward_mean'])
        history['eval_validity'].append(metrics['validity_mean'])
        history['eval_fitness'].append(metrics['fitness_mean'])
        history['eval_conformance'].append(metrics['conformance_mean'])
        history['eval_diversity'].append(metrics['diversity_mean'])
        history['valid_rate'].append(metrics['valid_rate'])
        history['unique_rate'].append(metrics['unique_rate'])
        history['novel_rate'].append(metrics['novel_rate'])

        # Save checkpoint only if this is the best model so far
        current_reward = metrics['reward_mean']
        if current_reward > best_reward:
            best_reward = current_reward
            best_epoch = epoch + 1

            checkpoint_path = os.path.join(config_merged['save_dir'], 'best')
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

            model.generator.save_weights(
                os.path.join(checkpoint_path, 'generator'))
            model.discriminator.save_weights(
                os.path.join(checkpoint_path, 'discriminator'))
            model.value_network.save_weights(
                os.path.join(checkpoint_path, 'value_network'))

            log(
                f"✨ New best model saved! Epoch {best_epoch}, Reward: {best_reward:.4f}")

            # Save epoch info
            with open(os.path.join(checkpoint_path, 'info.txt'), 'w') as f:
                f.write(f"Best Epoch: {best_epoch}\n")
                f.write(f"Best Reward: {best_reward:.4f}\n")
                f.write(f"Valid Rate: {metrics['valid_rate']:.2%}\n")
                f.write(f"Novel Rate: {metrics['novel_rate']:.2%}\n")

    # ─────────────────────────────────────────────────────────
    # 7. PLOT TRAINING CURVES
    # ─────────────────────────────────────────────────────────

    log("\n" + "=" * 80)
    log("Generating training plots...")
    log("=" * 80 + "\n")

    plot_training_history(history, config_merged['save_dir'])

    # ─────────────────────────────────────────────────────────
    # 8. FINAL EVALUATION WITH BEST MODEL
    # ─────────────────────────────────────────────────────────

    log("\n" + "=" * 80)
    log("Loading best model for final evaluation...")
    log("=" * 80 + "\n")

    log(f"Best model from epoch {best_epoch} with reward {best_reward:.4f}")

    # Load best model weights
    best_path = os.path.join(config_merged['save_dir'], 'best')
    model.generator.load_weights(os.path.join(best_path, 'generator'))
    model.discriminator.load_weights(os.path.join(best_path, 'discriminator'))
    model.value_network.load_weights(os.path.join(best_path, 'value_network'))
    log("Best model loaded successfully")

    # Generate larger sample
    log(f"Generating {config_merged['n_samples_eval']} final samples...")
    final_traces = generate_samples(
        model,
        data,
        n_samples=config_merged['n_samples_eval'],
        temperature=0.5
    )

    # Compute final metrics
    final_metrics = reward_function.evaluate_batch(final_traces)
    valid_traces = ProcessMetrics.valid_traces(final_traces)
    unique_traces = ProcessMetrics.unique_traces(final_traces)
    novel_traces = ProcessMetrics.novel_traces(final_traces, data.data)

    print("\n" + "=" * 80)
    print("FINAL EVALUATION - BEST MODEL")
    print("=" * 80)

    print(f"\nBest Checkpoint: Epoch {best_epoch}")
    print(f"Best Reward: {best_reward:.4f}")

    print("\nFinal Generation Quality:")
    print(f"  Total generated:      {len(final_traces)}")
    print(
        f"  Valid traces:         {len(valid_traces)} ({len(valid_traces)/len(final_traces):.1%})")
    print(
        f"  Unique traces:        {len(unique_traces)} ({len(unique_traces)/len(final_traces):.1%})")
    print(
        f"  Novel traces:         {len(novel_traces)} ({len(novel_traces)/len(final_traces):.1%})")
    print(f"  High quality (≥0.7):  {final_metrics['high_quality_rate']:.1%}")

    print("\nReward Breakdown:")
    print(f"  Validity:     {final_metrics['validity_mean']:.3f}")
    print(f"  Fitness:      {final_metrics['fitness_mean']:.3f}")
    print(f"  Conformance:  {final_metrics['conformance_mean']:.3f}")
    print(f"  Diversity:    {final_metrics['diversity_mean']:.3f}")
    print(f"  Total Reward: {final_metrics['reward_mean']:.3f}")

    print("\nSample Best Traces:")
    # Sort by reward and show top 5
    rewards = reward_function.compute_reward(final_traces).flatten()
    top_indices = np.argsort(rewards)[-5:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. [{rewards[idx]:.3f}] {' → '.join(final_traces[idx])}")

    print("\n" + "=" * 80)
    log(f"Best model saved in: {best_path}")
    log("Training completed successfully!")
    log("=" * 80)


if __name__ == '__main__':
    # Enable mixed precision for faster training (optional)
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    main()
