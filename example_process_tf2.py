"""
ProcessGAN Training Script (TensorFlow 2.x)

Modern training implementation using eager execution.
Compatible with TensorFlow 2.15+ and Python 3.10+.

Usage:
    python example_process_tf2.py
"""

import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from pathlib import Path

# Import ProcessGAN components (TF2 versions)
from utils.process_dataset import ProcessDataset
from utils.process_metrics import ProcessRewardFunction, ProcessMetrics
from models.process_gan_tf2 import ProcessGAN, matrices_to_traces
from optimizers.process_optimizer_tf2 import ProcessGANTrainer


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

config = {
    # Data
    'data_file': 'data/sample_event_log.csv',
    'max_activities': 15,
    'validation_split': 0.1,
    'test_split': 0.1,

    # Model architecture
    'z_dim': 16,
    'decoder_units': (128, 256, 512),
    'discriminator_units': (128, 64),
    'mlp_units': 128,
    'dropout_rate': 0.0,

    # Training
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'n_critic': 5,  # Discriminator steps per generator step

    # GAN/RL mixing
    'lambda_start': 1.0,  # Start with pure GAN
    'lambda_end': 0.6,    # End with 60% GAN, 40% RL
    'lambda_decay_start': 10,  # Start decaying lambda after epoch 10

    # Reward function
    'use_rl': True,
    'reward_weights': {
        'validity': 0.30,
        'fitness': 0.25,
        'conformance': 0.25,
        'diversity': 0.20
    },

    # Generation
    'n_samples_eval': 100,  # Number of samples to generate for evaluation

    # Checkpointing
    'save_dir': 'checkpoints/process_gan_tf2',
    'save_every': 10,  # Save model every N epochs

    # Gumbel-Softmax
    'temperature_start': 5.0,
    'temperature_end': 0.5,
    'temperature_decay': 0.95,

    # Logging
    'log_every': 1,  # Log every N epochs
}


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {msg}')


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
        progress = (epoch - config['lambda_decay_start']) / (config['epochs'] - config['lambda_decay_start'])
        return config['lambda_start'] + progress * (config['lambda_end'] - config['lambda_start'])


def compute_temperature(epoch, config):
    """Compute Gumbel-Softmax temperature"""
    temp = config['temperature_start'] * (config['temperature_decay'] ** epoch)
    return max(temp, config['temperature_end'])


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


def print_epoch_summary(epoch, epochs, metrics, losses, config):
    """Print summary of epoch results"""
    print("\n" + "=" * 80)
    print(f"EPOCH {epoch+1}/{epochs}")
    print("=" * 80)

    print("\nLosses:")
    print(f"  D Loss:       {losses['loss_D']:.4f}")
    print(f"  G Loss:       {losses['loss_G']:.4f}")
    if config['use_rl']:
        print(f"  RL Loss:      {losses['loss_RL']:.4f}")
        print(f"  V Loss:       {losses['loss_V']:.4f}")
    print(f"  Grad Penalty: {losses['grad_penalty']:.4f}")

    print("\nGeneration Metrics:")
    print(f"  Reward:       {metrics['reward_mean']:.3f} ± {metrics['reward_std']:.3f}")
    print(f"  Validity:     {metrics['validity_mean']:.3f} ± {metrics['validity_std']:.3f}")
    print(f"  Fitness:      {metrics['fitness_mean']:.3f} ± {metrics['fitness_std']:.3f}")
    print(f"  Conformance:  {metrics['conformance_mean']:.3f} ± {metrics['conformance_std']:.3f}")
    print(f"  Diversity:    {metrics['diversity_mean']:.3f} ± {metrics['diversity_std']:.3f}")

    print("\nQuality Metrics:")
    print(f"  Valid Rate:   {metrics['valid_rate']:.1%}")
    print(f"  Unique Rate:  {metrics['unique_rate']:.1%}")
    print(f"  Novel Rate:   {metrics['novel_rate']:.1%}")
    print(f"  High Quality: {metrics['high_quality_rate']:.1%} (reward >= 0.7)")

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
    log("ProcessGAN Training (TensorFlow 2.x)")
    log("=" * 80)

    # Check TensorFlow version
    log(f"TensorFlow version: {tf.__version__}")
    log(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    # ─────────────────────────────────────────────────────────
    # 1. LOAD DATASET
    # ─────────────────────────────────────────────────────────

    log("Loading dataset...")
    data = ProcessDataset(max_activities=config['max_activities'])

    # Check if data file exists
    if os.path.exists(config['data_file']):
        data.load_from_csv(
            config['data_file'],
            validation=config['validation_split'],
            test=config['test_split']
        )
    else:
        log(f"Data file not found: {config['data_file']}", level='WARNING')
        log("Creating synthetic dataset for demo...", level='WARNING')

        # Create synthetic traces
        synthetic_traces = [
            ['Start', 'Submit', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Review', 'Reject', 'Rework', 'Submit', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Approve', 'End'],
            ['Start', 'Submit', 'Review', 'Approve', 'Send', 'End'],
            ['Start', 'Submit', 'Check', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Review', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Validate', 'Approve', 'End'],
            ['Start', 'Submit', 'Review', 'Reject', 'End'],
        ] * 5  # Repeat to have ~40 traces

        data.generate_from_traces(
            synthetic_traces,
            validation=config['validation_split'],
            test=config['test_split']
        )

    stats = data.get_stats()
    log(f"Dataset loaded: {stats['total_traces']} traces")
    log(f"  Train: {stats['train_size']}, Val: {stats['val_size']}, Test: {stats['test_size']}")
    log(f"  Activities: {stats['num_activities']}, Flow types: {stats['num_flow_types']}")

    # ─────────────────────────────────────────────────────────
    # 2. CREATE REWARD FUNCTION
    # ─────────────────────────────────────────────────────────

    log("Creating reward function...")
    reward_function = ProcessRewardFunction(
        training_traces=data.data,
        weights=config['reward_weights']
    )
    log(f"Reward weights: {config['reward_weights']}")

    # ─────────────────────────────────────────────────────────
    # 3. BUILD MODEL
    # ─────────────────────────────────────────────────────────

    log("Building ProcessGAN model...")
    model = ProcessGAN(
        max_activities=data.max_activities,
        flow_types=data.flow_num_types,
        activity_types=data.activity_num_types,
        embedding_dim=config['z_dim'],
        decoder_units=config['decoder_units'],
        discriminator_units=config['discriminator_units'],
        mlp_units=config['mlp_units'],
        dropout_rate=config['dropout_rate']
    )
    log("Model built successfully")

    # ─────────────────────────────────────────────────────────
    # 4. CREATE TRAINER
    # ─────────────────────────────────────────────────────────

    log("Creating trainer...")
    trainer = ProcessGANTrainer(
        model,
        learning_rate=config['learning_rate'],
        gradient_penalty_weight=10.0,
        lambda_adv=0.6,
        lambda_reward=0.4
    )
    log("Trainer created")

    # Build models by calling them with sample input to create weights
    sample_z = tf.random.normal((1, config['z_dim']))
    sample_adj, sample_nodes = model.generator(sample_z, training=False)
    _ = model.discriminator(sample_adj, sample_nodes, training=False)
    _ = model.value_network(sample_adj, sample_nodes, training=False)

    # Count parameters
    total_g = sum([np.prod(v.shape) for v in model.generator.trainable_variables])
    total_d = sum([np.prod(v.shape) for v in model.discriminator.trainable_variables])
    total_v = sum([np.prod(v.shape) for v in model.value_network.trainable_variables])
    log(f"Total trainable parameters: {total_g + total_d + total_v:,}")

    # Create checkpoint directory
    create_checkpoint_dir(config['save_dir'])

    # ─────────────────────────────────────────────────────────
    # 5. TRAINING LOOP
    # ─────────────────────────────────────────────────────────

    log("\n" + "=" * 80)
    log("Starting training...")
    log("=" * 80 + "\n")

    steps_per_epoch = data.train_count // config['batch_size']

    for epoch in range(config['epochs']):

        # Compute dynamic parameters
        lambda_mix = compute_lambda(epoch, config)
        temperature = compute_temperature(epoch, config)

        # Reset metrics
        trainer.reset_metrics()

        # Training epoch
        for step in range(steps_per_epoch):
            # Get batch
            traces_batch, adj_batch, nodes_batch, _ = data.next_train_batch(config['batch_size'])

            # Training step
            losses = trainer.train_step(
                real_adj=adj_batch,
                real_nodes=nodes_batch,
                batch_size=config['batch_size'],
                n_critic=config['n_critic'],
                reward_function=reward_function,
                lambda_mix=lambda_mix,
                temperature=temperature
            )

        # Get average losses
        avg_losses = trainer.get_metrics()

        # Evaluation
        if (epoch + 1) % config['log_every'] == 0:
            metrics = evaluate_model(model, data, reward_function, config, epoch)
            print_epoch_summary(epoch, config['epochs'], metrics, avg_losses, config)

        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'epoch_{epoch+1}')
            model.generator.save_weights(os.path.join(checkpoint_path, 'generator'))
            model.discriminator.save_weights(os.path.join(checkpoint_path, 'discriminator'))
            model.value_network.save_weights(os.path.join(checkpoint_path, 'value_network'))
            log(f"Checkpoint saved: {checkpoint_path}")

    # ─────────────────────────────────────────────────────────
    # 6. FINAL EVALUATION
    # ─────────────────────────────────────────────────────────

    log("\n" + "=" * 80)
    log("Training completed! Running final evaluation...")
    log("=" * 80 + "\n")

    # Generate larger sample
    log(f"Generating {config['n_samples_eval']*5} final samples...")
    final_traces = generate_samples(
        model,
        data,
        n_samples=config['n_samples_eval'] * 5,
        temperature=0.5
    )

    # Compute final metrics
    final_metrics = reward_function.evaluate_batch(final_traces)
    valid_traces = ProcessMetrics.valid_traces(final_traces)
    unique_traces = ProcessMetrics.unique_traces(final_traces)
    novel_traces = ProcessMetrics.novel_traces(final_traces, data.data)

    print("\nFinal Generation Quality:")
    print(f"  Total generated:      {len(final_traces)}")
    print(f"  Valid traces:         {len(valid_traces)} ({len(valid_traces)/len(final_traces):.1%})")
    print(f"  Unique traces:        {len(unique_traces)} ({len(unique_traces)/len(final_traces):.1%})")
    print(f"  Novel traces:         {len(novel_traces)} ({len(novel_traces)/len(final_traces):.1%})")
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

    # Save final model
    final_path = os.path.join(config['save_dir'], 'final')
    model.generator.save_weights(os.path.join(final_path, 'generator'))
    model.discriminator.save_weights(os.path.join(final_path, 'discriminator'))
    model.value_network.save_weights(os.path.join(final_path, 'value_network'))
    log(f"\nFinal model saved: {final_path}")

    log("\n" + "=" * 80)
    log("Training completed successfully!")
    log("=" * 80)


if __name__ == '__main__':
    # Enable mixed precision for faster training (optional)
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')

    main()
