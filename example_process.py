"""
ProcessGAN Training Script

Example script for training ProcessGAN on event logs.
Generates synthetic process traces for data augmentation.

Usage:
    python example_process.py

Configuration:
    Edit config dict below to customize training
"""

import numpy as np
import tensorflow as tf
import os
from datetime import datetime

# Import ProcessGAN components
from utils.process_dataset import ProcessDataset
from utils.process_metrics import ProcessRewardFunction, ProcessMetrics
from models.process_gan import ProcessGANModel, matrices_to_traces
from optimizers.process_optimizer import ProcessGANOptimizer, training_step


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
    'discriminator_units': ((128, 64), 128, (128, 64)),

    # Training
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'n_critic': 5,  # Discriminator steps per generator step
    'dropout': 0.0,

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
    'save_dir': 'checkpoints/process_gan',
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        log(f'Created checkpoint directory: {save_dir}')


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


def generate_samples(session, model, dataset, n_samples, temperature=0.5):
    """Generate samples from model"""
    z = model.sample_z(n_samples)

    nodes, edges = session.run(
        [model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
        feed_dict={
            model.embeddings: z,
            model.training: False,
            model.temperature: temperature
        }
    )

    nodes = np.argmax(nodes, axis=-1)
    edges = np.argmax(edges, axis=-1)

    traces = matrices_to_traces(edges, nodes, dataset)

    return traces


def evaluate_model(session, model, dataset, reward_function, config, epoch):
    """Evaluate model by generating samples and computing metrics"""
    log('Generating samples for evaluation...')

    # Generate samples
    temperature = compute_temperature(epoch, config)
    traces = generate_samples(
        session,
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
    print(f"  D Loss:     {losses['loss_D']:.4f}")
    print(f"  G Loss:     {losses['loss_G']:.4f}")
    if config['use_rl']:
        print(f"  RL Loss:    {losses['loss_RL']:.4f}")
        print(f"  V Loss:     {losses['loss_V']:.4f}")
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
    log("ProcessGAN Training")
    log("=" * 80)

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
    model = ProcessGANModel(
        max_activities=data.max_activities,
        flow_types=data.flow_num_types,
        activity_types=data.activity_num_types,
        embedding_dim=config['z_dim'],
        decoder_units=config['decoder_units'],
        discriminator_units=config['discriminator_units'],
        soft_gumbel_softmax=True,
        hard_gumbel_softmax=False,
        batch_discriminator=False
    )
    log("Model built successfully")

    # ─────────────────────────────────────────────────────────
    # 4. CREATE OPTIMIZER
    # ─────────────────────────────────────────────────────────

    log("Creating optimizer...")
    optimizer = ProcessGANOptimizer(
        model,
        learning_rate=config['learning_rate'],
        feature_matching=True,
        gradient_penalty_weight=10.0
    )
    log("Optimizer created")

    # ─────────────────────────────────────────────────────────
    # 5. TENSORFLOW SESSION AND INITIALIZATION
    # ─────────────────────────────────────────────────────────

    log("Initializing TensorFlow session...")
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Count parameters
    total_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    log(f"Total trainable parameters: {total_params:,}")

    # Create checkpoint directory
    create_checkpoint_dir(config['save_dir'])

    # Saver for checkpoints
    saver = tf.Saver(max_to_keep=5)

    # ─────────────────────────────────────────────────────────
    # 6. TRAINING LOOP
    # ─────────────────────────────────────────────────────────

    log("\n" + "=" * 80)
    log("Starting training...")
    log("=" * 80 + "\n")

    steps_per_epoch = data.train_count // config['batch_size']

    for epoch in range(config['epochs']):

        # Compute dynamic parameters
        la = compute_lambda(epoch, config)
        temperature = compute_temperature(epoch, config)

        # Training epoch
        epoch_losses = []

        for step in range(steps_per_epoch):
            losses = training_step(
                session=session,
                model=model,
                optimizer=optimizer,
                data=data,
                batch_size=config['batch_size'],
                n_critic=config['n_critic'],
                la=la,
                reward_function=reward_function,
                epoch=epoch,
                use_rl=config['use_rl']
            )
            epoch_losses.append(losses)

        # Average losses
        avg_losses = {key: np.mean([l[key] for l in epoch_losses]) for key in epoch_losses[0].keys()}

        # Evaluation
        if (epoch + 1) % config['log_every'] == 0:
            metrics = evaluate_model(session, model, data, reward_function, config, epoch)
            print_epoch_summary(epoch, config['epochs'], metrics, avg_losses, config)

        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'model_epoch_{epoch+1}.ckpt')
            saver.save(session, checkpoint_path)
            log(f"Checkpoint saved: {checkpoint_path}")

    # ─────────────────────────────────────────────────────────
    # 7. FINAL EVALUATION
    # ─────────────────────────────────────────────────────────

    log("\n" + "=" * 80)
    log("Training completed! Running final evaluation...")
    log("=" * 80 + "\n")

    # Generate larger sample
    log(f"Generating {config['n_samples_eval']*5} final samples...")
    final_traces = generate_samples(
        session,
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
    print(f"  Total generated:    {len(final_traces)}")
    print(f"  Valid traces:       {len(valid_traces)} ({len(valid_traces)/len(final_traces):.1%})")
    print(f"  Unique traces:      {len(unique_traces)} ({len(unique_traces)/len(final_traces):.1%})")
    print(f"  Novel traces:       {len(novel_traces)} ({len(novel_traces)/len(final_traces):.1%})")
    print(f"  High quality (≥0.7): {final_metrics['high_quality_rate']:.1%}")

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
    final_path = os.path.join(config['save_dir'], 'model_final.ckpt')
    saver.save(session, final_path)
    log(f"\nFinal model saved: {final_path}")

    # Close session
    session.close()

    log("\n" + "=" * 80)
    log("Training completed successfully!")
    log("=" * 80)


if __name__ == '__main__':
    main()
