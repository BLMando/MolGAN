"""
ProcessGAN Training Script

Modern training implementation using TensorFlow 2.x eager execution.
Compatible with TensorFlow 2.15+ and Python 3.10+.

All configuration is managed via YAML config file.
No CLI parameter overrides are supported - edit config.yml instead.

Usage:
    # Train with default config
    python trainer.py
    
    # Train with custom config file
    python trainer.py --config my_config.yml
"""

import numpy as np
import tensorflow as tf
import os
from datetime import datetime

# Import refactored modules
from config import ConfigManager
from training import (
    DynamicRewardWeights,
    EarlyStoppingMonitor,
    CheckpointManager,
    LambdaMixScheduler,
    TemperatureScheduler,
    LearningRateScheduler,
    ModelEvaluator
)
from visualization import TrainingVisualizer

# Import ProcessGAN components
from utils.dataset import ProcessDataset
from utils.metrics import ProcessRewardFunction
from utils.model_utils import (
    load_reference_model,
    detect_start_end_activities,
    build_and_initialize_model,
    count_model_parameters
)
from models.process_gan import ProcessGAN
from optimizers.optimizer import ProcessGANTrainer


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {msg}')


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
    # 1. LOAD CONFIG FROM YAML AND PARSE ARGUMENTS
    # ─────────────────────────────────────────────────────────

    # Parse command-line arguments and load configuration
    config_manager = ConfigManager()
    args = config_manager.parse_args()
    config = config_manager.load_from_yaml(args.config)

    log("Configuration loaded successfully")
    log(f"  Config file: {args.config}")
    log(f"  ├─ data_file: {config.get('data_file', 'N/A')}")
    log(f"  ├─ max_activities: {config.get('max_activities', 'N/A')}")
    log(f"  ├─ z_dim: {config.get('z_dim', 'N/A')}")
    log(f"  ├─ epochs: {config.get('epochs', 'N/A')}")
    log(f"  ├─ batch_size: {config.get('batch_size', 'N/A')}")
    log(f"  ├─ learning_rate: {config.get('learning_rate', 'N/A')}")
    log(f"  ├─ save_dir: {config.get('save_dir', 'N/A')}")
    log(f"  └─ log_dir: {config.get('log_dir', 'N/A')}")

    # ─────────────────────────────────────────────────────────
    # 2. INITIALIZE TRAINING COMPONENTS
    # ─────────────────────────────────────────────────────────

    log("\nInitializing training components...")

    # Dynamic reward weights
    dynamic_rewards = DynamicRewardWeights(
        weights_early=config['reward_weights_early'],
        weights_late=config['reward_weights_late'],
        transition_epoch=config['reward_transition_epoch']
    )

    # Early stopping monitor
    early_stopping = EarlyStoppingMonitor(
        patience=config['early_stopping_patience'],
        metric=config['early_stopping_metric'],
        threshold=config['early_stopping_threshold'],
        mode='max'
    )

    # Lambda scheduler (GAN/RL mixing)
    lambda_scheduler = LambdaMixScheduler(
        lambda_start=config['lambda_start'],
        lambda_end=config['lambda_end'],
        decay_start=config['lambda_decay_start'],
        total_epochs=config['epochs']
    )

    # Temperature scheduler (Gumbel-Softmax)
    temperature_scheduler = TemperatureScheduler(
        temp_start=config['temperature_start'],
        temp_end=config['temperature_end'],
        decay_rate=config['temperature_decay']
    )

    # Learning rate scheduler (Cosine Annealing with Warm Restarts)
    lr_scheduler = LearningRateScheduler(
        lr_initial=config['learning_rate'],
        lr_min=config['learning_rate'] * 0.01,  # Min LR = 1% of initial
        cycle_length=50,  # Restart every 50 epochs
        cycle_mult=1.5,   # Increase cycle length by 1.5x each restart
        warmup_epochs=5   # 5 epochs of linear warmup
    )

    # ─────────────────────────────────────────────────────────
    # 4. LOAD DATASET
    # ─────────────────────────────────────────────────────────

    log("\nLoading dataset...")
    data = ProcessDataset(max_activities=config['max_activities'])

    # Load based on file extension
    if os.path.exists(config['data_file']):
        file_ext = os.path.splitext(config['data_file'])[1].lower()

        if file_ext == '.xes':
            data.load_from_xes(
                config['data_file'],
                validation=config['validation_split'],
                test=config['test_split']
            )
        else:
            log(f"Unknown file extension: {file_ext}. Expected '.xes'", level='ERROR')
            return
    else:
        log(f"Data file not found: {config['data_file']}", level='ERROR')
        return

    stats = data.get_stats()
    log(f"Dataset loaded: {stats['total_traces']} traces")
    log(f"Train: {stats['train_size']}, Val: {stats['val_size']}, Test: {stats['test_size']}")
    log(f"Activities: {stats['num_activities']}, Flow types: {stats['num_flow_types']}")

    # ─────────────────────────────────────────────────────────
    # 3. CREATE REWARD FUNCTION
    # ─────────────────────────────────────────────────────────

    log("Creating reward function...")

    # Load reference Petri net model
    reference_model = load_reference_model(
        data_file=config['data_file'],
        reward_function_class=ProcessRewardFunction,
        log_func=log
    )

    # Auto-detect START/END activities from dataset
    start_activity, end_activity = detect_start_end_activities(data.data)

    log(f"Auto-detected START activity: {start_activity}")
    log(f"Auto-detected END activity: {end_activity}")

    # Get initial reward weights (early phase)
    initial_weights = dynamic_rewards.get_weights(epoch=0)

    reward_function = ProcessRewardFunction(
        reference_model=reference_model,
        training_traces=data.data,
        weights=initial_weights,  # Use dynamic weights
        start_activity=start_activity,
        end_activity=end_activity
    )
    log(f"Initial reward weights (EARLY phase): {initial_weights}")

    # ─────────────────────────────────────────────────────────
    # 5. BUILD MODEL
    # ─────────────────────────────────────────────────────────

    log("\nBuilding ProcessGAN model...")
    model = ProcessGAN(
        max_activities=data.max_activities,
        flow_types=data.flow_num_types,
        activity_types=data.activity_num_types,
        embedding_dim=config['z_dim'],
        decoder_units=config['decoder_units'],
        discriminator_units=config['discriminator_units'],
        mlp_units=config['mlp_units'],
        dropout_rate=config['dropout_rate'],
        enforce_start=config['enforce_start']
    )

    # ─────────────────────────────────────────────────────────
    # 5. CREATE TRAINER
    # ─────────────────────────────────────────────────────────

    log("Creating trainer...")
    trainer = ProcessGANTrainer(
        model,
        dataset=data,
        learning_rate=config['learning_rate'],
        gradient_penalty_weight=10.0,
        lambda_adv=0.6,
        lambda_reward=0.4
    )

    # Build and initialize model
    build_and_initialize_model(model, config['z_dim'])

    # Count parameters
    param_counts = count_model_parameters(model)
    log(f"Total trainable parameters: {param_counts['total']:,}")
    log(f"  ├─ Generator: {param_counts['generator']:,}")
    log(f"  ├─ Discriminator: {param_counts['discriminator']:,}")
    log(f"  └─ Value Network: {param_counts['value_network']:,}")

    checkpoint_manager = CheckpointManager(
        save_dir=config['save_dir']
    )

    # ─────────────────────────────────────────────────────────
    # 6. TRAINING LOOP
    # ─────────────────────────────────────────────────────────

    log("\n" + "=" * 80)
    log("Starting training...")
    log("=" * 80 + "\n")

    steps_per_epoch = data.train_count // config['batch_size']

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

    for epoch in range(config['epochs']):

        current_weights = dynamic_rewards.get_weights(epoch)
        reward_function.update_weights(current_weights)

        # Get dynamic parameters from schedulers
        lambda_mix = lambda_scheduler.get_lambda(epoch)
        temperature = temperature_scheduler.get_temperature(epoch)
        current_lr = lr_scheduler.get_learning_rate(epoch)

        # Update learning rate
        trainer.update_learning_rate(current_lr)

        # Reset metrics
        trainer.reset_metrics()

        # Training epoch
        for step in range(steps_per_epoch):
            _, adj_batch, nodes_batch, _ = data.next_train_batch(
                config['batch_size'])

            # Training step (losses are accumulated in trainer.metrics)
            _ = trainer.train_step(
                real_adj=adj_batch,
                real_nodes=nodes_batch,
                batch_size=config['batch_size'],
                n_critic=config['n_critic'],
                reward_function=reward_function,
                lambda_mix=lambda_mix,
                temperature=temperature
            )

        # Get average losses over all steps
        avg_losses = trainer.get_metrics()

        # Initialize evaluator for this epoch
        evaluator = ModelEvaluator(
            model=model,
            dataset=data,
            reward_function=reward_function,
            config=config
        )

        # Compute validation losses
        val_losses = evaluator.compute_validation_losses(
            trainer=trainer,
            temperature=temperature
        )

        # Evaluation
        if (epoch + 1) % config['log_every'] == 0:
            metrics = evaluator.evaluate_samples(
                n_samples=config['n_samples_eval'],
                temperature=temperature
            )
            log(f"Epoch {epoch + 1}/{config['epochs']} - LR: {current_lr:.2e}, Lambda: {lambda_mix:.3f}, Temp: {temperature:.3f}")
            evaluator.print_epoch_summary(
                epoch=epoch,
                total_epochs=config['epochs'],
                eval_metrics=metrics,
                train_losses=avg_losses,
                val_losses=val_losses,
                use_rl=config['use_rl']
            )
        else:
            # Still compute metrics for best model tracking
            metrics = evaluator.evaluate_samples(
                n_samples=config['n_samples_eval'],
                temperature=temperature
            )

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

        # Save checkpoint using CheckpointManager
        checkpoint_manager.save_checkpoint(
            model=model,
            epoch=epoch,
            metrics=metrics
        )

        if early_stopping.check(epoch, metrics):
            log("\n" + "=" * 80)
            log("EARLY STOPPING TRIGGERED")
            log("=" * 80)
            best_info = early_stopping.get_best_info()
            log(f"Best {config['early_stopping_metric']}: {best_info['best_value']:.4f}")
            log(f"Best epoch: {best_info['best_epoch']}")
            log(f"Stopped at epoch: {best_info['stopped_epoch']}")
            log(f"Epochs saved: {config['epochs'] - (epoch + 1)}")
            break

    # ─────────────────────────────────────────────────────────
    # 7. PLOT TRAINING CURVES
    # ─────────────────────────────────────────────────────────

    log("\n" + "=" * 80)
    log("Generating training plots...")
    log("=" * 80 + "\n")

    visualizer = TrainingVisualizer(save_dir=config['save_dir'])
    visualizer.save_all_plots(history)

    best_info = checkpoint_manager.get_best_info()

    log("Training complete!")
    log(f"Best model: Epoch {best_info['epoch']}, Reward: {best_info['reward']:.4f}")
    log(f"Checkpoints saved in: {config['save_dir']}")
    log(
        f"Training plots saved in: {os.path.join(config['save_dir'], 'plots')}")
    log("=" * 80)


if __name__ == '__main__':
    main()
