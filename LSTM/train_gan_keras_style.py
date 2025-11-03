"""
Train Process Mining GAN using Keras .fit() API
Demonstrates how to use the new Keras-style training
"""

import numpy as np
import tensorflow as tf
import os
import sys

from xes_loader import load_xes_for_gan
from process_gan_keras_api import ProcessGAN
from callbacks import CallbackFactory
from dataset import ProcessTraceDataset
from evaluation import evaluate_generated_traces


def train_gan_keras_style(xes_filepath,
                          output_dir='../output',
                          # Data parameters
                          min_trace_length=2,
                          max_trace_length=10,
                          # Model parameters
                          noise_dim=128,
                          embedding_dim=64,
                          generator_lstm_units=256,
                          discriminator_lstm_units=256,
                          lstm_layers=2,
                          # Training parameters
                          batch_size=32,
                          epochs=100,
                          learning_rate=0.0001,
                          n_critic=5,
                          lambda_gp=10.0,
                          lambda_constraint=0.1,
                          # Other
                          use_validation=True,
                          verbose=True):
    """
    Train GAN

    Benefits over manual training loop:
    - Automatic progress bars
    - Built-in callbacks (ModelCheckpoint, EarlyStopping, TensorBoard, etc.)
    - Validation support
    - Easier to extend
    """

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("\n" + "="*80)
        print("PROCESS MINING GAN ")
        print("="*80)

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("LOADING XES FILE")
        print("="*80)

    data = load_xes_for_gan(
        xes_filepath=xes_filepath,
        min_trace_length=min_trace_length,
        max_trace_length=max_trace_length,
        min_activity_frequency=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        verbose=verbose
    )

    # ========================================================================
    # CREATE DATASETS
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("CREATING TF DATASETS")
        print("="*80)

    train_dataset = ProcessTraceDataset(
        traces=data['traces_train'],
        activity_to_idx=data['activity_to_idx'],
        max_length=data['max_trace_length']
    )

    tf_train_dataset = train_dataset.get_tf_dataset(
        batch_size=batch_size,
        shuffle=True
    )

    # Create validation dataset if requested
    val_dataset = None
    if use_validation:
        val_dataset = ProcessTraceDataset(
            traces=data['traces_val'],
            activity_to_idx=data['activity_to_idx'],
            max_length=data['max_trace_length']
        )
        tf_val_dataset = val_dataset.get_tf_dataset(
            batch_size=batch_size,
            shuffle=False
        )

    if verbose:
        print(f"\nDataset sizes:")
        print(f"  - Training: {len(data['traces_train'])} traces")
        print(f"  - Validation: {len(data['traces_val'])} traces")
        print(f"  - Test: {len(data['traces_test'])} traces")

    # ========================================================================
    # INITIALIZE GAN
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("INITIALIZING GAN (KERAS API)")
        print("="*80)

    gan = ProcessGAN(
        num_activities=data['num_activities'],
        max_trace_length=data['max_trace_length'],
        start_idx=data['special_tokens']['start_idx'],
        end_idx=data['special_tokens']['end_idx'],
        activity_frequencies=data['activity_frequencies'],
        noise_dim=noise_dim,
        embedding_dim=embedding_dim,
        generator_lstm_units=generator_lstm_units,
        discriminator_lstm_units=discriminator_lstm_units,
        lstm_layers=lstm_layers,
        n_critic=n_critic,
        lambda_gp=lambda_gp,
        lambda_constraint=lambda_constraint
    )

    # ========================================================================
    # COMPILE GAN
    # ========================================================================
    if verbose:
        print("\nCompiling model...")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Discriminator updates per generator update: {n_critic}")
        print(f"  Gradient penalty weight: {lambda_gp}")

    gan.compile(
        d_optimizer=tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.5, beta_2=0.9),
        g_optimizer=tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.5, beta_2=0.9)
    )

    if verbose:
        print("✓ Model compiled")

    # ========================================================================
    # SETUP CALLBACKS
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("SETTING UP CALLBACKS")
        print("="*80)
    
    # Create callback factory
    callback_factory = CallbackFactory(
        output_dir=output_dir,
        idx_to_activity=data['idx_to_activity']
    )
    
    callbacks = []

    # 1. Trace generation callback (show examples during training)
    callbacks.append(
        callback_factory.trace_generation(
            num_traces=5,
            display_every=10  # Show every 10 epochs
        )
    )

    # 2. Best model checkpoint (always save best model for evaluation)
    callbacks.append(
        callback_factory.best_model_checkpoint(
            monitor='val_g_loss' if use_validation else 'g_loss',
            mode='min',
            verbose=1
        )
    )
    
    # 3. Backup and restore (automatic recovery from interruptions)
    callbacks.append(
        callback_factory.backup_and_restore()
    )
    
    # 4. Quality Evaluation (PROPER TEST - computes metrics!)
    # callbacks.append(
    #     callback_factory.quality_evaluation(
    #         real_traces=data['traces_train'],
    #         activity_to_idx=data['activity_to_idx'],
    #         idx_to_activity=data['idx_to_activity'],
    #         eval_every=10,  # Evaluate every 10 epochs
    #         num_samples=100  # Generate 100 traces for evaluation
    #     )
    # )
   
    
    # 4. Early Stopping
    # callbacks.append(
    #     callback_factory.early_stopping(
    #         monitor='val_g_loss' if use_validation else 'g_loss',
    #         patience=10
    #     )
    # )
    
    # 5. Custom Learning Rate Scheduler
    # def lr_schedule(epoch, lr):
    #     if epoch > 50:
    #         return lr * 0.95
    #     return lr
    
    # callbacks.append(
    #     callback_factory.learning_rate_scheduler(lr_schedule)
    # )
    
    # 8. TensorBoard (visualization)
    # callbacks.append(
    #     callback_factory.tensorboard()
    # )


    if verbose:
        print(f"\n Setup {len(callbacks)} callbacks")
        for cb in callbacks:
            print(f"  - {cb.__class__.__name__}")

    # ========================================================================
    # TRAIN GAN
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("TRAINING GAN")
        print("="*80)
        print(f"\nTraining for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Steps per epoch: ~{len(data['traces_train']) // batch_size}")
        if use_validation:
            print(
                f"Validation steps: ~{len(data['traces_val']) // batch_size}")
        print()

    # Train the model
    gan.fit(
        tf_train_dataset,
        epochs=epochs,
        validation_data=tf_val_dataset if use_validation else None,
        callbacks=callbacks,
        verbose=1 if verbose else 0
    )

    if verbose:
        print("\n✓ Training completed!")

    # ========================================================================
    # GENERATE AND EVALUATE
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("GENERATING SYNTHETIC TRACES")
        print("="*80)

    synthetic_traces_indices = gan.generate_traces(
        num_samples=1000,
        temperature=0.5,
        return_indices=True
    )

    synthetic_traces = [
        [data['idx_to_activity'][idx] for idx in trace
         if idx in data['idx_to_activity']]
        for trace in synthetic_traces_indices
    ]

    if verbose:
        print(f"✓ Generated {len(synthetic_traces)} synthetic traces")

    # Evaluate
    evaluate_generated_traces(
        generated_traces=synthetic_traces,
        real_traces=data['traces_test'],  # Use test set for evaluation
        activity_to_idx=data['activity_to_idx'],
        idx_to_activity=data['idx_to_activity'],
        start_token='START',
        end_token='6',
        verbose=verbose
    )


    if verbose:
        print(f"\nResults saved to {output_dir}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Process Mining GAN using Keras .fit() API',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--output-dir', type=str, default='../output_keras',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lstm-units', type=int, default=256,
                        help='LSTM hidden units')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate for both optimizers')
    parser.add_argument('--n-critic', type=int, default=3,
                        help='Discriminator updates per generator update (lower = more balanced)')
    parser.add_argument('--no-validation', action='store_true',
                        help='Disable validation during training')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Disable verbose output')

    args = parser.parse_args()

    xes_file = '../data/helpdesk_parsed.xes'
    if not os.path.exists(xes_file):
        print(f"Error: XES file not found: {xes_file}")
        sys.exit(1)

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Train model
    train_gan_keras_style(
        xes_filepath=xes_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        n_critic=args.n_critic,
        generator_lstm_units=args.lstm_units,
        discriminator_lstm_units=args.lstm_units,
        use_validation=not args.no_validation,
        verbose=not args.no_verbose
    )

    print("\nProcess complete!")
    print(f"Check {args.output_dir}/ for all outputs")
    print(f"💾 Best model saved to: {args.output_dir}/checkpoints/best_model/")
