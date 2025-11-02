"""
Train Process Mining GAN from XES File
Complete end-to-end pipeline
"""

import numpy as np
import tensorflow as tf
import os
import sys

# Import our modules
from xes_loader import load_xes_for_gan, save_preprocessed_xes
from process_gan_tensorflow import ProcessGAN
from dataset import ProcessTraceDataset
from evaluation import evaluate_generated_traces, plot_trace_length_comparison


def train_gan_from_xes(xes_filepath,
                       output_dir='../output',
                       # Data parameters
                       min_trace_length=2,
                       max_trace_length=10,
                       min_activity_frequency=1,
                       # Model parameters
                       noise_dim=128,
                       embedding_dim=64,
                       generator_lstm_units=256,
                       discriminator_lstm_units=256,
                       lstm_layers=2,
                       # Training parameters
                       batch_size=32,
                       epochs=200,
                       learning_rate=0.0001,
                       n_critic=5,
                       lambda_gp=10.0,
                       lambda_constraint=0.1,
                       # Generation parameters
                       num_synthetic=1000,
                       generation_temperature=0.5,
                       # Other
                       eval_every=500,
                       save_checkpoints=True,
                       verbose=True):
    """
    Complete pipeline to train Process Mining GAN from XES file
    
    Parameters:
    -----------
    xes_filepath: str
        Path to XES file
    output_dir: str
        Directory for outputs (checkpoints, generated traces, etc.)
    ... (see parameter descriptions in code)
    
    Returns:
    --------
    results: dict
        Dictionary containing trained GAN, generated traces, and evaluation metrics
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    if verbose:
        print("\n" + "="*80)
        print(" PROCESS MINING GAN - TRAINING FROM XES FILE ")
        print("="*80)
        print(f"\nXES file: {xes_filepath}")
        print(f"Output directory: {output_dir}")
    
    # ========================================================================
    # STEP 1: LOAD AND PREPROCESS XES FILE
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 1: LOADING XES FILE")
        print("="*80)
    
    data = load_xes_for_gan(
        xes_filepath=xes_filepath,
        min_trace_length=min_trace_length,
        max_trace_length=max_trace_length,
        min_activity_frequency=min_activity_frequency,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        verbose=verbose
    )
    
    # Save preprocessed data
    preprocessed_path = os.path.join(output_dir, 'preprocessed_data.pkl')
    save_preprocessed_xes(data, preprocessed_path)
    
    # ========================================================================
    # STEP 2: CREATE TENSORFLOW DATASET
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 2: CREATING TENSORFLOW DATASET")
        print("="*80)
    
    train_dataset = ProcessTraceDataset(
        traces=data['traces_train'],
        activity_to_idx=data['activity_to_idx'],
        max_length=data['max_trace_length']
    )
    
    tf_train_dataset = train_dataset.get_tf_dataset(batch_size=batch_size, shuffle=True)
    
    if verbose:
        print(f"\nDataset configuration:")
        print(f"  - Training traces: {len(data['traces_train'])}")
        print(f"  - Validation traces: {len(data['traces_val'])}")
        print(f"  - Test traces: {len(data['traces_test'])}")
        print(f"  - Vocabulary size: {data['num_activities']}")
        print(f"  - Max trace length: {data['max_trace_length']}")
        print(f"  - Batch size: {batch_size}")
        
        print(f"\nExample traces:")
        for i in range(min(3, len(data['traces_train']))):
            # Filter out special tokens for display
            activities = [act for act in data['traces_train'][i]
                         if act not in ['<PAD>', '<UNK>']]
            trace_str = ' → '.join(activities[:8])
            if len(activities) > 8:
                trace_str += ' → ...'
            print(f"  {i+1}. {trace_str}")
    
    # ========================================================================
    # STEP 3: INITIALIZE GAN
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 3: INITIALIZING GAN")
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
        learning_rate=learning_rate,
        n_critic=n_critic,
        lambda_gp=lambda_gp,
        lambda_constraint=lambda_constraint
    )
    
    if verbose:
        print(f"\nGAN architecture:")
        print(f"  Generator:")
        print(f"    - Noise dimension: {noise_dim}")
        print(f"    - Embedding dimension: {embedding_dim}")
        print(f"    - LSTM units: {generator_lstm_units}")
        print(f"    - LSTM layers: {lstm_layers}")
        print(f"  Discriminator:")
        print(f"    - LSTM units: {discriminator_lstm_units}")
        print(f"    - LSTM layers: {lstm_layers}")
        print(f"  Training:")
        print(f"    - Learning rate: {learning_rate}")
        print(f"    - Discriminator updates per generator update: {n_critic}")
        print(f"    - Gradient penalty weight: {lambda_gp}")
        print(f"    - Constraint loss weight: {lambda_constraint}")
    
    # ========================================================================
    # STEP 4: TRAIN GAN
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 4: TRAINING GAN")
        print("="*80)
        print(f"\nTraining for {epochs} epochs...")
        print("(This may take a while depending on dataset size)")
    
    gan.train(
        dataset=tf_train_dataset,
        epochs=epochs,
        eval_every=eval_every,
        save_dir=checkpoints_dir if save_checkpoints else None,
        verbose=verbose,
        idx_to_activity=data['idx_to_activity'],
        num_preview_traces=5
    )
    
    if verbose:
        print("\n✓ Training completed!")
    
    # ========================================================================
    # STEP 5: GENERATE SYNTHETIC TRACES
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 5: GENERATING SYNTHETIC TRACES")
        print("="*80)
        print(f"\nGenerating {num_synthetic} synthetic traces...")
    
    synthetic_traces_indices = gan.generate_traces(
        num_samples=num_synthetic,
        temperature=generation_temperature,
        return_indices=True
    )
    
    # Convert indices to activity names
    synthetic_traces = [
        [data['idx_to_activity'][idx] for idx in trace 
         if idx in data['idx_to_activity']]
        for trace in synthetic_traces_indices
    ]
    
    if verbose:
        print(f"\n✓ Generated {len(synthetic_traces)} synthetic traces")
        print(f"\nExample synthetic traces:")
        for i in range(min(5, len(synthetic_traces))):
            # Filter out special tokens for display
            activities = [act for act in synthetic_traces[i]
                         if act not in ['<PAD>', '<UNK>']]
            trace_str = ' → '.join(activities[:8])
            if len(activities) > 8:
                trace_str += ' → ...'
            print(f"  {i+1}. {trace_str}")
    
    # ========================================================================
    # STEP 6: EVALUATE QUALITY
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 6: EVALUATING SYNTHETIC TRACES")
        print("="*80)
    
    metrics = evaluate_generated_traces(
        generated_traces=synthetic_traces,
        real_traces=data['traces_train'],
        activity_to_idx=data['activity_to_idx'],
        idx_to_activity=data['idx_to_activity'],
        start_token='START',
        end_token='6',
        verbose=verbose
    )
    
    # ========================================================================
    # STEP 7: VISUALIZE RESULTS
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 7: GENERATING VISUALIZATIONS")
        print("="*80)
    
    try:
        viz_path = os.path.join(output_dir, 'trace_length_comparison.png')
        plot_trace_length_comparison(
            synthetic_traces,
            data['traces_train'],
            save_path=viz_path
        )
        if verbose:
            print(f"✓ Saved visualization: {viz_path}")
    except Exception as e:
        if verbose:
            print(f"⚠ Visualization skipped: {e}")
    
    # ========================================================================
    # STEP 8: SAVE OUTPUTS
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 8: SAVING OUTPUTS")
        print("="*80)
    
    # Save synthetic traces as pickle
    import pickle
    synthetic_traces_path = os.path.join(output_dir, 'synthetic_traces.pkl')
    with open(synthetic_traces_path, 'wb') as f:
        pickle.dump(synthetic_traces, f)
    if verbose:
        print(f"✓ Saved synthetic traces: {synthetic_traces_path}")
    
    # Save synthetic traces as CSV (event log format)
    import pandas as pd
    synthetic_event_log = []
    for case_idx, trace in enumerate(synthetic_traces, start=1):
        timestamp = pd.Timestamp('2024-01-01')
        for activity in trace:
            if activity not in ['<PAD>', '<UNK>']:
                synthetic_event_log.append({
                    'case:concept:name': f'synthetic_{case_idx}',
                    'concept:name': activity,
                    'time:timestamp': timestamp
                })
                timestamp += pd.Timedelta(hours=1)
    
    synthetic_df = pd.DataFrame(synthetic_event_log)
    csv_path = os.path.join(output_dir, 'synthetic_event_log.csv')
    synthetic_df.to_csv(csv_path, index=False)
    if verbose:
        print(f"✓ Saved event log CSV: {csv_path}")
    
    # Save evaluation metrics
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    if verbose:
        print(f"✓ Saved metrics: {metrics_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print(" TRAINING COMPLETE - SUMMARY ")
        print("="*80)
        print(f"\nInput:")
        print(f"  - XES file: {xes_filepath}")
        print(f"  - Training traces: {len(data['traces_train'])}")
        print(f"  - Unique activities: {data['num_activities']}")
        
        print(f"\nTraining:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        
        print(f"\nOutput:")
        print(f"  - Synthetic traces: {len(synthetic_traces)}")
        print(f"  - Output directory: {output_dir}")
        
        print(f"\nQuality Metrics:")
        print(f"  - Diversity: {metrics['diversity']:.3f}")
        print(f"  - Novelty: {metrics['novelty']:.3f}")
        print(f"  - DFG F1-Score: {metrics['dfg_f1']:.3f}")
        print(f"  - START/END validity: {metrics['both_validity']:.3f}")
        print(f"  - Overall quality score: {metrics['overall_score']:.3f} (lower is better)")
        
        print(f"\nFiles created:")
        print(f"  - {preprocessed_path}")
        print(f"  - {synthetic_traces_path}")
        print(f"  - {csv_path}")
        print(f"  - {metrics_path}")
        if save_checkpoints:
            print(f"  - {checkpoints_dir}/ (model checkpoints)")
        
        print("\n" + "="*80)
    
    # Return results
    results = {
        'gan': gan,
        'data': data,
        'synthetic_traces': synthetic_traces,
        'metrics': metrics,
        'output_dir': output_dir
    }
    
    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    """
    Command line usage:
    python train_gan_from_xes.py <path_to_xes_file> [options]
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Process Mining GAN from XES file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='../output',
                       help='Output directory for results')
    parser.add_argument('--min-trace-length', type=int, default=2,
                       help='Minimum trace length')
    parser.add_argument('--max-trace-length', type=int, default=12,
                       help='Maximum trace length')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--num-synthetic', type=int, default=1000,
                       help='Number of synthetic traces to generate')
    parser.add_argument('--lstm-units', type=int, default=256,
                       help='LSTM hidden units')
    parser.add_argument('--no-verbose', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    xes_file = '../data/helpdesk_parsed.xes'
    # Check if XES file exists
    if not os.path.exists(xes_file):
        print(f"Error: XES file not found: {xes_file}")
        print("\nUsage:")
        print("  python train_gan_from_xes.py <xes_file> [options]")
        print("\nExample:")
        print("  python train_gan_from_xes.py my_process_log.xes --epochs 50")
        print("\nFor all options:")
        print("  python train_gan_from_xes.py --help")
        sys.exit(1)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train GAN
    results = train_gan_from_xes(
        xes_filepath=xes_file,
        output_dir=args.output_dir,
        min_trace_length=args.min_trace_length,
        max_trace_length=args.max_trace_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_synthetic=args.num_synthetic,
        generator_lstm_units=args.lstm_units,
        discriminator_lstm_units=args.lstm_units,
        verbose=not args.no_verbose
    )
    
    print("\n✓ Process complete!")
    print(f"✓ Check {args.output_dir}/ for all outputs")