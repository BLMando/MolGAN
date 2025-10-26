"""
Event Log Data Augmentation Script for ProcessGAN

This script loads the best trained model from checkpoints and performs
data augmentation by generating synthetic traces that expand the original
event log while preserving its process patterns.

Data Augmentation Features:
- Generates synthetic traces similar to original data
- Combines original + synthetic traces into augmented dataset
- Preserves original trace patterns and process behavior
- Increases dataset size for better model training/analysis

Usage:
    # Augment original log with 1000 synthetic traces
    python generate_event_log.py --checkpoint-dir checkpoints/process_gan/best --augment-samples 1000
    
    # Augment and save to XES file
    python generate_event_log.py --checkpoint-dir checkpoints/process_gan/best --augment-samples 500 --output-file augmented_log.xes
    
    # Augment with custom temperature for sampling
    python generate_event_log.py --checkpoint-dir checkpoints/process_gan/best --augment-samples 1000 --temperature 0.3
"""

import numpy as np
import tensorflow as tf
import argparse
import os
import json
from datetime import datetime
from pathlib import Path

# Import ProcessGAN components
from utils.process_dataset import ProcessDataset
from utils.process_metrics import ProcessRewardFunction, ProcessMetrics
from models.process_gan import ProcessGAN, matrices_to_traces


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {msg}')


def load_model_from_checkpoint(checkpoint_dir, dataset):
    """
    Load ProcessGAN model from checkpoint directory
    
    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., 'checkpoints/process_gan/best')
        dataset: ProcessDataset instance with loaded data
        
    Returns:
        model: Loaded ProcessGAN model
        info: Dictionary with checkpoint metadata
    """
    log(f"Loading model from checkpoint: {checkpoint_dir}")
    
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Load checkpoint info
    info_file = os.path.join(checkpoint_dir, 'info.txt')
    info = {}
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    info[key.strip()] = value.strip()
        log(f"Checkpoint info: {info}")
    else:
        log("No info.txt found in checkpoint directory", level='WARNING')
    
    # Create model with same architecture as training
    model = ProcessGAN(
        max_activities=dataset.max_activities,
        flow_types=dataset.flow_num_types,
        activity_types=dataset.activity_num_types,
        embedding_dim=32,  # Default from trainer config
        decoder_units=(64, 128, 128),
        discriminator_units=(64, 64),
        mlp_units=64,
        dropout_rate=0.2,
        enforce_start=True
    )
    
    # Build model by calling with sample input
    sample_z = tf.random.normal((1, 32))
    sample_adj, sample_nodes = model.generator(sample_z, training=False)
    _ = model.discriminator(sample_adj, sample_nodes, training=False)
    _ = model.value_network(sample_adj, sample_nodes, training=False)
    
    # Load weights
    generator_path = os.path.join(checkpoint_dir, 'generator')
    discriminator_path = os.path.join(checkpoint_dir, 'discriminator')
    value_network_path = os.path.join(checkpoint_dir, 'value_network')
    
    # Check if checkpoint files exist (TensorFlow creates .data-00000-of-00001 and .index files)
    def check_checkpoint_exists(base_path):
        """Check if TensorFlow checkpoint files exist for given base path"""
        data_file = base_path + '.data-00000-of-00001'
        index_file = base_path + '.index'
        return os.path.exists(data_file) and os.path.exists(index_file)
    
    generator_exists = check_checkpoint_exists(generator_path)
    discriminator_exists = check_checkpoint_exists(discriminator_path)
    value_network_exists = check_checkpoint_exists(value_network_path)
    
    if generator_exists:
        model.generator.load_weights(generator_path)
        log("✓ Generator weights loaded")
    else:
        raise FileNotFoundError(f"Generator weights not found: {generator_path}")
    
    if discriminator_exists:
        model.discriminator.load_weights(discriminator_path)
        log("✓ Discriminator weights loaded")
    else:
        raise FileNotFoundError(f"Discriminator weights not found: {discriminator_path}")
    
    if value_network_exists:
        model.value_network.load_weights(value_network_path)
        log("✓ Value network weights loaded")
    else:
        log("Value network weights not found, continuing without RL", level='WARNING')
    
    log("Model loaded successfully!")
    return model, info


def generate_traces(model, dataset, n_samples, temperature=0.5, batch_size=100):
    """
    Generate traces from the model
    
    Args:
        model: Loaded ProcessGAN model
        dataset: ProcessDataset instance
        n_samples: Number of traces to generate
        temperature: Gumbel-Softmax temperature for sampling
        batch_size: Batch size for generation
        
    Returns:
        traces: List of generated traces
    """
    log(f"Generating {n_samples} traces with temperature {temperature}")
    
    all_traces = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        
        # Generate latent vectors
        z = model.sample_z(current_batch_size)
        
        # Generate adjacency matrices and node features
        edges, nodes = model.generator(z, training=False, temperature=temperature)
        
        # Convert to numpy and argmax for discrete sampling
        edges_np = edges.numpy()
        nodes_np = nodes.numpy()
        
        edges_indices = np.argmax(edges_np, axis=-1)
        nodes_indices = np.argmax(nodes_np, axis=-1)
        
        # Convert to traces
        batch_traces = matrices_to_traces(edges_indices, nodes_np, dataset)
        all_traces.extend(batch_traces)
        
        if (batch_idx + 1) % 10 == 0:
            log(f"Generated {len(all_traces)}/{n_samples} traces")
    
    log(f"Generation complete: {len(all_traces)} traces")
    return all_traces


def evaluate_generated_traces(traces, dataset, reward_function=None):
    """
    Evaluate quality of generated traces
    
    Args:
        traces: List of generated traces
        dataset: ProcessDataset instance
        reward_function: Optional reward function for evaluation
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    log("Evaluating generated traces...")
    
    # Basic process metrics
    valid_traces = ProcessMetrics.valid_traces(traces)
    unique_traces = ProcessMetrics.unique_traces(traces)
    novel_traces = ProcessMetrics.novel_traces(traces, dataset.data)
    
    metrics = {
        'total_generated': len(traces),
        'valid_count': len(valid_traces),
        'valid_rate': len(valid_traces) / len(traces),
        'unique_count': len(unique_traces),
        'unique_rate': len(unique_traces) / len(traces),
        'novel_count': len(novel_traces),
        'novel_rate': len(novel_traces) / len(traces),
    }
    
    # Reward-based metrics if reward function provided
    if reward_function is not None:
        log("Computing reward-based metrics...")
        reward_metrics = reward_function.evaluate_batch(traces)
        metrics.update(reward_metrics)
    
    # Trace length statistics
    trace_lengths = [len(trace) for trace in traces]
    metrics['avg_length'] = np.mean(trace_lengths)
    metrics['min_length'] = np.min(trace_lengths)
    metrics['max_length'] = np.max(trace_lengths)
    metrics['std_length'] = np.std(trace_lengths)
    
    return metrics


def create_augmented_dataset(original_traces, synthetic_traces, activity_decoder):
    """
    Create augmented dataset by combining original and synthetic traces
    
    Args:
        original_traces: List of original traces from dataset
        synthetic_traces: List of generated synthetic traces
        activity_decoder: Dictionary mapping activity indices to names
        
    Returns:
        augmented_traces: Combined list of original + synthetic traces
        augmentation_stats: Statistics about the augmentation
    """
    log(f"Creating augmented dataset...")
    log(f"  Original traces: {len(original_traces)}")
    log(f"  Synthetic traces: {len(synthetic_traces)}")
    
    # Combine traces
    augmented_traces = original_traces + synthetic_traces
    
    # Compute augmentation statistics
    augmentation_stats = {
        'original_count': len(original_traces),
        'synthetic_count': len(synthetic_traces),
        'augmented_count': len(augmented_traces),
        'augmentation_ratio': len(synthetic_traces) / len(original_traces),
        'total_increase': len(synthetic_traces),
        'percentage_increase': (len(synthetic_traces) / len(original_traces)) * 100
    }
    
    log(f"  Augmented dataset: {len(augmented_traces)} traces")
    log(f"  Augmentation ratio: {augmentation_stats['augmentation_ratio']:.2f}x")
    log(f"  Percentage increase: {augmentation_stats['percentage_increase']:.1f}%")
    
    return augmented_traces, augmentation_stats


def save_augmented_dataset_to_xes(augmented_traces, output_file, activity_decoder, original_count):
    """
    Save augmented dataset to XES format with proper case IDs
    
    Args:
        augmented_traces: Combined list of original + synthetic traces
        output_file: Output XES file path
        activity_decoder: Dictionary mapping activity indices to names
        original_count: Number of original traces (for case ID separation)
    """
    log(f"Saving augmented dataset to XES file: {output_file}")
    
    import pm4py
    from pm4py.objects.log.obj import EventLog, Trace, Event
    
    # Create event log
    event_log = EventLog()
    
    for i, trace in enumerate(augmented_traces):
        # Create trace with appropriate case ID
        if i < original_count:
            case_id = f'Original_Case_{i+1}'
        else:
            case_id = f'Augmented_Case_{i-original_count+1}'
        
        pm4py_trace = Trace()
        pm4py_trace.attributes['concept:name'] = case_id
        
        # Add events to trace
        for j, activity_idx in enumerate(trace):
            if activity_idx in activity_decoder:
                activity_name = activity_decoder[activity_idx]
                
                # Create event
                event = Event()
                event['concept:name'] = activity_name
                event['time:timestamp'] = datetime.now()
                
                pm4py_trace.append(event)
        
        event_log.append(pm4py_trace)
    
    # Write XES file
    pm4py.write_xes(event_log, output_file)
    log(f"✓ Saved {len(augmented_traces)} traces to {output_file}")
    log(f"  - Original traces: {original_count}")
    log(f"  - Augmented traces: {len(augmented_traces) - original_count}")


def save_augmented_dataset_to_csv(augmented_traces, output_file, activity_decoder, original_count):
    """
    Save augmented dataset to CSV format with proper case IDs
    
    Args:
        augmented_traces: Combined list of original + synthetic traces
        output_file: Output CSV file path
        activity_decoder: Dictionary mapping activity indices to names
        original_count: Number of original traces (for case ID separation)
    """
    log(f"Saving augmented dataset to CSV file: {output_file}")
    
    import pandas as pd
    
    data = []
    for i, trace in enumerate(augmented_traces):
        # Create appropriate case ID
        if i < original_count:
            case_id = f'Original_Case_{i+1}'
        else:
            case_id = f'Augmented_Case_{i-original_count+1}'
        
        for j, activity_idx in enumerate(trace):
            if activity_idx in activity_decoder:
                activity_name = activity_decoder[activity_idx]
                data.append({
                    'case_id': case_id,
                    'activity': activity_name,
                    'timestamp': datetime.now(),
                    'position': j,
                    'trace_type': 'original' if i < original_count else 'augmented'
                })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    log(f"✓ Saved {len(augmented_traces)} traces to {output_file}")
    log(f"  - Original traces: {original_count}")
    log(f"  - Augmented traces: {len(augmented_traces) - original_count}")


def print_augmentation_summary(metrics, augmentation_stats, checkpoint_info):
    """Print summary of data augmentation results"""
    print("\n" + "=" * 80)
    print("EVENT LOG DATA AUGMENTATION SUMMARY")
    print("=" * 80)
    
    if checkpoint_info:
        print(f"\nModel Information:")
        for key, value in checkpoint_info.items():
            print(f"  {key}: {value}")
    
    print(f"\nData Augmentation Results:")
    print(f"  Original Dataset:     {augmentation_stats['original_count']} traces")
    print(f"  Synthetic Traces:     {augmentation_stats['synthetic_count']} traces")
    print(f"  Augmented Dataset:    {augmentation_stats['augmented_count']} traces")
    print(f"  Augmentation Ratio:   {augmentation_stats['augmentation_ratio']:.2f}x")
    print(f"  Percentage Increase:  {augmentation_stats['percentage_increase']:.1f}%")
    
    print(f"\nSynthetic Trace Quality:")
    print(f"  Valid Traces:         {metrics['valid_count']} ({metrics['valid_rate']:.1%})")
    print(f"  Unique Traces:        {metrics['unique_count']} ({metrics['unique_rate']:.1%})")
    print(f"  Novel Traces:         {metrics['novel_count']} ({metrics['novel_rate']:.1%})")
    
    print(f"\nTrace Length Statistics:")
    print(f"  Average Length:       {metrics['avg_length']:.1f}")
    print(f"  Min Length:           {metrics['min_length']}")
    print(f"  Max Length:           {metrics['max_length']}")
    print(f"  Std Deviation:        {metrics['std_length']:.1f}")
    
    if 'reward_mean' in metrics:
        print(f"\nQuality Metrics:")
        print(f"  Reward Score:        {metrics['reward_mean']:.3f} ± {metrics['reward_std']:.3f}")
        print(f"  Validity Score:      {metrics['validity_mean']:.3f}")
        print(f"  Fitness Score:       {metrics['fitness_mean']:.3f}")
        print(f"  Conformance Score:   {metrics['conformance_mean']:.3f}")
        print(f"  Diversity Score:     {metrics['diversity_mean']:.3f}")
        print(f"  High Quality Rate:   {metrics['high_quality_rate']:.1%}")
    
    print("=" * 80 + "\n")


def main():
    """Main data augmentation function"""
    
    log("=" * 80)
    log("ProcessGAN Event Log Data Augmentation")
    log("=" * 80)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Augment event logs with synthetic traces from trained ProcessGAN model')
    
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Path to checkpoint directory (e.g., checkpoints/process_gan/best)')
    parser.add_argument('--data-file', type=str, default='data/helpdesk_parsed.xes',
                        help='Path to original data file for dataset loading')
    parser.add_argument('--augment-samples', type=int, default=1000,
                        help='Number of synthetic traces to add for augmentation (default: 1000)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Gumbel-Softmax temperature for sampling (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for generation (default: 100)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path for augmented dataset (XES or CSV format)')
    parser.add_argument('--output-format', type=str, default='xes',
                        choices=['xes', 'csv'],
                        help='Output format (default: xes)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate generated traces with reward function')
    parser.add_argument('--reference-model', type=str, default=None,
                        help='Path to reference Petri net for evaluation')
    
    args = parser.parse_args()
    
    # Load dataset (needed for model architecture and trace decoding)
    log(f"Loading dataset from: {args.data_file}")
    dataset = ProcessDataset(max_activities=10)  # Default from trainer config
    
    if os.path.exists(args.data_file):
        if args.data_file.endswith('.xes'):
            dataset.load_from_xes(args.data_file, validation=0.15, test=0.1)
        elif args.data_file.endswith('.pnml'):
            dataset.load_from_petri_net(args.data_file, validation=0.15, test=0.1)
        else:
            log(f"Unsupported file format: {args.data_file}", level='ERROR')
            return
    else:
        log(f"Data file not found: {args.data_file}", level='ERROR')
        return
    
    log(f"Dataset loaded: {len(dataset.data)} traces")
    
    # Load model from checkpoint
    model, checkpoint_info = load_model_from_checkpoint(args.checkpoint_dir, dataset)
    
    # Create reward function for evaluation (if requested)
    reward_function = None
    if args.evaluate:
        log("Creating reward function for evaluation...")
        
        # Try to find reference model
        reference_model = None
        if args.reference_model and os.path.exists(args.reference_model):
            reference_model = ProcessRewardFunction.load_petri_net_model(args.reference_model)
            log(f"Using reference model: {args.reference_model}")
        elif args.data_file.endswith('.xes'):
            # Try to find companion PNML file
            base_path = os.path.splitext(args.data_file)[0]
            candidate_paths = [
                base_path.replace('_parsed', '_parsed_net') + '.pnml',
                base_path.capitalize().replace('_parsed', '_parsed_net') + '.pnml',
                base_path + '_net.pnml',
                base_path + '.pnml',
            ]
            
            for candidate in candidate_paths:
                if os.path.exists(candidate):
                    reference_model = ProcessRewardFunction.load_petri_net_model(candidate)
                    log(f"✓ Found companion Petri net: {candidate}")
                    break
        
        # Auto-detect START/END activities from dataset
        start_activity = None
        end_activity = None
        if dataset.data:
            from collections import Counter
            first_acts = [t[0] for t in dataset.data if len(t) > 0]
            last_acts = [t[-1] for t in dataset.data if len(t) > 0]
            if first_acts:
                start_activity = Counter(first_acts).most_common(1)[0][0]
            if last_acts:
                end_activity = Counter(last_acts).most_common(1)[0][0]

        log(f"Auto-detected START activity: {start_activity}")
        log(f"Auto-detected END activity: {end_activity}")

        reward_function = ProcessRewardFunction(
            reference_model=reference_model,
            training_traces=dataset.data,
            weights={'validity': 0.20, 'fitness': 0.40, 'conformance': 0.25, 'diversity': 0.15},
            start_activity=start_activity,  # NEW: Pass auto-detected START
            end_activity=end_activity        # NEW: Pass auto-detected END
        )
    
    # Generate synthetic traces for augmentation
    synthetic_traces = generate_traces(
        model, dataset, args.augment_samples, args.temperature, args.batch_size)
    
    # Evaluate synthetic traces
    metrics = evaluate_generated_traces(synthetic_traces, dataset, reward_function)
    
    # Create augmented dataset
    augmented_traces, augmentation_stats = create_augmented_dataset(
        dataset.data, synthetic_traces, dataset.activity_decoder)
    
    # Print summary
    print_augmentation_summary(metrics, augmentation_stats, checkpoint_info)
    
    # Save augmented dataset if output file specified
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if args.output_format == 'xes':
            save_augmented_dataset_to_xes(
                augmented_traces, args.output_file, dataset.activity_decoder, len(dataset.data))
        elif args.output_format == 'csv':
            save_augmented_dataset_to_csv(
                augmented_traces, args.output_file, dataset.activity_decoder, len(dataset.data))
    
    # Print sample traces
    print("Sample Original Traces:")
    for i, trace in enumerate(dataset.data[:5]):
        trace_str = ' → '.join([dataset.activity_decoder.get(idx, f'UNK_{idx}') for idx in trace])
        print(f"  {i+1:2d}. {trace_str}")
    
    print("\nSample Synthetic Traces:")
    for i, trace in enumerate(synthetic_traces[:5]):
        trace_str = ' → '.join([dataset.activity_decoder.get(idx, f'UNK_{idx}') for idx in trace])
        print(f"  {i+1:2d}. {trace_str}")
    
    log("Event log data augmentation completed successfully!")


if __name__ == '__main__':
    main()
