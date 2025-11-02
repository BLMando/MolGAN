"""
Event Log Data Augmentation Script for ProcessGAN (Refactored)

Uses modular export components for clean code organization.

Usage:
    python evaluate.py --checkpoint-dir checkpoints/process_gan/best --augment-samples 1000
"""

import numpy as np
import argparse
import os
from datetime import datetime
from collections import Counter

# Import refactored export modules
from export import CheckpointLoader, XESExporter

# Import ProcessGAN components
from utils.dataset import ProcessDataset
from utils.metrics import ProcessRewardFunction
from metrics.static_metrics import ProcessMetrics
from models.process_gan import matrices_to_traces


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {msg}')


def generate_traces(model, dataset, n_samples, temperature=0.5, batch_size=100):
    """
    Generate traces from the model

    Args:
        model: Loaded ProcessGAN model
        dataset: ProcessDataset instance
        n_samples: Number of traces to generate
        temperature: Gumbel-Softmax temperature
        batch_size: Batch size for generation

    Returns:
        List of generated traces (as activity name lists)
    """
    log(f"Generating {n_samples} traces (temperature={temperature})...")

    all_traces = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for i in range(n_batches):
        current_batch_size = min(batch_size, n_samples - len(all_traces))

        # Generate batch
        adj_batch, nodes_batch = model.generate(
            current_batch_size, temperature=temperature)

        # Convert to traces using dataset decoder
        batch_traces = matrices_to_traces(
            adj_batch.numpy(), nodes_batch.numpy(), dataset)

        all_traces.extend(batch_traces)

        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            log(f"  Generated {len(all_traces)}/{n_samples} traces")

    log(f"✓ Generation complete: {len(all_traces)} traces")
    return all_traces


def evaluate_generated_traces(traces, dataset, reward_function=None):
    """
    Evaluate generated traces

    Args:
        traces: List of generated traces
        dataset: ProcessDataset instance
        reward_function: Optional ProcessRewardFunction for quality metrics

    Returns:
        Dictionary with evaluation metrics
    """
    log(f"Evaluating {len(traces)} traces...")

    # Basic metrics using ProcessMetrics
    valid_traces = ProcessMetrics.valid_traces(
        traces,
        start_activity=dataset.activity_decoder[0] if 0 in dataset.activity_decoder else 'Start',
        end_activity=list(dataset.activity_decoder.values()
                          )[-1] if dataset.activity_decoder else 'End'
    )
    unique_traces = ProcessMetrics.unique_traces(traces)
    novel_traces = ProcessMetrics.novel_traces(traces, dataset.data)

    metrics = {
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


def create_augmentation_stats(original_count, synthetic_count):
    """
    Compute augmentation statistics

    Args:
        original_count: Number of original traces
        synthetic_count: Number of synthetic traces

    Returns:
        Dictionary with augmentation statistics
    """
    return {
        'original_count': original_count,
        'synthetic_count': synthetic_count,
        'augmented_count': original_count + synthetic_count,
        'augmentation_ratio': synthetic_count / original_count,
        'total_increase': synthetic_count,
        'percentage_increase': (synthetic_count / original_count) * 100
    }


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
    print(
        f"  Original Dataset:     {augmentation_stats['original_count']} traces")
    print(
        f"  Synthetic Traces:     {augmentation_stats['synthetic_count']} traces")
    print(
        f"  Augmented Dataset:    {augmentation_stats['augmented_count']} traces")
    print(
        f"  Augmentation Ratio:   {augmentation_stats['augmentation_ratio']:.2f}x")
    print(
        f"  Percentage Increase:  {augmentation_stats['percentage_increase']:.1f}%")

    print(f"\nSynthetic Trace Quality:")
    print(
        f"  Valid Traces:         {metrics['valid_count']} ({metrics['valid_rate']:.1%})")
    print(
        f"  Unique Traces:        {metrics['unique_count']} ({metrics['unique_rate']:.1%})")
    print(
        f"  Novel Traces:         {metrics['novel_count']} ({metrics['novel_rate']:.1%})")

    print(f"\nTrace Length Statistics:")
    print(f"  Average Length:       {metrics['avg_length']:.1f}")
    print(f"  Min Length:           {metrics['min_length']}")
    print(f"  Max Length:           {metrics['max_length']}")
    print(f"  Std Deviation:        {metrics['std_length']:.1f}")

    if 'reward_mean' in metrics:
        print(f"\nQuality Metrics:")
        print(
            f"  Reward Score:        {metrics['reward_mean']:.3f} ± {metrics['reward_std']:.3f}")
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
    parser = argparse.ArgumentParser(
        description='Augment event logs with synthetic traces from trained ProcessGAN model')

    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Path to checkpoint directory (e.g., checkpoints/process_gan/best)')
    parser.add_argument('--data-file', type=str, default='data/helpdesk_parsed.xes',
                        help='Path to original data file for dataset loading')
    parser.add_argument('--augment-samples', type=int, default=1000,
                        help='Number of synthetic traces to add (default: 1000)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Gumbel-Softmax temperature (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for generation (default: 100)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path for augmented dataset (XES or CSV)')
    parser.add_argument('--output-format', type=str, default='xes',
                        choices=['xes', 'csv'],
                        help='Output format (default: xes)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate generated traces with reward function')
    parser.add_argument('--reference-model', type=str, default=None,
                        help='Path to reference Petri net for evaluation')

    args = parser.parse_args()

    # Load dataset
    log(f"Loading dataset from: {args.data_file}")
    dataset = ProcessDataset(max_activities=10)

    if os.path.exists(args.data_file):
        if args.data_file.endswith('.xes'):
            dataset.load_from_xes(args.data_file, validation=0.15, test=0.1)
        else:
            log(f"Unsupported file format: {args.data_file}", level='ERROR')
            return
    else:
        log(f"Data file not found: {args.data_file}", level='ERROR')
        return

    log(f"Dataset loaded: {len(dataset.data)} traces")

    # Load model from checkpoint using CheckpointLoader
    model, checkpoint_info = CheckpointLoader.load(
        args.checkpoint_dir, dataset)

    # Create reward function for evaluation (if requested)
    reward_function = None
    if args.evaluate:
        log("Creating reward function for evaluation...")

        # Try to find reference model
        reference_model = None
        if args.reference_model and os.path.exists(args.reference_model):
            reference_model = ProcessRewardFunction.load_petri_net_model(
                args.reference_model)
            log(f"Using reference model: {args.reference_model}")

        # Auto-detect START/END activities
        start_activity = None
        end_activity = None
        if dataset.data:
            first_acts = [t[0] for t in dataset.data if len(t) > 0]
            last_acts = [t[-1] for t in dataset.data if len(t) > 0]
            if first_acts:
                start_activity = Counter(first_acts).most_common(1)[0][0]
            if last_acts:
                end_activity = Counter(last_acts).most_common(1)[0][0]

        log(f"Auto-detected START: {start_activity}, END: {end_activity}")

        reward_function = ProcessRewardFunction(
            reference_model=reference_model,
            training_traces=dataset.data,
            weights={'validity': 0.20, 'fitness': 0.40,
                     'conformance': 0.25, 'diversity': 0.15},
            start_activity=start_activity,
            end_activity=end_activity
        )

    # Generate synthetic traces
    synthetic_traces = generate_traces(
        model, dataset, args.augment_samples, args.temperature, args.batch_size)

    # Evaluate synthetic traces
    metrics = evaluate_generated_traces(
        synthetic_traces, dataset, reward_function)

    # Compute augmentation statistics
    augmentation_stats = create_augmentation_stats(
        len(dataset.data), len(synthetic_traces))

    log(f"Augmentation ratio: {augmentation_stats['augmentation_ratio']:.2f}x")
    log(
        f"Percentage increase: {augmentation_stats['percentage_increase']:.1f}%")

    # Print summary
    print_augmentation_summary(metrics, augmentation_stats, checkpoint_info)

    # Save augmented dataset if output file specified
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.output_format == 'xes':
            XESExporter.save_augmented(
                dataset.data, synthetic_traces, args.output_file)

    # Print sample traces
    print("Sample Original Traces:")
    for i, trace in enumerate(dataset.data[:5]):
        trace_str = ' → '.join(trace)
        print(f"  {i+1:2d}. {trace_str}")

    print("\nSample Synthetic Traces:")
    for i, trace in enumerate(synthetic_traces[:5]):
        trace_str = ' → '.join(trace)
        print(f"  {i+1:2d}. {trace_str}")

    log("Event log data augmentation completed successfully!")


if __name__ == '__main__':
    main()
