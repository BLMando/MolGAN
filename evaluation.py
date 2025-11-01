"""
Evaluation Utilities for Process Mining GAN
Assess quality of generated synthetic traces
"""

import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# BASIC TRACE METRICS
# ============================================================================

def compute_trace_diversity(traces):
    """
    Compute diversity of traces (ratio of unique traces)
    
    Parameters:
    -----------
    traces: list of lists or np.array
        Generated traces
    
    Returns:
    --------
    diversity: float
        Ratio of unique traces to total traces
    """
    if isinstance(traces, np.ndarray):
        traces = [tuple(trace) for trace in traces]
    else:
        traces = [tuple(trace) for trace in traces]
    
    unique_traces = len(set(traces))
    total_traces = len(traces)
    
    diversity = unique_traces / total_traces
    
    return diversity


def compute_validity_rate(generated_traces, real_traces):
    """
    Compute what percentage of generated traces also appear in real data
    
    Parameters:
    -----------
    generated_traces: list of lists
        Generated traces
    real_traces: list of lists
        Real traces from training data
    
    Returns:
    --------
    validity: float
        Proportion of generated traces that are "valid" (appear in real data)
    """
    real_set = set(tuple(trace) for trace in real_traces)
    generated_set = [tuple(trace) for trace in generated_traces]
    
    valid_count = sum(1 for trace in generated_set if trace in real_set)
    validity = valid_count / len(generated_traces)
    
    return validity


def compute_novelty_rate(generated_traces, real_traces):
    """
    Compute what percentage of generated traces are novel (not in real data)
    
    Parameters:
    -----------
    generated_traces: list of lists
        Generated traces
    real_traces: list of lists
        Real traces from training data
    
    Returns:
    --------
    novelty: float
        Proportion of generated traces that are novel
    """
    real_set = set(tuple(trace) for trace in real_traces)
    generated_set = [tuple(trace) for trace in generated_traces]
    
    novel_count = sum(1 for trace in generated_set if trace not in real_set)
    novelty = novel_count / len(generated_traces)
    
    return novelty


# ============================================================================
# DISTRIBUTION METRICS
# ============================================================================

def compute_activity_distribution_distance(generated_traces, real_traces, 
                                          activity_to_idx):
    """
    Compute distance between activity distributions
    
    Parameters:
    -----------
    generated_traces: list of lists
        Generated traces
    real_traces: list of lists
        Real traces
    activity_to_idx: dict
        Activity to index mapping
    
    Returns:
    --------
    js_divergence: float
        Jensen-Shannon divergence between distributions
    """
    num_activities = len(activity_to_idx)
    
    # Compute real distribution
    real_counts = np.zeros(num_activities)
    for trace in real_traces:
        for activity in trace:
            if activity in activity_to_idx:
                real_counts[activity_to_idx[activity]] += 1
    real_dist = real_counts / real_counts.sum()
    
    # Compute generated distribution
    gen_counts = np.zeros(num_activities)
    for trace in generated_traces:
        if isinstance(trace[0], str):
            # String activities
            for activity in trace:
                if activity in activity_to_idx:
                    gen_counts[activity_to_idx[activity]] += 1
        else:
            # Already indices
            for idx in trace:
                if 0 <= idx < num_activities:
                    gen_counts[idx] += 1
    gen_dist = gen_counts / gen_counts.sum()
    
    # Compute Jensen-Shannon divergence
    js_div = jensenshannon(real_dist, gen_dist)
    
    return js_div


def compute_trace_length_distance(generated_traces, real_traces):
    """
    Compute distance between trace length distributions
    
    Parameters:
    -----------
    generated_traces: list of lists
        Generated traces
    real_traces: list of lists
        Real traces
    
    Returns:
    --------
    distance: float
        Wasserstein distance between length distributions
    """
    real_lengths = [len(trace) for trace in real_traces]
    gen_lengths = [len(trace) for trace in generated_traces]
    
    # Compute Wasserstein distance
    distance = wasserstein_distance(real_lengths, gen_lengths)
    
    return distance


def compute_bigram_distribution_distance(generated_traces, real_traces):
    """
    Compute distance between bigram (directly-follows) distributions
    
    Parameters:
    -----------
    generated_traces: list of lists
        Generated traces
    real_traces: list of lists
        Real traces
    
    Returns:
    --------
    js_divergence: float
        Jensen-Shannon divergence between bigram distributions
    """
    # Extract all bigrams
    def get_bigrams(traces):
        bigrams = []
        for trace in traces:
            for i in range(len(trace) - 1):
                bigrams.append((trace[i], trace[i + 1]))
        return bigrams
    
    real_bigrams = get_bigrams(real_traces)
    gen_bigrams = get_bigrams(generated_traces)
    
    # Get all unique bigrams
    all_bigrams = list(set(real_bigrams + gen_bigrams))
    bigram_to_idx = {bg: idx for idx, bg in enumerate(all_bigrams)}
    
    # Compute distributions
    real_dist = np.zeros(len(all_bigrams))
    for bigram in real_bigrams:
        real_dist[bigram_to_idx[bigram]] += 1
    real_dist = real_dist / real_dist.sum() if real_dist.sum() > 0 else real_dist
    
    gen_dist = np.zeros(len(all_bigrams))
    for bigram in gen_bigrams:
        gen_dist[bigram_to_idx[bigram]] += 1
    gen_dist = gen_dist / gen_dist.sum() if gen_dist.sum() > 0 else gen_dist
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    real_dist += epsilon
    gen_dist += epsilon
    real_dist /= real_dist.sum()
    gen_dist /= gen_dist.sum()
    
    # Compute Jensen-Shannon divergence
    js_div = jensenshannon(real_dist, gen_dist)
    
    return js_div


# ============================================================================
# PROCESS MINING SPECIFIC METRICS
# ============================================================================

def compute_start_end_validity(generated_traces, start_token='<START>', 
                               end_token='<END>'):
    """
    Check if traces properly start with START and end with END
    
    Parameters:
    -----------
    generated_traces: list of lists
        Generated traces
    start_token: str or int
        START token
    end_token: str or int
        END token
    
    Returns:
    --------
    metrics: dict
        Dictionary with start_validity, end_validity, both_validity
    """
    start_valid = 0
    end_valid = 0
    both_valid = 0
    
    for trace in generated_traces:
        if len(trace) == 0:
            continue
        
        has_start = trace[0] == start_token
        has_end = trace[-1] == end_token
        
        if has_start:
            start_valid += 1
        if has_end:
            end_valid += 1
        if has_start and has_end:
            both_valid += 1
    
    n = len(generated_traces)
    
    return {
        'start_validity': start_valid / n if n > 0 else 0,
        'end_validity': end_valid / n if n > 0 else 0,
        'both_validity': both_valid / n if n > 0 else 0
    }


def compute_dfg_similarity(generated_traces, real_traces):
    """
    Compute similarity between DFGs of generated and real traces
    
    Parameters:
    -----------
    generated_traces: list of lists
        Generated traces
    real_traces: list of lists
        Real traces
    
    Returns:
    --------
    metrics: dict
        Dictionary with precision, recall, f1_score
    """
    def build_dfg(traces):
        dfg = set()
        for trace in traces:
            for i in range(len(trace) - 1):
                dfg.add((trace[i], trace[i + 1]))
        return dfg
    
    real_dfg = build_dfg(real_traces)
    gen_dfg = build_dfg(generated_traces)
    
    # Compute metrics
    intersection = real_dfg & gen_dfg
    
    precision = len(intersection) / len(gen_dfg) if len(gen_dfg) > 0 else 0
    recall = len(intersection) / len(real_dfg) if len(real_dfg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_real_edges': len(real_dfg),
        'num_gen_edges': len(gen_dfg),
        'num_common_edges': len(intersection)
    }


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_generated_traces(generated_traces, real_traces, 
                             activity_to_idx, idx_to_activity,
                             start_token='<START>', end_token='<END>',
                             verbose=True):
    """
    Comprehensive evaluation of generated traces
    
    Parameters:
    -----------
    generated_traces: list of lists or np.array
        Generated traces (can be indices or activity names)
    real_traces: list of lists
        Real traces from training/test data
    activity_to_idx: dict
        Activity to index mapping
    idx_to_activity: dict
        Index to activity mapping
    start_token: str or int
        START token
    end_token: str or int
        END token
    verbose: bool
        Print results
    
    Returns:
    --------
    metrics: dict
        Dictionary of all evaluation metrics
    """
    # Convert indices to activities if needed
    if isinstance(generated_traces, np.ndarray):
        generated_traces = [
            [idx_to_activity[idx] for idx in trace if idx in idx_to_activity]
            for trace in generated_traces
        ]
    elif len(generated_traces) > 0 and isinstance(generated_traces[0][0], (int, np.integer)):
        generated_traces = [
            [idx_to_activity[idx] for idx in trace if idx in idx_to_activity]
            for trace in generated_traces
        ]
    
    metrics = {}
    
    # Basic metrics
    if verbose:
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
    
    metrics['diversity'] = compute_trace_diversity(generated_traces)
    metrics['validity'] = compute_validity_rate(generated_traces, real_traces)
    metrics['novelty'] = compute_novelty_rate(generated_traces, real_traces)
    
    if verbose:
        print("\n1. BASIC METRICS:")
        print(f"   Diversity (unique traces): {metrics['diversity']:.4f}")
        print(f"   Validity (in training set): {metrics['validity']:.4f}")
        print(f"   Novelty (not in training): {metrics['novelty']:.4f}")
    
    # Distribution metrics
    metrics['activity_distribution_distance'] = compute_activity_distribution_distance(
        generated_traces, real_traces, activity_to_idx
    )
    metrics['length_distribution_distance'] = compute_trace_length_distance(
        generated_traces, real_traces
    )
    metrics['bigram_distribution_distance'] = compute_bigram_distribution_distance(
        generated_traces, real_traces
    )
    
    if verbose:
        print("\n2. DISTRIBUTION SIMILARITY:")
        print(f"   Activity dist. (JS divergence): {metrics['activity_distribution_distance']:.4f}")
        print(f"   Length dist. (Wasserstein): {metrics['length_distribution_distance']:.4f}")
        print(f"   Bigram dist. (JS divergence): {metrics['bigram_distribution_distance']:.4f}")
    
    # Process-specific metrics
    start_end_metrics = compute_start_end_validity(generated_traces, start_token, end_token)
    metrics.update(start_end_metrics)
    
    dfg_metrics = compute_dfg_similarity(generated_traces, real_traces)
    metrics['dfg_precision'] = dfg_metrics['precision']
    metrics['dfg_recall'] = dfg_metrics['recall']
    metrics['dfg_f1'] = dfg_metrics['f1_score']
    
    if verbose:
        print("\n3. PROCESS MINING METRICS:")
        print(f"   START token validity: {metrics['start_validity']:.4f}")
        print(f"   END token validity: {metrics['end_validity']:.4f}")
        print(f"   Both START & END: {metrics['both_validity']:.4f}")
        print(f"\n   DFG Precision: {metrics['dfg_precision']:.4f}")
        print(f"   DFG Recall: {metrics['dfg_recall']:.4f}")
        print(f"   DFG F1-Score: {metrics['dfg_f1']:.4f}")
        print(f"   Real DFG edges: {dfg_metrics['num_real_edges']}")
        print(f"   Generated DFG edges: {dfg_metrics['num_gen_edges']}")
        print(f"   Common edges: {dfg_metrics['num_common_edges']}")
    
    # Summary score (lower is better)
    metrics['overall_score'] = (
        metrics['activity_distribution_distance'] +
        metrics['bigram_distribution_distance'] +
        (1 - metrics['dfg_f1']) +
        (1 - metrics['both_validity'])
    ) / 4
    
    if verbose:
        print("\n4. OVERALL QUALITY SCORE:")
        print(f"   Score (lower is better): {metrics['overall_score']:.4f}")
        print("=" * 60)
    
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_trace_length_comparison(generated_traces, real_traces, save_path=None):
    """
    Plot comparison of trace length distributions
    """
    real_lengths = [len(trace) for trace in real_traces]
    gen_lengths = [len(trace) for trace in generated_traces]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(real_lengths, bins=30, alpha=0.7, label='Real', color='blue', density=True)
    plt.hist(gen_lengths, bins=30, alpha=0.7, label='Generated', color='orange', density=True)
    plt.xlabel('Trace Length')
    plt.ylabel('Density')
    plt.title('Trace Length Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    data = [real_lengths, gen_lengths]
    plt.boxplot(data, labels=['Real', 'Generated'])
    plt.ylabel('Trace Length')
    plt.title('Trace Length Comparison')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_activity_distribution_comparison(generated_traces, real_traces, 
                                         activity_to_idx, idx_to_activity,
                                         top_n=15, save_path=None):
    """
    Plot comparison of activity frequency distributions
    """
    # Count activities
    def count_activities(traces):
        counts = Counter()
        for trace in traces:
            if isinstance(trace[0], str):
                counts.update(trace)
            else:
                counts.update([idx_to_activity[idx] for idx in trace if idx in idx_to_activity])
        return counts
    
    real_counts = count_activities(real_traces)
    gen_counts = count_activities(generated_traces)
    
    # Get top activities from real data
    top_activities = [act for act, _ in real_counts.most_common(top_n)]
    
    # Prepare data for plotting
    real_freqs = [real_counts[act] / sum(real_counts.values()) for act in top_activities]
    gen_freqs = [gen_counts[act] / sum(gen_counts.values()) for act in top_activities]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_activities))
    width = 0.35
    
    ax.bar(x - width/2, real_freqs, width, label='Real', alpha=0.8, color='blue')
    ax.bar(x + width/2, gen_freqs, width, label='Generated', alpha=0.8, color='orange')
    
    ax.set_xlabel('Activity')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Top {top_n} Activity Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(top_activities, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example evaluation
    """
    
    # Create example data
    real_traces = [
        ['<START>', 'A', 'B', 'C', '<END>'],
        ['<START>', 'A', 'C', '<END>'],
        ['<START>', 'B', 'C', '<END>'],
        ['<START>', 'A', 'B', 'D', '<END>'],
    ] * 10
    
    # Simulate generated traces (with some variations)
    generated_traces = [
        ['<START>', 'A', 'B', 'C', '<END>'],
        ['<START>', 'A', 'C', '<END>'],
        ['<START>', 'B', 'C', '<END>'],
        ['<START>', 'A', 'D', '<END>'],  # Novel variant
        ['<START>', 'B', 'D', 'C', '<END>'],  # Novel variant
    ] * 8
    
    # Create vocabulary
    all_activities = set()
    for trace in real_traces + generated_traces:
        all_activities.update(trace)
    activity_to_idx = {act: idx for idx, act in enumerate(sorted(all_activities))}
    idx_to_activity = {idx: act for act, idx in activity_to_idx.items()}
    
    # Evaluate
    print("\nRunning comprehensive evaluation...")
    metrics = evaluate_generated_traces(
        generated_traces,
        real_traces,
        activity_to_idx,
        idx_to_activity,
        verbose=True
    )
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_trace_length_comparison(generated_traces, real_traces)
    plot_activity_distribution_comparison(
        generated_traces, real_traces,
        activity_to_idx, idx_to_activity,
        top_n=5
    )