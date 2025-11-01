"""
XES File Loader for Process Mining GAN
Complete pipeline for loading and preprocessing XES event logs
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import pickle
import os


# ============================================================================
# XES FILE LOADING (using pm4py)
# ============================================================================

def load_xes_file(filepath, verbose=True):
    """
    Load XES file using pm4py
    
    Parameters:
    -----------
    filepath: str
        Path to XES file
    verbose: bool
        Print loading information
    
    Returns:
    --------
    log: EventLog
        PM4Py event log object
    df: pd.DataFrame
        Event log as dataframe
    """
    try:
        import pm4py
    except ImportError:
        raise ImportError(
            "pm4py is required to load XES files.\n"
            "Install with: pip install pm4py"
        )
    
    if verbose:
        print(f"Loading XES file: {filepath}")
    
    # Import XES file
    log = pm4py.read_xes(filepath)
    
    if verbose:
        print(f"  - Loaded {len(log)} cases")
        print(f"  - Total events: {sum(len(trace) for trace in log)}")
    
    # Convert to dataframe for easier manipulation
    df = pm4py.convert_to_dataframe(log)
    
    if verbose:
        print(f"  - Dataframe shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
    
    return log, df


# ============================================================================
# TRACE EXTRACTION FROM XES
# ============================================================================

def extract_traces_from_xes(log_or_df, 
                            case_col='case:concept:name',
                            activity_col='concept:name',
                            timestamp_col='time:timestamp',
                            resource_col='org:resource',
                            min_length=2,
                            max_length=None,
                            add_special_tokens=True,
                            verbose=True):
    """
    Extract activity sequences (traces) from XES log
    
    Parameters:
    -----------
    log_or_df: EventLog or pd.DataFrame
        PM4Py event log or dataframe
    case_col: str
        Column name for case ID
    activity_col: str
        Column name for activity
    timestamp_col: str
        Column name for timestamp
    resource_col: str
        Column name for resource (optional)
    min_length: int
        Minimum trace length (filter out shorter traces)
    max_length: int, optional
        Maximum trace length (truncate longer traces)
    add_special_tokens: bool
        Whether to add <START> and <END> tokens
    verbose: bool
        Print extraction statistics
    
    Returns:
    --------
    traces: list of lists
        Each inner list is a sequence of activity names
    trace_metadata: list of dicts
        Metadata for each trace (case_id, length, timestamps, etc.)
    """
    try:
        import pm4py
    except ImportError:
        raise ImportError("pm4py required. Install with: pip install pm4py")
    
    # Convert to dataframe if EventLog
    if not isinstance(log_or_df, pd.DataFrame):
        df = pm4py.convert_to_dataframe(log_or_df)
    else:
        df = log_or_df
    
    if verbose:
        print("\nExtracting traces from XES...")
        print(f"  - Total events: {len(df)}")
        print(f"  - Unique cases: {df[case_col].nunique()}")
        print(f"  - Unique activities: {df[activity_col].nunique()}")
    
    traces = []
    trace_metadata = []
    
    filtered_short = 0
    filtered_long = 0
    
    # Group by case
    for case_id, group in df.groupby(case_col):
        # Sort by timestamp if available
        if timestamp_col in df.columns:
            group = group.sort_values(timestamp_col)
        
        # Extract activity sequence
        activity_sequence = group[activity_col].tolist()
        
        # Filter by minimum length
        if len(activity_sequence) < min_length:
            filtered_short += 1
            continue
        
        # Truncate if needed
        original_length = len(activity_sequence)
        if max_length and len(activity_sequence) > max_length:
            activity_sequence = activity_sequence[:max_length]
            filtered_long += 1
        
        # Add special tokens
        #if add_special_tokens:
        #    activity_sequence = ['<START>'] + activity_sequence + ['<END>']
        
        traces.append(activity_sequence)
        
        # Extract metadata
        metadata = {
            'case_id': case_id,
            'length': len(activity_sequence),
            'original_length': original_length,
            'truncated': original_length > len(activity_sequence) if max_length else False
        }
        
        if timestamp_col in group.columns:
            metadata['start_time'] = group[timestamp_col].iloc[0]
            metadata['end_time'] = group[timestamp_col].iloc[-1]
            metadata['duration'] = (metadata['end_time'] - metadata['start_time']).total_seconds()
        
        if resource_col in group.columns:
            metadata['resources'] = group[resource_col].tolist()
            metadata['unique_resources'] = len(set(metadata['resources']))
        
        trace_metadata.append(metadata)
    
    if verbose:
        print(f"\n  - Extracted {len(traces)} traces")
        print(f"  - Filtered (too short): {filtered_short}")
        if max_length:
            print(f"  - Truncated (too long): {filtered_long}")
        if len(traces) > 0:
            lengths = [len(t) for t in traces]
            print(f"  - Trace length: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
    
    return traces, trace_metadata


# ============================================================================
# ACTIVITY ANALYSIS
# ============================================================================

def analyze_activities(df_or_traces, activity_col='concept:name', verbose=True):
    """
    Analyze activity distribution in the event log
    
    Parameters:
    -----------
    df_or_traces: pd.DataFrame or list of lists
        Event log dataframe or list of traces
    activity_col: str
        Column name for activity (if dataframe)
    verbose: bool
        Print analysis
    
    Returns:
    --------
    activity_stats: dict
        Statistics about activities
    """
    # Extract activities
    if isinstance(df_or_traces, pd.DataFrame):
        activities = df_or_traces[activity_col].tolist()
    else:
        # Flatten traces
        activities = []
        for trace in df_or_traces:
            activities.extend(trace)
    
    # Count activities
    activity_counts = Counter(activities)

    # Remove only PAD and UNK tokens from counts for analysis
    tokens_to_exclude = {'<PAD>', '<UNK>'}
    activity_counts_filtered = Counter({
        k: v for k, v in activity_counts.items()
        if k not in tokens_to_exclude
    })
    
    total_events = sum(activity_counts.values())
    total_events_filtered = sum(activity_counts_filtered.values())

    stats = {
        'total_events': total_events,
        'unique_activities': len(activity_counts),
        'unique_activities_filtered': len(activity_counts_filtered),
        'activity_counts': activity_counts,
        'most_common': activity_counts_filtered.most_common(10),
        'least_common': activity_counts_filtered.most_common()[-10:] if len(activity_counts_filtered) >= 10 else [],
    }
    
    if verbose:
        print("\n" + "="*70)
        print("ACTIVITY ANALYSIS")
        print("="*70)
        print(f"\nTotal events: {total_events}")
        print(f"Unique activities (including special tokens): {stats['unique_activities']}")
        
        print(f"\nTop 10 most frequent activities (excluding special tokens):")
        for i, (activity, count) in enumerate(stats['most_common'][:10], 1):
            pct = 100 * count / total_events
            print(f"  {i:2d}. {activity:30s} : {count:6d} ({pct:5.2f}%)")
        
        if len(stats['least_common']) > 0:
            print(f"\nRarest activities:")
            for activity, count in stats['least_common'][:5]:
                pct = 100 * count / total_events
                print(f"      {activity:30s} : {count:6d} ({pct:5.2f}%)")
    
    return stats


# ============================================================================
# TRACE ANALYSIS
# ============================================================================

def analyze_traces(traces, verbose=True):
    """
    Analyze trace patterns and statistics
    
    Parameters:
    -----------
    traces: list of lists
        List of activity sequences
    verbose: bool
        Print analysis
    
    Returns:
    --------
    trace_stats: dict
        Statistics about traces
    """
    # Compute basic statistics
    trace_lengths = [len(trace) for trace in traces]
    
    # Count unique trace variants
    trace_variants = Counter([tuple(trace) for trace in traces])
    
    # Find most common patterns
    bigrams = []
    trigrams = []
    for trace in traces:
        for i in range(len(trace) - 1):
            bigrams.append((trace[i], trace[i+1]))
        for i in range(len(trace) - 2):
            trigrams.append((trace[i], trace[i+1], trace[i+2]))
    
    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)
    
    stats = {
        'num_traces': len(traces),
        'num_unique_variants': len(trace_variants),
        'variant_diversity': len(trace_variants) / len(traces),
        'trace_lengths': {
            'min': min(trace_lengths),
            'max': max(trace_lengths),
            'mean': np.mean(trace_lengths),
            'median': np.median(trace_lengths),
            'std': np.std(trace_lengths)
        },
        'most_common_variants': trace_variants.most_common(10),
        'most_common_bigrams': bigram_counts.most_common(10),
        'most_common_trigrams': trigram_counts.most_common(5)
    }
    
    if verbose:
        print("\n" + "="*70)
        print("TRACE ANALYSIS")
        print("="*70)
        print(f"\nTotal traces: {stats['num_traces']}")
        print(f"Unique trace variants: {stats['num_unique_variants']}")
        print(f"Variant diversity: {stats['variant_diversity']:.3f}")
        
        print(f"\nTrace length statistics:")
        print(f"  - Min:    {stats['trace_lengths']['min']}")
        print(f"  - Max:    {stats['trace_lengths']['max']}")
        print(f"  - Mean:   {stats['trace_lengths']['mean']:.2f}")
        print(f"  - Median: {stats['trace_lengths']['median']:.1f}")
        print(f"  - Std:    {stats['trace_lengths']['std']:.2f}")
        
        print(f"\nMost common trace variants:")
        for i, (variant, count) in enumerate(stats['most_common_variants'][:5], 1):
            pct = 100 * count / stats['num_traces']
            # Filter out special tokens for display
            variant_filtered = [act for act in variant if act not in {'<PAD>', '<UNK>'}]
            variant_str = ' → '.join(variant_filtered[:5])  # Show first 5 activities
            if len(variant_filtered) > 5:
                variant_str += ' → ...'
            print(f"  {i}. [{count:4d}, {pct:5.2f}%] {variant_str}")
        
        print(f"\nMost common activity transitions (bigrams, excluding special tokens):")
        # Filter bigrams to exclude special tokens
        special_tokens = {'<PAD>', '<UNK>'}
        filtered_bigrams = [(b, c) for b, c in stats['most_common_bigrams']
                           if b[0] not in special_tokens and b[1] not in special_tokens]
        for i, (bigram, count) in enumerate(filtered_bigrams[:5], 1):
            print(f"  {i}. {bigram[0]:20s} → {bigram[1]:20s} : {count:5d}")
    
    return stats


# ============================================================================
# COMPLETE XES LOADING PIPELINE
# ============================================================================

def load_xes_for_gan(xes_filepath,
                     min_trace_length=2,
                     max_trace_length=None,
                     min_activity_frequency=1,
                     train_ratio=0.7,
                     val_ratio=0.15,
                     test_ratio=0.15,
                     verbose=True):
    """
    Complete pipeline to load XES file and prepare for GAN training
    
    Parameters:
    -----------
    xes_filepath: str
        Path to XES file
    min_trace_length: int
        Minimum trace length to keep
    max_trace_length: int, optional
        Maximum trace length (longer traces will be truncated)
    min_activity_frequency: int
        Minimum frequency for an activity to be included
    train_ratio: float
        Proportion of data for training
    val_ratio: float
        Proportion of data for validation
    test_ratio: float
        Proportion of data for testing
    verbose: bool
        Print detailed information
    
    Returns:
    --------
    data: dict
        Dictionary containing all preprocessed data
    """
    if verbose:
        print("\n" + "="*70)
        print("XES FILE LOADING PIPELINE FOR PROCESS MINING GAN")
        print("="*70)
    
    # Step 1: Load XES file
    log, df = load_xes_file(xes_filepath, verbose=verbose)
    
    # Step 2: Extract traces
    traces, metadata = extract_traces_from_xes(
        df,
        min_length=min_trace_length,
        max_length=max_trace_length,
        add_special_tokens=True,
        verbose=verbose
    )
    
    if len(traces) == 0:
        raise ValueError("No traces extracted! Check your min_length parameter.")
    
    # Step 3: Analyze activities
    activity_stats = analyze_activities(traces, verbose=verbose)
    
    # Step 4: Analyze traces
    trace_stats = analyze_traces(traces, verbose=verbose)
    
    # Step 5: Build vocabulary
    if verbose:
        print("\n" + "="*70)
        print("BUILDING VOCABULARY")
        print("="*70)
    
    # Count all activities
    activity_counts = Counter()
    for trace in traces:
        activity_counts.update(trace)
    
    # Filter by frequency
    activities = [act for act, count in activity_counts.items() 
                 if count >= min_activity_frequency or act in {'<PAD>', '<UNK>'}]
    
    # Sort for consistency
    special_tokens = ['<PAD>', '<UNK>']
    regular_activities = sorted([a for a in activities if a not in special_tokens])
    all_activities = special_tokens + regular_activities
    
    # Create mappings
    activity_to_idx = {act: idx for idx, act in enumerate(all_activities)}
    idx_to_activity = {idx: act for act, idx in activity_to_idx.items()}
    
    if verbose:
        print(f"\nVocabulary size: {len(activity_to_idx)}")
        print(f"  - Special tokens: {len(special_tokens)}")
        print(f"  - Regular activities: {len(regular_activities)}")
        print(f"  - Minimum frequency: {min_activity_frequency}")
    
    # Compute activity frequencies (for GAN constraints)
    activity_freq_counts = np.zeros(len(activity_to_idx))
    for trace in traces:
        for activity in trace:
            if activity in activity_to_idx:
                activity_freq_counts[activity_to_idx[activity]] += 1
    activity_frequencies = activity_freq_counts / activity_freq_counts.sum()
    
    # Step 6: Split data
    if verbose:
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)
    
    n_traces = len(traces)
    n_train = int(n_traces * train_ratio)
    n_val = int(n_traces * val_ratio)
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(n_traces)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    traces_train = [traces[i] for i in train_idx]
    traces_val = [traces[i] for i in val_idx]
    traces_test = [traces[i] for i in test_idx]
    
    metadata_train = [metadata[i] for i in train_idx]
    metadata_val = [metadata[i] for i in val_idx]
    metadata_test = [metadata[i] for i in test_idx]
    
    if verbose:
        print(f"\nTrain set: {len(traces_train)} traces ({100*train_ratio:.1f}%)")
        print(f"Val set:   {len(traces_val)} traces ({100*val_ratio:.1f}%)")
        print(f"Test set:  {len(traces_test)} traces ({100*test_ratio:.1f}%)")
    
    # Step 7: Determine max trace length
    if max_trace_length is None:
        max_trace_length = max(len(t) for t in traces)
    
    if verbose:
        print(f"\nMax trace length: {max_trace_length}")
    
    # Prepare output
    data = {
        'traces_train': traces_train,
        'traces_val': traces_val,
        'traces_test': traces_test,
        'metadata_train': metadata_train,
        'metadata_val': metadata_val,
        'metadata_test': metadata_test,
        'activity_to_idx': activity_to_idx,
        'idx_to_activity': idx_to_activity,
        'num_activities': len(activity_to_idx),
        'max_trace_length': max_trace_length,
        'activity_frequencies': activity_frequencies,
        'activity_stats': activity_stats,
        'trace_stats': trace_stats,
        'special_tokens': {
            'pad_idx': activity_to_idx['<PAD>'],
            'start_idx': activity_to_idx['START'],
            'end_idx': activity_to_idx['6'],
            'unk_idx': activity_to_idx['<UNK>']
        }
    }
    
    if verbose:
        print("\n" + "="*70)
        print("LOADING COMPLETE!")
        print("="*70)
    
    return data


# ============================================================================
# SAVE/LOAD PREPROCESSED DATA
# ============================================================================

def save_preprocessed_xes(data, filepath):
    """Save preprocessed XES data to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n✓ Saved preprocessed data to: {filepath}")


def load_preprocessed_xes(filepath):
    """Load preprocessed XES data from file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"\n✓ Loaded preprocessed data from: {filepath}")
    return data


if __name__ == "__main__":
    print("\nXES Loader - Usage Example")
    print("="*70)
    print("\nTo use this module:")
    print("\n  from xes_loader import load_xes_for_gan")
    print("  data = load_xes_for_gan('your_file.xes')")
    print("\nSee train_gan_from_xes.py for complete example")