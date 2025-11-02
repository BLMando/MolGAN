"""
Data Preprocessing Utilities for Process Mining GAN
Supports XES, CSV, and other common event log formats
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import pickle


# ============================================================================
# EVENT LOG LOADING
# ============================================================================

def load_csv_event_log(filepath, case_col='case:concept:name', 
                       activity_col='concept:name', 
                       timestamp_col='time:timestamp',
                       sort_by_time=True):
    """
    Load event log from CSV file
    
    Parameters:
    -----------
    filepath: str
        Path to CSV file
    case_col: str
        Column name for case ID
    activity_col: str
        Column name for activity
    timestamp_col: str
        Column name for timestamp
    sort_by_time: bool
        Whether to sort events by timestamp
    
    Returns:
    --------
    df: pd.DataFrame
        Event log dataframe
    """
    df = pd.read_csv(filepath)
    
    # Ensure required columns exist
    required_cols = [case_col, activity_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV")
    
    # Sort by time if requested and column exists
    if sort_by_time and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([case_col, timestamp_col])
    
    return df


def load_xes_event_log(filepath):
    """
    Load event log from XES file
    Requires pm4py library: pip install pm4py
    
    Parameters:
    -----------
    filepath: str
        Path to XES file
    
    Returns:
    --------
    df: pd.DataFrame
        Event log dataframe
    """
    try:
        import pm4py
        from pm4py.objects.log.importer.xes import importer as xes_importer
    except ImportError:
        raise ImportError("pm4py is required to load XES files. Install with: pip install pm4py")
    
    # Import XES log
    log = xes_importer.apply(filepath)
    
    # Convert to dataframe
    df = pm4py.convert_to_dataframe(log)
    
    return df


# ============================================================================
# TRACE EXTRACTION
# ============================================================================

def extract_traces(df, case_col='case:concept:name', 
                  activity_col='concept:name',
                  timestamp_col=None,
                  resource_col=None,
                  min_length=2,
                  max_length=None):
    """
    Extract traces from event log
    
    Parameters:
    -----------
    df: pd.DataFrame
        Event log dataframe
    case_col: str
        Column name for case ID
    activity_col: str
        Column name for activity
    timestamp_col: str, optional
        Column name for timestamp
    resource_col: str, optional
        Column name for resource
    min_length: int
        Minimum trace length (shorter traces will be filtered)
    max_length: int, optional
        Maximum trace length (longer traces will be truncated)
    
    Returns:
    --------
    traces: list of lists
        Each inner list is a sequence of activities
    trace_metadata: dict
        Additional information about each trace
    """
    traces = []
    trace_metadata = []
    
    # Group by case
    for case_id, group in df.groupby(case_col):
        # Sort by timestamp if available
        if timestamp_col and timestamp_col in df.columns:
            group = group.sort_values(timestamp_col)
        
        # Extract activity sequence
        activity_sequence = group[activity_col].tolist()
        
        # Filter by length
        if len(activity_sequence) < min_length:
            continue
        
        if max_length and len(activity_sequence) > max_length:
            activity_sequence = activity_sequence[:max_length]
        
        traces.append(activity_sequence)
        
        # Extract metadata
        metadata = {'case_id': case_id, 'length': len(activity_sequence)}
        if timestamp_col and timestamp_col in group.columns:
            metadata['start_time'] = group[timestamp_col].iloc[0]
            metadata['end_time'] = group[timestamp_col].iloc[-1]
            metadata['duration'] = (metadata['end_time'] - metadata['start_time']).total_seconds()
        if resource_col and resource_col in group.columns:
            metadata['resources'] = group[resource_col].tolist()
        
        trace_metadata.append(metadata)
    
    return traces, trace_metadata


# ============================================================================
# VOCABULARY BUILDING
# ============================================================================

def build_vocabulary(traces, add_special_tokens=True, min_frequency=1):
    """
    Build activity vocabulary from traces
    
    Parameters:
    -----------
    traces: list of lists
        List of activity sequences
    add_special_tokens: bool
        Whether to add <PAD>, <START>, <END> tokens
    min_frequency: int
        Minimum frequency for an activity to be included
    
    Returns:
    --------
    activity_to_idx: dict
        Mapping from activity name to index
    idx_to_activity: dict
        Mapping from index to activity name
    activity_counts: Counter
        Frequency of each activity
    """
    # Count activity frequencies
    activity_counts = Counter()
    for trace in traces:
        activity_counts.update(trace)
    
    # Filter by minimum frequency
    activities = [act for act, count in activity_counts.items() 
                 if count >= min_frequency]
    
    # Sort alphabetically for consistency
    activities = sorted(activities)
    
    # Add special tokens
    if add_special_tokens:
        special_tokens = ['<PAD>', '<UNK>']
        activities = special_tokens + activities
    
    # Create mappings
    activity_to_idx = {act: idx for idx, act in enumerate(activities)}
    idx_to_activity = {idx: act for act, idx in activity_to_idx.items()}
    
    return activity_to_idx, idx_to_activity, activity_counts


# ============================================================================
# TRACE AUGMENTATION
# ============================================================================

def add_start_end_tokens(traces, start_token='START', end_token='6'):
    """
    Add START and END tokens to traces
    
    Parameters:
    -----------
    traces: list of lists
        Original traces
    start_token: str
        Start token to add
    end_token: str
        End token to add
    
    Returns:
    --------
    augmented_traces: list of lists
        Traces with start and end tokens
    """
    augmented_traces = []
    for trace in traces:
        augmented_trace = [start_token] + trace + [end_token]
        augmented_traces.append(augmented_trace)
    
    return traces


# ============================================================================
# TRAIN/VAL/TEST SPLIT
# ============================================================================

def split_traces(traces, trace_metadata=None, 
                train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                random_state=42):
    """
    Split traces into train/validation/test sets
    
    Parameters:
    -----------
    traces: list of lists
        All traces
    trace_metadata: list of dicts, optional
        Metadata for each trace
    train_ratio: float
        Proportion for training set
    val_ratio: float
        Proportion for validation set
    test_ratio: float
        Proportion for test set
    random_state: int
        Random seed for reproducibility
    
    Returns:
    --------
    splits: dict
        Dictionary with 'train', 'val', 'test' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n_traces = len(traces)
    n_train = int(n_traces * train_ratio)
    n_val = int(n_traces * val_ratio)
    
    # Shuffle indices
    np.random.seed(random_state)
    indices = np.random.permutation(n_traces)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    splits = {
        'train': [traces[i] for i in train_indices],
        'val': [traces[i] for i in val_indices],
        'test': [traces[i] for i in test_indices]
    }
    
    if trace_metadata:
        splits['train_metadata'] = [trace_metadata[i] for i in train_indices]
        splits['val_metadata'] = [trace_metadata[i] for i in val_indices]
        splits['test_metadata'] = [trace_metadata[i] for i in test_indices]
    
    return splits


# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================

def compute_trace_statistics(traces):
    """
    Compute statistics about traces
    
    Parameters:
    -----------
    traces: list of lists
        List of traces
    
    Returns:
    --------
    stats: dict
        Dictionary of statistics
    """
    trace_lengths = [len(trace) for trace in traces]
    
    # Compute activity statistics
    all_activities = []
    for trace in traces:
        all_activities.extend(trace)
    activity_counts = Counter(all_activities)
    
    # Compute n-gram statistics
    bigrams = []
    for trace in traces:
        for i in range(len(trace) - 1):
            bigrams.append((trace[i], trace[i + 1]))
    bigram_counts = Counter(bigrams)
    
    stats = {
        'num_traces': len(traces),
        'num_unique_activities': len(activity_counts),
        'avg_trace_length': np.mean(trace_lengths),
        'median_trace_length': np.median(trace_lengths),
        'min_trace_length': np.min(trace_lengths),
        'max_trace_length': np.max(trace_lengths),
        'std_trace_length': np.std(trace_lengths),
        'most_common_activities': activity_counts.most_common(10),
        'most_common_bigrams': bigram_counts.most_common(10),
        'num_unique_traces': len(set(tuple(trace) for trace in traces))
    }
    
    return stats


# ============================================================================
# DIRECTLY-FOLLOWS GRAPH (DFG)
# ============================================================================

def build_dfg(traces):
    """
    Build Directly-Follows Graph from traces
    
    Parameters:
    -----------
    traces: list of lists
        List of traces
    
    Returns:
    --------
    dfg: dict
        Dictionary mapping (activity_i, activity_j) -> frequency
    """
    dfg = defaultdict(int)
    
    for trace in traces:
        for i in range(len(trace) - 1):
            edge = (trace[i], trace[i + 1])
            dfg[edge] += 1
    
    return dict(dfg)


def dfg_to_adjacency_matrix(dfg, activity_to_idx):
    """
    Convert DFG to adjacency matrix
    
    Parameters:
    -----------
    dfg: dict
        Directly-follows graph
    activity_to_idx: dict
        Activity to index mapping
    
    Returns:
    --------
    adj_matrix: np.array [num_activities, num_activities]
        Adjacency matrix
    """
    n = len(activity_to_idx)
    adj_matrix = np.zeros((n, n), dtype=np.float32)
    
    for (act_i, act_j), freq in dfg.items():
        if act_i in activity_to_idx and act_j in activity_to_idx:
            i = activity_to_idx[act_i]
            j = activity_to_idx[act_j]
            adj_matrix[i, j] = freq
    
    return adj_matrix


# ============================================================================
# SAVE/LOAD PREPROCESSED DATA
# ============================================================================

def save_preprocessed_data(filepath, traces, activity_to_idx, 
                          idx_to_activity, trace_metadata=None):
    """
    Save preprocessed data to file
    
    Parameters:
    -----------
    filepath: str
        Path to save file
    traces: list of lists
        Processed traces
    activity_to_idx: dict
        Activity to index mapping
    idx_to_activity: dict
        Index to activity mapping
    trace_metadata: list of dicts, optional
        Trace metadata
    """
    data = {
        'traces': traces,
        'activity_to_idx': activity_to_idx,
        'idx_to_activity': idx_to_activity,
        'trace_metadata': trace_metadata
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved preprocessed data to {filepath}")


def load_preprocessed_data(filepath):
    """
    Load preprocessed data from file
    
    Parameters:
    -----------
    filepath: str
        Path to saved file
    
    Returns:
    --------
    data: dict
        Dictionary containing traces, mappings, and metadata
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data