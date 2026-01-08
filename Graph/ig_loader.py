"""
Instance Graph Loader - Parse .g files

Parses the Helpdesk instance graph format with:
- Vertices (v lines): nodes with activities, timestamps, features
- Edges (e lines): connections between nodes with labels
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def log(msg: str):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] {msg}')


class InstanceGraphLoader:
    """
    Load and parse instance graph (.g) files
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.activity_to_idx = {}
        self.idx_to_activity = {}
        self.edge_label_to_idx = {}
        self.idx_to_edge_label = {}

    def load_file(self, filepath: str) -> List[Dict]:
        """
        Load and parse .g file

        Args:
            filepath: Path to .g file

        Returns:
            List of traces, each trace is a dict with:
                - 'vertices': list of vertex dicts
                - 'edges': list of edge dicts
                - 'case_id': case identifier
        """
        if self.verbose:
            log(f'Loading instance graph file: {filepath}')

        traces = []
        current_trace = None
        in_trace = False

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                if not line or line.startswith('e_v node1'):
                    continue

                if line == 'XP':
                    if current_trace is not None and current_trace['vertices']:
                        traces.append(current_trace)

                    current_trace = {
                        'vertices': [],
                        'edges': [],
                        'case_id': None
                    }
                    in_trace = True
                    continue

                if not in_trace:
                    continue

               
                if line.startswith('v '):
                    vertex = self._parse_vertex_line(line, line_num)
                    if vertex:
                        current_trace['vertices'].append(vertex)              
                        if current_trace['case_id'] is None:
                            current_trace['case_id'] = vertex['case_id']

                elif line.startswith('e '):
                    edge = self._parse_edge_line(line, line_num)
                    if edge:
                        current_trace['edges'].append(edge)

        if current_trace is not None and current_trace['vertices']:
            traces.append(current_trace)

        if self.verbose:
            log(f'Loaded {len(traces)} traces from file')

        return traces

    def _parse_vertex_line(self, line: str, line_num: int) -> Optional[Dict]:
        """
        Parse vertex line: v node_id activity start_time end_time case_id norm_time trace_time prev_event_time

        Example:
        v 1.0  START "2012-04-03 14:55:37+00:00" "2012-04-03 14:55:37+00:00" 2.0 0.271606374316622 0.0 0.0
        """
        try:
            import re
            parts = re.findall(r'"[^"]*"|\S+', line)
            parts = parts[1:]

            vertex = {
                'node_id': float(parts[0]),
                'activity': parts[1].strip('"'),
                'start_timestamp': parts[2].strip('"') if len(parts) > 2 else None,
                'end_timestamp': parts[3].strip('"') if len(parts) > 3 else None,
                'case_id': float(parts[4]) if len(parts) > 4 else None,
                'norm_time': float(parts[5]) if len(parts) > 5 else 0.0,
                'trace_time': float(parts[6]) if len(parts) > 6 else 0.0,
                'prev_event_time': float(parts[7]) if len(parts) > 7 else 0.0
            }

            return vertex

        except (IndexError, ValueError) as e:
            if self.verbose:
                log(f'Warning: Could not parse vertex at line {line_num}: {e}')
            return None

    def _parse_edge_line(self, line: str, line_num: int) -> Optional[Dict]:
        """
        Parse edge line: e source target label case_id

        Example:
        e 1.0 2.0 START__1   2.0
        """
        try:
            parts = line.split()

            edge = {
                'source': float(parts[1]),
                'target': float(parts[2]),
                'label': parts[3],
                'case_id': float(parts[4]) if len(parts) > 4 else None
            }

            return edge

        except (IndexError, ValueError) as e:
            if self.verbose:
                log(f'Warning: Could not parse edge at line {line_num}: {e}')
            return None

    def build_vocabularies(self, traces: List[Dict]):
        """
        Build activity and edge label vocabularies from traces

        Args:
            traces: List of parsed traces
        """
        if self.verbose:
            log('Building vocabularies...')

        activities = set()
        for trace in traces:
            for vertex in trace['vertices']:
                activities.add(vertex['activity'])
        activities = sorted(list(activities))

        activities = ['<PAD>'] + activities

        self.activity_to_idx = {act: idx for idx, act in enumerate(activities)}
        self.idx_to_activity = {idx: act for idx, act in enumerate(activities)}

        edge_labels = set()
        for trace in traces:
            for edge in trace['edges']:
                edge_labels.add(edge['label'])

        edge_labels = sorted(list(edge_labels))

        self.edge_label_to_idx = {label: idx for idx,
                                  label in enumerate(edge_labels)}
        self.idx_to_edge_label = {idx: label for idx,
                                  label in enumerate(edge_labels)}

    def get_statistics(self, traces: List[Dict]) -> Dict:
        """
        Compute statistics from traces

        Args:
            traces: List of parsed traces

        Returns:
            Dictionary with statistics
        """
        num_nodes = [len(trace['vertices']) for trace in traces]
        num_edges = [len(trace['edges']) for trace in traces]

        stats = {
            'num_traces': len(traces),
            'min_nodes': min(num_nodes) if num_nodes else 0,
            'max_nodes': max(num_nodes) if num_nodes else 0,
            'avg_nodes': np.mean(num_nodes) if num_nodes else 0,
            'min_edges': min(num_edges) if num_edges else 0,
            'max_edges': max(num_edges) if num_edges else 0,
            'avg_edges': np.mean(num_edges) if num_edges else 0,
            'num_activities': len(self.activity_to_idx),
            'num_edge_labels': len(self.edge_label_to_idx)
        }

        return stats

    def print_statistics(self, traces: List[Dict]):
        """Print statistics"""
        stats = self.get_statistics(traces)

        print('\n' + '='*80)
        print('INSTANCE GRAPH STATISTICS')
        print('='*80)
        print(f"Number of traces: {stats['num_traces']}")
        print(
            f"Nodes per trace: min={stats['min_nodes']}, max={stats['max_nodes']}, avg={stats['avg_nodes']:.2f}")
        print(
            f"Edges per trace: min={stats['min_edges']}, max={stats['max_edges']}, avg={stats['avg_edges']:.2f}")
        print(f"Number of unique activities: {stats['num_activities']}")
        print(f"Number of unique edge labels: {stats['num_edge_labels']}")
        print('='*80 + '\n')


def load_ig_for_gan(filepath: str,
                    min_trace_nodes: int = 2,
                    max_trace_nodes: int = 8,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15,
                    verbose: bool = True) -> Dict:
    """
    Load instance graph file and prepare for GAN training

    Args:
        filepath: Path to .g file
        min_trace_nodes: Minimum number of nodes per trace
        max_trace_nodes: Maximum number of nodes per trace
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        verbose: Print progress

    Returns:
        Dictionary with:
            - traces_train, traces_val, traces_test: Split traces
            - activity_to_idx, idx_to_activity: Activity mappings
            - edge_label_to_idx, idx_to_edge_label: Edge label mappings
            - max_nodes: Maximum number of nodes
            - statistics: Data statistics
    """
    loader = InstanceGraphLoader(verbose=verbose)

    traces = loader.load_file(filepath)

    loader.build_vocabularies(traces)

    filtered_traces = []
    for trace in traces:
        num_nodes = len(trace['vertices'])
        if min_trace_nodes <= num_nodes <= max_trace_nodes:
            filtered_traces.append(trace)

    if verbose:
        log(f'Filtered to {len(filtered_traces)}/{len(traces)} traces '
            f'(nodes between {min_trace_nodes} and {max_trace_nodes})')

    n_total = len(filtered_traces)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    np.random.seed(42)
    indices = np.random.permutation(n_total)

    traces_train = [filtered_traces[i] for i in indices[:n_train]]
    traces_val = [filtered_traces[i] for i in indices[n_train:n_train+n_val]]
    traces_test = [filtered_traces[i] for i in indices[n_train+n_val:]]

    if verbose:
        log(f'Split: train={len(traces_train)}, val={len(traces_val)}, test={len(traces_test)}')

    if verbose:
        loader.print_statistics(filtered_traces)

    max_nodes = max(len(trace['vertices']) for trace in filtered_traces)

    return {
        'traces_train': traces_train,
        'traces_val': traces_val,
        'traces_test': traces_test,
        'activity_to_idx': loader.activity_to_idx,
        'idx_to_activity': loader.idx_to_activity,
        'edge_label_to_idx': loader.edge_label_to_idx,
        'idx_to_edge_label': loader.idx_to_edge_label,
        'max_nodes': max_nodes,
        'num_activities': len(loader.activity_to_idx),
        'num_edge_labels': len(loader.edge_label_to_idx),
        'statistics': loader.get_statistics(filtered_traces)
    }
