"""
Graph Dataset - Convert instance graphs to adjacency matrices and node features

Handles:
- Binary adjacency matrix construction (no edge types)
- Node feature matrices
- Padding for variable-length graphs
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
from datetime import datetime


def log(msg: str):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] {msg}')


class GraphDataset:
    """
    Convert instance graphs to binary adjacency matrices for GAN training
    (No edge type classification - binary edges only)
    """

    def __init__(self,
                 traces: List[Dict],
                 activity_to_idx: Dict[str, int],
                 max_nodes: int,
                 include_features: bool = True,
                 verbose: bool = True,
                 min_graph_size: int = None,
                 max_graph_size: int = None):
        """
        Args:
            traces: List of parsed traces from ig_loader
            activity_to_idx: Activity vocabulary mapping
            max_nodes: Maximum number of nodes per graph
            include_features: Whether to include temporal features
            verbose: Print progress
            min_graph_size: Minimum number of nodes (vertices) to include. None = no minimum
            max_graph_size: Maximum number of nodes (vertices) to include. None = no maximum
        """
        self.activity_to_idx = activity_to_idx
        self.idx_to_activity = {v: k for k, v in activity_to_idx.items()}
        self.max_nodes = max_nodes
        self.include_features = include_features
        self.verbose = verbose
        self.num_activities = len(activity_to_idx)
        
        # Filter traces by size if specified
        self.traces = self._filter_traces_by_size(traces, min_graph_size, max_graph_size)

        # No edge type classification (binary adjacency)
        # self.edge_classifier = EdgeTypeClassifier()  # REMOVED
        # self.num_edge_types = self.edge_classifier.num_types  # REMOVED

        # Preprocess all traces
        self.adjacency_matrices = []
        self.node_matrices = []
        self.feature_matrices = [] if include_features else None

        self._preprocess_traces()

    def _filter_traces_by_size(self, traces: List[Dict], min_size: int = None, max_size: int = None) -> List[Dict]:
        """
        Filter traces to only include graphs within specified size range
        
        Args:
            traces: List of traces
            min_size: Minimum number of vertices (nodes) to include
            max_size: Maximum number of vertices (nodes) to include
            
        Returns:
            Filtered list of traces
        """
        if min_size is None and max_size is None:
            return traces
        
        filtered_traces = []
        for trace in traces:
            num_vertices = len(trace['vertices'])
            
            # Check minimum size
            if min_size is not None and num_vertices < min_size:
                continue
            
            # Check maximum size
            if max_size is not None and num_vertices > max_size:
                continue
            
            filtered_traces.append(trace)
        
        if self.verbose:
            log(f'Filtered traces by size: {len(traces)} -> {len(filtered_traces)} traces')
            if min_size is not None:
                log(f'  Min size: {min_size} nodes')
            if max_size is not None:
                log(f'  Max size: {max_size} nodes')
        
        return filtered_traces


    def _preprocess_traces(self):
        """Convert all traces to matrices"""
        if self.verbose:
            log(f'Converting {len(self.traces)} traces to matrices...')

        for i, trace in enumerate(self.traces):
            # Build node ID to index mapping
            vertices = trace['vertices']
            node_id_to_idx = {v['node_id']
                : idx for idx, v in enumerate(vertices)}

            # Build adjacency matrix
            adj_matrix = self._build_adjacency(trace, node_id_to_idx)
            self.adjacency_matrices.append(adj_matrix)

            # Build node matrix (one-hot activities)
            node_matrix = self._build_nodes(trace)
            self.node_matrices.append(node_matrix)

            # Build feature matrix (optional)
            if self.include_features:
                feature_matrix = self._build_features(trace)
                self.feature_matrices.append(feature_matrix)

            if self.verbose and (i + 1) % 1000 == 0:
                log(f'  Processed {i + 1}/{len(self.traces)} traces')

        # Convert to numpy arrays
        self.adjacency_matrices = np.array(
            self.adjacency_matrices, dtype=np.float32)
        self.node_matrices = np.array(self.node_matrices, dtype=np.float32)
        if self.include_features:
            self.feature_matrices = np.array(
                self.feature_matrices, dtype=np.float32)

        if self.verbose:
            log(f'Adjacency matrices shape: {self.adjacency_matrices.shape}')
            log(f'Node matrices shape: {self.node_matrices.shape}')
            if self.include_features:
                log(f'Feature matrices shape: {self.feature_matrices.shape}')

    def _build_adjacency(self, trace: Dict, node_id_to_idx: Dict) -> np.ndarray:
        """
        Build binary adjacency matrix

        Returns:
            Adjacency matrix of shape (max_nodes, max_nodes) - binary
        """
        adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)

        # Fill adjacency matrix with binary edges
        edges = trace['edges']

        for edge in edges:
            src_id = edge['source']
            tgt_id = edge['target']

            # Map to indices
            if src_id not in node_id_to_idx or tgt_id not in node_id_to_idx:
                continue

            src_idx = node_id_to_idx[src_id]
            tgt_idx = node_id_to_idx[tgt_id]

            # Set binary adjacency (no edge type)
            if src_idx < self.max_nodes and tgt_idx < self.max_nodes:
                adj[src_idx, tgt_idx] = 1.0

        return adj

    def _build_nodes(self, trace: Dict) -> np.ndarray:
        """
        Build node matrix (one-hot encoded activities)

        Returns:
            Node matrix of shape (max_nodes, num_activities)
        """
        # Initialize with PAD token (index 0)
        # This ensures padding nodes are explicitly represented as PAD
        # instead of all-zeros, matching the generator's softmax output capability
        nodes = np.zeros(
            (self.max_nodes, self.num_activities), dtype=np.float32)
        nodes[:, 0] = 1.0  # Set all to PAD initially

        vertices = trace['vertices']
        for idx, vertex in enumerate(vertices):
            if idx >= self.max_nodes:
                break

            activity = vertex['activity']
            activity_idx = self.activity_to_idx.get(activity, 0)  # 0 = <PAD>
            
            # Clear PAD and set actual activity
            nodes[idx, 0] = 0.0
            nodes[idx, activity_idx] = 1.0

        return nodes

    def _build_features(self, trace: Dict) -> np.ndarray:
        """
        Build feature matrix (temporal features already normalized in .g file)

        Returns:
            Feature matrix of shape (max_nodes, num_features)
            Features: [norm_time, trace_time, prev_event_time]
        """
        num_features = 3
        features = np.zeros((self.max_nodes, num_features), dtype=np.float32)

        vertices = trace['vertices']
        for idx, vertex in enumerate(vertices):
            if idx >= self.max_nodes:
                break

            features[idx, 0] = vertex['norm_time']
            features[idx, 1] = vertex['trace_time']
            features[idx, 2] = vertex['prev_event_time']

        return features

    def get_tf_dataset(self, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle

        Returns:
            tf.data.Dataset yielding (adjacency, nodes) or (adjacency, nodes, features)
        """
        if self.include_features:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.adjacency_matrices, self.node_matrices, self.feature_matrices)
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.adjacency_matrices, self.node_matrices)
            )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.traces))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def compute_activity_frequencies(self) -> np.ndarray:
        """
        Compute normalized activity frequencies across all traces

        Returns:
            Array of shape (num_activities,) with frequencies
        """
        # Sum across all nodes and traces
        activity_counts = np.sum(self.node_matrices, axis=(0, 1))

        # Normalize
        activity_frequencies = activity_counts / np.sum(activity_counts)

        return activity_frequencies.astype(np.float32)

    def print_example(self, idx: int = 0):
        """Print example graph matrices"""
        if idx >= len(self.traces):
            print(f'Index {idx} out of range')
            return

        trace = self.traces[idx]
        adj = self.adjacency_matrices[idx]
        nodes = self.node_matrices[idx]

        print('\n' + '='*80)
        print(f'EXAMPLE GRAPH MATRICES (Case ID: {trace["case_id"]})')
        print('='*80)

        print('\nOriginal trace:')
        for v in trace['vertices']:
            print(f"  Node {v['node_id']}: {v['activity']}")

        print('\nNode matrix (activities):')
        for i in range(min(len(trace['vertices']), self.max_nodes)):
            activity_idx = np.argmax(nodes[i])
            activity = self.idx_to_activity[activity_idx]
            print(f"  Position {i}: {activity}")

        print('\nAdjacency matrix (edges):')
        for i in range(self.max_nodes):
            for j in range(self.max_nodes):
                if adj[i, j] > 0:
                    print(f"  {i} -> {j}")

        if self.include_features and self.feature_matrices is not None:
            features = self.feature_matrices[idx]
            print('\nTemporal features:')
            for i in range(min(len(trace['vertices']), 5)):
                print(f"  Node {i}: norm_time={features[i, 0]:.4f}, "
                      f"trace_time={features[i, 1]:.4f}, prev_time={features[i, 2]:.4f}")

        print('='*80 + '\n')

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        # Count edge types
        # No edge type classification (binary only)
        # edge_type_counts = {}
        # for edge_type, idx in self.edge_classifier.edge_types.items():
        #     count = np.sum(self.adjacency_matrices[:, :, :, idx])
        #     edge_type_counts[edge_type] = int(count)

        # Activity frequencies
        activity_freqs = self.compute_activity_frequencies()

        # Total edge count (binary)
        total_edges = int(np.sum(self.adjacency_matrices))
        
        # Graph size distribution
        graph_sizes = self.get_graph_size_distribution()

        stats = {
            'num_traces': len(self.traces),
            'num_activities': self.num_activities,
            # 'num_edge_types': self.num_edge_types,  # Not applicable
            'max_nodes': self.max_nodes,
            'total_edges': total_edges,
            # 'edge_type_counts': edge_type_counts,  # Not applicable
            'activity_frequencies': activity_freqs,
            'adjacency_shape': self.adjacency_matrices.shape,
            'nodes_shape': self.node_matrices.shape,
            'graph_sizes': graph_sizes
        }

        return stats
    
    def get_graph_size_distribution(self) -> Dict[str, any]:
        """
        Analyze the distribution of graph sizes in the dataset
        
        Returns:
            Dictionary with size statistics
        """
        sizes = []
        for trace in self.traces:
            sizes.append(len(trace['vertices']))
        
        sizes = np.array(sizes)
        
        return {
            'min': int(np.min(sizes)),
            'max': int(np.max(sizes)),
            'mean': float(np.mean(sizes)),
            'median': float(np.median(sizes)),
            'std': float(np.std(sizes)),
            'sizes': sizes,
            'histogram': np.histogram(sizes, bins=min(20, np.max(sizes) - np.min(sizes) + 1))
        }

    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_statistics()

        print('\n' + '='*80)
        print('GRAPH DATASET STATISTICS')
        print('='*80)
        print(f"Number of traces: {stats['num_traces']}")
        print(f"Number of activities: {stats['num_activities']}")
        print(f"Max nodes per graph: {stats['max_nodes']}")
        print(f"Total edges (binary): {stats['total_edges']}")

        # print('\nEdge type distribution:')  # Not applicable
        # for edge_type, count in stats['edge_type_counts'].items():
        #     print(f"  {edge_type}: {count}")
        
        print('\nGraph size distribution:')
        size_stats = stats['graph_sizes']
        print(f"  Min nodes: {size_stats['min']}")
        print(f"  Max nodes: {size_stats['max']}")
        print(f"  Mean nodes: {size_stats['mean']:.2f}")
        print(f"  Median nodes: {size_stats['median']:.1f}")
        print(f"  Std dev: {size_stats['std']:.2f}")

        print('\nTop 5 activities by frequency:')
        activity_freqs = stats['activity_frequencies']
        top_activities = np.argsort(activity_freqs)[::-1][:5]
        for idx in top_activities:
            activity = self.idx_to_activity[idx]
            freq = activity_freqs[idx]
            print(f"  {activity}: {freq:.4f}")

        print(f"\nMatrix shapes:")
        print(f"  Adjacency: {stats['adjacency_shape']}")
        print(f"  Nodes: {stats['nodes_shape']}")

        print('='*80 + '\n')
