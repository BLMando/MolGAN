

import pickle
import numpy as np
from datetime import datetime
from data_processing import XESLoader, PatternDiscovery, TraceEncoder


class ProcessDataset:
    """
    Dataset manager for process mining event logs

    """

    def __init__(self, max_activities=15):
        """
        Initialize ProcessDataset

        Args:
            max_activities: Maximum trace length
        """
        self.max_activities = max_activities

        # Dataset arrays
        self.data = None           # List of traces
        self.data_A = None         # Adjacency matrices
        self.data_X = None         # Node vectors
        self.data_F = None         # Feature matrices

        # Train/val/test splits
        self.train_idx = None
        self.validation_idx = None
        self.test_idx = None

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        self.train_count = 0
        self.validation_count = 0
        self.test_count = 0
        self.__len = 0

        # Components
        self.encoder = TraceEncoder(max_activities)
        self.pattern_discovery = PatternDiscovery()

        # Pattern info (for external access)
        self.petri_net = None
        self.pattern_map = {}
        self.dfg = None
        self.dfg_frequencies = {}
        self.pm4py_available = True

    @property
    def activity_encoder(self):
        """Access activity encoder from TraceEncoder"""
        return self.encoder.activity_encoder

    @property
    def activity_decoder(self):
        """Access activity decoder from TraceEncoder"""
        return self.encoder.activity_decoder

    @property
    def flow_encoder(self):
        """Access flow encoder from TraceEncoder"""
        return self.encoder.flow_encoder

    @property
    def flow_decoder(self):
        """Access flow decoder from TraceEncoder"""
        return self.encoder.flow_decoder

    @property
    def activity_num_types(self):
        """Number of activity types"""
        return self.encoder.activity_num_types

    @property
    def flow_num_types(self):
        """Number of flow types"""
        return self.encoder.flow_num_types

    def log(self, msg, date=True):
        """Print timestamped log message"""
        if date:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'{timestamp} {msg}')
        else:
            print(msg)

    def load_from_xes(self, xes_path, validation=0.1, test=0.1):
        """
        Load event log from XES file

        Args:
            xes_path: Path to XES file
            validation: Fraction for validation set
            test: Fraction for test set
        """
        self.log(f'Loading event log from {xes_path}...')

        # Step 1: Load XES file
        traces, event_log = XESLoader.load_xes(xes_path)

        self.log(f'Loaded {len(traces)} traces from XES file')

        # Step 2: Discover patterns
        patterns = self.pattern_discovery.discover_patterns(event_log)

        self.petri_net = patterns['petri_net']
        self.pattern_map = patterns['pattern_map']
        self.dfg = patterns['dfg']
        self.dfg_frequencies = patterns['dfg_frequencies']
        self.pm4py_available = patterns['pm4py_available']

        # Step 3: Generate dataset
        self._generate_from_traces(traces, validation, test)

    def _generate_from_traces(self, traces, validation, test):
        """
        Generate dataset from traces

        Args:
            traces: List of traces
            validation: Validation fraction
            test: Test fraction
        """
        self.log(f'Generating dataset from {len(traces)} traces...')

        self.data = traces

        # Create encoders
        self.encoder.create_encoders(traces)

        # Set discovered patterns
        self.encoder.set_patterns(self.pattern_map, self.dfg_frequencies)

        # Convert traces to graphs
        self.data_A, self.data_X, self.data_F, self.data = self.encoder.encode_traces(
            traces)

        self.__len = len(self.data)

        # Create splits
        self._generate_train_validation_test(validation, test)

        self.log(
            f'Dataset generated: {self.train_count} train, {self.validation_count} val, {self.test_count} test')

    def _generate_train_validation_test(self, validation_split, test_split):
        """
        Split dataset into train/val/test

        Args:
            validation_split: Fraction for validation
            test_split: Fraction for test
        """
        n = len(self.data)

        # Shuffle indices
        indices = np.random.permutation(n)

        # Split
        test_size = int(n * test_split)
        val_size = int(n * validation_split)

        self.test_idx = indices[:test_size]
        self.validation_idx = indices[test_size:test_size + val_size]
        self.train_idx = indices[test_size + val_size:]

        self.test_count = len(self.test_idx)
        self.validation_count = len(self.validation_idx)
        self.train_count = len(self.train_idx)

        # Create length-based buckets for batch grouping
        self._create_length_buckets()

    def _create_length_buckets(self):
        """Create buckets of trace indices grouped by length"""
        self.length_buckets = {
            '3-5': [],
            '6-8': [],
            '9-12': [],
            '13-15': []
        }

        for idx in self.train_idx:
            trace_len = len(self.data[idx])

            if 3 <= trace_len <= 5:
                self.length_buckets['3-5'].append(idx)
            elif 6 <= trace_len <= 8:
                self.length_buckets['6-8'].append(idx)
            elif 9 <= trace_len <= 12:
                self.length_buckets['9-12'].append(idx)
            elif 13 <= trace_len <= 15:
                self.length_buckets['13-15'].append(idx)

        # Log bucket sizes
        for bucket_name, indices in self.length_buckets.items():
            self.log(f'  Bucket {bucket_name}: {len(indices)} traces', date=False)

    def next_train_batch_grouped(self, batch_size):
        """
        Get next training batch with similar-length traces (reduces padding)

        Args:
            batch_size: Batch size

        Returns:
            (traces_batch, adjacency_batch, nodes_batch, features_batch)
        """
        # Filter non-empty buckets
        available_buckets = [name for name, indices in self.length_buckets.items()
                            if len(indices) >= batch_size]

        if not available_buckets:
            # Fallback to regular batch if no bucket has enough samples
            return self.next_train_batch(batch_size)

        # Randomly select bucket
        bucket_name = np.random.choice(available_buckets)
        bucket_indices = self.length_buckets[bucket_name]

        # Sample batch from bucket
        sampled_idx = np.random.choice(bucket_indices, size=batch_size, replace=False)

        traces_batch = [self.data[i] for i in sampled_idx]
        return traces_batch, self.data_A[sampled_idx], self.data_X[sampled_idx], self.data_F[sampled_idx]

    def next_train_batch(self, batch_size):
        """
        Get next training batch

        Args:
            batch_size: Batch size

        Returns:
            (traces_batch, adjacency_batch, nodes_batch, features_batch)
        """
        if self.train_counter + batch_size > self.train_count:
            self.train_counter = 0
            np.random.shuffle(self.train_idx)

        counter = self.train_counter
        self.train_counter += batch_size

        idx = self.train_idx[counter:counter + batch_size]

        traces_batch = [self.data[i] for i in idx]
        return traces_batch, self.data_A[idx], self.data_X[idx], self.data_F[idx]

    def next_validation_batch(self, batch_size):
        """
        Get next validation batch

        Args:
            batch_size: Batch size

        Returns:
            (traces_batch, adjacency_batch, nodes_batch, features_batch)
        """
        if self.validation_counter + batch_size > self.validation_count:
            self.validation_counter = 0

        counter = self.validation_counter
        self.validation_counter += batch_size

        idx = self.validation_idx[counter:counter + batch_size]

        traces_batch = [self.data[i] for i in idx]
        return traces_batch, self.data_A[idx], self.data_X[idx], self.data_F[idx]

    def matrices_to_trace(self, node_vector, adjacency_matrix=None, strict=True):
        """
        Convert node vector back to trace

        Args:
            node_vector: Node vector
            adjacency_matrix: Optional adjacency matrix (unused)
            strict: Stop at PAD token

        Returns:
            List of activity names
        """
        return self.encoder.decode_trace(node_vector, strict)

    def decode_flow_type(self, flow_value):
        """
        Decode flow type value

        Args:
            flow_value: Flow value from adjacency matrix

        Returns:
            Flow type label
        """
        return self.encoder.decode_flow_type(flow_value)

    def decode_batch(self, adjacency_batch, nodes_batch, strict=True):
        """
        Convert batch to traces

        Args:
            adjacency_batch: Adjacency matrices
            nodes_batch: Node vectors
            strict: Stop at PAD token

        Returns:
            List of traces
        """
        traces = []
        for node_vector in nodes_batch:
            trace = self.encoder.decode_trace(node_vector, strict)
            traces.append(trace)
        return traces

    def __len__(self):
        """Dataset size"""
        return self.__len

    def get_stats(self):
        """
        Get dataset statistics

        Returns:
            dict: Dataset statistics
        """
        return {
            'total_traces': len(self.data) if self.data else 0,
            'train_size': self.train_count,
            'val_size': self.validation_count,
            'test_size': self.test_count,
            'num_activities': self.activity_num_types,
            'num_flow_types': self.flow_num_types,
            'max_activities': self.max_activities
        }
