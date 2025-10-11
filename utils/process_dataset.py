"""
ProcessDataset: Dataset management for event logs in process mining

Adapts SparseMolecularDataset logic to process mining domain:
- Loads event logs from CSV/XES
- Converts traces to graph representation (adjacency matrix + node vector)
- Provides encoding/decoding for activities and flow types
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime


class ProcessDataset:
    """
    Dataset for process mining event logs

    Attributes:
        activity_encoder: Dict mapping activity names to indices
        activity_decoder: Dict mapping indices to activity names
        flow_encoder: Dict mapping flow types to indices
        flow_decoder: Dict mapping indices to flow types
        max_activities: Maximum number of activities per trace
        activity_num_types: Number of unique activity types
        flow_num_types: Number of flow control types
    """

    def __init__(self, max_activities=15):
        """
        Initialize ProcessDataset

        Args:
            max_activities: Maximum trace length (default: 15)
        """
        self.max_activities = max_activities

        # Encoders/decoders (populated during dataset generation)
        self.activity_encoder = {}
        self.activity_decoder = {}
        self.flow_encoder = {}
        self.flow_decoder = {}

        # Dataset arrays (populated after loading)
        self.data = None           # List of traces (as activity sequences)
        self.data_A = None         # Adjacency matrices (N, max_act, max_act)
        self.data_X = None         # Node vectors (N, max_act)
        self.data_F = None         # Features (N, max_act, n_features)

        # Train/val/test splits
        self.train_idx = None
        self.validation_idx = None
        self.test_idx = None

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        # Counters
        self.train_count = 0
        self.validation_count = 0
        self.test_count = 0
        self.__len = 0

    def load_from_csv(self, filename, case_id_col='case_id', activity_col='activity',
                     timestamp_col='timestamp', validation=0.1, test=0.1):
        """
        Load event log from CSV file

        Args:
            filename: Path to CSV file
            case_id_col: Column name for case ID
            activity_col: Column name for activity
            timestamp_col: Column name for timestamp
            validation: Fraction of data for validation (default: 0.1)
            test: Fraction of data for test (default: 0.1)
        """
        self.log(f'Loading event log from {filename}...')

        # Read CSV
        df = pd.read_csv(filename)

        # Sort by case and timestamp
        df = df.sort_values([case_id_col, timestamp_col])

        # Group by case to get traces
        traces = []
        for case_id, group in df.groupby(case_id_col):
            trace = group[activity_col].tolist()
            traces.append(trace)

        self.log(f'Loaded {len(traces)} traces from CSV')

        # Generate dataset
        self.generate_from_traces(traces, validation=validation, test=test)

    def generate_from_traces(self, traces, validation=0.1, test=0.1):
        """
        Generate dataset from list of traces

        Args:
            traces: List of traces (each trace is list of activity names)
            validation: Fraction for validation set
            test: Fraction for test set
        """
        self.log(f'Generating dataset from {len(traces)} traces...')

        self.data = traces

        # Create encoders/decoders
        self._generate_encoders_decoders()

        # Convert traces to graph representation
        self._generate_adjacency_and_nodes()

        # Create train/val/test splits
        self._generate_train_validation_test(validation, test)

        self.log(f'Dataset generated: {self.train_count} train, {self.validation_count} val, {self.test_count} test')

    def _generate_encoders_decoders(self):
        """
        Create encoders/decoders for activities and flow types
        """
        self.log('Creating activity encoder and decoder...')

        # Extract unique activities
        activity_set = set()
        for trace in self.data:
            activity_set.update(trace)

        # Add padding token
        activity_labels = sorted(list(activity_set)) + ['PAD']

        self.activity_encoder = {label: idx for idx, label in enumerate(activity_labels)}
        self.activity_decoder = {idx: label for idx, label in enumerate(activity_labels)}
        self.activity_num_types = len(activity_labels)

        self.log(f'Created activity encoder with {self.activity_num_types} types: {list(activity_set)}')

        # Flow types (control flow patterns)
        flow_labels = [
            'SEQUENCE',      # A → B (direct sequence)
            'XOR_SPLIT',     # A → B OR C (exclusive choice)
            'AND_SPLIT',     # A → B & C (parallel fork)
            'LOOP',          # B → A (loop back)
            'SKIP',          # A → C (skip B)
        ]

        self.flow_encoder = {label: idx for idx, label in enumerate(flow_labels)}
        self.flow_decoder = {idx: label for idx, label in enumerate(flow_labels)}
        self.flow_num_types = len(flow_labels)

        self.log(f'Created flow encoder with {self.flow_num_types} types')

    def _generate_adjacency_and_nodes(self):
        """
        Convert traces to adjacency matrices and node vectors
        """
        self.log('Converting traces to graph representation...')

        data_A = []  # Adjacency matrices
        data_X = []  # Node vectors
        data_F = []  # Features (simple: one-hot encoding for now)

        valid_traces = []

        for i, trace in enumerate(self.data):
            if len(trace) == 0 or len(trace) > self.max_activities:
                self.log(f'Skipping trace {i}: length {len(trace)} out of bounds', date=False)
                continue

            # Generate adjacency matrix
            A = self._trace_to_adjacency(trace)

            # Generate node vector
            X = self._trace_to_nodes(trace)

            # Generate features (placeholder: one-hot encoding)
            F = self._trace_to_features(trace)

            if A is not None and X is not None:
                data_A.append(A)
                data_X.append(X)
                data_F.append(F)
                valid_traces.append(trace)

        self.data = valid_traces
        self.data_A = np.array(data_A, dtype=np.int32)
        self.data_X = np.array(data_X, dtype=np.int32)
        self.data_F = np.array(data_F, dtype=np.float32)

        self.__len = len(self.data)

        self.log(f'Converted {len(self.data)} valid traces to graphs')

    def _trace_to_adjacency(self, trace):
        """
        Convert trace to adjacency matrix

        Args:
            trace: List of activity names

        Returns:
            Adjacency matrix (max_activities, max_activities) with flow types
        """
        n = len(trace)
        A = np.zeros((self.max_activities, self.max_activities), dtype=np.int32)

        # Simple sequential flow (can be extended to detect patterns)
        for i in range(n - 1):
            # Check for patterns
            if i > 0 and trace[i] == trace[i-1]:
                # Loop pattern
                flow_type = self.flow_encoder['LOOP']
            else:
                # Default: sequential
                flow_type = self.flow_encoder['SEQUENCE']

            A[i, i+1] = flow_type

        return A

    def _trace_to_nodes(self, trace):
        """
        Convert trace to node vector

        Args:
            trace: List of activity names

        Returns:
            Node vector (max_activities,) with activity indices
        """
        X = np.zeros(self.max_activities, dtype=np.int32)

        for i, activity in enumerate(trace):
            X[i] = self.activity_encoder[activity]

        # Padding
        for i in range(len(trace), self.max_activities):
            X[i] = self.activity_encoder['PAD']

        return X

    def _trace_to_features(self, trace):
        """
        Extract features for each activity (placeholder implementation)

        Args:
            trace: List of activity names

        Returns:
            Feature matrix (max_activities, n_features)
        """
        # Simple feature: one-hot encoding of activity type
        n_features = self.activity_num_types
        F = np.zeros((self.max_activities, n_features), dtype=np.float32)

        for i, activity in enumerate(trace):
            activity_idx = self.activity_encoder[activity]
            F[i, activity_idx] = 1.0

        return F

    def matrices_to_trace(self, node_vector, adjacency_matrix=None, strict=True):
        """
        Convert node vector (and optionally adjacency) back to trace

        Args:
            node_vector: Array of shape (max_activities,) with activity indices
            adjacency_matrix: Optional adjacency matrix (not used in simple version)
            strict: If True, stop at first PAD token

        Returns:
            List of activity names
        """
        trace = []

        for node_idx in node_vector:
            activity = self.activity_decoder[int(node_idx)]

            if activity == 'PAD':
                if strict:
                    break
                else:
                    continue

            trace.append(activity)

        return trace

    def _generate_train_validation_test(self, validation, test):
        """
        Split dataset into train/validation/test

        Args:
            validation: Fraction for validation
            test: Fraction for test
        """
        self.log('Creating train, validation and test sets...')

        n = len(self)
        validation_size = int(validation * n)
        test_size = int(test * n)
        train_size = n - validation_size - test_size

        # Random permutation
        self.all_idx = np.random.permutation(n)
        self.train_idx = self.all_idx[:train_size]
        self.validation_idx = self.all_idx[train_size:train_size + validation_size]
        self.test_idx = self.all_idx[train_size + validation_size:]

        self.train_count = train_size
        self.validation_count = validation_size
        self.test_count = test_size

        self.log(f'Split: {train_size} train, {validation_size} val, {test_size} test')

    def _next_batch(self, counter, count, idx, batch_size):
        """
        Get next batch from dataset

        Args:
            counter: Current position in dataset
            count: Total size of split
            idx: Indices for this split
            batch_size: Batch size (None = return all)

        Returns:
            [new_counter, traces, adjacency, nodes, features]
        """
        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            batch_idx = idx[counter:counter + batch_size]
            output = [
                [self.data[i] for i in batch_idx],
                self.data_A[batch_idx],
                self.data_X[batch_idx],
                self.data_F[batch_idx]
            ]

            counter += batch_size
        else:
            batch_idx = idx
            output = [
                [self.data[i] for i in batch_idx],
                self.data_A[batch_idx],
                self.data_X[batch_idx],
                self.data_F[batch_idx]
            ]

        return [counter] + output

    def next_train_batch(self, batch_size=None):
        """Get next training batch"""
        out = self._next_batch(
            counter=self.train_counter,
            count=self.train_count,
            idx=self.train_idx,
            batch_size=batch_size
        )
        self.train_counter = out[0]
        return out[1:]

    def next_validation_batch(self, batch_size=None):
        """Get next validation batch"""
        out = self._next_batch(
            counter=self.validation_counter,
            count=self.validation_count,
            idx=self.validation_idx,
            batch_size=batch_size
        )
        self.validation_counter = out[0]
        return out[1:]

    def next_test_batch(self, batch_size=None):
        """Get next test batch"""
        out = self._next_batch(
            counter=self.test_counter,
            count=self.test_count,
            idx=self.test_idx,
            batch_size=batch_size
        )
        self.test_counter = out[0]
        return out[1:]

    def save(self, filename):
        """Save dataset to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        self.log(f'Dataset saved to {filename}')

    def load(self, filename):
        """Load dataset from file"""
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        self.log(f'Dataset loaded from {filename}')

    @staticmethod
    def log(msg='', date=True):
        """Log message with timestamp"""
        if date:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {msg}')
        else:
            print(msg)

    def __len__(self):
        """Return dataset size"""
        return self.__len

    def get_stats(self):
        """Get dataset statistics"""
        stats = {
            'total_traces': len(self),
            'train_size': self.train_count,
            'val_size': self.validation_count,
            'test_size': self.test_count,
            'num_activities': self.activity_num_types,
            'num_flow_types': self.flow_num_types,
            'max_trace_length': self.max_activities,
            'activity_types': list(self.activity_encoder.keys()),
            'flow_types': list(self.flow_encoder.keys()),
        }

        # Length distribution
        trace_lengths = [len(trace) for trace in self.data]
        stats['avg_trace_length'] = np.mean(trace_lengths)
        stats['min_trace_length'] = np.min(trace_lengths)
        stats['max_trace_length'] = np.max(trace_lengths)

        return stats


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("ProcessDataset Example")
    print("=" * 60)

    # Create synthetic traces
    synthetic_traces = [
        ['Start', 'Submit', 'Review', 'Approve', 'End'],
        ['Start', 'Submit', 'Review', 'Reject', 'End'],
        ['Start', 'Submit', 'Approve', 'End'],
        ['Start', 'Submit', 'Review', 'Approve', 'Send', 'End'],
        ['Start', 'Submit', 'Review', 'Rework', 'Submit', 'Approve', 'End'],
    ]

    # Create dataset
    dataset = ProcessDataset(max_activities=10)
    dataset.generate_from_traces(synthetic_traces, validation=0.2, test=0.2)

    # Print stats
    print("\nDataset Statistics:")
    stats = dataset.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test batch loading
    print("\nTesting batch loading...")
    traces, adj, nodes, features = dataset.next_train_batch(batch_size=2)
    print(f"  Batch size: {len(traces)}")
    print(f"  Adjacency shape: {adj.shape}")
    print(f"  Nodes shape: {nodes.shape}")
    print(f"  Features shape: {features.shape}")

    # Test conversion
    print("\nTesting conversion back to trace:")
    reconstructed_trace = dataset.matrices_to_trace(nodes[0])
    print(f"  Original: {traces[0]}")
    print(f"  Reconstructed: {reconstructed_trace}")

    print("\n" + "=" * 60)
    print("ProcessDataset test completed successfully!")
    print("=" * 60)
