"""
Trace Encoder
Convert event log traces to graph representation (adjacency matrices + node vectors)
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} {msg}')


class TraceEncoder:
    """
    Encode traces as graphs with control flow patterns

    Converts process traces to:
    - Adjacency matrices (with flow type edges)
    - Node vectors (activity sequences)
    - Feature matrices (activity embeddings)
    """

    def __init__(self, max_activities: int):
        """
        Args:
            max_activities: Maximum number of activities per trace
        """
        self.max_activities = max_activities

        # Activity encoding
        self.activity_encoder = {}
        self.activity_decoder = {}
        self.activity_num_types = 0

        # Flow type encoding
        self.flow_encoder = {}
        self.flow_decoder = {}
        self.flow_num_types = 0

        # Pattern map from Petri net
        self.pattern_map = {}
        self.dfg_frequencies = {}

    def create_encoders(self, traces: List[List[str]]):
        """
        Create activity and flow encoders from traces

        Args:
            traces: List of traces (each trace is list of activity names)
        """
        log('[Trace Encoder] Creating activity encoder...')

        # Extract unique activities
        activity_set = set()
        for trace in traces:
            activity_set.update(trace)

        # Add padding token
        activity_labels = sorted(list(activity_set)) + ['PAD']

        self.activity_encoder = {label: idx for idx,
                                 label in enumerate(activity_labels)}
        self.activity_decoder = {idx: label for idx,
                                 label in enumerate(activity_labels)}
        self.activity_num_types = len(activity_labels)

        log(
            f'[Trace Encoder] Created activity encoder with {self.activity_num_types} types: {list(activity_set)}')

        # Flow types (control flow patterns)
        flow_labels = [
            'SEQUENCE',      # A → B (direct sequence)
            'XOR_SPLIT',     # A → B OR C (exclusive choice)
            'AND_SPLIT',     # A → B & C (parallel fork)
            'LOOP',          # B → A (loop back)
            'SKIP',          # A → C (skip B)
        ]

        self.flow_encoder = {label: idx for idx,
                             label in enumerate(flow_labels)}
        self.flow_decoder = {idx: label for idx,
                             label in enumerate(flow_labels)}
        self.flow_num_types = len(flow_labels)

        log(f'[Trace Encoder] Created flow encoder with {self.flow_num_types} types')

    def set_patterns(self, pattern_map: Dict, dfg_frequencies: Dict):
        """
        Set discovered patterns from PatternDiscovery

        Args:
            pattern_map: {(src, dst) -> pattern_type}
            dfg_frequencies: Normalized DFG frequencies
        """
        self.pattern_map = pattern_map
        self.dfg_frequencies = dfg_frequencies

    def encode_traces(self, traces: List[List[str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert traces to graph representation

        Args:
            traces: List of traces (activity sequences)

        Returns:
            (data_A, data_X, data_F):
            - data_A: Adjacency matrices (n, max_activities, max_activities)
            - data_X: Node vectors (n, max_activities)
            - data_F: Feature matrices (n, max_activities, n_features)
        """
        log('[Trace Encoder] Converting traces to graph representation...')

        data_A = []
        data_X = []
        data_F = []
        valid_traces = []

        for i, trace in enumerate(traces):
            if len(trace) == 0 or len(trace) > self.max_activities:
                continue

            # Generate matrices
            A = self._trace_to_adjacency(trace)
            X = self._trace_to_nodes(trace)
            F = self._trace_to_features(trace)

            if A is not None and X is not None:
                data_A.append(A)
                data_X.append(X)
                data_F.append(F)
                valid_traces.append(trace)

        data_A = np.array(data_A, dtype=np.int32)
        data_X = np.array(data_X, dtype=np.int32)
        data_F = np.array(data_F, dtype=np.float32)

        log(
            f'[Trace Encoder] Converted {len(valid_traces)}/{len(traces)} traces to graphs')

        return data_A, data_X, data_F, valid_traces

    def _trace_to_adjacency(self, trace: List[str]) -> np.ndarray:
        """
        Convert trace to adjacency matrix with flow types

        Args:
            trace: List of activity names

        Returns:
            Adjacency matrix (max_activities, max_activities) with flow types
        """
        n = len(trace)
        A = np.zeros((self.max_activities, self.max_activities),
                     dtype=np.int32)

        for i in range(n - 1):
            current_activity = trace[i]
            next_activity = trace[i + 1]

            # Detect flow type
            flow_type = self._detect_flow_type(
                current_activity, next_activity, i, trace)

            # Add offset +1 to avoid zero (0 = padding/no edge)
            A[i, i + 1] = flow_type + 1

        return A

    def _detect_flow_type(self, current: str, next_act: str, position: int, trace: List[str]) -> int:
        """
        Detect flow type between two activities

        Priority:
        1. Petri Net patterns (highest confidence)
        2. Loop detection (trace-level)
        3. DFG frequency analysis
        4. SKIP detection (statistical)
        5. SEQUENCE fallback

        Args:
            current: Current activity
            next_act: Next activity
            position: Position in trace
            trace: Full trace

        Returns:
            Flow type index
        """
        # Check Petri net patterns first
        key = (current, next_act)
        if key in self.pattern_map:
            pattern = self.pattern_map[key]

            if pattern == 'XOR_SPLIT':
                if self._validate_xor_with_dfg(current, next_act):
                    return self.flow_encoder['XOR_SPLIT']

            elif pattern == 'AND_SPLIT':
                return self.flow_encoder['AND_SPLIT']

            elif pattern == 'LOOP':
                return self.flow_encoder['LOOP']

            # XOR_JOIN → treat as SEQUENCE
            elif pattern == 'XOR_JOIN':
                return self.flow_encoder['SEQUENCE']

        # Loop detection (trace-level)
        if next_act in trace[:position]:
            return self.flow_encoder['LOOP']

        # DFG-based XOR detection
        if self._detect_xor_from_dfg(current, next_act):
            return self.flow_encoder['XOR_SPLIT']

        # SKIP detection
        if position < len(trace) - 2:
            if self._detect_skip_pattern(current, next_act):
                return self.flow_encoder['SKIP']

        # Default: SEQUENCE
        return self.flow_encoder['SEQUENCE']

    def _validate_xor_with_dfg(self, current: str, next_act: str) -> bool:
        """
        Validate XOR split using DFG frequencies

        Args:
            current: Source activity
            next_act: Target activity

        Returns:
            True if XOR split validated
        """
        if not self.dfg_frequencies:
            return True

        # Get outgoing edges
        outgoing = {dst: freq for (src, dst), freq in self.dfg_frequencies.items() if src == current}

        if len(outgoing) < 2 or next_act not in outgoing:
            return False

        # Check no dominant path (> 80%)
        return max(outgoing.values()) <= 0.8

    def _detect_xor_from_dfg(self, current: str, next_act: str) -> bool:
        """
        Detect XOR split from DFG (fallback)

        Args:
            current: Source activity
            next_act: Target activity

        Returns:
            True if XOR detected
        """
        if not self.dfg_frequencies:
            return False

        outgoing = {dst: freq for (src, dst), freq in self.dfg_frequencies.items() if src == current}

        # 2+ alternatives, reasonably balanced
        return len(outgoing) >= 2 and next_act in outgoing and max(outgoing.values()) < 0.8

    def _detect_skip_pattern(self, current: str, next_act: str) -> bool:
        """
        Detect SKIP pattern (statistical)

        Args:
            current: Current activity
            next_act: Next activity

        Returns:
            True if SKIP detected
        """
        if not self.dfg_frequencies:
            return False

        key = (current, next_act)
        if key not in self.dfg_frequencies:
            return False

        freq = self.dfg_frequencies[key]

        # Low frequency edge (< 10%) between common activities
        if freq < 0.1:
            current_freq = sum(
                f for (s, d), f in self.dfg_frequencies.items() if s == current)
            next_freq = sum(
                f for (s, d), f in self.dfg_frequencies.items() if d == next_act)

            if current_freq > 0.2 and next_freq > 0.2:
                return True

        return False

    def _trace_to_nodes(self, trace: List[str]) -> np.ndarray:
        """
        Convert trace to node vector

        Args:
            trace: List of activity names

        Returns:
            Node vector (max_activities,)
        """
        X = np.zeros(self.max_activities, dtype=np.int32)

        for i, activity in enumerate(trace):
            X[i] = self.activity_encoder[activity]

        # Padding
        for i in range(len(trace), self.max_activities):
            X[i] = self.activity_encoder['PAD']

        return X

    def _trace_to_features(self, trace: List[str]) -> np.ndarray:
        """
        Extract features (one-hot encoding)

        Args:
            trace: List of activity names

        Returns:
            Feature matrix (max_activities, n_features)
        """
        n_features = self.activity_num_types
        F = np.zeros((self.max_activities, n_features), dtype=np.float32)

        for i, activity in enumerate(trace):
            activity_idx = self.activity_encoder[activity]
            F[i, activity_idx] = 1.0

        return F

    def decode_trace(self, node_vector: np.ndarray, strict: bool = True) -> List[str]:
        """
        Convert node vector back to trace

        Args:
            node_vector: Node vector (max_activities,)
            strict: Stop at first PAD token

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

    def decode_flow_type(self, flow_value: int) -> str:
        """
        Decode flow type value (with offset correction)

        Args:
            flow_value: Value from adjacency matrix

        Returns:
            Flow type label
        """
        if flow_value == 0:
            return 'PADDING'

        flow_idx = flow_value - 1
        return self.flow_decoder.get(flow_idx, 'UNKNOWN')
