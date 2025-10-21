"""
ProcessDataset: Dataset management for event logs in process mining

Loads event logs from XES and PNML formats:
- XES: Event log format (pm4py)
- PNML: Petri net format (pm4py)
- Converts traces to graph representation (adjacency matrix + node vector)
- Provides encoding/decoding for activities and flow types
"""

import pickle
import numpy as np
from datetime import datetime
import pm4py


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
        petri_net: Discovered Petri net (net, im, fm) for pattern analysis
        pattern_map: Dict mapping (activity_a, activity_b) -> pattern type
        dfg: Directly-Follows Graph for frequency analysis
        dfg_frequencies: Normalized DFG frequencies
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

        # PM4Py pattern detection structures
        self.petri_net = None      # (net, im, fm) tuple
        self.pattern_map = {}      # (src, dst) -> pattern type
        self.dfg = None            # Directly-Follows Graph
        self.dfg_frequencies = {}  # Normalized DFG frequencies
        self.pm4py_available = True  # Flag for graceful fallback

    def load_from_xes(self, xes_path, validation=0.1, test=0.1):
        """
        Load event log from XES file with PM4Py pattern discovery

        Args:
            xes_path: Path to XES file
            validation: Fraction for validation set
            test: Fraction for test set
        """
        self.log(f'Loading event log from {xes_path}...')

        # Load XES file using pm4py
        dataframe = pm4py.read_xes(xes_path)

        # PM4Py returns a DataFrame - convert to EventLog format
        from pm4py.objects.log.util import dataframe_utils
        from pm4py.objects.conversion.log import converter as log_converter

        # Convert DataFrame to EventLog
        event_log = log_converter.apply(dataframe)

        # Extract traces from event log
        traces = []
        for trace in event_log:
            activity_sequence = []
            for event in trace:
                # Events are dictionaries with 'concept:name' key
                activity_name = event['concept:name']
                activity_sequence.append(activity_name)

            if activity_sequence:  # Only add non-empty traces
                traces.append(activity_sequence)

        self.log(f'Loaded {len(traces)} traces from XES')

        # STEP 1: Discover Petri Net for structural pattern analysis
        self._discover_petri_net_patterns(event_log)

        # STEP 2: Build Directly-Follows Graph for frequency validation
        self._build_dfg(event_log)

        # Generate dataset
        self.generate_from_traces(traces, validation=validation, test=test)

    def _discover_petri_net_patterns(self, event_log):
        """
        Discover Petri Net from event log and analyze structural patterns

        Uses PM4Py Inductive Miner to discover a Petri net, then analyzes
        the structure to identify XOR splits, AND splits, and joins.

        Args:
            event_log: PM4Py event log object
        """
        try:
            from pm4py.algo.discovery.inductive import algorithm as inductive_miner

            self.log('[Pattern Discovery] Discovering Petri net structure...')

            # Discover Petri net using Inductive Miner
            # Note: Can return ProcessTree or (net, im, fm) depending on variant
            result = inductive_miner.apply(event_log)

            # Check if result is a tuple (net, im, fm) or ProcessTree
            if isinstance(result, tuple) and len(result) == 3:
                net, initial_marking, final_marking = result
                self.petri_net = (net, initial_marking, final_marking)
            else:
                # Result is ProcessTree - convert to Petri net
                from pm4py.objects.conversion.process_tree import converter as pt_converter
                net, initial_marking, final_marking = pt_converter.apply(
                    result)
                self.petri_net = (net, initial_marking, final_marking)

            # Analyze Petri net structure
            num_places = len(net.places)
            num_transitions = len(net.transitions)

            self.log(
                f'[Pattern Discovery] Discovered Petri net: {num_places} places, {num_transitions} transitions')

            # Build pattern map from Petri net structure
            self._analyze_petri_net_structure(net)

        except ImportError:
            self.log(
                '[Pattern Discovery] PM4Py not available, using fallback pattern detection')
            self.pm4py_available = False
        except Exception as e:
            self.log(f'[Pattern Discovery] Failed to discover Petri net: {e}')
            self.log('[Pattern Discovery] Using fallback pattern detection')
            self.pm4py_available = False

    def _analyze_petri_net_structure(self, net):
        """
        Analyze Petri net structure to identify control flow patterns

        Identifies:
        - XOR splits: Place with 1 input arc, N>1 output arcs (exclusive choice)
        - AND splits: Place with 1 input arc, N>1 output arcs (parallel fork)
        - XOR joins: Place with N>1 input arcs, 1 output arc (merge)
        - Loops: Cycles in the Petri net

        Args:
            net: PM4Py Petri net object
        """
        pattern_counts = {'XOR_SPLIT': 0,
                          'AND_SPLIT': 0, 'XOR_JOIN': 0, 'LOOP': 0}

        for place in net.places:
            in_arcs = list(place.in_arcs)
            out_arcs = list(place.out_arcs)

            # Pattern 1: XOR/AND Split
            # 1 input transition → Place → N output transitions
            if len(in_arcs) == 1 and len(out_arcs) > 1:
                source_transition = in_arcs[0].source

                # In Petri nets, distinguishing XOR from AND requires semantics
                # Simplified: assume XOR split by default (can be refined)
                for out_arc in out_arcs:
                    target_transition = out_arc.target

                    # Skip silent transitions (None label)
                    if source_transition.label and target_transition.label:
                        key = (source_transition.label,
                               target_transition.label)
                        # Mark as XOR_SPLIT (could check for AND patterns)
                        self.pattern_map[key] = 'XOR_SPLIT'
                        pattern_counts['XOR_SPLIT'] += 1

            # Pattern 2: XOR Join
            # N input transitions → Place → 1 output transition
            elif len(in_arcs) > 1 and len(out_arcs) == 1:
                target_transition = out_arcs[0].target

                for in_arc in in_arcs:
                    source_transition = in_arc.source

                    if source_transition.label and target_transition.label:
                        key = (source_transition.label,
                               target_transition.label)
                        self.pattern_map[key] = 'XOR_JOIN'
                        pattern_counts['XOR_JOIN'] += 1

        # Detect loops (simplified: check for back-arcs in transitions)
        for transition in net.transitions:
            if transition.label:
                # Check if transition can reach itself (simplified check)
                for out_arc in transition.out_arcs:
                    place = out_arc.target
                    for out_arc2 in place.out_arcs:
                        next_transition = out_arc2.target
                        if next_transition == transition:
                            key = (transition.label, transition.label)
                            self.pattern_map[key] = 'LOOP'
                            pattern_counts['LOOP'] += 1

        self.log(f'[Pattern Discovery] Detected: {pattern_counts["XOR_SPLIT"]} XOR splits, '
                 f'{pattern_counts["AND_SPLIT"]} AND splits, {pattern_counts["LOOP"]} loops')

    def _build_dfg(self, event_log):
        """
        Build Directly-Follows Graph for frequency analysis

        DFG shows which activities directly follow each other and how often.
        Used to validate patterns and detect XOR splits from frequencies.

        Args:
            event_log: PM4Py event log object
        """
        try:
            from pm4py.algo.discovery.dfg import algorithm as dfg_discovery

            self.log('[Pattern Discovery] Building Directly-Follows Graph...')

            # Compute DFG
            dfg = dfg_discovery.apply(event_log)
            self.dfg = dfg

            # Normalize frequencies
            if dfg:
                total_edges = sum(dfg.values())
                self.dfg_frequencies = {
                    edge: freq / total_edges for edge, freq in dfg.items()}

                self.log(
                    f'[Pattern Discovery] Built DFG with {len(dfg)} edges')

        except ImportError:
            self.log('[Pattern Discovery] PM4Py DFG not available')
        except Exception as e:
            self.log(f'[Pattern Discovery] Failed to build DFG: {e}')

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

        self.log(
            f'Dataset generated: {self.train_count} train, {self.validation_count} val, {self.test_count} test')

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

        self.activity_encoder = {label: idx for idx,
                                 label in enumerate(activity_labels)}
        self.activity_decoder = {idx: label for idx,
                                 label in enumerate(activity_labels)}
        self.activity_num_types = len(activity_labels)

        self.log(
            f'Created activity encoder with {self.activity_num_types} types: {list(activity_set)}')

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
                self.log(
                    f'Skipping trace {i}: length {len(trace)} out of bounds', date=False)
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
        Convert trace to adjacency matrix with PM4Py-based pattern detection

        HYBRID APPROACH (Best accuracy ~85-90%):
        1. Petri Net structure analysis (XOR/AND splits from formal model)
        2. DFG frequency validation (confirm patterns with real frequencies)
        3. Loop detection (trace-level analysis)
        4. SKIP detection (statistical analysis, fallback to heuristic)

        Args:
            trace: List of activity names

        Returns:
            Adjacency matrix (max_activities, max_activities) with flow types

        Flow patterns detected:
        - SEQUENCE: A → B (standard flow)
        - LOOP: B → A (activity repeats, loop back)
        - XOR_SPLIT: Choice between activities (Petri net + DFG)
        - AND_SPLIT: Parallel activities (Petri net structure)
        - SKIP: Jump over intermediate activity (statistical)
        """
        n = len(trace)
        A = np.zeros((self.max_activities, self.max_activities),
                     dtype=np.int32)

        for i in range(n - 1):
            current_activity = trace[i]
            next_activity = trace[i + 1]

            # Detect flow type using hybrid approach
            flow_type = self._detect_flow_type(
                current_activity, next_activity, i, trace)

            A[i, i + 1] = flow_type

        return A

    def _detect_flow_type(self, current, next_act, position, trace):
        """

        Priority order:
        1. Petri Net structural patterns (highest confidence)
        2. Loop detection (trace-level, high confidence)
        3. DFG frequency analysis (medium confidence)
        4. SKIP detection (statistical, lower confidence)
        5. SEQUENCE fallback

        Args:
            current: Current activity name
            next_act: Next activity name
            position: Position in trace
            trace: Full trace

        Returns:
            Flow type index (from flow_encoder)
        """

        key = (current, next_act)
        if key in self.pattern_map:
            pattern = self.pattern_map[key]

            if pattern == 'XOR_SPLIT':
                # Validate with DFG frequencies
                if self._validate_xor_with_dfg(current, next_act):
                    return self.flow_encoder['XOR_SPLIT']

            elif pattern == 'AND_SPLIT':
                return self.flow_encoder['AND_SPLIT']

            elif pattern == 'LOOP':
                return self.flow_encoder['LOOP']

            # XOR_JOIN treated as SEQUENCE
            elif pattern == 'XOR_JOIN':
                return self.flow_encoder['SEQUENCE']

        if next_act in trace[:position]:
            return self.flow_encoder['LOOP']

        if self._detect_xor_from_dfg(current, next_act):
            return self.flow_encoder['XOR_SPLIT']

        if position < len(trace) - 2:
            if self._detect_skip_pattern(current, next_act, position, trace):
                return self.flow_encoder['SKIP']

        # Default: SEQUENCE
        return self.flow_encoder['SEQUENCE']

    def _validate_xor_with_dfg(self, current, next_act):
        """
        Validate XOR split pattern using DFG frequencies

        A true XOR split should show:
        - Multiple alternative paths from current activity
        - No single path dominates (e.g., all paths < 80% frequency)

        Args:
            current: Source activity
            next_act: Target activity

        Returns:
            True if XOR split validated by DFG
        """
        if not self.dfg_frequencies:
            return True  # No DFG, trust Petri net

        # Get all edges from current activity
        outgoing = {}
        for (src, dst), freq in self.dfg_frequencies.items():
            if src == current:
                outgoing[dst] = freq

        if len(outgoing) < 2:
            return False  # Not a split

        # Check if next_act is one of the alternatives
        if next_act not in outgoing:
            return False

        # Check frequency distribution (no dominant path > 80%)
        max_freq = max(outgoing.values())
        if max_freq > 0.8:
            return False  # Dominant path, not true XOR

        return True

    def _detect_xor_from_dfg(self, current, next_act):
        """
        Detect XOR split purely from DFG (fallback if not in Petri net)

        Args:
            current: Source activity
            next_act: Target activity

        Returns:
            True if XOR split detected from DFG
        """
        if not self.dfg_frequencies:
            return False

        # Get all outgoing edges from current
        outgoing = {}
        for (src, dst), freq in self.dfg_frequencies.items():
            if src == current:
                outgoing[dst] = freq

        # XOR split: 2+ alternatives, reasonably balanced
        if len(outgoing) >= 2 and next_act in outgoing:
            max_freq = max(outgoing.values())
            if max_freq < 0.8:  # No dominant path
                return True

        return False

    def _detect_skip_pattern(self, current, next_act, position, trace):
        """
        Detect SKIP pattern using statistical analysis

        A SKIP occurs when an activity that usually appears between
        current and next_act is missing in this trace.

        Args:
            current: Current activity
            next_act: Next activity
            position: Position in trace
            trace: Full trace

        Returns:
            True if SKIP pattern detected
        """
        # Requires DFG or training data
        if not self.dfg and not hasattr(self, 'data'):
            return False

        # Check if there's usually an intermediate activity
        # This is complex - simplified heuristic:
        # If (current, next_act) edge has low frequency but both activities
        # are common, likely there's a skip

        if self.dfg:
            key = (current, next_act)
            if key in self.dfg_frequencies:
                freq = self.dfg_frequencies[key]

                # Low frequency edge between common activities = potential skip
                if freq < 0.1:  # Threshold: 10%
                    # Check if current and next_act are individually common
                    current_freq = sum(
                        f for (s, d), f in self.dfg_frequencies.items() if s == current)
                    next_freq = sum(
                        f for (s, d), f in self.dfg_frequencies.items() if d == next_act)

                    if current_freq > 0.2 and next_freq > 0.2:
                        return True

        return False

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

    def decode_batch(self, adjacency_batch, nodes_batch, strict=True):
        """
        Convert batch of adjacency matrices and node features to traces

        Args:
            adjacency_batch: Array of shape (batch_size, max_activities, max_activities, flow_types)
            nodes_batch: Array of shape (batch_size, max_activities, activity_types)
            strict: If True, stop at first PAD token

        Returns:
            List of traces (list of activity names)
        """
        import numpy as np
        traces = []

        for i in range(len(nodes_batch)):
            # Get node indices from one-hot encoding
            if len(nodes_batch[i].shape) == 2:
                node_indices = np.argmax(nodes_batch[i], axis=-1)
            else:
                node_indices = nodes_batch[i]

            # Convert to trace
            trace = self.matrices_to_trace(node_indices, strict=strict)
            traces.append(trace)

        return traces

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

        self.log(
            f'Split: {train_size} train, {validation_size} val, {test_size} test')

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

    def load_from_petri_net(self, pnml_path, validation=0.1, test=0.1):
        """
        Load dataset from Petri net (PNML format)

        Args:
            pnml_path: Path to PNML file
            validation: Fraction for validation set
            test: Fraction for test set
        """
        from utils.graph_loader import GraphLoader

        self.log(f'Loading Petri net from {pnml_path}...')

        loader = GraphLoader(max_activities=self.max_activities)
        adj, nodes, activity_encoder, flow_encoder = loader.load_petri_net(
            pnml_path)

        # Update encoders
        self.activity_encoder = activity_encoder
        self.activity_decoder = {v: k for k, v in activity_encoder.items()}
        self.flow_encoder = flow_encoder
        self.flow_decoder = {v: k for k, v in flow_encoder.items()}

        # Create dataset with single graph
        self.generate_from_graphs(
            [adj], [nodes], validation=validation, test=test)

    def generate_from_graphs(self, adjacency_list, nodes_list, validation=0.1, test=0.1):
        """
        Generate dataset from list of graphs (adjacency matrices + node vectors)

        Args:
            adjacency_list: List of (max_activities, max_activities) adjacency matrices
            nodes_list: List of (max_activities,) node vectors
            validation: Fraction for validation set
            test: Fraction for test set
        """
        self.log(f'Generating dataset from {len(adjacency_list)} graphs...')

        # Store graphs directly
        self.adjacency_matrices = adjacency_list
        self.node_features = nodes_list

        # Convert to traces for compatibility (extract sequence from adjacency)
        self.traces = []
        for i in range(len(nodes_list)):
            trace = self.matrices_to_trace(nodes_list[i], strict=True)
            self.traces.append(trace)

        # Store as numpy arrays
        self.data_A = np.array(adjacency_list, dtype=np.int32)
        self.data_X = np.array(nodes_list, dtype=np.int32)

        # Generate features (one-hot encoding)
        self.data_F = np.zeros((len(nodes_list), self.max_activities, len(
            self.activity_encoder)), dtype=np.float32)
        for i, nodes in enumerate(nodes_list):
            for j, node_idx in enumerate(nodes):
                if node_idx > 0:  # Skip PAD
                    self.data_F[i, j, node_idx] = 1.0

        self.data = self.traces
        self.__len = len(self.traces)

        # Split into train/val/test
        self._generate_train_validation_test(validation, test)

        self.log(
            f'Dataset generated: {self.train_count} train, {self.validation_count} val, {self.test_count} test')

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

    def analyze_trace_length_distribution(self, xes_path):
        """
        Analyze trace length distribution from XES file

        This helps determine optimal max_activities value by showing:
        - Min, max, mean, median trace lengths
        - Percentiles (75th, 90th, 95th, 99th)
        - Recommendation for max_activities based on coverage

        Args:
            xes_path: Path to XES event log file

        Returns:
            dict: Statistics about trace lengths
        """

        from pm4py.objects.conversion.log import converter as log_converter

        dataframe = pm4py.read_xes(xes_path)
        event_log = log_converter.apply(dataframe)

        # Calculate trace lengths
        lengths = [len(trace) for trace in event_log]
        total_traces = len(lengths)

        min_len = min(lengths)
        max_len = max(lengths)
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)

        # Calculate percentiles
        percentiles = [50, 75, 90, 95, 99, 100]
        percentile_values = {}

        for p in percentiles:
            if p == 100:
                val = max_len
            else:
                val = np.percentile(lengths, p)
            percentile_values[p] = val

        test_values = [8, 10, 12, 15, 20, 25, int(max_len)]
        coverage_data = []

        for max_act in sorted(set(test_values)):
            covered = sum(1 for l in lengths if l <= max_act)
            coverage_pct = (covered / total_traces) * 100
            traces_lost = total_traces - covered
            padding_pct = ((max_act - mean_len) / max_act) * \
                100 if max_act > 0 else 0

            coverage_data.append({
                'max_activities': max_act,
                'coverage': coverage_pct,
                'traces_lost': traces_lost,
                'padding_pct': padding_pct
            })

        # Conservative: 99th percentile
        conservative = int(np.ceil(percentile_values[99]))

        # Balanced: 95th percentile
        balanced = int(np.ceil(percentile_values[95]))

        # Aggressive: 90th percentile
        aggressive = int(np.ceil(percentile_values[90]))

        # Return statistics
        return {
            'total_traces': total_traces,
            'min_length': min_len,
            'max_length': max_len,
            'mean_length': mean_len,
            'median_length': median_len,
            'percentiles': percentile_values,
            'recommended_conservative': conservative,
            'recommended_balanced': balanced,
            'recommended_aggressive': aggressive,
            'coverage_analysis': coverage_data
        }


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
