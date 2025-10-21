"""
Graph Loader for ProcessGAN

Loads Petri nets (PNML format) and converts them to ProcessGAN's internal 
representation (adjacency matrices + node features).

Supported formats:
- Petri Nets (PNML via pm4py)

Author: ProcessGAN Team
Date: 2024
"""

import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path

try:
    import pm4py
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    print("[WARNING] pm4py not available. Petri net loading disabled.")


class GraphLoader:
    """
    Unified loader for various graph formats used in process mining.

    Converts different graph representations to ProcessGAN format:
    - Adjacency matrix: (N, N) with flow type indices
    - Node features: (N,) with activity indices
    - Activity encoder: {activity_name: index}
    - Flow encoder: {flow_type: index}
    """

    def __init__(self, max_activities: int = 15):
        """
        Initialize graph loader

        Args:
            max_activities: Maximum number of activities/nodes in graph
        """
        self.max_activities = max_activities

        # Standard flow type encoding
        self.flow_encoder = {
            'NONE': 0,
            'SEQUENCE': 1,
            'PARALLEL': 2,
            'LOOP': 3,
            'CHOICE': 4,
            'SKIP': 5
        }
        self.flow_decoder = {v: k for k, v in self.flow_encoder.items()}

        # Activity encoder (to be built from data)
        self.activity_encoder = {'PAD': 0}
        self.activity_decoder = {0: 'PAD'}
        self.next_activity_idx = 1

    def _encode_activity(self, activity_name: str) -> int:
        """Encode activity name to index"""
        if activity_name not in self.activity_encoder:
            self.activity_encoder[activity_name] = self.next_activity_idx
            self.activity_decoder[self.next_activity_idx] = activity_name
            self.next_activity_idx += 1
        return self.activity_encoder[activity_name]

    def _pad_graph(self, adjacency: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad graph to max_activities size

        Args:
            adjacency: (n, n) adjacency matrix
            nodes: (n,) node vector

        Returns:
            Padded (max_activities, max_activities) adjacency, (max_activities,) nodes
        """
        n = len(nodes)
        if n > self.max_activities:
            print(f"[WARNING] Graph has {n} nodes, truncating to {self.max_activities}")
            adjacency = adjacency[:self.max_activities, :self.max_activities]
            nodes = nodes[:self.max_activities]
            n = self.max_activities

        # Pad adjacency
        padded_adj = np.zeros((self.max_activities, self.max_activities), dtype=np.int32)
        padded_adj[:n, :n] = adjacency

        # Pad nodes
        padded_nodes = np.zeros(self.max_activities, dtype=np.int32)
        padded_nodes[:n] = nodes

        return padded_adj, padded_nodes

    # ================================================================
    # Petri Net Loading
    # ================================================================

    def load_petri_net(self, pnml_path: str) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Load Petri net from PNML file

        Args:
            pnml_path: Path to PNML file

        Returns:
            (adjacency, nodes, activity_encoder, flow_encoder)

        Raises:
            ImportError: If pm4py not installed
            FileNotFoundError: If file doesn't exist
        """
        if not PM4PY_AVAILABLE:
            raise ImportError("pm4py is required for Petri net loading. Install with: pip install pm4py")

        if not Path(pnml_path).exists():
            raise FileNotFoundError(f"PNML file not found: {pnml_path}")

        print(f"[INFO] Loading Petri net from {pnml_path}...")

        # Load Petri net using pm4py
        net, initial_marking, final_marking = pm4py.read_pnml(pnml_path)

        # Extract transitions (activities) and arcs (flows)
        transitions = list(net.transitions)
        places = list(net.places)
        arcs = list(net.arcs)

        print(f"[INFO] Petri net has {len(transitions)} transitions, {len(places)} places, {len(arcs)} arcs")

        # Build activity list from transitions
        activity_names = []
        transition_to_idx = {}

        for i, trans in enumerate(transitions):
            # Use transition label if available, otherwise use name
            label = trans.label if trans.label else trans.name
            activity_names.append(label)
            transition_to_idx[trans] = i

        # Build node vector
        n = len(activity_names)
        nodes = np.zeros(n, dtype=np.int32)
        for i, name in enumerate(activity_names):
            nodes[i] = self._encode_activity(name)

        # Build adjacency matrix from arcs
        adjacency = np.zeros((n, n), dtype=np.int32)

        for arc in arcs:
            # Petri net has place-transition and transition-place arcs
            # We need to find transition-transition connections through places
            pass

        # Alternative: build transition-transition adjacency using place connections
        adjacency = self._build_transition_adjacency(net, transitions, transition_to_idx)

        # Pad to max_activities
        adjacency, nodes = self._pad_graph(adjacency, nodes)

        print(f"[INFO] Converted to {len(activity_names)} activities")

        return adjacency, nodes, self.activity_encoder.copy(), self.flow_encoder.copy()

    def _build_transition_adjacency(self, net, transitions, transition_to_idx):
        """
        Build transition-transition adjacency matrix from Petri net

        Two transitions are connected if:
        - First transition has output place
        - That place is input to second transition
        """
        n = len(transitions)
        adjacency = np.zeros((n, n), dtype=np.int32)

        # Build transition -> output places mapping
        trans_outputs = {trans: [] for trans in transitions}
        place_inputs = {place: [] for place in net.places}

        for arc in net.arcs:
            if arc.source in transitions:
                # Transition -> Place arc
                trans_outputs[arc.source].append(arc.target)
            elif arc.target in transitions:
                # Place -> Transition arc
                place_inputs[arc.target].append(arc.source)

        # Connect transitions through places
        for trans1 in transitions:
            idx1 = transition_to_idx[trans1]

            # For each output place of trans1
            for place in trans_outputs.get(trans1, []):
                # Find transitions that have this place as input
                for trans2 in transitions:
                    if trans1 != trans2:
                        # Check if place is input to trans2
                        for arc in net.arcs:
                            if arc.source == place and arc.target == trans2:
                                idx2 = transition_to_idx[trans2]

                                # Detect flow type
                                flow_type = self._detect_flow_type(net, trans1, place, trans2)
                                adjacency[idx1, idx2] = flow_type

        return adjacency

    def _detect_flow_type(self, net, trans1, place, trans2) -> int:
        """
        Detect flow type between two transitions in Petri net

        Heuristics:
        - Multiple outputs from place → PARALLEL (AND-split)
        - Back-edge (creates cycle) → LOOP
        - Multiple inputs to place → CHOICE (XOR-join)
        - Default → SEQUENCE
        """
        # Count outputs from place
        outputs = [arc for arc in net.arcs if arc.source == place]
        if len(outputs) > 1:
            return self.flow_encoder['PARALLEL']

        # Check for loops (simplified: check if trans2 appears before trans1 in some ordering)
        # This is heuristic-based
        if hasattr(trans1, 'name') and hasattr(trans2, 'name'):
            if trans2.name < trans1.name:  # Heuristic: backward flow
                return self.flow_encoder['LOOP']

        # Count inputs to place
        inputs = [arc for arc in net.arcs if arc.target == place]
        if len(inputs) > 1:
            return self.flow_encoder['CHOICE']

        # Default: sequential
        return self.flow_encoder['SEQUENCE']


# ================================================================
# Convenience Functions
# ================================================================

def load_petri_net_file(file_path: str, max_activities: int = 15) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Load Petri net from PNML file (convenience function)

    Args:
        file_path: Path to PNML file
        max_activities: Maximum number of activities

    Returns:
        (adjacency, nodes, activity_encoder, flow_encoder)
    """
    loader = GraphLoader(max_activities=max_activities)
    return loader.load_petri_net(file_path)
