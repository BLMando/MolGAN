"""
Graph Loader for ProcessGAN

Loads various graph formats (Petri nets, BPMN, GraphML, NetworkX) and converts
them to ProcessGAN's internal representation (adjacency matrices + node features).

Supported formats:
- Petri Nets (PNML via pm4py)
- BPMN (BPMN 2.0 XML via pm4py)
- GraphML (via NetworkX)
- NetworkX DiGraph (direct Python objects)
- Raw adjacency matrices + node lists

Author: ProcessGAN Team
Date: 2024
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    import pm4py
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    print("[WARNING] pm4py not available. Petri net and BPMN loading disabled.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("[WARNING] NetworkX not available. GraphML and NetworkX loading disabled.")


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
    # BPMN Loading
    # ================================================================

    def load_bpmn(self, bpmn_path: str) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Load BPMN model from XML file

        Args:
            bpmn_path: Path to BPMN XML file

        Returns:
            (adjacency, nodes, activity_encoder, flow_encoder)
        """
        if not PM4PY_AVAILABLE:
            raise ImportError("pm4py is required for BPMN loading. Install with: pip install pm4py")

        if not Path(bpmn_path).exists():
            raise FileNotFoundError(f"BPMN file not found: {bpmn_path}")

        print(f"[INFO] Loading BPMN from {bpmn_path}...")

        # Load BPMN using pm4py
        bpmn_graph = pm4py.read_bpmn(bpmn_path)

        # Extract nodes (tasks, gateways, events)
        nodes_list = bpmn_graph.get_nodes()
        flows = bpmn_graph.get_flows()

        print(f"[INFO] BPMN has {len(nodes_list)} nodes, {len(flows)} flows")

        # Filter to tasks only (activities)
        tasks = [node for node in nodes_list if hasattr(node, 'get_name')
                and node.__class__.__name__ in ['Task', 'Activity']]

        # Build activity list
        activity_names = []
        node_to_idx = {}

        for i, task in enumerate(tasks):
            name = task.get_name() if hasattr(task, 'get_name') else f"Task_{i}"
            activity_names.append(name)
            node_to_idx[task.get_id() if hasattr(task, 'get_id') else task] = i

        # Build node vector
        n = len(activity_names)
        nodes_vec = np.zeros(n, dtype=np.int32)
        for i, name in enumerate(activity_names):
            nodes_vec[i] = self._encode_activity(name)

        # Build adjacency from flows
        adjacency = np.zeros((n, n), dtype=np.int32)

        for flow in flows:
            source_id = flow.get_source() if hasattr(flow, 'get_source') else None
            target_id = flow.get_target() if hasattr(flow, 'get_target') else None

            if source_id in node_to_idx and target_id in node_to_idx:
                idx1 = node_to_idx[source_id]
                idx2 = node_to_idx[target_id]

                # Detect flow type from gateways
                flow_type = self._detect_bpmn_flow_type(bpmn_graph, flow, source_id, target_id)
                adjacency[idx1, idx2] = flow_type

        # Pad to max_activities
        adjacency, nodes_vec = self._pad_graph(adjacency, nodes_vec)

        print(f"[INFO] Converted to {len(activity_names)} activities")

        return adjacency, nodes_vec, self.activity_encoder.copy(), self.flow_encoder.copy()

    def _detect_bpmn_flow_type(self, bpmn_graph, flow, source_id, target_id) -> int:
        """Detect flow type in BPMN based on gateways"""
        # Check if flow goes through gateway
        nodes = bpmn_graph.get_nodes()

        # Find source and target nodes
        source_node = next((n for n in nodes if n.get_id() == source_id), None)
        target_node = next((n for n in nodes if n.get_id() == target_id), None)

        if source_node:
            node_type = source_node.__class__.__name__

            if 'Parallel' in node_type:
                return self.flow_encoder['PARALLEL']
            elif 'Exclusive' in node_type or 'XOR' in node_type:
                return self.flow_encoder['CHOICE']
            elif 'Inclusive' in node_type:
                return self.flow_encoder['CHOICE']

        # Check for loops (back edges)
        if hasattr(flow, 'is_back_edge') and flow.is_back_edge:
            return self.flow_encoder['LOOP']

        # Default: sequential
        return self.flow_encoder['SEQUENCE']

    # ================================================================
    # GraphML Loading
    # ================================================================

    def load_graphml(self, graphml_path: str) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Load graph from GraphML file

        Args:
            graphml_path: Path to GraphML file

        Returns:
            (adjacency, nodes, activity_encoder, flow_encoder)
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for GraphML loading. Install with: pip install networkx")

        if not Path(graphml_path).exists():
            raise FileNotFoundError(f"GraphML file not found: {graphml_path}")

        print(f"[INFO] Loading GraphML from {graphml_path}...")

        # Load using NetworkX
        G = nx.read_graphml(graphml_path)

        return self.load_networkx(G)

    # ================================================================
    # NetworkX Loading
    # ================================================================

    def load_networkx(self, G: Any) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Load graph from NetworkX DiGraph

        Expected node attributes:
        - 'label' or 'name': activity name

        Expected edge attributes:
        - 'type' or 'flow_type': flow type (SEQUENCE, PARALLEL, etc.)

        Args:
            G: NetworkX DiGraph

        Returns:
            (adjacency, nodes, activity_encoder, flow_encoder)
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required. Install with: pip install networkx")

        print(f"[INFO] Loading NetworkX graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges...")

        # Build node mapping
        node_list = list(G.nodes())
        n = len(node_list)
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        # Extract activity names from node attributes
        activity_names = []
        for node in node_list:
            attrs = G.nodes[node]
            # Try different attribute names
            name = attrs.get('label', attrs.get('name', attrs.get('activity', str(node))))
            activity_names.append(name)

        # Build node vector
        nodes = np.zeros(n, dtype=np.int32)
        for i, name in enumerate(activity_names):
            nodes[i] = self._encode_activity(name)

        # Build adjacency matrix
        adjacency = np.zeros((n, n), dtype=np.int32)

        for edge in G.edges(data=True):
            source, target, attrs = edge
            idx1 = node_to_idx[source]
            idx2 = node_to_idx[target]

            # Get flow type from edge attributes
            flow_type_str = attrs.get('type', attrs.get('flow_type', 'SEQUENCE'))
            flow_type_str = flow_type_str.upper()

            if flow_type_str in self.flow_encoder:
                flow_type = self.flow_encoder[flow_type_str]
            else:
                flow_type = self.flow_encoder['SEQUENCE']

            adjacency[idx1, idx2] = flow_type

        # Detect loops (cycles)
        try:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                if len(cycle) >= 2:
                    # Mark edges in cycle as LOOP
                    for i in range(len(cycle)):
                        node1 = cycle[i]
                        node2 = cycle[(i + 1) % len(cycle)]
                        idx1 = node_to_idx[node1]
                        idx2 = node_to_idx[node2]
                        if adjacency[idx1, idx2] == self.flow_encoder['SEQUENCE']:
                            adjacency[idx1, idx2] = self.flow_encoder['LOOP']
        except:
            pass  # If cycle detection fails, skip

        # Pad to max_activities
        adjacency, nodes = self._pad_graph(adjacency, nodes)

        print(f"[INFO] Converted to {len(activity_names)} activities")

        return adjacency, nodes, self.activity_encoder.copy(), self.flow_encoder.copy()

    # ================================================================
    # Raw Matrix Loading
    # ================================================================

    def load_adjacency_matrix(self,
                              adjacency: np.ndarray,
                              activity_labels: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Load from raw adjacency matrix and activity labels

        Args:
            adjacency: (n, n) adjacency matrix with flow type indices
            activity_labels: List of n activity names

        Returns:
            (adjacency, nodes, activity_encoder, flow_encoder)
        """
        n = len(activity_labels)

        if adjacency.shape[0] != n or adjacency.shape[1] != n:
            raise ValueError(f"Adjacency shape {adjacency.shape} doesn't match labels length {n}")

        # Build node vector
        nodes = np.zeros(n, dtype=np.int32)
        for i, name in enumerate(activity_labels):
            nodes[i] = self._encode_activity(name)

        # Pad to max_activities
        adjacency, nodes = self._pad_graph(adjacency, nodes)

        return adjacency, nodes, self.activity_encoder.copy(), self.flow_encoder.copy()


# ================================================================
# Convenience Functions
# ================================================================

def load_graph(file_path: str,
               format: Optional[str] = None,
               max_activities: int = 15) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Load graph from file, auto-detecting format from extension

    Args:
        file_path: Path to graph file
        format: Optional format override ('petri', 'bpmn', 'graphml')
        max_activities: Maximum number of activities

    Returns:
        (adjacency, nodes, activity_encoder, flow_encoder)
    """
    loader = GraphLoader(max_activities=max_activities)

    path = Path(file_path)

    # Auto-detect format from extension
    if format is None:
        ext = path.suffix.lower()
        if ext == '.pnml':
            format = 'petri'
        elif ext in ['.bpmn', '.xml']:
            format = 'bpmn'
        elif ext == '.graphml':
            format = 'graphml'
        else:
            raise ValueError(f"Cannot auto-detect format from extension: {ext}")

    # Load based on format
    if format == 'petri':
        return loader.load_petri_net(file_path)
    elif format == 'bpmn':
        return loader.load_bpmn(file_path)
    elif format == 'graphml':
        return loader.load_graphml(file_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_graphs_batch(file_paths: List[str],
                     max_activities: int = 15) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Load multiple graphs and combine into batch

    Args:
        file_paths: List of paths to graph files
        max_activities: Maximum number of activities

    Returns:
        (adjacency_batch, nodes_batch, activity_encoder, flow_encoder)
        adjacency_batch: (batch_size, max_activities, max_activities)
        nodes_batch: (batch_size, max_activities)
    """
    loader = GraphLoader(max_activities=max_activities)

    adjacency_list = []
    nodes_list = []

    for file_path in file_paths:
        adj, nodes, _, _ = load_graph(file_path, max_activities=max_activities)
        adjacency_list.append(adj)
        nodes_list.append(nodes)

    adjacency_batch = np.array(adjacency_list)
    nodes_batch = np.array(nodes_list)

    return adjacency_batch, nodes_batch, loader.activity_encoder, loader.flow_encoder
