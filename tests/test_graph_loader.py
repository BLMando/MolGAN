"""
Unit Tests for Graph Loader

Tests loading various graph formats into ProcessGAN.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.graph_loader import GraphLoader, load_graph


class TestGraphLoader(unittest.TestCase):
    """Test graph loading functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.loader = GraphLoader(max_activities=10)
        self.data_dir = Path(__file__).parent.parent / 'data'

    def test_loader_initialization(self):
        """Test GraphLoader initialization"""
        self.assertEqual(self.loader.max_activities, 10)
        self.assertIn('SEQUENCE', self.loader.flow_encoder)
        self.assertIn('PARALLEL', self.loader.flow_encoder)
        self.assertIn('LOOP', self.loader.flow_encoder)
        self.assertEqual(self.loader.flow_encoder['NONE'], 0)

    def test_encode_activity(self):
        """Test activity encoding"""
        idx1 = self.loader._encode_activity('Start')
        idx2 = self.loader._encode_activity('Submit')
        idx3 = self.loader._encode_activity('Start')  # Duplicate

        self.assertEqual(idx1, idx3)  # Same activity gets same index
        self.assertNotEqual(idx1, idx2)  # Different activities get different indices
        self.assertIn('Start', self.loader.activity_encoder)

    def test_pad_graph(self):
        """Test graph padding"""
        # Create small graph
        adj = np.array([[0, 1], [0, 0]], dtype=np.int32)
        nodes = np.array([1, 2], dtype=np.int32)

        # Pad to max_activities
        padded_adj, padded_nodes = self.loader._pad_graph(adj, nodes)

        self.assertEqual(padded_adj.shape, (self.loader.max_activities, self.loader.max_activities))
        self.assertEqual(padded_nodes.shape, (self.loader.max_activities,))
        self.assertEqual(padded_adj[0, 1], 1)  # Original edge preserved
        self.assertEqual(padded_nodes[0], 1)  # Original node preserved

    def test_graphml_loading(self):
        """Test loading GraphML file"""
        graphml_path = self.data_dir / 'sample_process.graphml'

        if not graphml_path.exists():
            self.skipTest(f"GraphML file not found: {graphml_path}")

        try:
            adj, nodes, activity_encoder, flow_encoder = self.loader.load_graphml(str(graphml_path))

            # Check shapes
            self.assertEqual(adj.shape, (self.loader.max_activities, self.loader.max_activities))
            self.assertEqual(nodes.shape, (self.loader.max_activities,))

            # Check that some activities were encoded
            self.assertGreater(len(activity_encoder), 1)  # At least PAD + one activity

            # Check flow types are present
            self.assertIn('SEQUENCE', flow_encoder)
            self.assertIn('CHOICE', flow_encoder)

        except ImportError as e:
            self.skipTest(f"NetworkX not available: {e}")

    def test_pnml_loading(self):
        """Test loading PNML file"""
        pnml_path = self.data_dir / 'sample_process.pnml'

        if not pnml_path.exists():
            self.skipTest(f"PNML file not found: {pnml_path}")

        try:
            adj, nodes, activity_encoder, flow_encoder = self.loader.load_petri_net(str(pnml_path))

            # Check shapes
            self.assertEqual(adj.shape, (self.loader.max_activities, self.loader.max_activities))
            self.assertEqual(nodes.shape, (self.loader.max_activities,))

            # Check that activities were encoded
            self.assertGreater(len(activity_encoder), 1)

        except ImportError as e:
            self.skipTest(f"pm4py not available: {e}")

    def test_load_adjacency_matrix(self):
        """Test loading from raw adjacency matrix"""
        # Create simple graph: Start → Submit → End
        adj = np.array([
            [0, 1, 0],  # Start → Submit
            [0, 0, 1],  # Submit → End
            [0, 0, 0]   # End
        ], dtype=np.int32)

        labels = ['Start', 'Submit', 'End']

        padded_adj, padded_nodes, activity_encoder, flow_encoder = self.loader.load_adjacency_matrix(adj, labels)

        # Check shapes
        self.assertEqual(padded_adj.shape, (self.loader.max_activities, self.loader.max_activities))
        self.assertEqual(padded_nodes.shape, (self.loader.max_activities,))

        # Check original structure preserved
        self.assertEqual(padded_adj[0, 1], 1)  # Start → Submit edge
        self.assertEqual(padded_adj[1, 2], 1)  # Submit → End edge

        # Check activities encoded
        self.assertIn('Start', activity_encoder)
        self.assertIn('Submit', activity_encoder)
        self.assertIn('End', activity_encoder)

    def test_networkx_loading(self):
        """Test loading from NetworkX DiGraph"""
        try:
            import networkx as nx
        except ImportError:
            self.skipTest("NetworkX not available")

        # Create simple graph
        G = nx.DiGraph()
        G.add_node('n1', label='Start', activity='Start')
        G.add_node('n2', label='Submit', activity='Submit')
        G.add_node('n3', label='End', activity='End')

        G.add_edge('n1', 'n2', type='SEQUENCE')
        G.add_edge('n2', 'n3', type='SEQUENCE')

        # Load
        adj, nodes, activity_encoder, flow_encoder = self.loader.load_networkx(G)

        # Check shapes
        self.assertEqual(adj.shape, (self.loader.max_activities, self.loader.max_activities))
        self.assertEqual(nodes.shape, (self.loader.max_activities,))

        # Check activities
        self.assertIn('Start', activity_encoder)
        self.assertIn('Submit', activity_encoder)
        self.assertIn('End', activity_encoder)

    def test_auto_format_detection(self):
        """Test automatic format detection from file extension"""
        test_cases = [
            ('test.pnml', 'petri'),
            ('test.graphml', 'graphml'),
            ('test.bpmn', 'bpmn'),
            ('test.xml', 'bpmn'),
        ]

        for filename, expected_format in test_cases:
            path = Path(filename)
            ext = path.suffix.lower()

            if ext == '.pnml':
                format_detected = 'petri'
            elif ext in ['.bpmn', '.xml']:
                format_detected = 'bpmn'
            elif ext == '.graphml':
                format_detected = 'graphml'
            else:
                format_detected = None

            self.assertEqual(format_detected, expected_format)


class TestProcessDatasetGraphLoading(unittest.TestCase):
    """Test ProcessDataset integration with graph loading"""

    def setUp(self):
        """Set up test fixtures"""
        from utils.process_dataset import ProcessDataset
        self.dataset = ProcessDataset(max_activities=10)
        self.data_dir = Path(__file__).parent.parent / 'data'

    def test_load_from_graphml(self):
        """Test ProcessDataset.load_from_graphml()"""
        graphml_path = self.data_dir / 'sample_process.graphml'

        if not graphml_path.exists():
            self.skipTest(f"GraphML file not found: {graphml_path}")

        try:
            self.dataset.load_from_graphml(str(graphml_path))

            # Check dataset was populated
            self.assertGreater(len(self.dataset), 0)
            self.assertIsNotNone(self.dataset.activity_encoder)
            self.assertIsNotNone(self.dataset.flow_encoder)

            # Check data structures
            self.assertIsNotNone(self.dataset.data_A)
            self.assertIsNotNone(self.dataset.data_X)

        except ImportError as e:
            self.skipTest(f"NetworkX not available: {e}")

    def test_generate_from_graphs(self):
        """Test ProcessDataset.generate_from_graphs()"""
        # Create sample graphs
        adj1 = np.zeros((10, 10), dtype=np.int32)
        adj1[0, 1] = 1  # Start → Submit
        adj1[1, 2] = 1  # Submit → End

        nodes1 = np.array([1, 2, 3] + [0]*7, dtype=np.int32)  # Start, Submit, End + padding

        # Build encoders
        self.dataset.activity_encoder = {'PAD': 0, 'Start': 1, 'Submit': 2, 'End': 3}
        self.dataset.activity_decoder = {v: k for k, v in self.dataset.activity_encoder.items()}
        self.dataset.flow_encoder = {'NONE': 0, 'SEQUENCE': 1}
        self.dataset.flow_decoder = {v: k for k, v in self.dataset.flow_encoder.items()}

        # Generate dataset
        self.dataset.generate_from_graphs([adj1], [nodes1], validation=0.0, test=0.0)

        # Check dataset
        self.assertEqual(len(self.dataset), 1)
        self.assertGreater(len(self.dataset.traces), 0)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
