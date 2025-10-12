# ProcessGAN Graph Input Guide

Complete guide for loading process graphs directly into ProcessGAN.

---

## Overview

ProcessGAN now supports loading process models directly as **graph data** instead of only text-based event logs. This enables:

- **Process Model Augmentation**: Generate variants of existing process models
- **Richer Flow Information**: Preserve parallel/choice/loop structures from models
- **Cross-Format Pipeline**: Integrate with process mining tools (pm4py, ProM)
- **Direct Model Training**: Train on discovered Petri nets or BPMN models

---

## Supported Formats

| Format | Extension | Library | Description |
|--------|-----------|---------|-------------|
| **Petri Nets** | `.pnml` | pm4py | Standard process modeling formalism |
| **BPMN** | `.bpmn`, `.xml` | pm4py | Business Process Model and Notation |
| **GraphML** | `.graphml` | NetworkX | General graph interchange format |
| **NetworkX** | (Python object) | NetworkX | Direct Python graph objects |
| **Raw Matrices** | (NumPy arrays) | - | Direct adjacency matrix input |

---

## Installation

### Required Dependencies

```bash
# Core dependencies (already in environment)
conda activate ProcessGAN

# Verify installations
python -c "import pm4py; print('pm4py:', pm4py.__version__)"
python -c "import networkx; print('networkx:', networkx.__version__)"
```

### Optional: Install in New Environment

```bash
# Create environment from YAML
conda env create -f environment_processggan.yml
conda activate ProcessGAN
```

---

## Quick Start

### 1. Train on GraphML

```bash
python example_process_tf2.py \
  --data data/sample_process.graphml \
  --input-format graphml \
  --epochs 100
```

### 2. Train on Petri Net

```bash
python example_process_tf2.py \
  --data data/sample_process.pnml \
  --input-format petri \
  --epochs 100
```

### 3. Train on BPMN

```bash
python example_process_tf2.py \
  --data data/sample_process.bpmn \
  --input-format bpmn \
  --epochs 100
```

---

## Format-Specific Guides

### Petri Nets (PNML)

**What is a Petri Net?**
- Formal modeling technique for concurrent systems
- Consists of **places** (states) and **transitions** (activities)
- Connected by directed arcs

**Loading Petri Nets:**

```python
from utils.process_dataset import ProcessDataset

# Load Petri net
dataset = ProcessDataset()
dataset.load_from_petri_net('models/purchase_order.pnml')

# Train model
# ... (see training examples)
```

**Creating PNML Files:**

Option A: Export from Process Mining Tools
```python
import pm4py

# Discover Petri net from event log
log = pm4py.read_xes('event_log.xes')
net, im, fm = pm4py.discover_petri_net_inductive(log)

# Save as PNML
pm4py.write_pnml(net, im, fm, 'model.pnml')
```

Option B: Manual PNML (see `data/sample_process.pnml` for template)

**ProcessGAN Conversion:**
- **Transitions** → Activities
- **Arcs** → Flow types
- **Parallel places** (multiple outputs) → PARALLEL flow
- **Choice places** (multiple transitions) → CHOICE flow
- **Back-arcs** → LOOP flow

---

### BPMN (Business Process Model and Notation)

**What is BPMN?**
- Industry-standard notation for business processes
- Includes **tasks**, **gateways**, **events**, **flows**
- More expressive than Petri nets (supports subprocesses, data)

**Loading BPMN:**

```python
from utils.process_dataset import ProcessDataset

# Load BPMN model
dataset = ProcessDataset()
dataset.load_from_bpmn('models/order_fulfillment.bpmn')
```

**Creating BPMN Files:**

Option A: Export from BPMN Modelers
- Camunda Modeler
- Signavio
- bpmn.io

Option B: Discover from Event Log
```python
import pm4py

log = pm4py.read_xes('event_log.xes')
bpmn_graph = pm4py.discover_bpmn_inductive(log)
pm4py.write_bpmn(bpmn_graph, 'model.bpmn')
```

**ProcessGAN Conversion:**
- **Tasks** → Activities
- **Parallel Gateway** → PARALLEL flow
- **Exclusive Gateway** (XOR) → CHOICE flow
- **Inclusive Gateway** (OR) → CHOICE flow
- **Sequence Flows** → SEQUENCE flow
- **Back-flows** → LOOP flow

---

### GraphML (NetworkX Format)

**What is GraphML?**
- XML-based graph interchange format
- Supported by NetworkX, Gephi, yEd
- Flexible attribute system

**Loading GraphML:**

```python
from utils.process_dataset import ProcessDataset

# Load GraphML
dataset = ProcessDataset()
dataset.load_from_graphml('models/process_graph.graphml')
```

**Creating GraphML Files:**

```python
import networkx as nx

# Create graph
G = nx.DiGraph()

# Add nodes with attributes
G.add_node('n1', label='Start', activity='Start')
G.add_node('n2', label='Submit', activity='Submit')
G.add_node('n3', label='End', activity='End')

# Add edges with flow types
G.add_edge('n1', 'n2', type='SEQUENCE', flow_type='SEQUENCE')
G.add_edge('n2', 'n3', type='SEQUENCE', flow_type='SEQUENCE')

# Save as GraphML
nx.write_graphml(G, 'process_graph.graphml')
```

**Required Node Attributes:**
- `label` or `name` or `activity`: Activity name

**Required Edge Attributes:**
- `type` or `flow_type`: Flow type (SEQUENCE, PARALLEL, LOOP, CHOICE, SKIP)

**Example GraphML Structure:**
See `data/sample_process.graphml` for complete example.

---

### NetworkX Direct Loading

**What is NetworkX?**
- Python library for complex network analysis
- Used internally by ProcessGAN for graph operations

**Loading NetworkX DiGraph:**

```python
import networkx as nx
from utils.process_dataset import ProcessDataset

# Create graph programmatically
G = nx.DiGraph()
G.add_node('start', label='Start')
G.add_node('submit', label='Submit Order')
G.add_node('end', label='End')
G.add_edge('start', 'submit', type='SEQUENCE')
G.add_edge('submit', 'end', type='SEQUENCE')

# Load into ProcessGAN
dataset = ProcessDataset()
dataset.load_from_networkx(G)
```

**Use Cases:**
- Programmatic graph generation
- Integration with graph analysis pipelines
- Custom process model synthesis

---

### Raw Adjacency Matrix

**Direct Matrix Input:**

```python
import numpy as np
from utils.graph_loader import GraphLoader

# Create adjacency matrix
# Rows/cols = activities, values = flow types
adj = np.array([
    [0, 1, 0],  # Start → Submit (SEQUENCE)
    [0, 0, 1],  # Submit → End (SEQUENCE)
    [0, 0, 0]   # End
], dtype=np.int32)

labels = ['Start', 'Submit', 'End']

# Load
loader = GraphLoader(max_activities=15)
adj_padded, nodes_padded, encoders, flow_enc = loader.load_adjacency_matrix(adj, labels)
```

**Flow Type Encoding:**
```python
flow_types = {
    0: 'NONE',      # No connection
    1: 'SEQUENCE',  # Sequential flow
    2: 'PARALLEL',  # Parallel (AND)
    3: 'LOOP',      # Repetition
    4: 'CHOICE',    # Choice (XOR)
    5: 'SKIP'       # Optional
}
```

---

## Flow Type Detection

ProcessGAN automatically detects flow types from graph structure:

### Petri Net Detection

| Pattern | Flow Type | Detection Rule |
|---------|-----------|----------------|
| **Sequence** | SEQUENCE | Single input → Single output |
| **AND-split** | PARALLEL | Place with multiple output arcs |
| **AND-join** | PARALLEL | Place with multiple input arcs |
| **XOR-split** | CHOICE | Transition with multiple outputs |
| **Loop** | LOOP | Back-edge (backward flow) |

### BPMN Detection

| Gateway Type | Flow Type | Mapping |
|--------------|-----------|---------|
| Parallel Gateway | PARALLEL | Direct mapping |
| Exclusive Gateway | CHOICE | XOR semantics |
| Inclusive Gateway | CHOICE | OR semantics (treated as choice) |
| Sequence Flow | SEQUENCE | Default |
| Back-edge | LOOP | Cycle detection |

### GraphML Detection

- **Explicit**: Read from `type` or `flow_type` edge attribute
- **Implicit**: Detect cycles for LOOP
- **Default**: SEQUENCE for unspecified edges

---

## Advanced Usage

### Hybrid Training: Event Logs + Process Models

Combine event logs with process models for training:

```python
from utils.process_dataset import ProcessDataset

dataset = ProcessDataset()

# Load event log
dataset.load_from_csv('logs/real_traces.csv')

# Add discovered process model
# (This requires extending ProcessDataset to support merging)
# TODO: Implement dataset.add_from_petri_net()
```

### Batch Loading Multiple Graphs

```python
from utils.graph_loader import load_graphs_batch

# Load multiple models
file_paths = [
    'models/variant1.graphml',
    'models/variant2.graphml',
    'models/variant3.graphml'
]

adjacency_batch, nodes_batch, activity_enc, flow_enc = load_graphs_batch(file_paths)

# Use in training
# ...
```

### Process Model Augmentation Pipeline

```python
import pm4py
from utils.process_dataset import ProcessDataset
from models.process_gan_tf2 import ProcessGAN

# Step 1: Discover model from logs
log = pm4py.read_xes('real_logs.xes')
net, im, fm = pm4py.discover_petri_net_inductive(log)
pm4py.write_pnml(net, im, fm, 'discovered_model.pnml')

# Step 2: Load into ProcessGAN
dataset = ProcessDataset()
dataset.load_from_petri_net('discovered_model.pnml')

# Step 3: Train GAN
model = ProcessGAN(...)
# ... training ...

# Step 4: Generate variants
z = np.random.normal(0, 1, (100, 128))
adj_variants, node_variants = model.generator(z, training=False)

# Step 5: Export variants as new models
# (Requires conversion back to PNML/BPMN)
```

### Convert Generated Graphs to PNML

```python
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils

def convert_to_petri_net(adjacency, nodes, activity_decoder):
    """Convert ProcessGAN output to Petri net"""
    net = PetriNet("Generated Process")

    # Create transitions from nodes
    transitions = {}
    for i in range(len(nodes)):
        if nodes[i] > 0:  # Skip PAD
            activity = activity_decoder[nodes[i]]
            trans = PetriNet.Transition(f"t{i}", activity)
            net.transitions.add(trans)
            transitions[i] = trans

    # Create places and arcs from adjacency
    place_counter = 0
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if adjacency[i, j] > 0:  # Edge exists
                place = PetriNet.Place(f"p{place_counter}")
                net.places.add(place)

                petri_utils.add_arc_from_to(transitions[i], place, net)
                petri_utils.add_arc_from_to(place, transitions[j], net)

                place_counter += 1

    # Create markings (simplified)
    im = Marking()
    fm = Marking()

    return net, im, fm

# Use:
net, im, fm = convert_to_petri_net(adj, nodes, activity_decoder)
pm4py.write_pnml(net, im, fm, 'generated_model.pnml')
```

---

## Troubleshooting

### Issue: pm4py not installed

**Error:**
```
ImportError: pm4py is required for Petri net loading
```

**Solution:**
```bash
pip install pm4py
# or
conda install -c conda-forge pm4py
```

### Issue: NetworkX not installed

**Error:**
```
ModuleNotFoundError: No module named 'networkx'
```

**Solution:**
```bash
pip install networkx
# or
conda install networkx
```

### Issue: Invalid PNML file

**Error:**
```
xml.etree.ElementTree.ParseError: syntax error
```

**Solution:**
- Validate PNML file structure
- Ensure well-formed XML
- Check for missing closing tags
- Use pm4py to validate: `pm4py.read_pnml('file.pnml')`

### Issue: Missing node/edge attributes in GraphML

**Error:**
```
KeyError: 'label' or 'activity'
```

**Solution:**
- Ensure nodes have `label`, `name`, or `activity` attribute
- Ensure edges have `type` or `flow_type` attribute
- See `data/sample_process.graphml` for correct format

### Issue: Graph too large

**Warning:**
```
Graph has 20 nodes, truncating to 15
```

**Solution:**
- Increase `max_activities` parameter:
```python
dataset = ProcessDataset(max_activities=25)
```

or

```bash
python example_process_tf2.py --max-activities 25 --input-format graphml --data model.graphml
```

### Issue: No edges detected

**Problem:** Adjacency matrix is all zeros

**Solution:**
- Check edge directionality (must be DiGraph, not Graph)
- Verify flow type encoding (valid values: SEQUENCE, PARALLEL, LOOP, CHOICE, SKIP)
- Ensure arcs in Petri net connect transitions through places

---

## Examples

### Example 1: Simple Purchase Order Process

**GraphML Format:**
```xml
<!-- See data/sample_process.graphml -->
<graphml>
  <graph edgedefault="directed">
    <node id="n0"><data key="label">Start</data></node>
    <node id="n1"><data key="label">Submit</data></node>
    <node id="n2"><data key="label">Review</data></node>
    <node id="n3"><data key="label">Approve</data></node>
    <node id="n4"><data key="label">End</data></node>

    <edge source="n0" target="n1"><data key="type">SEQUENCE</data></edge>
    <edge source="n1" target="n2"><data key="type">SEQUENCE</data></edge>
    <edge source="n2" target="n3"><data key="type">CHOICE</data></edge>
    <edge source="n3" target="n4"><data key="type">SEQUENCE</data></edge>
  </graph>
</graphml>
```

**Usage:**
```bash
python example_process_tf2.py \
  --data data/sample_process.graphml \
  --input-format graphml \
  --epochs 100 \
  --batch-size 16
```

### Example 2: Complex Process with Loops

**NetworkX Code:**
```python
import networkx as nx

G = nx.DiGraph()

# Main flow
G.add_edges_from([
    ('start', 'submit', {'type': 'SEQUENCE'}),
    ('submit', 'review', {'type': 'SEQUENCE'}),
    ('review', 'approve', {'type': 'CHOICE'}),
    ('review', 'reject', {'type': 'CHOICE'}),
    ('approve', 'end', {'type': 'SEQUENCE'}),
    ('reject', 'rework', {'type': 'SEQUENCE'}),
    ('rework', 'submit', {'type': 'LOOP'})  # Loop back
])

# Add labels
for node in G.nodes():
    G.nodes[node]['label'] = node.capitalize()

# Save and load
nx.write_graphml(G, 'complex_process.graphml')
```

### Example 3: Parallel Gateway

**Petri Net Structure:**
```
Start → [P1] → Submit → [P2] → ParallelSplit → [P3, P4]
                                   ↓              ↓
                              [P3] Review     [P4] Validate
                                   ↓              ↓
                             ParallelJoin ← [P5, P6]
                                   ↓
                              [P7] → End
```

**ProcessGAN Interpretation:**
- ParallelSplit: Place P2 with two output transitions → PARALLEL flow
- ParallelJoin: Place P7 with two input transitions → Merge point

---

## Best Practices

1. **Use Discovered Models**: Start with process models discovered from real event logs
2. **Validate Structure**: Ensure graphs have clear start/end nodes
3. **Normalize Activity Names**: Use consistent naming conventions
4. **Test with Simple Models**: Start with small graphs (5-10 activities)
5. **Preserve Semantics**: Maintain control flow meaning when converting formats
6. **Document Flow Types**: Clearly specify flow type semantics in your domain
7. **Version Control Models**: Track process model versions alongside event logs

---

## API Reference

### GraphLoader Class

```python
from utils.graph_loader import GraphLoader

loader = GraphLoader(max_activities=15)

# Load from file
adj, nodes, activity_enc, flow_enc = loader.load_graphml('model.graphml')
adj, nodes, activity_enc, flow_enc = loader.load_petri_net('model.pnml')
adj, nodes, activity_enc, flow_enc = loader.load_bpmn('model.bpmn')

# Load from NetworkX
adj, nodes, activity_enc, flow_enc = loader.load_networkx(G)

# Load from matrix
adj, nodes, activity_enc, flow_enc = loader.load_adjacency_matrix(adj_matrix, labels)
```

### ProcessDataset Integration

```python
from utils.process_dataset import ProcessDataset

dataset = ProcessDataset(max_activities=15)

# Load methods
dataset.load_from_petri_net('model.pnml', validation=0.1, test=0.1)
dataset.load_from_bpmn('model.bpmn', validation=0.1, test=0.1)
dataset.load_from_graphml('model.graphml', validation=0.1, test=0.1)
dataset.load_from_networkx(G, validation=0.1, test=0.1)

# Batch loading
dataset.generate_from_graphs(adjacency_list, nodes_list, validation=0.1, test=0.1)
```

---

## Related Documentation

- [README.md](README.md) - Main project documentation
- [ARCHITETTURA_PROCESSGGAN.md](Docs/ARCHITETTURA_PROCESSGGAN.md) - Technical architecture
- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - Graph visualization guide
- [README_TF2_MIGRATION.md](README_TF2_MIGRATION.md) - TensorFlow migration guide

---

## Citation

If you use ProcessGAN graph input features in your research:

```bibtex
@misc{processggan2024,
  title={{ProcessGAN: Adapting MolGAN for Process Mining with Graph Input Support}},
  author={Roselli, Paolo},
  year={2024},
  howpublished={\url{https://github.com/paoloroselli/MolGAN}}
}
```

---

## Support

For issues with graph loading:
1. Check this guide's troubleshooting section
2. Verify input file format matches specification
3. Test with provided sample files (`data/sample_process.*`)
4. Open GitHub issue with error message and file example

---

Last Updated: 2024-10-12
