# Graph Process GAN

Complete Graph-based ProcessGAN implementation for instance graph data with parallelism support.

## Overview

This module transforms the LSTM-based ProcessGAN into a **graph-based** approach optimized for instance graph data (.g format). Key advantages:

- ✅ Native support for **parallel activities** (AND-split/join)
- ✅ **Graph neural networks** (R-GCN) for structure-aware processing
- ✅ Automatic **edge type classification** (SEQUENCE, AND_SPLIT, AND_JOIN, LOOP, XOR)
- ✅ **Keras API** compatibility with full callback support
- ✅ Process mining **constraints** (START/END, connectivity, activity frequencies)

## Architecture

### Generator

- Input: Random noise z ~ N(0,1)
- Output: (Adjacency matrix, Node features)
- Sampling: Gumbel-Softmax for differentiable discrete sampling
- Temperature annealing: 5.0 → 0.5 for more discrete outputs

### Discriminator

- Architecture: R-GCN → Global Pooling → MLP → Score
- R-GCN: Relational Graph Convolutional Networks with per-edge-type message passing
- Output: Wasserstein distance estimate

### Training

- Loss: Wasserstein GAN with Gradient Penalty (WGAN-GP)
- n_critic: 5 discriminator updates per generator update
- Constraints: START/END nodes, activity frequencies, connectivity, no self-loops
- Optimizer: Adam with gradient clipping

## Data Format

Instance graphs (.g files):

```
t # 0  (trace ID)
v 0 START 2010-11-05T09:33:00.000+01:00 ...  (vertex: id, activity, timestamp, features)
v 1 Activity_1 2010-11-05T09:36:00.000+01:00 ...
e 0 1 complete 0  (edge: source, target, label, case_id)
e 1 2 complete 0
...
```

Edge types automatically classified:

- **SEQUENCE**: Single outgoing edge
- **AND_SPLIT**: Multiple outgoing edges from same node
- **AND_JOIN**: Multiple incoming edges to same node
- **LOOP**: Edge back to previous node
- **XOR**: Single path selection (inferred)

## Files

### Phase 1: Data Loading ✅

- `ig_loader.py` - Parse .g files, extract nodes/edges, build vocabularies
- `graph_dataset.py` - Convert graphs to adjacency matrices + node features
- `graph_utils.py` - Utilities (gumbel_softmax, losses, pooling)

### Phase 2: Model Architecture ✅

- `graph_generator.py` - Generate graphs from noise
- `graph_discriminator.py` - Classify real/fake graphs
- `rgcn_layers.py` - Relational GCN implementation
- `graph_constraints.py` - Process mining constraints

### Phase 3: Training Loop ✅

- `graph_process_gan.py` - Main GAN class with Keras API
- `train_graph_gan.py` - Complete training script with callbacks

### Phase 4: Evaluation (TODO)

- `graph_evaluation.py` - Metrics (validity, diversity, fitness)
- `conformance.py` - Conformance checking

### Phase 5: Export (TODO)

- `graph_exporter.py` - Export to .xes, .pnml, .g formats

## Usage

### Quick Start

```python
# 1. Load data
from ig_loader import load_ig_for_gan

graphs, vocab, edge_types = load_ig_for_gan('data/Helpdesk_igs_complete.g')

# 2. Create dataset
from graph_dataset import GraphDataset

dataset = GraphDataset(graphs, max_nodes=20, vocab=vocab, edge_type_vocab=edge_types)
adj_matrices, node_matrices = dataset.get_all_matrices()

# 3. Train GAN
from train_graph_gan import train_graph_gan

gan, history = train_graph_gan(
    data_path='data/Helpdesk_igs_complete.g',
    output_dir='output_graph_gan',
    max_nodes=20,
    batch_size=32,
    epochs=500
)

# 4. Generate graphs
generated_adj, generated_nodes = gan.generate_graphs(
    num_samples=100,
    temperature=0.5,
    hard=True
)
```

### Training Script

```bash
# Basic training
python train_graph_gan.py --data ../../data/Helpdesk_igs_complete.g

# Custom hyperparameters
python train_graph_gan.py \
    --data ../../data/Helpdesk_igs_complete.g \
    --output ../../output_graph_gan \
    --max-nodes 20 \
    --batch-size 32 \
    --epochs 500 \
    --n-critic 5 \
    --lambda-gp 10.0 \
    --lambda-constraint 0.1 \
    --d-lr 0.0001 \
    --g-lr 0.0001

# Monitor with TensorBoard
tensorboard --logdir ../../output_graph_gan/logs
```

### Callbacks Included

1. **ModelCheckpoint** - Save best model based on validation loss
2. **EarlyStopping** - Stop when no improvement (patience=20)
3. **TensorBoard** - Visualize training metrics
4. **CSVLogger** - Log metrics to CSV
5. **ReduceLROnPlateau** - Reduce LR when stuck
6. **PeriodicCheckpoint** - Save every 50 epochs

### Metrics Tracked

- `d_loss` - Discriminator loss
- `g_loss` - Generator loss
- `gradient_penalty` - GP term
- `constraint_loss` - Process constraints
- `d_real_score` - Discriminator score for real graphs
- `d_fake_score` - Discriminator score for fake graphs
- `temperature` - Gumbel-Softmax temperature

## Testing

Each module has standalone tests:

```bash
# Test data loading
python ig_loader.py

# Test dataset creation
python graph_dataset.py

# Test utilities
python graph_utils.py

# Test generator
python graph_generator.py

# Test discriminator
python graph_discriminator.py

# Test R-GCN layers
python rgcn_layers.py

# Test constraints
python graph_constraints.py

# Test main GAN
python graph_process_gan.py
```

## Hyperparameters

### Model Architecture

- `max_nodes`: 20 (maximum nodes per graph)
- `noise_dim`: 128 (latent dimension)
- `generator_hidden_dims`: (256, 512, 1024)
- `rgcn_hidden_dims`: (128, 64)
- `mlp_hidden_dims`: (128, 64)
- `generator_dropout`: 0.0
- `discriminator_dropout`: 0.3

### Training

- `n_critic`: 5 (discriminator updates per generator update)
- `lambda_gp`: 10.0 (gradient penalty weight)
- `lambda_constraint`: 0.1 (constraint loss weight)
- `batch_size`: 32
- `epochs`: 500
- `d_lr`: 0.0001 (discriminator learning rate)
- `g_lr`: 0.0001 (generator learning rate)
- `beta_1`: 0.5 (Adam beta_1)
- `beta_2`: 0.9 (Adam beta_2)

### Temperature Scheduling

- `temp_start`: 5.0 (initial temperature)
- `temp_min`: 0.5 (minimum temperature)
- `temp_decay`: 0.99995 (decay rate per step)

## Output Structure

```
output_graph_gan/
├── checkpoints/
│   └── YYYYMMDD_HHMMSS/
│       ├── best/           # Best model (lowest val_d_loss)
│       ├── epoch_50/       # Periodic checkpoints
│       ├── epoch_100/
│       └── final/          # Final model + metadata
│           ├── model       # Model weights
│           ├── metadata.json
│           ├── sample_adjacency.npy
│           └── sample_nodes.npy
└── logs/
    └── YYYYMMDD_HHMMSS/
        ├── training_metrics.csv  # CSV log
        └── events.out.tfevents.* # TensorBoard events
```

## Data Statistics (Helpdesk)

- **Traces**: 3804
- **Nodes per trace**: 3-16 (avg 5.6)
- **Activities**: 12 (START, END, 1-9, PAD)
- **Edge types**: 5 (SEQUENCE, AND_SPLIT, AND_JOIN, LOOP, XOR)
- **Edge labels**: 47 unique labels

## Comparison: LSTM vs Graph

| Aspect         | LSTM Approach        | Graph Approach                     |
| -------------- | -------------------- | ---------------------------------- |
| Data structure | Sequential traces    | Instance graphs                    |
| Parallelism    | ❌ Cannot model      | ✅ Native support                  |
| Architecture   | LSTM encoder/decoder | R-GCN                              |
| Edge types     | N/A                  | 5 types (auto-detected)            |
| Constraints    | Sequence only        | Full graph constraints             |
| Best for       | Linear processes     | Complex processes with parallelism |

## Process Constraints

Implemented in `graph_constraints.py`:

1. **start_node_loss** - First node should be START activity
2. **end_node_loss** - Last active node should be END activity
3. **activity_frequency_loss** - Match target activity distribution (KL divergence)
4. **connectivity_loss** - Penalize isolated nodes
5. **structural_validity_loss** - Penalize self-loops

Total constraint loss: weighted sum of individual constraints.

## Known Issues & Fixes

### Fixed Issues ✅

1. **R-GCN normalization bug** - in_degree shape mismatch
   - Solution: Use `reduce_sum` → `expand_dims` for proper broadcasting
2. **Timestamp parsing** - Quoted timestamps in .g files
   - Solution: Regex pattern with optional quotes

### Current Status

- ✅ Data loading tested (3804 traces)
- ✅ Models created and compiled
- ✅ Training script with full callbacks
- 🔄 Full training run pending
- 🔄 Evaluation metrics pending

## Next Steps

**Phase 4: Evaluation**

- Implement validity metrics (START/END, connectivity)
- Implement diversity metrics (unique traces, structural diversity)
- Implement fitness metrics (conformance, precision, generalization)
- Compare with real data distribution

**Phase 5: Export & Validation**

- Export generated graphs to .xes, .pnml, .g formats
- Conformance checking with process models
- Statistical tests (KS test, EMD)
- Visual comparison

## References

- Instance graphs: ISO/IEC/IEEE 42010:2011
- R-GCN: Schlichtkrull et al. (2018) "Modeling Relational Data with Graph Convolutional Networks"
- WGAN-GP: Gulrajani et al. (2017) "Improved Training of Wasserstein GANs"
- Gumbel-Softmax: Jang et al. (2017) "Categorical Reparameterization with Gumbel-Softmax"

## License

Same as parent project.

## Authors

Developed as part of Big Data Analytics course project.
