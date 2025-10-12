# MolGAN & ProcessGAN

This repository contains two implementations:
1. **MolGAN**: Original implementation for molecular graph generation
2. **ProcessGAN**: Adaptation for process mining event log generation and data augmentation

---

## 🧪 MolGAN (Original)

TensorFlow implementation of MolGAN: An implicit generative model for small molecular graphs ([arXiv:1805.11973](https://arxiv.org/abs/1805.11973))

### Dependencies
* **python>=3.6**
* **tensorflow>=1.7.0**: https://tensorflow.org
* **rdkit**: https://www.rdkit.org
* **numpy**, **scikit-learn**

### Quick Start
```bash
# Setup legacy environment
conda env create -f environment_legacy.yml
conda activate MolGAN_legacy

# Download and prepare molecular dataset
bash download_dataset.sh
python utils/sparse_molecular_dataset.py

# Run training
python example.py
```

---

## 🔄 ProcessGAN (Process Mining Adaptation)

Adaptation of MolGAN for generating synthetic event logs in process mining. Supports both **TensorFlow 1.x (legacy)** and **TensorFlow 2.x (modern)**.

### Features
- **Data Augmentation**: Generate synthetic event logs from limited real data
- **Multi-Objective Reward**: Validity, Fitness, Conformance, Diversity
- **R-GCN Architecture**: Handles multiple control flow types (sequence, parallel, loop, choice)
- **Two Implementations**: TF 1.x (stable) and TF 2.x (modern, faster)

### Quick Start

#### Option A: TensorFlow 2.x (Recommended)

```bash
# Setup modern environment
conda env create -f environment_processggan.yml
conda activate ProcessGAN

# Prepare your event log (CSV format)
# Required columns: case_id, activity, timestamp (optional)

# Run training
python example_process_tf2.py \
  --data data/sample_event_log.csv \
  --epochs 100 \
  --batch-size 32 \
  --output results/
```

#### Option B: TensorFlow 1.x (Legacy)

```bash
# Setup legacy environment
conda env create -f environment_legacy.yml
conda activate MolGAN_legacy

# Run training
python example_process.py \
  --data data/sample_event_log.csv \
  --epochs 100 \
  --batch-size 32 \
  --output results/
```

### Architecture Overview

```
┌─────────────┐
│  Event Log  │
│   (CSV/XES) │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  ProcessDataset     │
│  - Encode activities│
│  - Build adjacency  │
│  - Extract traces   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│           ProcessGAN Model              │
│  ┌───────────────┐   ┌───────────────┐ │
│  │  Generator    │   │ Discriminator │ │
│  │  z → (A, X)   │   │ (A, X) → [0,1]│ │
│  │               │   │               │ │
│  │ • Dense layers│   │ • R-GCN layers│ │
│  │ • Gumbel-     │   │ • Aggregation │ │
│  │   Softmax     │   │ • Classification│ │
│  └───────────────┘   └───────────────┘ │
└─────────────────────────────────────────┘
       │                       │
       ▼                       ▼
┌──────────────┐      ┌─────────────────┐
│ Reward Fn    │      │  WGAN-GP Loss   │
│ • Validity   │      │  + Gradient     │
│ • Fitness    │      │    Penalty      │
│ • Conformance│      └─────────────────┘
│ • Diversity  │
└──────────────┘
```

### Key Components

| Component | TF 1.x File | TF 2.x File | Description |
|-----------|-------------|-------------|-------------|
| Dataset | [utils/process_dataset.py](utils/process_dataset.py) | *(same)* | Load and encode event logs |
| Reward | [utils/process_metrics.py](utils/process_metrics.py) | *(same)* | Multi-objective reward function |
| Model | [models/process_gan.py](models/process_gan.py) | [models/process_gan_tf2.py](models/process_gan_tf2.py) | Generator + Discriminator |
| Optimizer | [optimizers/process_optimizer.py](optimizers/process_optimizer.py) | [optimizers/process_optimizer_tf2.py](optimizers/process_optimizer_tf2.py) | WGAN-GP training |
| Training | [example_process.py](example_process.py) | [example_process_tf2.py](example_process_tf2.py) | End-to-end training script |

### Usage Examples

#### 1. Basic Training

```python
from utils.process_dataset import ProcessDataset
from models.process_gan_tf2 import ProcessGAN
from optimizers.process_optimizer_tf2 import ProcessGANTrainer

# Load dataset
dataset = ProcessDataset()
dataset.load_from_csv('data/my_event_log.csv',
                      case_id_col='case_id',
                      activity_col='activity')

# Create model
model = ProcessGAN(
    max_activities=dataset.max_activities,
    flow_types=len(dataset.flow_encoder),
    activity_types=len(dataset.activity_encoder),
    embedding_dim=128
)

# Train
trainer = ProcessGANTrainer(model)
trainer.train(dataset, epochs=100, batch_size=32)
```

#### 2. Generate Synthetic Traces

```python
import numpy as np

# Generate random latent vectors
z = np.random.normal(0, 1, (100, 128))

# Generate traces
adjacency, nodes = model.generator(z, training=False)

# Decode to traces
traces = dataset.decode_batch(adjacency.numpy(), nodes.numpy())

# Print first 5 traces
for i, trace in enumerate(traces[:5]):
    print(f"Trace {i+1}: {' → '.join(trace)}")
```

#### 3. Evaluate Quality

```python
from utils.process_metrics import ProcessRewardFunction

# Create reward function
reward_fn = ProcessRewardFunction(
    reference_dataset=dataset,
    weights={'validity': 0.3, 'fitness': 0.3,
             'conformance': 0.2, 'diversity': 0.2}
)

# Evaluate generated traces
rewards = reward_fn.compute_reward(traces)
print(f"Average reward: {rewards.mean():.3f}")
print(f"Validity: {reward_fn.validity_score(traces).mean():.3f}")
print(f"Fitness: {reward_fn.fitness_score(traces).mean():.3f}")
```

### Configuration

#### Reward Function Weights

Edit in your training script or pass as arguments:

```python
reward_weights = {
    'validity': 0.3,      # Syntax correctness (Start/End, no self-loops)
    'fitness': 0.3,       # Token replay fitness on reference model
    'conformance': 0.2,   # Alignment with reference traces
    'diversity': 0.2      # Edit distance from training set
}
```

#### Model Hyperparameters

```python
config = {
    'embedding_dim': 128,           # Latent space dimension
    'generator_units': [128, 256, 512],  # Dense layer sizes
    'discriminator_units': [128, 64],    # R-GCN + aggregation
    'dropout_rate': 0.3,            # Dropout probability
    'learning_rate': 1e-4,          # Adam learning rate
    'gradient_penalty_weight': 10.0 # WGAN-GP lambda
}
```

### Performance Benchmarks

| Version | Training Speed | Memory | Compatibility | Recommended |
|---------|---------------|--------|---------------|-------------|
| **TF 2.x** | ~2x faster | ~30% less | Python 3.8-3.11 | ✅ Yes |
| **TF 1.x** | Baseline | Baseline | Python 3.6-3.7 | Legacy only |

**Test Environment**: MacBook Pro M1, 16GB RAM, 1000 traces

### Documentation

| Document | Description |
|----------|-------------|
| [ARCHITETTURA_PROCESSGGAN.md](Docs/ARCHITETTURA_PROCESSGGAN.md) | Complete technical architecture (450+ lines) |
| [README_PROCESSGGAN.md](README_PROCESSGGAN.md) | ProcessGAN user guide |
| [README_TF2_MIGRATION.md](README_TF2_MIGRATION.md) | Migration guide TF 1.x → 2.x |
| [GUIDA_ADATTAMENTO_PROCESS_MINING.md](Docs/GUIDA_ADATTAMENTO_PROCESS_MINING.md) | Original adaptation plan (Italian) |
| [DOCUMENTAZIONE_MOLGAN.md](Docs/DOCUMENTAZIONE_MOLGAN.md) | Original MolGAN docs (Italian) |

### Input Data Format

#### CSV Format
```csv
case_id,activity,timestamp
1,Start,2024-01-01 08:00
1,Submit Application,2024-01-01 08:15
1,Review Application,2024-01-01 10:30
1,Approve,2024-01-01 11:00
1,End,2024-01-01 11:05
2,Start,2024-01-01 09:00
...
```

#### XES Format
```xml
<log>
  <trace>
    <string key="concept:name" value="1"/>
    <event>
      <string key="concept:name" value="Start"/>
      <date key="time:timestamp" value="2024-01-01T08:00:00"/>
    </event>
    ...
  </trace>
</log>
```

### Troubleshooting

#### Issue: "Module 'tensorflow' has no attribute 'placeholder'"
**Solution**: You're using TF 2.x code with TF 1.x. Use `example_process_tf2.py` with `environment_processggan.yml`.

#### Issue: "AttributeError: module 'tensorflow' has no attribute 'Session'"
**Solution**: You're using TF 1.x code with TF 2.x. Use `example_process.py` with `environment_legacy.yml`.

#### Issue: Low reward scores
**Solution**:
- Increase training epochs (try 200-500)
- Adjust reward weights based on your priority
- Ensure sufficient training data (>50 traces recommended)
- Check if reference model is too restrictive

#### Issue: Out of memory
**Solution**:
- Reduce `batch_size` (try 16 or 8)
- Reduce `max_activities` if possible
- Use TF 2.x version (more memory efficient)

### Testing

```bash
# Run unit tests for reward function
python -m pytest tests/test_process_reward.py -v

# Expected: 30 tests passed (100%)
```

### Project Structure

```
MolGAN/
├── data/
│   ├── sample_event_log.csv         # Sample process dataset
│   └── gdb9.sdf                      # Molecular dataset (original)
├── models/
│   ├── __init__.py                   # MolGAN models (molecules)
│   ├── gan.py                        # Original GAN
│   ├── process_gan.py                # ProcessGAN (TF 1.x)
│   └── process_gan_tf2.py            # ProcessGAN (TF 2.x) ⭐
├── optimizers/
│   ├── gan.py                        # Original optimizers
│   ├── process_optimizer.py          # ProcessGAN optimizer (TF 1.x)
│   └── process_optimizer_tf2.py      # ProcessGAN optimizer (TF 2.x) ⭐
├── utils/
│   ├── layers.py                     # R-GCN layers
│   ├── sparse_molecular_dataset.py   # Molecular dataset
│   ├── process_dataset.py            # Event log dataset ⭐
│   └── process_metrics.py            # Reward function ⭐
├── tests/
│   └── test_process_reward.py        # Unit tests (30 tests)
├── Docs/
│   ├── ARCHITETTURA_PROCESSGGAN.md   # Technical documentation
│   ├── DOCUMENTAZIONE_MOLGAN.md      # Original MolGAN docs
│   └── GUIDA_ADATTAMENTO_PROCESS_MINING.md
├── example.py                         # Original MolGAN training
├── example_process.py                 # ProcessGAN training (TF 1.x)
├── example_process_tf2.py             # ProcessGAN training (TF 2.x) ⭐
├── environment_legacy.yml             # TF 1.x environment
├── environment_processggan.yml        # TF 2.x environment ⭐
├── README_PROCESSGGAN.md              # ProcessGAN guide
└── README_TF2_MIGRATION.md            # Migration guide
```

⭐ = Recommended for new projects

---

## Citation

### Original MolGAN

```bibtex
@article{de2018molgan,
  title={{MolGAN: An implicit generative model for small molecular graphs}},
  author={De Cao, Nicola and Kipf, Thomas},
  journal={ICML 2018 workshop on Theoretical Foundations and Applications of Deep Generative Models},
  year={2018}
}
```

### ProcessGAN Adaptation

If you use ProcessGAN in your research, please cite both the original MolGAN paper and this repository:

```bibtex
@misc{processggan2024,
  title={{ProcessGAN: Adapting MolGAN for Process Mining Event Log Generation}},
  author={Roselli, Paolo},
  year={2024},
  howpublished={\url{https://github.com/paoloroselli/MolGAN}}
}
```

---

## License

MIT License - see LICENSE file for details

## Feedback & Contributions

- **Original MolGAN**: [Nicola De Cao](mailto:nicola.decao@gmail.com)
- **ProcessGAN Adaptation**: [Paolo Roselli](mailto:paolo.roselli@example.com)

For questions, issues, or contributions related to ProcessGAN, please open an issue on GitHub.

---

## Acknowledgments

This work builds upon the original MolGAN implementation by De Cao & Kipf (2018). The ProcessGAN adaptation was developed for process mining research and data augmentation in business process management.
