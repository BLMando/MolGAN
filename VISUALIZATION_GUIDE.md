# ProcessGAN Visualization Guide

Complete guide for visualizing generated process graphs interactively.

---

## 📊 Visualization Options

### 1. Interactive Desktop Viewer (Matplotlib)

**Best for**: Quick exploration, offline use, presentations

```bash
python interactive_viewer.py --data data/sample_event_log.csv --n-samples 10
```

**Features**:
- ✅ Navigate with buttons or keyboard shortcuts
- ✅ Regenerate samples on-the-fly
- ✅ View graph + trace simultaneously
- ✅ No browser required

**Controls**:
| Action | Button | Keyboard |
|--------|--------|----------|
| Next graph | Click "Next" | Right arrow or N |
| Previous graph | Click "Previous" | Left arrow or P |
| Regenerate all | Click "Regenerate" | R |
| Quit | Close window | Q |

**Screenshot**:
```
┌──────────────────────────────────────────────────┐
│         Interactive ProcessGAN Graph Viewer      │
├──────────────────────────────────────────────────┤
│                                                  │
│        [Network Graph Visualization]             │
│                                                  │
│   Start → Submit → Review → Approve → End       │
│                                                  │
├──────────────────────────────────────────────────┤
│  Trace: Start → Submit → Review → Approve → End │
├──────────────────────────────────────────────────┤
│    [Previous]    [Regenerate]    [Next]         │
└──────────────────────────────────────────────────┘
```

---

### 2. Web-based Viewer (Plotly)

**Best for**: Sharing results, detailed exploration, publications

```bash
python web_viewer.py --data data/sample_event_log.csv --n-samples 6 --output dashboard.html
```

**Features**:
- ✅ Opens automatically in browser
- ✅ Zoom, pan, and hover for details
- ✅ Multiple graphs in dashboard layout
- ✅ Shareable HTML file (no dependencies)
- ✅ High-quality export (PNG/SVG)

**Interactive Controls**:
- **Zoom**: Scroll or drag selection box
- **Pan**: Click and drag
- **Reset**: Double-click
- **Hover**: Show node/edge details
- **Download**: Use camera icon (top-right)

**Dashboard Layout**:
```
┌─────────────────────────────────────────┐
│  ProcessGAN Generated Graphs Dashboard │
├───────────────┬─────────────────────────┤
│   Sample 1    │      Sample 2           │
│   [Graph]     │      [Graph]            │
├───────────────┼─────────────────────────┤
│   Sample 3    │      Sample 4           │
│   [Graph]     │      [Graph]            │
├───────────────┼─────────────────────────┤
│   Sample 5    │      Sample 6           │
│   [Graph]     │      [Graph]            │
└───────────────┴─────────────────────────┘
```

---

### 3. Static Image Export

**Best for**: Reports, papers, simple testing

```bash
python test_visualization.py
# Output: visualizations/*.png
```

**Features**:
- ✅ No dependencies required
- ✅ Fast batch generation
- ✅ High-resolution PNG (300 DPI)
- ✅ Works without display/GUI

**Generated Files**:
```
visualizations/
├── trace_1.png          # Sequential trace diagram
├── trace_2.png
├── trace_3.png
├── real_graph_1.png     # Network graph visualization
├── real_adjacency_1.png # Adjacency matrix heatmap
└── comparison.png       # Side-by-side comparison
```

---

## 🎨 Visualization Components

### Graph Elements

| Element | Description | Visual Representation |
|---------|-------------|----------------------|
| **Start Node** | Process start | 🟢 Green box/circle |
| **End Node** | Process end | 🔴 Pink box/circle |
| **Activity Node** | Regular activity | 🔵 Blue box/circle |
| **Sequence Edge** | Sequential flow | Blue solid arrow → |
| **Parallel Edge** | Parallel split/join | Purple solid arrow |
| **Loop Edge** | Repetition | Orange dashed arrow ⤿ |
| **Choice Edge** | XOR decision | Red dotted arrow |
| **Skip Edge** | Optional path | Green solid arrow |

### Color Scheme

```python
# Node colors
Start activities:  #90EE90 (Light green)
End activities:    #FFB6C6 (Light pink)
Regular activities: #87CEEB (Sky blue)

# Edge colors
SEQUENCE:  #2E86AB (Blue)
PARALLEL:  #A23B72 (Purple)
LOOP:      #F18F01 (Orange)
CHOICE:    #C73E1D (Red)
SKIP:      #6A994E (Green)
```

---

## 🔧 Advanced Usage

### Custom Model Checkpoint

Load a trained model:

```bash
python interactive_viewer.py \
  --data data/my_event_log.csv \
  --model results/checkpoints/model_epoch_100.h5 \
  --n-samples 20
```

### Generate Specific Samples

Control the random seed for reproducibility:

```bash
python web_viewer.py \
  --data data/sample_event_log.csv \
  --seed 42 \
  --n-samples 10
```

### Export Dashboard

Create shareable HTML dashboard:

```bash
python web_viewer.py \
  --data data/sample_event_log.csv \
  --mode dashboard \
  --n-samples 12 \
  --output results/dashboard_$(date +%Y%m%d).html
```

### Batch Visualization

Generate visualizations for multiple models:

```bash
#!/bin/bash
for epoch in 50 100 150 200; do
  python web_viewer.py \
    --data data/sample_event_log.csv \
    --model results/model_epoch_${epoch}.h5 \
    --output viz/epoch_${epoch}.html \
    --n-samples 6
done
```

---

## 📝 Python API Usage

### Programmatic Visualization

```python
from utils.process_dataset import ProcessDataset
from models.process_gan_tf2 import ProcessGAN
from utils.visualization import ProcessGraphVisualizer
import numpy as np

# Load data
dataset = ProcessDataset()
dataset.load_from_csv('data/sample_event_log.csv')

# Create model
model = ProcessGAN(
    max_activities=dataset.max_activities,
    flow_types=len(dataset.flow_encoder),
    activity_types=len(dataset.activity_encoder),
    embedding_dim=128
)

# Generate samples
z = np.random.normal(0, 1, (10, 128))
adjacency, nodes = model.generator(z, training=False)

# Create visualizer
activity_decoder = {v: k for k, v in dataset.activity_encoder.items()}
flow_decoder = {v: k for k, v in dataset.flow_encoder.items()}
visualizer = ProcessGraphVisualizer(activity_decoder, flow_decoder)

# Visualize single graph
fig = visualizer.visualize_graph(
    adjacency[0].numpy(),
    nodes[0].numpy(),
    title="Generated Process Graph",
    figsize=(12, 8),
    save_path='output/graph.png'
)

# Visualize trace
traces = dataset.decode_batch(adjacency.numpy(), nodes.numpy())
fig = visualizer.visualize_trace(
    traces[0],
    title="Process Trace",
    save_path='output/trace.png'
)

# Visualize batch
fig = visualizer.visualize_batch(
    adjacency.numpy(),
    nodes.numpy(),
    n_samples=4,
    save_path='output/batch.png'
)
```

### Training Metrics Visualization

```python
from utils.visualization import plot_training_metrics, plot_reward_components

# During training, collect metrics
losses_d = []
losses_g = []
rewards = []
validity_scores = []
fitness_scores = []
conformance_scores = []
diversity_scores = []

# After training
fig = plot_training_metrics(
    losses_d, losses_g, rewards,
    save_path='results/training_metrics.png'
)

fig = plot_reward_components(
    validity_scores, fitness_scores,
    conformance_scores, diversity_scores,
    save_path='results/reward_components.png'
)
```

---

## 🐛 Troubleshooting

### Issue: Matplotlib window doesn't open

**Solution**: Check if you're using SSH or headless environment
```bash
# Use web viewer instead
python web_viewer.py --data data/sample_event_log.csv

# Or set matplotlib backend
export MPLBACKEND=TkAgg
python interactive_viewer.py --data data/sample_event_log.csv
```

### Issue: Plotly not installed

**Solution**: Install with pip
```bash
pip install plotly kaleido
# Or update conda environment
conda env update -f environment_processggan.yml
```

### Issue: Graphs look cluttered

**Solution**: Reduce number of samples or adjust layout
```bash
# Show fewer samples
python web_viewer.py --data data/sample_event_log.csv --n-samples 3

# Use single graph mode
python web_viewer.py --data data/sample_event_log.csv --mode single
```

### Issue: Can't see node labels

**Solution**: Increase figure size
```python
fig = visualizer.visualize_graph(
    adjacency, nodes,
    figsize=(16, 12)  # Larger figure
)
```

### Issue: Interactive viewer is slow

**Solution**: Reduce complexity
```bash
# Generate fewer samples
python interactive_viewer.py --data data/sample_event_log.csv --n-samples 5

# Or use static images
python test_visualization.py
```

---

## 📚 Examples

### Example 1: Quick Visualization Test

```bash
# Test with sample data (no training required)
python test_visualization.py
open visualizations/trace_1.png
```

### Example 2: Interactive Exploration

```bash
# Launch interactive viewer
python interactive_viewer.py \
  --data data/sample_event_log.csv \
  --n-samples 20 \
  --seed 42
```

### Example 3: Create Publication Figures

```bash
# Generate high-quality dashboard
python web_viewer.py \
  --data data/sample_event_log.csv \
  --n-samples 6 \
  --output paper/figures/generated_graphs.html

# Open in browser and use camera icon to export PNG/SVG
```

### Example 4: Compare Different Models

```python
from utils.visualization import ProcessGraphVisualizer
import matplotlib.pyplot as plt

# Load multiple model checkpoints
models = [
    load_model('results/epoch_50.h5'),
    load_model('results/epoch_100.h5'),
    load_model('results/epoch_150.h5')
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, model in enumerate(models):
    z = np.random.normal(0, 1, (1, 128))
    adj, nodes = model.generator(z, training=False)

    visualizer.visualize_graph(
        adj[0].numpy(), nodes[0].numpy(),
        title=f"Epoch {(i+1)*50}",
        ax=axes[i]
    )

plt.savefig('results/model_comparison.png', dpi=300)
```

---

## 🎯 Best Practices

1. **Start with static images** for quick tests
2. **Use interactive viewer** for exploration and debugging
3. **Use web viewer** for sharing and presentations
4. **Set random seed** for reproducible visualizations
5. **Save HTML files** for interactive sharing with collaborators
6. **Export high-resolution PNG** (300 DPI) for papers
7. **Use dashboard mode** to compare multiple samples
8. **Customize colors** to match your domain/branding

---

## 🔗 Related Documentation

- [README.md](README.md) - Main project documentation
- [ARCHITETTURA_PROCESSGGAN.md](Docs/ARCHITETTURA_PROCESSGGAN.md) - Technical architecture
- [README_PROCESSGGAN.md](README_PROCESSGGAN.md) - ProcessGAN user guide
- [utils/visualization.py](utils/visualization.py) - Visualization API reference

---

## 💡 Tips

- **Keyboard shortcuts** are faster than clicking buttons
- **Hover over nodes/edges** in web viewer to see details
- **Double-click** in web viewer to reset zoom
- **Right-click** in web viewer for more options
- **Use `--seed`** parameter for reproducible results
- **Regenerate button** (R key) creates new random samples
- **HTML files** are standalone - share them via email/cloud

---

## 📧 Support

For visualization issues or feature requests, please open an issue on GitHub with:
- Visualization script used
- Error message (if any)
- Expected vs actual behavior
- Screenshots (if applicable)
