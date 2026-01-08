# Graph Process GAN

A **Graph-based Generative Adversarial Network** for process mining, built with TensorFlow/Keras.

## Overview

This project generates synthetic process graphs representing event logs. It uses a GAN architecture with:

- **Generator**: Transforms latent noise into process graphs (adjacency matrices + node features)
- **Discriminator**: Uses R-GCN layers to classify real vs. generated graphs
- **Wasserstein Loss**: For stable training with gradient penalty

## Project Structure

```
Graph/
├── graph_generator.py       # Generator network (z → graphs)
├── graph_discriminator.py   # Discriminator with R-GCN
├── graph_process_gan.py     # Main GAN model (Keras API)
├── train_graph_gan.py       # Training script
├── graph_dataset.py         # Data loading utilities
├── graph_constraints.py     # Process constraints (START/END nodes, timing)
├── graph_evaluation.py      # Evaluation metrics
├── rgcn_layers.py           # R-GCN layer implementation
├── graph_utils.py           # Helper functions (Gumbel-Softmax, pooling)
└── displayGraphs.py         # Visualization tools
```

## Quick Start

### Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy

### Training

```bash
python Graph/train_graph_gan.py --data_path <path_to_data.g> --output_dir ./output
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_nodes` | 5 | Maximum nodes per graph |
| `epochs` | 200 | Training epochs |
| `batch_size` | 32 | Batch size |
| `noise_dim` | 128 | Latent vector dimension |
