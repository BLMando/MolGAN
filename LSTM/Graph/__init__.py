"""
Graph-based Process GAN Module

Components:
- ig_loader: Instance graph file loader
- graph_dataset: Convert graphs to matrices
- graph_generator: Generate process graphs
- graph_discriminator: Classify real vs fake graphs
- rgcn_layers: Relational GCN layers
- graph_constraints: Process mining constraints
- graph_process_gan: Main GAN class with Keras API
- graph_utils: Utility functions
"""

from .ig_loader import InstanceGraphLoader, load_ig_for_gan
from .graph_dataset import GraphDataset, EdgeTypeClassifier
from .graph_generator import GraphGenerator, GraphGeneratorWithFeatures
from .graph_discriminator import GraphDiscriminator, GraphDiscriminatorWithFeatures
from .rgcn_layers import RGCNLayer, RGCNStack
from .graph_constraints import GraphProcessConstraints
from .graph_process_gan import GraphProcessGAN
from .graph_utils import (
    gumbel_softmax,
    wasserstein_loss,
    gradient_penalty_graph,
    global_mean_pool,
    global_max_pool,
    compute_graph_statistics,
    visualize_graph
)

__all__ = [
    # Loaders
    'InstanceGraphLoader',
    'load_ig_for_gan',
    
    # Dataset
    'GraphDataset',
    'EdgeTypeClassifier',
    
    # Models
    'GraphGenerator',
    'GraphGeneratorWithFeatures',
    'GraphDiscriminator',
    'GraphDiscriminatorWithFeatures',
    'GraphProcessConstraints',
    'GraphProcessGAN',
    
    # Layers
    'RGCNLayer',
    'RGCNStack',
    
    # Utils
    'gumbel_softmax',
    'wasserstein_loss',
    'gradient_penalty_graph',
    'global_mean_pool',
    'global_max_pool',
    'compute_graph_statistics',
    'visualize_graph',
]

__version__ = '1.0.0'
