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
- graph_conversion: Convert between data formats
- graph_evaluation: Evaluate generated graphs with IG metrics
- ig_metrics: Instance Graph evaluation metrics (Acc, MC, AG)
"""

from .ig_loader import InstanceGraphLoader, load_ig_for_gan
from .graph_dataset import GraphDataset
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
from .graph_conversion import (
    matrix_to_networkx,
    trace_to_networkx,
    batch_matrices_to_networkx,
    traces_to_networkx,
    txt_file_to_networkx,
    txt_files_to_networkx
)
from .graph_evaluation import (
    evaluate_generated_graphs,
    evaluate_from_txt_files,
    print_evaluation_results
)
from . import ig_metrics

__all__ = [
    # Loaders
    'InstanceGraphLoader',
    'load_ig_for_gan',
    
    # Dataset
    'GraphDataset',
    
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
    
    # Conversion
    'matrix_to_networkx',
    'trace_to_networkx',
    'batch_matrices_to_networkx',
    'traces_to_networkx',
    'txt_file_to_networkx',
    'txt_files_to_networkx',
    
    # Evaluation
    'evaluate_generated_graphs',
    'evaluate_from_txt_files',
    'print_evaluation_results',
    'ig_metrics',
]

__version__ = '1.0.0'

