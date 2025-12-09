"""
Graph Discriminator - Classify real vs generated process graphs

Architecture:
    (Adjacency, Nodes) → R-GCN layers → Global pooling → MLP → Score
    
Uses R-GCN to process multi-relational graph structure,
then pools node features and classifies.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from rgcn_layers import RGCNStack
from rgcn_layers import RGCNStack
from graph_utils import global_mean_pool, global_sum_pool


class GraphDiscriminator(keras.Model):
    """
    Discriminator network: (Adjacency, Nodes) → Score
    
    Uses R-GCN to process graphs with multiple edge types,
    then classifies as real or fake using Wasserstein loss.
    """
    
    def __init__(self,
                 num_activities,
                 rgcn_hidden_dims=(128, 64),
                 mlp_hidden_dims=(128, 64),
                 dropout_rate=0.3,
                 use_layer_norm=True,
                 pooling_method='max',
                 name='graph_discriminator'):
        """
        Args:
            num_activities: Number of activity types (input features)
            rgcn_hidden_dims: Hidden dimensions for R-GCN layers
            mlp_hidden_dims: Hidden dimensions for MLP classifier
            dropout_rate: Dropout rate
            use_layer_norm: Use layer normalization in R-GCN
            pooling_method: 'mean' or 'max' for graph pooling
            name: Model name
        """
        super(GraphDiscriminator, self).__init__(name=name)
        
        self.num_activities = num_activities
        self.rgcn_hidden_dims = rgcn_hidden_dims
        self.mlp_hidden_dims = mlp_hidden_dims
        self.pooling_method = pooling_method
        
        # R-GCN stack for processing graph structure
        self.rgcn = RGCNStack(
            hidden_dims=rgcn_hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )
        
        # MLP classifier
        self.classifier_layers = []
        for i, units in enumerate(mlp_hidden_dims):
            self.classifier_layers.append(
                layers.Dense(
                    units,
                    kernel_constraint=tf.keras.constraints.MaxNorm(1.0),
                    name=f'mlp_{i}'
                )
            )
            self.classifier_layers.append(layers.LeakyReLU(0.2))
            self.classifier_layers.append(layers.Dropout(dropout_rate))
        
        # Output layer (no activation for WGAN)
        self.output_layer = layers.Dense(
            1,
            kernel_constraint=tf.keras.constraints.MaxNorm(1.0),
            name='output'
        )
    
    def call(self, adjacency, nodes, training=False):
        """
        Forward pass
        
        Args:
            adjacency: Binary adjacency matrices (batch, max_nodes, max_nodes)
            nodes: Node feature matrices (batch, max_nodes, num_activities)
            training: Training mode flag
            
        Returns:
            scores: Discriminator scores (batch, 1) for WGAN
        """
        # Process through R-GCN
        node_features = self.rgcn(nodes, adjacency, training=training)
        
        # Global pooling to get graph-level representation
        if self.pooling_method == 'mean':
            graph_features = global_mean_pool(node_features)
        elif self.pooling_method == 'sum':
            graph_features = global_sum_pool(node_features)
        elif self.pooling_method == 'max':
            graph_features = tf.reduce_max(node_features, axis=1)
        else:
            raise ValueError(f'Unknown pooling method: {self.pooling_method}')
        
        # Pass through MLP classifier
        h = graph_features
        for layer in self.classifier_layers:
            h = layer(h, training=training) if isinstance(layer, layers.Dropout) else layer(h)
        
        # Output score
        scores = self.output_layer(h)
        
        return scores


class GraphDiscriminatorWithFeatures(GraphDiscriminator):
    """
    Extended discriminator that also considers temporal features
    """
    
    def __init__(self,
                 num_activities,
                 num_features=3,
                 rgcn_hidden_dims=(128, 64),
                 mlp_hidden_dims=(128, 64),
                 dropout_rate=0.3,
                 use_layer_norm=True,
                 pooling_method='mean',
                 name='graph_discriminator_features'):
        """
        Args:
            num_features: Number of temporal features
            Other args same as GraphDiscriminator
        """
        super().__init__(
            num_activities=num_activities,
            rgcn_hidden_dims=rgcn_hidden_dims,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            pooling_method=pooling_method,
            name=name
        )
        
        self.num_features = num_features
        
        # Feature processing layer
        self.feature_processor = layers.Dense(
            rgcn_hidden_dims[-1] // 2,
            activation='relu',
            name='feature_processor'
        )
        
        # Projection layer to bring concatenated features back to expected dimension
        self.feature_projection = layers.Dense(
            rgcn_hidden_dims[-1],
            activation='relu',
            name='feature_projection'
        )
    
    def call(self, adjacency, nodes, features=None, training=False):
        """
        Forward pass with features
        
        Args:
            adjacency: Adjacency matrices
            nodes: Node matrices
            features: Temporal features (batch, max_nodes, num_features)
            training: Training mode
            
        Returns:
            scores: Discriminator scores
        """
        # Process through R-GCN
        node_features = self.rgcn(nodes, adjacency, training=training)
        
        # If features provided, concatenate and project them
        if features is not None:
            processed_features = self.feature_processor(features)
            node_features = tf.concat([node_features, processed_features], axis=-1)
            node_features = self.feature_projection(node_features)
        
        # Global pooling
        if self.pooling_method == 'mean':
            graph_features = global_mean_pool(node_features)
        elif self.pooling_method == 'sum':
            graph_features = global_sum_pool(node_features)
        else:
            graph_features = tf.reduce_max(node_features, axis=1)
        
        # Classify
        h = graph_features
        for layer in self.classifier_layers:
            h = layer(h, training=training) if isinstance(layer, layers.Dropout) else layer(h)
        
        scores = self.output_layer(h)
        
        return scores
