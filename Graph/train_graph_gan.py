"""
Training script for Graph Process GAN

Full training pipeline with:
- Data loading from .g files
- Model creation and compilation
- Comprehensive callbacks (ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger)
- Validation during training
- Automatic checkpointing and logging
"""

from graph_process_gan import GraphProcessGAN
from graph_dataset import GraphDataset
from ig_loader import load_ig_for_gan
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

def save_graph_to_txt(adj_matrix, node_matrix, idx_to_activity, filename, feature_matrix=None):
    """
    Save a single graph to .txt in NetworkX-readable format (adjacency matrix as edges).
    
    Args:
        adj_matrix: (max_nodes, max_nodes, num_edge_types) adjacency
        node_matrix: (max_nodes, num_activities) node labels
        idx_to_activity: Dict for activity names
        filename: Output .txt file path
        feature_matrix: (max_nodes, num_features) temporal features (optional)
    """
    max_nodes = adj_matrix.shape[0]
    num_activities = node_matrix.shape[1]
    
    with open(filename, 'w') as f:
        f.write("# Nodes: node_id: activity\n")
        f.write("# Edges: Edge source target\n\n")
        
        # Write nodes (only non-zero activity nodes)
        for node_id in range(max_nodes):
            activity_probs = node_matrix[node_id]
            if np.sum(activity_probs) > 0:  # Node exists
                activity_idx = np.argmax(activity_probs)
                activity = idx_to_activity.get(activity_idx, f"UNK_{activity_idx}")
                
                # Skip PAD nodes
                if activity == '<PAD>':
                    continue
                    
                f.write(f"Node {node_id}: {activity}\n")
        
        f.write("\n")
        
        # Write edges (support both binary adjacency and multi-channel)
        for source in range(max_nodes):
            for target in range(max_nodes):
                # Support 2D binary adjacency or 3D multi-channel adjacency
                if adj_matrix.ndim == 2:
                    has_edge = adj_matrix[source, target] > 0.5
                elif adj_matrix.ndim == 3:
                    has_edge = np.sum(adj_matrix[source, target, :]) > 0.5
                else:
                    # Fallback: treat any non-zero as edge
                    has_edge = np.any(adj_matrix[source, target] > 0)

                if has_edge:
                    f.write(f"Edge {source} {target}\n")

        # Write temporal features if provided
        if feature_matrix is not None:
            f.write("\n# Temporal Features: node_id: norm_time trace_time prev_event_time\n")
            for node_id in range(max_nodes):
                # Only write features for existing nodes (non-zero activity)
                if np.sum(node_matrix[node_id]) > 0:
                    feats = feature_matrix[node_id]
                    f.write(f"Features {node_id}: {feats[0]:.4f} {feats[1]:.4f} {feats[2]:.4f}\n")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))


class SampleGraphsCallback(keras.callbacks.Callback):
    """Callback to save sample graphs every 20 epochs"""
    
    def __init__(self, output_dir, idx_to_activity, num_samples=5, interval=20):
        super().__init__()
        self.output_dir = output_dir
        self.idx_to_activity = idx_to_activity
        self.num_samples = num_samples
        self.interval = interval
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            print(f'\n Generating {self.num_samples} sample graphs at epoch {epoch + 1}...')
            
            # Generate samples
            sample_adj, sample_nodes, sample_features = self.model.generate_graphs(
                num_samples=self.num_samples,
                temperature=0.5,
                hard=True
            )
            
            # Save each sample
            for i in range(self.num_samples):
                adj = sample_adj[i]
                nodes = sample_nodes[i]
                features = sample_features[i]
                filename = os.path.join(self.output_dir, f'epoch_{epoch + 1}_sample_{i}.txt')
                save_graph_to_txt(adj, nodes, self.idx_to_activity, filename, features)
            
            print(f'  Saved to {self.output_dir}')


class DualLearningRateScheduler(keras.callbacks.Callback):
    """Custom LR scheduler for GAN with two optimizers"""

    def __init__(self, decay_rate=0.98, min_lr=1e-7, verbose=1):
        super().__init__()
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        # Get current learning rates
        d_lr = float(self.model.d_optimizer.learning_rate.numpy())
        g_lr = float(self.model.g_optimizer.learning_rate.numpy())

        # Apply decay
        new_d_lr = max(d_lr * self.decay_rate, self.min_lr)
        new_g_lr = max(g_lr * self.decay_rate, self.min_lr)

        # Set new learning rates
        self.model.d_optimizer.learning_rate.assign(new_d_lr)
        self.model.g_optimizer.learning_rate.assign(new_g_lr)

        if self.verbose and epoch % 10 == 0:  # Print every 10 epochs
            print(
                f'\nEpoch {epoch}: D_LR = {new_d_lr:.2e}, G_LR = {new_g_lr:.2e}')


def create_callbacks(
    checkpoint_dir,
    log_dir, 
    monitor='val_d_loss', 
    early_stopping_monitor='val_constraint_loss',
    patience=15
    ):
    """
    Create all important callbacks for training

    Args:
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for TensorBoard logs
        monitor: Metric to monitor
        early_stopping_monitor: Metric to monitor for early stopping
        patience: Patience for early stopping

    Returns:
        List of callback objects
    """
    callbacks = []

    # ================================================================
    # 1. ModelCheckpoint - Save best model
    # ================================================================
    checkpoint_path = os.path.join(checkpoint_dir, 'best', 'model')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        mode='min',
        save_best_only=True,
        # Save weights (not full model - GAN structure is complex)
        save_weights_only=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # ================================================================
    # 2. EarlyStopping - Stop when no improvement
    # ================================================================
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=early_stopping_monitor,
        patience=patience,
        mode='min',
        verbose=1,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)

    # ================================================================
    # 3. TensorBoard - Visualize training
    # ================================================================
    os.makedirs(log_dir, exist_ok=True)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,  # Don't log histograms (too slow for GANs)
        write_graph=False,  # Don't log graph (too large)
        update_freq='epoch',
        profile_batch=0  # Disable profiling
    )
    callbacks.append(tensorboard_callback)

    # ================================================================
    # 4. CSVLogger - Log metrics to CSV
    # ================================================================
    csv_path = os.path.join(log_dir, 'training_metrics.csv')

    csv_logger = keras.callbacks.CSVLogger(
        filename=csv_path,
        separator=',',
        append=False
    )
    callbacks.append(csv_logger)

    # ================================================================
    # 5. Custom LR Scheduler - Exponential decay for both optimizers
    # ================================================================
    lr_scheduler = DualLearningRateScheduler(
        decay_rate=0.99,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(lr_scheduler)

    # ================================================================
    # 6. BackupAndRestore - Automatic checkpoint and recovery
    # ================================================================
    backup_dir = os.path.join(checkpoint_dir, 'backup')

    backup_restore = keras.callbacks.BackupAndRestore(
        backup_dir=backup_dir,
        # Save every epoch (can also use integer for batch frequency)
        save_freq='epoch',
        delete_checkpoint=False  # Keep backup for manual recovery
    )
    callbacks.append(backup_restore)

    return callbacks


def train_graph_gan(
    data_path,
    output_dir='../output_graph_gan',
    # Model hyperparameters
    max_nodes=20,
    noise_dim=128,
    generator_hidden_dims=(256, 512, 1024),
    rgcn_hidden_dims=(128, 64),
    mlp_hidden_dims=(128, 64),
    generator_dropout=0.1,
    discriminator_dropout=0.3,
    # Training hyperparameters
    n_critic=3,
    lambda_gp=10.0,
    lambda_constraint=0.5,
    lambda_structure=15.0,
    lambda_degree=10.0,
    lambda_sparsity=0.0,
    lambda_time_monotonic=10.0,
    lambda_start=3.0,
    lambda_end=6.0,
    batch_size=32,
    epochs=500,
    validation_split=0.1,
    # Data filtering
    min_graph_size=None,
    max_graph_size=None,
    # Optimizer
    d_lr=0.0001,
    g_lr=0.0001,
    beta_1=0.5,
    beta_2=0.9,
    # Temperature scheduling
    temp_start=5.0,
    temp_min=0.5,
    temp_decay=0.9995,
    # Early stopping
    early_stopping_patience=100,
    seed=42
):
    """
    Main training function

    Args:
        data_path: Path to .g file
        output_dir: Output directory for checkpoints and logs
        max_nodes: Maximum nodes per graph
        noise_dim: Latent dimension
        generator_hidden_dims: Generator MLP dimensions
        rgcn_hidden_dims: R-GCN hidden dimensions
        mlp_hidden_dims: Discriminator MLP dimensions
        generator_dropout: Generator dropout rate
        discriminator_dropout: Discriminator dropout rate
        n_critic: Discriminator updates per generator update
        lambda_gp: Gradient penalty weight
        lambda_constraint: Constraint loss weight
        lambda_structure: Structural validity (loop prevention) weight
        lambda_degree: Degree constraint weight
        lambda_sparsity: Sparsity (varying node counts) weight
        lambda_time_monotonic: Monotonic time weight
        batch_size: Training batch size
        epochs: Maximum epochs
        validation_split: Fraction of data for validation
        min_graph_size: Minimum number of nodes to include (None = no filter)
        max_graph_size: Maximum number of nodes to include (None = no filter)
        d_lr: Discriminator learning rate
        g_lr: Generator learning rate
        beta_1: Adam beta_1
        beta_2: Adam beta_2
        temp_start: Initial Gumbel-Softmax temperature
        temp_min: Minimum temperature
        temp_decay: Temperature decay rate
        early_stopping_patience: Patience for early stopping
        seed: Random seed
    """
    # Set seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)

    print('='*80)
    print('GRAPH PROCESS GAN TRAINING')
    print('='*80)

    # ================================================================
    # 1. LOAD DATA
    # ================================================================
    print(f'\n Loading data from: {data_path}')

    data = load_ig_for_gan(data_path, max_trace_nodes=max_nodes)

    # Extract all graphs (train + val + test)
    graphs = data['traces_train'] + data['traces_val'] + data['traces_test']

    vocab = {
        'activity_to_idx': data['activity_to_idx'],
        'idx_to_activity': data['idx_to_activity']
    }

    num_activities = data['num_activities']
    start_idx = vocab['activity_to_idx']['START']
    end_idx = vocab['activity_to_idx']['END']

    print(f'  Loaded {len(graphs)} graphs')
    print(f'  Activities: {num_activities}')
    print(f'  START index: {start_idx}, END index: {end_idx}')

    # ================================================================
    # 2. CREATE DATASET
    # ================================================================
    print('\n Creating dataset...')
    
    if min_graph_size is not None or max_graph_size is not None:
        print(f'  Filtering graphs by size:')
        if min_graph_size is not None:
            print(f'    Min nodes: {min_graph_size}')
        if max_graph_size is not None:
            print(f'    Max nodes: {max_graph_size}')

    dataset = GraphDataset(
        traces=graphs,
        activity_to_idx=vocab['activity_to_idx'],
        max_nodes=max_nodes,
        include_features=True,  # Enable temporal features
        verbose=True,
        min_graph_size=min_graph_size,
        max_graph_size=max_graph_size
    )

    # Get matrices
    adjacency_matrices = np.array(dataset.adjacency_matrices)
    node_matrices = np.array(dataset.node_matrices)
    feature_matrices = np.array(dataset.feature_matrices)
    #num_edge_types = dataset.num_edge_types  # Not needed anymore (binary adjacency)

    print(f'  Adjacency shape: {adjacency_matrices.shape}')
    print(f'  Nodes shape: {node_matrices.shape}')
    print(f'  Features shape: {feature_matrices.shape}')
    # print(f'  Edge types: {num_edge_types}')  # No longer applicable (binary)

    # Compute activity frequencies for constraints
    activity_counts = np.sum(node_matrices, axis=(0, 1))
    activity_frequencies = activity_counts / activity_counts.sum()

    print(f'  Activity frequencies:')
    for activity, idx in sorted(vocab['activity_to_idx'].items(), key=lambda x: x[1]):
        print(f'    {activity:>10s}: {activity_frequencies[idx]:.4f}')

    # ================================================================
    # 3. TRAIN/VAL SPLIT
    # ================================================================
    print(f'\n Splitting data (validation: {validation_split*100:.0f}%)')

    num_samples = len(adjacency_matrices)
    num_val = int(num_samples * validation_split)
    num_train = num_samples - num_val

    # Shuffle
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_adj = adjacency_matrices[train_indices]
    train_nodes = node_matrices[train_indices]
    train_features = feature_matrices[train_indices]
    val_adj = adjacency_matrices[val_indices]
    val_nodes = node_matrices[val_indices]
    val_features = feature_matrices[val_indices]

    print(f'  Training samples: {num_train}')
    print(f'  Validation samples: {num_val}')

    # Create TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_adj, train_nodes, train_features))
    train_dataset = train_dataset.shuffle(1000).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_adj, val_nodes, val_features))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # ================================================================
    # 4. CREATE MODEL
    # ================================================================
    print('\n Creating Graph ProcessGAN...')

    gan = GraphProcessGAN(
        max_nodes=max_nodes,
        num_activities=num_activities,
        start_idx=start_idx,
        end_idx=end_idx,
        activity_frequencies=activity_frequencies,
        noise_dim=noise_dim,
        generator_hidden_dims=generator_hidden_dims,
        generator_dropout=generator_dropout,
        rgcn_hidden_dims=rgcn_hidden_dims,
        mlp_hidden_dims=mlp_hidden_dims,
        discriminator_dropout=discriminator_dropout,
        n_critic=n_critic,
        lambda_gp=lambda_gp,
        lambda_constraint=lambda_constraint,
        lambda_structure=lambda_structure,
        lambda_degree=lambda_degree,
        lambda_sparsity=lambda_sparsity,
        lambda_time_monotonic=lambda_time_monotonic,
        lambda_start=lambda_start,
        lambda_end=lambda_end,
        temp_start=temp_start,
        temp_min=temp_min,
        temp_decay=temp_decay
    )

    print(f'  Generator parameters:')
    print(f'    Noise dim: {noise_dim}')
    print(f'    Hidden dims: {generator_hidden_dims}')
    print(f'    Dropout: {generator_dropout}')

    print(f'  Discriminator parameters:')
    print(f'    R-GCN dims: {rgcn_hidden_dims}')
    print(f'    MLP dims: {mlp_hidden_dims}')
    print(f'    Dropout: {discriminator_dropout}')

    print(f'  Training parameters:')
    print(f'    batch size: {batch_size}')
    print(f'    epochs: {epochs}')
    print(f'    validation split: {validation_split}')
    print(f'    n_critic: {n_critic}')
    print(f'    lambda_gp: {lambda_gp}')
    print(f'    lambda_constraint: {lambda_constraint}')

    # ================================================================
    # 5. COMPILE MODEL
    # ================================================================
    print('\n Compiling model...')

    d_optimizer = keras.optimizers.Adam(
        learning_rate=d_lr, beta_1=beta_1, beta_2=beta_2, clipnorm=1.0)
    g_optimizer = keras.optimizers.Adam(
        learning_rate=g_lr, beta_1=beta_1, beta_2=beta_2, clipnorm=1.0)

    gan.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer)

    print(f'  Discriminator LR: {d_lr}')
    print(f'  Generator LR: {g_lr}')
    print(f'  Adam betas: ({beta_1}, {beta_2})')

    # ================================================================
    # 6. SETUP CALLBACKS
    # ================================================================
    print('\n Setting up callbacks...')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints', timestamp)
    log_dir = os.path.join(output_dir, 'logs', timestamp)
    samples_dir = os.path.join(output_dir, 'samples', timestamp)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        monitor='val_constraint_loss',
        early_stopping_monitor='val_g_loss',
        patience=early_stopping_patience
    )

    # Add sample graphs callback
    sample_callback = SampleGraphsCallback(
        output_dir=samples_dir, 
        idx_to_activity=vocab['idx_to_activity']
    )
    callbacks.append(sample_callback)

    print(f'  Checkpoint dir: {checkpoint_dir}')
    print(f'  Log dir: {log_dir}')
    print(f'  Early stopping patience: {early_stopping_patience}')
    print(f'  Callbacks: {len(callbacks)}')

    # ================================================================
    # 7. TRAIN
    # ================================================================
    print('\n Starting training...')
    print('='*80)

    history = gan.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print('\n Training complete!')

    # ================================================================
    # 8. SAVE FINAL MODEL
    # ================================================================
    print('\n Saving final model...')

    final_path = os.path.join(checkpoint_dir, 'final')
    os.makedirs(final_path, exist_ok=True)
    gan.save_weights(os.path.join(final_path, 'model'))

    # Save metadata
    metadata = {
        'num_activities': num_activities,
        # 'num_edge_types': num_edge_types,  # Not applicable (binary adjacency)
        'max_nodes': max_nodes,
        'vocab': vocab,
        'activity_frequencies': activity_frequencies.tolist(),
        'training_samples': num_train,
        'validation_samples': num_val,
        'final_epoch': len(history.history['d_loss']),
        'final_d_loss': float(history.history['d_loss'][-1]),
        'final_g_loss': float(history.history['g_loss'][-1])
    }

    import json
    with open(os.path.join(final_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'  Model saved to: {final_path}')

    # ================================================================
    # 9. GENERATE SAMPLE GRAPHS
    # ================================================================
    print('\n Generating sample graphs...')

    sample_adj, sample_nodes, sample_features = gan.generate_graphs(
        num_samples=100,
        temperature=0.5,
        hard=True
    )

    print(f'  Generated {len(sample_adj)} graphs')
    print(f'  Sample adjacency shape: {sample_adj.shape}')
    print(f'  Sample nodes shape: {sample_nodes.shape}')
    print(f'  Sample features shape: {sample_features.shape}')

    # Save samples
    np.save(os.path.join(final_path, 'sample_adjacency.npy'), sample_adj)
    np.save(os.path.join(final_path, 'sample_nodes.npy'), sample_nodes)

    print('\n' + '='*80)
    print('TRAINING COMPLETE!')
    print('='*80)
    print(f'\nCheckpoint directory: {checkpoint_dir}')
    print(f'TensorBoard logs: {log_dir}')
    print(f'\nTo visualize training:')
    print(f'  tensorboard --logdir {log_dir}')

    return gan, history


if __name__ == '__main__':
    """Run training"""
    import argparse

    parser = argparse.ArgumentParser(description='Train Graph Process GAN')

    # Data
    parser.add_argument('--data', type=str,
                        default='../data/Helpdesk_igs_complete.g',
                        help='Path to .g file')
    parser.add_argument('--output', type=str,
                        default='../output_graph_gan',
                        help='Output directory')

    # Model architecture
    parser.add_argument('--max-nodes', type=int, default=5,
                        help='Maximum nodes per graph')
    parser.add_argument('--noise-dim', type=int, default=128,
                        help='Latent noise dimension')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum epochs')
    parser.add_argument('--n-critic', type=int, default=2,
                        help='Discriminator updates per generator update')
    parser.add_argument('--lambda-gp', type=float, default=10.0,
                        help='Gradient penalty weight')
    parser.add_argument('--lambda-constraint', type=float, default=1.0,
                        help='Constraint loss weight')
    parser.add_argument('--lambda-structure', type=float, default=5.0,
                        help='Structural validity weight')
    parser.add_argument('--lambda-degree', type=float, default=5.0,
                        help='Degree constraint weight')
    parser.add_argument('--lambda-sparsity', type=float, default=0.1,
                        help='Sparsity loss weight')
    parser.add_argument('--lambda-time-monotonic', type=float, default=10.0,
                        help='Monotonic time loss weight')
    parser.add_argument('--lambda-start', type=float, default=5.0,
                        help='Start constraint weight')
    parser.add_argument('--lambda-end', type=float, default=5.0,
                        help='End constraint weight')
    parser.add_argument('--temp-decay', type=float, default=0.9995,
                        help='Temperature decay factor')

    # Optimizer
    parser.add_argument('--d-lr', type=float, default=0.0001,
                        help='Discriminator learning rate')
    parser.add_argument('--g-lr', type=float, default=0.00005,
                        help='Generator learning rate')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Validation split fraction')

    args = parser.parse_args()

    # Train
    gan, history = train_graph_gan(
        data_path=args.data,
        output_dir=args.output,
        max_nodes=args.max_nodes,
        noise_dim=args.noise_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        lambda_constraint=args.lambda_constraint,
        d_lr=args.d_lr,
        g_lr=args.g_lr,
        validation_split=args.validation_split,
        seed=args.seed,
        min_graph_size=5,
        max_graph_size=args.max_nodes,
        lambda_time_monotonic=args.lambda_time_monotonic,
        lambda_start=args.lambda_start,
        lambda_end=args.lambda_end,
        temp_decay=args.temp_decay
    )
