"""
Checkpoint Loader
Load ProcessGAN models from checkpoint directories
"""

import os
import tensorflow as tf
from datetime import datetime
from typing import Dict, Tuple, Any

# Import ProcessGAN
from models.process_gan import ProcessGAN


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {msg}')


class CheckpointLoader:
    """
    Load ProcessGAN models from checkpoint directories

    Handles:
    - Checkpoint metadata loading (info.txt)
    - Model architecture recreation
    - Weight loading for generator, discriminator, value network
    """

    @staticmethod
    def load(checkpoint_dir: str, dataset: Any) -> Tuple[ProcessGAN, Dict[str, str]]:
        """
        Load ProcessGAN model from checkpoint directory

        Args:
            checkpoint_dir: Path to checkpoint directory (e.g., 'checkpoints/process_gan/best')
            dataset: ProcessDataset instance with loaded data

        Returns:
            (model, info): Loaded ProcessGAN model and checkpoint metadata
        """
        log(f"Loading model from checkpoint: {checkpoint_dir}")

        # Check if checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_dir}")

        # Load checkpoint metadata
        info = CheckpointLoader._load_checkpoint_info(checkpoint_dir)

        # Create model with architecture
        model = CheckpointLoader._create_model(dataset)

        # Load weights
        CheckpointLoader._load_weights(model, checkpoint_dir)

        log("Model loaded successfully!")
        return model, info

    @staticmethod
    def _load_checkpoint_info(checkpoint_dir: str) -> Dict[str, str]:
        """
        Load checkpoint metadata from info.txt

        Args:
            checkpoint_dir: Path to checkpoint directory

        Returns:
            Dictionary with checkpoint metadata
        """
        info_file = os.path.join(checkpoint_dir, 'info.txt')
        info = {}

        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        info[key.strip()] = value.strip()
            log(f"Checkpoint info: {info}")
        else:
            log("No info.txt found in checkpoint directory", level='WARNING')

        return info

    @staticmethod
    def _create_model(dataset: Any) -> ProcessGAN:
        """
        Create ProcessGAN model with training architecture

        Args:
            dataset: ProcessDataset instance

        Returns:
            ProcessGAN model (uninitialized)
        """
        # Create model with same architecture as training
        model = ProcessGAN(
            max_activities=dataset.max_activities,
            flow_types=dataset.flow_num_types,
            activity_types=dataset.activity_num_types,
            embedding_dim=32,  # Default from trainer config
            decoder_units=(64, 128, 128),
            discriminator_units=(64, 64),
            mlp_units=64,
            dropout_rate=0.2,
            enforce_start=True
        )

        # Build model by calling with sample input
        sample_z = tf.random.normal((1, 32))
        sample_adj, sample_nodes = model.generator(sample_z, training=False)
        _ = model.discriminator(sample_adj, sample_nodes, training=False)
        _ = model.value_network(sample_adj, sample_nodes, training=False)

        log("Model architecture created")
        return model

    @staticmethod
    def _load_weights(model: ProcessGAN, checkpoint_dir: str):
        """
        Load weights for generator, discriminator, value network

        Args:
            model: ProcessGAN model
            checkpoint_dir: Path to checkpoint directory
        """
        generator_path = os.path.join(checkpoint_dir, 'generator')
        discriminator_path = os.path.join(checkpoint_dir, 'discriminator')
        value_network_path = os.path.join(checkpoint_dir, 'value_network')

        # Load generator weights
        if CheckpointLoader._check_checkpoint_exists(generator_path):
            model.generator.load_weights(generator_path)
            log("✓ Generator weights loaded")
        else:
            raise FileNotFoundError(
                f"Generator weights not found: {generator_path}")

        # Load discriminator weights
        if CheckpointLoader._check_checkpoint_exists(discriminator_path):
            model.discriminator.load_weights(discriminator_path)
            log("✓ Discriminator weights loaded")
        else:
            raise FileNotFoundError(
                f"Discriminator weights not found: {discriminator_path}")

        # Load value network weights (optional)
        if CheckpointLoader._check_checkpoint_exists(value_network_path):
            model.value_network.load_weights(value_network_path)
            log("✓ Value network weights loaded")
        else:
            log("Value network weights not found, continuing without RL", level='WARNING')

    @staticmethod
    def _check_checkpoint_exists(base_path: str) -> bool:
        """
        Check if TensorFlow checkpoint files exist

        Args:
            base_path: Base path to checkpoint (without extensions)

        Returns:
            True if checkpoint files exist
        """
        data_file = base_path + '.data-00000-of-00001'
        index_file = base_path + '.index'
        return os.path.exists(data_file) and os.path.exists(index_file)
