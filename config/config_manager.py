import os
import yaml
import argparse
from typing import Dict, Any


class ConfigManager:
    """
    Configuration management

    Handles:
    - YAML config file loading
    - Nested structure flattening
    - Tuple string parsing
    - Configuration validation
    - CLI argument parsing
    """

    @staticmethod
    def parse_tuple_string(s: str) -> tuple:
        """
        Convert string representation of tuple to actual tuple

        Examples:
            "(128, 256, 256, 128)" → (128, 256, 256, 128)
            "(64, 64)" → (64, 64)

        Args:
            s: String representation of tuple or actual tuple/list

        Returns:
            Tuple of integers
        """
        if isinstance(s, (tuple, list)):
            return tuple(s)

        # Remove parentheses and spaces, then split by comma
        s = s.strip('()').replace(' ', '')
        if not s:
            return ()

        return tuple(int(x) for x in s.split(','))

    @staticmethod
    def load_from_yaml(yaml_path: str = 'config.yml') -> Dict[str, Any]:
        """
        Load configuration from YAML file and convert to flat dict

        Args:
            yaml_path: Path to YAML config file

        Returns:
            Flat dictionary with all config parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Flatten nested structure
        flat_config = {}

        # Data section
        if 'data' in config:
            flat_config['data_file'] = config['data']['data_file']
            flat_config['max_activities'] = config['data']['max_activities']
            flat_config['validation_split'] = config['data']['validation_split']
            flat_config['test_split'] = config['data']['test_split']

        # Model section
        if 'model' in config:
            flat_config['z_dim'] = config['model']['z_dim']
            flat_config['decoder_units'] = ConfigManager.parse_tuple_string(
                config['model']['decoder_units'])
            flat_config['discriminator_units'] = ConfigManager.parse_tuple_string(
                config['model']['discriminator_units'])
            flat_config['mlp_units'] = config['model']['mlp_units']
            flat_config['dropout_rate'] = config['model'].get(
                'dropout_rate', 0.1)
            flat_config['enforce_start'] = config['model'].get(
                'enforce_start', True)

        # Training section
        if 'training' in config:
            flat_config['batch_size'] = config['training']['batch_size']
            flat_config['epochs'] = config['training']['epochs']
            flat_config['learning_rate'] = config['training']['learning_rate']
            flat_config['learning_rate_D'] = config['training']['learning_rate_D']
            flat_config['learning_rate_V'] = config['training']['learning_rate_V']
            flat_config['n_critic'] = config['training']['n_critic']
            flat_config['gradient_penalty_weight'] = config['training'].get(
                'gradient_penalty_weight', 10.0)

            flat_config['temperature_start'] = config['training']['temperature_start']
            flat_config['temperature_end'] = config['training']['temperature_end']
            flat_config['temperature_decay'] = config['training']['temperature_decay']

            flat_config['lambda_start'] = config['training']['lambda_start']
            flat_config['lambda_end'] = config['training']['lambda_end']
            flat_config['lambda_decay_start'] = config['training']['lambda_decay_start']

            flat_config['early_stopping_patience'] = config['training']['early_stopping_patience']
            flat_config['early_stopping_metric'] = config['training']['early_stopping_metric']
            flat_config['early_stopping_threshold'] = config['training']['early_stopping_threshold']

        # Reward section
        if 'reward' in config:
            flat_config['use_rl'] = config['reward']['use_rl']
            flat_config['reward_weights_early'] = config['reward']['weights_early']
            flat_config['reward_weights_late'] = config['reward']['weights_late']
            flat_config['reward_transition_epoch'] = config['reward']['transition_epoch']

        # Evaluation section
        if 'evaluation' in config:
            flat_config['n_samples_eval'] = config['evaluation']['n_samples_eval']
            flat_config['eval_freq'] = config['evaluation'].get('eval_freq', 5)

        # Output section
        if 'output' in config:
            flat_config['save_dir'] = config['output']['save_dir']
            flat_config['save_every'] = config['output']['save_every']
            flat_config['log_every'] = config['output']['log_every']
            flat_config['log_dir'] = config['output'].get(
                'log_dir', 'results/logs')

        return flat_config

    @staticmethod
    def parse_args():
        """
        Parse command-line arguments

        Returns:
            Parsed arguments namespace
        """
        parser = argparse.ArgumentParser(
            description='ProcessGAN Training with YAML Config')

        # Config file (only CLI argument)
        parser.add_argument('--config', type=str, default='config.yml',
                            help='Path to YAML configuration file (default: config.yml)')

        return parser.parse_args()

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required keys
        required_keys = [
            'data_file', 'max_activities', 'z_dim', 'epochs',
            'batch_size', 'learning_rate', 'save_dir'
        ]

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate ranges
        if config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")

        if config['epochs'] <= 0:
            raise ValueError("epochs must be positive")

        if config['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")

        if config['max_activities'] <= 0:
            raise ValueError("max_activities must be positive")

        if config['z_dim'] <= 0:
            raise ValueError("z_dim must be positive")
