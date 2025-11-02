"""
Training Visualization
Plot training history, losses, rewards, quality metrics
"""

import os
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {msg}')


class TrainingVisualizer:
    """
    Create training plots and visualizations

    Generates comprehensive plots for:
    - Training/validation losses
    - Reward components
    - Quality metrics (valid/unique/novel rates)
    - Overview dashboard
    """

    def __init__(self, save_dir: str):
        """
        Initialize visualizer

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        self.plots_dir = os.path.join(save_dir, 'plots')
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_losses(self, history: Dict):
        """
        Plot training vs validation losses

        Args:
            history: Dictionary with training history
        """
        epochs = history['epoch']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training vs Validation Losses',
                     fontsize=16, fontweight='bold')

        # Discriminator Loss
        axes[0, 0].plot(epochs, history['loss_D'], 'b-',
                        linewidth=2, label='Training')
        axes[0, 0].plot(epochs, history['val_loss_D'], 'r--',
                        linewidth=2, label='Validation')
        axes[0, 0].set_title('Discriminator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Generator Loss
        axes[0, 1].plot(epochs, history['loss_G'], 'b-',
                        linewidth=2, label='Training')
        axes[0, 1].plot(epochs, history['val_loss_G'], 'r--',
                        linewidth=2, label='Validation')
        axes[0, 1].set_title('Generator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # RL Loss
        axes[1, 0].plot(epochs, history['loss_RL'], 'b-',
                        linewidth=2, label='Training')
        axes[1, 0].plot(epochs, history['val_loss_RL'], 'r--',
                        linewidth=2, label='Validation')
        axes[1, 0].set_title('Reinforcement Learning Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Value Network Loss
        axes[1, 1].plot(epochs, history['loss_V'], 'b-',
                        linewidth=2, label='Training')
        axes[1, 1].plot(epochs, history['val_loss_V'], 'r--',
                        linewidth=2, label='Validation')
        axes[1, 1].set_title('Value Network Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_validation_losses.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_rewards(self, history: Dict):
        """
        Plot reward metrics

        Args:
            history: Dictionary with training history
        """
        epochs = history['epoch']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reward Metrics', fontsize=16, fontweight='bold')

        # Total Reward
        axes[0, 0].plot(epochs, history['eval_reward'], 'b-',
                        linewidth=2, label='Evaluation')
        axes[0, 0].plot(epochs, history['reward_mean'], 'r--',
                        linewidth=2, alpha=0.7, label='Training')
        axes[0, 0].set_title('Total Reward')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Validity
        axes[0, 1].plot(epochs, history['eval_validity'], 'g-', linewidth=2)
        axes[0, 1].set_title('Validity Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Validity')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

        # Fitness
        axes[1, 0].plot(epochs, history['eval_fitness'], 'orange', linewidth=2)
        axes[1, 0].set_title('Fitness Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Fitness')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])

        # Diversity
        axes[1, 1].plot(epochs, history['eval_diversity'],
                        'purple', linewidth=2)
        axes[1, 1].set_title('Diversity Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Diversity')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'rewards.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_quality_metrics(self, history: Dict):
        """
        Plot quality metrics (valid/unique/novel rates)

        Args:
            history: Dictionary with training history
        """
        epochs = history['epoch']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Generation Quality Metrics',
                     fontsize=16, fontweight='bold')

        # Valid Rate
        axes[0].plot(epochs, [r * 100 for r in history['valid_rate']],
                     'b-', linewidth=2)
        axes[0].set_title('Valid Traces (%)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Percentage')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 100])

        # Unique Rate
        axes[1].plot(epochs, [r * 100 for r in history['unique_rate']],
                     'g-', linewidth=2)
        axes[1].set_title('Unique Traces (%)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Percentage')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 100])

        # Novel Rate
        axes[2].plot(epochs, [r * 100 for r in history['novel_rate']],
                     'r-', linewidth=2)
        axes[2].set_title('Novel Traces (%)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Percentage')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'quality_metrics.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_overview(self, history: Dict):
        """
        Plot comprehensive overview dashboard

        Args:
            history: Dictionary with training history
        """
        epochs = history['epoch']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Overview', fontsize=16, fontweight='bold')

        # D + G Loss
        axes[0, 0].plot(epochs, history['loss_D'], 'b-',
                        linewidth=2, label='Discriminator')
        axes[0, 0].plot(epochs, history['loss_G'], 'r-',
                        linewidth=2, label='Generator')
        axes[0, 0].set_title('GAN Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Gradient Penalty
        axes[0, 1].plot(epochs, history['grad_penalty'], 'orange', linewidth=2)
        axes[0, 1].set_title('Gradient Penalty (WGAN-GP)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Penalty')
        axes[0, 1].grid(True, alpha=0.3)

        # Reward Components
        axes[0, 2].plot(epochs, history['eval_validity'],
                        label='Validity', linewidth=2)
        axes[0, 2].plot(epochs, history['eval_fitness'],
                        label='Fitness', linewidth=2)
        axes[0, 2].plot(epochs, history['eval_conformance'],
                        label='Conformance', linewidth=2)
        axes[0, 2].plot(epochs, history['eval_diversity'],
                        label='Diversity', linewidth=2)
        axes[0, 2].set_title('Reward Components')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        axes[0, 2].set_ylim([0, 1])

        # RL Losses
        axes[1, 0].plot(epochs, history['loss_RL'], 'g-',
                        linewidth=2, label='RL Loss')
        axes[1, 0].plot(epochs, history['loss_V'], 'm-',
                        linewidth=2, label='V Loss')
        axes[1, 0].set_title('RL Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Total Reward
        axes[1, 1].plot(epochs, history['eval_reward'], 'b-',
                        linewidth=2, marker='o', markersize=4)
        axes[1, 1].set_title('Total Reward (Evaluation)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True, alpha=0.3)

        # Quality Summary
        axes[1, 2].plot(
            epochs, [r * 100 for r in history['valid_rate']], label='Valid', linewidth=2)
        axes[1, 2].plot(
            epochs, [r * 100 for r in history['unique_rate']], label='Unique', linewidth=2)
        axes[1, 2].plot(
            epochs, [r * 100 for r in history['novel_rate']], label='Novel', linewidth=2)
        axes[1, 2].set_title('Quality Rates (%)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Percentage')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        axes[1, 2].set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'overview.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_all_plots(self, history: Dict):
        """
        Generate and save all plots

        Args:
            history: Dictionary with training history
        """
        log(f"📊 Generating training plots...")

        self.plot_losses(history)
        self.plot_rewards(history)
        self.plot_quality_metrics(history)
        self.plot_overview(history)

        log(f"📊 Training plots saved in: {self.plots_dir}")
        log(f"  - training_validation_losses.png")
        log(f"  - rewards.png")
        log(f"  - quality_metrics.png")
        log(f"  - overview.png")
