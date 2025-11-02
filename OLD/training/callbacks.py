"""
Training Callbacks
Early stopping, dynamic reward weights, checkpoint management
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {msg}')


class DynamicRewardWeights:
    """
    Manages automatic transition between early and late reward weights

    Early phase: Focus on validity (getting valid traces)
    Late phase: Balanced rewards (optimizing fitness, conformance, diversity)
    """

    def __init__(self, weights_early: Dict[str, float], weights_late: Dict[str, float],
                 transition_epoch: int):
        """
        Initialize dynamic reward weights

        Args:
            weights_early: Reward weights for early phase (focus on validity)
            weights_late: Reward weights for late phase (balanced optimization)
            transition_epoch: Epoch number to transition from early to late phase
        """
        self.weights_early = weights_early
        self.weights_late = weights_late
        self.transition_epoch = transition_epoch
        self.current_phase = 'early'

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get reward weights for current epoch

        Args:
            epoch: Current training epoch

        Returns:
            Dictionary of reward weights for current phase
        """
        new_phase = 'late' if epoch >= self.transition_epoch else 'early'

        # Log phase transition
        if new_phase != self.current_phase and epoch > 0:
            self.current_phase = new_phase
            weights = self.weights_late if new_phase == 'late' else self.weights_early
            log(f"🔄 REWARD WEIGHTS TRANSITION: {self.current_phase.upper()} PHASE")
            log(f"   New weights: {weights}")

        return self.weights_late if epoch >= self.transition_epoch else self.weights_early

    def get_phase(self, epoch: int) -> str:
        """
        Get current training phase

        Args:
            epoch: Current training epoch

        Returns:
            'early' or 'late'
        """
        return 'late' if epoch >= self.transition_epoch else 'early'


class EarlyStoppingMonitor:
    """
    Custom early stopping for manual training loop

    Monitors a metric and stops training if:
    1. Metric doesn't improve for N epochs (patience)
    2. Metric reaches threshold target
    """

    def __init__(self, patience: int = 20, metric: str = 'valid_rate',
                 threshold: float = 0.85, mode: str = 'max'):
        """
        Initialize early stopping monitor

        Args:
            patience: Number of epochs to wait for improvement
            metric: Metric name to monitor (e.g., 'valid_rate', 'reward_mean')
            threshold: Target threshold - stop if reached
            mode: 'max' (higher is better) or 'min' (lower is better)
        """
        self.patience = patience
        self.metric = metric
        self.threshold = threshold
        self.mode = mode

        self.best_value = -float('inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = None

    def check(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop

        Args:
            epoch: Current training epoch
            metrics: Dictionary of metrics from evaluation

        Returns:
            True if training should stop, False otherwise
        """
        if self.metric not in metrics:
            log(f"⚠️  Warning: Metric '{self.metric}' not found in metrics.",
                level='WARNING')
            return False

        current = metrics[self.metric]

        # Check if threshold reached
        threshold_reached = (self.mode == 'max' and current >= self.threshold) or \
            (self.mode == 'min' and current <= self.threshold)

        if threshold_reached:
            log(f"Threshold reached: {self.metric}={current:.4f} {'≥' if self.mode == 'max' else '≤'} {self.threshold}")
            log(f"Early stopping triggered at epoch {epoch + 1}")
            self.stopped_epoch = epoch
            return True

        # Check for improvement
        improved = (self.mode == 'max' and current > self.best_value) or \
                   (self.mode == 'min' and current < self.best_value)

        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            log(f"New best {self.metric}: {current:.4f} (epoch {epoch + 1})")
        else:
            self.wait += 1
            if self.wait % 5 == 0:
                log(f"No improvement for {self.wait}/{self.patience} epochs (best: {self.best_value:.4f} @ epoch {self.best_epoch + 1})")

            if self.wait >= self.patience:
                log(
                    f"Early stopping: No improvement for {self.patience} epochs")
                log(f"Best {self.metric}: {self.best_value:.4f} at epoch {self.best_epoch + 1}")
                self.stopped_epoch = epoch
                return True

        return False

    def get_best_info(self) -> Dict[str, Any]:
        """
        Get information about best model

        Returns:
            Dictionary with best_value, best_epoch, stopped_epoch
        """
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch + 1,
            'stopped_epoch': self.stopped_epoch + 1 if self.stopped_epoch is not None else None
        }


class CheckpointManager:
    """
    Manages model checkpoints during training

    Handles:
    - Checkpoint directory creation
    - Model saving
    - Info file generation
    - Best model tracking
    """

    def __init__(self, save_dir: str):
        """
        Initialize checkpoint manager

        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = save_dir
        self.best_dir = os.path.join(save_dir, 'best')

        # Best model tracking
        self.best_reward = -float('inf')
        self.best_epoch = 0
        self.best_metrics = {}

        # Create directories
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.best_dir).mkdir(parents=True, exist_ok=True)

        log(f'Checkpoint directory ready: {self.save_dir}')

    def save_checkpoint(self, model, epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoint (automatically tracks best model)

        Args:
            model: ProcessGAN model to save
            epoch: Current training epoch
            metrics: Dictionary of metrics to save
        """
        # Check if this is the best model based on reward_mean
        current_reward = metrics.get('reward_mean', -float('inf'))
        is_best = current_reward > self.best_reward

        if is_best:
            self.best_reward = current_reward
            self.best_epoch = epoch + 1
            self.best_metrics = metrics.copy()

        # Determine save directory
        save_path = self.best_dir if is_best else self.save_dir

        # Save model weights
        model.generator.save_weights(os.path.join(save_path, 'generator'))
        model.discriminator.save_weights(
            os.path.join(save_path, 'discriminator'))
        model.value_network.save_weights(
            os.path.join(save_path, 'value_network'))

        # Save checkpoint info (filter only numeric metrics)
        numeric_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

        info = {
            'epoch': epoch + 1,
            'metrics': numeric_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(os.path.join(save_path, 'info.txt'), 'w') as f:
            json.dump(info, f, indent=2)

        if is_best:
            log(
                f'✨ New BEST model saved! Epoch {epoch + 1}, Reward: {current_reward:.4f}')
        else:
            log(f'💾 Checkpoint saved (epoch {epoch + 1})')

    def get_save_dir(self) -> str:
        """Get save directory path"""
        return self.save_dir

    def get_best_dir(self) -> str:
        """Get best model directory path"""
        return self.best_dir

    def get_best_info(self) -> Dict[str, Any]:
        """
        Get information about the best saved model

        Returns:
            Dictionary with best epoch, reward, and metrics
        """
        return {
            'epoch': self.best_epoch,
            'reward': self.best_reward,
            'metrics': self.best_metrics
        }
