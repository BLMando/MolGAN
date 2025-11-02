"""
Training Schedulers
Lambda mixing, temperature decay, learning rate schedules
"""

import math


class LambdaMixScheduler:
    """
    Schedule GAN/RL mixing coefficient

    Pure GAN (lambda=1.0) for initial epochs, then gradually transitions
    to RL-guided training (lambda→0.2) to incorporate reward optimization.
    """

    def __init__(self, lambda_start: float, lambda_end: float,
                 decay_start: int, total_epochs: int):
        """
        Initialize lambda scheduler

        Args:
            lambda_start: Initial lambda value (e.g., 1.0 for pure GAN)
            lambda_end: Final lambda value (e.g., 0.2 for RL-heavy)
            decay_start: Epoch to start decay (e.g., 50)
            total_epochs: Total training epochs
        """
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.decay_start = decay_start
        self.total_epochs = total_epochs

    def get_lambda(self, epoch: int) -> float:
        """
        Compute lambda for current epoch

        Args:
            epoch: Current training epoch

        Returns:
            Lambda value for GAN/RL mixing
        """
        if epoch < self.decay_start:
            return self.lambda_start
        else:
            # Linear decay from lambda_start to lambda_end
            progress = (epoch - self.decay_start) / \
                (self.total_epochs - self.decay_start)
            return self.lambda_start + progress * (self.lambda_end - self.lambda_start)


class TemperatureScheduler:
    """
    Schedule Gumbel-Softmax temperature

    High temperature (e.g., 5.0) for exploration in early training,
    exponentially decaying to low temperature (e.g., 0.5) for exploitation.
    """

    def __init__(self, temp_start: float, temp_end: float, decay_rate: float):
        """
        Initialize temperature scheduler

        Args:
            temp_start: Initial temperature (e.g., 5.0)
            temp_end: Minimum temperature (e.g., 0.5)
            decay_rate: Exponential decay rate (e.g., 0.96)
        """
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.decay_rate = decay_rate

    def get_temperature(self, epoch: int) -> float:
        """
        Compute temperature for current epoch

        Args:
            epoch: Current training epoch

        Returns:
            Temperature value for Gumbel-Softmax
        """
        temp = self.temp_start * (self.decay_rate ** epoch)
        return max(temp, self.temp_end)


class LearningRateScheduler:
    """
    Cosine Annealing Learning Rate Scheduler with Warm Restarts

    Ideal for GAN training as it:
    - Allows exploration of different local minima via periodic restarts
    - Prevents premature convergence in adversarial training
    - Smooth decay within each cycle

    Uses cosine decay: lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
    where t is current step in cycle, T is cycle length.
    """

    def __init__(self, lr_initial: float, lr_min: float = 1e-6,
                 cycle_length: int = 50, cycle_mult: float = 1.5,
                 warmup_epochs: int = 5):
        """
        Initialize learning rate scheduler

        Args:
            lr_initial: Initial/maximum learning rate (e.g., 1e-4)
            lr_min: Minimum learning rate (e.g., 1e-6)
            cycle_length: Initial cycle length in epochs (e.g., 50)
            cycle_mult: Cycle length multiplier after each restart (e.g., 1.5)
            warmup_epochs: Number of warmup epochs (linear ramp-up)
        """
        self.lr_initial = lr_initial
        self.lr_min = lr_min
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.warmup_epochs = warmup_epochs

        self.current_cycle = 0
        self.current_cycle_length = cycle_length
        self.cycle_start_epoch = warmup_epochs

    def get_learning_rate(self, epoch: int) -> float:
        """
        Compute learning rate for current epoch

        Args:
            epoch: Current training epoch

        Returns:
            Learning rate value
        """
        # Warmup phase: linear ramp-up
        if epoch < self.warmup_epochs:
            return self.lr_initial * (epoch + 1) / self.warmup_epochs

        # Determine position in current cycle
        epoch_in_training = epoch - self.warmup_epochs

        # Check if we need to start a new cycle
        if epoch_in_training >= self.cycle_start_epoch + self.current_cycle_length:
            self.current_cycle += 1
            self.cycle_start_epoch = epoch_in_training
            self.current_cycle_length = int(
                self.cycle_length * (self.cycle_mult ** self.current_cycle))

        # Cosine annealing within cycle
        t = epoch_in_training - self.cycle_start_epoch
        T = self.current_cycle_length

        cos_decay = 0.5 * (1 + math.cos(math.pi * t / T))
        lr = self.lr_min + (self.lr_initial - self.lr_min) * cos_decay

        return lr

    def reset_cycle(self):
        """Reset to start of first cycle (useful for manual restarts)"""
        self.current_cycle = 0
        self.current_cycle_length = self.cycle_length
        self.cycle_start_epoch = self.warmup_epochs
