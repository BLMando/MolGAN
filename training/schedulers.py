"""
Training Schedulers
Lambda mixing, temperature decay, learning rate schedules
"""


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
