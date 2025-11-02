import numpy as np
from typing import List, Dict, Optional, Set
from collections import Counter

# Import specialized scorers
from metrics import ValidityScorer, FitnessScorer, ConformanceScorer, DiversityScorer


class ProcessRewardFunction:
    """
    Multi-objective reward function coordinator

    Computes reward R(trace) ∈ [0,1] as weighted sum of:
    - Validity: Syntactic rules (START/END, length, no loops)
    - Fitness: Token replay fitness
    - Conformance: Alignment cost
    - Diversity: Edit distance from training
    """

    def __init__(self, reference_model=None, training_traces=None, weights=None,
                 start_activity=None, end_activity=None, core_activities=None):
        """
        Initialize reward function

        Args:
            reference_model: Petri net (net, im, fm) for fitness/conformance
            training_traces: Training traces for diversity
            weights: Dict with keys ['validity', 'fitness', 'conformance', 'diversity']
            start_activity: START activity (auto-detected if None)
            end_activity: END activity (auto-detected if None)
            core_activities: Core activities set (auto-detected if None)
        """
        self.reference_model = reference_model
        self.training_traces = training_traces if training_traces is not None else []

        # Auto-detect activities
        if start_activity is None:
            self.start_activity = self._detect_start_activity()
        else:
            self.start_activity = start_activity

        if end_activity is None:
            self.end_activity = self._detect_end_activity()
        else:
            self.end_activity = end_activity

        if core_activities is None:
            self.core_activities = self._detect_core_activities()
        else:
            self.core_activities = core_activities

        self.expected_pattern = self._detect_expected_pattern()

        # Weights setup
        if weights is None:
            self.weights = {
                'validity': 0.30,
                'fitness': 0.25,
                'conformance': 0.25,
                'diversity': 0.20
            }
        else:
            self.weights = weights

        # Normalize weights
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            for key in self.weights:
                self.weights[key] /= weight_sum

        # Initialize scorers
        self.validity_scorer = ValidityScorer(
            training_traces=self.training_traces,
            start_activity=self.start_activity,
            end_activity=self.end_activity,
            core_activities=self.core_activities
        )

        self.fitness_scorer = FitnessScorer(
            reference_model=reference_model,
            core_activities=self.core_activities,
            start_activity=self.start_activity,
            end_activity=self.end_activity
        )

        self.conformance_scorer = ConformanceScorer(
            reference_model=reference_model,
            expected_pattern=self.expected_pattern
        )

        self.diversity_scorer = DiversityScorer(
            training_traces=self.training_traces
        )

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update reward weights dynamically

        Args:
            new_weights: New weights dictionary
        """
        self.weights = new_weights.copy()

        # Normalize
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            for key in self.weights:
                self.weights[key] /= weight_sum

    def _detect_start_activity(self) -> str:
        """Auto-detect START activity"""
        if not self.training_traces:
            return 'Start'

        first_activities = Counter(
            trace[0] for trace in self.training_traces if len(trace) > 0
        )

        if not first_activities:
            return 'Start'

        return first_activities.most_common(1)[0][0]

    def _detect_end_activity(self) -> str:
        """Auto-detect END activity"""
        if not self.training_traces:
            return 'End'

        last_activities = Counter(
            trace[-1] for trace in self.training_traces if len(trace) > 0
        )

        if not last_activities:
            return 'End'

        return last_activities.most_common(1)[0][0]

    def _detect_core_activities(self, min_frequency: float = 0.20) -> Set[str]:
        """Auto-detect core activities (>= 20% frequency)"""
        if not self.training_traces:
            return set()

        activity_count = {}
        total_traces = len(self.training_traces)

        for trace in self.training_traces:
            unique_activities = set(trace)
            for activity in unique_activities:
                activity_count[activity] = activity_count.get(activity, 0) + 1

        core_activities = {
            activity for activity, count in activity_count.items()
            if count / total_traces >= min_frequency
        }

        core_activities.add(self.start_activity)
        core_activities.add(self.end_activity)

        return core_activities

    def _detect_expected_pattern(self) -> List[str]:
        """Auto-detect expected trace pattern (most common trace)"""
        if not self.training_traces:
            return [self.start_activity, self.end_activity]

        trace_counts = Counter(tuple(trace) for trace in self.training_traces)

        if not trace_counts:
            return [self.start_activity, self.end_activity]

        return list(trace_counts.most_common(1)[0][0])

    @staticmethod
    def load_petri_net_model(pnml_path: str):
        """
        Load Petri net from PNML file

        Args:
            pnml_path: Path to PNML file

        Returns:
            Tuple (net, initial_marking, final_marking) or None
        """
        return FitnessScorer.load_petri_net(pnml_path)

    def compute_reward(self, traces: List[List[str]]) -> np.ndarray:
        """
        Compute reward for batch of traces

        Args:
            traces: List of traces

        Returns:
            Array of shape (len(traces), 1) with rewards ∈ [0,1]
        """
        batch_size = len(traces)
        rewards = np.zeros(batch_size)

        # Compute sub-rewards using specialized scorers
        validity_scores = self.validity_scorer.score(traces)
        fitness_scores = self.fitness_scorer.score(traces)
        conformance_scores = self.conformance_scorer.score(traces)
        diversity_scores = self.diversity_scorer.score(traces)

        # Weighted sum
        for i in range(batch_size):
            rewards[i] = (
                self.weights['validity'] * validity_scores[i] +
                self.weights['fitness'] * fitness_scores[i] +
                self.weights['conformance'] * conformance_scores[i] +
                self.weights['diversity'] * diversity_scores[i]
            )

        return rewards.reshape(-1, 1)

    def validity_score(self, traces: List[List[str]]) -> np.ndarray:
        """Compute validity scores (delegated to ValidityScorer)"""
        return self.validity_scorer.score(traces)

    def fitness_score(self, traces: List[List[str]]) -> np.ndarray:
        """Compute fitness scores (delegated to FitnessScorer)"""
        return self.fitness_scorer.score(traces)

    def conformance_score(self, traces: List[List[str]]) -> np.ndarray:
        """Compute conformance scores (delegated to ConformanceScorer)"""
        return self.conformance_scorer.score(traces)

    def diversity_score(self, traces: List[List[str]]) -> np.ndarray:
        """Compute diversity scores (delegated to DiversityScorer)"""
        return self.diversity_scorer.score(traces)

    def evaluate_batch(self, traces: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate batch of traces and return detailed metrics

        Args:
            traces: List of traces

        Returns:
            Dict with mean/std for each sub-reward and total reward
        """
        validity = self.validity_score(traces)
        fitness = self.fitness_score(traces)
        conformance = self.conformance_score(traces)
        diversity = self.diversity_score(traces)
        total_reward = self.compute_reward(traces).flatten()

        metrics = {
            'validity_mean': np.mean(validity),
            'validity_std': np.std(validity),
            'fitness_mean': np.mean(fitness),
            'fitness_std': np.std(fitness),
            'conformance_mean': np.mean(conformance),
            'conformance_std': np.std(conformance),
            'diversity_mean': np.mean(diversity),
            'diversity_std': np.std(diversity),
            'reward_mean': np.mean(total_reward),
            'reward_std': np.std(total_reward),
            'high_quality_rate': np.mean(total_reward >= 0.7),
        }

        return metrics
