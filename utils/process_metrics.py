"""
ProcessMetrics: Reward function for evaluating generated event logs

Implements multi-objective reward function with four components:
1. Validity: Syntactic correctness [0,1]
2. Fitness: Token replay fitness [0,1]
3. Conformance: Alignment cost [0,1]
4. Diversity: Edit distance from training [0,1]
"""

import numpy as np
from typing import List, Dict, Tuple


class ProcessRewardFunction:
    """
    Multi-objective reward function for process mining traces

    Computes reward R(trace) ∈ [0,1] as weighted sum of:
    - Validity: Syntactic rules (Start/End, length, no loops)
    - Fitness: Token replay fitness (simplified)
    - Conformance: Alignment cost (simplified)
    - Diversity: Edit distance from training set
    """

    def __init__(self, reference_model=None, training_traces=None, weights=None):
        """
        Initialize reward function

        Args:
            reference_model: Reference process model (Petri net, BPMN) - optional
            training_traces: List of training traces for diversity calculation
            weights: Dict with keys ['validity', 'fitness', 'conformance', 'diversity']
                    If None, uses default: {0.30, 0.25, 0.25, 0.20}
        """
        self.reference_model = reference_model
        self.training_traces = training_traces if training_traces is not None else []

        if weights is None:
            self.weights = {
                'validity': 0.30,
                'fitness': 0.25,
                'conformance': 0.25,
                'diversity': 0.20
            }
        else:
            self.weights = weights

        # Validate weights sum to 1
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            print(f"Warning: Weights sum to {weight_sum}, normalizing...")
            for key in self.weights:
                self.weights[key] /= weight_sum

    def compute_reward(self, traces: List[List[str]]) -> np.ndarray:
        """
        Compute reward for batch of traces

        Args:
            traces: List of traces (each trace is list of activity names)

        Returns:
            Array of shape (len(traces), 1) with rewards ∈ [0,1]
        """
        batch_size = len(traces)
        rewards = np.zeros(batch_size)

        # Compute sub-rewards
        validity_scores = self.validity_score(traces)
        fitness_scores = self.fitness_score(traces)
        conformance_scores = self.conformance_score(traces)
        diversity_scores = self.diversity_score(traces)

        # Weighted sum
        for i in range(batch_size):
            rewards[i] = (
                self.weights['validity'] * validity_scores[i] +
                self.weights['fitness'] * fitness_scores[i] +
                self.weights['conformance'] * conformance_scores[i] +
                self.weights['diversity'] * diversity_scores[i]
            )

        return rewards.reshape(-1, 1)

    # ─────────────────────────────────────────────────────────
    # SUB-REWARD 1: VALIDITY [0,1]
    # ─────────────────────────────────────────────────────────

    def validity_score(self, traces: List[List[str]]) -> np.ndarray:
        """
        Syntactic validity score

        Checks:
        - Trace is not empty
        - Starts with 'Start'
        - Ends with 'End'
        - Reasonable length (3-20 activities)
        - No consecutive duplicate activities
        - 'End' only appears at last position

        Args:
            traces: List of traces

        Returns:
            Array of shape (len(traces),) with validity scores ∈ [0,1]
        """
        scores = []

        for trace in traces:
            score = 1.0

            # Rule 1: Non-empty trace
            if len(trace) == 0:
                scores.append(0.0)
                continue

            # Rule 2: Must start with 'Start'
            if trace[0] != 'Start':
                score *= 0.5

            # Rule 3: Must end with 'End'
            if trace[-1] != 'End':
                score *= 0.5

            # Rule 4: Reasonable length (3-20 activities)
            if not (3 <= len(trace) <= 20):
                score *= 0.7

            # Rule 5: No consecutive duplicate activities
            for i in range(len(trace) - 1):
                if trace[i] == trace[i+1]:
                    score *= 0.8
                    break

            # Rule 6: 'End' only at last position
            if 'End' in trace[:-1]:
                score *= 0.3

            scores.append(score)

        return np.array(scores, dtype=np.float32)

    # ─────────────────────────────────────────────────────────
    # SUB-REWARD 2: FITNESS [0,1]
    # ─────────────────────────────────────────────────────────

    def fitness_score(self, traces: List[List[str]]) -> np.ndarray:
        """
        Token replay fitness (simplified version)

        In full implementation, would use Petri net token replay.
        Simplified version checks:
        - Presence of expected activities
        - Logical ordering

        Args:
            traces: List of traces

        Returns:
            Array of shape (len(traces),) with fitness scores ∈ [0,1]
        """
        scores = []

        # Expected activity patterns (simplified)
        expected_activities = {'Start', 'End'}
        core_activities = {'Submit', 'Review', 'Approve'}

        for trace in traces:
            if len(trace) == 0:
                scores.append(0.0)
                continue

            fitness = 1.0

            # Check presence of expected activities
            trace_set = set(trace)
            if not expected_activities.issubset(trace_set):
                fitness *= 0.5

            # Bonus for core activities
            core_present = len(trace_set.intersection(core_activities))
            core_bonus = core_present / len(core_activities)
            fitness *= (0.5 + 0.5 * core_bonus)

            # Check logical ordering
            # Start should be before End
            try:
                start_idx = trace.index('Start')
                end_idx = trace.index('End')
                if start_idx >= end_idx:
                    fitness *= 0.3
            except ValueError:
                pass

            # If using reference model, compute proper token replay
            if self.reference_model is not None:
                fitness = self._token_replay_fitness(trace, self.reference_model)

            scores.append(fitness)

        return np.array(scores, dtype=np.float32)

    def _token_replay_fitness(self, trace: List[str], model) -> float:
        """
        Proper token replay fitness (placeholder for Petri net implementation)

        Args:
            trace: Single trace
            model: Reference model (Petri net)

        Returns:
            Fitness score ∈ [0,1]
        """
        # Placeholder: In full implementation, use PM4Py token replay
        # For now, return simplified fitness
        return 0.8

    # ─────────────────────────────────────────────────────────
    # SUB-REWARD 3: CONFORMANCE [0,1]
    # ─────────────────────────────────────────────────────────

    def conformance_score(self, traces: List[List[str]]) -> np.ndarray:
        """
        Conformance checking via alignment (simplified)

        In full implementation, would use A* alignment algorithm.
        Simplified version checks structural similarity.

        Args:
            traces: List of traces

        Returns:
            Array of shape (len(traces),) with conformance scores ∈ [0,1]
        """
        scores = []

        # Expected trace pattern (simplified)
        expected_pattern = ['Start', 'Submit', 'Review', 'Approve', 'End']

        for trace in traces:
            if len(trace) == 0:
                scores.append(0.0)
                continue

            # Simplified alignment: Longest Common Subsequence (LCS)
            lcs_length = self._longest_common_subsequence(trace, expected_pattern)

            # Conformance based on LCS ratio
            max_len = max(len(trace), len(expected_pattern))
            conformance = lcs_length / max_len if max_len > 0 else 0.0

            # If using reference model, compute proper alignment
            if self.reference_model is not None:
                conformance = self._alignment_conformance(trace, self.reference_model)

            scores.append(conformance)

        return np.array(scores, dtype=np.float32)

    def _alignment_conformance(self, trace: List[str], model) -> float:
        """
        Proper alignment-based conformance (placeholder)

        Args:
            trace: Single trace
            model: Reference model

        Returns:
            Conformance score ∈ [0,1]
        """
        # Placeholder: In full implementation, use PM4Py alignment
        return 0.75

    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Compute length of longest common subsequence

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Length of LCS
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    # ─────────────────────────────────────────────────────────
    # SUB-REWARD 4: DIVERSITY [0,1]
    # ─────────────────────────────────────────────────────────

    def diversity_score(self, traces: List[List[str]]) -> np.ndarray:
        """
        Diversity score based on edit distance from training set

        High score = trace is different from training (good for augmentation)
        Low score = trace is similar to training (potential duplicate)

        Args:
            traces: List of traces

        Returns:
            Array of shape (len(traces),) with diversity scores ∈ [0,1]
        """
        scores = []

        if len(self.training_traces) == 0:
            # No training traces: return neutral score
            return np.ones(len(traces), dtype=np.float32) * 0.5

        for trace in traces:
            if len(trace) == 0:
                scores.append(0.0)
                continue

            # Compute minimum edit distance to training traces
            min_distance = float('inf')

            # Sample training traces for efficiency (max 100)
            sample_size = min(100, len(self.training_traces))
            sample_indices = np.random.choice(len(self.training_traces), sample_size, replace=False)

            for idx in sample_indices:
                train_trace = self.training_traces[idx]
                distance = self.edit_distance(trace, train_trace)
                min_distance = min(min_distance, distance)

            # Normalize distance
            max_len = max(len(trace), max(len(self.training_traces[i]) for i in sample_indices))
            diversity = min_distance / max_len if max_len > 0 else 0.0

            # Clip: too much diversity (>0.8) might be unrealistic
            diversity = min(diversity, 0.8) / 0.8

            scores.append(diversity)

        return np.array(scores, dtype=np.float32)

    @staticmethod
    def edit_distance(seq1: List[str], seq2: List[str]) -> int:
        """
        Levenshtein edit distance between two sequences

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Edit distance (number of operations: insert, delete, substitute)
        """
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=np.int32)

        # Initialize
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j

        # Dynamic programming
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    cost = 0
                else:
                    cost = 1

                dp[i, j] = min(
                    dp[i-1, j] + 1,      # Deletion
                    dp[i, j-1] + 1,      # Insertion
                    dp[i-1, j-1] + cost  # Substitution
                )

        return dp[m, n]

    # ─────────────────────────────────────────────────────────
    # EVALUATION AND ANALYSIS
    # ─────────────────────────────────────────────────────────

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
            'high_quality_rate': np.mean(total_reward >= 0.7),  # % with reward >= 0.7
        }

        return metrics

    def print_trace_analysis(self, trace: List[str]) -> None:
        """
        Print detailed analysis of a single trace

        Args:
            trace: Single trace
        """
        print("=" * 60)
        print(f"Trace Analysis: {' → '.join(trace)}")
        print("=" * 60)

        validity = self.validity_score([trace])[0]
        fitness = self.fitness_score([trace])[0]
        conformance = self.conformance_score([trace])[0]
        diversity = self.diversity_score([trace])[0]
        reward = self.compute_reward([trace])[0, 0]

        print(f"Validity:     {validity:.3f} (α={self.weights['validity']:.2f})")
        print(f"Fitness:      {fitness:.3f} (β={self.weights['fitness']:.2f})")
        print(f"Conformance:  {conformance:.3f} (γ={self.weights['conformance']:.2f})")
        print(f"Diversity:    {diversity:.3f} (δ={self.weights['diversity']:.2f})")
        print("-" * 60)
        print(f"Total Reward: {reward:.3f}")

        if reward >= 0.8:
            quality = "Excellent ✓✓"
        elif reward >= 0.6:
            quality = "Good ✓"
        elif reward >= 0.4:
            quality = "Mediocre ⚠"
        else:
            quality = "Poor ✗"

        print(f"Quality:      {quality}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS FOR EVALUATION
# ─────────────────────────────────────────────────────────

class ProcessMetrics:
    """
    Additional helper metrics for process mining evaluation
    """

    @staticmethod
    def valid_traces(traces: List[List[str]]) -> List[List[str]]:
        """Return only valid traces (start with Start, end with End)"""
        return [t for t in traces if len(t) >= 3 and t[0] == 'Start' and t[-1] == 'End']

    @staticmethod
    def unique_traces(traces: List[List[str]]) -> List[List[str]]:
        """Return unique traces"""
        seen = set()
        unique = []
        for trace in traces:
            trace_tuple = tuple(trace)
            if trace_tuple not in seen:
                seen.add(trace_tuple)
                unique.append(trace)
        return unique

    @staticmethod
    def novel_traces(traces: List[List[str]], training_traces: List[List[str]]) -> List[List[str]]:
        """Return traces not in training set"""
        training_set = set(tuple(t) for t in training_traces)
        return [t for t in traces if tuple(t) not in training_set]

    @staticmethod
    def compute_statistics(traces: List[List[str]]) -> Dict[str, float]:
        """Compute basic statistics"""
        if len(traces) == 0:
            return {}

        lengths = [len(t) for t in traces]

        stats = {
            'total_traces': len(traces),
            'avg_length': np.mean(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'std_length': np.std(lengths),
            'unique_rate': len(ProcessMetrics.unique_traces(traces)) / len(traces),
        }

        return stats


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("ProcessRewardFunction Example")
    print("=" * 60)

    # Create reward function
    training_traces = [
        ['Start', 'Submit', 'Review', 'Approve', 'End'],
        ['Start', 'Submit', 'Review', 'Reject', 'End'],
        ['Start', 'Submit', 'Approve', 'End'],
    ]

    reward_fn = ProcessRewardFunction(training_traces=training_traces)

    # Test traces
    test_traces = [
        ['Start', 'Submit', 'Review', 'Approve', 'End'],  # Perfect
        ['Start', 'Submit', 'Check', 'Approve', 'End'],   # Novel activity
        ['Submit', 'Review', 'End'],                      # Missing Start
        ['Start', 'Review', 'Review', 'End'],             # Duplicate
        [],                                                # Empty
    ]

    print("\nEvaluating test traces:")
    print("-" * 60)

    for i, trace in enumerate(test_traces):
        print(f"\nTrace {i+1}: {' → '.join(trace) if trace else '(empty)'}")
        reward = reward_fn.compute_reward([trace])[0, 0]
        print(f"  Reward: {reward:.3f}")

    print("\n" + "=" * 60)

    # Detailed analysis
    print("\nDetailed Analysis:")
    reward_fn.print_trace_analysis(test_traces[0])

    # Batch evaluation
    print("\nBatch Evaluation:")
    metrics = reward_fn.evaluate_batch(test_traces)
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

    print("\n" + "=" * 60)
    print("ProcessRewardFunction test completed successfully!")
    print("=" * 60)
