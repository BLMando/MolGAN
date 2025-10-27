"""
Diversity Scorer
Edit distance-based diversity evaluation for process traces
"""

import numpy as np
from typing import List
from datetime import datetime


def log(msg):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} {msg}')


class DiversityScorer:
    """
    Edit distance-based diversity checker for process traces

    Evaluates trace diversity using:
    - Levenshtein edit distance from training set
    - Novelty detection (not in training)
    - Uniqueness analysis (no duplicates)
    """

    def __init__(self, training_traces: List[List[str]]):
        """
        Initialize diversity scorer

        Args:
            training_traces: Training traces for diversity comparison
        """
        self.training_traces = training_traces
        log(f'[Diversity] Training set: {len(training_traces)} traces')

    def score(self, traces: List[List[str]]) -> np.ndarray:
        """
        Compute diversity score for traces

        High score = trace is different from training (good)
        Low score = trace is similar to training (potential duplicate)

        Args:
            traces: List of traces

        Returns:
            Array of diversity scores ∈ [0,1]
        """
        scores = []

        if len(self.training_traces) == 0:
            # No training traces: neutral score
            return np.ones(len(traces), dtype=np.float32) * 0.5

        for trace in traces:
            if len(trace) == 0:
                scores.append(0.0)
                continue

            # Compute minimum edit distance to training
            min_distance = float('inf')

            # Sample training traces for efficiency (max 100)
            sample_size = min(100, len(self.training_traces))
            sample_indices = np.random.choice(
                len(self.training_traces), sample_size, replace=False)

            for idx in sample_indices:
                train_trace = self.training_traces[idx]
                distance = self.edit_distance(trace, train_trace)
                min_distance = min(min_distance, distance)

            # Normalize distance
            max_len = max(len(trace), max(
                len(self.training_traces[i]) for i in sample_indices))
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
            Edit distance (insertions + deletions + substitutions)
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

        return int(dp[m, n])

    @staticmethod
    def novel_traces(traces: List[List[str]], training_set: List[List[str]]) -> List[List[str]]:
        """
        Filter traces that are NOT in training set

        Args:
            traces: Generated traces
            training_set: Training traces

        Returns:
            List of novel traces
        """
        # Convert training traces to set of tuples for fast lookup
        training_tuples = {tuple(trace) for trace in training_set}

        novel = []
        for trace in traces:
            if tuple(trace) not in training_tuples:
                novel.append(trace)

        return novel

    @staticmethod
    def unique_traces(traces: List[List[str]]) -> List[List[str]]:
        """
        Filter unique traces (remove duplicates)

        Args:
            traces: List of traces

        Returns:
            List of unique traces
        """
        seen = set()
        unique = []

        for trace in traces:
            trace_tuple = tuple(trace)
            if trace_tuple not in seen:
                seen.add(trace_tuple)
                unique.append(trace)

        return unique
