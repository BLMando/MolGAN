"""
Conformance Scorer
Alignment-based conformance checking for process traces
"""

import numpy as np
from typing import List, Optional
from datetime import datetime


def log(msg):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} {msg}')


class ConformanceScorer:
    """
    Alignment-based conformance checker for process traces

    Evaluates conformance using:
    - PM4Py A* alignment (if model available)
    - LCS-based similarity (fallback)
    """

    def __init__(self, reference_model=None, expected_pattern: List[str] = None):
        """
        Initialize conformance scorer

        Args:
            reference_model: Petri net tuple (net, im, fm)
            expected_pattern: Expected trace pattern for LCS fallback
        """
        self.reference_model = reference_model
        self.expected_pattern = expected_pattern or ['Start', 'End']

        if reference_model:
            log('[Conformance] Using Petri net for alignment')
        else:
            log('[Conformance] Using LCS-based conformance (no reference model)')

    def score(self, traces: List[List[str]]) -> np.ndarray:
        """
        Compute conformance score for traces

        Args:
            traces: List of traces

        Returns:
            Array of conformance scores ∈ [0,1]
        """
        scores = []

        for trace in traces:
            if len(trace) == 0:
                scores.append(0.0)
                continue

            # Simplified: LCS with expected pattern
            lcs_length = self._longest_common_subsequence(
                trace, self.expected_pattern)
            max_len = max(len(trace), len(self.expected_pattern))
            conformance = lcs_length / max_len if max_len > 0 else 0.0

            # Use reference model if available
            if self.reference_model is not None:
                conformance = self._alignment_conformance(
                    trace, self.reference_model)

            scores.append(conformance)

        return np.array(scores, dtype=np.float32)

    def _alignment_conformance(self, trace: List[str], model) -> float:
        """
        Proper alignment-based conformance using PM4Py

        Args:
            trace: Single trace
            model: Petri net tuple (net, im, fm)

        Returns:
            Conformance score ∈ [0,1]
        """
        try:
            from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
            from pm4py.objects.log.obj import EventLog, Trace, Event

            # Convert trace to PM4Py format
            pm4py_trace = Trace()
            for activity in trace:
                event = Event()
                event['concept:name'] = activity
                pm4py_trace.append(event)

            # Create single-trace event log
            event_log = EventLog()
            event_log.append(pm4py_trace)

            # Compute alignment
            if isinstance(model, tuple) and len(model) == 3:
                net, im, fm = model

                alignments_result = alignments.apply(
                    event_log, net, im, fm,
                    variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR
                )

                if alignments_result and len(alignments_result) > 0:
                    alignment = alignments_result[0]
                    cost = alignment['cost']

                    # Normalize: conformance = 1 / (1 + normalized_cost)
                    trace_length = len(trace)
                    normalized_cost = cost / max(trace_length, 1)
                    conformance = 1.0 / (1.0 + normalized_cost)

                    return float(conformance)
                else:
                    return 0.5
            else:
                return 0.75

        except Exception as e:
            # Fallback: LCS-based conformance
            lcs_length = self._longest_common_subsequence(
                trace, self.expected_pattern)
            max_len = max(len(trace), len(self.expected_pattern))
            return lcs_length / max_len if max_len > 0 else 0.0

    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Compute length of longest common subsequence

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            LCS length
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
