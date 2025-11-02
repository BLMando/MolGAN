from typing import List, Dict
from metrics import DiversityScorer
import numpy as np
class ProcessMetrics:
    """
    Additional helper metrics for process mining evaluation
    """

    @staticmethod
    def valid_traces(traces: List[List[str]], start_activity='Start', end_activity='End') -> List[List[str]]:
        """
        Return only valid traces with strict position constraints

        Args:
            traces: List of traces
            start_activity: START activity name
            end_activity: END activity name

        Returns:
            List of valid traces
        """
        return [
            t for t in traces
            if len(t) >= 3
            and t[0] == start_activity
            and t[-1] == end_activity
            and t.count(start_activity) == 1
            and t.count(end_activity) == 1
            and start_activity not in t[1:]
            #and end_activity not in t[:-1]
        ]

    @staticmethod
    def unique_traces(traces: List[List[str]]) -> List[List[str]]:
        """Return unique traces (remove duplicates)"""
        return DiversityScorer.unique_traces(traces)

    @staticmethod
    def novel_traces(traces: List[List[str]], training_traces: List[List[str]]) -> List[List[str]]:
        """Return traces not in training set"""
        return DiversityScorer.novel_traces(traces, training_traces)

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
