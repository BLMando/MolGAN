"""
Validity Scorer
Syntactic correctness validation for process traces
"""

import numpy as np
from typing import List, Set
from datetime import datetime


def log(msg):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} {msg}')


class ValidityScorer:
    """
    Syntactic validity checker for process traces

    Validates traces based on:
    - START/END activity presence and position
    - Trace length constraints
    - No consecutive duplicates
    - Core activities presence
    """

    def __init__(self, training_traces: List[List[str]], start_activity: str = None,
                 end_activity: str = None, core_activities: Set[str] = None):
        """
        Initialize validity scorer

        Args:
            training_traces: Training traces for auto-detection
            start_activity: START activity name (auto-detected if None)
            end_activity: END activity name (auto-detected if None)
            core_activities: Core activities set (auto-detected if None)
        """
        self.training_traces = training_traces

        # Auto-detect or use provided
        self.start_activity = start_activity or self._detect_start_activity()
        self.end_activity = end_activity or self._detect_end_activity()
        self.core_activities = core_activities or self._detect_core_activities()

        log(f'[Validity] START: {self.start_activity}, END: {self.end_activity}')
        log(f'[Validity] Core activities: {len(self.core_activities)}')

    def _detect_start_activity(self) -> str:
        """
        Auto-detect START activity (most common first activity)

        Returns:
            Name of START activity (default: 'Start')
        """
        if not self.training_traces:
            return 'Start'

        # Count first activities
        first_activities = {}
        for trace in self.training_traces:
            if len(trace) > 0:
                first_act = trace[0]
                first_activities[first_act] = first_activities.get(
                    first_act, 0) + 1

        if not first_activities:
            return 'Start'

        return max(first_activities, key=first_activities.get)

    def _detect_end_activity(self) -> str:
        """
        Auto-detect END activity (most common last activity)

        Returns:
            Name of END activity (default: 'End')
        """
        if not self.training_traces:
            return 'End'

        # Count last activities
        last_activities = {}
        for trace in self.training_traces:
            if len(trace) > 0:
                last_act = trace[-1]
                last_activities[last_act] = last_activities.get(
                    last_act, 0) + 1

        if not last_activities:
            return 'End'

        return max(last_activities, key=last_activities.get)

    def _detect_core_activities(self, min_frequency: float = 0.20) -> Set[str]:
        """
        Auto-detect core activities (appear in >= 20% of traces)

        Args:
            min_frequency: Minimum frequency threshold (default: 0.20)

        Returns:
            Set of core activity names
        """
        if not self.training_traces:
            return set()

        # Count activity occurrences
        activity_count = {}
        total_traces = len(self.training_traces)

        for trace in self.training_traces:
            unique_activities = set(trace)
            for activity in unique_activities:
                activity_count[activity] = activity_count.get(activity, 0) + 1

        # Filter by frequency
        core_activities = {
            activity for activity, count in activity_count.items()
            if count / total_traces >= min_frequency
        }

        # Always include START and END
        core_activities.add(self.start_activity)
        core_activities.add(self.end_activity)

        return core_activities

    def score(self, traces: List[List[str]]) -> np.ndarray:
        """
        Compute validity score for traces

        Checks:
        - Non-empty trace
        - Starts with START (exactly once, only at position 0)
        - Ends with END (exactly once, only at last position)
        - Reasonable length (3-20 activities)
        - No consecutive duplicates

        Args:
            traces: List of traces

        Returns:
            Array of validity scores ∈ [0,1]
        """
        scores = []

        for trace in traces:
            score = 1.0

            # Rule 1: Non-empty
            if len(trace) == 0:
                scores.append(0.0)
                continue

            # Rule 2: START at position 0 (HARD CONSTRAINT)
            if trace[0] != self.start_activity:
                scores.append(0.0)
                continue

            # Rule 2b: START only at position 0
            if self.start_activity in trace[1:]:
                scores.append(0.0)
                continue

            # Rule 2c: START exactly once
            if trace.count(self.start_activity) != 1:
                scores.append(0.0)
                continue

            # Rule 3: END at last position
            if trace[-1] != self.end_activity:
                score *= 0.5

            # Rule 3b: END exactly once
            if trace.count(self.end_activity) != 1:
                scores.append(0.0)
                continue

            # Rule 3c: END only at last position (HARD CONSTRAINT)
            if self.end_activity in trace[:-1]:
                scores.append(0.0)
                continue

            # Rule 4: Reasonable length (3-20)
            if not (3 <= len(trace) <= 20):
                score *= 0.7

            # Rule 5: No consecutive duplicates
            for i in range(len(trace) - 1):
                if trace[i] == trace[i+1]:
                    score *= 0.8
                    break

            scores.append(score)

        return np.array(scores, dtype=np.float32)
