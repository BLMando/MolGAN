"""
Fitness Scorer
Token replay fitness evaluation for process traces
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from datetime import datetime


def log(msg):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} {msg}')


class FitnessScorer:
    """
    Token replay fitness checker for process traces

    Evaluates trace fitness against reference model using:
    - PM4Py token replay (if model available)
    - Simplified heuristics (fallback)
    """

    def __init__(self, reference_model=None, core_activities: Set[str] = None,
                 start_activity: str = 'Start', end_activity: str = 'End'):
        """
        Initialize fitness scorer

        Args:
            reference_model: Petri net tuple (net, im, fm) from PM4Py
            core_activities: Core activities set
            start_activity: START activity name
            end_activity: END activity name
        """
        self.reference_model = reference_model
        self.core_activities = core_activities or {
            start_activity, end_activity}
        self.start_activity = start_activity
        self.end_activity = end_activity

        if reference_model:
            log('[Fitness] Using Petri net for token replay')
        else:
            log('[Fitness] Using simplified heuristics (no reference model)')

    @staticmethod
    def load_petri_net(pnml_path: str) -> Optional[Tuple]:
        """
        Load Petri net model from PNML file

        Args:
            pnml_path: Path to PNML file

        Returns:
            Tuple (net, initial_marking, final_marking) or None
        """
        try:
            import pm4py

            net, initial_marking, final_marking = pm4py.read_pnml(pnml_path)

            log(
                f'[Fitness] Petri net loaded: {len(net.places)} places, {len(net.transitions)} transitions')

            return (net, initial_marking, final_marking)

        except Exception as e:
            log(f'[Fitness] Failed to load Petri net: {e}')
            return None

    def score(self, traces: List[List[str]]) -> np.ndarray:
        """
        Compute fitness score for traces

        Args:
            traces: List of traces

        Returns:
            Array of fitness scores ∈ [0,1]
        """
        scores = []

        expected_activities = {self.start_activity, self.end_activity}

        for trace in traces:
            if len(trace) == 0:
                scores.append(0.0)
                continue

            fitness = 1.0

            # Check presence of START/END
            trace_set = set(trace)
            if not expected_activities.issubset(trace_set):
                fitness *= 0.5

            # Bonus for core activities
            if self.core_activities:
                core_present = len(
                    trace_set.intersection(self.core_activities))
                core_bonus = core_present / len(self.core_activities)
                fitness *= (0.5 + 0.5 * core_bonus)

            # Check logical ordering (START before END)
            try:
                start_idx = trace.index(self.start_activity)
                end_idx = trace.index(self.end_activity)
                if start_idx >= end_idx:
                    fitness *= 0.3
            except ValueError:
                pass

            # Use reference model if available
            if self.reference_model is not None:
                fitness = self._token_replay_fitness(
                    trace, self.reference_model)

            scores.append(fitness)

        return np.array(scores, dtype=np.float32)

    def _token_replay_fitness(self, trace: List[str], model) -> float:
        """
        Proper token replay fitness using PM4Py

        Args:
            trace: Single trace
            model: Petri net tuple (net, im, fm)

        Returns:
            Fitness score ∈ [0,1]
        """
        try:
            from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
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

            # Compute token replay
            if isinstance(model, tuple) and len(model) == 3:
                net, im, fm = model
                fitness_result = replay_fitness.apply(
                    event_log, net, im, fm,
                    variant=replay_fitness.Variants.TOKEN_BASED
                )

                # Extract average fitness
                fitness = fitness_result['average_trace_fitness']
                return float(fitness)
            else:
                return 0.8

        except Exception as e:
            # Fallback to simplified fitness
            expected = self.core_activities if self.core_activities else {
                self.start_activity, self.end_activity}
            trace_set = set(trace)
            overlap = len(expected.intersection(trace_set))
            return overlap / len(expected) if expected else 0.5
