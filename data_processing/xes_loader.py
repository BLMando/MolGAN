"""
XES Loader
Load and parse XES event log files using PM4Py
"""

import pm4py
from datetime import datetime
from typing import List


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} {msg}')


class XESLoader:
    """
    Load XES event log files

    Handles:
    - XES file loading via PM4Py
    - DataFrame to EventLog conversion
    - Trace extraction (activity sequences)
    """

    @staticmethod
    def load_xes(xes_path: str):
        """
        Load event log from XES file

        Args:
            xes_path: Path to XES file

        Returns:
            tuple: (traces, event_log) where:
                - traces: List of activity sequences [[act1, act2, ...], ...]
                - event_log: PM4Py EventLog object for pattern discovery

        Raises:
            FileNotFoundError: If XES file doesn't exist
            Exception: If loading fails
        """
        log(f'Loading event log from {xes_path}...')

        # Load XES file using pm4py
        dataframe = pm4py.read_xes(xes_path)

        # PM4Py returns a DataFrame - convert to EventLog format
        from pm4py.objects.log.util import dataframe_utils
        from pm4py.objects.conversion.log import converter as log_converter

        # Convert DataFrame to EventLog
        event_log = log_converter.apply(dataframe)

        # Extract traces from event log
        traces = []
        for trace in event_log:
            activity_sequence = []
            for event in trace:
                # Events are dictionaries with 'concept:name' key
                activity_name = event['concept:name']
                activity_sequence.append(activity_name)

            if activity_sequence:  # Only add non-empty traces
                traces.append(activity_sequence)

        log(f'Loaded {len(traces)} traces from XES')

        return traces, event_log
