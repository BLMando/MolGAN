"""
XES Exporter
Export process traces to XES (eXtensible Event Stream) format
"""

import os
from datetime import datetime
from typing import List, Dict


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {msg}')


class XESExporter:
    """
    Export process traces to XES format using PM4Py

    Handles:
    - Trace to PM4Py EventLog conversion
    - Case ID generation (original vs augmented)
    - XES file writing
    """

    @staticmethod
    def save_traces(traces: List[List[str]], output_file: str, case_id_prefix: str = 'Case'):
        """
        Save traces to XES file

        Args:
            traces: List of traces (activity name lists)
            output_file: Output XES file path
            case_id_prefix: Prefix for case IDs (default: 'Case')
        """
        log(f"Saving {len(traces)} traces to XES file: {output_file}")

        import pm4py

        # Create event log
        event_log = XESExporter._create_event_log(
            traces, case_id_prefix, trace_type='trace')

        # Write XES file
        pm4py.write_xes(event_log, output_file)
        log(f"✓ Saved {len(traces)} traces to {output_file}")

    @staticmethod
    def save_augmented(original_traces: List[List[str]], synthetic_traces: List[List[str]],
                       output_file: str):
        """
        Save augmented dataset (original + synthetic) to XES file

        Args:
            original_traces: List of original traces
            synthetic_traces: List of generated synthetic traces
            output_file: Output XES file path
        """
        log(f"Saving augmented dataset to XES file: {output_file}")
        log(f"  Original traces: {len(original_traces)}")
        log(f"  Synthetic traces: {len(synthetic_traces)}")

        import pm4py
        from pm4py.objects.log.obj import EventLog

        # Create event log with both original and synthetic traces
        event_log = EventLog()

        # Add original traces
        original_log = XESExporter._create_event_log(
            original_traces, 'Original_Case', trace_type='original'
        )
        event_log.extend(original_log)

        # Add synthetic traces
        synthetic_log = XESExporter._create_event_log(
            synthetic_traces, 'Augmented_Case', trace_type='augmented'
        )
        event_log.extend(synthetic_log)

        # Write XES file
        pm4py.write_xes(event_log, output_file)
        log(f"✓ Saved {len(original_traces) + len(synthetic_traces)} traces to {output_file}")
        log(f"  - Original traces: {len(original_traces)}")
        log(f"  - Augmented traces: {len(synthetic_traces)}")

    @staticmethod
    def _create_event_log(traces: List[List[str]], case_id_prefix: str, trace_type: str):
        """
        Create PM4Py EventLog from traces

        Args:
            traces: List of traces (activity name lists)
            case_id_prefix: Prefix for case IDs
            trace_type: Type label for traces ('original', 'augmented', or 'trace')

        Returns:
            PM4Py EventLog object
        """
        from pm4py.objects.log.obj import EventLog, Trace, Event

        event_log = EventLog()

        for i, trace in enumerate(traces):
            # Create case ID
            case_id = f'{case_id_prefix}_{i+1}'

            # Create PM4Py trace
            pm4py_trace = Trace()
            pm4py_trace.attributes['concept:name'] = case_id
            pm4py_trace.attributes['trace_type'] = trace_type

            # Add events to trace
            for j, activity in enumerate(trace):
                event = Event()
                event['concept:name'] = activity
                event['time:timestamp'] = datetime.now()
                event['position'] = j

                pm4py_trace.append(event)

            event_log.append(pm4py_trace)

        return event_log
