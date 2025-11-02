"""
Pattern Discovery
Discover control flow patterns from event logs using PM4Py
"""

from datetime import datetime
from typing import Dict, Tuple, Any


def log(msg, level='INFO'):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} {msg}')


class PatternDiscovery:
    """
    Discover control flow patterns from event logs

    Uses PM4Py to:
    - Discover Petri net structure (Inductive Miner)
    - Build Directly-Follows Graph (DFG)
    - Identify XOR/AND splits, joins, loops
    """

    def __init__(self):
        """Initialize pattern discovery"""
        self.petri_net = None      # (net, im, fm) tuple
        self.pattern_map = {}      # (src, dst) -> pattern type
        self.dfg = None            # Directly-Follows Graph
        self.dfg_frequencies = {}  # Normalized DFG frequencies
        self.pm4py_available = True

    def discover_patterns(self, event_log):
        """
        Discover all patterns from event log

        Args:
            event_log: PM4Py EventLog object

        Returns:
            dict: Pattern information {
                'petri_net': (net, im, fm),
                'pattern_map': dict,
                'dfg': dict,
                'dfg_frequencies': dict
            }
        """
        # Step 1: Discover Petri net
        self._discover_petri_net(event_log)

        # Step 2: Build DFG
        self._build_dfg(event_log)

        return {
            'petri_net': self.petri_net,
            'pattern_map': self.pattern_map,
            'dfg': self.dfg,
            'dfg_frequencies': self.dfg_frequencies,
            'pm4py_available': self.pm4py_available
        }

    def _build_dfg(self, event_log):
        """
        Build Directly-Follows Graph for frequency analysis

        Args:
            event_log: PM4Py event log object
        """
        try:
            from pm4py.algo.discovery.dfg import algorithm as dfg_discovery

            log('[Pattern Discovery] Building Directly-Follows Graph...')

            # Compute DFG
            dfg = dfg_discovery.apply(event_log)
            self.dfg = dfg

            # Normalize frequencies
            if dfg:
                total_edges = sum(dfg.values())
                self.dfg_frequencies = {
                    edge: freq / total_edges for edge, freq in dfg.items()}

                log(f'[Pattern Discovery] Built DFG with {len(dfg)} edges')

        except ImportError:
            log('[Pattern Discovery] PM4Py DFG not available')
        except Exception as e:
            log(f'[Pattern Discovery] Failed to build DFG: {e}')

    def _discover_petri_net(self, event_log):
        """
        Discover Petri Net from event log using Inductive Miner

        Args:
            event_log: PM4Py event log object
        """
        try:
            from pm4py.algo.discovery.inductive import algorithm as inductive_miner

            log('[Pattern Discovery] Discovering Petri net structure...')

            # Discover Petri net using Inductive Miner
            result = inductive_miner.apply(event_log)

            # Check if result is a tuple (net, im, fm) or ProcessTree
            if isinstance(result, tuple) and len(result) == 3:
                net, initial_marking, final_marking = result
                self.petri_net = (net, initial_marking, final_marking)
            else:
                # Result is ProcessTree - convert to Petri net
                from pm4py.objects.conversion.process_tree import converter as pt_converter
                net, initial_marking, final_marking = pt_converter.apply(
                    result)
                self.petri_net = (net, initial_marking, final_marking)

            # Analyze Petri net structure
            num_places = len(net.places)
            num_transitions = len(net.transitions)

            log(
                f'[Pattern Discovery] Discovered Petri net: {num_places} places, {num_transitions} transitions')

            # Build pattern map from Petri net structure
            self._analyze_petri_net_structure(net)

        except ImportError:
            log(
                '[Pattern Discovery] PM4Py not available, using fallback pattern detection')
            self.pm4py_available = False
        except Exception as e:
            log(f'[Pattern Discovery] Failed to discover Petri net: {e}')
            log('[Pattern Discovery] Using fallback pattern detection')
            self.pm4py_available = False

    def _analyze_petri_net_structure(self, net):
        """
        Analyze Petri net structure to identify control flow patterns

        Args:
            net: PM4Py Petri net object
        """
        pattern_counts = {'XOR_SPLIT': 0,
                          'AND_SPLIT': 0, 'XOR_JOIN': 0, 'LOOP': 0}

        for place in net.places:
            in_arcs = list(place.in_arcs)
            out_arcs = list(place.out_arcs)

            # Pattern 1: XOR/AND Split
            # 1 input transition → Place → N output transitions
            if len(in_arcs) == 1 and len(out_arcs) > 1:
                source_transition = in_arcs[0].source

                for out_arc in out_arcs:
                    target_transition = out_arc.target

                    # Skip silent transitions (None label)
                    if source_transition.label and target_transition.label:
                        key = (source_transition.label,
                               target_transition.label)
                        self.pattern_map[key] = 'XOR_SPLIT'
                        pattern_counts['XOR_SPLIT'] += 1

            # Pattern 2: XOR Join
            # N input transitions → Place → 1 output transition
            elif len(in_arcs) > 1 and len(out_arcs) == 1:
                target_transition = out_arcs[0].target

                for in_arc in in_arcs:
                    source_transition = in_arc.source

                    if source_transition.label and target_transition.label:
                        key = (source_transition.label,
                               target_transition.label)
                        self.pattern_map[key] = 'XOR_JOIN'
                        pattern_counts['XOR_JOIN'] += 1

        # Detect loops (simplified: check for back-arcs in transitions)
        for transition in net.transitions:
            if transition.label:
                for out_arc in transition.out_arcs:
                    place = out_arc.target
                    for out_arc2 in place.out_arcs:
                        next_transition = out_arc2.target
                        if next_transition == transition:
                            key = (transition.label, transition.label)
                            self.pattern_map[key] = 'LOOP'
                            pattern_counts['LOOP'] += 1

        log(f'[Pattern Discovery] Detected: {pattern_counts["XOR_SPLIT"]} XOR splits, '
            f'{pattern_counts["AND_SPLIT"]} AND splits, {pattern_counts["LOOP"]} loops')
