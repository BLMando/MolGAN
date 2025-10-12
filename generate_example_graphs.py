#!/usr/bin/env python3
"""
Generate Example Graph Files

Creates sample graph files in different formats for testing ProcessGAN graph input:
- GraphML (NetworkX format)
- PNML (Petri net format) - requires pm4py
- BPMN (BPMN 2.0 format) - requires pm4py

These files represent a simple Purchase Order process.
"""

import networkx as nx
from pathlib import Path

# Create data directory if needed
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)


def create_sample_process_graph():
    """
    Create a simple Purchase Order process as NetworkX DiGraph

    Process flow:
    Start → Submit → Review → Approve → Send → End
                ↓ (reject)
            Reject → Rework → Submit (loop)
    """
    G = nx.DiGraph()

    # Add nodes with labels
    nodes = [
        ('n0', {'label': 'Start', 'activity': 'Start'}),
        ('n1', {'label': 'Submit', 'activity': 'Submit'}),
        ('n2', {'label': 'Review', 'activity': 'Review'}),
        ('n3', {'label': 'Approve', 'activity': 'Approve'}),
        ('n4', {'label': 'Reject', 'activity': 'Reject'}),
        ('n5', {'label': 'Rework', 'activity': 'Rework'}),
        ('n6', {'label': 'Send', 'activity': 'Send'}),
        ('n7', {'label': 'End', 'activity': 'End'}),
    ]

    G.add_nodes_from(nodes)

    # Add edges with flow types
    edges = [
        ('n0', 'n1', {'type': 'SEQUENCE', 'flow_type': 'SEQUENCE'}),  # Start → Submit
        ('n1', 'n2', {'type': 'SEQUENCE', 'flow_type': 'SEQUENCE'}),  # Submit → Review
        ('n2', 'n3', {'type': 'CHOICE', 'flow_type': 'CHOICE'}),      # Review → Approve (choice)
        ('n2', 'n4', {'type': 'CHOICE', 'flow_type': 'CHOICE'}),      # Review → Reject (choice)
        ('n3', 'n6', {'type': 'SEQUENCE', 'flow_type': 'SEQUENCE'}),  # Approve → Send
        ('n4', 'n5', {'type': 'SEQUENCE', 'flow_type': 'SEQUENCE'}),  # Reject → Rework
        ('n5', 'n1', {'type': 'LOOP', 'flow_type': 'LOOP'}),          # Rework → Submit (loop)
        ('n6', 'n7', {'type': 'SEQUENCE', 'flow_type': 'SEQUENCE'}),  # Send → End
    ]

    G.add_edges_from(edges)

    return G


def save_graphml(G, filename):
    """Save graph as GraphML"""
    print(f"[INFO] Saving GraphML to {filename}...")
    nx.write_graphml(G, filename)
    print(f"[SUCCESS] GraphML saved: {filename}")


def save_as_pnml(G, filename):
    """
    Convert NetworkX graph to Petri net and save as PNML

    Note: This is a simplified conversion. For production use,
    consider using proper process discovery algorithms.
    """
    try:
        import pm4py
        from pm4py.objects.petri_net.obj import PetriNet, Marking
        from pm4py.objects.petri_net.utils import petri_utils

        print(f"[INFO] Converting to Petri net and saving to {filename}...")

        # Create Petri net
        net = PetriNet("Purchase Order Process")

        # Create transitions from nodes
        transitions = {}
        for node, attrs in G.nodes(data=True):
            label = attrs.get('label', attrs.get('activity', str(node)))
            trans = PetriNet.Transition(node, label)
            net.transitions.add(trans)
            transitions[node] = trans

        # Create places between transitions
        place_counter = 0
        for source, target in G.edges():
            place_name = f"p{place_counter}"
            place = PetriNet.Place(place_name)
            net.places.add(place)

            # Add arcs
            petri_utils.add_arc_from_to(transitions[source], place, net)
            petri_utils.add_arc_from_to(place, transitions[target], net)

            place_counter += 1

        # Create initial and final markings
        # Initial marking: place before Start transition
        start_place = PetriNet.Place("start_place")
        net.places.add(start_place)
        start_trans = [t for t in net.transitions if 'Start' in t.label][0]
        petri_utils.add_arc_from_to(start_place, start_trans, net)

        initial_marking = Marking()
        initial_marking[start_place] = 1

        # Final marking: place after End transition
        end_place = PetriNet.Place("end_place")
        net.places.add(end_place)
        end_trans = [t for t in net.transitions if 'End' in t.label][0]
        petri_utils.add_arc_from_to(end_trans, end_place, net)

        final_marking = Marking()
        final_marking[end_place] = 1

        # Save as PNML
        pm4py.write_pnml(net, initial_marking, final_marking, filename)
        print(f"[SUCCESS] PNML saved: {filename}")

    except ImportError:
        print("[WARNING] pm4py not installed. Skipping PNML generation.")
        print("          Install with: pip install pm4py")
    except Exception as e:
        print(f"[ERROR] Failed to generate PNML: {e}")


def save_as_bpmn(G, filename):
    """
    Convert NetworkX graph to BPMN and save as XML

    Note: This is a simplified conversion.
    """
    try:
        import pm4py
        from pm4py.objects.bpmn.obj import BPMN

        print(f"[INFO] Converting to BPMN and saving to {filename}...")

        # Create BPMN object
        bpmn_graph = BPMN()

        # Add tasks from nodes
        bpmn_nodes = {}
        for node, attrs in G.nodes(data=True):
            label = attrs.get('label', attrs.get('activity', str(node)))

            # Create appropriate BPMN element
            if 'Start' in label:
                bpmn_node = BPMN.StartEvent(name=label)
            elif 'End' in label:
                bpmn_node = BPMN.EndEvent(name=label)
            else:
                bpmn_node = BPMN.Task(name=label)

            bpmn_graph.add_node(bpmn_node)
            bpmn_nodes[node] = bpmn_node

        # Add flows from edges
        for source, target, attrs in G.edges(data=True):
            flow = BPMN.Flow(bpmn_nodes[source], bpmn_nodes[target])
            bpmn_graph.add_flow(flow)

        # Save as BPMN XML
        pm4py.write_bpmn(bpmn_graph, filename)
        print(f"[SUCCESS] BPMN saved: {filename}")

    except ImportError:
        print("[WARNING] pm4py not installed. Skipping BPMN generation.")
        print("          Install with: pip install pm4py")
    except Exception as e:
        print(f"[ERROR] Failed to generate BPMN: {e}")


def main():
    print("="*70)
    print("  ProcessGAN Example Graph Generator")
    print("="*70)
    print()

    # Create sample graph
    print("[INFO] Creating sample Purchase Order process graph...")
    G = create_sample_process_graph()
    print(f"[INFO] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print()

    # Save in different formats
    save_graphml(G, data_dir / 'sample_process.graphml')
    print()

    save_as_pnml(G, data_dir / 'sample_process.pnml')
    print()

    save_as_bpmn(G, data_dir / 'sample_process.bpmn')
    print()

    # Summary
    print("="*70)
    print("  Summary")
    print("="*70)
    print("\nGenerated files:")
    for fmt, filename in [
        ('GraphML', 'sample_process.graphml'),
        ('PNML', 'sample_process.pnml'),
        ('BPMN', 'sample_process.bpmn')
    ]:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"  ✓ {fmt:10s} {filepath}")
        else:
            print(f"  ✗ {fmt:10s} {filepath} (skipped)")

    print("\nProcess structure:")
    print("  Start → Submit → Review → Approve → Send → End")
    print("                      ↓")
    print("                   Reject → Rework → (back to Submit)")
    print()
    print("Usage:")
    print("  python -c \"from utils.process_dataset import ProcessDataset; ds = ProcessDataset(); ds.load_from_graphml('data/sample_process.graphml')\"")
    print()


if __name__ == '__main__':
    main()
