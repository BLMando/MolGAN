import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import networkx as nx
import os
from pathlib import Path
import sys

def load_graph_from_txt(filename):
    """
    Load a graph from .txt file in NetworkX format.
    
    Returns:
        G: NetworkX DiGraph
    """
    G = nx.DiGraph()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Node'):
                parts = line.split(': ')
                node_id = int(parts[0].split()[1])
                activity = parts[1]
                G.add_node(node_id, activity=activity)
            elif line.startswith('Edge'):
                parts = line.split()
                source = int(parts[1])
                target = int(parts[2])
                G.add_edge(source, target)
    return G

# Path where your samples were saved
samples_dir = Path("../../output_graph_gan/samples")

if not samples_dir.exists():
    raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

# Get all .txt files
txt_files = list(samples_dir.glob("*.txt"))
if not txt_files:
    raise FileNotFoundError(f"No .txt files found in {samples_dir}")

print(f"Found {len(txt_files)} sample graphs")

num_to_show = 100  # how many to visualize
if len(txt_files) < num_to_show:
    num_to_show = len(txt_files)

indices = np.random.choice(len(txt_files), num_to_show, replace=False)

for i in indices:
    filename = txt_files[i]

    # Load graph
    G = load_graph_from_txt(filename)

    # Create labels from node activities (original nodes)
    labels = {node: data['activity'] for node, data in G.nodes(data=True)}

    # Collapse nodes by activity: build a new graph H where each node is an activity
    activity_to_nodes = {}
    for n, act in labels.items():
        activity_to_nodes.setdefault(act, []).append(n)

    activities = list(activity_to_nodes.keys())

    # Ensure START/END are included only if they actually appear in the graph
    # (some generators may add placeholders; keep visualization faithful to data)
    activities = [a for a in activities if not (str(a).upper() in ("START", "END") and len(activity_to_nodes.get(a, [])) == 0)]
    H = nx.DiGraph()
    for act in activities:
        H.add_node(act)

    # Add edges between activities; count multiplicity in 'weight'
    for u, v in G.edges():
        au = labels.get(u, str(u))
        av = labels.get(v, str(v))
        if H.has_edge(au, av):
            H[au][av]['weight'] += 1
        else:
            H.add_edge(au, av, weight=1)

    # Remove START/END from the collapsed graph if they are isolated (no incident edges)
    # This ensures we don't force-show START/END when they don't participate in any edge
    start_activities = [a for a in activities if str(a).upper() == 'START']
    end_activities = [a for a in activities if str(a).upper() == 'END']

    for s in start_activities:
        if H.has_node(s) and H.degree(s) == 0:
            # remove from H and from activities list
            try:
                H.remove_node(s)
            except Exception:
                pass
            if s in activities:
                activities.remove(s)
            if s in activity_to_nodes:
                del activity_to_nodes[s]

    for e in end_activities:
        if H.has_node(e) and H.degree(e) == 0:
            try:
                H.remove_node(e)
            except Exception:
                pass
            if e in activities:
                activities.remove(e)
            if e in activity_to_nodes:
                del activity_to_nodes[e]

    # --- Interactive step mode: enabled only when --step is passed on the command line
    step_mode = ('--step' in sys.argv)

    # Prepare pair color mapping for edges.
    # Instead of a unique color per edge, assign the same color to edges that share
    # the same source node. This lets us show those edges together and color them
    # consistently in the legend.
    edge_pairs = list(H.edges())
    if edge_pairs:
        # Ordered list of unique sources as they appear in edge_pairs
        unique_sources = []
        for u, v in edge_pairs:
            if u not in unique_sources:
                unique_sources.append(u)

        S = len(unique_sources)
        source_hues = np.linspace(0, 1, max(S, 1), endpoint=False)
        source_colors = [mcolors.hsv_to_rgb((h, 0.75, 1.0)) for h in source_hues]
        source_color_map = {unique_sources[i]: source_colors[i % len(source_colors)] for i in range(len(unique_sources))}

        # Map each edge to the color of its source
        pair_color_map = {e: source_color_map[e[0]] for e in edge_pairs}
    else:
        pair_color_map = {}

    # Create a small curvature map to avoid perfectly overlapping straight edges
    pair_rad_map = {}
    for idx, e in enumerate(edge_pairs):
        # alternate small radii to spread parallel edges
        rad = ((idx % 5) - 2) * 0.06  # values like -0.12, -0.06, 0.0, 0.06, 0.12
        pair_rad_map[e] = rad

    # Build reveal steps grouped by source: reveal all edges from the same source together.
    # The order of steps follows the first occurrence of each source in edge_pairs.
    reveal_steps = []
    if edge_pairs:
        # map source -> list of its edges (preserve order)
        source_to_edges = {s: [] for s in unique_sources}
        for e in edge_pairs:
            source_to_edges[e[0]].append(e)
        for s in unique_sources:
            if source_to_edges[s]:
                reveal_steps.append(source_to_edges[s])

    def compute_reveal_groups(graph, start_nodes):
        """Compute an ordered list of edge-groups to reveal.

        Each group is a list of (src, tgt) edges that are considered parallel
        (multiple outgoing from the same source discovered at the same BFS step).
        """
        groups = []
        from collections import deque

        if start_nodes:
            q = deque(start_nodes)
            visited = set(start_nodes)
        else:
            roots = [n for n, d in graph.in_degree() if d == 0]
            q = deque(roots)
            visited = set(roots)

        # BFS: for each popped node, collect outgoing edges to not-yet-visited targets
        while q:
            u = q.popleft()
            new_vs = [v for v in graph.successors(u) if v not in visited]
            if new_vs:
                group = [(u, v) for v in new_vs]
                groups.append(group)
                for v in new_vs:
                    visited.add(v)
                    q.append(v)

        # Add any remaining edges (between nodes not reached from starts) as singletons
        remaining = []
        for u, v in graph.edges():
            if (u, v) not in [e for g in groups for e in g]:
                remaining.append((u, v))
        for e in remaining:
            groups.append([e])

        return groups

    # Detect start/end activities by label (case-insensitive)
    start_activities = [a for a in activities if str(a).upper() == 'START']
    end_activities = [a for a in activities if str(a).upper() == 'END']

    # Build horizontal layered layout (left-to-right) on collapsed graph H
    layers = {}
    from collections import deque

    # Initialize BFS from start activities if present, otherwise from activities with in-degree 0
    if start_activities:
        queue = deque()
        for s in start_activities:
            layers[s] = 0
            queue.append(s)
    else:
        roots = [n for n, d in H.in_degree() if d == 0]
        if not roots:
            # fallback: pick an arbitrary activity as root
            roots = [list(H.nodes())[0]] if len(H.nodes()) > 0 else []
        queue = deque()
        for r in roots:
            layers[r] = 0
            queue.append(r)

    while queue:
        u = queue.popleft()
        for v in H.successors(u):
            new_layer = layers[u] + 1
            if v not in layers or layers[v] > new_layer:
                layers[v] = new_layer
                queue.append(v)

    # Any activities not reached: place them after the current max layer
    if layers:
        max_layer = max(layers.values())
    else:
        max_layer = 0
    for n in H.nodes():
        if n not in layers:
            max_layer += 1
            layers[n] = max_layer

    # If end activity(s) exist, force them to the last layer
    if end_activities:
        last_layer = max(layers.values())
        for e in end_activities:
            layers[e] = last_layer

    # Group activities by layer and assign positions
    layer_to_nodes = {}
    for n, l in layers.items():
        layer_to_nodes.setdefault(l, []).append(n)

    pos = {}
    n_layers = max(layer_to_nodes.keys()) + 1 if layer_to_nodes else 1
    for l, nodes_in_layer in layer_to_nodes.items():
        x = l
        k = len(nodes_in_layer)
        if k == 1:
            ys = [0.5]
        else:
            ys = np.linspace(0.1, 0.9, k)
        for idx, n in enumerate(nodes_in_layer):
            pos[n] = np.array([x, ys[idx]])

    # Normalize positions so graph spans horizontally nicely
    xs = np.array([p[0] for p in pos.values()]) if pos else np.array([0])
    if xs.ptp() == 0:
        scale_x = 1.0
    else:
        scale_x = 10.0 / xs.ptp()
    for n in pos:
        pos[n][0] = pos[n][0] * scale_x

    # Generate bright colors for activities (avoid dark shades)
    M = max(len(activities), 1)
    cmap_hues = np.linspace(0, 1, M, endpoint=False)
    bright_colors = [mcolors.hsv_to_rgb((h, 0.85, 1.0)) for h in cmap_hues]
    activity_color_map = {activities[i]: bright_colors[i % len(bright_colors)] for i in range(len(activities))}

    # Create figure with a 2-column GridSpec: main graph (left) + legend (right).
    # Using GridSpec + subplots avoids tight_layout incompatibility with manually
    # added axes and ensures the legend area doesn't overlap the graph.
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.72, 0.28], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis('off')
    # Figure-level title (placed at the top of the figure)
    fig.suptitle(f"{filename.name}", fontsize=14, fontweight='bold', y=0.98)

    # Draw nodes (initially without edges)
    node_list = activities
    node_colors = [activity_color_map[a] for a in node_list]
    nx.draw_networkx_nodes(H, pos, nodelist=node_list, node_color=node_colors, node_size=900, ax=ax)

    # Prepare interactive reveal groups (fallback BFS groups) but prefer reveal_steps
    reveal_groups = compute_reveal_groups(H, start_activities)
    revealed_edges = set()

    def redraw(all_highlighted=None):
        ax.clear()
        # redraw nodes
        nx.draw_networkx_nodes(H, pos, nodelist=node_list, node_color=node_colors, node_size=900, ax=ax)
        nx.draw_networkx_labels(H, pos, labels={a: a for a in activities}, font_size=10, font_color='black', ax=ax)

        # draw unrevealed edges in light gray, with per-edge curvature to avoid overlap
        unrevealed = [e for e in H.edges() if e not in revealed_edges]
        for e in unrevealed:
            rad = pair_rad_map.get(e, 0.0)
            nx.draw_networkx_edges(H, pos, edgelist=[e], edge_color='lightgray', arrows=True, arrowsize=20, ax=ax, connectionstyle=f'arc3,rad={rad}')

        # draw revealed edges with pair-specific color and thicker linewidth
        if revealed_edges:
            for e in revealed_edges:
                color = pair_color_map.get(e, activity_color_map.get(e[0], 'k'))
                rad = pair_rad_map.get(e, 0.0)
                nx.draw_networkx_edges(H, pos, edgelist=[e], edge_color=[color], arrows=True, arrowsize=20, width=3.0, ax=ax, connectionstyle=f'arc3,rad={rad}')

    ax.set_axis_off()
    # redraw; GridSpec handles spacing so no tight_layout rectangle is needed here
    plt.draw()

    # Show initial graph with nodes only
    redraw()
    plt.show(block=False)

    # In step mode, reveal edges grouped by source (reveal_steps) so edges from the
    # same source are shown together and share the same color.
    if step_mode and reveal_steps:
        total_steps = len(reveal_steps)
        print(f"Interactive reveal: {total_steps} steps. Press Enter to reveal next (q + Enter to quit).")
        for group in reveal_steps:
            user_in = input()
            if user_in.lower().strip() == 'q':
                break
            for e in group:
                revealed_edges.add(e)
            redraw()
            plt.pause(0.01)
        # final pause to inspect
        print("Reveal finished. Close the plot window or press Enter to continue.")
        try:
            input()
        except Exception:
            pass
    else:
        # If not step_mode or no groups, draw all edges with pair colors and non-overlapping curvature
        for e in H.edges():
            color = pair_color_map.get(e, activity_color_map.get(e[0], 'k'))
            rad = pair_rad_map.get(e, 0.0)
            nx.draw_networkx_edges(H, pos, edgelist=[e], edge_color=[color], arrows=True, arrowsize=20, ax=ax, connectionstyle=f'arc3,rad={rad}')

    # Draw labels on activity-nodes with readable font
    nx.draw_networkx_labels(H, pos, labels={a: a for a in activities}, font_size=10, font_color='black', ax=ax)

    # Title: filename on top
    # title already set in redraw

    # Create a legend for edges on the right showing pair colors
    # Use the previously created pair_color_map so legend colors match interactive drawing
    if edge_pairs:
        handles = [Line2D([0], [0], color=pair_color_map[p], lw=3) for p in edge_pairs]
        labels = [f"{u} -> {v}" for u, v in edge_pairs]
        # Draw legend inside the dedicated legend axis so it's always visible
        legend = legend_ax.legend(handles=handles, labels=labels, loc='center', frameon=True)
        # Color only the legend handles (markers/lines); keep text black for readability
        for lh, pair in zip(handles, edge_pairs):
            try:
                lh.set_color(pair_color_map[pair])
                lh.set_linewidth(3.0)
            except Exception:
                pass
        for txt in legend.get_texts():
            txt.set_color('black')
            txt.set_fontsize(9)
        # give legend area a subtle background to improve contrast
        legend_ax.set_facecolor('#ffffff')
    else:
        # fallback: if no edges, show activity legend (node colors)
        handles = [Patch(facecolor=activity_color_map[a], edgecolor='none', label=str(a)) for a in activities]
        legend = legend_ax.legend(handles=handles, loc='center', frameon=True)
        for txt in legend.get_texts():
            txt.set_color('black')
            txt.set_fontsize(9)
        legend_ax.set_facecolor('#ffffff')

    # Remove axes for cleaner look
    ax.set_axis_off()

    # GridSpec already manages spacing between axes; avoid calling tight_layout
    # because it raises a UserWarning with mixed axes (legend artists).
    plt.draw()
    plt.show()