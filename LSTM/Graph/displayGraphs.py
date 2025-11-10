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
        edges_ordered: List of (source, target) tuples in original file order
    """
    G = nx.DiGraph()
    edges_ordered = []  # Preserve original edge order
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
                edges_ordered.append((source, target))  # Preserve order
    return G, edges_ordered

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

    # Load graph and preserve original edge order
    G, edges_ordered = load_graph_from_txt(filename)

    # Create labels from node activities (original nodes)
    labels = {node: data['activity'] for node, data in G.nodes(data=True)}

    # Use distinct nodes (default and only behavior)
    activities = list(G.nodes())
    activity_to_nodes = {n: [n] for n in G.nodes()}

    # Ensure START/END are included only if they actually appear in the graph
    # (some generators may add placeholders; keep visualization faithful to data)
    activities = [a for a in activities if not (str(labels.get(a, a)).upper() in ("START", "END") and len(activity_to_nodes.get(a, [])) == 0)]
    H = nx.DiGraph()
    for act in activities:
        H.add_node(act)

    # Convert edges to activity pairs, preserving ALL duplicates and original order
    edge_pairs = []
    for u, v in edges_ordered:
        au = u
        av = v
        edge_pairs.append((au, av))
        # Add to H for layout purposes (duplicates will be handled in visualization)
        if not H.has_edge(au, av):
            H.add_edge(au, av)

    # Remove START/END from the graph if they are isolated (no incident edges)
    # This ensures we don't force-show START/END when they don't participate in any edge
    start_activities = [a for a in activities if str(labels.get(a, a)).upper() == 'START']
    end_activities = [a for a in activities if str(labels.get(a, a)).upper() == 'END']

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

    # Group consecutive edges by same source node (parallelism detection)
    # Edges with same consecutive source get same color
    edge_groups = []  # List of (start_idx, end_idx, source_node)
    if edge_pairs:
        current_source = edge_pairs[0][0]
        group_start = 0
        
        for idx in range(1, len(edge_pairs)):
            if edge_pairs[idx][0] != current_source:
                # End of current group
                edge_groups.append((group_start, idx - 1, current_source))
                current_source = edge_pairs[idx][0]
                group_start = idx
        
        # Add last group
        edge_groups.append((group_start, len(edge_pairs) - 1, current_source))
    
    # Assign same color to edges in the same group (consecutive edges from same source)
    num_groups = len(edge_groups)
    group_hues = np.linspace(0, 1, max(num_groups, 1), endpoint=False)
    group_colors = [mcolors.hsv_to_rgb((h, 0.75, 1.0)) for h in group_hues]
    
    pair_color_map = {}
    for group_idx, (start_idx, end_idx, source) in enumerate(edge_groups):
        color = group_colors[group_idx % len(group_colors)]
        for edge_idx in range(start_idx, end_idx + 1):
            pair_color_map[edge_idx] = color


    # Create a small curvature map to avoid perfectly overlapping straight edges
    pair_rad_map = {}
    for idx in range(len(edge_pairs)):
        # alternate small radii to spread parallel edges
        rad = ((idx % 5) - 2) * 0.06  # values like -0.12, -0.06, 0.0, 0.06, 0.12
        pair_rad_map[idx] = rad

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

    # Create node labels - show activity names
    node_labels = {n: labels[n] for n in activities}

    # Track revealed edges for step mode (by index)
    revealed_edge_indices = set()

    def redraw(all_highlighted=None):
        ax.clear()
        # redraw nodes
        nx.draw_networkx_nodes(H, pos, nodelist=node_list, node_color=node_colors, node_size=900, ax=ax)
        nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=10, font_color='black', ax=ax)

        # draw unrevealed edges in light gray, with per-edge curvature to avoid overlap
        for idx in range(len(edge_pairs)):
            if idx not in revealed_edge_indices:
                e = edge_pairs[idx]
                rad = pair_rad_map.get(idx, 0.0)
                nx.draw_networkx_edges(H, pos, edgelist=[e], edge_color='lightgray', arrows=True, arrowsize=20, ax=ax, connectionstyle=f'arc3,rad={rad}')

        # draw revealed edges with unique color and thicker linewidth
        if revealed_edge_indices:
            for idx in revealed_edge_indices:
                e = edge_pairs[idx]
                color = pair_color_map.get(idx, 'k')
                rad = pair_rad_map.get(idx, 0.0)
                nx.draw_networkx_edges(H, pos, edgelist=[e], edge_color=[color], arrows=True, arrowsize=20, width=3.0, ax=ax, connectionstyle=f'arc3,rad={rad}')

    ax.set_axis_off()
    # redraw; GridSpec handles spacing so no tight_layout rectangle is needed here
    plt.draw()

    # Show initial graph with nodes only
    redraw()
    plt.show(block=False)

    # In step mode, reveal edges GROUP by GROUP (consecutive edges with same source)
    if step_mode and edge_pairs:
        total_steps = len(edge_groups)
        print(f"Interactive reveal: {total_steps} groups. Press Enter to reveal next group (q + Enter to quit).")
        for group_idx, (start_idx, end_idx, source) in enumerate(edge_groups):
            user_in = input()
            if user_in.lower().strip() == 'q':
                break
            # Reveal all edges in this group at once
            for edge_idx in range(start_idx, end_idx + 1):
                revealed_edge_indices.add(edge_idx)
            redraw()
            plt.pause(0.01)
        # final pause to inspect
        print("Reveal finished. Close the plot window or press Enter to continue.")
        try:
            input()
        except Exception:
            pass
    else:
        # If not step_mode, draw all edges with group colors and non-overlapping curvature
        for idx in range(len(edge_pairs)):
            e = edge_pairs[idx]
            color = pair_color_map.get(idx, 'k')
            rad = pair_rad_map.get(idx, 0.0)
            nx.draw_networkx_edges(H, pos, edgelist=[e], edge_color=[color], arrows=True, arrowsize=20, ax=ax, connectionstyle=f'arc3,rad={rad}')

    # Draw labels on activity-nodes with readable font
    nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=10, font_color='black', ax=ax)

    # Title: filename on top
    # title already set in redraw

    # Create a legend for edges on the right showing unique edge colors
    # Each edge (including duplicates) has its own color in original file order
    if edge_pairs:
        handles = [Line2D([0], [0], color=pair_color_map[i], lw=3) for i in range(len(edge_pairs))]
        edge_legend_labels = [f"{labels[u]} → {labels[v]}" for u, v in edge_pairs]
        # Draw legend inside the dedicated legend axis so it's always visible
        legend = legend_ax.legend(handles=handles, labels=edge_legend_labels, loc='center', frameon=True)
        # Color only the legend handles (markers/lines); keep text black for readability
        for idx, lh in enumerate(handles):
            try:
                lh.set_color(pair_color_map[idx])
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