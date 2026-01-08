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
    edges_ordered = []
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
                edges_ordered.append((source, target))
    return G, edges_ordered



samples_dir = Path("../../output_graph_gan/samples/20251201_105053")

if not samples_dir.exists():
    raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
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
    G, edges_ordered = load_graph_from_txt(filename)
    labels = {node: data['activity'] for node, data in G.nodes(data=True)}
    
    activities = list(G.nodes())
    activity_to_nodes = {n: [n] for n in G.nodes()}
    activities = [a for a in activities if not (str(labels.get(a, a)).upper() in ("START", "END") and len(activity_to_nodes.get(a, [])) == 0)]
    
    H = nx.DiGraph()
    for act in activities:
        H.add_node(act)

    edge_pairs = []
    for u, v in edges_ordered:
        au = u
        av = v
        edge_pairs.append((au, av))
 
        if not H.has_edge(au, av):
            H.add_edge(au, av)
    start_activities = [a for a in activities if str(labels.get(a, a)).upper() == 'START']
    end_activities = [a for a in activities if str(labels.get(a, a)).upper() == 'END']

    for s in start_activities:
        if H.has_node(s) and H.degree(s) == 0:
            
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


    step_mode = ('--step' in sys.argv)

    edge_groups = []
    if edge_pairs:
        current_source = edge_pairs[0][0]
        group_start = 0
        
        for idx in range(1, len(edge_pairs)):
            if edge_pairs[idx][0] != current_source:
                edge_groups.append((group_start, idx - 1, current_source))
                current_source = edge_pairs[idx][0]
                group_start = idx
        
        edge_groups.append((group_start, len(edge_pairs) - 1, current_source))
    
    num_groups = len(edge_groups)
    group_hues = np.linspace(0, 1, max(num_groups, 1), endpoint=False)
    group_colors = [mcolors.hsv_to_rgb((h, 0.75, 1.0)) for h in group_hues]
    
    pair_color_map = {}
    for group_idx, (start_idx, end_idx, source) in enumerate(edge_groups):
        color = group_colors[group_idx % len(group_colors)]
        for edge_idx in range(start_idx, end_idx + 1):
            pair_color_map[edge_idx] = color

    pair_rad_map = {}
    for idx in range(len(edge_pairs)):
        rad = ((idx % 5) - 2) * 0.06
        pair_rad_map[idx] = rad

    start_activities = [a for a in activities if str(a).upper() == 'START']
    end_activities = [a for a in activities if str(a).upper() == 'END']

    layers = {}
    from collections import deque

    if start_activities:
        queue = deque()
        for s in start_activities:
            layers[s] = 0
            queue.append(s)
    else:
        roots = [n for n, d in H.in_degree() if d == 0]
        if not roots:
            
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

    if layers:
        max_layer = max(layers.values())
    else:
        max_layer = 0
    for n in H.nodes():
        if n not in layers:
            max_layer += 1
            layers[n] = max_layer

    if end_activities:
        last_layer = max(layers.values())
        for e in end_activities:
            layers[e] = last_layer

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
    
    xs = np.array([p[0] for p in pos.values()]) if pos else np.array([0])
    if xs.ptp() == 0:
        scale_x = 1.0
    else:
        scale_x = 10.0 / xs.ptp()
    for n in pos:
        pos[n][0] = pos[n][0] * scale_x

    M = max(len(activities), 1)
    cmap_hues = np.linspace(0, 1, M, endpoint=False)
    bright_colors = [mcolors.hsv_to_rgb((h, 0.85, 1.0)) for h in cmap_hues]
    activity_color_map = {activities[i]: bright_colors[i % len(bright_colors)] for i in range(len(activities))}

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.72, 0.28], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis('off')
    fig.suptitle(f"{filename.name}", fontsize=14, fontweight='bold', y=0.98)

    node_list = activities
    node_colors = [activity_color_map[a] for a in node_list]
    nx.draw_networkx_nodes(H, pos, nodelist=node_list, node_color=node_colors, node_size=900, ax=ax)

    node_labels = {n: labels[n] for n in activities}

    revealed_edge_indices = set()

    def redraw(all_highlighted=None):
        ax.clear()
        nx.draw_networkx_nodes(H, pos, nodelist=node_list, node_color=node_colors, node_size=900, ax=ax)
        nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=10, font_color='black', ax=ax)

        for idx in range(len(edge_pairs)):
            if idx not in revealed_edge_indices:
                e = edge_pairs[idx]
                rad = pair_rad_map.get(idx, 0.0)
                nx.draw_networkx_edges(H, pos, edgelist=[e], edge_color='lightgray', arrows=True, arrowsize=20, ax=ax, connectionstyle=f'arc3,rad={rad}')

        if revealed_edge_indices:
            for idx in revealed_edge_indices:
                e = edge_pairs[idx]
                color = pair_color_map.get(idx, 'k')
                rad = pair_rad_map.get(idx, 0.0)
                nx.draw_networkx_edges(H, pos, edgelist=[e], edge_color=[color], arrows=True, arrowsize=20, width=3.0, ax=ax, connectionstyle=f'arc3,rad={rad}')

    ax.set_axis_off()
    plt.draw()

    redraw()
    plt.show(block=False)

    if step_mode and edge_pairs:
        total_steps = len(edge_groups)
        print(f"Interactive reveal: {total_steps} groups. Press Enter to reveal next group (q + Enter to quit).")
        for group_idx, (start_idx, end_idx, source) in enumerate(edge_groups):
            user_in = input()
            if user_in.lower().strip() == 'q':
                break
            for edge_idx in range(start_idx, end_idx + 1):
                revealed_edge_indices.add(edge_idx)
            redraw()
            plt.pause(0.01)
        print("Reveal finished. Close the plot window or press Enter to continue.")
        try:
            input()
        except Exception:
            pass
    else:
        for idx in range(len(edge_pairs)):
            e = edge_pairs[idx]
            color = pair_color_map.get(idx, 'k')
            rad = pair_rad_map.get(idx, 0.0)
            nx.draw_networkx_edges(H, pos, edgelist=[e], edge_color=[color], arrows=True, arrowsize=20, ax=ax, connectionstyle=f'arc3,rad={rad}')

    nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=10, font_color='black', ax=ax)

    if edge_pairs:
        handles = [Line2D([0], [0], color=pair_color_map[i], lw=3) for i in range(len(edge_pairs))]
        edge_legend_labels = [f"{labels[u]} → {labels[v]}" for u, v in edge_pairs]
        legend = legend_ax.legend(handles=handles, labels=edge_legend_labels, loc='center', frameon=True)
        for idx, lh in enumerate(handles):
            try:
                lh.set_color(pair_color_map[idx])
                lh.set_linewidth(3.0)
            except Exception:
                pass
        for txt in legend.get_texts():
            txt.set_color('black')
            txt.set_fontsize(9)
        legend_ax.set_facecolor('#ffffff')
    else:
        handles = [Patch(facecolor=activity_color_map[a], edgecolor='none', label=str(a)) for a in activities]
        legend = legend_ax.legend(handles=handles, loc='center', frameon=True)
        for txt in legend.get_texts():
            txt.set_color('black')
            txt.set_fontsize(9)
        legend_ax.set_facecolor('#ffffff')

    ax.set_axis_off()
    plt.draw()
    plt.show()