import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from pathlib import Path

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
    
    # Create labels from node activities
    labels = {node: data['activity'] for node, data in G.nodes(data=True)}
    
    # Optional: color nodes (use node id for simplicity)
    node_colors = list(G.nodes())
    
    plt.figure(figsize=(8, 7))
    pos = nx.spring_layout(G)  # Layout for better visualization
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        cmap="viridis",
        edge_color="gray",
        node_size=500,
        font_size=9,
        arrows=True,  # Show direction
        arrowsize=20
    )
    plt.title(f"Generated Graph\n{filename.name}", fontsize=12, fontweight='bold')
    plt.suptitle(f"File: {filename.name}", fontsize=10, y=0.02, color='gray')
    plt.tight_layout()
    plt.show()