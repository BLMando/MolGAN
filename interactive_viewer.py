#!/usr/bin/env python3
"""
Interactive ProcessGAN Graph Viewer

Displays generated graphs in an interactive window with navigation controls.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path

from utils.process_dataset import ProcessDataset
from models.process_gan import ProcessGAN
from utils.visualization import ProcessGraphVisualizer


class InteractiveGraphViewer:
    """Interactive viewer for ProcessGAN graphs"""

    def __init__(self, dataset, model, visualizer, z_dim=128, n_samples=10):
        self.dataset = dataset
        self.model = model
        self.visualizer = visualizer
        self.z_dim = z_dim
        self.n_samples = n_samples

        # Generate samples
        print(f"[INFO] Generating {n_samples} samples...")
        self.z_samples = np.random.normal(0, 1, (n_samples, z_dim))
        self.adjacency_batch, self.nodes_batch = model.generator(
            self.z_samples, training=False)
        self.adjacency_batch = self.adjacency_batch.numpy()
        self.nodes_batch = self.nodes_batch.numpy()
        self.traces = dataset.decode_batch(
            self.adjacency_batch, self.nodes_batch)

        # Current index
        self.current_idx = 0

        # Create figure
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('Interactive ProcessGAN Graph Viewer',
                          fontsize=16, fontweight='bold')

        # Main graph axis
        self.ax_graph = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)

        # Trace axis
        self.ax_trace = plt.subplot2grid((3, 2), (2, 0), colspan=2)

        # Button axes
        ax_prev = plt.axes([0.3, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.6, 0.02, 0.1, 0.04])
        ax_regen = plt.axes([0.45, 0.02, 0.1, 0.04])

        # Create buttons
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_regen = Button(ax_regen, 'Regenerate')

        # Connect callbacks
        self.btn_prev.on_clicked(self.show_previous)
        self.btn_next.on_clicked(self.show_next)
        self.btn_regen.on_clicked(self.regenerate)

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Show first graph
        self.update_display()

    def update_display(self):
        """Update the display with current graph"""
        # Clear axes
        self.ax_graph.clear()
        self.ax_trace.clear()

        idx = self.current_idx
        adj = self.adjacency_batch[idx]
        nodes = self.nodes_batch[idx]
        trace = self.traces[idx]

        # Update title
        self.ax_graph.set_title(
            f'Generated Graph {idx + 1}/{self.n_samples}',
            fontsize=14, fontweight='bold', pad=10
        )

        # Visualize graph
        self._draw_graph_on_axis(adj, nodes, self.ax_graph)

        # Visualize trace
        self._draw_trace_on_axis(trace, self.ax_trace)

        # Redraw
        self.fig.canvas.draw()

    def _draw_graph_on_axis(self, adjacency, nodes, ax):
        """Draw graph on given axis"""
        import networkx as nx

        # Decode
        if len(adjacency.shape) == 3:
            adj_indices = np.argmax(adjacency, axis=-1)
        else:
            adj_indices = adjacency

        if len(nodes.shape) == 2:
            node_indices = np.argmax(nodes, axis=-1)
        else:
            node_indices = nodes

        # Create graph
        G = nx.DiGraph()
        activity_labels = {}
        node_colors = []
        active_nodes = []

        for i in range(len(node_indices)):
            activity_idx = node_indices[i]
            if activity_idx in self.visualizer.activity_decoder:
                activity_name = self.visualizer.activity_decoder[activity_idx]
                if activity_name not in ['PAD', '<PAD>', '']:
                    G.add_node(i)
                    activity_labels[i] = activity_name
                    active_nodes.append(i)

                    # Color
                    if 'Start' in activity_name:
                        node_colors.append(
                            self.visualizer.node_colors['Start'])
                    elif 'End' in activity_name:
                        node_colors.append(self.visualizer.node_colors['End'])
                    else:
                        node_colors.append(
                            self.visualizer.node_colors['default'])

        # Add edges
        edge_colors = []
        edge_labels = {}
        edge_styles = []

        for i in active_nodes:
            for j in active_nodes:
                flow_idx = adj_indices[i, j]
                if flow_idx > 0 and flow_idx in self.visualizer.flow_decoder:
                    flow_type = self.visualizer.flow_decoder[flow_idx]
                    G.add_edge(i, j)
                    edge_labels[(i, j)] = flow_type
                    edge_colors.append(
                        self.visualizer.flow_colors.get(flow_type, '#888888'))

                    if flow_type == 'LOOP':
                        edge_styles.append('dashed')
                    elif flow_type == 'CHOICE':
                        edge_styles.append('dotted')
                    else:
                        edge_styles.append('solid')

        # Layout
        if len(G.nodes()) > 0:
            try:
                pos = self.visualizer._hierarchical_layout(G, active_nodes)
            except:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

            # Draw
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=2000, alpha=0.9, ax=ax)

            for (u, v), style in zip(G.edges(), edge_styles):
                edge_color = edge_colors[list(G.edges()).index((u, v))]
                nx.draw_networkx_edges(G, pos, [(u, v)],
                                       edge_color=[edge_color],
                                       style=style, width=2.5, alpha=0.7,
                                       arrowsize=20, arrowstyle='->',
                                       connectionstyle='arc3,rad=0.1', ax=ax)

            nx.draw_networkx_labels(G, pos, activity_labels,
                                    font_size=9, font_weight='bold', ax=ax)

            nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                         font_size=7, ax=ax)

            # Legend
            import matplotlib.patches as mpatches
            legend_elements = [
                mpatches.Patch(color=color, label=flow_type)
                for flow_type, color in self.visualizer.flow_colors.items()
                if flow_type in edge_labels.values()
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax.axis('off')

    def _draw_trace_on_axis(self, trace, ax):
        """Draw trace on given axis"""
        if not trace:
            ax.text(0.5, 0.5, 'No valid trace', ha='center', va='center')
            ax.axis('off')
            return

        for i, activity in enumerate(trace):
            # Color
            if 'Start' in activity:
                color = self.visualizer.node_colors['Start']
            elif 'End' in activity:
                color = self.visualizer.node_colors['End']
            else:
                color = self.visualizer.node_colors['default']

            # Box
            import matplotlib.patches as mpatches
            rect = mpatches.FancyBboxPatch((i*1.5, 0), 1.2, 0.6,
                                           boxstyle="round,pad=0.05",
                                           facecolor=color,
                                           edgecolor='black',
                                           linewidth=2)
            ax.add_patch(rect)

            # Text
            ax.text(i*1.5 + 0.6, 0.3, activity,
                    ha='center', va='center',
                    fontsize=8, fontweight='bold')

            # Arrow
            if i < len(trace) - 1:
                ax.arrow(i*1.5 + 1.2, 0.3, 0.2, 0,
                         head_width=0.15, head_length=0.1,
                         fc='black', ec='black')

        ax.set_xlim(-0.3, len(trace)*1.5)
        ax.set_ylim(-0.2, 0.8)
        ax.axis('off')
        ax.set_title(f'Trace: {" → ".join(trace)}', fontsize=10, pad=10)

    def show_next(self, event):
        """Show next graph"""
        self.current_idx = (self.current_idx + 1) % self.n_samples
        self.update_display()

    def show_previous(self, event):
        """Show previous graph"""
        self.current_idx = (self.current_idx - 1) % self.n_samples
        self.update_display()

    def regenerate(self, event):
        """Regenerate all samples"""
        print("[INFO] Regenerating samples...")
        self.z_samples = np.random.normal(0, 1, (self.n_samples, self.z_dim))
        self.adjacency_batch, self.nodes_batch = self.model.generator(
            self.z_samples, training=False)
        self.adjacency_batch = self.adjacency_batch.numpy()
        self.nodes_batch = self.nodes_batch.numpy()
        self.traces = self.dataset.decode_batch(
            self.adjacency_batch, self.nodes_batch)
        self.current_idx = 0
        self.update_display()
        print("[INFO] Done!")

    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'right' or event.key == 'n':
            self.show_next(None)
        elif event.key == 'left' or event.key == 'p':
            self.show_previous(None)
        elif event.key == 'r':
            self.regenerate(None)
        elif event.key == 'q':
            plt.close(self.fig)

    def show(self):
        """Show the viewer"""
        print("\n[INFO] Interactive Viewer Controls:")
        print("  - Next: Click 'Next' or press Right/N")
        print("  - Previous: Click 'Previous' or press Left/P")
        print("  - Regenerate: Click 'Regenerate' or press R")
        print("  - Quit: Press Q or close window")
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Interactive ProcessGAN Graph Viewer')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to event log CSV')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model checkpoint (optional)')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--z-dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    if args.seed:
        np.random.seed(args.seed)

    print("[INFO] Loading dataset...")
    dataset = ProcessDataset()
    dataset.load_from_csv(args.data, case_id_col='case_id',
                          activity_col='activity')

    print(f"[INFO] Dataset: {len(dataset.data)} traces, "
          f"{dataset.max_activities} activities, "
          f"{len(dataset.flow_encoder)} flow types")

    # Create visualizer
    activity_decoder = {v: k for k, v in dataset.activity_encoder.items()}
    flow_decoder = {v: k for k, v in dataset.flow_encoder.items()}
    visualizer = ProcessGraphVisualizer(activity_decoder, flow_decoder)

    # Create model
    print("[INFO] Creating model...")
    model = ProcessGAN(
        max_activities=dataset.max_activities,
        flow_types=len(dataset.flow_encoder),
        activity_types=len(dataset.activity_encoder),
        embedding_dim=args.z_dim,
        decoder_units=[128, 256, 512],
        discriminator_units=[128, 64],
        mlp_units=64,
        dropout_rate=0.3
    )

    if args.model:
        print(f"[INFO] Loading model from {args.model}...")
        # TODO: Load checkpoint
        print("  [WARNING] Checkpoint loading not implemented yet")

    # Create and show viewer
    viewer = InteractiveGraphViewer(
        dataset, model, visualizer,
        z_dim=args.z_dim,
        n_samples=args.n_samples
    )
    viewer.show()


if __name__ == '__main__':
    main()
