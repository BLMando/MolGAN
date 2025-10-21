#!/usr/bin/env python3
"""
Web-based Interactive ProcessGAN Graph Viewer

Uses Plotly for interactive, zoomable, draggable graph visualization.
Opens in browser automatically.
"""

import argparse
import numpy as np
from pathlib import Path
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARNING] Plotly not installed. Install with: pip install plotly")

from utils.process_dataset import ProcessDataset
from models.process_gan import ProcessGAN


class ProcessGraphWebViewer:
    """Web-based interactive viewer using Plotly"""

    def __init__(self, dataset, model, z_dim=128):
        self.dataset = dataset
        self.model = model
        self.z_dim = z_dim

        # Decoders
        self.activity_decoder = {v: k for k,
                                 v in dataset.activity_encoder.items()}
        self.flow_decoder = {v: k for k, v in dataset.flow_encoder.items()}

        # Colors
        self.flow_colors = {
            'SEQUENCE': '#2E86AB',
            'PARALLEL': '#A23B72',
            'LOOP': '#F18F01',
            'CHOICE': '#C73E1D',
            'SKIP': '#6A994E'
        }

        self.node_colors = {
            'Start': '#90EE90',
            'End': '#FFB6C6',
            'default': '#87CEEB'
        }

    def generate_samples(self, n_samples=10):
        """Generate graph samples"""
        print(f"[INFO] Generating {n_samples} samples...")
        z = np.random.normal(0, 1, (n_samples, self.z_dim))
        adjacency_batch, nodes_batch = self.model.generator(z, training=False)
        adjacency_batch = adjacency_batch.numpy()
        nodes_batch = nodes_batch.numpy()
        traces = self.dataset.decode_batch(adjacency_batch, nodes_batch)
        return adjacency_batch, nodes_batch, traces

    def create_graph_figure(self, adjacency, nodes, trace, title="Generated Graph"):
        """Create Plotly figure for a single graph"""
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

        # Build NetworkX graph
        G = nx.DiGraph()
        activity_labels = {}
        node_color_map = {}
        active_nodes = []

        for i in range(len(node_indices)):
            activity_idx = node_indices[i]
            if activity_idx in self.activity_decoder:
                activity_name = self.activity_decoder[activity_idx]
                if activity_name not in ['PAD', '<PAD>', '']:
                    G.add_node(i)
                    activity_labels[i] = activity_name
                    active_nodes.append(i)

                    # Color
                    if 'Start' in activity_name:
                        node_color_map[i] = self.node_colors['Start']
                    elif 'End' in activity_name:
                        node_color_map[i] = self.node_colors['End']
                    else:
                        node_color_map[i] = self.node_colors['default']

        # Add edges
        edge_info = []
        for i in active_nodes:
            for j in active_nodes:
                flow_idx = adj_indices[i, j]
                if flow_idx > 0 and flow_idx in self.flow_decoder:
                    flow_type = self.flow_decoder[flow_idx]
                    G.add_edge(i, j)
                    edge_info.append({
                        'source': i,
                        'target': j,
                        'type': flow_type,
                        'color': self.flow_colors.get(flow_type, '#888888')
                    })

        # Layout
        if len(G.nodes()) == 0:
            return None

        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = {i: (i, 0) for i in G.nodes()}

        # Create Plotly figure
        fig = go.Figure()

        # Add edges
        for edge in edge_info:
            x0, y0 = pos[edge['source']]
            x1, y1 = pos[edge['target']]

            # Arrow annotation
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=edge['color'],
                opacity=0.7
            )

            # Edge label
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            fig.add_trace(go.Scatter(
                x=[mid_x], y=[mid_y],
                mode='text',
                text=[edge['type']],
                textfont=dict(size=8, color='#333'),
                hoverinfo='skip',
                showlegend=False
            ))

        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_hover = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(activity_labels[node])
            node_colors.append(node_color_map[node])
            node_hover.append(f"{activity_labels[node]}<br>Node ID: {node}")

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=40,
                color=node_colors,
                line=dict(width=2, color='black')
            ),
            text=node_text,
            textfont=dict(size=10, color='black', family='Arial Black'),
            hovertext=node_hover,
            hoverinfo='text',
            showlegend=False
        ))

        # Layout
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>Trace: {' → '.join(trace)}</sub>",
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )

        return fig

    def create_dashboard(self, n_samples=6):
        """Create multi-graph dashboard"""
        adjacency_batch, nodes_batch, traces = self.generate_samples(n_samples)

        # Create subplots
        rows = (n_samples + 1) // 2
        fig = make_subplots(
            rows=rows, cols=2,
            subplot_titles=[f"Sample {i+1}" for i in range(n_samples)],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        for idx in range(n_samples):
            row = idx // 2 + 1
            col = idx % 2 + 1

            # Generate individual figure
            single_fig = self.create_graph_figure(
                adjacency_batch[idx],
                nodes_batch[idx],
                traces[idx],
                title=f"Sample {idx+1}"
            )

            if single_fig:
                # Add traces to subplot
                for trace in single_fig.data:
                    fig.add_trace(trace, row=row, col=col)

                # Copy annotations
                for annotation in single_fig.layout.annotations:
                    annotation.xref = f'x{idx+1}' if idx > 0 else 'x'
                    annotation.yref = f'y{idx+1}' if idx > 0 else 'y'
                    annotation.axref = f'x{idx+1}' if idx > 0 else 'x'
                    annotation.ayref = f'y{idx+1}' if idx > 0 else 'y'
                    fig.add_annotation(annotation)

        # Update layout
        fig.update_layout(
            title_text="ProcessGAN Generated Graphs Dashboard",
            showlegend=False,
            height=400 * rows,
            hovermode='closest'
        )

        # Hide axes
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

        return fig

    def save_html(self, fig, output_path='viewer.html'):
        """Save figure as HTML"""
        fig.write_html(output_path, auto_open=True)
        print(f"[INFO] Saved to: {Path(output_path).absolute()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Web-based ProcessGAN Graph Viewer')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to event log CSV')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--n-samples', type=int, default=6,
                        help='Number of samples to display')
    parser.add_argument('--z-dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--output', type=str, default='viewer.html',
                        help='Output HTML file')
    parser.add_argument('--mode', type=str, default='dashboard',
                        choices=['single', 'dashboard'],
                        help='Viewer mode')
    return parser.parse_args()


def main():
    if not PLOTLY_AVAILABLE:
        print("[ERROR] Plotly is required for web viewer")
        print("Install with: pip install plotly")
        return

    args = parse_args()

    print("[INFO] Loading dataset...")
    dataset = ProcessDataset()
    dataset.load_from_csv(args.data, case_id_col='case_id',
                          activity_col='activity')

    print(f"[INFO] Dataset: {len(dataset.traces)} traces")

    print("[INFO] Creating model...")
    model = ProcessGAN(
        max_activities=dataset.max_activities,
        flow_types=len(dataset.flow_encoder),
        activity_types=len(dataset.activity_encoder),
        embedding_dim=args.z_dim,
        decoder_units=[128, 256, 512],
        discriminator_units=[128, 64],
        mlp_units=[64],
        dropout_rate=0.3
    )

    if args.model:
        print(f"[INFO] Loading model from {args.model}...")
        print("  [WARNING] Checkpoint loading not implemented yet")

    # Create viewer
    viewer = ProcessGraphWebViewer(dataset, model, z_dim=args.z_dim)

    if args.mode == 'dashboard':
        print(f"[INFO] Creating dashboard with {args.n_samples} samples...")
        fig = viewer.create_dashboard(n_samples=args.n_samples)
    else:
        print("[INFO] Creating single graph...")
        adj, nodes, traces = viewer.generate_samples(n_samples=1)
        fig = viewer.create_graph_figure(adj[0], nodes[0], traces[0])

    # Save and open
    viewer.save_html(fig, args.output)
    print(f"\n[SUCCESS] Interactive viewer opened in browser!")
    print(f"  File: {Path(args.output).absolute()}")


if __name__ == '__main__':
    main()
