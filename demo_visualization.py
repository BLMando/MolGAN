#!/usr/bin/env python3
"""
Demo visualization without TensorFlow dependencies

Shows how the interactive graph visualizations would look.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
from pathlib import Path


class SimpleGraphDemo:
    """Simple demo of graph visualization without GAN"""

    def __init__(self, csv_path):
        # Load traces from CSV
        df = pd.read_csv(csv_path)
        self.traces = []
        for case_id, group in df.groupby('case_id'):
            trace = group['activity'].tolist()
            self.traces.append(trace)

        self.current_idx = 0

        # Create figure
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('ProcessGAN Graph Visualization Demo',
                         fontsize=16, fontweight='bold')

        # Graph axis
        self.ax_graph = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)

        # Trace axis
        self.ax_trace = plt.subplot2grid((3, 2), (2, 0), colspan=2)

        # Buttons
        ax_prev = plt.axes([0.3, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.6, 0.02, 0.1, 0.04])

        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')

        self.btn_prev.on_clicked(self.show_previous)
        self.btn_next.on_clicked(self.show_next)

        # Keyboard
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.update_display()

    def update_display(self):
        """Update display with current trace"""
        self.ax_graph.clear()
        self.ax_trace.clear()

        trace = self.traces[self.current_idx]

        # Draw graph view
        self._draw_graph(trace, self.ax_graph)

        # Draw trace view
        self._draw_trace(trace, self.ax_trace)

        self.fig.canvas.draw()

    def _draw_graph(self, trace, ax):
        """Draw graph as network"""
        import networkx as nx

        G = nx.DiGraph()

        # Colors
        node_colors_map = {
            'Start': '#90EE90',
            'End': '#FFB6C6',
            'default': '#87CEEB'
        }

        # Add nodes
        node_colors = []
        for i, activity in enumerate(trace):
            G.add_node(i, label=activity)
            if 'Start' in activity:
                node_colors.append(node_colors_map['Start'])
            elif 'End' in activity:
                node_colors.append(node_colors_map['End'])
            else:
                node_colors.append(node_colors_map['default'])

        # Add edges
        for i in range(len(trace) - 1):
            G.add_edge(i, i + 1)

        # Layout
        pos = {}
        n = len(trace)
        for i in range(n):
            # Hierarchical left-to-right
            x = i / (n - 1) if n > 1 else 0.5
            y = 0.5 + 0.1 * np.sin(i * 0.5)  # Slight wave for visual appeal
            pos[i] = (x, y)

        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=2500, alpha=0.9, ax=ax)

        nx.draw_networkx_edges(G, pos, edge_color='#2E86AB',
                              width=3, alpha=0.7, arrowsize=25,
                              arrowstyle='->', ax=ax)

        labels = {i: trace[i] for i in range(len(trace))}
        nx.draw_networkx_labels(G, pos, labels,
                               font_size=9, font_weight='bold', ax=ax)

        ax.set_title(f'Graph View: Trace {self.current_idx + 1}/{len(self.traces)}',
                    fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')

    def _draw_trace(self, trace, ax):
        """Draw trace as sequence"""
        for i, activity in enumerate(trace):
            # Color
            if 'Start' in activity:
                color = '#90EE90'
            elif 'End' in activity:
                color = '#FFB6C6'
            else:
                color = '#87CEEB'

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
        """Next trace"""
        self.current_idx = (self.current_idx + 1) % len(self.traces)
        self.update_display()

    def show_previous(self, event):
        """Previous trace"""
        self.current_idx = (self.current_idx - 1) % len(self.traces)
        self.update_display()

    def on_key_press(self, event):
        """Keyboard shortcuts"""
        if event.key in ['right', 'n']:
            self.show_next(None)
        elif event.key in ['left', 'p']:
            self.show_previous(None)
        elif event.key == 'q':
            plt.close(self.fig)

    def show(self):
        """Show viewer"""
        print("\n" + "="*60)
        print("ProcessGAN Visualization Demo")
        print("="*60)
        print("\nShowing real traces from dataset")
        print("(Generated graphs will look similar but with GAN output)")
        print("\nControls:")
        print("  Next: Click 'Next' or press Right/N")
        print("  Previous: Click 'Previous' or press Left/P")
        print("  Quit: Press Q or close window")
        print("="*60 + "\n")
        plt.show()


def main():
    csv_path = 'data/sample_event_log.csv'

    if not Path(csv_path).exists():
        print(f"[ERROR] File not found: {csv_path}")
        print("Please create sample data first.")
        return

    print("[INFO] Loading traces from CSV...")
    viewer = SimpleGraphDemo(csv_path)
    viewer.show()


if __name__ == '__main__':
    main()
