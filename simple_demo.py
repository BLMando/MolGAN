#!/usr/bin/env python3
"""
Simple visualization demo - no external dependencies except matplotlib

Demonstrates the interactive viewer concept.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
from pathlib import Path


class InteractiveTraceViewer:
    """Interactive viewer for process traces"""

    def __init__(self, traces):
        self.traces = traces
        self.current_idx = 0

        # Create figure
        self.fig, self.axes = plt.subplots(2, 1, figsize=(14, 8))
        self.fig.suptitle('ProcessGAN Interactive Viewer Demo',
                         fontsize=16, fontweight='bold')

        # Buttons
        ax_prev = plt.axes([0.3, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.6, 0.02, 0.1, 0.04])

        self.btn_prev = Button(ax_prev, '← Previous')
        self.btn_next = Button(ax_next, 'Next →')

        self.btn_prev.on_clicked(self.show_previous)
        self.btn_next.on_clicked(self.show_next)

        # Keyboard
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.update_display()

    def update_display(self):
        """Update display"""
        for ax in self.axes:
            ax.clear()

        trace = self.traces[self.current_idx]

        # Graph view (circular layout)
        self._draw_circular_graph(trace, self.axes[0])

        # Trace view (sequential)
        self._draw_sequential_trace(trace, self.axes[1])

        self.fig.canvas.draw()

    def _draw_circular_graph(self, trace, ax):
        """Draw as circular graph"""
        n = len(trace)
        if n == 0:
            return

        # Calculate positions on circle
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 1.0
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)

        # Draw edges (connections)
        for i in range(n - 1):
            ax.arrow(x[i], y[i],
                    x[i+1] - x[i], y[i+1] - y[i],
                    head_width=0.08, head_length=0.06,
                    fc='#2E86AB', ec='#2E86AB',
                    alpha=0.6, length_includes_head=True)

        # Draw nodes
        for i, activity in enumerate(trace):
            # Color
            if 'Start' in activity:
                color = '#90EE90'
            elif 'End' in activity:
                color = '#FFB6C6'
            else:
                color = '#87CEEB'

            # Node circle
            circle = plt.Circle((x[i], y[i]), 0.15,
                              facecolor=color,
                              edgecolor='black',
                              linewidth=2,
                              zorder=10)
            ax.add_patch(circle)

            # Label
            ax.text(x[i], y[i], activity,
                   ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   zorder=11)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Graph View: Trace {self.current_idx + 1}/{len(self.traces)}',
                    fontsize=12, fontweight='bold', pad=20)

    def _draw_sequential_trace(self, trace, ax):
        """Draw as sequential trace"""
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

        ax.set_xlim(-0.3, len(trace)*1.5 if trace else 1)
        ax.set_ylim(-0.2, 0.8)
        ax.axis('off')
        ax.set_title(f'Sequential View: {" → ".join(trace)}',
                    fontsize=10, pad=10)

    def show_next(self, event):
        self.current_idx = (self.current_idx + 1) % len(self.traces)
        self.update_display()

    def show_previous(self, event):
        self.current_idx = (self.current_idx - 1) % len(self.traces)
        self.update_display()

    def on_key_press(self, event):
        if event.key in ['right', 'n']:
            self.show_next(None)
        elif event.key in ['left', 'p']:
            self.show_previous(None)
        elif event.key == 'q':
            plt.close(self.fig)

    def show(self):
        print("\n" + "="*70)
        print("       ProcessGAN Interactive Graph Viewer - Demo")
        print("="*70)
        print("\n📊 Showing real traces from dataset")
        print("   (With trained model, these would be GAN-generated graphs)")
        print("\n⌨️  Controls:")
        print("   • Next:     Click 'Next →' or press Right Arrow / N")
        print("   • Previous: Click '← Previous' or press Left Arrow / P")
        print("   • Quit:     Press Q or close window")
        print("\n💡 Features in full version:")
        print("   • Interactive node dragging (NetworkX)")
        print("   • Multiple edge types with different colors")
        print("   • Hierarchical/force-directed layouts")
        print("   • Real-time regeneration")
        print("   • Zoom and pan")
        print("="*70 + "\n")
        plt.show()


def load_traces_from_csv(csv_path):
    """Load traces from CSV"""
    df = pd.read_csv(csv_path)
    traces = []
    for case_id, group in df.groupby('case_id'):
        trace = group['activity'].tolist()
        traces.append(trace)
    return traces


def main():
    csv_path = 'data/sample_event_log.csv'

    if Path(csv_path).exists():
        print("[INFO] Loading traces from CSV...")
        traces = load_traces_from_csv(csv_path)
        print(f"[INFO] Loaded {len(traces)} traces")
    else:
        print("[INFO] Using example traces...")
        traces = [
            ['Start', 'Submit', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Review', 'Reject', 'Rework', 'Submit', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Approve', 'End'],
            ['Start', 'Submit', 'Review', 'Approve', 'Send', 'End'],
            ['Start', 'Submit', 'Check', 'Review', 'Approve', 'End']
        ]

    viewer = InteractiveTraceViewer(traces)
    viewer.show()


if __name__ == '__main__':
    main()
