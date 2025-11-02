"""
Custom Keras Callbacks for Process Mining GAN Training

This module provides a CallbackFactory class that creates and configures
all available callbacks for GAN training.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


class CallbackFactory:
    """
    Factory class for creating training callbacks

    This class centralizes all callback creation and configuration,
    making it easy to add or remove callbacks from training.

    Parameters:
    -----------
    output_dir: str
        Base output directory for saving checkpoints, logs, etc.
    idx_to_activity: dict, optional
        Mapping from activity indices to activity names (required for trace generation)
    """

    def __init__(self, output_dir, idx_to_activity=None):
        self.output_dir = output_dir
        self.idx_to_activity = idx_to_activity

    def trace_generation(self, num_traces=5, display_every=1):
        """
        Create callback to display generated trace examples during training

        Parameters:
        -----------
        num_traces: int, default=5
            Number of example traces to generate and display
        display_every: int, default=1
            Display traces every N epochs

        Returns:
        --------
        TraceGenerationCallback instance
        """
        if self.idx_to_activity is None:
            raise ValueError(
                "idx_to_activity must be provided to use trace_generation callback")

        return TraceGenerationCallback(
            idx_to_activity=self.idx_to_activity,
            num_traces=num_traces,
            display_every=display_every
        )

    def best_model_checkpoint(self, monitor='val_g_loss', mode='min', verbose=1):
        """
        Create callback to save the best model during training

        Parameters:
        -----------
        monitor: str, default='val_g_loss'
            Metric to monitor (e.g., 'val_g_loss', 'g_loss', 'val_d_loss')
        mode: str, default='min'
            One of {'min', 'max'}. Whether to minimize or maximize the metric
        verbose: int, default=1
            Verbosity level (0=silent, 1=print messages)

        Returns:
        --------
        BestModelCheckpoint instance
        """
        filepath = os.path.join(self.output_dir, 'checkpoints', 'best_model')

        return BestModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            mode=mode,
            verbose=verbose
        )

    def tensorboard(self, log_dir='tensorboard_logs', histogram_freq=1, write_graph=True):
        """
        Create TensorBoard callback for visualization

        Parameters:
        -----------
        log_dir: str, default='tensorboard_logs'
            Directory name for TensorBoard logs
        histogram_freq: int, default=1
            Frequency (in epochs) for logging histograms
        write_graph: bool, default=True
            Whether to visualize the model graph

        Returns:
        --------
        TensorBoard instance
        """
        filepath = os.path.join(self.output_dir, log_dir)

        return tf.keras.callbacks.TensorBoard(
            log_dir=filepath,
            histogram_freq=histogram_freq,
            write_graph=write_graph
        )

    def early_stopping(self, monitor='val_g_loss', patience=20, mode='min',
                       restore_best_weights=True):
        """
        Create early stopping callback

        Parameters:
        -----------
        monitor: str, default='val_g_loss'
            Metric to monitor
        patience: int, default=20
            Number of epochs with no improvement after which training will be stopped
        mode: str, default='min'
            One of {'min', 'max'}
        restore_best_weights: bool, default=True
            Whether to restore model weights from the best epoch

        Returns:
        --------
        EarlyStopping instance
        """
        return tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=restore_best_weights,
            verbose=1
        )

    def learning_rate_scheduler(self, schedule_fn):
        """
        Create learning rate scheduler callback

        Parameters:
        -----------
        schedule_fn: callable
            Function that takes (epoch, current_lr) and returns new learning rate

        Returns:
        --------
        LearningRateScheduler instance
        """
        return tf.keras.callbacks.LearningRateScheduler(schedule_fn, verbose=1)


    def backup_and_restore(self, backup_dir='backup', save_freq='epoch'):
        """
        Create backup and restore callback for automatic recovery

        Parameters:
        -----------
        backup_dir: str, default='backup'
            Directory name for backup files
        save_freq: str or int, default='epoch'
            'epoch' to save after each epoch, or integer for batch frequency

        Returns:
        --------
        BackupAndRestore instance
        """
        filepath = os.path.join(self.output_dir, backup_dir)

        return tf.keras.callbacks.BackupAndRestore(
            backup_dir=filepath,
            save_freq=save_freq
        )
    
    def quality_evaluation(self, real_traces, activity_to_idx, idx_to_activity,
                          eval_every=10, num_samples=100):
        """
        Create callback to evaluate generation quality during training
        
        This is a PROPER TEST - computes metrics and logs them!
        
        Parameters:
        -----------
        real_traces: list of lists
            Real training traces for comparison
        activity_to_idx: dict
            Activity name to index mapping
        idx_to_activity: dict
            Index to activity name mapping
        eval_every: int, default=10
            Evaluate every N epochs
        num_samples: int, default=100
            Number of samples to generate for evaluation
        
        Returns:
        --------
        QualityEvaluationCallback instance
        """
        if self.idx_to_activity is None:
            idx_to_activity_arg = idx_to_activity
        else:
            idx_to_activity_arg = self.idx_to_activity
            
        return QualityEvaluationCallback(
            real_traces=real_traces,
            activity_to_idx=activity_to_idx,
            idx_to_activity=idx_to_activity_arg,
            eval_every=eval_every,
            num_samples=num_samples
        )



# ============================================================================
# CUSTOM CALLBACK IMPLEMENTATIONS
# ============================================================================

class TraceGenerationCallback(keras.callbacks.Callback):
    """Callback to generate and display example traces during training"""

    def __init__(self, idx_to_activity, num_traces=5, display_every=1):
        super().__init__()
        self.idx_to_activity = idx_to_activity
        self.num_traces = num_traces
        self.display_every = display_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.display_every != 0:
            return

        temp = self.model.temperature.numpy()

        print(f"\n{'='*70}")
        print(f"Generated Samples (Epoch {epoch + 1}, Temperature={temp:.3f})")
        print('='*70)

        traces = self.model.generate_traces(
            num_samples=self.num_traces,
            temperature=temp,
            return_indices=True
        )

        for i, trace_indices in enumerate(traces, 1):
            activities = []
            for idx in trace_indices:
                if idx in self.idx_to_activity:
                    activity = self.idx_to_activity[idx]
                    if activity not in ['<PAD>', '<UNK>']:
                        activities.append(activity)

            if activities:
                trace_str = ' → '.join(activities)
            else:
                trace_str = "(empty trace)"

            print(f"  {i}. {trace_str}")

        print('='*70 + '\n')


class BestModelCheckpoint(keras.callbacks.Callback):
    """Save the best model based on a monitored metric"""

    def __init__(self, filepath, monitor='val_g_loss', mode='min', verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        if mode == 'min':
            self.best = np.inf
            self.monitor_op = np.less
        elif mode == 'max':
            self.best = -np.inf
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got: {mode}")

    def on_train_begin(self, logs=None):
        if self.verbose > 0:
            print(
                f"\nBestModelCheckpoint: Monitoring '{self.monitor}' (mode={self.mode})")
            print(f"Will save best model to: {self.filepath}\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            if self.verbose > 0 and epoch == 0:
                print(f"\nWarning: Metric '{self.monitor}' not found in logs.")
                print(f"Available metrics: {list(logs.keys())}\n")
            return

        if self.monitor_op(current, self.best):
            if self.verbose > 0:
                improvement = current - self.best
                print(f"\n{'='*70}")
                print(f"Epoch {epoch + 1}: {self.monitor} improved!")
                print(f"  Previous best: {self.best:.6f}")
                print(f"  Current value: {current:.6f}")
                print(f"  Improvement:   {improvement:.6f}")
                print(f"Saving model to {self.filepath}")
                print('='*70 + '\n')

            self.best = current

            os.makedirs(self.filepath, exist_ok=True)

            weights_path = os.path.join(self.filepath, 'best_model_weights.h5')
            try:
                self.model.save_weights(weights_path)
            except Exception as e:
                print(f"Error saving model weights: {e}")
                return

            info_path = os.path.join(self.filepath, 'best_info.txt')
            try:
                with open(info_path, 'w') as f:
                    f.write("="*50 + "\n")
                    f.write("BEST MODEL INFORMATION\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Epoch: {epoch + 1}\n")
                    f.write(f"Monitored metric: {self.monitor}\n")
                    f.write(f"Best value: {current:.6f}\n\n")
                    f.write("All metrics at this epoch:\n")
                    f.write("-"*50 + "\n")
                    for key, value in sorted(logs.items()):
                        if isinstance(value, (int, float)):
                            f.write(f"{key:30s}: {value:.6f}\n")
                        else:
                            f.write(f"{key:30s}: {value}\n")
            except Exception as e:
                print(f"Error saving model info: {e}")


class QualityEvaluationCallback(keras.callbacks.Callback):
    """
    Evaluate generation quality during training
    
    This is a PROPER TEST that computes metrics and logs them to TensorBoard/CSV.
    Unlike TraceGenerationCallback which just prints traces.
    """
    
    def __init__(self, real_traces, activity_to_idx, idx_to_activity, 
                 eval_every=10, num_samples=100):
        super().__init__()
        self.real_traces = real_traces
        self.activity_to_idx = activity_to_idx
        self.idx_to_activity = idx_to_activity
        self.eval_every = eval_every
        self.num_samples = num_samples
        
        # Store real traces as set for fast lookup
        self.real_set = set(tuple(trace) for trace in real_traces)
        
        # Track metrics over time
        self.history = {
            'epoch': [],
            'diversity': [],
            'novelty': [],
            'start_validity': [],
            'end_validity': [],
            'avg_length': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        """Evaluate quality every N epochs"""
        if (epoch + 1) % self.eval_every != 0:
            return
        
        logs = logs or {}
        
        print(f"\n{'='*70}")
        print(f"Quality Evaluation (Epoch {epoch + 1})")
        print('='*70)
        
        # Generate samples
        try:
            traces_indices = self.model.generate_traces(
                num_samples=self.num_samples,
                temperature=0.5,
                return_indices=True
            )
        except Exception as e:
            print(f"Error generating traces: {e}")
            return
        
        # Convert to activity names and clean
        synthetic_traces = []
        for trace_indices in traces_indices:
            activities = []
            for idx in trace_indices:
                if idx in self.idx_to_activity:
                    activity = self.idx_to_activity[idx]
                    if activity not in ['<PAD>', '<UNK>']:
                        activities.append(activity)
            if len(activities) > 0:
                synthetic_traces.append(activities)
        
        if len(synthetic_traces) == 0:
            print("Warning: No valid traces generated")
            return
        
        # ====================================================================
        # COMPUTE METRICS (proper evaluation!)
        # ====================================================================
        
        # 1. Diversity: ratio of unique traces
        unique_traces = len(set(tuple(t) for t in synthetic_traces))
        diversity = unique_traces / len(synthetic_traces)
        
        # 2. Novelty: ratio of traces not in training data
        novel_count = sum(1 for t in synthetic_traces if tuple(t) not in self.real_set)
        novelty = novel_count / len(synthetic_traces)
        
        # 3. START token validity
        start_token = 'START'
        start_valid = sum(1 for t in synthetic_traces if len(t) > 0 and t[0] == start_token)
        start_validity = start_valid / len(synthetic_traces)
        
        # 4. END token presence
        end_token = '6'  # Your END token
        end_valid = sum(1 for t in synthetic_traces if end_token in t)
        end_validity = end_valid / len(synthetic_traces)
        
        # 5. Average trace length
        avg_length = np.mean([len(t) for t in synthetic_traces])
        real_avg_length = np.mean([len(t) for t in self.real_traces])
        
        # ====================================================================
        # LOG METRICS (these get tracked!)
        # ====================================================================
        
        logs['gen_diversity'] = diversity
        logs['gen_novelty'] = novelty
        logs['gen_start_validity'] = start_validity
        logs['gen_end_validity'] = end_validity
        logs['gen_avg_length'] = avg_length
        
        # Store in history
        self.history['epoch'].append(epoch + 1)
        self.history['diversity'].append(diversity)
        self.history['novelty'].append(novelty)
        self.history['start_validity'].append(start_validity)
        self.history['end_validity'].append(end_validity)
        self.history['avg_length'].append(avg_length)
        
        # ====================================================================
        # PRINT RESULTS
        # ====================================================================
        
        print(f"\nMetrics (from {len(synthetic_traces)} generated traces):")
        print(f"  Diversity:        {diversity:.3f}  (target: >0.6)")
        print(f"  Novelty:          {novelty:.3f}  (target: 0.4-0.8)")
        print(f"  START validity:   {start_validity:.3f}  (target: ~1.0)")
        print(f"  END validity:     {end_validity:.3f}  (target: ~1.0)")
        print(f"  Avg length:       {avg_length:.1f}  (real: {real_avg_length:.1f})")
        
        # Show example traces
        print(f"\nExample generated traces:")
        for i, trace in enumerate(synthetic_traces[:3], 1):
            trace_str = ' → '.join(trace[:8])
            if len(trace) > 8:
                trace_str += ' → ...'
            print(f"  {i}. {trace_str}")
        
        print('='*70 + '\n')
    
    def on_train_end(self, logs=None):
        """Save evaluation history at end of training"""
        if len(self.history['epoch']) == 0:
            return
        
        try:
            import pandas as pd
            df = pd.DataFrame(self.history)
            
            # Save to CSV
            history_path = 'quality_history.csv'
            df.to_csv(history_path, index=False)
            print(f"\n✓ Saved quality metrics history to: {history_path}")
            
            # Create simple plot if matplotlib available
            try:
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle('Generation Quality Over Training')
                
                # Diversity
                axes[0, 0].plot(df['epoch'], df['diversity'], 'b-', marker='o')
                axes[0, 0].set_ylabel('Diversity')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].grid(True)
                axes[0, 0].axhline(y=0.6, color='r', linestyle='--', alpha=0.3, label='Target')
                axes[0, 0].legend()
                
                # Novelty
                axes[0, 1].plot(df['epoch'], df['novelty'], 'g-', marker='o')
                axes[0, 1].set_ylabel('Novelty')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].grid(True)
                axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Target')
                axes[0, 1].legend()
                
                # START validity
                axes[1, 0].plot(df['epoch'], df['start_validity'], 'orange', marker='o')
                axes[1, 0].set_ylabel('START Validity')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].grid(True)
                axes[1, 0].set_ylim([0, 1.1])
                
                # Average length
                axes[1, 1].plot(df['epoch'], df['avg_length'], 'purple', marker='o')
                axes[1, 1].set_ylabel('Avg Trace Length')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].grid(True)
                
                plt.tight_layout()
                plot_path = 'quality_history.png'
                plt.savefig(plot_path, dpi=150)
                print(f"✓ Saved quality plot to: {plot_path}")
                plt.close()
                
            except ImportError:
                pass  # matplotlib not available
                
        except Exception as e:
            print(f"Warning: Could not save quality history: {e}")
