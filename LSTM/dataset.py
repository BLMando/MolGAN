import numpy as np
import tensorflow as tf

class ProcessTraceDataset:
    """
    Dataset handler for process traces
    """

    def __init__(self, traces, activity_to_idx, max_length,
                 pad_token='<PAD>', start_token='START', end_token='6'):
        """
        Parameters:
        -----------
        traces: list of lists
            Each inner list is a sequence of activity names
            Example: [['Start', 'A', 'B', 'End'], ['Start', 'C', 'End'], ...]
        activity_to_idx: dict
            Mapping from activity name to integer index
        max_length: int
            Maximum trace length (for padding/truncating)
        """
        self.traces = traces
        self.activity_to_idx = activity_to_idx
        self.idx_to_activity = {v: k for k, v in activity_to_idx.items()}
        self.max_length = max_length
        self.num_activities = len(activity_to_idx)

        # Special token indices
        self.pad_idx = activity_to_idx[pad_token]
        self.start_idx = activity_to_idx[start_token]
        self.end_idx = activity_to_idx[end_token]

        # Preprocess traces
        self.processed_traces = self._preprocess_traces()

        # Compute activity frequencies for constraint matching
        self.activity_frequencies = self._compute_frequencies()

    def _preprocess_traces(self):
        """
        Convert traces to integer sequences and pad/truncate
        """
        processed = []
        for trace in self.traces:
            # Convert to indices
            indices = [self.activity_to_idx.get(
                act, self.pad_idx) for act in trace]

            # Pad or truncate
            if len(indices) < self.max_length:
                indices += [self.pad_idx] * (self.max_length - len(indices))
            else:
                indices = indices[:self.max_length]

            processed.append(indices)

        return np.array(processed, dtype=np.int32)

    def _compute_frequencies(self):
        """
        Compute normalized activity frequencies
        """
        counts = np.bincount(self.processed_traces.flatten(),
                             minlength=self.num_activities)
        frequencies = counts / counts.sum()
        return frequencies.astype(np.float32)

    def get_tf_dataset(self, batch_size, shuffle=True):
        """
        Create TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.processed_traces)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.traces))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
