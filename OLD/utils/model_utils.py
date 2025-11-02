"""
Model Utilities
Helper functions for model setup and configuration
"""

import os
import numpy as np
import tensorflow as tf
from collections import Counter
from typing import Optional, Tuple, Dict


def find_companion_pnml(xes_path: str) -> Optional[str]:
    """
    Find companion PNML file for an XES file

    Tries common naming patterns:
    - helpdesk_parsed.xes → Helpdesk_parsed_net.pnml
    - helpdesk_parsed.xes → helpdesk_parsed_net.pnml
    - helpdesk_parsed.xes → helpdesk_parsed.pnml

    Args:
        xes_path: Path to XES file

    Returns:
        Path to companion PNML file if found, None otherwise
    """
    base_path = os.path.splitext(xes_path)[0]

    candidate_paths = [
        base_path.replace('_parsed', '_parsed_net') +
        '.pnml',  # _parsed → _parsed_net
        base_path.capitalize().replace('_parsed', '_parsed_net') +
        '.pnml',  # Capitalize first letter
        base_path + '_net.pnml',  # Add _net suffix
        base_path + '.pnml',  # Simple replacement
    ]

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate

    return None


def detect_start_end_activities(traces: list) -> Tuple[Optional[str], Optional[str]]:
    """
    Auto-detect START and END activities from traces

    Uses the most common first and last activities in the dataset.

    Args:
        traces: List of traces (each trace is a list of activities)

    Returns:
        Tuple (start_activity, end_activity)
    """
    if not traces:
        return None, None

    # Get first and last activities
    first_activities = [t[0] for t in traces if len(t) > 0]
    last_activities = [t[-1] for t in traces if len(t) > 0]

    start_activity = None
    end_activity = None

    if first_activities:
        start_activity = Counter(first_activities).most_common(1)[0][0]

    if last_activities:
        end_activity = Counter(last_activities).most_common(1)[0][0]

    return start_activity, end_activity


def load_reference_model(data_file: str, reward_function_class, log_func=None):
    """
    Load reference Petri net model for fitness/conformance evaluation

    Automatically finds companion PNML file for XES inputs.

    Args:
        data_file: Path to data file (XES)
        reward_function_class: ProcessRewardFunction class (for load_petri_net_model)
        log_func: Optional logging function

    Returns:
        Reference model (net, im, fm) or None if not found
    """
    log = log_func or (lambda msg: print(msg))

    reference_model = None
    reference_model_path = None

    file_ext = os.path.splitext(data_file)[1].lower()

    # If input is XES, try to find companion PNML file
    if file_ext == '.xes':
        reference_model_path = find_companion_pnml(data_file)

        if reference_model_path:
            log(f"Found companion Petri net: {reference_model_path}")
        else:
            log("⚠ No companion PNML file found")

    # Load reference model if path found
    if reference_model_path:
        log("Loading Petri net as reference model for fitness/conformance...")
        reference_model = reward_function_class.load_petri_net_model(
            reference_model_path)

        if reference_model:
            log("Reference model loaded")
        else:
            log("⚠ Failed to load reference model - using simplified metrics")
    else:
        log("⚠ No reference model available - using simplified fitness/conformance metrics")

    return reference_model


def build_and_initialize_model(model, z_dim: int):
    """
    Build model by creating weights with sample input

    Args:
        model: ProcessGAN model instance
        z_dim: Latent dimension size

    Returns:
        None (model is initialized in-place)
    """
    import tensorflow as tf

    # Build models by calling them with sample input to create weights
    sample_z = tf.random.normal((1, z_dim))
    sample_adj, sample_nodes = model.generator(sample_z, training=False)
    _ = model.discriminator(sample_adj, sample_nodes, training=False)
    _ = model.value_network(sample_adj, sample_nodes, training=False)


def count_model_parameters(model) -> Dict[str, int]:
    """
    Count trainable parameters in ProcessGAN model

    Args:
        model: ProcessGAN model instance

    Returns:
        Dictionary with parameter counts:
        - 'generator': Generator parameters
        - 'discriminator': Discriminator parameters
        - 'value_network': Value network parameters
        - 'total': Total parameters
    """
    total_g = sum([np.prod(v.shape)
                  for v in model.generator.trainable_variables])
    total_d = sum([np.prod(v.shape)
                  for v in model.discriminator.trainable_variables])
    total_v = sum([np.prod(v.shape)
                  for v in model.value_network.trainable_variables])

    return {
        'generator': int(total_g),
        'discriminator': int(total_d),
        'value_network': int(total_v),
        'total': int(total_g + total_d + total_v)
    }
