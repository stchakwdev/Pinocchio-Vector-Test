"""
Pinocchio Vector Test - Mechanistic Interpretability Experiment

Detects if a language model "knows" it's lying when scheming by extracting
a "truth direction" from residual stream activations.
"""

from .model_utils import get_device, load_model, clear_memory
from .data_generation import (
    FactualPair,
    SchemingPrompt,
    generate_factual_pairs,
    generate_scheming_prompts,
    generate_hallucination_prompts,
)
from .activation_extraction import (
    extract_residual_activations,
    batch_extract_activations,
    find_token_position,
)
from .truth_probe import (
    compute_truth_direction,
    project_onto_truth,
    evaluate_probe_accuracy,
)
from .statistics import (
    compute_dprime,
    compute_cohens_d,
    hypothesis_testing,
    layer_discriminability_analysis,
)
from .visualization import (
    plot_score_distributions,
    plot_layerwise_heatmap,
    plot_dprime_by_layer,
    plot_roc_curve,
)

__version__ = "0.1.0"
__all__ = [
    # Model utilities
    "get_device",
    "load_model",
    "clear_memory",
    # Data generation
    "FactualPair",
    "SchemingPrompt",
    "generate_factual_pairs",
    "generate_scheming_prompts",
    "generate_hallucination_prompts",
    # Activation extraction
    "extract_residual_activations",
    "batch_extract_activations",
    "find_token_position",
    # Truth probe
    "compute_truth_direction",
    "project_onto_truth",
    "evaluate_probe_accuracy",
    # Statistics
    "compute_dprime",
    "compute_cohens_d",
    "hypothesis_testing",
    "layer_discriminability_analysis",
    # Visualization
    "plot_score_distributions",
    "plot_layerwise_heatmap",
    "plot_dprime_by_layer",
    "plot_roc_curve",
]
