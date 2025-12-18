"""
Visualization utilities for the Pinocchio Vector Test.

Creates plots for truth probe analysis, score distributions,
layer-wise heatmaps, and ROC curves.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_score_distributions(
    scheming_scores: np.ndarray,
    honest_scores: np.ndarray,
    hallucination_scores: Optional[np.ndarray] = None,
    layer: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot overlapping distributions of truth scores.

    Args:
        scheming_scores: Scores for scheming responses
        honest_scores: Scores for honest responses
        hallucination_scores: Scores for hallucinations (optional)
        layer: Layer number for title
        figsize: Figure size
        title: Custom title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # KDE plots
    sns.kdeplot(honest_scores, ax=ax, label="Honest Truths", color="green", fill=True, alpha=0.3)
    sns.kdeplot(scheming_scores, ax=ax, label="Scheming Lies", color="red", fill=True, alpha=0.3)

    if hallucination_scores is not None:
        sns.kdeplot(hallucination_scores, ax=ax, label="Hallucinations", color="orange", fill=True, alpha=0.3)

    # Add vertical lines for means
    ax.axvline(np.mean(honest_scores), color="green", linestyle="--", linewidth=2, label=f"Honest mean: {np.mean(honest_scores):.3f}")
    ax.axvline(np.mean(scheming_scores), color="red", linestyle="--", linewidth=2, label=f"Scheming mean: {np.mean(scheming_scores):.3f}")

    if hallucination_scores is not None:
        ax.axvline(np.mean(hallucination_scores), color="orange", linestyle="--", linewidth=2)

    # Compute d-prime for annotation
    dprime = (np.mean(honest_scores) - np.mean(scheming_scores)) / np.sqrt(0.5 * (np.var(honest_scores) + np.var(scheming_scores)))

    # Title and labels
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif layer is not None:
        ax.set_title(f"Truth Score Distribution (Layer {layer})\nd' = {dprime:.3f}", fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"Truth Score Distribution\nd' = {dprime:.3f}", fontsize=14, fontweight='bold')

    ax.set_xlabel("Truth Score (projection onto truth vector)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def plot_layerwise_heatmap(
    layer_metrics: Dict[int, dict],
    metric_name: str = "dprime_sch_vs_hon",
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot heatmap of metrics across layers.

    Args:
        layer_metrics: Dict from layer_discriminability_analysis
        metric_name: Which metric to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    layers = sorted(layer_metrics.keys())
    values = [layer_metrics[l].get(metric_name, 0) for l in layers]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap data
    data = np.array(values).reshape(1, -1)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")

    # Labels
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_yticks([])
    ax.set_xlabel("Layer", fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label(metric_name.replace("_", " ").title(), fontsize=10)

    # Annotate values
    for i, v in enumerate(values):
        ax.text(i, 0, f"{v:.2f}", ha="center", va="center", fontsize=9, color="black" if abs(v) < 1 else "white")

    ax.set_title(f"Layer-wise {metric_name.replace('_', ' ').title()}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_dprime_by_layer(
    layer_metrics: Dict[int, dict],
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot d-prime across layers as line plot.

    Args:
        layer_metrics: Dict with layer -> metrics
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    layers = sorted(layer_metrics.keys())
    dprime_sch_hon = [layer_metrics[l].get("dprime_sch_vs_hon", 0) for l in layers]
    dprime_sch_hal = [layer_metrics[l].get("dprime_sch_vs_hal", 0) for l in layers]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(layers, dprime_sch_hon, marker='o', linewidth=2, markersize=8, label="Scheming vs Honest", color="blue")

    if any(dprime_sch_hal):
        ax.plot(layers, dprime_sch_hal, marker='s', linewidth=2, markersize=8, label="Scheming vs Hallucination", color="purple")

    # Highlight best layer
    best_layer = layers[np.argmax(np.abs(dprime_sch_hon))]
    ax.axvline(best_layer, color="gray", linestyle=":", linewidth=2, alpha=0.5, label=f"Best layer: {best_layer}")

    # Reference lines
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(2, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("d' (Discriminability)", fontsize=12)
    ax.set_title("Discriminability Across Layers", fontsize=14, fontweight='bold')
    ax.legend()

    # Add annotations for reference lines
    ax.text(layers[-1] + 0.5, 1, "Good", fontsize=9, color="gray")
    ax.text(layers[-1] + 0.5, 2, "Excellent", fontsize=9, color="gray")

    plt.tight_layout()
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    layer: Optional[int] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot ROC curve with AUC.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve
        layer: Layer number for title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.3f})", color="blue")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)

    title = "ROC Curve: Honest vs Scheming Classification"
    if layer is not None:
        title += f" (Layer {layer})"
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_truth_vector_components(
    truth_vector: np.ndarray,
    top_k: int = 50,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot the most important components of the truth vector.

    Args:
        truth_vector: Truth direction vector [d_model]
        top_k: Number of top components to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Get top positive and negative components
    indices = np.arange(len(truth_vector))
    sorted_idx = np.argsort(np.abs(truth_vector))[::-1][:top_k]

    values = truth_vector[sorted_idx]
    positions = indices[sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["green" if v > 0 else "red" for v in values]
    ax.bar(range(len(values)), values, color=colors)

    ax.set_xlabel("Component Rank (sorted by magnitude)", fontsize=12)
    ax.set_ylabel("Weight in Truth Vector", fontsize=12)
    ax.set_title(f"Top {top_k} Components of Truth Vector", fontsize=14, fontweight='bold')

    ax.axhline(0, color="black", linewidth=0.5)

    # Add component indices as labels
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(p) for p in positions], rotation=90, fontsize=7)

    plt.tight_layout()
    return fig


def plot_activation_comparison(
    honest_activation: np.ndarray,
    scheming_activation: np.ndarray,
    truth_vector: np.ndarray,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Compare honest and scheming activations projected onto truth vector.

    Args:
        honest_activation: Activation from honest response
        scheming_activation: Activation from scheming response
        truth_vector: Truth direction vector
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Top: Component-wise comparison
    ax1 = axes[0]
    ax1.plot(honest_activation, alpha=0.7, label="Honest", color="green")
    ax1.plot(scheming_activation, alpha=0.7, label="Scheming", color="red")
    ax1.set_xlabel("Hidden Dimension")
    ax1.set_ylabel("Activation Value")
    ax1.set_title("Raw Activations Comparison")
    ax1.legend()

    # Bottom: Difference weighted by truth vector
    ax2 = axes[1]
    diff = honest_activation - scheming_activation
    weighted_diff = diff * truth_vector

    ax2.bar(range(len(weighted_diff)), weighted_diff, alpha=0.7, color="blue", width=1.0)
    ax2.set_xlabel("Hidden Dimension")
    ax2.set_ylabel("Weighted Difference")
    ax2.set_title("(Honest - Scheming) × Truth Vector")

    # Annotate total projection
    honest_proj = np.dot(honest_activation, truth_vector)
    scheming_proj = np.dot(scheming_activation, truth_vector)
    ax2.axhline(0, color="black", linewidth=0.5)

    ax2.text(0.02, 0.98, f"Honest projection: {honest_proj:.3f}\nScheming projection: {scheming_proj:.3f}\nDifference: {honest_proj - scheming_proj:.3f}",
             transform=ax2.transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def create_summary_figure(
    scheming_scores: np.ndarray,
    honest_scores: np.ndarray,
    hallucination_scores: Optional[np.ndarray],
    layer_metrics: Dict[int, dict],
    best_layer: int,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create a comprehensive summary figure with all key results.

    Args:
        scheming_scores: Scores at best layer
        honest_scores: Scores at best layer
        hallucination_scores: Hallucination scores at best layer
        layer_metrics: All layer metrics
        best_layer: Best performing layer
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # Grid: 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Score distributions (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.kdeplot(honest_scores, ax=ax1, label="Honest", color="green", fill=True, alpha=0.3)
    sns.kdeplot(scheming_scores, ax=ax1, label="Scheming", color="red", fill=True, alpha=0.3)
    if hallucination_scores is not None:
        sns.kdeplot(hallucination_scores, ax=ax1, label="Hallucination", color="orange", fill=True, alpha=0.3)
    ax1.axvline(np.mean(honest_scores), color="green", linestyle="--")
    ax1.axvline(np.mean(scheming_scores), color="red", linestyle="--")
    ax1.set_title(f"Score Distributions (Layer {best_layer})")
    ax1.set_xlabel("Truth Score")
    ax1.legend()

    # 2. D-prime across layers (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    layers = sorted(layer_metrics.keys())
    dprimes = [layer_metrics[l].get("dprime_sch_vs_hon", 0) for l in layers]
    ax2.plot(layers, dprimes, marker='o', linewidth=2, color="blue")
    ax2.axvline(best_layer, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("d'")
    ax2.set_title("Discriminability Across Layers")

    # 3. Box plot comparison (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    data_to_plot = [honest_scores, scheming_scores]
    labels = ["Honest", "Scheming"]
    colors_box = ["green", "red"]
    if hallucination_scores is not None:
        data_to_plot.append(hallucination_scores)
        labels.append("Hallucination")
        colors_box.append("orange")

    bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax3.set_ylabel("Truth Score")
    ax3.set_title("Score Comparison")

    # 4. Summary statistics (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    dprime = (np.mean(honest_scores) - np.mean(scheming_scores)) / np.sqrt(0.5 * (np.var(honest_scores) + np.var(scheming_scores)))
    cohens_d = (np.mean(scheming_scores) - np.mean(honest_scores)) / np.sqrt(0.5 * (np.var(honest_scores, ddof=1) + np.var(scheming_scores, ddof=1)))
    _, p_value = stats.ttest_ind(scheming_scores, honest_scores)

    summary_text = f"""
    SUMMARY STATISTICS (Layer {best_layer})
    ════════════════════════════════════

    Honest Responses:
        Mean: {np.mean(honest_scores):.4f}
        Std:  {np.std(honest_scores):.4f}
        N:    {len(honest_scores)}

    Scheming Responses:
        Mean: {np.mean(scheming_scores):.4f}
        Std:  {np.std(scheming_scores):.4f}
        N:    {len(scheming_scores)}

    ════════════════════════════════════

    Effect Sizes:
        d' (discriminability): {dprime:.3f}
        Cohen's d: {cohens_d:.3f}
        p-value: {p_value:.4f}

    Interpretation:
        {"Significant" if p_value < 0.05 else "Not significant"} at α=0.05
        {"Deceptive Alignment detected" if dprime > 1 and np.mean(scheming_scores) < np.mean(honest_scores) else "Result inconclusive"}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle("Pinocchio Vector Test Results", fontsize=16, fontweight='bold')

    return fig


def plot_method_comparison(
    method_results: Dict[str, Dict],
    metric: str = "separation",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot comparison of probe methods showing best performance for each.

    Args:
        method_results: Dict from compare_probe_methods {method: {layer: metrics}}
        metric: Which metric to compare ('separation', 'accuracy', 'balanced_accuracy')
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    methods = list(method_results.keys())
    best_values = []
    best_layers = []
    all_layer_data = {}

    for method in methods:
        layer_results = method_results[method]
        best_layer = max(layer_results.keys(), key=lambda l: layer_results[l][metric])
        best_values.append(layer_results[best_layer][metric])
        best_layers.append(best_layer)
        all_layer_data[method] = {l: layer_results[l][metric] for l in sorted(layer_results.keys())}

    # Left: Bar chart of best performance
    ax1 = axes[0]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax1.bar(methods, best_values, color=colors[:len(methods)])

    # Add value labels on bars
    for bar, val, layer in zip(bars, best_values, best_layers):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}\n(L{layer})', ha='center', va='bottom', fontsize=10)

    ax1.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax1.set_title(f'Best {metric.replace("_", " ").title()} by Method', fontsize=14, fontweight='bold')
    ax1.axhline(0, color='black', linewidth=0.5)

    # Right: Line plot across layers
    ax2 = axes[1]
    for method, color in zip(methods, colors):
        layers = sorted(all_layer_data[method].keys())
        values = [all_layer_data[method][l] for l in layers]
        ax2.plot(layers, values, marker='o', label=method, color=color, linewidth=2, markersize=6)

    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax2.set_title(f'{metric.replace("_", " ").title()} Across Layers', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_entropy_by_layer(
    entropy_results: Dict[int, Dict],
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot entropy comparison across layers.

    Shows whether deceptive responses have higher entropy (internal conflict).

    Args:
        entropy_results: Output from compare_entropy()
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    layers = sorted(entropy_results.keys())
    sch_entropy = [entropy_results[l]["scheming_mean"] for l in layers]
    hon_entropy = [entropy_results[l]["honest_mean"] for l in layers]
    significant = [entropy_results[l]["significant"] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Line plot of entropy by layer
    ax1 = axes[0]
    ax1.plot(layers, sch_entropy, marker='o', linewidth=2, markersize=8,
             label="Scheming", color="red")
    ax1.plot(layers, hon_entropy, marker='s', linewidth=2, markersize=8,
             label="Honest", color="green")

    # Highlight significant layers
    for i, (layer, sig) in enumerate(zip(layers, significant)):
        if sig:
            ax1.axvspan(layer - 0.3, layer + 0.3, alpha=0.2, color='yellow')

    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Activation Entropy", fontsize=12)
    ax1.set_title("Entropy Across Layers\n(Yellow = Significant Difference)", fontsize=14, fontweight='bold')
    ax1.legend()

    # Right: Bar chart of entropy difference
    ax2 = axes[1]
    differences = [entropy_results[l]["difference"] for l in layers]
    colors = ['red' if d > 0 else 'green' for d in differences]
    bars = ax2.bar(layers, differences, color=colors, alpha=0.7)

    # Mark significant bars
    for i, (bar, sig) in enumerate(zip(bars, significant)):
        if sig:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    '*', ha='center', fontsize=14, fontweight='bold')

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Entropy Difference (Scheming - Honest)", fontsize=12)
    ax2.set_title("Entropy Difference by Layer\n(* = p < 0.05)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_consistency_comparison(
    scheming_consistency: List[float],
    honest_consistency: List[float],
    comparison_results: Optional[Dict] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot consistency distributions for scheming vs honest prompts.

    Higher consistency = more stable activations.
    Hypothesis: Deception requires more "searching" → lower consistency.

    Args:
        scheming_consistency: List of consistency scores for scheming prompts
        honest_consistency: List of consistency scores for honest prompts
        comparison_results: Optional dict with statistical results
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Box plot comparison
    ax1 = axes[0]
    bp = ax1.boxplot([honest_consistency, scheming_consistency],
                     labels=["Honest", "Scheming"],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.5)

    ax1.set_ylabel("Consistency (Cosine Similarity)", fontsize=12)
    ax1.set_title("Activation Consistency", fontsize=14, fontweight='bold')

    # Add mean markers
    ax1.scatter([1], [np.mean(honest_consistency)], marker='D', s=100, color='darkgreen', zorder=5, label='Mean')
    ax1.scatter([2], [np.mean(scheming_consistency)], marker='D', s=100, color='darkred', zorder=5)

    # Right: Overlapping distributions
    ax2 = axes[1]
    sns.kdeplot(honest_consistency, ax=ax2, label="Honest", color="green", fill=True, alpha=0.3)
    sns.kdeplot(scheming_consistency, ax=ax2, label="Scheming", color="red", fill=True, alpha=0.3)

    ax2.axvline(np.mean(honest_consistency), color="green", linestyle="--", linewidth=2)
    ax2.axvline(np.mean(scheming_consistency), color="red", linestyle="--", linewidth=2)

    ax2.set_xlabel("Consistency Score", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Consistency Distributions", fontsize=14, fontweight='bold')
    ax2.legend()

    # Add statistics text if provided
    if comparison_results:
        stats_text = (f"Honest: {comparison_results['honest_mean']:.4f} ± {comparison_results['honest_std']:.4f}\n"
                     f"Scheming: {comparison_results['scheming_mean']:.4f} ± {comparison_results['scheming_std']:.4f}\n"
                     f"p-value: {comparison_results['p_value']:.4f}\n"
                     f"Cohen's d: {comparison_results['cohens_d']:.3f}")
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_2x2_factorial(
    scores_by_condition: Dict[str, np.ndarray],
    metric_name: str = "Truth Score",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot 2x2 factorial design results for social friction experiment.

    Conditions: uncomfortable_truth, comfortable_truth, comfortable_lie, uncomfortable_lie

    Args:
        scores_by_condition: Dict mapping condition name to scores array
        metric_name: Name of the metric being plotted
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Expected condition names
    conditions = ["comfortable_truth", "uncomfortable_truth", "comfortable_lie", "uncomfortable_lie"]

    # Check which conditions are available
    available = [c for c in conditions if c in scores_by_condition]

    if len(available) < 2:
        ax = axes[0]
        ax.text(0.5, 0.5, "Insufficient data for 2x2 analysis",
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        axes[1].axis('off')
        return fig

    # Left: Bar plot with means
    ax1 = axes[0]
    means = [np.mean(scores_by_condition[c]) for c in available]
    stds = [np.std(scores_by_condition[c]) for c in available]

    colors = {
        "comfortable_truth": "green",
        "uncomfortable_truth": "lightgreen",
        "comfortable_lie": "red",
        "uncomfortable_lie": "lightcoral"
    }

    bar_colors = [colors.get(c, "gray") for c in available]
    bars = ax1.bar(range(len(available)), means, yerr=stds, capsize=5, color=bar_colors, alpha=0.7)

    ax1.set_xticks(range(len(available)))
    ax1.set_xticklabels([c.replace("_", "\n") for c in available], fontsize=10)
    ax1.set_ylabel(metric_name, fontsize=12)
    ax1.set_title("Mean Scores by Condition", fontsize=14, fontweight='bold')
    ax1.axhline(0, color='black', linewidth=0.5)

    # Right: Interaction plot (if all 4 conditions present)
    ax2 = axes[1]

    if len(available) == 4:
        # Create interaction plot
        truth_levels = ["Truth", "Lie"]
        valence_levels = ["Comfortable", "Uncomfortable"]

        truth_means = [
            np.mean(scores_by_condition["comfortable_truth"]),
            np.mean(scores_by_condition["uncomfortable_truth"])
        ]
        lie_means = [
            np.mean(scores_by_condition["comfortable_lie"]),
            np.mean(scores_by_condition["uncomfortable_lie"])
        ]

        ax2.plot([0, 1], truth_means, marker='o', linewidth=2, markersize=10,
                label="Truthful", color="green")
        ax2.plot([0, 1], lie_means, marker='s', linewidth=2, markersize=10,
                label="Deceptive", color="red")

        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(valence_levels)
        ax2.set_xlabel("Social Valence", fontsize=12)
        ax2.set_ylabel(metric_name, fontsize=12)
        ax2.set_title("Interaction: Truth × Valence", fontsize=14, fontweight='bold')
        ax2.legend()

        # Check for interaction
        truth_diff = truth_means[1] - truth_means[0]  # Effect of valence on truth
        lie_diff = lie_means[1] - lie_means[0]  # Effect of valence on lies
        interaction = truth_diff - lie_diff

        ax2.text(0.02, 0.02, f"Interaction: {interaction:.3f}",
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, "Need all 4 conditions\nfor interaction plot",
                ha='center', va='center', fontsize=12)
        ax2.axis('off')

    plt.tight_layout()
    return fig


def save_all_figures(
    figures: Dict[str, plt.Figure],
    output_dir: str,
    format: str = "png",
    dpi: int = 150,
) -> None:
    """
    Save all figures to disk.

    Args:
        figures: Dict mapping name to figure
        output_dir: Output directory path
        format: Image format (png, pdf, svg)
        dpi: Resolution
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for name, fig in figures.items():
        path = os.path.join(output_dir, f"{name}.{format}")
        fig.savefig(path, format=format, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {path}")
