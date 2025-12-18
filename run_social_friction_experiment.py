#!/usr/bin/env python3
"""
Social Friction Experiment - v1
Tests whether models show different activations based on anticipated social response.

2x2 Factorial Design:
- Truth dimension: truthful vs deceptive
- Valence dimension: positive (comfortable) vs negative (uncomfortable) social response

Hypothesis: Models show measurable activation differences based on anticipated
social friction, regardless of whether they "know" they're lying.
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.absolute()))

from src.model_utils import get_device, load_model, clear_memory, get_model_info
from src.data_generation import (
    generate_factual_pairs, generate_social_friction_prompts,
    SocialFrictionPrompt
)
from src.activation_extraction import batch_extract_activations
from src.truth_probe import (
    compute_truth_direction, batch_project_onto_truth,
    evaluate_probe_accuracy, get_best_method_and_layer, compare_probe_methods
)
from src.statistics import (
    analyze_2x2_factorial, factorial_summary_table,
    compare_entropy, entropy_summary_table
)
from src.visualization import (
    plot_score_distributions, plot_entropy_by_layer,
    plot_2x2_factorial, save_all_figures
)


def main():
    print("=" * 70)
    print("SOCIAL FRICTION EXPERIMENT - v1")
    print("Testing: Does the model anticipate social consequences?")
    print("=" * 70)

    # ====================
    # SETUP
    # ====================
    DEVICE = get_device()
    print(f"\nUsing device: {DEVICE}")

    if DEVICE == "mps":
        DTYPE = torch.float32
        BATCH_SIZE = 8
    elif DEVICE == "cuda":
        DTYPE = torch.float16
        BATCH_SIZE = 16
    else:
        DTYPE = torch.float32
        BATCH_SIZE = 4

    print(f"Precision: {DTYPE}")
    print(f"Batch size: {BATCH_SIZE}")

    CONFIG = {
        "layers": list(range(10, 22)),  # Pythia-1.4B layers
        "n_pairs_per_category": 25,
        "train_ratio": 0.8,
        "output_dir": Path("./data/results/social_friction"),
    }
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    # ====================
    # LOAD MODEL
    # ====================
    MODEL_NAME = "EleutherAI/pythia-1.4b-deduped"
    print(f"\nLoading {MODEL_NAME}...")

    model = load_model(
        model_name=MODEL_NAME,
        device=DEVICE,
        dtype=DTYPE,
        verbose=True
    )

    model_info = get_model_info(model)
    print("\nModel Configuration:")
    for k, v in model_info.items():
        print(f"  {k}: {v}")

    # ====================
    # PHASE 1: TRAIN TRUTH PROBE (for comparison)
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 1: Truth Probe Training (for comparison)")
    print("=" * 70)

    factual_pairs = generate_factual_pairs(n_per_category=CONFIG["n_pairs_per_category"])
    print(f"\nGenerated {len(factual_pairs)} factual pairs")

    np.random.seed(42)
    indices = np.random.permutation(len(factual_pairs))
    split_idx = int(len(factual_pairs) * CONFIG["train_ratio"])
    train_pairs = [factual_pairs[i] for i in indices[:split_idx]]

    print(f"Training pairs: {len(train_pairs)}")

    print("\nExtracting activations for TRUE statements...")
    true_statements = [p.true_statement for p in train_pairs]
    true_activations = batch_extract_activations(
        model=model,
        texts=true_statements,
        layers=CONFIG["layers"],
        position=-1,
        show_progress=True
    )

    print("Extracting activations for FALSE statements...")
    false_statements = [p.false_statement for p in train_pairs]
    false_activations = batch_extract_activations(
        model=model,
        texts=false_statements,
        layers=CONFIG["layers"],
        position=-1,
        show_progress=True
    )

    # Find best layer using mass_mean (what worked before)
    print("\nComputing truth direction...")
    truth_vectors = compute_truth_direction(
        true_activations, false_activations,
        method="mass_mean", normalize=True
    )

    # Use layer 13 (where entropy was significant) or find best
    best_layer = 13  # From entropy analysis
    print(f"Using layer {best_layer} (significant entropy difference found here)")

    torch.save(truth_vectors, CONFIG["output_dir"] / "truth_vectors.pt")

    clear_memory(DEVICE)

    # ====================
    # PHASE 2: LOAD SOCIAL FRICTION PROMPTS
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 2: Social Friction Prompts")
    print("=" * 70)

    friction_prompts = generate_social_friction_prompts()

    print("\nPrompts per condition:")
    for condition, prompts in friction_prompts.items():
        print(f"  {condition}: {len(prompts)}")

    total_prompts = sum(len(p) for p in friction_prompts.values())
    print(f"Total: {total_prompts} prompts")

    # ====================
    # PHASE 3: EXTRACT ACTIVATIONS FOR EACH CONDITION
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 3: Extract Activations by Condition")
    print("=" * 70)

    activations_by_condition = {}
    scores_by_condition = {}

    for condition, prompts in friction_prompts.items():
        print(f"\nExtracting activations for {condition}...")
        prompt_texts = [p.prompt for p in prompts]

        acts = batch_extract_activations(
            model=model,
            texts=prompt_texts,
            layers=CONFIG["layers"],
            position=-1,
            show_progress=True
        )

        activations_by_condition[condition] = acts

        # Project onto truth vector at best layer
        scores = batch_project_onto_truth(acts[best_layer], truth_vectors[best_layer])
        scores_by_condition[condition] = scores

        print(f"  {condition}: mean={scores.mean():.4f}, std={scores.std():.4f}")

    # ====================
    # PHASE 4: 2x2 FACTORIAL ANALYSIS
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 4: 2x2 Factorial Analysis")
    print("=" * 70)

    factorial_results = analyze_2x2_factorial(scores_by_condition)

    print("\n" + factorial_summary_table(factorial_results))

    # ====================
    # PHASE 5: ENTROPY ANALYSIS BY CONDITION
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 5: Entropy Analysis by Condition")
    print("=" * 70)

    # Compare entropy: truthful vs deceptive
    truth_acts = {
        layer: torch.cat([
            activations_by_condition['uncomfortable_truth'][layer],
            activations_by_condition['comfortable_truth'][layer]
        ], dim=0)
        for layer in CONFIG["layers"]
    }

    lie_acts = {
        layer: torch.cat([
            activations_by_condition['uncomfortable_lie'][layer],
            activations_by_condition['comfortable_lie'][layer]
        ], dim=0)
        for layer in CONFIG["layers"]
    }

    entropy_truth_vs_lie = compare_entropy(lie_acts, truth_acts)

    print("\nEntropy: Lies vs Truths (across all conditions)")
    print("-" * 50)
    sig_layers = []
    for layer in CONFIG["layers"]:
        r = entropy_truth_vs_lie[layer]
        sig_marker = "*" if r["significant"] else " "
        if r["significant"]:
            sig_layers.append(layer)
        print(f"  Layer {layer}: Lie={r['scheming_mean']:.4f}, Truth={r['honest_mean']:.4f}, "
              f"diff={r['difference']:.4f}, p={r['p_value']:.4f} {sig_marker}")

    if sig_layers:
        print(f"\n>>> SIGNIFICANT ENTROPY DIFFERENCES at layers: {sig_layers} <<<")

    # ====================
    # PHASE 6: GENERATE VISUALIZATIONS
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 6: Generating Visualizations")
    print("=" * 70)

    figures = {}

    # 1. 2x2 factorial plot
    print("  Creating 2x2 factorial plot...")
    figures['factorial_2x2'] = plot_2x2_factorial(scores_by_condition, "Truth Score")

    # 2. Entropy by layer
    print("  Creating entropy by layer plot...")
    figures['entropy_truth_vs_lie'] = plot_entropy_by_layer(entropy_truth_vs_lie)

    # 3. Score distributions by condition
    print("  Creating score distributions...")
    # Combine for distribution plot
    all_truth = np.concatenate([
        scores_by_condition['uncomfortable_truth'],
        scores_by_condition['comfortable_truth']
    ])
    all_lies = np.concatenate([
        scores_by_condition['uncomfortable_lie'],
        scores_by_condition['comfortable_lie']
    ])
    figures['truth_vs_lie_dist'] = plot_score_distributions(
        all_lies, all_truth, layer=best_layer
    )

    # Save all figures
    print("\nSaving figures...")
    save_all_figures(figures, str(CONFIG["output_dir"]), format="png", dpi=150)

    # ====================
    # SAVE RESULTS
    # ====================
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        "experiment": "social_friction_v1",
        "model": MODEL_NAME,
        "best_layer": best_layer,
        "factorial_analysis": {
            "means": convert_to_serializable(factorial_results["means"]),
            "main_effect_truth": convert_to_serializable(factorial_results["main_effect_truth"]),
            "main_effect_valence": convert_to_serializable(factorial_results["main_effect_valence"]),
            "interaction": convert_to_serializable(factorial_results["interaction"]),
        },
        "entropy_significant_layers": sig_layers,
        "prompt_counts": {
            condition: len(prompts)
            for condition, prompts in friction_prompts.items()
        }
    }

    with open(CONFIG["output_dir"] / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ====================
    # FINAL SUMMARY
    # ====================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nModel: {MODEL_NAME}")
    print(f"Analysis layer: {best_layer}")

    me_truth = factorial_results["main_effect_truth"]
    me_val = factorial_results["main_effect_valence"]

    print(f"\nMain Effect of TRUTH (truthful vs deceptive):")
    print(f"  p-value: {me_truth['p_value']:.4f} ({'*' if me_truth['significant'] else 'n.s.'})")
    print(f"  Cohen's d: {me_truth['cohens_d']:.3f}")

    print(f"\nMain Effect of VALENCE (positive vs negative):")
    print(f"  p-value: {me_val['p_value']:.4f} ({'*' if me_val['significant'] else 'n.s.'})")
    print(f"  Cohen's d: {me_val['cohens_d']:.3f}")

    print(f"\nInteraction (Truth x Valence):")
    print(f"  Value: {factorial_results['interaction']['value']:.4f}")
    print(f"  {factorial_results['interaction']['interpretation']}")

    # Key findings
    print("\n" + "-" * 40)
    print("KEY FINDINGS:")
    print("-" * 40)

    if me_truth['significant']:
        print(f"  - TRUTH effect detected (p={me_truth['p_value']:.4f})")
    else:
        print("  - No significant TRUTH effect")

    if me_val['significant']:
        print(f"  - VALENCE effect detected (p={me_val['p_value']:.4f})")
        print(f"    Model shows different activations for anticipated social response!")
    else:
        print("  - No significant VALENCE effect")

    if sig_layers:
        print(f"  - Entropy differences found at layers: {sig_layers}")

    print(f"\nResults saved to: {CONFIG['output_dir']}")
    print("Experiment complete!")

    return results


if __name__ == "__main__":
    main()
