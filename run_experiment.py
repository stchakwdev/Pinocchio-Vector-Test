#!/usr/bin/env python3
"""
Pinocchio Vector Test - Improved Experiment v4
Detects if a model internally represents deception differently from honest responses.

Changes in v4:
- Added entropy probe (tests internal conflict hypothesis)
- Switched to Pythia-1.4B (faster iteration)
- Added debug output for probe accuracy
- Added visualization generation
"""

import os
import sys
import json
import warnings
from pathlib import Path

# Numerical
import numpy as np
import pandas as pd
from scipy import stats

# PyTorch
import torch

# Progress bars
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Import our modules
from src.model_utils import get_device, load_model, clear_memory, get_model_info
from src.data_generation import (
    generate_factual_pairs, generate_scheming_prompts,
    generate_honest_prompts, generate_hallucination_prompts,
    generate_negated_pairs
)
from src.activation_extraction import batch_extract_activations
from src.truth_probe import (
    compute_truth_direction, batch_project_onto_truth,
    evaluate_probe_accuracy, compare_probe_methods, get_best_method_and_layer
)
from src.statistics import (
    hypothesis_testing, layer_discriminability_analysis, compute_roc_curve, interpret_results,
    compare_entropy, entropy_summary_table
)
from src.visualization import (
    plot_score_distributions, plot_dprime_by_layer, plot_roc_curve as plot_roc,
    create_summary_figure, plot_method_comparison, save_all_figures,
    plot_entropy_by_layer
)


def debug_activations(activations, name=""):
    """Print debug info about activations."""
    for layer, acts in activations.items():
        print(f"  {name} Layer {layer}: shape={acts.shape}, "
              f"mean={acts.mean():.4f}, std={acts.std():.4f}, "
              f"min={acts.min():.4f}, max={acts.max():.4f}")
        break  # Just show first layer for brevity


def main():
    print("=" * 70)
    print("PINOCCHIO VECTOR TEST - v3 (Pythia-1.4B + Visualizations)")
    print("=" * 70)

    # ====================
    # SETUP
    # ====================
    DEVICE = get_device()
    print(f"\nUsing device: {DEVICE}")

    if DEVICE == "mps":
        DTYPE = torch.float32  # Use float32 for better precision on MPS
        BATCH_SIZE = 8
    elif DEVICE == "cuda":
        DTYPE = torch.float16
        BATCH_SIZE = 16
    else:
        DTYPE = torch.float32
        BATCH_SIZE = 4

    print(f"Precision: {DTYPE}")
    print(f"Batch size: {BATCH_SIZE}")

    # Configuration - Pythia-1.4B has 24 layers
    CONFIG = {
        "layers": list(range(10, 22)),  # Focus on middle-late layers for 1.4B
        "n_pairs_per_category": 25,
        "train_ratio": 0.8,
        "probe_methods": ["difference_in_means", "pca", "logistic", "mass_mean"],
        "output_dir": Path("./data/results"),
    }
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    # ====================
    # LOAD MODEL - Pythia-1.4B (faster!)
    # ====================
    MODEL_NAME = "EleutherAI/pythia-1.4b-deduped"
    print(f"\nLoading {MODEL_NAME}...")
    print("This is much faster than 6.9B!")

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
    # PHASE 1: TRUTH DIRECTION EXTRACTION
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 1: Truth Direction Extraction")
    print("=" * 70)

    factual_pairs = generate_factual_pairs(n_per_category=CONFIG["n_pairs_per_category"])
    print(f"\nGenerated {len(factual_pairs)} factual pairs")

    # Split train/test
    np.random.seed(42)
    indices = np.random.permutation(len(factual_pairs))
    split_idx = int(len(factual_pairs) * CONFIG["train_ratio"])
    train_pairs = [factual_pairs[i] for i in indices[:split_idx]]
    test_pairs = [factual_pairs[i] for i in indices[split_idx:]]
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")

    # Extract activations
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

    # DEBUG: Check activations
    print("\n>>> DEBUG: Activation Statistics <<<")
    debug_activations(true_activations, "TRUE")
    debug_activations(false_activations, "FALSE")

    # Check if true/false have different distributions
    for layer in [CONFIG["layers"][0], CONFIG["layers"][-1]]:
        true_mean = true_activations[layer].mean().item()
        false_mean = false_activations[layer].mean().item()
        raw_sep = true_mean - false_mean
        print(f"  Layer {layer} raw separation: {raw_sep:.6f}")

    # Extract test activations
    print("\nExtracting test set activations...")
    test_true = [p.true_statement for p in test_pairs]
    test_false = [p.false_statement for p in test_pairs]
    test_true_acts = batch_extract_activations(model, test_true, CONFIG["layers"], show_progress=True)
    test_false_acts = batch_extract_activations(model, test_false, CONFIG["layers"], show_progress=True)

    # ====================
    # PHASE 1.5: COMPARE PROBE METHODS
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 1.5: Comparing Probe Methods")
    print("=" * 70)

    method_comparison = compare_probe_methods(
        true_activations, false_activations,
        test_true_acts, test_false_acts,
        methods=CONFIG["probe_methods"]
    )

    best_method, best_layer, best_sep = get_best_method_and_layer(method_comparison, metric="separation")

    print(f"\nMethod Comparison:")
    print("-" * 60)
    for method, layer_results in method_comparison.items():
        best_l = max(layer_results.keys(), key=lambda l: layer_results[l]["separation"])
        m = layer_results[best_l]
        print(f"  {method:20s}: acc={m['accuracy']:.3f}, sep={m['separation']:.4f} (layer {best_l})")

    print(f"\n*** BEST: {best_method} at layer {best_layer} (sep={best_sep:.4f}) ***")

    # Use best method
    truth_vectors = compute_truth_direction(
        true_activations, false_activations,
        method=best_method, normalize=True
    )
    torch.save(truth_vectors, CONFIG["output_dir"] / "truth_vectors.pt")

    # Evaluate on test set
    test_metrics = evaluate_probe_accuracy(
        truth_vectors[best_layer],
        test_true_acts[best_layer],
        test_false_acts[best_layer]
    )
    print(f"\n>>> TEST SET PERFORMANCE (Layer {best_layer}) <<<")
    print(f"  Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"  Separation: {test_metrics['separation']:.4f}")
    print(f"  True mean: {test_metrics['true_mean']:.4f}, False mean: {test_metrics['false_mean']:.4f}")

    clear_memory(DEVICE)

    # ====================
    # PHASE 2: SCHEMING SETUP
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 2: Scheming Setup")
    print("=" * 70)

    scheming_prompts = generate_scheming_prompts()
    honest_prompts = generate_honest_prompts()
    hallucination_prompts = generate_hallucination_prompts()

    print(f"\nScheming prompts: {len(scheming_prompts)}")
    print(f"Honest prompts: {len(honest_prompts)}")
    print(f"Hallucination prompts: {len(hallucination_prompts)}")

    # ====================
    # PHASE 3: LIE DETECTION TEST
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 3: Lie Detection Test")
    print("=" * 70)

    print("\nExtracting activations...")
    sch_acts = batch_extract_activations(model, [p.prompt for p in scheming_prompts], CONFIG["layers"], show_progress=True)
    hon_acts = batch_extract_activations(model, [p.prompt for p in honest_prompts], CONFIG["layers"], show_progress=True)
    hal_acts = batch_extract_activations(model, [p.prompt for p in hallucination_prompts], CONFIG["layers"], show_progress=True)

    # Project onto truth vector
    sch_scores_by_layer = {}
    hon_scores_by_layer = {}
    hal_scores_by_layer = {}

    for layer in CONFIG["layers"]:
        sch_scores_by_layer[layer] = batch_project_onto_truth(sch_acts[layer], truth_vectors[layer])
        hon_scores_by_layer[layer] = batch_project_onto_truth(hon_acts[layer], truth_vectors[layer])
        hal_scores_by_layer[layer] = batch_project_onto_truth(hal_acts[layer], truth_vectors[layer])

    sch_scores = sch_scores_by_layer[best_layer]
    hon_scores = hon_scores_by_layer[best_layer]
    hal_scores = hal_scores_by_layer[best_layer]

    print(f"\nTruth Scores at Layer {best_layer}:")
    print("=" * 50)
    print(f"SCHEMING (n={len(sch_scores)}): mean={sch_scores.mean():.4f}, std={sch_scores.std():.4f}")
    print(f"HONEST (n={len(hon_scores)}): mean={hon_scores.mean():.4f}, std={hon_scores.std():.4f}")
    print(f"HALLUCINATION (n={len(hal_scores)}): mean={hal_scores.mean():.4f}, std={hal_scores.std():.4f}")

    separation = hon_scores.mean() - sch_scores.mean()
    print(f"\nSeparation (Honest - Scheming): {separation:.4f}")
    print(f"Direction: {'Honest > Scheming' if separation > 0 else 'Scheming > Honest (unexpected)'}")

    # ====================
    # PHASE 4: STATISTICAL ANALYSIS
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 4: Statistical Analysis")
    print("=" * 70)

    test_results = hypothesis_testing(sch_scores, hon_scores, hal_scores, alpha=0.05)
    svh = test_results["scheming_vs_honest"]

    print(f"\nScheming vs Honest:")
    print(f"  t-stat: {svh['t_test']['statistic']:.4f}, p-value: {svh['t_test']['p_value']:.6f}")
    print(f"  d': {svh['dprime']:.4f}, Cohen's d: {svh['cohens_d']:.4f}")
    print(f"  Significant: {svh['significant']}")

    # ROC
    fpr, tpr, auc = compute_roc_curve(hon_scores, sch_scores)
    print(f"\nROC AUC: {auc:.4f}")

    # Layer analysis for visualizations
    layer_metrics = {}
    for layer in CONFIG["layers"]:
        sch = sch_scores_by_layer[layer]
        hon = hon_scores_by_layer[layer]
        hal = hal_scores_by_layer[layer]
        dprime_sh = (hon.mean() - sch.mean()) / np.sqrt(0.5 * (hon.var() + sch.var()))
        dprime_shhal = (hal.mean() - sch.mean()) / np.sqrt(0.5 * (hal.var() + sch.var())) if len(hal) > 0 else 0
        layer_metrics[layer] = {
            "dprime_sch_vs_hon": dprime_sh,
            "dprime_sch_vs_hal": dprime_shhal
        }

    # ====================
    # PHASE 4.5: ENTROPY ANALYSIS (Social Friction Detection)
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 4.5: Entropy Analysis (Internal Conflict Detection)")
    print("=" * 70)
    print("\nHypothesis: Deception creates internal conflict → higher entropy")

    # Compare entropy between scheming and honest activations
    entropy_results = compare_entropy(sch_acts, hon_acts, hal_acts)

    # Find layers with significant entropy differences
    sig_entropy_layers = [l for l, r in entropy_results.items() if r["significant"]]

    print(f"\nEntropy Analysis Results:")
    print("-" * 50)
    for layer in CONFIG["layers"]:
        r = entropy_results[layer]
        sig_marker = "*" if r["significant"] else " "
        print(f"  Layer {layer}: Sch={r['scheming_mean']:.4f}, Hon={r['honest_mean']:.4f}, "
              f"diff={r['difference']:.4f}, p={r['p_value']:.4f} {sig_marker}")

    if sig_entropy_layers:
        print(f"\n>>> SIGNIFICANT ENTROPY DIFFERENCE at layers: {sig_entropy_layers} <<<")
        best_entropy_layer = max(sig_entropy_layers, key=lambda l: abs(entropy_results[l]["difference"]))
        best_entropy_diff = entropy_results[best_entropy_layer]["difference"]
        print(f"  Best layer: {best_entropy_layer} (diff={best_entropy_diff:.4f})")
        if best_entropy_diff > 0:
            print("  Direction: Scheming has HIGHER entropy (supports internal conflict hypothesis)")
        else:
            print("  Direction: Honest has higher entropy (unexpected)")
    else:
        print("\n>>> No significant entropy differences found <<<")

    # Create entropy summary table
    entropy_df = entropy_summary_table(entropy_results)
    print(f"\nEntropy Summary Table:")
    print(entropy_df.to_string(index=False))

    # ====================
    # PHASE 5: GENERATE VISUALIZATIONS
    # ====================
    print("\n" + "=" * 70)
    print("PHASE 5: Generating Visualizations")
    print("=" * 70)

    figures = {}

    # 1. Score distributions
    print("  Creating score distributions plot...")
    figures['score_distributions'] = plot_score_distributions(
        sch_scores, hon_scores, hal_scores, layer=best_layer
    )

    # 2. D-prime by layer
    print("  Creating d-prime by layer plot...")
    figures['dprime_by_layer'] = plot_dprime_by_layer(layer_metrics)

    # 3. ROC curve
    print("  Creating ROC curve...")
    figures['roc_curve'] = plot_roc(fpr, tpr, auc, layer=best_layer)

    # 4. Method comparison
    print("  Creating method comparison plot...")
    figures['method_comparison'] = plot_method_comparison(method_comparison, metric="separation")

    # 5. Summary dashboard
    print("  Creating summary dashboard...")
    figures['summary'] = create_summary_figure(
        sch_scores, hon_scores, hal_scores, layer_metrics, best_layer
    )

    # 6. Entropy analysis
    print("  Creating entropy by layer plot...")
    figures['entropy_by_layer'] = plot_entropy_by_layer(entropy_results)

    # Save all figures
    print("\nSaving figures...")
    save_all_figures(figures, str(CONFIG["output_dir"]), format="png", dpi=150)

    # ====================
    # SAVE RESULTS
    # ====================
    # Prepare entropy results for JSON
    entropy_for_json = {}
    for layer, r in entropy_results.items():
        entropy_for_json[str(layer)] = {
            "scheming_mean": r["scheming_mean"],
            "honest_mean": r["honest_mean"],
            "difference": r["difference"],
            "p_value": r["p_value"],
            "significant": r["significant"],
            "cohens_d": r["cohens_d"]
        }

    results = {
        "model": MODEL_NAME,
        "best_method": best_method,
        "best_layer": best_layer,
        "test_accuracy": float(test_metrics['accuracy']),
        "test_separation": float(test_metrics['separation']),
        "scheming_vs_honest": {
            "separation": float(separation),
            "p_value": float(svh['t_test']['p_value']),
            "dprime": float(svh['dprime']),
            "cohens_d": float(svh['cohens_d']),
            "auc": float(auc),
            "significant": bool(svh['significant'])
        },
        "entropy_analysis": {
            "hypothesis": "Deception creates internal conflict -> higher entropy",
            "significant_layers": sig_entropy_layers,
            "by_layer": entropy_for_json
        },
        "prompt_counts": {
            "scheming": len(scheming_prompts),
            "honest": len(honest_prompts)
        }
    }

    with open(CONFIG["output_dir"] / "results_v4.json", "w") as f:
        json.dump(results, f, indent=2)

    # ====================
    # FINAL SUMMARY
    # ====================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Best method: {best_method} at layer {best_layer}")
    print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Scheming vs Honest separation: {separation:.4f}")
    print(f"p-value: {svh['t_test']['p_value']:.6f} ({'*' if svh['significant'] else 'n.s.'})")
    print(f"d': {svh['dprime']:.4f}")
    print(f"AUC: {auc:.4f}")

    if svh['significant'] and separation > 0:
        print("\n>>> HYPOTHESIS A: Evidence of deception awareness <<<")
    elif svh['significant'] and separation < 0:
        print("\n>>> UNEXPECTED: Scheming scores higher than honest <<<")
    else:
        print("\n>>> NO SIGNIFICANT DIFFERENCE (Truth Probe) <<<")

    # Entropy findings
    print(f"\n--- Entropy Analysis (Internal Conflict) ---")
    if sig_entropy_layers:
        print(f"Significant entropy differences at layers: {sig_entropy_layers}")
        print(">>> ENTROPY PROBE: Supports internal conflict hypothesis <<<")
    else:
        print("No significant entropy differences found")

    print(f"\nVisualizations saved to: {CONFIG['output_dir']}")
    print("Experiment complete!")

    return results


if __name__ == "__main__":
    main()
