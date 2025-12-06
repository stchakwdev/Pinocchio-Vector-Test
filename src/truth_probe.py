"""
Truth probe computation and projection.

Implements the Linear Representation Hypothesis to extract a "truth direction"
from residual stream activations and project new activations onto it.
"""

from typing import Dict, List, Optional, Literal, Tuple
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


def compute_truth_direction(
    true_activations: Dict[int, torch.Tensor],
    false_activations: Dict[int, torch.Tensor],
    method: Literal["difference_in_means", "pca", "logistic"] = "difference_in_means",
    normalize: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    Compute truth direction vector for each layer.

    Args:
        true_activations: Dict mapping layer to true statement activations [n_true, d_model]
        false_activations: Dict mapping layer to false statement activations [n_false, d_model]
        method: Extraction method
            - 'difference_in_means': Simple mean difference (recommended)
            - 'pca': First principal component of differences
            - 'logistic': Logistic regression weight vector
        normalize: Whether to normalize to unit vector

    Returns:
        Dict mapping layer to truth direction vector [d_model]

    Example:
        >>> truth_vectors = compute_truth_direction(true_acts, false_acts)
        >>> truth_vectors[16].shape  # torch.Size([4096])
    """
    truth_vectors = {}

    for layer in true_activations.keys():
        true_acts = true_activations[layer]
        false_acts = false_activations[layer]

        if method == "difference_in_means":
            direction = _difference_in_means(true_acts, false_acts)
        elif method == "pca":
            direction = _pca_direction(true_acts, false_acts)
        elif method == "logistic":
            direction = _logistic_direction(true_acts, false_acts)
        else:
            raise ValueError(f"Unknown method: {method}")

        if normalize:
            norm = direction.norm()
            if norm > 1e-8:
                direction = direction / norm

        truth_vectors[layer] = direction

    return truth_vectors


def _difference_in_means(
    true_acts: torch.Tensor,
    false_acts: torch.Tensor,
) -> torch.Tensor:
    """
    Compute truth direction as difference in means.

    v_truth = mean(true) - mean(false)
    """
    mean_true = true_acts.mean(dim=0)
    mean_false = false_acts.mean(dim=0)
    return mean_true - mean_false


def _pca_direction(
    true_acts: torch.Tensor,
    false_acts: torch.Tensor,
) -> torch.Tensor:
    """
    Compute truth direction using PCA on differences.

    More robust to noise than simple difference-in-means.
    """
    # Compute pairwise differences
    n_true = true_acts.shape[0]
    n_false = false_acts.shape[0]

    # Sample pairs to avoid O(n^2) computation
    n_pairs = min(n_true * n_false, 1000)
    differences = []

    for _ in range(n_pairs):
        i = np.random.randint(n_true)
        j = np.random.randint(n_false)
        diff = true_acts[i] - false_acts[j]
        differences.append(diff.cpu().numpy())

    differences = np.stack(differences)

    # PCA to find principal direction
    pca = PCA(n_components=1)
    pca.fit(differences)

    direction = torch.from_numpy(pca.components_[0]).float()

    # Ensure direction points toward "true"
    mean_true = true_acts.mean(dim=0).cpu()
    mean_false = false_acts.mean(dim=0).cpu()
    if (mean_true @ direction) < (mean_false @ direction):
        direction = -direction

    return direction.to(true_acts.device)


def _logistic_direction(
    true_acts: torch.Tensor,
    false_acts: torch.Tensor,
) -> torch.Tensor:
    """
    Compute truth direction using logistic regression.

    The weight vector of a linear classifier separating true from false.
    """
    # Prepare data
    X = torch.cat([true_acts, false_acts], dim=0).cpu().numpy()
    y = np.concatenate([
        np.ones(len(true_acts)),
        np.zeros(len(false_acts))
    ])

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X, y)

    # Extract weight vector
    direction = torch.from_numpy(clf.coef_[0]).float()
    return direction.to(true_acts.device)


def project_onto_truth(
    activation: torch.Tensor,
    truth_vector: torch.Tensor,
) -> float:
    """
    Project activation onto truth direction.

    Higher score = more "truthful" representation
    Lower score = more "untruthful" representation

    Args:
        activation: Activation vector [d_model]
        truth_vector: Truth direction vector [d_model] (should be normalized)

    Returns:
        Scalar projection score
    """
    return (activation @ truth_vector).item()


def batch_project_onto_truth(
    activations: torch.Tensor,
    truth_vector: torch.Tensor,
) -> np.ndarray:
    """
    Project multiple activations onto truth direction.

    Args:
        activations: Activation matrix [n_samples, d_model]
        truth_vector: Truth direction vector [d_model]

    Returns:
        Array of projection scores [n_samples]
    """
    scores = activations @ truth_vector
    return scores.cpu().numpy()


def evaluate_probe_accuracy(
    truth_vector: torch.Tensor,
    true_activations: torch.Tensor,
    false_activations: torch.Tensor,
    threshold: float = 0.0,
) -> dict:
    """
    Evaluate truth probe accuracy on a test set.

    Args:
        truth_vector: Truth direction [d_model]
        true_activations: True statement activations [n_true, d_model]
        false_activations: False statement activations [n_false, d_model]
        threshold: Classification threshold (default: 0.0, meaning positive = true)

    Returns:
        Dict with accuracy metrics
    """
    true_scores = batch_project_onto_truth(true_activations, truth_vector)
    false_scores = batch_project_onto_truth(false_activations, truth_vector)

    # Classification: score > threshold => predicted true
    true_correct = (true_scores > threshold).sum()
    false_correct = (false_scores <= threshold).sum()

    n_true = len(true_scores)
    n_false = len(false_scores)
    total = n_true + n_false

    accuracy = (true_correct + false_correct) / total
    true_positive_rate = true_correct / n_true
    true_negative_rate = false_correct / n_false

    return {
        "accuracy": accuracy,
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "balanced_accuracy": (true_positive_rate + true_negative_rate) / 2,
        "true_mean": true_scores.mean(),
        "false_mean": false_scores.mean(),
        "true_std": true_scores.std(),
        "false_std": false_scores.std(),
        "separation": true_scores.mean() - false_scores.mean(),
    }


def cross_validate_probe(
    true_activations: torch.Tensor,
    false_activations: torch.Tensor,
    n_folds: int = 5,
    method: str = "difference_in_means",
) -> dict:
    """
    Cross-validate truth probe to check generalization.

    Args:
        true_activations: True statement activations [n_true, d_model]
        false_activations: False statement activations [n_false, d_model]
        n_folds: Number of cross-validation folds
        method: Truth direction extraction method

    Returns:
        Dict with cross-validation results
    """
    n_true = len(true_activations)
    n_false = len(false_activations)

    # Create fold indices
    true_indices = np.arange(n_true)
    false_indices = np.arange(n_false)
    np.random.shuffle(true_indices)
    np.random.shuffle(false_indices)

    true_folds = np.array_split(true_indices, n_folds)
    false_folds = np.array_split(false_indices, n_folds)

    accuracies = []
    separations = []

    for fold in range(n_folds):
        # Split into train/test
        true_test_idx = true_folds[fold]
        false_test_idx = false_folds[fold]

        true_train_idx = np.concatenate([true_folds[i] for i in range(n_folds) if i != fold])
        false_train_idx = np.concatenate([false_folds[i] for i in range(n_folds) if i != fold])

        # Train probe on training set
        true_train = true_activations[true_train_idx]
        false_train = false_activations[false_train_idx]

        truth_vec = compute_truth_direction(
            {0: true_train},
            {0: false_train},
            method=method,
        )[0]

        # Evaluate on test set
        true_test = true_activations[true_test_idx]
        false_test = false_activations[false_test_idx]

        metrics = evaluate_probe_accuracy(truth_vec, true_test, false_test)
        accuracies.append(metrics["accuracy"])
        separations.append(metrics["separation"])

    return {
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "mean_separation": np.mean(separations),
        "std_separation": np.std(separations),
        "fold_accuracies": accuracies,
    }


def find_best_layer(
    true_activations: Dict[int, torch.Tensor],
    false_activations: Dict[int, torch.Tensor],
    method: str = "difference_in_means",
) -> Tuple[int, dict]:
    """
    Find the layer with the best truth probe discrimination.

    Args:
        true_activations: Dict mapping layer to activations
        false_activations: Dict mapping layer to activations
        method: Direction extraction method

    Returns:
        Tuple of (best_layer, all_layer_metrics)
    """
    results = {}

    for layer in true_activations.keys():
        truth_vec = compute_truth_direction(
            {layer: true_activations[layer]},
            {layer: false_activations[layer]},
            method=method,
        )[layer]

        metrics = evaluate_probe_accuracy(
            truth_vec,
            true_activations[layer],
            false_activations[layer],
        )
        results[layer] = metrics

    # Find best layer by separation
    best_layer = max(results.keys(), key=lambda l: results[l]["separation"])

    return best_layer, results
