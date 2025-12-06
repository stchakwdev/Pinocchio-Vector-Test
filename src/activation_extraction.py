"""
Activation extraction utilities using TransformerLens hooks.

Extracts residual stream activations at specified layers and token positions.
"""

from typing import Dict, List, Optional, Union, Callable
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm


def extract_residual_activations(
    model: HookedTransformer,
    text: str,
    layers: List[int],
    position: int = -1,
    activation_type: str = "resid_post",
) -> Dict[int, torch.Tensor]:
    """
    Extract residual stream activations at specified layers.

    Args:
        model: TransformerLens model
        text: Input text
        layers: List of layer indices to extract
        position: Token position (-1 for last token)
        activation_type: Type of residual stream ('resid_pre', 'resid_mid', 'resid_post')

    Returns:
        Dict mapping layer index to activation tensor [d_model]

    Example:
        >>> activations = extract_residual_activations(model, "Hello world", [12, 16, 20])
        >>> activations[16].shape  # torch.Size([4096])
    """
    # Define hook names for the specified layers
    hook_names = [f"blocks.{layer}.hook_{activation_type}" for layer in layers]

    # Create filter function
    def name_filter(name: str) -> bool:
        return any(hook_name in name for hook_name in hook_names)

    # Run model with caching
    _, cache = model.run_with_cache(
        text,
        names_filter=name_filter,
        remove_batch_dim=True,
    )

    # Extract activations at specified position
    activations = {}
    for layer in layers:
        key = f"blocks.{layer}.hook_{activation_type}"
        if key in cache:
            # Shape: [seq_len, d_model] -> [d_model] at position
            activations[layer] = cache[key][position, :].clone()

    return activations


def batch_extract_activations(
    model: HookedTransformer,
    texts: List[str],
    layers: List[int],
    position: int = -1,
    activation_type: str = "resid_post",
    show_progress: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    Extract activations for multiple texts.

    Args:
        model: TransformerLens model
        texts: List of input texts
        layers: List of layer indices
        position: Token position (-1 for last token)
        activation_type: Type of residual stream
        show_progress: Show progress bar

    Returns:
        Dict mapping layer to stacked activations [n_texts, d_model]
    """
    all_activations = {layer: [] for layer in layers}

    iterator = tqdm(texts, desc="Extracting activations") if show_progress else texts

    for text in iterator:
        acts = extract_residual_activations(
            model, text, layers, position, activation_type
        )
        for layer in layers:
            all_activations[layer].append(acts[layer])

    # Stack into tensors
    return {
        layer: torch.stack(acts)
        for layer, acts in all_activations.items()
    }


def find_token_position(
    model: HookedTransformer,
    text: str,
    target: str,
    occurrence: int = 1,
) -> int:
    """
    Find the position of a target token in tokenized text.

    Args:
        model: TransformerLens model
        text: Input text
        target: Target string to find
        occurrence: Which occurrence to find (1 = first)

    Returns:
        Token position (0-indexed)

    Raises:
        ValueError: If target not found

    Example:
        >>> pos = find_token_position(model, "Hello world", "world")
        >>> pos  # 1 (or wherever 'world' is tokenized)
    """
    # Tokenize the text
    tokens = model.to_tokens(text, prepend_bos=True)
    str_tokens = model.to_str_tokens(text, prepend_bos=True)

    # Search for target
    count = 0
    for i, tok in enumerate(str_tokens):
        # Check for exact match or substring match
        tok_clean = tok.strip().lower()
        target_clean = target.strip().lower()

        if target_clean in tok_clean or tok_clean == target_clean:
            count += 1
            if count == occurrence:
                return i

    # If not found, try looking for the target as a sequence of tokens
    full_text = "".join(str_tokens)
    if target.lower() in full_text.lower():
        # Find approximate position
        target_start = full_text.lower().find(target.lower())
        char_count = 0
        for i, tok in enumerate(str_tokens):
            char_count += len(tok)
            if char_count > target_start:
                return i

    raise ValueError(
        f"Target '{target}' not found in tokens: {str_tokens}"
    )


def extract_activation_at_token(
    model: HookedTransformer,
    text: str,
    target_token: str,
    layers: List[int],
    activation_type: str = "resid_post",
    occurrence: int = 1,
) -> Dict[int, torch.Tensor]:
    """
    Extract activation at a specific target token.

    Args:
        model: TransformerLens model
        text: Input text
        target_token: Token to analyze
        layers: List of layer indices
        activation_type: Type of residual stream
        occurrence: Which occurrence of target (1 = first)

    Returns:
        Dict mapping layer to activation at target position

    Example:
        >>> acts = extract_activation_at_token(model, "I am human", "human", [16])
    """
    position = find_token_position(model, text, target_token, occurrence)
    return extract_residual_activations(model, text, layers, position, activation_type)


def get_token_info(
    model: HookedTransformer,
    text: str,
) -> List[dict]:
    """
    Get detailed information about each token in the text.

    Args:
        model: TransformerLens model
        text: Input text

    Returns:
        List of dicts with token info (position, string, id)
    """
    tokens = model.to_tokens(text, prepend_bos=True)
    str_tokens = model.to_str_tokens(text, prepend_bos=True)

    return [
        {
            "position": i,
            "string": str_tokens[i],
            "token_id": tokens[0, i].item(),
        }
        for i in range(len(str_tokens))
    ]


def compare_activations(
    activation1: torch.Tensor,
    activation2: torch.Tensor,
) -> dict:
    """
    Compare two activations with various metrics.

    Args:
        activation1: First activation [d_model]
        activation2: Second activation [d_model]

    Returns:
        Dict with comparison metrics
    """
    # Normalize for cosine similarity
    norm1 = activation1 / (activation1.norm() + 1e-8)
    norm2 = activation2 / (activation2.norm() + 1e-8)

    return {
        "cosine_similarity": (norm1 @ norm2).item(),
        "euclidean_distance": (activation1 - activation2).norm().item(),
        "dot_product": (activation1 @ activation2).item(),
        "mean_diff": (activation1 - activation2).mean().item(),
        "max_diff": (activation1 - activation2).abs().max().item(),
    }


def extract_with_generation(
    model: HookedTransformer,
    prompt: str,
    layers: List[int],
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> tuple:
    """
    Generate text and extract activations during generation.

    Args:
        model: TransformerLens model
        prompt: Input prompt
        layers: Layers to cache
        max_new_tokens: Tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (generated_text, activations_per_token)
    """
    # First generate the text
    generated = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_type="str",
    )

    # Then extract activations for the full sequence
    activations = extract_residual_activations(
        model,
        generated,
        layers,
        position=-1,  # Last generated token
    )

    return generated, activations
