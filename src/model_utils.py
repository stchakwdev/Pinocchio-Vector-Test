"""
Model utilities for loading and configuring Pythia with TransformerLens.
Supports MPS (Apple Silicon), CUDA, and CPU backends.
"""

import gc
from typing import Optional, Literal

import torch
from transformer_lens import HookedTransformer


def get_device() -> str:
    """
    Detect the best available device.

    Priority: CUDA > MPS > CPU

    Returns:
        Device string for PyTorch
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(
    model_name: str = "EleutherAI/pythia-6.9b-deduped",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> HookedTransformer:
    """
    Load a model with TransformerLens for mechanistic interpretability.

    Args:
        model_name: HuggingFace model identifier
        device: Target device ('cuda', 'mps', 'cpu', or None for auto)
        dtype: Model precision (float16 recommended for memory efficiency)
        verbose: Print loading information

    Returns:
        HookedTransformer model with activation hooks

    Example:
        >>> model = load_model("EleutherAI/pythia-6.9b-deduped")
        >>> model.cfg.n_layers  # 32
        >>> model.cfg.d_model   # 4096
    """
    if device is None:
        device = get_device()

    if verbose:
        print(f"Loading {model_name} on {device} with {dtype}...")

    # Disable gradient computation for inference
    torch.set_grad_enabled(False)

    # Load model with TransformerLens
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )

    if verbose:
        print(f"Model loaded successfully!")
        print(f"  - Layers: {model.cfg.n_layers}")
        print(f"  - Hidden dim: {model.cfg.d_model}")
        print(f"  - Heads: {model.cfg.n_heads}")
        print(f"  - Context length: {model.cfg.n_ctx}")
        print(f"  - Vocab size: {model.cfg.d_vocab}")

    return model


def clear_memory(device: Optional[str] = None) -> None:
    """
    Clear GPU/MPS memory cache.

    Call this between phases to prevent OOM errors.

    Args:
        device: Device to clear (None for auto-detect)
    """
    gc.collect()

    if device is None:
        device = get_device()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == "mps":
        # MPS memory management
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


def get_model_info(model: HookedTransformer) -> dict:
    """
    Extract model configuration information.

    Args:
        model: Loaded HookedTransformer model

    Returns:
        Dictionary with model configuration
    """
    return {
        "model_name": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_heads": model.cfg.n_heads,
        "d_head": model.cfg.d_head,
        "n_ctx": model.cfg.n_ctx,
        "d_vocab": model.cfg.d_vocab,
        "device": str(model.cfg.device),
        "dtype": str(model.cfg.dtype),
    }


def estimate_memory_usage(
    model_name: str = "EleutherAI/pythia-6.9b-deduped",
    dtype: torch.dtype = torch.float16,
    batch_size: int = 1,
    seq_len: int = 512,
) -> dict:
    """
    Estimate memory usage for model and activations.

    Args:
        model_name: Model identifier
        dtype: Model precision
        batch_size: Batch size for inference
        seq_len: Sequence length

    Returns:
        Dictionary with memory estimates in GB
    """
    # Pythia-6.9B parameters
    n_params = 6.9e9
    bytes_per_param = 2 if dtype == torch.float16 else 4

    model_memory_gb = (n_params * bytes_per_param) / (1024**3)

    # Activation memory (rough estimate)
    # 32 layers * 4096 hidden * seq_len * batch_size * 2 bytes
    d_model = 4096
    n_layers = 32
    activation_memory_gb = (
        n_layers * d_model * seq_len * batch_size * bytes_per_param
    ) / (1024**3)

    return {
        "model_memory_gb": round(model_memory_gb, 2),
        "activation_memory_gb": round(activation_memory_gb, 2),
        "total_estimate_gb": round(model_memory_gb + activation_memory_gb, 2),
        "recommended_vram_gb": round((model_memory_gb + activation_memory_gb) * 1.5, 2),
    }
