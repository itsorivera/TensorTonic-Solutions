import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.

    Args:
        x: Input array of shape (..., d_model)
        gamma: Scale parameter of shape (d_model,)
        beta: Shift parameter of shape (d_model,)
        eps: Small constant for numerical stability

    Returns:
        Normalized array of same shape as x
    """
    # Compute mean across the last dimension (d_model)
    mean = np.mean(x, axis=-1, keepdims=True)
    
    # Compute variance across the last dimension (d_model)
    var = np.var(x, axis=-1, keepdims=True)
    
    # Normalize: (x - mean) / sqrt(var + eps)
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # Scale and shift using learnable parameters
    return gamma * x_norm + beta