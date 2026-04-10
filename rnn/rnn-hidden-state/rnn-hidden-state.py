import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    """
    Initialize the hidden state for an RNN.
    Args:
        batch_size: The number of sequences in the batch.
        hidden_dim: The dimensionality of the hidden state.
    Returns:
        A zero-initialized array of shape (batch_size, hidden_dim) and type float32.
    """
    return np.zeros((batch_size, hidden_dim), dtype=np.float32)
