import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Initialize the positional encoding matrix with zeros
    pe = np.zeros((seq_length, d_model))
    
    # Create a vector of positions (0, 1, ..., seq_length-1)
    # Reshape to (seq_length, 1) for broadcasting
    position = np.arange(seq_length).reshape(-1, 1)
    
    # Calculate the division term for frequencies
    # Formula: exp(2i * -log(10000.0) / d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sine to even indices (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply cosine to odd indices (1, 3, 5, ...)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe