import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch_size, seq_length, input_dim = X.shape
    hidden_dim = h_0.shape[1]
    
    h_all = []
    h_prev = h_0
    
    for t in range(seq_length):
        x_t = X[:, t, :]  # (batch_size, input_dim)
        # h_t = tanh(h_{t-1} W_hh^T + x_t W_xh^T + b_h)
        h_t = np.tanh(np.dot(h_prev, W_hh.T) + np.dot(x_t, W_xh.T) + b_h)
        h_all.append(h_t)
        h_prev = h_t
        
    h_all_stacked = np.stack(h_all, axis=1)
    return h_all_stacked, h_prev