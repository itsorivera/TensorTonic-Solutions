import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # 1. Compute pre-activation gradient (gradient through tanh)
    # dh_next is (batch, hidden_dim), h_t is (batch, hidden_dim)
    # Derivative of tanh(z) is 1 - tanh^2(z) = 1 - h_t^2
    dtanh = dh_next * (1 - h_t**2)
    
    # 2. Compute gradient w.r.t. W_hh
    # dtanh is (batch, hidden_dim), h_prev is (batch, hidden_dim)
    # dW_hh = dtanh^T @ h_prev
    dW_hh = np.dot(dtanh.T, h_prev)
    
    # 3. Compute gradient w.r.t. h_prev
    # dtanh is (batch, hidden_dim), W_hh is (hidden_dim, hidden_dim)
    # dh_prev = dtanh @ W_hh
    dh_prev = np.dot(dtanh, W_hh)
    
    return dh_prev, dW_hh