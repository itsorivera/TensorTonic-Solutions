import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        batch_size, seq_length, input_dim = X.shape
        
        if h_0 is None:
            h_0 = np.zeros((batch_size, self.hidden_dim))
            
        h_prev = h_0
        h_all = []
        
        # 1. Compute hidden states through time
        for t in range(seq_length):
            x_t = X[:, t, :]
            # h_t = tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)
            h_t = np.tanh(np.dot(x_t, self.W_xh.T) + np.dot(h_prev, self.W_hh.T) + self.b_h)
            h_all.append(h_t)
            h_prev = h_t
            
        # Stack hidden states: (batch, seq_length, hidden_dim)
        h_all_stacked = np.stack(h_all, axis=1)
        
        # 2. Project hidden states to output dimension
        # We can flatten to (batch * seq_length, hidden_dim) for efficient matrix multiplication
        h_flat = h_all_stacked.reshape(-1, self.hidden_dim)
        y_flat = np.dot(h_flat, self.W_hy.T) + self.b_y
        
        # Reshape back to (batch, seq_length, output_dim)
        y_seq = y_flat.reshape(batch_size, seq_length, -1)
        
        return y_seq, h_prev