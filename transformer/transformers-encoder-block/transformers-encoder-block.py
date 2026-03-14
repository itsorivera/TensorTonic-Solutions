import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B, N, d_model = Q.shape
    d_k = d_model // num_heads
    
    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)
    
    Q_split = Q_proj.reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    K_split = K_proj.reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    V_split = V_proj.reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    
    scores = np.matmul(Q_split, K_split.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    context = np.matmul(weights, V_split)
    
    context = context.transpose(0, 2, 1, 3).reshape(B, N, d_model)
    return np.dot(context, W_o)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    inner = np.dot(x, W1) + b1
    activated = np.maximum(0, inner)
    return np.dot(activated, W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    mha_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_1 = layer_norm(x + mha_out, gamma1, beta1)
    
    ff_out = feed_forward(x_1, W1, b1, W2, b2)
    x_2 = layer_norm(x_1 + ff_out, gamma2, beta2)
    
    return x_2