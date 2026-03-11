import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q: Queries [batch, seq_len_q, d_model]
        K: Keys [batch, seq_len_k, d_model]
        V: Values [batch, seq_len_v, d_model]
        W_q, W_k, W_v: weight matrices for projections [d_model, d_model]
        W_o: weight matrix for final output [d_model, d_model]
        num_heads: number of attention heads
        
    Returns:
        output: [batch, seq_len_q, d_model]
    """
    batch_size, seq_len_q, d_model = Q.shape
    _, seq_len_k, _ = K.shape
    d_k = d_model // num_heads
    
    # 1. Linear projections
    Q = Q @ W_q  # [batch, seq_len_q, d_model]
    K = K @ W_k  # [batch, seq_len_k, d_model]
    V = V @ W_v  # [batch, seq_len_v, d_model]
    
    # 2. Reshape and transpose to separate heads
    # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
    Q = Q.reshape(batch_size, seq_len_q, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len_k, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len_k, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # 3. Scaled dot-product attention
    # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    context = weights @ V  # [batch, num_heads, seq_len_q, d_k]
    
    # 4. Concatenate heads
    # [batch, num_heads, seq_len_q, d_k] -> [batch, seq_len_q, num_heads, d_k] -> [batch, seq_len_q, d_model]
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, d_model)
    
    # 5. Final linear projection
    output = context @ W_o
    
    return output