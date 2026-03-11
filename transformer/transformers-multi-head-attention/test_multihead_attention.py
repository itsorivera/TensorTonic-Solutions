import numpy as np
from multihead_attention import multi_head_attention

def test_multi_head_attention_shapes():
    # Example 1
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8
    
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    W_o = np.random.randn(d_model, d_model)
    
    output = multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("Example 1 shape test passed!")

    # Example 2
    batch_size = 1
    seq_len = 5
    d_model = 32
    num_heads = 4
    
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    W_o = np.random.randn(d_model, d_model)
    
    output = multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("Example 2 shape test passed!")

def test_multi_head_attention_logic():
    # Verify that it's doing something meaningful (not just returning input)
    batch_size = 1
    seq_len = 2
    d_model = 4
    num_heads = 2
    
    Q = np.eye(2, 4).reshape(1, 2, 4)
    K = np.eye(2, 4).reshape(1, 2, 4)
    V = np.eye(2, 4).reshape(1, 2, 4)
    
    # Use identity matrices for weights to simplify
    W_q = np.eye(d_model)
    W_k = np.eye(d_model)
    W_v = np.eye(d_model)
    W_o = np.eye(d_model)
    
    output = multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads)
    
    # Check if output is different from input (since softmax and projections are involved)
    assert not np.allclose(output, Q), "Output should be transformed by attention"
    print("Logic test passed!")

if __name__ == "__main__":
    test_multi_head_attention_shapes()
    test_multi_head_attention_logic()
    print("All tests passed!")
