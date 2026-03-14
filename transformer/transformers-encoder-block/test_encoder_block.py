import numpy as np
from transformer_encoder_block import encoder_block

def test_encoder_block_shapes():
    # Example 1
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8
    d_ff = 256
    
    x = np.random.randn(batch_size, seq_len, d_model)
    
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    W_o = np.random.randn(d_model, d_model)
    
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)
    
    gamma1 = np.ones(d_model)
    beta1 = np.zeros(d_model)
    gamma2 = np.ones(d_model)
    beta2 = np.zeros(d_model)
    
    output = encoder_block(x, W_q, W_k, W_v, W_o, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, num_heads)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("Example 1 shape test passed!")

    # Example 2
    batch_size = 1
    seq_len = 5
    d_model = 32
    num_heads = 4
    d_ff = 128
    
    x = np.random.randn(batch_size, seq_len, d_model)
    
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    W_o = np.random.randn(d_model, d_model)
    
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)
    
    gamma1 = np.ones(d_model)
    beta1 = np.zeros(d_model)
    gamma2 = np.ones(d_model)
    beta2 = np.zeros(d_model)
    
    output = encoder_block(x, W_q, W_k, W_v, W_o, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, num_heads)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("Example 2 shape test passed!")

def test_encoder_block_logic():
    # Simple check to ensure output differs meaningfully from the original input
    batch_size = 1
    seq_len = 2
    d_model = 4
    num_heads = 2
    d_ff = 8
    
    x = np.ones((batch_size, seq_len, d_model))
    
    W_q = np.eye(d_model, d_model)
    W_k = np.eye(d_model, d_model)
    W_v = np.eye(d_model, d_model)
    W_o = np.eye(d_model, d_model)
    
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)
    
    gamma1 = np.ones(d_model)
    beta1 = np.zeros(d_model)
    gamma2 = np.ones(d_model)
    beta2 = np.zeros(d_model)
    
    output = encoder_block(x, W_q, W_k, W_v, W_o, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, num_heads)
    
    assert not np.allclose(output, x), "Output should be transformed by encoder block"
    print("Logic test passed!")

if __name__ == "__main__":
    test_encoder_block_shapes()
    test_encoder_block_logic()
    print("All tests passed!")
