import numpy as np
from feed_forward import feed_forward

def test_feed_forward_shapes():
    """
    Test the output shapes of the feed_forward function with example dimensions.
    """
    # Example 1
    batch_size, seq_len, d_model = 2, 10, 64
    d_ff = 256
    x = np.random.randn(batch_size, seq_len, d_model)
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)
    
    output = feed_forward(x, W1, b1, W2, b2)
    print(f"Test 1 - Input: {x.shape}, Output: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"

    # Example 2
    batch_size, seq_len, d_model = 1, 5, 32
    d_ff = 128
    x = np.random.randn(batch_size, seq_len, d_model)
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)
    
    output = feed_forward(x, W1, b1, W2, b2)
    print(f"Test 2 - Input: {x.shape}, Output: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"

def test_feed_forward_logic():
    """
    Test the logic of the feed_forward function using a simple deterministic example.
    """
    x = np.array([[[1.0, -1.0]]]) # Shape (1, 1, 2)
    W1 = np.array([[1.0, 0.0], [0.0, 1.0]]) # Identity-like
    b1 = np.array([0.5, -0.5])
    W2 = np.array([[1.0, 0.0], [0.0, 1.0]]) # Identity-like
    b2 = np.array([0.0, 0.0])
    
    # xW1 + b1 = [1.0, -1.0] + [0.5, -0.5] = [1.5, -1.5]
    # ReLU(1.5, -1.5) = [1.5, 0.0]
    # [1.5, 0.0] * W2 + b2 = [1.5, 0.0]
    
    expected_output = np.array([[[1.5, 0.0]]])
    output = feed_forward(x, W1, b1, W2, b2)
    
    print(f"Test Logic - Expected: {expected_output}, Got: {output}")
    np.testing.assert_allclose(output, expected_output), "Logic test failed"

if __name__ == "__main__":
    try:
        test_feed_forward_shapes()
        test_feed_forward_logic()
        print("All tests passed successfully!")
    except Exception as e:
        print(f"Tests failed: {e}")
        exit(1)
