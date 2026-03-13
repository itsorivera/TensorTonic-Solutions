import numpy as np
from normalized_layer import layer_norm

def test_layer_norm_shapes():
    """
    Test the output shapes of the layer_norm function.
    """
    batch_size, seq_len, d_model = 2, 10, 64
    x = np.random.randn(batch_size, seq_len, d_model)
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)
    
    output = layer_norm(x, gamma, beta)
    print(f"Test Shape - Input: {x.shape}, Output: {output.shape}")
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

def test_layer_norm_logic():
    """
    Test the logic of the layer_norm function using the example from problem.md.
    """
    x = np.array([[1, 2, 3, 4]], dtype=float)
    gamma = np.array([1, 1, 1, 1], dtype=float)
    beta = np.array([0, 0, 0, 0], dtype=float)
    
    # Expected output from problem.md: [[-1.34, -0.45, 0.45, 1.34]]
    # Let's calculate precisely for validation
    mean = np.mean(x, axis=-1, keepdims=True) # 2.5
    var = np.var(x, axis=-1, keepdims=True)  # ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
    expected_output = (x - mean) / np.sqrt(var + 1e-6)
    
    output = layer_norm(x, gamma, beta)
    
    print(f"Test Logic - Input: {x}")
    print(f"Test Logic - Expected: {expected_output}")
    print(f"Test Logic - Got: {output}")
    
    np.testing.assert_allclose(output, expected_output, atol=1e-2), "Logic test failed"
    
    # Check if mean is approx 0 and std is approx 1
    np.testing.assert_allclose(np.mean(output), 0, atol=1e-6)
    np.testing.assert_allclose(np.std(output), 1, atol=1e-2)

def test_layer_norm_parameters():
    """
    Test if gamma and beta are applied correctly.
    """
    x = np.array([[1, 2, 3, 4]], dtype=float)
    gamma = np.array([2, 2, 2, 2], dtype=float)
    beta = np.array([1, 1, 1, 1], dtype=float)
    
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + 1e-6)
    expected_output = 2 * x_norm + 1
    
    output = layer_norm(x, gamma, beta)
    np.testing.assert_allclose(output, expected_output, atol=1e-6), "Parameter test failed"

if __name__ == "__main__":
    try:
        test_layer_norm_shapes()
        test_layer_norm_logic()
        test_layer_norm_parameters()
        print("All tests passed successfully!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
