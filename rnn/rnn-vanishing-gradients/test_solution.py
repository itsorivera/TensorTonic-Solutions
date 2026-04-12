import unittest
import numpy as np
from solution import compute_gradient_norm_decay

class TestVanishingGradient(unittest.TestCase):
    def test_vanishing_case(self):
        # T=4, W_hh = 0.5 * Identity
        # Norms: 1.0, 0.5, 0.25, 0.125
        T = 4
        W_hh = 0.5 * np.eye(3)
        expected = [1.0, 0.5, 0.25, 0.125]
        result = compute_gradient_norm_decay(T, W_hh)
        np.testing.assert_array_almost_equal(result, expected)

    def test_exploding_case(self):
        # T=3, W_hh = 2.0 * Identity
        # Norms: 1.0, 2.0, 4.0
        T = 3
        W_hh = 2.0 * np.eye(2)
        expected = [1.0, 2.0, 4.0]
        result = compute_gradient_norm_decay(T, W_hh)
        np.testing.assert_array_almost_equal(result, expected)

    def test_spectral_norm_complex(self):
        # W_hh with known spectral norm
        # W = [[3, 0], [0, 1]] -> Spectral norm is 3
        T = 3
        W_hh = np.array([[3.0, 0.0], [0.0, 1.0]])
        expected = [1.0, 3.0, 9.0]
        result = compute_gradient_norm_decay(T, W_hh)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sequence_length_one(self):
        T = 1
        W_hh = np.random.randn(3, 3)
        result = compute_gradient_norm_decay(T, W_hh)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 1.0)

if __name__ == '__main__':
    unittest.main()
