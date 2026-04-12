import unittest
import numpy as np
from solution import bptt_single_step

class TestBPTT(unittest.TestCase):
    def test_shapes(self):
        batch_size, hidden_dim, input_dim = 2, 3, 4
        dh_next = np.random.randn(batch_size, hidden_dim)
        h_t = np.random.randn(batch_size, hidden_dim)
        h_prev = np.random.randn(batch_size, hidden_dim)
        x_t = np.random.randn(batch_size, input_dim)
        W_hh = np.random.randn(hidden_dim, hidden_dim)
        
        dh_prev, dW_hh = bptt_single_step(dh_next, h_t, h_prev, x_t, W_hh)
        
        self.assertEqual(dh_prev.shape, (batch_size, hidden_dim))
        self.assertEqual(dW_hh.shape, (hidden_dim, hidden_dim))

    def test_identity_case(self):
        # Input: dh_next = ones, h_t = zeros, W_hh = identity
        # Output: dh_prev = ones
        batch_size, hidden_dim = 2, 3
        dh_next = np.ones((batch_size, hidden_dim))
        h_t = np.zeros((batch_size, hidden_dim))
        h_prev = np.random.randn(batch_size, hidden_dim)
        x_t = np.random.randn(batch_size, 4)
        W_hh = np.eye(hidden_dim)
        
        dh_prev, dW_hh = bptt_single_step(dh_next, h_t, h_prev, x_t, W_hh)
        
        # dtanh = dh_next * (1 - 0^2) = dh_next = ones
        # dh_prev = dtanh @ W_hh = ones @ identity = ones
        np.testing.assert_array_almost_equal(dh_prev, np.ones((batch_size, hidden_dim)))

    def test_manual_values(self):
        # batch 1, hidden 1
        dh_next = np.array([[0.5]])
        h_t = np.array([[0.1]]) # tanh(z) = 0.1
        h_prev = np.array([[0.2]])
        x_t = np.array([[0.3]])
        W_hh = np.array([[0.8]])
        
        # dtanh = 0.5 * (1 - 0.1^2) = 0.5 * 0.99 = 0.495
        # dW_hh = 0.495 * 0.2 = 0.099
        # dh_prev = 0.495 * 0.8 = 0.396
        
        dh_prev, dW_hh = bptt_single_step(dh_next, h_t, h_prev, x_t, W_hh)
        
        np.testing.assert_array_almost_equal(dh_prev, np.array([[0.396]]))
        np.testing.assert_array_almost_equal(dW_hh, np.array([[0.099]]))

if __name__ == '__main__':
    unittest.main()
