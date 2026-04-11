import unittest
import numpy as np
from solution import rnn_forward

class TestRNNForward(unittest.TestCase):
    def test_shapes_example1(self):
        batch_size, seq_length, input_dim = 2, 5, 4
        hidden_dim = 3
        
        X = np.random.randn(batch_size, seq_length, input_dim)
        h_0 = np.random.randn(batch_size, hidden_dim)
        W_xh = np.random.randn(hidden_dim, input_dim)
        W_hh = np.random.randn(hidden_dim, hidden_dim)
        b_h = np.random.randn(hidden_dim)
        
        h_all, h_final = rnn_forward(X, h_0, W_xh, W_hh, b_h)
        
        self.assertEqual(h_all.shape, (batch_size, seq_length, hidden_dim))
        self.assertEqual(h_final.shape, (batch_size, hidden_dim))
        np.testing.assert_array_almost_equal(h_all[:, -1, :], h_final)
        self.assertTrue(np.all(h_all >= -1.0) and np.all(h_all <= 1.0))

    def test_shapes_example2(self):
        batch_size, seq_length, input_dim = 4, 10, 8
        hidden_dim = 16
        
        X = np.random.randn(batch_size, seq_length, input_dim)
        h_0 = np.random.randn(batch_size, hidden_dim)
        W_xh = np.random.randn(hidden_dim, input_dim)
        W_hh = np.random.randn(hidden_dim, hidden_dim)
        b_h = np.random.randn(hidden_dim)
        
        h_all, h_final = rnn_forward(X, h_0, W_xh, W_hh, b_h)
        
        self.assertEqual(h_all.shape, (batch_size, seq_length, hidden_dim))
        self.assertEqual(h_final.shape, (batch_size, hidden_dim))
        np.testing.assert_array_almost_equal(h_all[:, -1, :], h_final)

    def test_manual_values(self):
        # Small controlled test case
        # X: (1, 2, 1) -> batch 1, time 2, input_dim 1
        X = np.array([[[1.0], [2.0]]])
        # h_0: (1, 1) -> batch 1, hidden_dim 1
        h_0 = np.array([[0.0]])
        # W_xh: (1, 1)
        W_xh = np.array([[0.5]])
        # W_hh: (1, 1)
        W_hh = np.array([[0.1]])
        # b_h: (1,)
        b_h = np.array([0.0])
        
        # Step 1:
        # x_1 = 1.0, h_0 = 0.0
        # z_1 = 0.0 * 0.1 + 1.0 * 0.5 + 0.0 = 0.5
        # h_1 = tanh(0.5)
        h1_expected = np.tanh(0.5)
        
        # Step 2:
        # x_2 = 2.0, h_1 = tanh(0.5)
        # z_2 = tanh(0.5) * 0.1 + 2.0 * 0.5 + 0.0
        # h_2 = tanh(z_2)
        h2_expected = np.tanh(h1_expected * 0.1 + 2.0 * 0.5)
        
        h_all, h_final = rnn_forward(X, h_0, W_xh, W_hh, b_h)
        
        expected_all = np.array([[[h1_expected], [h2_expected]]])
        np.testing.assert_array_almost_equal(h_all, expected_all)
        np.testing.assert_array_almost_equal(h_final, np.array([[h2_expected]]))

if __name__ == '__main__':
    unittest.main()
