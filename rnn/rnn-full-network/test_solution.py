import unittest
import numpy as np
from solution import VanillaRNN

class TestVanillaRNN(unittest.TestCase):
    def test_forward_shapes(self):
        batch_size, seq_length, input_dim = 2, 4, 3
        hidden_dim, output_dim = 5, 2
        
        rnn = VanillaRNN(input_dim, hidden_dim, output_dim)
        X = np.random.randn(batch_size, seq_length, input_dim)
        
        y_seq, h_final = rnn.forward(X)
        
        self.assertEqual(y_seq.shape, (batch_size, seq_length, output_dim))
        self.assertEqual(h_final.shape, (batch_size, hidden_dim))

    def test_h0_initialization(self):
        batch_size, seq_length, input_dim = 1, 1, 1
        hidden_dim, output_dim = 1, 1
        
        rnn = VanillaRNN(input_dim, hidden_dim, output_dim)
        # Set weights to something predictable (identity for now, though scaling applies)
        rnn.W_xh = np.array([[1.0]])
        rnn.W_hh = np.array([[0.0]])
        rnn.b_h = np.array([0.0])
        rnn.W_hy = np.array([[1.0]])
        rnn.b_y = np.array([0.0])
        
        X = np.array([[[1.0]]])
        h_0 = np.array([[0.0]])
        
        y_seq, h_final = rnn.forward(X, h_0)
        
        # h_1 = tanh(1*1 + 0*0 + 0) = tanh(1.0)
        # y_1 = 1*tanh(1.0) + 0 = tanh(1.0)
        expected_val = np.tanh(1.0)
        np.testing.assert_array_almost_equal(h_final, np.array([[expected_val]]))
        np.testing.assert_array_almost_equal(y_seq, np.array([[[expected_val]]]))

    def test_xavier_initialization(self):
        # Check if weights are initialized approximately within range
        input_dim, hidden_dim, output_dim = 100, 100, 100
        rnn = VanillaRNN(input_dim, hidden_dim, output_dim)
        
        # W_xh scale should be sqrt(2 / (100+100)) = sqrt(1/100) = 0.1
        # Std dev of initialization should be around 0.1
        self.assertLess(np.abs(np.std(rnn.W_xh) - 0.1), 0.05)
        self.assertLess(np.abs(np.std(rnn.W_hh) - 0.1), 0.05)
        
        # Biases should be zero
        np.testing.assert_array_equal(rnn.b_h, np.zeros(hidden_dim))
        np.testing.assert_array_equal(rnn.b_y, np.zeros(output_dim))

if __name__ == '__main__':
    unittest.main()
