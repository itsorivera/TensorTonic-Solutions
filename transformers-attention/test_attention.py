import torch
import unittest
import math
from transformers_scaled_dot_product_attention import scaled_dot_product_attention

class TestScaledDotProductAttention(unittest.TestCase):
    
    def test_basic_shape(self):
        """Test if the output shape is correct for equal sequence lengths."""
        batch_size, seq_len, d_k, d_v = 2, 8, 32, 32
        q = torch.randn(batch_size, seq_len, d_k)
        k = torch.randn(batch_size, seq_len, d_k)
        v = torch.randn(batch_size, seq_len, d_v)
        
        output = scaled_dot_product_attention(q, k, v)
        self.assertEqual(output.shape, (batch_size, seq_len, d_v))

    def test_cross_attention_shape(self):
        """Test if it handles different sequence lengths for queries vs keys/values."""
        batch_size, seq_len_q, seq_len_k, d_k, d_v = 1, 3, 5, 64, 64
        q = torch.randn(batch_size, seq_len_q, d_k)
        k = torch.randn(batch_size, seq_len_k, d_k)
        v = torch.randn(batch_size, seq_len_k, d_v)
        
        output = scaled_dot_product_attention(q, k, v)
        self.assertEqual(output.shape, (batch_size, seq_len_q, d_v))

    def test_masking(self):
        """Test if the mask correctly ignores specified positions."""
        batch_size, seq_len, d_k = 1, 4, 16
        q = torch.randn(batch_size, seq_len, d_k)
        k = torch.randn(batch_size, seq_len, d_k)
        v = torch.randn(batch_size, seq_len, d_k)
        
        # Mask that only allows attending to the first 2 tokens
        mask = torch.tensor([[[1, 1, 0, 0]]]) # shape (1, 1, 4) matches (batch, seq_q, seq_k) via broadcasting
        
        output = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # If we change the values of the masked positions in V, 
        # the output should NOT change because their attention weights should be 0.
        v_modified = v.clone()
        v_modified[:, 2:, :] += 100.0 # Modify masked parts
        
        output_modified = scaled_dot_product_attention(q, k, v_modified, mask=mask)
        
        self.assertTrue(torch.allclose(output, output_modified, atol=1e-6))

    def test_scaling_logic(self):
        """Verifies that scaling is applied by checking raw computation."""
        # Use small identity-like matrices to make computation predictable
        d_k = 4
        q = torch.ones(1, 1, d_k)
        k = torch.ones(1, 1, d_k)
        v = torch.ones(1, 1, d_k)
        
        # Raw dot product is 4.0 (1*1 + 1*1 + 1*1 + 1*1)
        # Scaled dot product is 4.0 / sqrt(4) = 4.0 / 2.0 = 2.0
        # softmax(2.0) for a single element is 1.0
        # Output should be 1.0 * V = [1, 1, 1, 1]
        
        output = scaled_dot_product_attention(q, k, v)
        expected = torch.ones(1, 1, d_k)
        self.assertTrue(torch.allclose(output, expected))

if __name__ == "__main__":
    unittest.main()
