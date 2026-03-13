import torch
import torch.nn as nn
import math
from transformers_embedding import create_embedding_layer, embed_tokens

def test_embedder():
    vocab_size = 100
    d_model = 64
    
    # Test 1: Shape check for sequence input
    tokens1 = torch.tensor([5, 12, 3])
    embedding = create_embedding_layer(vocab_size, d_model)
    output1 = embed_tokens(embedding, tokens1, d_model)
    print(f"Test 1 - Input: {tokens1.shape}, Output: {output1.shape}")
    assert output1.shape == (3, 64)
    
    # Test 2: Shape check for batch input
    tokens2 = torch.tensor([[0, 99], [1, 2]])
    output2 = embed_tokens(embedding, tokens2, d_model)
    print(f"Test 2 - Input: {tokens2.shape}, Output: {output2.shape}")
    assert output2.shape == (2, 2, 64)
    
    # Test 3: Scaling check
    # Let's manually check scaling if we know the weight
    test_idx = torch.tensor([42])
    with torch.no_grad():
        manual_lookup = embedding.weight[42]
        expected_output = manual_lookup * math.sqrt(d_model)
        actual_output = embed_tokens(embedding, test_idx, d_model)
        
        diff = torch.abs(actual_output - expected_output).max()
        print(f"Test 3 - Scaling difference: {diff.item()}")
        assert diff < 1e-6
        
    print("All tests passed!")

if __name__ == "__main__":
    test_embedder()
