import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer with scaled initialization.
    """
    embedding = nn.Embedding(vocab_size, d_model)
    # Initialization should be scaled (e.g., standard normal / sqrt(d_model))
    nn.init.normal_(embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
    return embedding

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Look up embeddings for each token index in the input
    embedded = embedding(tokens)
    # Scale the output embeddings by sqrt(d_model) as specified in the Transformer paper
    return embedded * math.sqrt(d_model)
