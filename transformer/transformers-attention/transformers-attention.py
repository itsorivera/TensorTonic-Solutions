import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Queries tensor [batch_size, seq_len_q, d_k]
        K: Keys tensor [batch_size, seq_len_k, d_k]
        V: Values tensor [batch_size, seq_len_k, d_v]
        mask: Optional mask tensor [batch_size, seq_len_q, seq_len_k] 
              where 1 means "attend" and 0 means "mask out".
              
    Returns:
        Output tensor [batch_size, seq_len_q, d_v]
    """
    d_k = Q.size(-1)
    
    # 1. Compute dot product scores: (batch, seq_len_q, seq_len_k)
    # K.transpose(-2, -1) swaps the last two dimensions to allow multiplication
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 2. Scale the scores to prevent large values
    scores = scores / math.sqrt(d_k)
    
    # 3. Apply mask if provided
    if mask is not None:
        # mask == 0 indicates positions to hide
        # We replace them with a very small number so softmax results in 0
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 4. Apply softmax to get attention weights (along the last dimension)
    attention_weights = F.softmax(scores, dim=-1)
    
    # 5. Multiply weights by values: (batch, seq_len_q, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output
