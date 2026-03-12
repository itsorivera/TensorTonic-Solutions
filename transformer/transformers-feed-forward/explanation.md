# Position-wise Feed-Forward Network (FFN)

This component is a fundamental part of each Transformer layer (both in the Encoder and Decoder). While self-attention allows the model to relate different tokens in the sequence, the Feed-Forward Network provides the model with the ability to process each position independently and learn richer representations through non-linear transformations.

## Architecture

The FFN consists of two linear transformations with a ReLU activation in between:

$$FFN(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2$$

Where:

- $x$ is the input tensor of shape `(batch_size, seq_len, d_model)`.
- $W_1$ and $b_1$ project the input to a higher-dimensional space (typically $d_{ff} = 4 \times d_{model}$).
- $W_2$ and $b_2$ project the hidden representation back to the original $d_{model}$ dimension.

### Key Concepts

1.  **Position-wise**: The same linear transformations are applied to each position (token) in the sequence independently. This means there is no interaction between different positions within this layer; interactions are handled solely by the attention mechanisms.
2.  **Expansion and Contraction**: The network first expands the dimensionality (from $d_{model}$ to $d_{ff}$) to allow the model to learn more complex features, and then contracts it back to maintain consistent dimensionality throughout the Transformer stack.
3.  **Non-linearity**: The ReLU activation function introduces non-linearity, enabling the model to approximate complex mapping functions.

## Implementation Details

The implementation uses **NumPy** for efficient matrix operations:

```python
def feed_forward(x, W1, b1, W2, b2):
    # First linear transformation (expansion)
    hidden = np.dot(x, W1) + b1

    # ReLU activation (non-linearity)
    relu_out = np.maximum(0, hidden)

    # Second linear transformation (contraction)
    output = np.dot(relu_out, W2) + b2

    return output
```
