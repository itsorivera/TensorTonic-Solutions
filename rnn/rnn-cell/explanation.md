# Understanding the RNN Cell: A Deep Dive into Low-Level Implementation

Recurrent Neural Networks (RNNs) are designed to process sequential data by maintaining a "memory" (hidden state) of past inputs. This document explains the implementation details of the single RNN cell you just built.

---

## 1. The Mathematical Foundation

![RNN Architecture](/rnn/assets/rnn-architecture.png)

_Figure 1: RNN Architecture. The hidden state $h_t$ is computed as a function of the current input $x_t$ and the previous hidden state $h_{t-1}$._


The theoretical equation for a hidden state update is:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

In this equation:

- $x_t$ is the current input vector.
- $h_{t-1}$ is the "memory" from the previous time step.
- $h_t$ is the new hidden state (output).
- $W_{xh}$ and $W_{hh}$ are the weight matrices.
- $b_h$ is the bias vector.
- $\tanh$ is the activation function.

---

## 2. From Math to Code: Dimensional Analysis

In a real-world scenario, we process data in **batches** for computational efficiency. This changes how we perform matrix multiplications.

### Linear Transformations

To implement $W_{xh} x_t$ with a batch dimension:

- $x_t$ has shape `(batch_size, input_dim)`
- $W_{xh}$ has shape `(hidden_dim, input_dim)`

If we multiply directly: `(batch_size, input_dim) @ (hidden_dim, input_dim)` → **Dimension Mismatch!**

To fix this, we **transpose** the weight matrix:

- $W_{xh}^T$ has shape `(input_dim, hidden_dim)`
- `x_t @ W_{xh}^T` → `(batch_size, input_dim) @ (input_dim, hidden_dim)` = `(batch_size, hidden_dim)`

The same logic applies to the hidden state transformation:  
`h_prev @ W_hh^T` → `(batch_size, hidden_dim) @ (hidden_dim, hidden_dim)` = `(batch_size, hidden_dim)`

---

## 3. Why the Hyperbolic Tangent ($\tanh$)?

We use $\tanh$ as the activation function for three main reasons:

1.  **Normalization**: It bounds outputs to the range $[-1, 1]$. Without this, repeatedly multiplying values over many time steps could lead to values exploding to infinity.
2.  **Zero-Centered Output**: $\tanh$ outputs are zero-centered, which helps the gradient flow during backpropagation compared to non-zero-centered activations like ReLU or Sigmoid.
3.  **Smoothness**: As a differentiable function, it provides smooth gradients necessary for the optimization process (Gradient Descent).

---

## 4. Key Concept: Weight Sharing

A defining feature of the RNN cell is that the same weights ($W_{xh}, W_{hh}$) and bias ($b_h$) are reused for every single step in a sequence.

**Why?**

- **Parameter Efficiency**: It significantly reduces the number of parameters the model needs to learn.
- **Translation Invariance in Time**: It allows the model to detect a pattern regardless of when it appears in the sequence (e.g., a "subject" followed by a "verb" in a sentence).

---

## 5. Summary of the Implementation

Our NumPy implementation:

```python
h_next = np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)
```

1.  **Linear Combination**: We calculate the weighted influence of both the new input and the old memory.
2.  **Bias Addition**: We shift the result by adding $b_h$. NumPy's "broadcasting" ensures $b_h$ is added to every sample in the batch.
3.  **Non-Linearity**: We apply $\tanh$ to introduce non-linear relationships, allowing the network to learn complex patterns.
