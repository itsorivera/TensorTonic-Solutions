# Vanilla RNN Architecture: From Recurrence to Projection

The **Vanilla RNN** (also known as the Elman Network) is the foundational architecture for sequential processing. It combines a recurrent cell for temporal dependency with a projection layer for generating outputs.

---

## 1. Architectural Components

Our implementation consists of two main stages for every time step:

### A. The Recurrent State Update

The core of the RNN is its ability to update a latent representation (hidden state) based on the current input and its own history.
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

- **$W_{xh}$**: Transmits the influence of the current input feature.
- **$W_{hh}$**: Preserves the temporal context from the past.

### B. The Output Projection

Unlike the hidden state (which is designed for memory), the output layer maps the current context to the specific output dimension (e.g., vocabulary size in NLP or number of classes).
$$y_t = W_{hy} h_t + b_y$$

---

## 2. Weight Initialization: Xavier (Glorot)

Proper initialization is critical for training deep architectures. We use **Xavier Uniform/Normal Initialization**, which scales weights based on the number of input and output units (_fan-in_ and _fan-out_).

- **Objective**: Maintain a consistent variance of activations across layers.
- **Formula used**: $\text{std} = \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$
- **Impact**: Prevents signals from shrinking to zero (vanishing) or exploding during the first few passes of training.

---

## 3. Engineering Implementation: The Reshaping Trick

While the recurrence happens step-by-step, the output projection can be optimized for hardware.

1.  **Temporal Loop**: We iterate through the sequence length $T$ to compute $h_t$. This part is inherently sequential and difficult to parallelize over time.
2.  **Matrix Projection**: Once all $h_t$ are collected in a tensor of shape `(Batch, Time, Hidden)`, we **reshape** it to `(Batch * Time, Hidden)`.
3.  **Vectorized Linear Layer**: We perform a single massive matrix multiplication:
    `[Batch * Time, Hidden] @ [Hidden, Output]^T`
4.  **Restoring Shape**: Finally, we reshape back to `(Batch, Time, Output)` to restore the temporal structure.

This approach is significantly faster than performing $T$ individual projections inside the loop.

---

## 4. Input/Output Layouts

- **Input $X$**: `(N, T, I)` - Batch $N$, sequence length $T$, input features $I$.
- **Hidden $H$**: `(N, T, H)` - The "internal memory" for every step.
- **Output $Y$**: `(N, T, O)` - Logits projected for every time step.

---

## 5. Limitations

While elegant, this architecture is rarely used in production today because the **Hidden-to-Hidden recurrence** is highly sensitive to the vanishing/exploding gradient phenomena, making it difficult to learn dependencies beyond 10-20 steps.
