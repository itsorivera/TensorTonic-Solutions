# RNN Architecture: Hidden State Fundamentals

The hidden state mechanism is the core component that enables Recurrent Neural Networks (RNNs) to process sequential data by maintaining a persistent internal representation of information across time steps.

## 1. Recurrent Memory Mechanism
Unlike standard feedforward networks that treat inputs independently, RNNs utilize a temporal feedback loop. The **Hidden State ($h_t$)** serves as a compressed vector representation of all preceding inputs in the sequence.

Formally, the transition is defined as:
$$h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

Where:
- $h_{t-1} \in \mathbb{R}^{B \times H}$: Previous hidden state (context from the past).
- $x_t \in \mathbb{R}^{B \times I}$: Current input at time step $t$.
- $W_{hh} \in \mathbb{R}^{H \times H}$: Hidden-to-hidden weight matrix.
- $W_{xh} \in \mathbb{R}^{I \times H}$: Input-to-hidden weight matrix.
- $b_h \in \mathbb{R}^{H}$: Bias vector.
- $\sigma$: Non-linear activation function, typically `tanh`.

## 2. Zero Initialization Rationale ($h_0$)
At the first time step ($t=1$), the network lack any prior context. Initializing $h_0$ as a zero vector is the industry standard for the following reasons:
- **Activation Neutrality**: In a `tanh` activation space (ranging from -1 to 1), zero represents a neutral starting point. This prevents biasing the network's initial state before any data has been observed.
- **Gradient Flow**: Starting at zero avoids the saturation regions of the `tanh` function ($f'(0) = 1$), which facilitates stable gradient backpropagation during the initial stages of training.
- **Deterministic Baseline**: It ensures consistent behavior across different sequences within a batch, providing a stable reference for the weight updates.

## 3. Tensor Dimensionality
In machine learning engineering, operations are vectorized for computational efficiency. The initial hidden state must adhere to the shape **$(B, H)$**:
- **Batch size ($B$)**: Enables parallel execution on SIMD (Single Instruction, Multiple Data) architectures like GPUs.
- **Hidden dimension ($H$)**: A hyperparameter that determines the network's representational capacity. A higher $H$ allows for more complex memory patterns but increases computational overhead and the risk of overfitting.

## 4. Precision and Data Types: `float32` vs `float64`
While numerical analysis often defaults to 64-bit precision, deep learning frameworks predominantly use **`float32`** (Single Precision):
- **Computational Efficiency**: GPUs and specialized AI accelerators (like TPUs) are optimized for 32-bit (and increasingly 16-bit) arithmetic.
- **Memory Bandwidth**: Reducing precision halves memory traffic, which is often the primary bottleneck in training high-performance models.
- **Regularization Effect**: The slight noise introduced by lower precision can act as a subtle form of regularization, and high precision is rarely required for the convergence of stochastic gradient descent.

## 5. Architectural Implications: Vanishing Gradients
A rigorous understanding of Vanilla RNNs requires acknowledging the **Vanishing Gradient Problem**. Because the hidden state is updated via repeated matrix multiplications and non-linearities, the gradient of the loss with respect to early time steps ($h_1$) decays exponentially. This fundamental limitation led to the development of gated architectures like LSTMs and GRUs, which utilize "cell states" to mitigate information decay.

---

## Key Concept Review

**1. What is $h_t$?**
It is a compressed vector representation of the entire sequence history up to time step $t$.

**2. Why initialize with zeros?**
It represents a complete "lack of prior information" and serves as a neutral starting point within the `tanh` activation range.

**3. Why use the shape $(B, H)$?**
It enables vectorization for high computational efficiency (Batching) and provides a dedicated space for feature representation (Hidden Dimension).
