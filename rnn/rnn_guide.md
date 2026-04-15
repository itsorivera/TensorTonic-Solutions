# RNN Architecture Guide: Synthesis & Integration

This document maps the modular components found in the `rnn/` subdirectories to their functional implementation. While the subdirectories isolate concepts for rigorous study, this guide explains how they synthesize into a complete recurrent system.

## Architectural Mapping

### 1. State Initialization

**Subdirectory:** `rnn-hidden-state`
**Role:** Defining the initial memory buffer ($h_0$) before any sequence data is processed.

- **Logic:** In `rnn-full-network/rnn-full-network.py` (Line 22), we initialize as zeros: `h_0 = np.zeros((batch_size, self.hidden_dim))`.
- **Engineering Note:** While zeros are standard, for some tasks, $h_0$ can be a learnable parameter.

### 2. The Recurrent Nucleus (The Cell)

**Subdirectory:** `rnn-cell`
**Role:** The fundamental transformation that blends current input with past memory.

- **Math:** $h_t = \tanh(x_t W_{xh}^T + h_{t-1} W_{hh}^T + b_h)$
- **Logic:** Cleanly isolated in `rnn-cell/rnn-cell.py` (Line 8) and integrated in the sequence loop of `rnn-full-network/rnn-full-network.py` (Line 31).

### 3. Temporal Unrolling

**Subdirectory:** `rnn-forward-sequence`
**Role:** Managing the flow of the hidden state across the time dimension ($T$).

- **Logic:** In `rnn-full-network/rnn-full-network.py` (Lines 28-33), the `for t in range(seq_length):` loop handles the sequential dependency.
- **Constraint:** This part is inherently $O(T)$ sequential, unlike the parallelizable nature of Transformers.

### 4. Weight Initialization (Rigor)

**Integrated in:** `rnn-full-network`
**Role:** Preventing signal degradation at initialization.

- **Logic:** `rnn-full-network/rnn-full-network.py` (Lines 8-10) implements **Xavier Initialization**: `np.sqrt(2.0 / (fan_in + fan_out))`.
- **Purpose:** Essential to keep activations within the non-saturated regions of the $\tanh$ function.

### 5. Output Projection

**Integrated in:** `rnn-full-network`
**Role:** Mapping the latent memory ($H$) to the observable output space ($Y$).

- **Logic:** `rnn-full-network/rnn-full-network.py` (Lines 40-41) uses the "Reshaping Trick" to transform `(Batch, Time, Hidden)` -> `(Batch*Time, Hidden)` for a single efficient matrix multiplication.

### 6. Backpropagation Through Time (BPTT)

**Subdirectory:** `rnn-bptt`
**Role:** Extending the chain rule across the temporal sequence.

- **Logic:** `rnn-bptt/rnn-bptt.py` (Lines 12-22) calculates gradients for $W_{hh}$ and the previous hidden state $h_{t-1}$.
- **Critical Step:** The gradient $\partial h_t / \partial h_{t-1}$ involves the derivative of tanh: $1 - h_t^2$ (Line 12).

---

## 🔬 The "First-Principles" Analysis: Vanishing Gradients

**Subdirectory:** `rnn-vanishing-gradients`

This is where the engineering rigor meets research. By analyzing the **Spectral Norm** of $W_{hh}$ (isolated in `rnn-vanishing-gradients/rnn-vanishing-gradients.py`), we can mathematically predict if the network will effectively learn long-term patterns or if the signal will decay into numerical noise.

### Why this Synthesis Matters

By deconstructing the RNN into these pieces, we reveal:

1.  **Memory Bottleneck:** Why forcing everything into a single fixed-size $h_t$ limits capacity.
2.  **Gradient Flow:** The mathematical justification for the transition to LSTMs (which use a "Constant Error Carousel" to solve the issues found in `rnn-vanishing-gradients`).
