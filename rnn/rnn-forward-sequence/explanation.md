# Low-Level Sequential Processing: Unrolling the RNN

Processing a single RNN cell is a static operation, but processing a sequence is a **dynamic process**. In this exercise, we implemented the "Forward Pass Through Time," often referred to as **unrolling** the RNN.

---

## 1. The Concept of "Unrolling"

While we often visualize RNNs with a loop arrow in diagrams, in practice, we "unroll" them across the temporal dimension during the forward pass. This means applying the **exact same logic and parameters** ($W_{xh}, W_{hh}, b_h$) to every element in the sequence, one after another.

- **Input**: A sequence $X = [x_1, x_2, \dots, x_T]$
- **State Transition**: $h_t = f(x_t, h_{t-1})$
- **Persistence**: The output $h_t$ at time $t$ becomes the "previous" state $h_{t-1}$ for the next calculation at time $t+1$.

---

## 2. Managing the 3D Input Tensor

In Machine Learning Engineering, we mostly deal with data in specific tensor layouts. For sequential data, the standard layout is:
`[Batch Size, Sequence Length, Input Dimension]`

When we iterate through the sequence:

1.  We slice the tensor along the **Time Axis** (Axis 1): `X[:, t, :]`.
2.  This gives us a 2D matrix of shape `(batch_size, input_dim)` for that specific time step.
3.  We feed this into our RNN logic alongside the current hidden state.

---

## 3. Mathematical Recurrence in Code

The implementation follows the recursive formula:
$$h_t = \tanh(x_t W_{xh}^T + h_{t-1} W_{hh}^T + b_h)$$

From a low-level perspective:

- **Matrix Multiplication**: Using `np.dot` or the `@` operator.
- **Transposition**: Weight matrices are often stored as `(hidden_dim, input_dim)`, so they must be transposed to align with the batch-first data: `(Batch, Input) @ (Input, Hidden)`.
- **Broadcasting**: The bias $b_h$ is a 1D vector of size `(hidden_dim)`. NumPy's broadcasting automatically spreads this addition across all samples in the batch.

---

## 4. Collecting Hidden States: `h_all` vs `h_final`

A common question in RNN design is what to return. This exercise returns both:

1.  **`h_all` (The Hidden History)**:
    - Shape: `(batch_size, seq_length, hidden_dim)`
    - This is used when the next layer in the architecture also expects a sequence (e.g., in Many-to-Many or stacked RNN architectures).
    - We collect these in a list and use `np.stack(..., axis=1)` to rebuild the temporal dimension.

2.  **`h_final` (The Summary)**:
    - Shape: `(batch_size, hidden_dim)`
    - This is the "digest" of the entire sequence. It's often used for classification tasks (Many-to-One) where only the final temporal context is needed.
    - Engineering check: `h_all[:, -1, :]` must always equal `h_final`.

---

## 5. Engineering Considerations: Memory & Efficiency

- **List Appending vs Pre-allocation**: In our implementation, we append to a list and then stack. While pre-allocating an array with `np.zeros` might be slightly more memory-efficient for extremely large sequences, `np.stack` is highly optimized and provides a cleaner loop logic for most use cases.
- **Vanishing Gradients**: Although not directly visible in the forward pass, this unrolling process is exactly why RNNs struggle with long-term dependencies. The same $W_{hh}$ matrix is multiplied repeatedly; if its eigenvalues are less than 1, the signal (and the gradient) eventually disappears.
