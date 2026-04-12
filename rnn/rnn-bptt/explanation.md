# Backpropagation Through Time (BPTT): Low-Level Gradient Derivation

Implementing the backward pass of a Recurrent Neural Network (RNN) requires a rigorous application of the chain rule across temporal dependencies. This process is known as **Backpropagation Through Time (BPTT)**.

In this exercise, we implemented a single step of BPTT, which forms the building block for training RNNs.

---

## 1. The Recursive Gradient Flow

In an RNN, the state $h_t$ depends on $h_{t-1}$. When we compute the gradient of the loss $L$ with respect to the parameters, the gradient must flow backwards from $h_t$ to $h_{t-1}$.

The total gradient at time $t$ ($\frac{\partial L}{\partial h_t}$) is typically the sum of:

1.  The gradient from the loss at the current time step (if applicable).
2.  The gradient flowing back from the future time step $t+1$.

---

## 2. Mathematical Derivation

### The Forward Step (Recap)

$$z_t = h_{t-1} W_{hh}^T + x_t W_{xh}^T + b_h$$
$$h_t = \tanh(z_t)$$

### The Backward Step

To compute the local gradients, we apply the chain rule in reverse order:

#### A. Gradient through the Activation ($\tanh$)

The derivative of $\tanh(z)$ is $1 - \tanh^2(z)$. Since we already have $h_t = \tanh(z_t)$, we can compute the pre-activation gradient ($d z_t$) directly:
$$d z_t = \frac{\partial L}{\partial h_t} \odot (1 - h_t^2)$$
In code: `dtanh = dh_next * (1 - h_t**2)`

#### B. Gradient w.r.t. Weights ($W_{hh}$)

The recurrent weight matrix $W_{hh}$ contributes to the transformation of $h_{t-1}$. Its gradient is the outer product of the incoming error and the previous state:
$$\frac{\partial L}{\partial W_{hh}} = d z_t^T \cdot h_{t-1}$$
In a batch setting: `dW_hh = np.dot(dtanh.T, h_prev)`

#### C. Gradient w.r.t. Previous State ($h_{t-1}$)

To continue the backpropagation to the previous time step, we need to pass the gradient through the linear transformation:
$$\frac{\partial L}{\partial h_{t-1}} = d z_t \cdot W_{hh}$$
In code: `dh_prev = np.dot(dtanh, W_hh)`

---

## 3. Engineering Insights: Vanishing Gradients

The term `dtanh = dh_next * (1 - h_t**2)` is the primary culprit behind the **Vanishing Gradient Problem**.

- The derivative of $\tanh$ is capped at **1.0** (when $z=0$) and approaches **0.0** as $|z|$ increases.
- During BPTT, we multiply the gradient by these values repeatedly across many time steps.
- If the values in $W_{hh}$ are also small, the gradient shrinks exponentially, effectively "forgetting" long-term dependencies.

---

## 4. Implementation Summary

From an MLE perspective, efficient BPTT implementation relies on:

1.  **Caching**: We must cache $h_t$ and $h_{t-1}$ during the forward pass to use them in the backward pass.
2.  **Vectorization**: Performing these operations across the batch dimension using optimized BLAS routines (via `np.dot`).
3.  **Transposition alignment**: Ensuring that the matrix dimensions match the batch-first convention (`Batch x Hidden`).
