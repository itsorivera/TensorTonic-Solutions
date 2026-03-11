# Multi-Head Attention: Concept and Implementation

Multi-Head Attention is an enhancement of the attention mechanism that allows the model to simultaneously attend to information from different representation subspaces. Instead of computing a single attention pass, it performs multiple attention operations in parallel.

---

## 🧠 Why Multi-Head?

While single-head attention can focus on specific parts of a sequence, Multi-Head Attention allows the model to:

1.  **Attend to multiple positions** at once from different perspectives.
2.  **Separate representation subspaces**: One head might focus on syntactic relationships, while another focuses on semantic ones.
3.  **Improve stability**: Parallel heads help the model capture more robust features.

---

## 📐 The Mechanism

The operation is defined as:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}\_1, \dots, \text{head}\_h)W^O $$

Where each head is:

$$ \text{head}\_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

And:

- $Q, K, V$ are Queries, Keys, and Values.
- $W_i^Q, W_i^K, W_i^V$ are projection matrices for each head.
- $\text{Attention}$ is the Scaled Dot-Product Attention.
- $h$ is the number of heads.
- $d_k = d_{v} = d_{model} / h$.

---

## 🚀 Step-by-Step Implementation

### 1. Linear Projections

We project the input $Q, K, V$ into $d_{model}$ dimensions. In practice, we do this once for all heads combined and then split them.

```python
Q = Q @ W_q
K = K @ W_k
V = V @ W_v
```

### 2. Splitting into Heads

We reshape and transpose the tensors so that the heads are in a separate dimension.

- Shape change: `[batch, seq_len, d_model]` $\rightarrow$ `[batch, h, seq_len, d_k]`

```python
Q = Q.reshape(batch, seq, h, d_k).transpose(0, 2, 1, 3)
```

### 3. Scaled Dot-Product Attention

We apply attention to each head independently. The calculation is done in parallel using matrix multiplication.

```python
scores = (Q @ K.T) / sqrt(d_k)
weights = softmax(scores)
context = weights @ V
```

### 4. Concatenation and Projection

We merge the heads back into a single vector and apply a final linear projection $W^O$.

- Shape change: `[batch, h, seq_len, d_k]` $\rightarrow$ `[batch, seq_len, d_model]`

```python
context = context.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
output = context @ W_o
```

---

## 🎨 Visual Workflow

![Multi-Head Attention Dynamics](/transformer/assets/multi-head-attention.png)
