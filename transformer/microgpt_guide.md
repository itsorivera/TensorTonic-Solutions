# MicroGPT Architecture Guide

This document maps the components found in the `transformer/` subdirectories to their integrated implementation within [microgpt.py](./microgpt.py). While the subdirectories provide a modular, piece-by-piece breakdown for learning, `microgpt.py` demonstrates how these pieces work together in a functional LLM.

## 🗺️ Architectural Mapping

### 1. Tokenization

**Subdirectory:** `transformers-tokenization`
**Role:** Converts raw text into integer IDs that the model can process.

- **Logic in `microgpt.py`:**
  - **Vocabulary Creation:** Lines 24-27 define the unique characters and the `BOS` (Beginning of Sequence) token.
  - **Encoding:** Line 157 uses `uchars.index(ch)` to convert characters to IDs before feeding them to the model.

### 2. Embeddings

**Subdirectory:** `transformers-embedding`
**Role:** Maps discrete token IDs to continuous vectors (word representations).

- **Logic in `microgpt.py`:**
  - **Initialization:** Line 81 initializes the `wte` (weight token embedding) matrix.
  - **Lookup:** Line 109 retrieves the specific embedding for the current token: `tok_emb = state_dict['wte'][token_id]`.

### 3. Positional Encoding

**Subdirectory:** `transformers-positional-encoding`
**Role:** Injects information about the order of tokens, as Transformers process sequence elements in parallel.

- **Logic in `microgpt.py`:**
  - **Initialization:** Line 81 initializes the `wpe` (weight position embedding) matrix.
  - **Lookup:** Line 110 retrieves the vector for the current index: `pos_emb = state_dict['wpe'][pos_id]`.
  - **Integration:** Line 111 sums the token and position embeddings.

### 4. Layer Normalization (RMSNorm)

**Subdirectory:** `transformers-layer-normalization`
**Role:** Stabilizes training by normalizing the mean and variance of activations. `microgpt.py` uses **RMSNorm**, a modern variant.

- **Logic in `microgpt.py`:**
  - **Implementation:** Lines 103-106 define the `rmsnorm` function.
  - **Usage:** Applied at lines 112, 117, and 137 before moving into attention or MLP blocks (Pre-Norm architecture).

### 5. Multi-Head Attention

**Subdirectories:** `transformers-attention`, `transformers-multi-head-attention`
**Role:** Allows the model to focus on different parts of the input sequence simultaneously.

- **Logic in `microgpt.py`:**
  - **Projections:** Lines 118-120 project the input into Queries (Q), Keys (K), and Values (V).
  - **Heads:** Lines 124-132 contain the loop `for h in range(n_head):` which computes attention for each head independently before concatenating them.
  - **Softmax:** Line 130 applies the attention weights.

### 6. Feed-Forward Network (MLP)

**Subdirectory:** `transformers-feed-forward`
**Role:** A position-wise neural network that processes the output of the attention layer.

- **Logic in `microgpt.py`:**
  - **Structure:** Lines 138-141 implement the two-layer MLP.
  - **Activation:** Line 139 uses the `.relu()` activation function.

### 7. Transformer Block (The Assembly)

**Subdirectory:** `transformers-encoder-block`
**Role:** The fundamental unit that combines Attention, Feed-Forward, and Residual Connections.

- **Logic in `microgpt.py`:**
  - **Cycle:** Lines 114-141 represent one complete Transformer layer.
  - **Residual Connections:** Lines 134 and 141 perform the `x + residual` addition, which is crucial for training deep networks.

---

## 🧠 The "Glue": Autograd Engine

While not a subdirectory in the `transformer/` folder, the **`Value` class (Lines 30-73)** is the heart of the script. It implements a mini-Autograd engine that:

1.  **Tracks operations:** Building a computation graph automatically.
2.  **Backpropagation:** Using the `backward()` method to calculate gradients using the chain rule.

This engine is what allows the model to "learn" by updating the parameters in the **Adam Optimizer (Lines 147-184)**.

---

## 💡 How to use this Guide

1.  **Bottom-Up:** Study a specific subdirectory (e.g., `transformers-attention`) to understand the math.
2.  **Integration:** Use this guide to find where that math lives in `microgpt.py`.
3.  **Observation:** Run `microgpt.py` to see how all these pieces collaborate to generate names (the inference part starts at Line 187).
