# Positional Encoding: Giving Order to Transformers

## 1. Introduction

The Transformer architecture, unlike Recurrent Neural Networks (RNNs) or LSTMs, processes all tokens in a sequence simultaneously (in parallel). This architecture is **permutation invariant**: if you shuffle the words in a sentence, the self-attention mechanism itself would output the exact same representations for each word, just in a different order.

To fix this, we must inject information about the *relative or absolute position* of the tokens in the sequence. We do this by adding a **Positional Encoding** vector to each input embedding.

This directory (`transformers-positional-encoding`) implements the sinusoidal method originally proposed in the paper "Attention Is All You Need".

## 2. Key Concepts

### 2.1 The Problem of Order
In a sentence like "The *dog* chased the *cat*", swapping "dog" and "cat" changes the meaning completely. An embedding layer alone maps "dog" to the same vector $v_{dog}$ regardless of whether it appears first or last. Positional encodings ensure that $v_{dog} + p_{1}$ (dog at pos 1) is distinct from $v_{dog} + p_{4}$ (dog at pos 4).

### 2.2 Additive Injection
We do not concatenate the position information. Instead, we **add** the positional vector element-wise to the token embedding.
$$Input = Embedding(Token) + PositionalEncoding(Position)$$
This works because the high-dimensional space allows the model to learn to separate semantic information (embedding) from positional information (encoding).

## 3. Mathematical Foundation

The positional encoding $PE$ is a matrix of size $(L, d_{model})$, where $L$ is the sequence length and $d_{model}$ is the dimension of the embedding.

For a position $pos$ and dimension $i$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
*   $pos$: The position of the token in the sequence ($0, 1, 2, ..., L-1$).
*   $i$: The dimension index ($0, 1, ..., d_{model}/2 - 1$).
*   $d_{model}$: The size of the embedding vector (e.g., 512).

### 3.1 Frequency and Wavelength
The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.
*   **Low dimensions (small $i$)**: High frequency, change rapidly.
*   **High dimensions (large $i$)**: Low frequency, change slowly.

## 4. Logical Intuition

### 4.1 Why Sinusoids?
1.  **Unique Positions**: Each position $pos$ gets a unique combination of sine and cosine values across the coordinate dimensions.
2.  **Relative Positioning**: For any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$. This allows the model to easily learn to attend to relative positions (e.g., "the word 3 steps back").
    $$\sin(\omega(t+k)) = \sin(\omega t)\cos(\omega k) + \cos(\omega t)\sin(\omega k)$$
3.  **Extrapolation**: Training on short sequences doesn't break the logic for longer sequences during inference (though performance may vary).

### 4.2 Why 10000?
The number 10000 is an arbitrary large constant chosen to create a wide range of frequencies, ensuring that the lowest frequency wave completes only a small part of a cycle even for very long sequences.

## 5. Implementation Details

The Python implementation (`positional_encoding`) uses vectorized NumPy operations for efficiency:

1.  **Div Term Initialization**:
    Instead of computing the division for every position loop, we precompute the frequency term:
    `div_term` $= \exp(2i \cdot -\frac{\ln(10000)}{d_{model}})$
    This is equivalent to $\frac{1}{10000^{2i/d_{model}}}$ but numerically stable.

2.  **Vectorized Position Array**:
    We create a column vector `position` of shape $(seq\_length, 1)$.

3.  **Sine and Cosine Application**:
    *   **Even indices** ($2i$) in the output matrix are filled with $\sin(position \times div\_term)$.
    *   **Odd indices** ($2i+1$) are filled with $\cos(position \times div\_term)$.

This results in a matrix where each row is a unique positional vector ready to be added to your word embeddings.
