# Transformer Encoder Block

This directory contains the implementation of a full Transformer Encoder Block, joining the individual pieces of the Transformer architecture like Multi-Head Attention, Feed-Forward Network, and Layer Normalization.

## Key Concepts

The Transformer Encoder Block combines several components to process tokens deeply. An encoder is built by stacking these blocks $N$ times. Each block contains two main sub-layers:

1. **Multi-Head Self-Attention Mechanism:** Allows sequence tokens to attend to each other to gather context.
2. **Position-wise Feed-Forward Network:** Applies non-linear mathematical transformations to each token's representation independently.

Crucially, both of these sub-layers are surrounded by:

- **Residual (Skip) Connections:** Help gradient flow backward by adding the input $x$ of the sub-layer directly to its output ($x + \text{Sublayer}(x)$).
- **Layer Normalization:** Normalizes the activation over the feature dimension, stabilizing the internal states and making learning smoother and faster.

## Implementation Details

The overarching equation for the encoder block can be broken into two steps:

1. First sublayer: $x' = \text{LayerNorm}(x + \text{MultiHeadAttention}(x, x, x))$
2. Second sublayer: $\text{output} = \text{LayerNorm}(x' + \text{FeedForward}(x'))$

### 1. `layer_norm(x, gamma, beta)`

Computes the mean and variance along the last dimension (`axis=-1`), normalizes the values, and shifts by the learnable parameters $\gamma$ (scale) and $\beta$ (bias).

### 2. `multi_head_attention(Q, K, V, ...)`

Projects the tokens into Queries, Keys, and Values. It splits them into multiple "heads" to allow the model to jointly attend to information from different representation subspaces. It calculates scaled dot-product attention before concatenating the heads and applying a final linear projection.

### 3. `feed_forward(x, ...)`

A dense, fully connected setup composed of two linear transformations with a ReLU activation in between. It acts on the sequence position-by-position symmetrically.
$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$

### 4. `encoder_block(...)`

Ties all pieces together, enforcing the strict `LayerNorm(x + Layer(x))` rule for both the Multi-Head Attention step and the Feed-Forward step.

## Example

Given an input batch $x$ of shape `(2, 10, 64)` (Batch Size = 2, Sequence Length = 10, Hidden Dimension = 64) and target `num_heads = 8`, the encoder processes this matrix iteratively through MHA and FFN.

The output maintains its shape exactly `(2, 10, 64)`, enabling deep architectural stacking, passing the output of block 1 directly as the input to block 2 without dimensional mismatch.