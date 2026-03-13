# Transformer Tokenization: A Fundamental Preprocessing Step

## 1. Introduction

Tokenization is the first and most fundamental step in Natural Language Processing (NLP) pipelines, particularly for Transformer models. Before a neural network can process text, the raw strings must be converted into a numerical format. This process involves breaking down text into smaller units called **tokens** and mapping each token to a unique integer ID.

In the context of the implementation provided in this directory (`transformers_tokenization`), we focus on a **Word-Level Tokenizer**. This approach treats each unique word as a distinct token, creating a direct mapping between words and integer indices.

## 2. Key Concepts

### 2.1 Tokens and Vocabulary ($V$)
- **Token**: The atomic unit of text processing. In our implementation, a token is a word (ex: "hello", "world"). In more advanced models (like BERT or GTP), tokens can be sub-words or characters.
- **Vocabulary ($V$)**: The set of all unique tokens known to the model. The size of the vocabulary ($|V|$) determines the dimension of the input embedding layer.

### 2.2 Special Tokens
Transformer models rely on special markers to handle specific sequence processing tasks. Our implementation includes four standard special tokens:

1.  **`<PAD>` (Padding Token)**: Used to fill sequences to a uniform length. Transformers process data in batches (matrices), requiring all sequences in a batch to have the same dimensions.
2.  **`<UNK>` (Unknown Token)**: Represents words that are not in the vocabulary (Out-Of-Vocabulary or OOV words). This prevents the model from crashing on unseen data.
3.  **`<BOS>` (Beginning of Sequence)**: Marks the start of a sentence. Crucial for autoregressive generation tasks.
4.  **`<EOS>` (End of Sequence)**: Marks the end of a sentence, signaling the model to stop generating text.

### 2.3 Word-Level vs. Subword-Level
-   **Word-Level**: Splits text by spaces. Simple and intuitive but suffers from a large vocabulary size and poor handling of rare words.
-   **Subword-Level (BPE, WordPiece)**: Splits words into meaningful sub-parts (e.g., "playing" -> "play", "##ing"). Used by modern Transformers to balance vocabulary size and expressiveness. *Note: Our current implementation is Word-Level.*

## 3. Mathematical Foundation

### 3.1 Mapping Functions

Let $\mathcal{T}$ be the set of strict text strings and $\mathbb{Z}_{\ge 0}$ be the set of non-negative integers.

We define a vocabulary $V$ as an ordered set of unique tokens:
$$V = \{t_0, t_1, t_2, ..., t_{|V|-1}\}$$

The tokenizer defines two primary bijective (or surjective in the case of UNK) mappings:

1.  **Encoding ($E$)**: Maps a sequence of words $W = (w_1, w_2, ..., w_n)$ to a sequence of indices $I$.
    $$E(w) = \begin{cases} 
    \text{index}(w) & \text{if } w \in V \\
    \text{index}(\texttt{<UNK>}) & \text{if } w \notin V 
    \end{cases}$$
    
    The resulting sequence is $I = (E(w_1), E(w_2), ..., E(w_n))$.

2.  **Decoding ($D$)**: Maps a sequence of indices $I$ back to text.
    $$D(i) = t_i \quad \text{where } t_i \in V$$

### 3.2 One-Hot Encoding (Conceptual)
Ideally, each token index $i$ corresponds to a **one-hot vector** $v_i \in \{0,1\}^{|V|}$, where the $i$-th component is 1 and all others are 0.
$$v_{\text{hello}} = [0, 0, ..., 1, ..., 0]$$
In practice, for efficiency, we pass the integer index $i$ directly to an **Embedding Layer**, which performs a lookup operation equivalent to multiplying the one-hot vector by the embedding matrix.

## 4. Logical Intuition & Implementation

### 4.1 Implementation Logic (`SimpleTokenizer`)

The tokenizer is implemented as a class `SimpleTokenizer` with three main logical steps:

1.  **Initialization (`__init__`)**:
    -   We establish two dictionaries: `word_to_id` (string $\to$ int) for encoding and `id_to_word` (int $\to$ string) for decoding.
    -   We define our special tokens first.

2.  **Vocabulary Building (`build_vocab`)**:
    -   **Priority to Special Tokens**: We always assign IDs $0, 1, 2, 3$ to `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` respectively. This convention makes debugging and downstream tensor masking easier.
    -   **Iterative Discovery**: We iterate through the training corpus, splitting text by whitespace. Every *new* unique word found is assigned the next available integer ID.
    -   *Complexity*: This operation is linear with respect to the total number of words in the training corpus $O(N)$.

3.  **Encoding & Decoding**:
    -   **Encode**: We split the input string. For every word, we check if it exists in our `word_to_id` map. If yes, return the ID. If no, logical fallback to the ID of `<UNK>`.
    -   **Decode**: We look up the integer in `id_to_word`.

### 4.2 Why this matters for Transformers?

The Transformer architecture (Attention mechanisms, Feed-Forward Networks) relies entirely on matrix multiplication. It cannot "read" strings. 
-   **Tokenization** bridges the gap between human-readable language (discrete symbols) and machine-processable data (continuous vectors).
-   The quality of tokenization directly impacts the model's performance. A poor tokenizer that treats "Transformers" and "transformers" as unrelated concepts (due to case sensitivity) or produces too many `<UNK>` tokens will severely limit the model's understanding.

## 5. Summary

This `SimpleTokenizer` implementation solves the problem of converting raw text into a numerical format required for the Embedding layer of a Transformer. It introduces the critical concept of a **Vocabulary** and **Special Tokens**, handling the "unknown" via a fallback mechanism, ensuring the robustness of the pipeline.
