# Transformer Tokenization: A Fundamental Preprocessing Step

## 1. Architectural Context

Tokenization is **Step 0** in any language model. Before the Transformer can perform matrix multiplications with Embeddings or Attention, raw text must be converted into integers (IDs).

**Flow:**
`Raw Text` $\rightarrow$ `Tokenizer` $\rightarrow$ `List of IDs` $\rightarrow$ `Embedding Layer`

In this implementation (`transformers_tokenization`), we focus on a **Word-Level Tokenizer**. It treats each unique word as a token, creating a direct mapping between words and integer indices.

## 2. Key Concepts

### 2.1 Tokens and Vocabulary ($V$)

- **Token**: The atomic unit. Here a token is a whole word ("hello", "world").
- **Vocabulary ($V$)**: The set of all known tokens. The vocabulary size ($|V|$) determines the dimension of the network's first Embedding layer.

### 2.2 Special Tokens

Crucial markers for processing sequences:

1. **`<PAD>` (0)**: Pads sequences so that all in a batch have the same length.
2. **`<UNK>` (1)**: Out-of-vocabulary words.
3. **`<BOS>` (2)**: Marks the Beginning Of Sequence.
4. **`<EOS>` (3)**: Marks the End Of Sequence.

## 3. Mathematical Foundation

### 3.1 Mapping Functions

The vocabulary $V$ is an ordered set of unique tokens:
$$V = \{t_0, t_1, t_2, ..., t_{|V|-1}\}$$

Bijective Mapping:

- **Encoding ($E$)**: Text to Index (ID).
  $$E(w) = \text{index}(w) \text{ if } w \in V \text{ (else, index(<UNK>))}$$
- **Decoding ($D$)**: Index to Text.

## 4. Tensor Shapes

Although the Tokenizer produces native Python lists, conceptually the input/output for the model looks like this:

- **Input (`encode`)**: `String` or `List[String]`
- **Output (`encode`)**: `List[int]` (Typically converted to a 1D or 2D Tensor of shape `(batch_size, seq_len)` before entering the Transformer).

## 5. Minimal Executable Example (Unit Example)

```python
from transformers_tokenization import SimpleTokenizer

text = "The dog chased the cat"
corpus = [text]

# 1. Instantiate and build vocabulary
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(corpus)

# 2. Encode (Text -> IDs)
encoded = tokenizer.encode(text)
print(f"Encoded: {encoded}")
# Expected output: Encoded: [4, 5, 6, 7, 8]  (Assuming indices after special tokens)

# 3. Decode (IDs -> Text)
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
# Expected output: Decoded: the dog chased the cat
```
