# LLM Tokenizers: ew and most recent techniques

## What Are Tokenizers?

Tokenizers are fundamental components that serve as the bridge between human language and machine-readable data. They break down text into discrete components called "tokens"  which are numerical representations that Large Language Models (LLMs) can understand.
Tokenization occurs as a preprocessing step in the LLM pipeline:

```
Raw Text → Tokenizer → Token IDs → Embeddings → Model Processing → Output
```

## How Tokenizers Work

### Basic Process

1. **Text Input**: Raw text is provided to the tokenizer
2. **Segmentation**: Text is broken down into smaller units (tokens)
3. **Vocabulary Mapping**: Each token is mapped to a unique numerical ID
4. **Numerical Output**: The sequence of token IDs is passed to the model

### Token Types

- **Character-level**: Individual characters become tokens
- **Word-level**: Whole words become tokens  
- **Subword-level**: Parts of words become tokens (most common in modern LLMs)

## Most Common Tokenization Techniques

### 1. Byte Pair Encoding (BPE)

**How it works**: BPE is a data compression technique adapted for NLP that starts with individual characters and iteratively merges the most frequent pair of tokens to form new, longer tokens.

**Process**:

- Initialize vocabulary with individual characters
- Find the most frequent pair of consecutive tokens
- Merge this pair into a new token
- Repeat until desired vocabulary size is reached

**Used by**: GPT models, RoBERTa

### 2. WordPiece

**How it works**: Similar to BPE but uses likelihood-based merging instead of frequency-based merging. WordPiece chooses symbol pairs that result in the largest increase in likelihood upon merging.

**Key differences from BPE**:

- Uses likelihood maximization for pair selection
- Places "##" at the beginning of subword tokens
- Greedy algorithm leveraging likelihood instead of count frequency

**Used by**: BERT, DistilBERT

### 3. SentencePiece

**How it works**: A re-implementation of subword units that treats the input as a raw input stream without pre-tokenization. It supports both BPE and Unigram algorithms and can handle any language without requiring language-specific preprocessing.

**Key features**:

- Language-agnostic approach
- No pre-tokenization required
- Supports multiple segmentation algorithms
- Handles whitespace as regular characters

**Used by**: T5, ALBERT, XLNet

### 4. Unigram Language Model

**How it works**: Unlike BPE and WordPiece which build vocabulary bottom-up, Unigram starts with a large vocabulary and progressively removes tokens that have the least impact on the overall likelihood of the training data.

**Process**:

- Start with large initial vocabulary
- Iteratively remove tokens with minimal impact on likelihood
- Continue until desired vocabulary size is reached

### State-of-the-Art Considerations

Recent techniques combine multiple tokenization strategies.
Tokenizer can be trained on domain specific data

## Performance Factors

- **Vocabulary Size**: Larger vocabularies can capture more semantic information but increase computational cost
- **Compression Ratio**: Balance between token efficiency and semantic preservation  
- **Language Coverage**: Ability to handle multiple languages and special characters
- **Out-of-Vocabulary Handling**: Graceful degradation for unseen text

So all sequences have the same length, you can add padding to the short sentences with masked attention and shorten (truncate) longer sequences.
Most recent techniques deal with that kind of limits.

## Encoding and decoding

Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs.
There are multiple rules that can govern that process, which is why we need to instantiate the tokenizer using the name of the model, to make sure we use the same rules that were used when the model was pretrained.
The second step is to convert those tokens into numbers, so we can build a tensor out of them and feed them to the model.
To do this, the tokenizer has a vocabulary, which is the part we download when we instantiate it.
Again, we need to use the same vocabulary used when the model was pretrained.

Decoding is going the other way around: from vocabulary indices, we want to get a string.
Note that the decode method not only converts the indices back to tokens, but also groups together the tokens that were part of the same words to produce a readable sentence. This behavior will be extremely useful when we use models that predict new text (either text generated from a prompt, or for sequence-to-sequence problems like translation or summarization).

## Tokenizing numbers: a long history

<https://www.artfish.ai/p/how-would-you-tokenize-or-break-down>

## Sources

- [The Technical User's Introduction to LLM Tokenization](https://christophergs.com/blog/understanding-llm-tokenization) - February 28, 2024
- [Introduction to LLM Tokenization - Airbyte](https://airbyte.com/data-engineering-resources/llm-tokenization) - September 3, 2024
- [Tokenization in large language models, explained](https://seantrott.substack.com/p/tokenization-in-large-language-models) - May 2, 2024
- [Hugging Face Tokenizers Course](https://huggingface.co/learn/llm-course/en/chapter2/4)
- [WordPiece: Subword-based tokenization algorithm - Towards Data Science](https://towardsdatascience.com/wordpiece-subword-based-tokenization-algorithm-1fbd14394ed7/) - January 24, 2025
- [Google SentencePiece GitHub Repository](https://github.com/google/sentencepiece)
