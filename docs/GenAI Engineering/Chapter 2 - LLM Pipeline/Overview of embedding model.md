----

description: Definition and overview of embedding models
----

Add check overview: <https://huggingface.co/spaces/hesamation/primer-llm-embedding?section=what_makes_a_good_embedding>?

# What are embeddings ?

Embeddings are at the code of Language, Vision and Speech models.
Simply put, it is the process of encoding an object (text, image, audio ...) into a vector representations.
Once represented as a vector, we can calculate metrics that renders the quality of such an embedding model compared to new objects of the same family.
Vectors are usually high-dimensional and we want to transform them into our lower-dimensional representation, which we call embeddings or also latent space.
This latent space captures the important features and similarities in a simpler form. For example, an embedding model might take a 2000-word document and represent it in 300-dimensional space, meaning simply 300 numbers in a list, which is much smaller than the original space but still retains the meaningful relationships between words.

Embedding models understand the relationships between words, sentences or objects. The objective is to define, measure and keep a meaningful relationship between these objects.

## Types of embeddings

### 𝗦𝗽𝗮𝗿𝘀𝗲 𝗘𝗺𝗯𝗲𝗱𝗱𝗶𝗻𝗴𝘀

Think keyword-based representations where most values are zero. Great for exact matching but limited for semantic understanding.

### 𝗗𝗲𝗻𝘀𝗲 𝗘𝗺𝗯𝗲𝗱𝗱𝗶𝗻𝗴𝘀

The most common type - every dimension has a value. These capture semantic meaning really well, and come in many different lengths.

### 𝗤𝘂𝗮𝗻𝘁𝗶𝘇𝗲𝗱 𝗘𝗺𝗯𝗲𝗱𝗱𝗶𝗻𝗴𝘀

Compressed versions of dense embeddings that reduce memory usage by using fewer bits per dimension. Perfect when you need to save storage space.

### 𝗕𝗶𝗻𝗮𝗿𝘆 𝗘𝗺𝗯𝗲𝗱𝗱𝗶𝗻𝗴𝘀

Ultra-compressed embeddings using only 0s and 1s. Super fast for similarity calculations but with reduced accuracy.

### V𝗮𝗿𝗶𝗮𝗯𝗹𝗲 𝗗𝗶𝗺𝗲𝗻𝘀𝗶𝗼𝗻𝘀 (𝗠𝗮𝘁𝗿𝘆𝗼𝘀𝗵𝗸𝗮)

These embeddings let you use just the first 8, 16, 32, etc. dimensions while still retaining most of the information. This ability comes during model training: earlier dimensions capture more information than later ones. You can truncate a 3072-dimension vector to 512 dimensions and still get great performance.

### 𝗠𝘂𝗹𝘁𝗶-𝗩𝗲𝗰𝘁𝗼𝗿 (𝗖𝗼𝗹𝗕𝗘𝗥𝗧)

Instead of one vector per object, you get many vectors that represent different parts of your object (like tokens for text, patches for images). This enables "late interaction" - comparing individual parts of texts rather than whole documents. Way more nuanced than single-vector approaches.

𝗦𝗼 𝘄𝗵𝗶𝗰𝗵 𝘀𝗵𝗼𝘂𝗹𝗱 𝘆𝗼𝘂 𝗰𝗵𝗼𝗼𝘀𝗲?

- Dense for general semantic search.
- Matryoshka when you need flexible performance/cost trade-offs.
- Multi-vector for precise text matching.
- Quantized/Binary when storage and speed matter most.

## What are embeddings using transformer techniques

This [cohere's blog](https://docs.cohere.com/docs/text-embeddings) explains why embedding is useful and given the complexity of human language how sentence embedding actually works in simple terms.

A very famous library (from Hugging Face) known to train embedding model is "Sentence Transformer" (<https://sbert.net/>)
Here's an example with a few sentences encoded using the "all-MiniLM" model.

```python
from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
```

The embedding model will always produce embeddings of the same fixed size. You can then compute the similarity of complex objects by computing the similarity of the respective embeddings.
The more dimension, the more "information" we can store in the vector. However, the more computationaly intensive it will be.
So research is aiming at having the best trade-off between size and performance.
Here's an example of embedding model with a dimension reduction while keeping performance relatively equivalent: (<https://huggingface.co/blog/matryoshka>)

## Embedding Model evaluation and metrics

When running embeddings, the performance is often evaluated using the cosine similarity.

## Fine tuning an embedding model

The main reasons you want to fine tune an embedding model are:

- to bring context and vocabulary to your embedding model. Given an encoding strategy, you may or may not have the right context in the original model.
- to have domain specific performance: the same sentence in different domain may relate differently and have different surroundings
- to improve the semantic similarity of your retriever

[Example on how to fine tune with sentence transformers](https://huggingface.co/blog/train-sentence-transformers)

In [this example](https://www.philschmid.de/fine-tune-embedding-model-for-rag) Philipp Schmidt demonstrates a 7% improvement (77% to 83%) fine-tuning the embedding model in the context of RAG.
Moreover, using the "matryoshka" methodology defined above, you can reduce the embedding dimension by a factor of 10 while keeping the same performance.

So fine-tuning is a key friend when you need topic and nlp context because it will also helps you to trade some compute performance against the business performance.
However, from a practical point of view, do not start by fine-tuning.
Start simply by evaluating your solution/pipeline and define criteria for evaluation.
Once you reach a certain threshold comes the time you can focus on each component and twick those.

<https://huggingface.co/blog/embedding-quantization>

## Multi vector embeddings

Single Vector Embeddings
In single vector embeddings, each document or text chunk is represented by one dense vector in high-dimensional space. This is the traditional approach where:

Each document gets tokenized and encoded into a single fixed-size vector (e.g., 768 or 1536 dimensions)
The entire semantic meaning of the document is compressed into this one vector
Vector databases store and search using these single representations
Similarity search finds the closest vectors using cosine similarity or other distance metrics

Multi-Vector Embeddings
Multi-vector embeddings represent each document using multiple vectors rather than just one. This approach:

Splits documents into smaller chunks or uses different embedding strategies
Creates multiple vectors per document to capture different aspects or granularities
Can represent hierarchical information (document-level, paragraph-level, sentence-level)
Allows for more nuanced retrieval by matching against the most relevant vector representation

## VectorDBs

docs/img/vectordb_weaviate.jpeg

## Resources

<https://encord.com/blog/embeddings-machine-learning/?utm_source=linkedin>
<https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings>
<https://weaviate.io/blog/muvera>

[single_vs_multi](../../img/vector_embeddings_diagram.html)
