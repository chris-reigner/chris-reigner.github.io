
# What is encoding ?

The encoder part in the original transformer is responsible for understanding and extracting the relevant information from the input text.
It then outputs a continuous representation (embedding) of the input text that is passed to the decoder.

# What is decoding ?

To illustrate and understand decoding, you can follow this [nice post from Maxime Labonne](https://huggingface.co/blog/mlabonne/decoding-strategies).
It explores how does a model like GPT2 "produces" the text.

Actually the tokenizer translates each token from the input text into a corresponding token ID.
Then the LLMs calculate logits which are scores assigned to every possible token in the vocabulary.
They are then converted into probabilities using a softmax function.
Eventually, how these probabilities are used to select the next token is part of the decoding strategy.

# Encoder only models

BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only architecture based on the Transformer's encoder module. The BERT model is pretrained on a large text corpus using masked language modeling (illustrated in the figure below) and next-sentence prediction tasks.
The main idea behind masked language modeling is to mask (or replace) random word tokens in the input sequence and then train the model to predict the original masked tokens based on the surrounding context.

Bert are bidirectional which means they can learn representations of the text by attending to words on both sides.

These models are good for tasks that require understanding the input like sentence classification or named entity recognition.

## Decoder only models

Over the years, researchers have built upon the original encoder-decoder transformer architecture and developed several decoder-only models that have proven to be highly effective in various natural language processing tasks. The most notable models include the GPT family.
The GPT (Generative Pre-trained Transformer) series are decoder-only models pre-trained on large-scale unsupervised text data and finetuned for specific tasks such as text classification, sentiment analysis, question-answering, and summarization.
Compared to Bert, they can only use context from the previous tokens to predict the next token.

They are good at generative tasks like generate text or image.

For instance, GPT-2 uses byte pair encoding (BPE) to tokenize words and generate a token embedding. Positional encodings are added to the token embeddings to indicate the position of each token in the sequence. The input embeddings are passed through multiple decoder blocks to output some final hidden state. Within each decoder block, GPT-2 uses a masked self-attention layer which means GPT-2 can’t attend to future tokens. It is only allowed to attend to tokens on the left. This is different from BERT’s [mask] token because, in masked self-attention, an attention mask is used to set the score to 0 for future tokens.

The output from the decoder is passed to a language modeling head, which performs a linear transformation to convert the hidden states into logits. The label is the next token in the sequence, which are created by shifting the logits to the right by one. The cross-entropy loss is calculated between the shifted logits and the labels to output the next most likely token.

## Encoder-decoder models or sequence-to-sequence

They are good at generative tasks that require an input such as translation or summarization
Next to the traditional encoder and decoder architectures, there have been advancements in the development of new encoder-decoder models that leverage the strengths of both components. These models often incorporate novel techniques, pre-training objectives, or architectural modifications to enhance their performance in various natural language processing tasks
Recently, these models have lost popularity towards decoder only models.

<https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder>

## Cross encoder / bi-encoder

<https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings2/>
<https://weaviate.io/blog/cross-encoders-as-reranker>

## Speech and Audio data

Whisper is an encoder-decoder (sequence-to-sequence) transformer pretrained on 680,000 hours of labeled audio data. This amount of pretraining data enables zero-shot performance on audio tasks in English and many other languages. The decoder allows Whisper to map the encoders learned speech representations to useful outputs, such as text, without additional fine-tuning. Whisper just works out of the box.

## Image classification

## <https://huggingface.co/blog/mlabonne/decoding-strategies>
