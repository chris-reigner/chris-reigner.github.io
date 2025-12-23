# Transformers (WIP)

## Transformer architecture

The original transformer architecture:
![Transformer Architecture](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers-dark.svg)

## Play with transformers

Yes transformers are fun !
Here is a way to play with them:
<https://poloclub.github.io/transformer-explainer/>

## Attention layers

A key point of transformer models is that they are built with ***attention layers***.
The original technical paper [Attention is all you Need!](https://arxiv.org/abs/1706.03762) will provide you with a deep dive on this.

Evolution of this layer: flash attention etc...

There are two main approaches for training a transformer model:

- Masked language modeling (MLM): Used by encoder models like BERT, this approach randomly masks some tokens in the input and trains the model to predict the original tokens based on the surrounding context. This allows the model to learn bidirectional context (looking at words both before and after the masked word).
- Causal language modeling (CLM): Used by decoder models like GPT, this approach predicts the next token based on all previous tokens in the sequence. The model can only use context from the left (previous tokens) to predict the next token.

Language models work by being trained to predict the probability of a word given the context of surrounding words. This gives them a foundational understanding of language that can generalize to other tasks.

You will find more details and resources from the great [LLM Hugging Face course](https://huggingface.co/learn/llm-course/chapter1/1)

## Merging

<https://huggingface.co/blog/mlabonne/merge-models>

## Resources

<https://ml-course.github.io/master/notebooks/08%20-%20Transformers.html>
