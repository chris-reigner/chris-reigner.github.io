
# What are adapters and why are they important ?

Adapter-based methods add extra trainable parameters after the attention and fully-connected layers of a frozen pretrained model to reduce memory-usage and speed up training. The method varies depending on the adapter, it could simply be an extra added layer or it could be expressing the weight updates ∆W as a low-rank decomposition of the weight matrix. Either way, the adapters are typically small but demonstrate comparable performance to a fully finetuned model and enable training larger models with fewer resources.
This is an additional module injected to a frozen base model. Each PEFT method can use one or more types of adapters.
<https://huggingface.co/docs/peft/conceptual_guides/adapter>

This [blog](https://sebastianraschka.com/blog/2023/llm-finetuning-llama-adapter.html) offers more technical details.


# Fine tuning techniques (WIP)

## Supervised Fine Tuning (SFT)

Given access to a pre-trained model, full fine-tuning, which involves adjusting all the model’s weights for a new task, is a viable approach.
However, this method can be resource-intensive and may pose risks such as overfitting, especially with smaller datasets, or catastrophic forgetting.
Supervised finetuning, involves training the model with instruction-output pairs, where the instruction serves as the input and the output is the model’s desired response.

This stage uses smaller datasets compared to pretraining, as creating instruction-output pairs requires significant effort, often involving humans or another high-quality LLM.
The process focuses on refining the model’s ability to produce specific outputs based on given instructions.

### What is the difference between transfer learning and SFT ?

Transfer learning focuses on how to transfer the knowledge gained from one task to accelerate learning for a new, related task or domain.
The key benefits is the small amount of data required to perform transfer learning with the little !!!!! probability for overfitting given the amount of data
Another transfer learning option is feature based (embeddings then task e.g. classification)

### What are the limits of fine-tuning ?

- knowledge that is too far (new language) than pre-training can lead to poor results
- Fine-tuning on a new knowledge could increase hallucinations and could erase some knowledge that generalized well (catastrophic forgetting)


# Why memory is such a big deal ?

### How to estimate the memory consumption for SFT ?

Estimated memory consumption: parameters + gradients + optimizer states + activations

## Estimate memory requirements for full fine-tuning LLM ?

We'll go through a common example: Mistral Instruct 7 Billion

- Model parameters memory: assuming it occupies 4 bytes (32-bits), the model needs 7e9 * 4 bytes = 28GB
- Gradient calculation: similar to the model memory for float32 i.e. 28 GB
- Backward pass: need to compute and store intermediate activations for backpropagation: Number of layers *size of activations per layer* batch size = Memory required for storing intermediate activations. It can be estimated for Mistral (not public) to 54.98 GB
- Optimizer step: estimated to 109.96 GB for float 32

# Alternatives to fine-tuning

Alternatives to tuning a model (In context learning ; Zero/Few shot inference)

## Pruning and distillation

<https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model>
<https://snorkel.ai/blog/llm-distillation-demystified-a-complete-guide/>

## Domain tuning

## Transfer learning


## Data preparation for large model tuning

# full fine tuning

# Long context scaling

<https://www.gradient.ai/blog/scaling-rotational-embeddings-for-long-context-language-models>
