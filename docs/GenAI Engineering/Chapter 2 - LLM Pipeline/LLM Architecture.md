
## Non determinism in LLMs

If you're intestested, you can go through the entire articile of [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
You can use their code to make your LLM deterministic: <https://github.com/thinking-machines-lab/batch_invariant_ops>

Here's the TLDR: You can bring down the temperature to 0, the non-determinism can still exist because of kernels behavior.

• **The Problem**: LLM inference produces different outputs for identical inputs, even at temperature 0, making reproducible results nearly impossible.
• **Common Misconception**: The widely believed "concurrency + floating point" hypothesis suggests GPU parallelism causes nondeterminism, but this misses the real issue.
• **Root Cause**: Floating-point non-associativity means (a+b)+c ≠ a+(b+c), causing different results when numbers are added in different orders.
• **Real Culprit (Simple Example)**: Imagine you ask ChatGPT "What's 2+2?" When the server is busy with 100 other users, your request gets processed in a batch of 101.
When it's quiet, your request gets processed alone in a batch of 1. Even though it's the same question, the math operations inside the model happen differently because of
the different batch sizes - like doing homework alone vs. in a crowded classroom where the teacher has to handle everyone differently.
• **Forward Pass Reality**: Individual LLM forward passes are actually deterministic, but the system becomes nondeterministic due to varying batch contexts.
• **Solution Strategy**: Use fixed reduction strategies regardless of batch size, avoiding split-reduction techniques that change based on parallelism needs.
• **Performance Trade-off**: Batch-invariant kernels sacrifice some performance (about 20% slower) but maintain mathematical correctness.
• **Real Impact**: Tests showed 80 unique completions from 1000 identical requests, but batch-invariant kernels produced identical results every time.
• **Broader Benefits**: True determinism enables genuine on-policy reinforcement learning and eliminates the need for off-policy corrections in training.

Most Modern LLMs use decoder-only architecture.

<https://github.com/rasbt/LLMs-from-scratch>

## Positional encoding

<https://github.com/harrisonpim/positional-embeddings?tab=readme-ov-file>

## Mixture of Experts

<https://huggingface.co/blog/moe#fine-tuning-moes>
<https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts>

New release of expert parallelism using transformers directly:
<https://huggingface.co/blog/faster-transformers>
<https://www.linkedin.com/posts/akshay-pachaar_youre-in-an-ml-engineer-interview-at-mistralai-activity-7383487927080857600-v5yN?utm_source=share&utm_medium=member_android&rcm=ACoAAAVV2dEBAuuJCv1jGmfAXdBgR9YAUI0StlM>

<https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison>

## Build from scratch

- Qwen3 from scratch: <https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3>
-
