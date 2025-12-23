# LLM Inference Optimization techniques (WIP)

## API vs Model serving

The inference optimization techniques should be relevant to the type of serving you choose.
It is clear that invoking an LLM model served on a cluster or invoking an API lead to different trade-offs and optimization.
Yet, the main metrics one can follow are similar.

## Embrace async

## LLM serving optimization

There are 3 main strategies of optimized inference (data, model and system).

*Source: A Survey on Efficient Inference for Large Language Models*

### Model optimization

There are two main steps:

- prefill: processing the full prompt length at once and cache intermediate steps (kv caching). It contributes to little latency.
- decode: new tokens generation based on all (or partially) previous tokens. Harder to paralllelized to it is the part that takes the most time

#### Quantization

Quantization is a technique that reduces the precision of the tensors in the model.
A few library like bitsandbites, trl, unsloth... allows you to run quantize your model so it's faster at inference.

Read more in this article: <https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html>

### System-level optimization

### Batch inference

As the decoding stage is taking time, batching multiple queries together improve the utilization and enables processing multiple requests at once.

*Source: <https://www.baseten.co/blog/continuous-vs-dynamic-batching-for-ai-inference/>*

Continuous batching can be considered a variation of Dynamic Batching, only that it works at the token level instead of the request level. For continuous batching, a layer of the model is applied to the next token of each request.
In this manner, the same model weights could generate the N’th token of one response and the Nx100 token of another. Once a sequence in a batch has completed generation, a new sequence can be inserted in its place, yielding higher GPU utilization than static batching.

### Decoding methods

As decoding is taking the most time, optimizing this step is the most important. In time, users moved through a few methods:

- greedy decoding: you take the most probably token at each step which is clearly a limit
- beam search: you take the n most likely tokens to be sampled to select a token from the top k likely options
- speculative decoding: you do not need all previous tokens you rather make educated guesses about future tokens all within a forward pass. Eventually, a verification mechanism ensure the robustness.

You can find more details about decoding techniques [here](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)

Speculative decoding explained [here](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)

### Key-value cache and optimized attention mechanism

Key-value caching are now embedded into most transformers based library.
As the decode phase generates a single token at each time step. Still, each token depends on all previous tokens' key and value tensors (including the input tokens’ KV tensors computed at prefill and any new KV tensors computed until the current time step).

To avoid recomputing all these tensors for all tokens at each time step, it’s possible to cache them in GPU memory.

### Parallelization Optimization Techniques

One way to reduce the per-device memory footprint of the model weights is to distribute the model over several GPUs. This enables the running of larger models or batches of inputs. Based on how the model weights are split, there are three common ways of parallelizing the model: pipeline, sequence, and tensor.

**Pipeline parallelism** involves sharding the model (vertically) into chunks, each comprising a subset of layers executed on a separate device. Thus, each device's memory requirement for storing model weights is effectively quartered.

**Limitation**: Because this execution flow is sequential, some devices may remain IDLE while waiting for the output of the previous layers.

**Tensor parallelism** involves sharding (horizontally) individual layers of the model into smaller, independent blocks of computation that can be executed on different devices.
In transformer models, the Attention Blocks and MLP (normalization) layers will benefit from Tensor Parallelism because large LLMs have multiple Attention Heads. To speed up the computation of Attention Matrices, which is done independently and in parallel, we could split them into one per device.

**Tensor parallelism has limitations**. It requires layers to be divided into independent, manageable blocks. This does not apply to operations like LayerNorm and Dropout, which are replicated across the tensor-parallel group. These layers are computationally inexpensive but require considerable memory to store activations.

To mitigate this bottleneck, **sequence parallelism** partitions these operations along the “sequence dimension” where the Tensor Parallelised layers are, making them memory efficient.

### Data-level optimization

Prompt compression and prompt caching are discussed in the next sections as they are eligible for API inference.

## API inference

### Prompt caching

Model prompts often contain repetitive content, like system prompts and common instructions. routes API requests to servers that recently processed the same prompt, making it cheaper and faster than processing a prompt from scratch (as explained above).
This can reduce latency by up to 80% and cost by up to 75% according to OpenAI. Recent model all offers this behavior by default.

**Tips**:
Cache hits are only possible for exact prefix matches within a prompt. To realize caching benefits, place static content like instructions and examples at the beginning of your prompt, and put variable content, such as user-specific information, at the end. This also applies to images and tools, which must be identical between requests.

The following elements are cached:

- messages
- images
- tool use
- structured outputs

### Prompt compression

Prompt compression is the process of systematically reducing the number of tokens fed into a large language model to retain or closely match the output quality comparable to that of the original, uncompressed prompt.
This leads to cost reduction, speed improvement and token limits optimization.

One of the main challenges in prompt compression is maintaining the essential context while reducing prompt length. When too much information is removed, the LLM may: misinterpret the user’s intent, provide vague or irrelevant responses, omit critical details necessary for accurate answers

Mitigation strategies are:

- Start first by basic summarization before using more advanced techniques
- Set a compression threshold to preserve most context
- In some cases, you just need entities to be served directly in the prompt. Use pydantic to extract and keep defined format.

## Metrics to monitor

- Time to First Token (TTFT): The time it takes for the first token to be generated
- Time between Tokens (TBT): The interval between each token generation
- Tokens per Second (TPS) or throughput: The rate at which tokens are generated
- Time per Output Token (TPOT): The time it takes to generate each output token
- Total Latency: The total time required to complete a response
- GPU Utilization (models using GPU): percentage of time during which the GPU is actively processing tasks



## Main bottelnecks

There are 2 major bottlenecks: 
- compute-bound (computation needed) 
- memory bandwidth (moving data between CPU and GPU for instance)

## Sources

Great resource on LLM optimization: <https://multimodalai.substack.com/p/understanding-llm-optimization-techniques>
KV cache explained: <https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms>
<https://bentoml.com/llm/inference-optimization>
continuous batching : <https://huggingface.co/blog/continuous_batching>

