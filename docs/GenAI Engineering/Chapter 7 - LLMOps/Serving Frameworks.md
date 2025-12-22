# Open Source Frameworks

## Model Inference

You'll need to choose whether you want to run model inference locally (windows or MAC) or on cloud/premise infrastructure.
You can switch from one environment to another but generally speaking, you need to run small language models (SLMs) quantized or not locally.
You (usually) have only 1 GPU locally while starting quite fast having multiple on cloud so the considerations are different.

There are many different options, so you need to ask yourself the following questions:

- do I need a managed serving engine or do I want to customize the serving ?
- do I want to serve my own LLM or am I happy with what opensource has to offer ?

### Local inference

**Ollama** is one of my favorite. It is a managed framework that runs easily on both windows and MAC.
It is an abstraction layer over llama.cpp, GGML and GGUF exposing an OpenAI-compatible interface.
You can customize your model with ollama-compatible modelfiles and run quantized model.
It comes with a lot of models available in a 1 line command. Ollama manages the orchestration while leaving the below layer to C++ llama.cpp library.

**Llama.cpp** offers more flexibility has it is a layer under Ollama. It has a UI (<https://docs.openwebui.com/getting-started/quick-start/starting-with-llama-cpp/>) interface to chat directly with the model.
It comes with a CLI and server integrated. It also contains a library to quantize models locally, transform weights into a compatible format like GGUF etc...
It is recommended if you need more customization while keeping a managed way to optimize the model loading and inference.

### Cloud inference

You can use the previous frameworks to serve your model on CLoud/Premise. However, the objectives are usually not the same. On cloud, you'll probably have more compute, would like to optimize your resources and would need a more resilient architecture as you won't switch every now and then.

#### Overview

**vLLM** is one of the best in class that is a great trade-off between simplicity of serving and production ready inference scalability.

RedHat ran a benchmark comparing vLLM to Ollama: <https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking#comparison_1__default_settings_showdown>
In summary, vLLM is a clear winner as it scales on throughputs and responsiveness with concurrent requests.

It also ran a benchmark comparing vLLM to llama.cpp and found out that vLLM is still outperforming both on throughtputs and responsiveness for high concurrent requests.
<https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case>

So what can be vLLM's weakness ?
As I experienced it, llama.cpp is more portable and has more flexiility than vLLM. You have also more contraints running vLLM depending on your infrastructure.
So the first items to check are:

- your target architecture (framework, GPUs, model...)
- your target model size
- your target model usage (# clients, # requests)
- your target model constraints (throughput, latency etc...)

You can see more requests and details in the LLMOps section.

**SGLang** in some benchmarks (<https://lmsys.org/blog/2024-07-25-sglang-llama3/>) outperforms vLLM in particular for smallLM.
It is a great alternative to test. Again, it will depends on your target deployments.

#### Feature comparison

| Feature                        | vLLM         | llama.cpp     | TGI           | Ollama        | sglang        | TensorRT      |
|-------------------------------|--------------|---------------|---------------|---------------|---------------|---------------|
| Language                      | Python/CUDA  | C/C++         | Rust/Python   | Go/C/TS       | Python/Rust/C++| C++/Python    |
| Model Support                 | HF, Llama, MoE, Multi-modal | GGUF, Llama, HF | HF, Llama, Falcon, StarCoder, BLOOM | Llama, Gemma, Mistral, Qwen, etc. | Llama, Qwen, DeepSeek, Gemma, Mistral, GLM, GPT, etc. | ONNX, HF, Llama, custom |
| Quantization                  | GPTQ, AWQ, INT4/8, FP8, TensorRT | 1.5-8bit, GGUF | bitsandbytes, GPTQ, AWQ, Marlin, fp8 | GGUF, Safetensors | FP4/FP8/INT4/AWQ/GPTQ, Multi-LoRA | INT8/FP16/FP8          |
| Hardware Support              | NVIDIA, AMD, Intel, ARM, TPU, TensorRT | CPU, GPU (NVIDIA, AMD, Apple, Vulkan, etc.) | NVIDIA, AMD, Intel, Gaudi, TPU | CPU, GPU, Docker | NVIDIA (GB200/B300/H100/A100), AMD (MI355/MI300), Intel Xeon, TPU, Ascend NPU | NVIDIA GPU, Triton      |
| API Compatibility             | OpenAI, HF, REST, Open WebUI | OpenAI, REST   | OpenAI, REST   | OpenAI, REST  | OpenAI, REST, HF, custom | Triton, REST, custom   |
| Streaming                     | Yes          | Yes           | Yes           | Yes           | Yes           | Yes           |
| Distributed/Parallel Serving  | Yes          | Partial       | Yes           | Partial       | Yes (tensor/pipeline/expert/data parallelism) | Yes (via Triton)        |
| Prefix/Attention Caching      | Yes (Paged)  | Yes           | Yes           | Yes           | Yes (RadixAttention, paged) | Yes           |
| Multi-LoRA                    | Yes          | Yes           | Yes           | Yes           | Yes           | No            |
| Observability/Tracing         | OpenTelemetry| Basic         | OpenTelemetry | 3rd party     | OpenTelemetry, metrics | OpenTelemetry |
| Web UI                        | Open WebUI, 3rd party | Basic         | Swagger       | Many          | SGLang Web UI, 3rd party | Triton UI      |
| Docker Support                | Yes          | Yes           | Yes           | Yes           | Yes           | Yes           |
| License                       | Apache-2.0   | MIT           | Apache-2.0    | MIT           | Apache-2.0     | Apache-2.0    |

#### Benchmarks

Note that this benchmark table is a very high level summary and should be used with caution depending on your target architecture/model/usage.

## 2. Performance Benchmarks (2024)

| Framework   | Model         | Hardware         | Throughput (tokens/s) | Latency (ms) | Notes                  |
|-------------|--------------|------------------|-----------------------|--------------|------------------------|
| vLLM        | Llama-2-7B    | A100 80GB        | ~10,000+              | <100         | Continuous batching, TensorRT support |
| llama.cpp   | Llama-2-7B    | M2 Pro (Apple)   | ~5,000                | <200         | Quantized, Metal       |
| TGI         | Llama-2-7B    | A100 80GB        | ~8,000                | <120         | Tensor parallelism     |
| Ollama      | Llama-2-7B    | M2 Pro (Apple)   | ~4,000                | <250         | GGUF, local            |
| sglang      | Llama-3-8B    | H100/GB200       | ~12,000+ (prefill), ~15,000+ (decode) | <80         | Large-scale expert parallelism |
| TensorRT    | Llama-2-7B    | A100/H100        | ~10,000+              | <100         | INT8/FP16/FP8 quantization |

## Sources

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [llama.cpp Docs](https://github.com/ggml-org/llama.cpp)
- [TGI Docs](https://huggingface.co/docs/text-generation-inference/)
- [Ollama Docs](https://ollama.com/)
- [HuggingFace Model Benchmarks](https://huggingface.co/docs/text-generation-inference/benchmarks)
- <https://www.linkedin.com/posts/arazvant_why-do-i-use-ollama-for-most-of-my-local-llm-activity-7389632890915135489-lo9u?utm_source=share&utm_medium=member_android&rcm=ACoAAAVV2dEBAuuJCv1jGmfAXdBgR9YAUI0StlM>
