
## What is a CPU ?

Central Processing Units (CPUs) are general-purpose processors used in all computers and servers. CPUs are widely available and suitable for running small models or serving infrequent requests. However, they lack the parallel processing power to run LLMs efficiently. For production-grade LLM inference, especially with larger models or high request volumes, CPUs often fall short in both latency and throughput.

## What is a GPU ?

Graphics Processing Units (GPUs) were originally designed for graphics rendering and digital visualization tasks. As they could perform highly parallel operations, they also turned out to be a great fit for ML and AI workloads. Today, GPUs are the default choice for both training and inference of GenAI like LLMs.

The architecture of GPUs is optimized for matrix multiplication and tensor operations, which are core components of transformer-based models. Modern inference frameworks and runtimes (e.g., vLLM, SGLang, LMDeploy, TensorRT-LLM, and Hugging Face TGI) are designed to take full advantage of GPU acceleration.

You can find here a fascinating (rather dense) free book on GPU and how to make them work collectively and at scale (as this is the main goal): <https://jax-ml.github.io/scaling-book/gpus/#what-is-a-gpu>

## What is a TPU ?

Tensor Processing Units (TPUs) are custom-built by Google to accelerate AI workloads like training and inference. Compared with GPUs, TPUs are designed from the ground up for tensor operations — the fundamental math behind neural networks. This specialization makes TPUs faster and more efficient than GPUs for many AI-based compute tasks, like LLM inference.

TPUs are behind some of the most advanced AI applications today: agents, recommendation systems and personalization, image, video & audio synthesis, and more. Google uses TPUs in Search, Photos, Maps, and to power Gemini and DeepMind models.

## So what ?

First, make sure that your use case requires beyond CPU usage as this is a whole new world.
Then, while TPU have been created specifically for matrices operations like ML/LLM models, there is still, to date, a lot of infrastructure running on GPUs.

Not all models architecture support TPU infrastructure which shows GPUs are more general purpose compute oriented.
TPU (if model allows) is normally more efficient at inference time for large scale recommendation engines or training large LLMs.
