# Resources

## Engineering

- [Weaviate Academy — multi-tenancy example](https://weaviate.io/developers/academy/py/multitenancy/overview)

## LLM Fundamentals and Architecture

- [Transformer explainer](https://poloclub.github.io/transformer-explainer/)
- [The attention mechanism (Benjamin Warner)](https://benjaminwarner.dev/2023/07/01/attention-mechanism) — transformers from scratch
- [Language model architectures (Pruna AI slides)](https://github.com/PrunaAI/ai-efficiency-courses/blob/main/slides/01-language_model_architectures.pdf)
- [Embeddings in machine learning](https://encord.com/blog/embeddings-machine-learning/?utm_source=linkedin)
- [Decoding / sampling strategies (top-k, etc.)](https://huggingface.co/blog/mlabonne/decoding-strategies)
- [Safetensors](https://huggingface.co/docs/safetensors/en/index)
- [A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223.pdf)

### Mixture of Experts

- [A visual guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
- [nanoMoE](https://cameronrwolfe.substack.com/p/nano-moe)
- [MoE architecture in transformer models](https://machinelearningmastery.com/mixture-of-experts-architecture-in-transformer-models/)
- [Gist (Phil Schmid)](https://gist.github.com/philschmid/d188034c759811a7183e7949e1fa0aa4)

## LLM Fine-Tuning

- [Unsloth notebooks](https://github.com/unslothai/notebooks/)
- [Data preparation (Unsloth datasets guide)](https://docs.unsloth.ai/basics/datasets-guide)
- [14 free LLM fine-tuning notebooks](https://levelup.gitconnected.com/14-free-large-language-models-fine-tuning-notebooks-532055717cb7)
- [Mastering LLM fine-tuning — top learning resources](https://pub.towardsai.net/mastering-large-language-model-llm-fine-tuning-top-learning-resources-dcef012256fd)
- [Best resources on building datasets to train LLMs](https://levelup.gitconnected.com/best-resources-on-building-datasets-to-trian-llms-f6c6e02fc375)
- [Awesome AI data guided projects](https://github.com/youssefHosni/Awesome-AI-Data-Guided-Projects?tab=readme-ov-file)
- [pyreft — representation fine-tuning (Stanford)](https://github.com/stanfordnlp/pyreft) — [README](https://github.com/stanfordnlp/pyreft?tab=readme-ov-file), [notebook (Colab)](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing#scrollTo=NL0yGhbe3EFk)
- [Lightning Studio fine-tuning pipelines (ZenML)](https://www.zenml.io/blog/automating-lightning-studio-ml-pipelines-for-fine-tuning-llm)
- [TRL Jobs (Hugging Face)](https://github.com/huggingface/trl-jobs)
- [Primer on PEFT](https://vinija.ai/nlp/parameter-efficient-fine-tuning/)
- [Efficient training on a single GPU (Transformers docs)](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [Fine-tuning post (LinkedIn)](https://www.linkedin.com/posts/eordax_ai-genai-finetuning-activity-7208179993086959616-wLbB/)
- [Continuous pre-training (Unsloth)](https://unsloth.ai/blog/contpretraining)
- [3x faster training with packing (Unsloth)](https://docs.unsloth.ai/new/3x-faster-training-packing)
- [Fine-tune multimodal LLMs with TRL (Phil Schmid)](https://www.philschmid.de/fine-tune-multimodal-llms-with-trl)
- [Fine-tune LLMs in 2024 with TRL (Phil Schmid)](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) — **Blocked**: dependencies
- [Fine-tune ModernBERT (Phil Schmid)](https://www.philschmid.de/fine-tune-modern-bert-in-2025) — routing via BERT fine-tuning
- [Ludwig](https://ludwig.ai/latest/)
- [ORPO with Unsloth (Colab)](https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing)
- [GRPO with Qwen 0.5B Instruct](https://github.com/aburkov/theLMbook/blob/main/GRPO_Qwen_0_5_Instruct.ipynb)
- [GRPO notebook 1 (Colab)](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing)

## Optimization and Efficiency

- [Understanding LLM optimization techniques (Neural Bits)](https://neuralbits.substack.com/p/understanding-llm-optimization-techniques)
- [Understanding LLM optimization techniques (Multimodal AI)](https://multimodalai.substack.com/p/understanding-llm-optimization-techniques)
- [ML engineering — beginner to advanced tips and tricks](https://github.com/stas00/ml-engineering?tab=readme-ov-file)
- [QLoRA and 4-bit quantization explained](https://mccormickml.com/2024/09/14/qlora-and-4bit-quantization/)
- [AI engineer optimization post (LinkedIn)](https://www.linkedin.com/posts/aishwarya-srinivasan_if-youre-an-ai-engineer-trying-to-optimize-activity-7334975771561680900-BXr-?utm_source=share&utm_medium=member_android&rcm=ACoAAAVV2dEBAuuJCv1jGmfAXdBgR9YAUI0StlM)
- [AI engineer optimization post (LinkedIn, share link)](https://www.linkedin.com/posts/aishwarya-srinivasan_if-youre-an-ai-engineer-trying-to-optimize-share-7334815669756047361-JCzs/)
- [Awesome AI efficiency (Pruna AI)](https://github.com/PrunaAI/awesome-ai-efficiency) — [README](https://github.com/PrunaAI/awesome-ai-efficiency?tab=readme-ov-file)
- [AI efficiency courses (Pruna AI)](https://github.com/PrunaAI/ai-efficiency-courses)
- [OptiPFair — pruning](https://github.com/peremartra/optipfair)
- [Knowledge distillation of Llama (notebook)](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/7_1_knowledge_distillation_Llama.ipynb)
- [Prune and distill Llama 3.1 8B to Minitron 4B (NVIDIA)](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model)
- [Distillation (video)](https://www.youtube.com/watch?v=JE7SuP049mQ)
- [Half a million evaluations on quantized LLMs (Neural Magic)](https://neuralmagic.com/blog/we-ran-over-half-a-million-evaluations-on-quantized-llms-heres-what-we-found/)
- [Quantization explained](https://mohamedarbinsibi.onrender.com/posts/quantization)
- [Deep Compression (pruning, quantization, Huffman coding)](https://arxiv.org/abs/1510.00149)

## Inference and Serving

- [vLLM project](https://github.com/vllm-project)
- [SGLang with Unsloth models](https://unsloth.ai/docs/basics/inference-and-deployment/sglang-guide) — **To test**
- [DeepSeek open infra index](https://github.com/deepseek-ai/open-infra-index/tree/main)
- [vLLM KV cache notebook (Rearchitecting LLMs, CH08)](https://github.com/peremartra/Rearchitecting-LLMs/blob/main/CH08/CH08_NB02_T4_KVCache_vLLM.ipynb) — re-add Rearchitecting-LLMs
- [Coding the KV cache in LLMs (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)
- [KV cache analyzer](https://github.com/bs258q/kv-cache-analyzer)
- [SIE — model gateway (Superlinked)](https://github.com/superlinked/sie)
- [LoRAX — multi-LoRA serving](https://github.com/predibase/lorax?tab=readme-ov-file)
- [BitNet](https://github.com/microsoft/BitNet)
- [Running custom GGUF models with Ollama](https://www.linkedin.com/pulse/running-custom-gguf-models-ollama-talles-carvalho-ckfif)
- [Decoding strategies (Maxime Labonne)](https://huggingface.co/blog/mlabonne/decoding-strategies)
- [AI engineer optimization post (LinkedIn)](https://www.linkedin.com/posts/aishwarya-srinivasan_if-youre-an-ai-engineer-trying-to-optimize-activity-7334975771561680900-BXr-) — to finish, with [this video](https://www.youtube.com/watch?v=J3vUMwG8dks)
- [Triton Inference Server tutorials](https://github.com/triton-inference-server/tutorials/tree/main)
- [Scaling test-time compute (HF blog post)](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute)

## Embeddings

- [EmbeddingGemma](https://huggingface.co/blog/embeddinggemma)
- [Training sentence transformers](https://huggingface.co/blog/train-sentence-transformers)
- [Fine-tune an embedding model for RAG (Phil Schmid)](https://www.philschmid.de/fine-tune-embedding-model-for-rag)
- [LEAF distillation — text embedding models (MongoDB)](https://www.mongodb.com/company/blog/engineering/leaf-distillation-state-of-the-art-text-embedding-models)
- [Embedding quantization](https://huggingface.co/blog/embedding-quantization)
- [Fine-tune and deploy embedding models on SageMaker](https://www.philschmid.de/sagemaker-train-deploy-embedding-models)

## RAG and Retrieval

- [RAG zero to hero guide](https://github.com/KalyanKS-NLP/rag-zero-to-hero-guide?tab=readme-ov-file)
- [Contextual retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval)
- [RAGatouille book](https://www.sakunaharinda.xyz/ragatouille-book/intro.html)
- [Cross-encoders as rerankers](https://weaviate.io/blog/cross-encoders-as-reranker)
- [Local RAG app with NVIDIA NIM and Milvus](https://www.linkedin.com/pulse/build-rag-app-nvidia-nim-milvus-running-locally-janakiram-msv-lch4c/?utm_source=share&utm_medium=member_android&utm_campaign=share_via)
- [Advanced RAG (Weaviate)](https://weaviate.io/blog/advanced-rag) — nice overview
- [RAG Techniques (Nir Diamant)](https://github.com/NirDiamant/RAG_Techniques)
- [Controllable RAG agent](https://github.com/NirDiamant/Controllable-RAG-Agent)
- [RAG from scratch (YouTube playlist)](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)
- [RAPTOR](https://github.com/parthsarthi03/RAPTOR)
- [LightRAG](https://github.com/HKUDS/LightRAG)
- [GraphRAG (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [GraphRAG v1 cookbook (LlamaIndex)](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/GraphRAG_v1.ipynb)
- [Graphiti](https://github.com/getzep/graphiti)
- [Stop chunking your data (Weaviate, LinkedIn)](https://www.linkedin.com/posts/weaviate-io_stop-chunking-your-data-youre-probably-activity-7399115168061816832-phi6)
- [Pyversity — result diversification](https://github.com/Pringled/pyversity)
- [High-precision RAG for table-heavy documents](https://medium.com/kx-systems/high-precision-rag-for-table-heavy-documents-using-langchain-unstructured-io-kdb-ai-22f7830eac9a)
- [Multi-document agentic RAG (Lightning AI)](https://lightning.ai/blochsphere/studios/multi-document-agentic-rag-for-quantum-computing)
- [Multimodal RAG on slide decks (LlamaParse)](https://github.com/run-llama/llama_parse/blob/main/examples/multimodal/multimodal_rag_slide_deck.ipynb) — **must read**
- [Persona-based access GenAI application (AWS workshop)](https://catalog.workshops.aws/persona-based-access-genai-application/en-US/01-introduction)
- [LEANN — RAG over large document collections](https://github.com/yichuan-w/LEANN)
- [BM25S](https://huggingface.co/blog/xhluca/bm25s) — test questions on a DB

## LLMOps, MLOps and Production

- [MLOps principles](https://ml-ops.org/content/mlops-principles)
- [LLM Twin course](https://github.com/decodingml/llm-twin-course?tab=readme-ov-file) — end-to-end, production-like, advanced optimization engineering
- [ZenML LLMOps guide](https://docs.zenml.io/user-guides/llmops-guide)
- [Productionize LLMs](https://mrmaheshrajput.medium.com/how-to-productionize-large-language-models-llms-060a4cb1a169)
- [Energy forecasting MLOps course](https://github.com/iusztinpaul/energy-forecasting)
- [Marvelous MLOps (marvel characters)](https://github.com/marvelousmlops/marvel-characters)
- [Dagger Python SDK](https://archive.docs.dagger.io/0.9/sdk/python/628797/get-started/)
- [MLflow Recipes](https://mlflow.org/docs/latest/recipes.html)
- [LLMOps vs MLOps (Iguazio)](https://www.iguazio.com/blog/llmops-vs-mlops-understanding-the-differences/)
- [Operationalizing generative AI — how it differs from MLOps (AWS)](https://aws.amazon.com/fr/blogs/machine-learning/operationalizing-generative-ai-how-it-differs-from-mlops/)
- [Why we need MLOps (LinkedIn)](https://www.linkedin.com/posts/eordax_genai-mlops-llmops-ugcPost-7172157303452823553-AVCf)
- [LLMOps glossary (Klu)](https://klu.ai/glossary/llm-ops)
- [LLMOps database — use cases (ZenML)](https://www.zenml.io/llmops-database)
- [LLMOps Python package](https://github.com/callmesora/llmops-python-package/tree/main)
- [LLMOps (Blaizzy)](https://github.com/Blaizzy/LLMOps/tree/main) — to follow
- [LLMOps maturity model (Azure)](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/concept-llmops-maturity?view=azureml-api-2)

## LLM Evaluation

- [Hugging Face evaluation guidebook](https://github.com/huggingface/evaluation-guidebook?tab=readme-ov-file)
- [RAG evaluation (HF cookbook)](https://huggingface.co/learn/cookbook/rag_evaluation)
- [Retrieval evaluation — the art of RAG, part 4](https://medium.com/@rossashman/the-art-of-rag-part-4-retrieval-evaluation-427bb5db0475)
- [RAGET — banking supervision example](https://docs.giskard.ai/en/latest/reference/notebooks/RAGET_Banking_Supervision.html)
- [Visualize RAG (Renumics)](https://github.com/Renumics/renumics-rag/blob/main/notebooks/visualize_rag_tutorial.ipynb)
- [GLIDE (Emerton Data)](https://github.com/EmertonData/glide) — [license](https://github.com/EmertonData/glide?tab=License-1-ov-file)
- [Numeric evals for LLM-as-a-judge (Arize)](https://arize.com/blog-course/numeric-evals-for-llm-as-a-judge/)
- [Preference leakage in LLM-as-a-judge](https://github.com/David-Li0406/Preference-Leakage)
- [OpenEvals (LangChain)](https://github.com/langchain-ai/openevals)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [Evidently — LLM evaluation tutorial](https://docs.evidentlyai.com/get-started/tutorial-llm)
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/index)
- [Evalchemy — distributed evaluation](https://github.com/mlfoundations/evalchemy)
- [MixEval](https://mixeval.github.io)
- [Evaluating LLMs with Giskard and MLflow (Databricks)](https://www.databricks.com/blog/evaluating-large-language-models-giskard-mlflow)
- [Track LLM evaluation with SageMaker managed MLflow and FMEval](https://aws.amazon.com/fr/blogs/machine-learning/track-llm-model-evaluation-using-amazon-sagemaker-managed-mlflow-and-fmeval/)
- [Evaluate abstractive summarization (OpenAI cookbook)](https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/How_to_eval_abstractive_summarization.ipynb)
- [Exploring LLM leaderboards](https://medium.com/@olga.zem/exploring-llm-leaderboards-8527eac97431)
- [Sycophantic AI](https://huggingface.co/blog/Rakshit122/sycophantic-ai)

## OCR and Document Extraction

- [Mistral cookbook](https://github.com/mistralai/cookbook)
- [Instabase — unstructured document understanding](https://instabase.com)
- [Docling — parsing](https://github.com/DS4SD/docling) — further test in docs/examples

## Utils

- [The Incredible PyTorch](https://github.com/ritchieng/the-incredible-pytorch)

## Engineering Practices and Git

- [Microsoft engineering playbook](https://microsoft.github.io/code-with-engineering-playbook/) — [YouTube playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [The guide to Git I never had](https://dev.to/glasskube/the-guide-to-git-i-never-had-1450)
- [Git explained (LinkedIn)](https://www.linkedin.com/posts/rajatgajbhiye_i-will-never-understand-git-it-is-so-confusing-activity-7231890613313396736-XJZA/?utm_source=share&utm_medium=member_android)
- [GitOps for ML workshop (Iterative)](https://github.com/iterative/workshop-e2e-gitops-ml)
- [Scale GitOps in the enterprise with an IDP (Humanitec)](https://humanitec.com/blog/scale-gitops-in-the-enterprise-with-an-idp)

## Python

- [Comparison of PDM, Poetry and Rye](https://www.playfulpython.com/environment-tools-pdm-poetry-rye/)
- [Rye](https://github.com/astral-sh/rye)
- [PDM](https://github.com/pdm-project/pdm)
- [Ruff vs mypy / pyright / pyre](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-mypy-or-pyright-or-pyre) — type checkers
- [Clean code in Python](https://testdriven.io/blog/clean-code-python/)
- [SOLID MLOps code with Pydantic and ABC](https://mlops.community/make-your-mlops-code-base-solid-with-pydantic-and-pythons-abc/)
- [Multiprocessing vs asyncio vs threading (Stack Overflow)](https://stackoverflow.com/questions/61351844/difference-between-multiprocessing-asyncio-threading-and-concurrency-futures-i)
- [Weaviate async Python client](https://weaviate.notion.site/Async-Python-Client-Usage-21c70562ccd68020b569c668be40d584)
- [GitDiagram](https://github.com/ahmedkhaleel2004/gitdiagram)
- [C4 model diagrams](https://c4model.com/diagrams#diagrams)
- [Diagrams as code](https://diagrams.mingrammer.com/docs/getting-started/examples)
- [Markmap](https://markmap.js.org/repl)
- [Manim — math animation videos](https://github.com/manimCommunity/manim)
- [The guide to Git I never had](https://dev.to/glasskube/the-guide-to-git-i-never-had-1450)
- [Git explained (LinkedIn)](https://www.linkedin.com/posts/rajatgajbhiye_i-will-never-understand-git-it-is-so-confusing-activity-7231890613313396736-XJZA/)
- [uithub API](https://uithub.com/openapi.html#/operations/getRepositoryContents)

## ML Evaluation

- [The Little Book of ML Metrics](https://github.com/NannyML/The-Little-Book-of-ML-Metrics)
- [Google ML glossary](https://developers.google.com/machine-learning/glossary)

## Responsible AI (ML/LLM)

- [Evidently AI](https://www.evidentlyai.com)
- [NannyML — drift](https://www.nannyml.com/library)
- [LangFair (CVS Health)](https://github.com/cvs-health/langfair)
- [Scikit-lego fairness](https://scikit-lego.netlify.app/fairness.html)
- [Phare LLM benchmark (Giskard)](https://phare.giskard.ai/)
- [Foundation Model Transparency Index (Stanford HAI)](https://hai.stanford.edu/news/introducing-foundation-model-transparency-index) — [May 2024 edition](https://crfm.stanford.edu/fmti/May-2024/index.html)
- [Responsible NLP (Meta)](https://github.com/facebookresearch/ResponsibleNLP)
- [Awesome production ML — outlier and anomaly detection](https://github.com/EthicalML/awesome-production-machine-learning?tab=readme-ov-file#outlier-and-anomaly-detection)
- [Petri — open-source AI safety auditing (Anthropic)](https://www.anthropic.com/research/petri-open-source-auditing)
- [Introduction to explainable AI with Shapley values (SHAP)](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)
- [InterpretML](https://github.com/interpretml/interpret) — SHAP-like
- [InterpretML — MSA](https://interpret.ml/docs/msa.html)
- [Captum — PyTorch model interpretability](https://captum.ai)
- [Shapash (MAIF)](https://github.com/MAIF/shapash)
- [Responsible AI Toolbox (Microsoft)](https://github.com/microsoft/responsible-ai-toolbox) — to test
- [AI Explainability 360 (IBM)](https://aix360.res.ibm.com)
- [Explainer Dashboard](https://explainerdashboard.readthedocs.io/en/latest/explainers.html)
- [TrustyAI explainability examples](https://github.com/trustyai-explainability/trustyai-explainability-python-examples?tab=readme-ov-file)
- [PyTorch Concepts — interpretable models](https://pytorch-concepts.readthedocs.io/en/latest/index.html)
- [EconML — causal inference](https://github.com/py-why/EconML)
- [Scaling monosemanticity (Anthropic)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [Circuit tracer tutorial](https://github.com/safety-research/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb)
- [LLM circuit graph (Neuronpedia)](https://www.neuronpedia.org/gemma-2-2b/graph?slug=cristuftest&pruningThreshold=0.6&densityThreshold=0.99)

## Great Visualizations

- [LLM visualization](https://bbycroft.net/llm)

## GPUs

- [Modal GPU glossary](https://modal.com/gpu-glossary/readme)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started) — train on GPU/CPU
- [VRAM calculator](https://apxml.com/tools/vram-calculator)
- [Cloud instance comparison (Vantage)](https://instances.vantage.sh/?id=ae5bdee9e41785d5d7f919b242e670aa7e5a2ea0)
- [Memory-efficient weight loading (LLMs from scratch)](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
- [mlxtop](https://github.com/maximpri/mlxtop)
- [Coiled](https://cloud.coiled.io/get-started)
- [Ori GPU cloud — free credits](https://www.ori.co/alphasignal-1000)
- [OpenLLM (BentoML)](https://github.com/bentoml/OpenLLM)
- [Cloud comparison](https://comparecloud.in)

## Reinforcement Learning

- [Prime Agent (Prime Intellect)](https://github.com/PrimeIntellect-ai/prime-agent) — **To test**: RL improvement
- [GRPO with Qwen 0.5B Instruct (theLMbook)](https://github.com/aburkov/theLMbook/blob/main/GRPO_Qwen_0_5_Instruct.ipynb)
- [GRPO notebook 1 (Colab)](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing)
- [GRPO notebook 2 (Colab)](https://colab.research.google.com/drive/1RmLv-pCMBBsQGXQIM8yF-OdCNyoylUR1?usp=sharing)
- [GRPO notebook 3 (Colab)](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing)
- [R1 reasoning (Unsloth)](https://unsloth.ai/blog/r1-reasoning)
- [The state of LLM reasoning model training (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [Tool-calling router fine-tuning with GRPO](https://github.com/debashis1983/llama-tool-router-grpo)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/user/running.html)
- [RewardBench](https://allenai.org/blog/rewardbench-the-first-benchmark-leaderboard-for-reward-models-used-in-rlhf-1d4d7d04a90b)
- [RL notebook (Colab)](https://colab.research.google.com/drive/1ZcQeq7irohIMP1bBh_n3H9Ak3Dm5Yu1y?usp=sharing#scrollTo=9BICzsovb6-H)

## Agents

### Frameworks and Platforms

- [Hermes Agent (Nous Research)](https://github.com/NousResearch/hermes-agent) — [docs](https://hermes-agent.nousresearch.com)
- [Agent Lightning (Microsoft)](https://github.com/microsoft/agent-lightning?tab=readme-ov-file)
- [Atomic Agents](https://github.com/BrainBlend-AI/atomic-agents)
- [AutoAgent](https://github.com/HKUDS/AutoAgent?tab=readme-ov-file#-quick-start)
- [OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [Strands Agents — prompt engineering for safety](https://strandsagents.com/0.1.x/user-guide/safety-security/prompt-engineering/)
- [Mixture of Agents (Together)](https://github.com/togethercomputer/moa)
- [Google A2A](https://github.com/google/A2A)
- [AG-UI — agent-user interaction protocol](https://docs.ag-ui.com/introduction) — [notebook](https://colab.research.google.com/drive/1ZcQeq7irohIMP1bBh_n3H9Ak3Dm5Yu1y?usp=sharing)
- [AWS Labs agent repositories](https://github.com/orgs/awslabs/repositories?q=agent)
- [Amazon Bedrock agent samples](https://github.com/awslabs/amazon-bedrock-agent-samples)
- [Multi-agent business expert with Amazon Bedrock](https://aws.amazon.com/fr/blogs/machine-learning/build-an-intelligent-multi-agent-business-expert-using-amazon-bedrock/)
- [LangChain Python REPL tool](https://python.langchain.com/docs/integrations/tools/python/)

### Agent Researcher

- [Research Rabbit](https://github.com/langchain-ai/research-rabbit)

### Agent Memory

- [LangGraph persistence — memory store](https://docs.langchain.com/oss/python/langgraph/persistence#memory-store) — test in finance
- [Memento — agents learning from experience](https://github.com/Agent-on-the-Fly/Memento)
- [LLM agents learn from experience without fine-tuning (VentureBeat)](https://venturebeat.com/ai/this-new-framework-lets-llm-agents-learn-from-experience-no-fine-tuning)
- [A-MEM — agentic memory](https://github.com/agiresearch/A-mem)
- [OpenMemory](https://github.com/CaviraOSS/OpenMemory/tree/main) — promising memory SDK
- [Cognee with Ollama](https://docs.cognee.ai/tutorials/setup-ollama)

### Agent Training and Self-Improvement

- [ART — agent reinforcement trainer (OpenPipe)](https://github.com/OpenPipe/ART?tab=readme-ov-file)
- [Training agents (live video)](https://www.youtube.com/live/cixmqTsi2A4)
- [ToolOrchestra (NVIDIA)](https://github.com/NVlabs/ToolOrchestra/tree/main)
- [SkillOpt (Microsoft)](https://github.com/microsoft/SkillOpt/tree/main)
- [STORM (Stanford OVAL)](https://github.com/stanford-oval/storm)
- [Santander AI](https://github.com/SantanderAI)

### AgentOps

- [AgentOps open course](https://github.com/MLOps-Courses/agentops-open-course)
- [AgentOps](https://github.com/AgentOps-AI/agentops) — see also [this video](https://www.youtube.com/watch?v=Pz9YeBs_afo)

## MCP

- [Model Context Protocol announcement (Anthropic)](https://www.anthropic.com/news/model-context-protocol)
- [Code execution with MCP (Anthropic)](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [MCP registry](https://github.com/modelcontextprotocol/registry/tree/main)
- [LinkedIn MCP server](https://github.com/stickerdaniel/linkedin-mcp-server?tab=readme-ov-file) — **important**
- [FastAPI MCP](https://github.com/tadata-org/fastapi_mcp)
- [Pydantic AI MCP server](https://ai.pydantic.dev/mcp/server/)
- [DSPy with MCP](https://dspy.ai/tutorials/mcp/)
- [Experiments with MCP](https://github.com/Vaibhavs10/experiments-with-mcp)
- [MCP from gitingest (X post)](https://x.com/akshay_pachaar/status/1918283739760828795)
- [MCP crash course — simple server setup](https://github.com/daveebbelaar/ai-cookbook/tree/main/mcp/crash-course/3-simple-server-setup)
- [Run MCP servers with AWS Lambda](https://github.com/awslabs/run-model-context-protocol-servers-with-aws-lambda)
- [Azure OpenAI MCP sample](https://github.com/monuminu/AOAI_Samples/tree/main/mcp_aoai) — **Blocked**: test once Azure access is available

## AI Security

- [LiteLLM supply chain attack (Giskard)](https://www.giskard.ai/knowledge/litellm-supply-chain-attack-2026) — [notebook](https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw?usp=sharing#scrollTo=s-_vu8HuGEb-)
- [Pickle security (Hugging Face)](https://huggingface.co/docs/hub/security-pickle)
- [Safetensors](https://huggingface.co/docs/safetensors/en/index)
- [picklescan](https://github.com/mmaitre314/picklescan) / [ModelScan](https://github.com/protectai/modelscan)
- [MLflow JFrog Artifactory plugin](https://pypi.org/project/mlflow-jfrog-plugin/#history)
- [AI security posture management (Palo Alto)](https://www.paloaltonetworks.com/cyberpedia/ai-security-posture-management-aispm)
- [OWASP — LLM and GenAI security solutions landscape](https://genai.owasp.org/resource/llm-and-generative-ai-security-solutions-landscape/)
- [garak — LLM vulnerability scanner (NVIDIA)](https://github.com/NVIDIA/garak)
- [DeepTeam — red teaming](https://github.com/confident-ai/deepteam)
- [Prompt injection challenge (Giskard)](https://red.giskard.ai/register)
- [Securing the Model Context Protocol (LinkedIn)](https://www.linkedin.com/pulse/securing-model-context-protocol-mcp-challenges-best-muayad-sayed-ali-sot4e/)
- [The S in MCP stands for security](https://elenacross7.medium.com/️-the-s-in-mcp-stands-for-security-91407b33ed6b)
- [Small samples can poison LLMs (Anthropic)](https://www.anthropic.com/research/small-samples-poison)

## Guardrails

- [How to use guardrails (OpenAI cookbook)](https://cookbook.openai.com/examples/how_to_use_guardrails)
- [Awesome AI guardrails](https://github.com/enguard-ai/awesome-ai-guardails/tree/main)
- [LLM Guard with Bedrock](https://github.com/protectai/llm-guard/tree/main/examples)
- [Fiddler AI guardrails](https://www.fiddler.ai/webinars/product-updates-ai-guardrails-integrations)
- [Bedrock Guardrails image content filters](https://aws.amazon.com/fr/blogs/machine-learning/amazon-bedrock-guardrails-image-content-filters-provide-industry-leading-safeguards-helping-customer-block-up-to-88-of-harmful-multimodal-content-generally-available-today/) — **Blocked**: test once Bedrock access is available

## Kubernetes and Cloud

- [Argo CD](https://argo-cd.readthedocs.io/en/stable/)
- [8 alternatives to Kubeflow (ZenML)](https://www.zenml.io/blog/8-alternatives-to-kubeflow-for-ml-workflow-orchestration-and-why-you-might-switch)
- [Deploying models to Kubernetes with MLflow and KServe (MinIO)](https://blog.min.io/deploying-models-to-kubernetes-with-aistor-mlflow-and-kserve/)
- [A guide to Kubernetes, part 2 (SwirlAI)](https://www.newsletter.swirlai.com/p/a-guide-to-kubernetes-part-2-different)
- [Azure Service Fabric cluster templates](https://github.com/Azure-Samples/service-fabric-cluster-templates)
- [Azure code samples](https://learn.microsoft.com/en-us/samples/browse/)

## Tabular Foundation Models

- [TabFM — zero-shot foundation model for tabular data (Google Research)](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/) — [repo](https://github.com/google-research/tabfm)
- [TabPFN](https://github.com/PriorLabs/tabpfn?tab=readme-ov-file) — [Nature paper](https://www.nature.com/articles/s41586-024-08328-6)
- [TabICL (Soda Inria)](https://github.com/soda-inria/tabicl)
- [Seldon technical report (Neuralk)](https://technical-report.neuralk.ai/seldon-technical-report.pdf)

## Time Series Foundation Models

- [TimesFM (Google Research)](https://github.com/google-research/timesfm)
- [Toto (Datadog)](https://github.com/DataDog/toto)

## Finance and Trading

- [Building AI Agents for Finance — chapter 5 (Packt)](https://github.com/PacktPublishing/Building-AI-Agents-for-Finance/tree/main/Chapter%205) — **To test, absolutely**
- [AI4Finance Foundation](https://github.com/ai4finance-foundation)
- [Qwen Open Finance R 8B (DragonLLM)](https://huggingface.co/DragonLLM/Qwen-Open-Finance-R-8B)
- [Machine learning for trading (Stefan Jansen)](https://github.com/stefan-jansen/machine-learning-for-trading)
- [Momentum strategy with residual alpha and Yang-Zhang volatility](https://medium.com/@Ansique/building-a-sophisticated-momentum-strategy-residual-alpha-yang-zhang-volatility-a3884247c2ed) — test finance strategy
- [Decoding alpha — volatility and causal inference](https://antonio-velazquez-bustamante.medium.com/decoding-alpha-how-volatility-and-causal-inference-unlock-predictive-trading-455754bcb7b9)

## NLP, NER and PII

- [GLiNER fine-tuning (CFM case study)](https://huggingface.co/blog/cfm-case-study)
- [GLiNER2](https://github.com/fastino-ai/GLiNER2)
- [Nemotron PII detection (NVIDIA)](https://huggingface.co/blog/nvidia/nemotron-pii)
- [PII and de-identification collection (OpenMed)](https://huggingface.co/collections/OpenMed/pii-and-de-identification)
- [Document classifier with LLMs as labelers (theLMbook)](https://github.com/aburkov/theLMbook/blob/main/document_classifier_with_LLMs_as_labelers.ipynb)

## Speech and Vision

### Speech

- [Kyutai STT](https://kyutai.org/next/stt)
- [Moshi (Kyutai)](https://github.com/kyutai-labs/moshi) — run on local machine
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Chatterbox (Resemble AI)](https://github.com/resemble-ai/chatterbox)
- [VibeVoice (Microsoft)](https://microsoft.github.io/VibeVoice/)
- [Qwen3-TTS open-sourced (LinkedIn)](https://www.linkedin.com/posts/qwen3-tts-is-officially-live-weve-open-sourced-share-7420091707704193024-uc3l) — **To review**: Pocket TTS vs Qwen TTS; test TTS deployment

### Vision and VLMs

- [An introduction to VLMs](https://towardsdatascience.com/an-introduction-to-vlms-the-future-of-computer-vision-models-5f5aeaafb282)
- [VoRA](https://github.com/Hon-Wong/VoRA)
- [smol-vision](https://github.com/merveenoyan/smol-vision/tree/main)
- [YOLOv10](https://github.com/THU-MIG/yolov10)
- [Fine-tune Idefics2 for multi-page PDF QA](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Idefics2/Fine_tune_Idefics2_for_multi_page_PDF_question_answering_on_DUDE.ipynb)
- [Transformers tutorials (Niels Rogge)](https://github.com/NielsRogge/Transformers-Tutorials/tree/master)
- [Vision notebook (Colab)](https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing)
- Roboflow football detection: `pip install git+https://github.com/roboflow/sports.git`

## World Models

- [LE-WM](https://github.com/lucas-maes/le-wm)
- [World models — Utopia](https://github.com/deeplethe/utopia)

## Sovereign AI

- [Sovereign AI — Clever Cloud × SpiNNcloud](https://www.clever.cloud/fr/blog/presse/2025/11/25/spinncloud-et-clever-cloud-unissent-leurs-forces-pour-une-plateforme-europeenne-de-calcul-pour-lia/)

## AWS and SageMaker

- [Secure MLOps with SageMaker](https://github.com/aws-samples/amazon-sagemaker-secure-mlops?tab=readme-ov-file) — best repo; Service Catalog and SageMaker domain setup
- [SageMaker Pipelines examples](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-pipelines)
- [SageMaker Studio examples](https://github.com/aws/amazon-sagemaker-examples/tree/main/aws_sagemaker_studio)
- [SageMaker notebook lifecycle configuration samples](https://github.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples)
- [SageMaker pipelines with GitHub Actions](https://github.com/aws-samples/mlops-sagemaker-github-actions)
- [SageMaker custom project templates](https://github.com/aws-samples/sagemaker-custom-project-templates/tree/main)
- [MLOps with SageMaker (includes Jenkins examples)](https://github.com/aws-samples/mlops-amazon-sagemaker)
- [MLOps Workload Orchestrator](https://docs.aws.amazon.com/solutions/latest/mlops-workload-orchestrator/using-the-aws-management-console.html)
- [SageMaker with MLflow](https://aws.amazon.com/fr/blogs/aws/manage-ml-and-generative-ai-experiments-using-amazon-sagemaker-with-mlflow/)
- [SageMaker Core Python SDK](https://aws.amazon.com/blogs/machine-learning/introducing-sagemaker-core-a-new-object-oriented-python-sdk-for-amazon-sagemaker/)
- [Multi-tenant LoRA serving with SageMaker](https://aws.amazon.com/blogs/machine-learning/efficient-and-cost-effective-multi-tenant-lora-serving-with-amazon-sagemaker/)
- [AWS SAM workshop — CI/CD unit tests](https://catalog.workshops.aws/complete-aws-sam/en-US/module-4-cicd/module-4-cicd-codepipeline/70-unittests)
- [CodePipeline tutorial](https://docs.aws.amazon.com/codepipeline/latest/userguide/tutorials-simple-codecommit.html)
- [Amazon Bedrock samples](https://github.com/aws-samples/amazon-bedrock-samples/tree/main)
- [Import custom models in Amazon Bedrock](https://aws.amazon.com/blogs/aws/import-custom-models-in-amazon-bedrock-preview/)
- [Multimodal RAG on PDFs with Bedrock](https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag-solutions/multimodal-rag-pdf/rag/multimodal-rag-pdf.ipynb) — **Blocked**: needs Bedrock access
