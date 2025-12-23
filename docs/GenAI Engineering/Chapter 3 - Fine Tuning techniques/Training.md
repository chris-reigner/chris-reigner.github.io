# Training steps (WIP)

## What are pre-trained or foundation models ?

## What is pre-training ?

Data preparation: Pre-training requires massive datasets (e.g., Llama 3.1 was trained on 15 trillion tokens) that need careful curation, cleaning, deduplication, and tokenization. Modern pre-training pipelines implement sophisticated filtering to remove low-quality or problematic content.
Distributed training: Combine different parallelization strategies: data parallel (batch distribution), pipeline parallel (layer distribution), and tensor parallel (operation splitting). These strategies require optimized network communication and memory management across GPU clusters.
Training optimization: Use adaptive learning rates with warm-up, gradient clipping and normalization to prevent explosions, mixed-precision training for memory efficiency, and modern optimizers (AdamW, Lion) with tuned hyperparameters.
Monitoring: Track key metrics (loss, gradients, GPU stats) using dashboards, implement targeted logging for distributed training issues, and set up performance profiling to identify bottlenecks in computation and communication across devices.

## What is continued pre-training ?

Continued Pre-Training refers to the cost effective alternative to pre-training. In this process, we further train a base pre-trained LLM on a large corpus of domain-specific text documents. This augments the model’s general knowledge with specific information from the particular domain.
Like pre-training it is performed in a self supervised manner (i.e. no labels) using for example some texts for a particular subject.

There are huge benefits because you can have totally new dataset but it is at risk of catastrophic forgetting (the fact that previous knowledge would be forgotten).

## What is post-training ?

### Post training datasets

Storage & chat templates: Because of the conversational structure, post-training datasets are stored in a specific format like ShareGPT or OpenAI/HF. Then, these formats are mapped to a chat template like ChatML or Alpaca to produce the final samples the model is trained on.
Synthetic data generation: Create instruction-response pairs based on seed data using frontier models like GPT-4o. This approach allows for flexible and scalable dataset creation with high-quality answers. Key considerations include designing diverse seed tasks and effective system prompts.
Data enhancement: Enhance existing samples using techniques like verified outputs (using unit tests or solvers), multiple answers with rejection sampling, Auto-Evol, Chain-of-Thought, Branch-Solve-Merge, personas, etc.
Quality filtering: Traditional techniques involve rule-based filtering, removing duplicates or near-duplicates (with MinHash or embeddings), and n-gram decontamination. Reward models and judge LLMs complement this step with fine-grained and customizable quality control.

<https://github.com/mlabonne/llm-datasets>
<https://github.com/NVIDIA/NeMo-Curator>
<https://distilabel.argilla.io/dev/sections/pipeline_samples/>



## Model Merging

Model merging
It is another method than simultaneous and sequential multitask fine-tuning. It fine tunes the model on different tasks separately but in parallel and then merge the different models together
It is also one way to do federated learning
Different than ensemble method as it mixes some parameters for constituents models together.
A few methods (experimentals):

- summing: linear combination of model weights usually. One can also create «task vectors» (from same model base, extract vectors for a specific task giving idiosyncrasies). If not the same base model, projections can be used to sum 2 different combinations
- Layer stacking: stack layers from different models and requires final fine-tuning once merged. It can be used for MixtureOfExperts; It can also be used to run depth wise upscaling (increase the size of the model)
- Concatenation:

<https://magazine.sebastianraschka.com/p/research-papers-in-january-2024?open=false#%C2%A7understanding-model-merging-and-weight-averaging>


## Resources

If you want real-life experience training an LLM I strongly recommend [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook).
It is a guide to the methodology used to train real life LLMs.