# Async Python Client Usage

### Async use cases

**E-commerce search platform:**

- Consider an online store where hundreds of customers are simultaneously searching for products. Each search query must look through tens of millions of products and return results quickly to maintain a good user experience. This must hold true, even during busy periods like sales events.

**Personal productivity app:**

- Imagine a note-taking application where users can save documents, to-do lists, and research materials. Users frequently perform multiple operations simultaneously - saving a new document while searching through their existing notes in multiple tabs. The app should be able to handle all of these operations happening at the same time.

**Customer support chatbot:**

- Consider a RAG-powered support system. When a customer asks a question, the system must: (1) search the knowledge base for information, (2) gather context, and (3) send the retrieved context to an LLM for response generation. The complexity and length of these queries make it even more important to efficiently coordinate these multi-step workflows, to ensure fast response times even under high query volumes.

In all these use cases, the common thread is **I/O-bound operations**, from distributed sources such as from multiple users. The async client shines when you need to coordinate multiple network calls (to main API, embedding models, or LLMs) efficiently. This is a basic explanation of how concurrency helps to speed up operations.

![Async overview](../../img/mermaid-diagram-2025-07-03-163630.png)

But, even when compared to parallelized sync client calls (e.g. through threads of multiple processes), the async client is more efficient, as you will see.

So let’s dive in to the world of async Python client.

### Concurrency in Python

We assume familiarity with concurrency. If this is new to you, we recommend skimming through [this Python Documentation section](https://docs.python.org/3/library/concurrency.html), or online tutorials such as [this](https://realpython.com/python-concurrency/) or [this](https://www.geeksforgeeks.org/python/python-program-with-concurrency/).

You can perform concurrent operations in Python using multi-processing (`multiprocessing`), multi-threading (`multithreading`) and coroutines (`asyncio`).

![Concurrency](../../img/Concurrency.png)

Multi-processing is the most isolated and resource-heavy, as each “process” can be thought as separate programs. At the other end of the scale are coroutines, where multiple tasks are coordinated in one thread to minimize unproductive time in any one operation.

Concurrency in Weaviate are typically required where end users are involved; for data insertions, for search queries or for RAG queries. They come at somewhat unpredictable rates - as a distributed set of varying sizes and rates - and therefore apps must be able to handle a variety of loads.

They are also typically I/O-bound, meaning that the application host, is doing a lot of waiting. This is caused by network latency and model inference from the integrated AI models.

To illustrate, take a look at the below system diagram of Weaviate that illustrates the steps in a retrieval-augmented-generation (RAG) query.

![Weaviate Latency](../../img/weaviate_latency.png)

This diagram shows that in one query, there may be 6(!) steps that involve data transport through a network. That’s even before we discuss the time taken for model inference and time taken for search inside Weaviate.

All this is to say that these are perfect for the use of `async/await` pattern coroutines that can be spun up at scale with very little overhead.

### When to Choose Async vs Sync Client

Given the above, is there a use case at all for the sync client? Well, the async client may be the best choice for concurrency unless there are specific reasons to **not** use it. For example - due to:

- Framework incompatibility (e.g. Django/Flask)
- Overhead of existing codebase
- Library incompatibility

However, in many cases the async client will be preferable, such as with new projects, or for where the highest throughput is desired. The async client offers:

- More efficient concurrency (lighter than threads)
- Compatibility with modern frameworks (e.g. FastAPI)
