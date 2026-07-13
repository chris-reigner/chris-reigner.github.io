# Context Manager for AI Agents

> *"The context window is not infinite storage — it is a finite, high-stakes workspace. The context manager is the architect that decides what lives there."*

As agentic systems grow in complexity — executing dozens of tool calls, spanning multiple turns, or running for hours — the naive approach of blindly appending everything to the context window breaks down. Context managers are the engineering layer that keeps agents coherent, cost-efficient, and capable across long horizons.

---

## What Is a Context Manager?

A **context manager** is the subsystem responsible for deciding what tokens occupy the LLM's context window at every inference step.

More precisely, it is the set of policies and mechanisms that govern:

- What information enters the context window (injection)
- How existing content is shaped and filtered (transformation)
- What gets compressed, evicted, or replaced when the window fills (eviction)
- How state is preserved and reconstructed across compression cycles (continuity)

Anthropic's Applied AI team has framed this discipline broadly as **"context engineering"** — *"the design, structuring, filtering, prioritization, ordering and governance of the information that an AI system uses during reasoning, planning or execution."*

> [Effective context engineering for AI agents — Anthropic Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

---

## What a Context Manager Is NOT

It is easy to confuse the context manager with adjacent systems. The boundaries matter.

| System | What it does | Where it lives |
|---|---|---|
| **Context Manager** | Governs what is *currently* in the active context window | Inside the inference loop |
| **Memory Layer** | Persists information *across* sessions (episodic, semantic, procedural) | External store (DB, files, vector index) |
| **RAG / Retrieval** | *Fetches* relevant documents from external knowledge at query time | External store → injected into context |
| **KV Cache** | Caches attention computation for unchanged prompt prefixes | Infrastructure / model serving layer |
| **Tool Router** | Decides *which* tools to call | Agent orchestration layer |

A context manager does not store memories. It does not retrieve documents. It does not persist anything beyond the current run. It only manages what is *actively inside* the window right now.

!!! note "The key distinction"
    The memory layer asks: *"What should the agent remember across sessions?"*
    The context manager asks: *"Of everything available right now, what should go into these N tokens?"*

---

## Why It Matters

LLMs are inherently stateless. Each API call starts from scratch. The context window is the only "working memory" the model has during a run. As agents execute long workflows:

- A single tool call can dump thousands of tokens (a file read, a search result, an API response)
- Tool schemas, system instructions, and retrieved docs all compete for the same budget

> [AI Agent Context Compression Strategies — Zylos Research](https://zylos.ai/research/2026-02-28-ai-agent-context-compression-strategies)

The financial argument is equally compelling: a customer service agent processing 10,000 conversations/day costs ~$255,000/year with unmanaged context and ~$102,000/year after a 60% context reduction via active management.

> [Context budgets: how to allocate tokens for AI agents — Wire Blog](https://usewire.io/blog/context-budgets-how-to-allocate-tokens-for-ai-agents/)

**Financial operations are among the heaviest context consumers.** A single SEC filing, earnings call transcript, or loan agreement can exceed 50,000 tokens. Financial agents that process contracts, trade logs, or regulatory documents at scale face compounding API costs: at enterprise volume — thousands of concurrent agent sessions running inside a bank or trading firm — unmanaged context turns into millions in annual infrastructure spend. The 60% cost reduction shown above is not hypothetical at that scale; it is a business-critical engineering constraint.

**Context management is also inference optimization.** LLM inference has two distinct phases: *prefill* (the model processes your input tokens) and *decoding* (the model generates output tokens). Prefill time scales with input length — roughly linearly for standard attention, worse with longer sequences for some architectures. A 100K-token context can add 2–5 seconds of prefill latency before the first output token appears. For agents embedded in real-time workflows — live trading signals, customer-facing support bots, interactive coding assistants — this is not just a cost issue. It is a hard performance bottleneck. A smaller context window means faster time-to-first-token and lower end-to-end latency for every inference call the agent makes.

---

## Core Capabilities

### 1. Token Budget Management

The context window is a finite resource with competing claimants. Effective management requires treating it as an **explicit budget** divided into categories, not a bag to fill arbitrarily.

**A production-grade budget allocation:**

| Category | Recommended Share |
|---|---|
| System instructions / persona | 10–15% |
| Tool definitions (JSON schemas) | 15–20% |
| Retrieved knowledge (RAG passages) | 30–40% |
| Conversation / action history | 20–30% |
| Output buffer (reserved for response) | 10–15% |

> [Context budgets: how to allocate tokens for AI agents — Wire Blog](https://usewire.io/blog/context-budgets-how-to-allocate-tokens-for-ai-agents/)

The context manager enforces this budget at each inference step, tracking token counts per category and triggering compaction or eviction when limits are approached. A common production threshold: trigger compression when context utilization exceeds **70% of available budget**.

---

### 2. Context Injection

The context manager controls what goes *in* before the model runs:

- **System prompt calibration**: structured with XML tags or Markdown headers into distinct sections (background, instructions, tool guidance, output format). Anthropic's principle: "right altitude" — specific enough to guide, flexible enough not to be brittle.
- **Few-shot examples**: injected selectively based on current task type, not exhaustively pre-loaded.
- **Retrieved documents**: dynamically injected at query time from the RAG layer; the context manager decides how much of the retrieved content to include given the current budget.
- **State summaries / progress files**: for long-running agents, a structured summary of prior work is injected at session start (e.g., `claude-progress.txt` in Anthropic's agent harness).
- **Pinned memory blocks** (Letta architecture): specific blocks are always compiled into the system prompt; others are fetched on demand.

> [Effective harnesses for long-running agents — Anthropic Engineering](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

---

### 3. Relevance Filtering and Context Prioritization

Rather than including all available context, a smart context manager applies relevance scoring:

- **Recency weighting**: recent turns carry higher weight by default unless past turns are explicitly relevant.
- **Semantic similarity scoring**: assess whether historical turns are relevant to the current query before including them.
- **Role-based filtering** (multi-agent): each agent receives only the context relevant to its function.
- **Importance tagging**: content marked high-salience (decisions, errors, file paths, entity states) is preserved across compression cycles, while low-salience content (intermediate search results, resolved errors) is evicted first.
- **Just-in-time loading**: agents maintain lightweight identifiers (file paths, query strings, URLs) rather than preloading full content, fetching only when a tool call actually needs it.

> [Adaptive Context Management — Relevance AI](https://relevanceai.com/blog/adaptive-context-management-for-production-ai-agents)

!!! note "Is this a semantic cache?"
    No — but the confusion is understandable. Both use semantic similarity, but they solve different problems:

    - A **semantic cache** stores full LLM responses keyed by query embedding. When a new query arrives, it checks if a semantically similar query was already answered and, if so, returns the cached response — *skipping the model call entirely*.
    - **Relevance filtering** uses semantic similarity as a *scoring heuristic* to decide which past turns or documents to include in the current prompt. The model still runs; you are only controlling what it sees.

    Semantic caching is about avoiding inference. Relevance filtering is about shaping the input to inference.

**Lossless vs. Lossy.** Relevance filtering is **lossless by design** — when implemented correctly:

- The content filtered out is **not deleted**. It remains in conversation history, a vector store, or an external buffer.
- It is only excluded from the active context window for this inference step.
- If a later turn makes that content relevant again, it can be retrieved and re-injected.

This is different from the lossy strategies covered in the Deep Dive section:

| Strategy | Original data preserved? | Classification |
|---|---|---|
| Relevance filtering | Yes — lives in external store | **Lossless** |
| Sliding window truncation | No — oldest messages dropped permanently | **Lossy** |
| LLM summarization | No — original replaced by a summary | **Lossy** |
| Observation masking | Partially — reasoning kept, raw output dropped | **Near-lossless** |

The trade-off is latency: if you filter something out and need it later, you must retrieve it — adding a lookup step. Summarization avoids that lookup but pays with permanent fidelity loss. Relevance filtering preserves fidelity at the cost of requiring a retrieval layer beneath it.

---

### 4. Tool Output Truncation and Summarization

Tool outputs are among the most context-polluting artifacts in agentic systems. A single file read, web search result, or database query can consume tens of thousands of tokens.

**Strategies:**

- **Goal-aware compaction**: truncate or summarize tool output around what is directly relevant to the current sub-task, not the full raw result.
- **Structured extraction**: from a meeting transcript, extract only key decisions and action items.
- **Observation masking**: replace older tool outputs with a placeholder label rather than keeping the full content (see [Deep Dive: History Compression](#5-history-compression-and-summarization)).
- **Anthropic `clear_tool_uses_20250919`**: an API-level strategy that automatically clears stale tool calls and results from the context when approaching limits.

Anthropic measured: *"In a 100-turn web search evaluation, context editing enabled agents to complete workflows that would otherwise fail due to context exhaustion — while reducing token consumption by 84%."*

> [Context editing — Claude API Docs](https://platform.claude.com/docs/en/build-with-claude/context-editing)

---

### 5. History Compression and Summarization

When the conversation or action history outgrows the available budget, the context manager must evict old content while preserving the signal needed for the agent to continue coherently. This is the most technically nuanced capability — covered in depth in the section below.

---

### 6. State Tracking and Continuity

Across compression cycles and context window resets, the context manager must maintain coherent run state:

- **Progress files**: Anthropic's agent harness uses `claude-progress.txt` so the agent can reconstruct task state when starting with a fresh context window.
- **Structured anchor state** (Factory.ai): four sections — intent, file modifications, decisions made, next steps — injected at each new session.
- **Memory blocks** (Letta): blocks are compiled from current database state into the system prompt at each inference call, ensuring the agent always sees the latest persisted state.
- **Compaction blocks** (Anthropic API): a `compaction` block serves as the authoritative state summary; all content prior to it is automatically dropped on subsequent requests.

> [Effective harnesses for long-running agents — Anthropic Engineering](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
> [Memory Blocks — Letta Blog](https://www.letta.com/blog/memory-blocks)

---

## Technical Deep Dive: History Compression and Summarization

This is the central challenge of context management for long-horizon agents. The core problem, named **context rot** by Factory.ai, is this: *"A model that remembers everything eventually remembers nothing useful. The noise from stale intermediate steps drowns the signal from the current task."*

The design space spans five major strategies, ranging from zero-overhead truncation to learned compression.

---

### Strategy 1: Sliding Window (Truncation)

**Mechanism:** Maintain a FIFO buffer of the most recent N messages or T tokens. When the buffer overflows, the oldest entry is discarded.

    History:  [msg1, msg2, msg3, msg4, msg5, msg6]
    Window=4: [            msg3, msg4, msg5, msg6]
                           ^ hard cut here

**Two variants:**

- **Message-based**: discard when `len(messages) > N` — simple but ignores token length variance.
- **Token-based**: discard when `token_count(messages) > T` — strictly superior, respects actual model constraints.

**Properties:**

- Zero computation overhead (no LLM call required)
- Completely lossless for retained content
- Hard information loss: anything outside the window is permanently gone
- No continuity across sessions

**Best for:** Short-horizon tasks, single-session chatbots, pipelines where historical context beyond a few turns is genuinely irrelevant.

> [Context management — OpenAI Agents SDK](https://openai.github.io/openai-agents-python/context/)

---

### Strategy 2: Full Reconstruction Summarization

**Mechanism:** When total history tokens exceed a threshold T, send the entire history to an LLM and generate a summary from scratch. The summary replaces the full history.

    History:  [msg1, msg2, msg3, msg4, msg5] → LLM → [SUMMARY_1]
    Next run: [SUMMARY_1, msg6, msg7, msg8] → LLM → [SUMMARY_2]

**Used by:** Early `ConversationSummaryMemory` in LangChain, OpenAI's `compact` endpoint.

**Properties:**

- High compression ratio (OpenAI's compact endpoint achieves 99.3% token removal)
- Progressive information loss across cycles ("summary drift") — details gradually disappear
- High compute cost: must process full history each time compression triggers
- At 99.3% removal, the resulting representation is opaque and cannot be inspected for correctness

**Production evaluation (Factory.ai, 36,000+ messages):**

| Method | Overall (0–5) | Accuracy (0–5) |
|---|---|---|
| Anchored iterative (Factory) | 3.70 | 4.04 |
| Anthropic SDK compaction | 3.44 | 3.74 |
| OpenAI compact | 3.35 | 3.43 |

Full reconstruction consistently scores lowest on accuracy. Structured approaches win.

> [Evaluating Context Compression for AI Agents — Factory.ai](https://factory.ai/news/evaluating-compression)

---

### Strategy 3: Anchored Iterative Summarization

**Mechanism:** Rather than regenerating the full summary, maintain a persistent **anchor state** and only process the *newly-evicted span* when compression triggers. The partial summary is merged into the anchor.

    Anchor state (always present):
      session_intent: "Refactor the auth module"
      file_modifications: [auth.py, tests/test_auth.py]
      decisions_made: ["Use JWT tokens", "Deprecate session cookies"]
      next_steps: ["Update middleware", "Add refresh token logic"]

    On compression trigger:
      1. Identify the newly-truncated span [msg_n .. msg_n+k]
      2. Summarize ONLY that span
      3. Merge partial summary → anchor state
      4. Discard the span

**Why structure matters:** *"Structure forces preservation — by dedicating sections to specific information types, the summary cannot silently drop file paths or skip decisions. Each section acts as a checklist."*
— Factory.ai Engineering

**Properties:**

- Low overhead: only the newly-evicted delta is processed, not the full history
- High accuracy: structured sections prevent silent omission of critical details
- Weak spot: artifact tracking (exact file path retention) remains challenging — scores 2.19–2.45/5.0 across all tested methods

**Used by:** Factory.ai's production coding agent.

> [Compressing Context — Factory.ai](https://factory.ai/news/compressing-context)

---

### Strategy 4: Observation Masking

**Mechanism:** Replace older tool outputs (observations) with a lightweight placeholder token or label, rather than summarizing them. The reasoning trace and action decisions are preserved verbatim; only the raw observations are masked.

    Before masking:
      [TOOL_CALL: search("React hooks")]
      [OBSERVATION: "React hooks are functions that let you use state and lifecycle..."
       ... 4,000 tokens of search results ...]

    After masking:
      [TOOL_CALL: search("React hooks")]
      [OBSERVATION: <masked>]

**Key insight from JetBrains Research (NeurIPS DL4C Workshop, December 2025):**

Simple observation masking is **as effective as LLM summarization** for most agent tasks, at a fraction of the cost.

Results on SWE-bench:

- Observation masking: 2.6% **higher** solve rate than summarization with Qwen3-Coder 480B
- Both methods cut costs by 50%+ vs. unbounded context
- LLM summarization ran ~15% more steps (agents ran longer), partially offsetting efficiency gains
- Summarization overhead: over **7% of total cost** per instance for large models

**Why masking works:** Many tool outputs carry zero marginal value after the next few turns. Intermediate search results that were already acted upon, error traces that were resolved, partial file reads that were superseded — masking them is equivalent to summarizing them as "this was processed," which is exactly what a useful summary would say.

**Recommendation (JetBrains):** Use observation masking as the primary defense. Fall back to selective LLM summarization only for extended trajectories where semantic continuity genuinely matters.

> [Cutting Through the Noise — JetBrains Research Blog](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
> [The Complexity Trap — NeurIPS DL4C 2025 (GitHub)](https://github.com/JetBrains-Research/the-complexity-trap)

---

### Strategy 5: Hierarchical and Tiered Compression

**Mechanism:** Operate on multiple granularity levels simultaneously. Recent content stays verbatim; older content is progressively compressed at increasing abstraction levels.

**Three-tier pattern (production):**

    Tier 1 (verbatim):    Last 5–10 turns — exact wording, full tool outputs
    Tier 2 (compressed):  Recent session summary — medium abstraction, key decisions preserved
    Tier 3 (extracted):   Long-term facts — entity states, critical file paths, system decisions

**NexusSum** (arXiv:2505.24575, 2025): A three-stage hierarchical multi-agent framework that formalizes this:

1. **Preprocessing**: convert raw dialogue to descriptive prose
2. **Narrative Summarization**: generate an initial summary
3. **Iterative Compression**: refine for length while preserving key details

**Recursive variant:** When a session accumulates multiple compression cycles, the intermediate summaries themselves can be recursively compressed into higher-level abstracts. Risk: compounding drift — each step introduces subtle rewording shifts that accumulate.

> [NexusSum — arXiv 2505.24575](https://arxiv.org/html/2505.24575v1)
> [LLM Chat History Summarization Guide — Mem0.ai](https://mem0.ai/blog/llm-chat-history-summarization-guide-2025)

---

### Strategy 6: Failure-Driven Guideline Optimization (ACON)

**Mechanism:** Treat context compression as an optimization problem with iterative self-improvement. A compression policy (expressed as natural language guidelines) is refined based on observed failures.

    Loop:
      1. Collect paired trajectories:
           (full-context success, compressed-context failure) pairs
      2. Failure analysis: LLM examines why the compressed context caused failure
           → "The agent lost track of the current file path after compression"
      3. Guideline update: compression prompt is updated to preserve that info class
           → "Always preserve all file paths and their modification status"
      4. Distillation: optimized logic is distilled into a smaller, cheaper compressor

**Results** (ACON, arXiv:2510.00615, ICLR 2025):

- 26–54% reduction in peak token usage
- Maintains **95%+ task accuracy** when distilled into smaller models
- +32% on AppWorld, +20% on OfficeBench, +46% on Multi-objective QA
- Model-agnostic, gradient-free: works with any API-accessible LLM

> [ACON — arXiv 2510.00615](https://arxiv.org/abs/2510.00615)

---

### Lossless vs. Lossy: Comparison Matrix

| Strategy | Lossless? | Overhead | Token Reduction | Best For |
|---|---|---|---|---|
| Sliding window (token-based) | Lossless (within window) | Zero | Variable | Short sessions |
| Observation masking | Lossless for reasoning | Near-zero | 50%+ | Tool-heavy agents |
| Full reconstruction summarization | Lossy | High (full history) | ~95%+ | Simple chatbots |
| Anchored iterative summarization | Lossy (structured) | Low (delta only) | ~98% | Long coding sessions |
| Hierarchical / tiered | Lossy (graduated) | Medium | Variable | Long conversations |
| ACON guideline optimization | Lossy (95%+ accurate) | Medium (optimization phase) | 26–54% | Long-horizon tasks |
| KV cache compression (KVzip) | Near-lossless | Low (inference-time) | 3–4× (cache) | Inference optimization |

**KVzip** (arXiv:2505.23416, 2025) operates at the KV cache level — evicting low-importance KV pairs based on their importance for context reconstruction. Achieves 3–4× memory reduction with negligible accuracy loss, scaling to 170K tokens.

> [KVzip — arXiv 2505.23416](https://arxiv.org/abs/2505.23416)

---

## Implementations in the Wild

### Letta (formerly MemGPT)

The original paper introduced the LLM-as-operating-system paradigm: the model manages its own memory like an OS manages RAM and disk via paging.

**Two-tier architecture:**

- **Main context ("RAM")**: system instructions + writeable core memory blocks + recent message history
- **External context ("Disk")**: recall storage (full message history) + archival storage (semantic facts)

The agent uses dedicated tools (`archival_memory_insert`, `core_memory_replace`, `conversation_search`) to self-manage what flows between tiers. Context blocks are compiled from the current database state into the system prompt at each inference via Jinja templating.

> [MemGPT concepts — Letta Docs](https://docs.letta.com/concepts/memgpt/)
> [Memory Blocks — Letta Blog](https://www.letta.com/blog/memory-blocks)

---

### Anthropic: Compaction API

Server-side automatic summarization triggered by a token threshold:

    context_management={
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 150000},  # min: 50,000
                "pause_after_compaction": False,
                "instructions": "Preserve all file paths, variable names, and technical decisions."
                # Custom instructions REPLACE the default prompt entirely
            }
        ]
    }

When triggered:

1. Generates a `compaction` block containing the summary
2. On subsequent requests, all message blocks prior to the compaction block are automatically dropped
3. Prompt caching is preserved across compaction events

> [Compaction — Claude API Docs](https://platform.claude.com/docs/en/build-with-claude/compaction)
> [Managing context on the Claude Developer Platform — Anthropic](https://www.anthropic.com/news/context-management)

---

### LangChain: ConversationSummaryBufferMemory

The most sophisticated of LangChain's built-in memory types — a hybrid buffer + summarization approach:

    from langchain.memory import ConversationSummaryBufferMemory

    memory = ConversationSummaryBufferMemory(
        llm=llm,              # can be a cheaper/faster model than the agent LLM
        max_token_limit=2000  # token threshold (not message count)
    )

**Behavior:**

- Recent messages stay verbatim in the buffer up to `max_token_limit`
- When the buffer exceeds the limit, the **oldest** messages are summarized and the raw content is dropped
- `prune()`: explicitly triggers compression of the oldest buffer content
- `predict_new_summary()`: generates what the summary would be given a new exchange (useful for testing)

The key advantage over `ConversationSummaryMemory`: recent interactions remain in raw form (maximum fidelity) while older ones are compressed.

> [ConversationSummaryBufferMemory — LangChain Python Docs](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.summary_buffer.ConversationSummaryBufferMemory.html)
> [Conversational Memory for LLMs — Pinecone](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)

---

### OpenAI Agents SDK: Sessions + Compaction

**Two session modes:**

- `OpenAIConversationsSession`: server-side history via the Conversations API
- `OpenAIResponsesCompactionSession`: wraps the Responses API with automatic compaction

**Explicit management API:**

    session.get_items()   # retrieve all items in history
    session.pop_item()    # remove and return most recent item
    session.clear()       # remove all items

**Memory distillation** (a separate capability): extracts high-quality, durable signals (facts, preferences, key decisions) from conversation history and records them as persistent "memory notes" that survive across sessions.

> [Sessions overview — OpenAI Agents SDK](https://openai.github.io/openai-agents-python/sessions/)

---

## Key Takeaways

1. **Treat context as a budget, not a bag.** Allocate token categories explicitly before agents run.
2. **Observation masking beats summarization in cost-efficiency.** For tool-heavy agents, masking old observations is simpler, cheaper, and often equally effective as LLM-based summarization.
3. **Structure defeats drift.** LLM-based summarization without forced structure produces opaque summaries that silently drop critical details across compression cycles.
4. **The effective limit of any model is ~65–70% of its nominal context.** Design your budget ceilings accordingly.
5. **File paths and artifact tracking remain the hardest problem.** All production approaches score poorly (2.19–2.45/5.0) at preserving exact artifact references across compression.

---

## Resources

- [Effective context engineering for AI agents — Anthropic Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Context Engineering Guide — PromptingGuide.ai](https://www.promptingguide.ai/guides/context-engineering-guide)
- [Compaction — Claude API Docs](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [Context editing — Claude API Docs](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- [Effective harnesses for long-running agents — Anthropic Engineering](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Evaluating Context Compression for AI Agents — Factory.ai](https://factory.ai/news/evaluating-compression)
- [Cutting Through the Noise — JetBrains Research Blog](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- [The Complexity Trap — NeurIPS DL4C 2025 (GitHub)](https://github.com/JetBrains-Research/the-complexity-trap)
- [ACON: Optimizing Context Compression — arXiv 2510.00615](https://arxiv.org/abs/2510.00615)
- [NexusSum: Hierarchical Summarization — arXiv 2505.24575](https://arxiv.org/html/2505.24575v1)
- [KVzip: KV Cache Compression — arXiv 2505.23416](https://arxiv.org/abs/2505.23416)
- [MemGPT / Letta Docs](https://docs.letta.com/concepts/memgpt/)
- [LangChain ConversationSummaryBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.summary_buffer.ConversationSummaryBufferMemory.html)
- [Context budgets: how to allocate tokens — Wire Blog](https://usewire.io/blog/context-budgets-how-to-allocate-tokens-for-ai-agents/)
- [AI Agent Context Compression Strategies — Zylos Research](https://zylos.ai/research/2026-02-28-ai-agent-context-compression-strategies)
- [Cognitive Memory in LLMs — arXiv 2504.02441](https://arxiv.org/html/2504.02441v1)
