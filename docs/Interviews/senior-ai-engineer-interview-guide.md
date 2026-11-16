# Senior AI Engineer Interview Guide (2026)

A bank of **precise questions with precise answers** for interviewing senior AI engineers. Broad topics are split into several narrow questions. Each answer lists the expected points (several valid options where relevant), quantified results where they exist, and the sources to check them.

Answers are folded so the page can be used live during an interview. Case studies and system design exercises are kept separate.

| Level | Meaning |
|---|---|
| 🟢 **Core** | Any senior AI engineer should answer correctly and quickly |
| 🟠 **Senior** | Differentiates senior from mid-level: mechanisms, numbers, trade-offs |
| 🔴 **Staff+** | Depth: internals, maths, cross-cutting enterprise concerns |

## 1. Software engineering principles in practice

### SE-01 · Dependency inversion for LLM clients 🟢

**Q:** What does the Dependency Inversion Principle say, and how do you apply it to code that calls LLM providers?

??? success "Answer"
    - Principle: high-level modules must not depend on low-level modules; both depend on **abstractions**.
    - Application:
        - Define a port, e.g. `class LLMClient(Protocol): async def complete(self, messages, *, schema=None) -> Completion`.
        - One **adapter** per provider (OpenAI, Anthropic, Mistral, vLLM) that maps provider types and errors to your own types.
        - Inject the adapter via constructor or a DI container; business logic imports only the port.
        - Tests use a `FakeLLMClient` implementing the same protocol.
    - This is the ports-and-adapters (hexagonal) architecture.

    **Sources:** [Hexagonal architecture (Cockburn)](https://alistair.cockburn.us/hexagonal-architecture/) · [PEP 544 Protocols](https://peps.python.org/pep-0544/) · R. C. Martin, *Clean Architecture* (2017)

### SE-02 · Retry policy for LLM APIs 🟢

**Q:** Which errors do you retry when calling an LLM API, and with what backoff formula?

??? success "Answer"
    - **Retry:** HTTP 429 (rate limit), 500/502/503/504, 529 (overloaded), timeouts, connection resets.
    - **Do not retry:** 400 (bad request, context length exceeded), 401/403, 404, content-policy refusals. These are deterministic.
    - Honour the `Retry-After` header when present.
    - **Exponential backoff with full jitter:** `sleep = random.uniform(0, min(cap, base * 2 ** attempt))`. Jitter de-synchronises clients and avoids retry storms.
    - Bound attempts (3–5) and the total deadline; libraries: `tenacity`, or provider SDKs' built-in `max_retries`.

    **Sources:** [Exponential Backoff and Jitter (AWS Architecture Blog)](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) · [tenacity](https://tenacity.readthedocs.io/)

### SE-03 · Idempotency keys 🟠

**Q:** What is an idempotency key and how do you implement it for a side-effecting tool (e.g. "create ticket")?

??? success "Answer"
    - A client-generated unique key for **one logical operation**; repeating the request with the same key returns the original result instead of executing twice.
    - Implementation:
        - Key derived deterministically for agents, e.g. `hash(run_id, step_id, tool_name, canonical_args)`.
        - Server table `idempotency(key PRIMARY KEY, request_hash, status, response, created_at)`; insert-if-absent inside the same transaction as the side effect.
        - Same key + different payload → reject (409/422).
        - Expire keys after a TTL (Stripe uses 24 h).
    - Needed because a timeout does not tell you whether the side effect happened.

    **Sources:** [Stripe — Idempotent requests](https://docs.stripe.com/api/idempotent_requests) · [Designing robust APIs with idempotency (Stripe blog)](https://stripe.com/blog/idempotency)

### SE-04 · Circuit breaker 🟢

**Q:** Describe the states of a circuit breaker and how you would use one in front of an LLM provider.

??? success "Answer"
    - **Closed:** calls pass; failures counted in a rolling window.
    - **Open:** after a failure threshold (e.g. 50% errors over 20 calls), calls fail fast for a cool-down period.
    - **Half-open:** after the cool-down, a few trial calls; success → closed, failure → open.
    - LLM use: on open, fall back to a secondary provider/model, a cached answer, or a degraded response. Combine with per-call timeouts and retries (retries inside, breaker outside).

    **Sources:** [CircuitBreaker (Martin Fowler)](https://martinfowler.com/bliki/CircuitBreaker.html) · M. Nygard, *Release It!* (2nd ed., 2018)

### SE-05 · Blocking calls in async code 🟢

**Q:** A FastAPI `async def` endpoint calls the LLM with a synchronous HTTP client. What happens under load and how do you fix it?

??? success "Answer"
    - The sync call **blocks the event loop**: every other request on that worker waits, so latency grows with concurrency and throughput collapses.
    - Fixes (any of):
        1. Use an async client (`httpx.AsyncClient`, `AsyncOpenAI`, `AsyncAnthropic`) and `await` it.
        2. Declare the endpoint with `def` instead of `async def`: FastAPI runs it in a thread pool.
        3. Offload explicitly: `await asyncio.to_thread(sync_call, ...)` or `run_in_threadpool`.
    - Reuse one client (connection pool) per process rather than creating one per request.

    **Sources:** [FastAPI — Concurrency and async/await](https://fastapi.tiangolo.com/async/) · [Async (this handbook)](../ML%20Engineering/Chapter%201%20-%20Engineering/Async.md)

### SE-06 · Bounded parallel LLM calls 🟠

**Q:** You must summarise 10,000 documents with an API limited to 50 concurrent requests. Write the concurrency pattern.

??? success "Answer"
    - `asyncio.Semaphore(50)` around each call, launched with `asyncio.TaskGroup` (Python 3.11+) or `asyncio.gather`:

            sem = asyncio.Semaphore(50)

            async def one(doc):
                async with sem:
                    return await client.summarise(doc)

            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(one(d)) for d in docs]

    - `TaskGroup` gives structured concurrency: one failure cancels the siblings. Use `gather(..., return_exceptions=True)` if partial results are acceptable.
    - For large offline jobs, the provider **Batch API** is usually better: about 50% cheaper and no client-side rate limiting.

    **Sources:** [asyncio.TaskGroup](https://docs.python.org/3/library/asyncio-task.html#task-groups) · [asyncio.Semaphore](https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore) · [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)

### SE-07 · Guaranteeing structured outputs 🟠

**Q:** What are the options to get valid JSON matching a schema from an LLM, and how do they differ?

??? success "Answer"
    1. **Provider-native structured outputs** (JSON schema, `strict` mode): constrained decoding server-side; schema adherence is guaranteed.
    2. **Tool / function calling** with a schema: widely supported, adherence high but not always guaranteed without strict mode.
    3. **Constrained decoding on self-hosted models**: grammar/FSM masking of invalid tokens (Outlines, XGrammar, llguidance; integrated in vLLM and SGLang).
    4. **Validate-and-retry** (Instructor, Pydantic AI): parse with Pydantic, send validation errors back to the model, retry.
    - Constrained decoding guarantees **syntax**, not **semantics**: still validate business rules.
    - Evidence: OpenAI reported 100% schema adherence with Structured Outputs vs under 40% for `gpt-4-0613` with prompting only.

    **Sources:** [Introducing Structured Outputs (OpenAI)](https://openai.com/index/introducing-structured-outputs-in-the-api/) · [Outlines paper, arXiv:2307.09702](https://arxiv.org/abs/2307.09702) · [XGrammar, arXiv:2411.15100](https://arxiv.org/abs/2411.15100) · [Instructor](https://python.useinstructor.com/)

### SE-08 · Testing code that calls an LLM 🟢

**Q:** How do you unit-test business logic that calls an LLM without calling the model?

??? success "Answer"
    - **Fake implementing the port** (see SE-01) returning scripted responses, including errors and malformed outputs.
    - **HTTP-level mocking**: `respx` (httpx) or `responses` (requests).
    - **Record/replay**: `vcrpy` / `pytest-recording` cassettes of real responses, replayed in CI.
    - **Contract tests** against the real API, run nightly, to detect provider changes.
    - Quality of the model's answers is tested by **evals** (section 8), not unit tests.

    **Sources:** [respx](https://lundberg.github.io/respx/) · [vcrpy](https://vcrpy.readthedocs.io/) · [pytest-recording](https://github.com/kiwicom/pytest-recording)

### SE-09 · Prompt versioning 🟠

**Q:** What are the options to version and release prompts, and their trade-offs?

??? success "Answer"
    1. **Prompts in the repository** (templates as files): reviewed in PRs, released with the code, traceable by git SHA. Simple, but changing a prompt needs a deploy.
    2. **Prompt registry** (Langfuse, MLflow Prompt Registry, LangSmith): versions and labels (`production`, `staging`), runtime fetch with caching, change without deploy. Needs governance and eval gates to avoid untested changes.
    3. **Hybrid**: registry as source of truth, version pinned in config, CI syncs and runs evals.
    - In all cases: log the **prompt version in every trace**, and gate changes with an eval suite.

    **Sources:** [Prompt Release Lifecycle (this handbook)](../AI%20Agents%20Engineering/5.%20Agent%20Ops/prompt%20release%20lifecycle.md) · [Langfuse prompt management](https://langfuse.com/docs/prompts/get-started) · [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/)

### SE-10 · ADR format 🟢

**Q:** What sections does an Architecture Decision Record contain, and how do you handle a decision that changes later?

??? success "Answer"
    - Nygard format: **Title** (numbered), **Status** (proposed / accepted / deprecated / superseded), **Context** (forces and constraints), **Decision** (in active voice, "We will…"), **Consequences** (positive and negative).
    - MADR adds **considered options** with pros/cons and decision drivers.
    - Stored as Markdown in the repo, e.g. `docs/adr/0007-use-pgvector.md`, reviewed by PR.
    - ADRs are **immutable**: a changed decision gets a new ADR that supersedes the old one, and the old status becomes "Superseded by ADR-0012".

    **Sources:** [Documenting Architecture Decisions (Nygard, 2011)](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) · [MADR](https://adr.github.io/madr/) · [adr.github.io](https://adr.github.io/)

### SE-11 · DORA metrics 🟢

**Q:** Name the DORA software delivery metrics and the "elite" thresholds.

??? success "Answer"
    - **Deployment frequency**: on demand (multiple per day).
    - **Lead time for changes** (commit → production): under one day.
    - **Change failure rate**: about 5%.
    - **Failed deployment recovery time**: under one hour.
    - DORA 2024 also introduced **rework rate** (unplanned deployments to fix user-facing bugs).
    - Throughput metrics (first two) and stability metrics (last two) are tracked together.

    **Sources:** [DORA — Software delivery performance metrics](https://dora.dev/guides/dora-metrics-four-keys/) · [DORA 2024 Accelerate State of DevOps Report](https://dora.dev/research/2024/dora-report/)

### SE-12 · Evidence on AI coding productivity 🟠

**Q:** What do controlled studies say about AI coding assistants' impact on productivity?

??? success "Answer"
    - **GitHub Copilot RCT (2022):** developers completed an HTTP-server task **55.8% faster**: greenfield, small, well-specified task.
    - **METR RCT (2025):** 16 experienced open-source developers on their own large repos were **19% slower** with AI tools, while believing they were about 20% faster.
    - **DORA 2024:** a 25% increase in AI adoption was associated with **−1.5% delivery throughput and −7.2% delivery stability**, despite higher self-reported productivity.
    - Takeaway: gains depend on task type, codebase familiarity and review/verification capacity; measure with delivery metrics, not perception.

    **Sources:** [Peng et al., arXiv:2302.06590](https://arxiv.org/abs/2302.06590) · [METR study](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/) · [DORA 2024 report](https://dora.dev/research/2024/dora-report/)

### SE-13 · Reproducible Python AI builds 🟢

**Q:** How do you make a Python AI service build reproducibly from laptop to production?

??? success "Answer"
    - **Lockfile** with exact versions and hashes (`uv.lock`, `poetry.lock`); CI installs with `uv sync --locked` (fails if the lock is stale) or `--frozen`.
    - Pin the Python version (`.python-version`, `requires-python`).
    - Pin the **PyTorch / CUDA index** explicitly (CUDA wheel variants differ).
    - Docker: base image pinned by **digest**, multi-stage build (builder with compilers, slim runtime), non-root user, `.dockerignore`.
    - Model weights pinned by revision/commit hash, not `main`.

    **Sources:** [uv — Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) · [UV (this handbook)](../ML%20Engineering/Chapter%201%20-%20Engineering/uv.md) · [Docker (this handbook)](../ML%20Engineering/Chapter%201%20-%20Engineering/Docker.md)

### SE-14 · Configuration and secrets 🟢

**Q:** Where do API keys, model names and thresholds live in a twelve-factor AI service?

??? success "Answer"
    - Config that varies per environment (endpoints, model names, feature flags) → **environment variables**, loaded and validated at startup (e.g. `pydantic-settings`).
    - Secrets → **secret manager** (Vault, AWS Secrets Manager, Azure Key Vault), injected at runtime; never in code, images or logs; rotated.
    - Behavioural config (thresholds, prompt versions) → versioned config files or a config/flag service, so it can change without rebuilding.
    - Pre-commit / CI secret scanning (gitleaks, trufflehog).

    **Sources:** [The Twelve-Factor App — Config](https://12factor.net/config) · [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) · [gitleaks](https://github.com/gitleaks/gitleaks)

### SE-15 · Data validation in pipelines 🟢

**Q:** Which tools validate data in an ML/LLM pipeline, and at which level does each work?

??? success "Answer"
    - **Pydantic**: record level (API payloads, LLM outputs, tool arguments).
    - **Pandera**: DataFrame schemas (types, ranges, nullability, custom checks) as code, usable as decorators in pipelines.
    - **Great Expectations**: expectation suites on datasets, data docs, checkpoints; suited to data platform teams.
    - **dbt tests**: in the warehouse (unique, not null, relationships).
    - Place checks at **boundaries** (ingestion, before training/indexing, before serving); decide fail-fast vs quarantine per check.

    **Sources:** [Pandera](https://pandera.readthedocs.io/) · [Great Expectations](https://docs.greatexpectations.io/) · [Pydantic](https://docs.pydantic.dev/)

### SE-16 · OpenTelemetry for LLM calls 🟠

**Q:** What are the OpenTelemetry GenAI semantic conventions and which attributes should every LLM span carry?

??? success "Answer"
    - A standard schema for GenAI telemetry (spans, metrics, events), so traces are portable across backends (Langfuse, Phoenix, Datadog, Grafana…).
    - Key span attributes:
        - `gen_ai.operation.name` (`chat`, `embeddings`, `execute_tool`, `invoke_agent`)
        - `gen_ai.provider.name` (formerly `gen_ai.system`), `gen_ai.request.model`, `gen_ai.response.model`
        - `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
        - `gen_ai.request.temperature`, `gen_ai.request.max_tokens`, `gen_ai.response.finish_reasons`
    - Prompt/completion content is **opt-in** (PII risk); add your own attributes for prompt version, tenant and cost.

    **Sources:** [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) · [Observability (this handbook)](../AI%20Agents%20Engineering/5.%20Agent%20Ops/observability.md)

### SE-17 · Reviewing AI-generated code 🟠

**Q:** List the defects you specifically check for when reviewing AI-generated Python that calls an LLM.

??? success "Answer"
    - **Hallucinated APIs and packages**: imports of non-existent packages (supply-chain risk, "slopsquatting"), wrong SDK method names or parameters.
    - **Security**: secrets in code, user input concatenated into prompts or shell/SQL, `eval`/`exec`, unsafe deserialisation (`pickle`), PII in logs.
    - **Reliability**: no timeouts, unbounded retries, blocking calls in `async`, broad `except Exception: pass`, no validation of model output.
    - **Tests**: tests that assert on mocks only, or were weakened to pass.
    - Evidence: across 576k generated samples from 16 LLMs, **19.7% of recommended packages did not exist** (about 5% for commercial models vs about 22% for open models).

    **Sources:** [Spracklen et al., "We Have a Package for You!", arXiv:2406.10279](https://arxiv.org/abs/2406.10279) · [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)

## 2. Coding development practices with AI agents

### AC-01 · Content of a context file 🟢

**Q:** What should go in `CLAUDE.md` / `AGENTS.md`, and what should not?

??? success "Answer"
    - **In:** build/test/lint commands, repository map, conventions not enforced by tools, non-obvious gotchas, workflow rules (branching, PR format).
    - **Out:** anything a linter/formatter already enforces, long docs (link them instead), secrets, stale information.
    - Keep it short (it is loaded into every session and consumes context); use hierarchy: user-level, project root, and subdirectory files loaded when working there; imports (`@path`) for detail.
    - `AGENTS.md` is the cross-tool open format (Codex, Cursor, Jules, etc.); `CLAUDE.md` is Claude Code's.

    **Sources:** [Claude Code — Memory](https://docs.claude.com/en/docs/claude-code/memory) · [AGENTS.md](https://agents.md/) · [Claude Code best practices (Anthropic)](https://www.anthropic.com/engineering/claude-code-best-practices)

### AC-02 · Extension mechanisms of coding agents 🟠

**Q:** In Claude Code (or an equivalent agent), what is the difference between the context file, skills, slash commands, hooks, subagents and MCP servers?

??? success "Answer"
    | Mechanism | Loaded / triggered | Use for |
    |---|---|---|
    | `CLAUDE.md` | Always in context | Project facts and conventions |
    | Skills (`SKILL.md`) | Only name/description loaded; body loaded on demand when relevant | Reusable procedures, scripts, domain know-how |
    | Slash commands | Invoked explicitly by the user | Repeatable prompts/workflows |
    | Hooks | Deterministic shell commands on lifecycle events (`PreToolUse`, `PostToolUse`, `Stop`…) | Rules that must **always** apply (format, block commands) |
    | Subagents | Separate context window, own tools/prompt | Isolated investigation, review, parallel work |
    | MCP servers | External tools/resources via protocol | Integrations (DB, tickets, browser) |

    **Sources:** [Claude Code — Hooks](https://docs.claude.com/en/docs/claude-code/hooks) · [Agent Skills (Anthropic)](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) · [Claude Code — Subagents](https://docs.claude.com/en/docs/claude-code/sub-agents) · [Agent Skills (this handbook)](../AI%20Agents%20Engineering/3.%20Runtime%20Engineering/agent%20skills.md)

### AC-03 · Hooks vs instructions 🟢

**Q:** Why enforce "never run `git push --force`" with a hook rather than an instruction in the context file?

??? success "Answer"
    - Instructions are **probabilistic**: the model can ignore or forget them, especially in long contexts.
    - Hooks are **deterministic** code executed by the harness. In Claude Code a `PreToolUse` hook exiting with code **2** blocks the tool call and returns stderr to the model.
    - Rule of thumb: guidance → instructions; invariants (security, formatting, forbidden commands) → hooks or permission deny rules.

    **Sources:** [Claude Code — Hooks reference](https://docs.claude.com/en/docs/claude-code/hooks) · [Claude Code — Settings and permissions](https://docs.claude.com/en/docs/claude-code/settings)

### AC-04 · Explore–plan–code–commit 🟢

**Q:** Describe the recommended workflow for a multi-file change with a coding agent, and why planning comes first.

??? success "Answer"
    1. **Explore:** ask the agent to read relevant files, explicitly *without* writing code.
    2. **Plan:** ask for a written plan (plan mode / "think hard"); review and correct it; optionally save it to a file.
    3. **Code:** implement step by step, running tests/linters after each step.
    4. **Commit / PR:** small commits, descriptive messages, PR reviewed by a human.
    - Why: without exploration and planning, agents jump to code and solve the wrong problem; correcting a plan is much cheaper than correcting a diff.

    **Sources:** [Claude Code best practices (Anthropic)](https://www.anthropic.com/engineering/claude-code-best-practices) · [Coding Agent (this handbook)](../AI%20Agents%20Engineering/5.%20Agent%20Ops/coding-agent.md)

### AC-05 · TDD with an agent 🟠

**Q:** How do you run test-driven development with a coding agent, and how do you stop it from gaming the tests?

??? success "Answer"
    1. Ask the agent to write tests from the expected behaviour (input/output pairs), stating explicitly it is doing TDD and must **not** write the implementation.
    2. Run tests, **confirm they fail**.
    3. **Commit the tests.**
    4. Ask for an implementation that passes the tests **without modifying them**; iterate until green.
    5. Review for overfitting (special-casing test inputs), optionally with an independent subagent.
    - Protect test files: commit before implementation, check the diff touches no test files, or use a hook that blocks edits under `tests/`.

    **Sources:** [Claude Code best practices — TDD workflow](https://www.anthropic.com/engineering/claude-code-best-practices) · K. Beck, *Test-Driven Development: By Example* (2002)

### AC-06 · Running agents in parallel 🟠

**Q:** How do you run several coding agents in parallel on the same repository without conflicts?

??? success "Answer"
    - **git worktrees**: `git worktree add ../repo-feature-x -b feature-x`; each agent works in its own checkout and branch, sharing one `.git`.
    - Alternatives: separate clones, containers/devcontainers per agent, or cloud agents working on branches.
    - **Headless mode** for automation (`claude -p "..."` in CI or scripts, with JSON output).
    - Split work along independent boundaries (different modules) to limit merge conflicts; merge via PRs.

    **Sources:** [git-worktree](https://git-scm.com/docs/git-worktree) · [Claude Code — Common workflows (worktrees)](https://docs.claude.com/en/docs/claude-code/common-workflows) · [Claude Code — Headless mode](https://docs.claude.com/en/docs/claude-code/headless)

### AC-07 · Managing context in long agent sessions 🟠

**Q:** What techniques keep a coding agent effective in long sessions?

??? success "Answer"
    - `/clear` between unrelated tasks; `/compact` (or auto-compaction) to summarise history.
    - Delegate wide searches to **subagents** so only their summary enters the main context.
    - Persist state outside the context: plan files, progress/todo files, commit history.
    - Limit loaded tool definitions (each MCP server's tools consume context); prefer on-demand skills.
    - Reason: performance degrades as context grows ("context rot"), and tokens cost money and latency.

    **Sources:** [Effective context engineering for AI agents (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) · [Context Rot (Chroma research)](https://research.trychroma.com/context-rot) · [Context Manager (this handbook)](../AI%20Agents%20Engineering/3.%20Runtime%20Engineering/context-manager.md)

### AC-08 · Sandboxing and permissions 🟠

**Q:** What are the options to restrict what a coding agent can do on a machine?

??? success "Answer"
    - **Permission modes**: ask before each edit/command (default), accept edits, plan-only.
    - **Allow/deny rules** per tool and command pattern in settings (e.g. allow `Bash(npm test:*)`, deny `Bash(curl:*)`, deny reading `.env`).
    - **OS-level sandbox**: filesystem write restriction to the workspace and network egress allowlist.
    - **Containers / devcontainers / VMs** with no production credentials; skip-permission modes only inside such isolation.
    - Least-privilege tokens (read-only GitHub tokens, scoped cloud roles).

    **Sources:** [Claude Code — Security](https://docs.claude.com/en/docs/claude-code/security) · [Claude Code sandboxing (Anthropic)](https://www.anthropic.com/engineering/claude-code-sandboxing) · [Claude Code — Devcontainer](https://docs.claude.com/en/docs/claude-code/devcontainer)

### AC-09 · Prompt injection against coding agents 🟠

**Q:** Through which channels can a coding agent be prompt-injected, and what are the mitigations?

??? success "Answer"
    - **Vectors:** README/docs/comments in the repo, GitHub issues and PR comments, fetched web pages, dependency source code, MCP tool outputs, malicious skills or MCP servers.
    - **Impact:** secret exfiltration (via `curl`, DNS, git push), destructive commands, backdoored code.
    - **Mitigations:** network egress allowlist, no secrets in the environment, approval for commands outside an allowlist, pinned and reviewed MCP servers/skills, human review of diffs, running untrusted repos only in isolated containers.
    - Framework: the "lethal trifecta" (private data + untrusted content + external communication) → remove at least one leg.

    **Sources:** [The lethal trifecta (Simon Willison)](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) · [OWASP Top 10 for Agentic Applications (this handbook)](../AI%20Agents%20Engineering/5.%20Agent%20Ops/vulnerabilities.md)

### AC-10 · Coding agent benchmarks 🟠

**Q:** Name the main coding-agent benchmarks, what they measure, and their limits.

??? success "Answer"
    - **SWE-bench Verified:** 500 real GitHub issues (Python repos) human-validated as solvable; graded by hidden unit tests.
    - **Terminal-Bench:** end-to-end tasks in a terminal environment (build, debug, configure).
    - **Aider Polyglot:** 225 hard Exercism exercises in C++, Go, Java, JavaScript, Python, Rust.
    - **LiveCodeBench:** problems collected continuously from contests, filtered by date to limit contamination.
    - **Limits:** training-data contamination, scores depend heavily on the **scaffold/harness**, Python-centric, single-attempt scores hide variance → run an internal eval on your own tasks.

    **Sources:** [SWE-bench Verified (OpenAI)](https://openai.com/index/introducing-swe-bench-verified/) · [Terminal-Bench](https://www.tbench.ai/) · [Aider leaderboards](https://aider.chat/docs/leaderboards/) · [LiveCodeBench, arXiv:2403.07974](https://arxiv.org/abs/2403.07974)

## 3. Fine-tuning LLMs / SLMs

### FT-01 · Fine-tuning vs RAG vs prompting 🟢

**Q:** For each need below, which technique do you use first: new factual knowledge, strict output format, domain tone, lower cost at high volume?

??? success "Answer"
    | Need | First choice | Why |
    |---|---|---|
    | New / changing factual knowledge | **RAG** | Updatable, citable, access-controlled |
    | Strict output format | **Structured outputs / prompting**, then SFT | Constrained decoding is cheaper than training |
    | Domain tone, style, behaviour | **Few-shot prompting**, then SFT/LoRA | Fine-tuning shapes behaviour well |
    | Cost/latency at high volume | **Distil / fine-tune a smaller model** | Small model replaces a large one on a narrow task |
    - Evidence: RAG consistently outperformed unsupervised fine-tuning for knowledge injection (Ovadia et al.); LIMA aligned a 65B model with only 1,000 curated examples (behaviour, not knowledge).

    **Sources:** [Ovadia et al., arXiv:2312.05934](https://arxiv.org/abs/2312.05934) · [LIMA, arXiv:2305.11206](https://arxiv.org/abs/2305.11206) · [OpenAI — Fine-tuning best practices](https://platform.openai.com/docs/guides/fine-tuning-best-practices) · [Fine tuning overview (this handbook)](../GenAI%20Engineering/Chapter%203%20-%20Fine%20Tuning%20techniques/Fine%20tuning%20overview.md)

### FT-02 · Memory for full fine-tuning 🟠

**Q:** How many bytes per parameter does mixed-precision full fine-tuning with Adam need? Apply it to a 7B model.

??? success "Answer"
    - **16 bytes/parameter**: fp16/bf16 weights (2) + fp16/bf16 gradients (2) + fp32 master weights (4) + Adam first moment (4) + second moment (4).
    - 7B × 16 B = **112 GB**, plus activations (depend on batch × sequence length × layers × hidden size).
    - So a 7B full fine-tune does not fit on one 80 GB GPU without sharding (ZeRO/FSDP), 8-bit optimisers or offloading.

    **Sources:** [ZeRO, arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · [Fine tuning overview — memory (this handbook)](../GenAI%20Engineering/Chapter%203%20-%20Fine%20Tuning%20techniques/Fine%20tuning%20overview.md)

### FT-03 · ZeRO stages and FSDP 🟠

**Q:** What do ZeRO stages 1, 2 and 3 shard, and how does FSDP relate?

??? success "Answer"
    - **Stage 1:** shards optimizer states → up to 4× memory reduction.
    - **Stage 2:** + gradients → up to 8×.
    - **Stage 3:** + parameters → memory reduction linear in the number of GPUs; parameters are all-gathered layer by layer for forward/backward (more communication).
    - **ZeRO-Offload / Infinity:** move states to CPU RAM / NVMe.
    - **PyTorch FSDP** (and FSDP2) implements the stage-3 idea natively; `SHARD_GRAD_OP` ≈ stage 2.

    **Sources:** [ZeRO, arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · [PyTorch FSDP, arXiv:2304.11277](https://arxiv.org/abs/2304.11277) · [Ultra-Scale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

### FT-04 · LoRA mechanics 🟠

**Q:** Write the LoRA update, explain the initialisation, and compute the trainable parameters for a 4096×4096 matrix with r=16.

??? success "Answer"
    - `h = W₀x + (α/r)·B·A·x`, with `W₀` frozen, `A ∈ ℝ^(r×k)`, `B ∈ ℝ^(d×r)`, r ≪ min(d, k).
    - Init: `A` random Gaussian, `B` **zeros** → ΔW = 0 at step 0, so training starts exactly from the base model.
    - Parameters: r·(d + k) = 16 × 8192 = **131,072** vs 16.8M for the full matrix (**0.78%**).
    - After training, `W = W₀ + (α/r)BA` can be **merged**: no extra inference latency.
    - Evidence: on GPT-3 175B, 10,000× fewer trainable parameters and 3× less GPU memory than full fine-tuning, with on-par quality.

    **Sources:** [LoRA, arXiv:2106.09685](https://arxiv.org/abs/2106.09685) · [Supervised Fine Tuning (this handbook)](../GenAI%20Engineering/Chapter%203%20-%20Fine%20Tuning%20techniques/Supervised_fine_tuning.md)

### FT-05 · QLoRA 🟠

**Q:** What are the three techniques introduced by QLoRA, and what did it achieve?

??? success "Answer"
    1. **4-bit NormalFloat (NF4)**: quantisation levels matched to normally distributed weights.
    2. **Double quantisation**: quantise the quantisation constants too, saving about 0.37 bits/parameter (about 3 GB on a 65B model).
    3. **Paged optimisers**: NVIDIA unified memory pages optimizer states to CPU during memory spikes.
    - Base weights stored in NF4, de-quantised to bf16 for compute; LoRA adapters trained in bf16.
    - Result: fine-tuning a **65B model on a single 48 GB GPU**; Guanaco reached 99.3% of ChatGPT's level on the Vicuna benchmark after 24 h.

    **Sources:** [QLoRA, arXiv:2305.14314](https://arxiv.org/abs/2305.14314) · [Quantization (this handbook)](../GenAI%20Engineering/Chapter%203%20-%20Fine%20Tuning%20techniques/Quantization.md)

### FT-06 · LoRA hyperparameters and variants 🟠

**Q:** Give sensible starting values for LoRA rank, alpha, target modules and learning rate, and name two variants that improve it.

??? success "Answer"
    - **Rank** 8–64 (higher for harder/larger shifts); **alpha** = r or 2r; **dropout** 0–0.1.
    - **Target modules:** all linear layers (q, k, v, o, gate, up, down) usually beat attention-only at equal budget.
    - **Learning rate** about 1e-4 to 2e-4 (higher than full fine-tuning), 1–3 epochs, cosine schedule with warm-up.
    - **rsLoRA:** scaling α/√r instead of α/r, stable at high ranks.
    - **DoRA:** decomposes weights into magnitude and direction, LoRA on the direction; closer to full fine-tuning quality.
    - Trade-off: LoRA learns less on hard targets (code, maths) than full fine-tuning but **forgets less** of the base model.

    **Sources:** [rsLoRA, arXiv:2312.03732](https://arxiv.org/abs/2312.03732) · [DoRA, arXiv:2402.09353](https://arxiv.org/abs/2402.09353) · [LoRA Learns Less and Forgets Less, arXiv:2405.09673](https://arxiv.org/abs/2405.09673) · [Practical Tips for Finetuning LLMs Using LoRA (Raschka)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

### FT-07 · Chat templates and loss masking 🟢

**Q:** Why must you apply the model's chat template and mask the prompt tokens in SFT?

??? success "Answer"
    - **Chat template:** the model was trained with specific special tokens and role markers; a different format at training or inference degrades output and can break stopping (missing EOS → endless generation). Use `tokenizer.apply_chat_template`.
    - **Loss masking:** set labels of system/user tokens to `-100` so loss is computed only on assistant tokens; otherwise the model learns to generate user prompts and the gradient signal is diluted.
    - In TRL `SFTTrainer`: `completion_only_loss` (prompt–completion data) or `assistant_only_loss` (conversational data).

    **Sources:** [Hugging Face — Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating) · [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

### FT-08 · Activation memory techniques 🟠

**Q:** Explain gradient checkpointing, gradient accumulation and sequence packing, with their costs.

??? success "Answer"
    - **Gradient checkpointing:** store activations only at checkpoints and recompute the rest in backward → activation memory about O(√n) in layers, cost about one extra forward pass (about 20–33% more compute).
    - **Gradient accumulation:** effective batch = micro-batch × accumulation steps × number of GPUs; same maths as a large batch, more steps of wall time.
    - **Sequence packing:** concatenate short examples to fill the context, removing padding; requires attention isolation between examples (position IDs / FlashAttention varlen), otherwise examples attend to each other. Hugging Face measured up to **2× training throughput** with padding-free packing.

    **Sources:** [Chen et al., Sublinear Memory Cost, arXiv:1604.06174](https://arxiv.org/abs/1604.06174) · [Packing with Flash Attention 2 (Hugging Face blog)](https://huggingface.co/blog/packing-with-FA2) · [Training Optimization techniques (this handbook)](../GenAI%20Engineering/Chapter%203%20-%20Fine%20Tuning%20techniques/Training%20Optimization%20techniques.md)

### FT-09 · DPO loss 🔴

**Q:** Write the DPO objective and explain the role of β and the reference model.

??? success "Answer"
    - `L = −E[ log σ( β·[ log πθ(y_w|x)/π_ref(y_w|x) − log πθ(y_l|x)/π_ref(y_l|x) ] ) ]`, with `y_w` preferred and `y_l` rejected.
    - Derived from the KL-constrained RLHF objective: the policy is its own implicit reward model → **no reward model, no sampling during training**.
    - **π_ref** (usually the SFT model) anchors the policy; **β** controls how far it may move (higher β = stay closer).
    - Variants: **IPO** (less overfitting), **KTO** (binary good/bad labels, no pairs), **ORPO** (SFT + preference in one stage, no reference model).

    **Sources:** [DPO, arXiv:2305.18290](https://arxiv.org/abs/2305.18290) · [KTO, arXiv:2402.01306](https://arxiv.org/abs/2402.01306) · [ORPO, arXiv:2403.07691](https://arxiv.org/abs/2403.07691) · [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)

### FT-10 · PPO vs GRPO 🔴

**Q:** How does GRPO differ from PPO, and when is it the right choice?

??? success "Answer"
    - **PPO-RLHF:** policy + reference + **reward model** + **value (critic) model**; advantage from the critic; four models in memory.
    - **GRPO:** sample a group of G completions per prompt, advantage `Aᵢ = (rᵢ − mean(r)) / std(r)`; **no critic**, KL penalty to the reference model; much lower memory.
    - Best with **verifiable rewards** (RLVR): unit tests, maths answers, schema validity, exact match.
    - Evidence: DeepSeek-R1-Zero (RL only, no SFT) raised AIME 2024 pass@1 from **15.6% to 71.0%**. InstructGPT earlier showed RLHF made a 1.3B model preferred to the 175B GPT-3.
    - KL penalty prevents **reward hacking** (exploiting reward model flaws) and language drift.

    **Sources:** [DeepSeekMath (GRPO), arXiv:2402.03300](https://arxiv.org/abs/2402.03300) · [DeepSeek-R1, arXiv:2501.12948](https://arxiv.org/abs/2501.12948) · [InstructGPT, arXiv:2203.02155](https://arxiv.org/abs/2203.02155) · [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)

### FT-11 · Distillation 🟠

**Q:** What are the main distillation methods and a quantified example of each?

??? success "Answer"
    - **Logit (soft-label) distillation:** student minimises KL to the teacher's temperature-softened distribution (Hinton). Example: **DistilBERT** is 40% smaller, 60% faster and keeps 97% of BERT's language understanding.
    - **Sequence-level / synthetic data distillation:** student fine-tuned on teacher-generated outputs (needs only API access). Example: **DeepSeek-R1-Distill** models (SFT on 800k R1 samples); R1-Distill-Qwen-32B beat o1-mini on several reasoning benchmarks.
    - **On-policy distillation:** student generates, teacher scores/corrects its tokens; reduces train/test mismatch.
    - Check the teacher's licence/terms before distilling from a commercial model.

    **Sources:** [Hinton et al., arXiv:1503.02531](https://arxiv.org/abs/1503.02531) · [DistilBERT, arXiv:1910.01108](https://arxiv.org/abs/1910.01108) · [DeepSeek-R1, arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

### FT-12 · Dataset size and quality 🟢

**Q:** How many examples do you need for SFT, and which data-quality steps are mandatory?

??? success "Answer"
    - Size: improvements typically visible from **50–100 well-crafted examples** for API fine-tuning; a few thousand for robust task models; LIMA used 1,000 curated examples for general alignment.
    - Mandatory steps:
        - **Deduplication** (exact + near-duplicate with MinHash); deduplicated training data made models emit memorised text **10× less often**.
        - **Decontamination** against eval sets (n-gram overlap, e.g. 13-grams as in GPT-3).
        - Coverage of edge cases and refusals; consistent format; PII scrubbing.
    - Quality beats quantity; review a sample manually.

    **Sources:** [OpenAI — Fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning) · [Lee et al., Deduplicating Training Data, arXiv:2107.06499](https://arxiv.org/abs/2107.06499) · [GPT-3 paper (decontamination), arXiv:2005.14165](https://arxiv.org/abs/2005.14165) · [Data for LLM (this handbook)](../GenAI%20Engineering/Chapter%207%20-%20LLMOps/Data%20for%20LLM.md)

### FT-13 · Catastrophic forgetting 🟠

**Q:** How do you detect and limit catastrophic forgetting after fine-tuning?

??? success "Answer"
    - **Detect:** run general benchmarks (e.g. MMLU, IFEval, safety sets) and previous task evals before and after; compare to the base model.
    - **Limit:**
        - parameter-efficient methods (LoRA forgets less than full FT),
        - **replay**: mix 5–20% general instruction data into the training set,
        - lower learning rate and fewer epochs, early stopping on held-out loss,
        - model merging (e.g. averaging fine-tuned and base weights).

    **Sources:** [LoRA Learns Less and Forgets Less, arXiv:2405.09673](https://arxiv.org/abs/2405.09673) · [Luo et al., forgetting in continual fine-tuning, arXiv:2308.08747](https://arxiv.org/abs/2308.08747) · [Evaluation in LLM training and fine-tuning (this handbook)](../GenAI%20Engineering/Chapter%205%20-%20Evaluation/Evaluation%20in%20LLM%20training%20and%20fine-tuning.md)

### FT-14 · Small models and encoders 🟢

**Q:** Give three tasks where a small model or an encoder beats a frontier LLM in production, with an example model.

??? success "Answer"
    - **Zero-shot NER / PII detection:** GLiNER (bidirectional encoder, CPU-friendly) reported outperforming ChatGPT on zero-shot NER benchmarks.
    - **Classification / routing / intent detection:** fine-tuned ModernBERT or DeBERTa; milliseconds on CPU, 8,192-token context for ModernBERT.
    - **Reranking / embeddings:** cross-encoders and embedding models of about 100M–600M parameters.
    - **Narrow generation on device / at high volume:** Phi-3-mini (3.8B) reports MMLU about 69%, comparable to Mixtral 8x7B and GPT-3.5.
    - Pattern: **cascade**: small model first, escalate to a large model when confidence is low.

    **Sources:** [GLiNER, arXiv:2311.08526](https://arxiv.org/abs/2311.08526) · [ModernBERT, arXiv:2412.13663](https://arxiv.org/abs/2412.13663) · [Phi-3, arXiv:2404.14219](https://arxiv.org/abs/2404.14219) · [Model Choice (this handbook)](../GenAI%20Engineering/Chapter%203%20-%20Fine%20Tuning%20techniques/Model_choice.md)

## 4. LLM serving and optimisation

### SV-01 · Prefill vs decode bottlenecks 🟢

**Q:** Why is prefill compute-bound and decode memory-bandwidth-bound?

??? success "Answer"
    - A matmul with weights does about 2 FLOPs per parameter per token. Weights (2 bytes/parameter in bf16) must be read from HBM once per forward pass.
    - **Prefill:** all N prompt tokens processed in one pass → about N FLOPs per byte read → high arithmetic intensity → **compute-bound** → drives TTFT.
    - **Decode:** one token per sequence per pass → about 1 FLOP per byte at batch 1 → **memory-bound** → drives inter-token latency.
    - H100 ridge point ≈ 989 TFLOPS (bf16 dense) / 3.35 TB/s ≈ **295 FLOPs/byte**: decode needs a batch in the hundreds before compute saturates (less in practice because KV cache reads grow too).

    **Sources:** [How to Scale Your Model — Inference (Google DeepMind)](https://jax-ml.github.io/scaling-book/inference/) · [LLM Inference Unveiled (roofline), arXiv:2402.16363](https://arxiv.org/abs/2402.16363) · [Inference Optimizations (this handbook)](../GenAI%20Engineering/Chapter%203%20-%20Fine%20Tuning%20techniques/Inference%20Optimizations.md)

### SV-02 · Decode speed upper bound 🟢

**Q:** What is the maximum single-stream decode speed of a 7B bf16 model on an H100 SXM, and on an A100 80GB?

??? success "Answer"
    - tokens/s ≲ memory bandwidth / bytes read per token ≈ bandwidth / model size.
    - Model: 7B × 2 bytes = 14 GB.
    - **H100 SXM** (3.35 TB/s): ≈ **240 tokens/s**. **A100 80GB SXM** (about 2.0 TB/s): ≈ **145 tokens/s**.
    - Real numbers are lower (KV cache reads, kernel efficiency of 60–80%). Quantising to 4-bit roughly quadruples the bound.

    **Sources:** [NVIDIA H100 datasheet](https://www.nvidia.com/en-us/data-center/h100/) · [NVIDIA A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/) · [How to Scale Your Model — Inference](https://jax-ml.github.io/scaling-book/inference/)

### SV-03 · KV cache size 🟠

**Q:** Give the KV cache formula and compute it for Llama-3-8B at 32k tokens in bf16.

??? success "Answer"
    - Bytes per token = `2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_value`.
    - Llama-3-8B: 32 layers, 8 KV heads, head_dim 128, 2 bytes → 2 × 32 × 8 × 128 × 2 = **131,072 B = 128 KiB/token**.
    - 32,768 tokens → **4 GiB per sequence**. A batch of 16 such sequences needs 64 GiB, more than the 16 GB of weights.
    - Same maths for Llama-2-7B (32 KV heads, no GQA): **512 KiB/token**, 4× more.

    **Sources:** [Llama 3 model config (Hugging Face)](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json) · [vLLM paper, arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

### SV-04 · MHA, MQA, GQA, MLA 🟠

**Q:** Compare multi-head, multi-query, grouped-query and multi-head latent attention in terms of KV cache.

??? success "Answer"
    - **MHA:** one K/V head per query head → largest KV cache.
    - **MQA:** one K/V head shared by all query heads → smallest cache, some quality loss.
    - **GQA:** G groups of query heads share one K/V head (Llama 3 8B: 32 query heads, 8 KV heads → 4× smaller than MHA); quality close to MHA, speed close to MQA.
    - **MLA (DeepSeek):** K/V compressed into a low-rank latent vector cached per token, up-projected at attention time. DeepSeek-V2 reported **93.3% smaller KV cache** and 5.76× max generation throughput vs DeepSeek 67B.

    **Sources:** [MQA, arXiv:1911.02150](https://arxiv.org/abs/1911.02150) · [GQA, arXiv:2305.13245](https://arxiv.org/abs/2305.13245) · [DeepSeek-V2, arXiv:2405.04434](https://arxiv.org/abs/2405.04434)

### SV-05 · Continuous batching 🟢

**Q:** What is continuous (iteration-level) batching and why does it matter?

??? success "Answer"
    - **Static batching:** a batch runs until its longest sequence finishes; finished slots sit idle; new requests wait.
    - **Continuous batching:** the scheduler adds and removes sequences **at every decoding iteration**, keeping the GPU full.
    - Evidence: Orca reported **36.9× throughput** vs FasterTransformer on GPT-3 175B at the same latency; Anyscale measured up to **23×** throughput vs naive batching.
    - Implemented in vLLM, SGLang, TensorRT-LLM, TGI.

    **Sources:** [Orca (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) · [How continuous batching enables 23x throughput (Anyscale)](https://www.anyscale.com/blog/continuous-batching-llm-inference)

### SV-06 · PagedAttention 🟠

**Q:** How does PagedAttention work and what problem does it solve?

??? success "Answer"
    - Problem: pre-allocating contiguous KV memory for the maximum length wastes memory (internal/external fragmentation); existing systems used only **20–40%** of KV memory for actual tokens.
    - Solution: KV cache split into fixed-size **blocks** (16 tokens by default in vLLM) allocated on demand; a per-sequence **block table** maps logical to physical blocks (like OS virtual memory paging).
    - Enables sharing with **copy-on-write** (parallel sampling, beam search, common prefixes).
    - Result: waste **under 4%**, **2–4× throughput** vs FasterTransformer and Orca at the same latency.

    **Sources:** [vLLM / PagedAttention, arXiv:2309.06180](https://arxiv.org/abs/2309.06180) · [vLLM blog](https://blog.vllm.ai/2023/06/20/vllm.html)

### SV-07 · Prefix caching 🟠

**Q:** How does prefix caching work in vLLM/SGLang, and how do you structure prompts to benefit?

??? success "Answer"
    - KV blocks are hashed by their token content (and preceding tokens); requests sharing a prefix reuse cached KV and **skip that part of prefill**.
    - **SGLang RadixAttention** keeps a radix tree of prefixes with LRU eviction; reported **up to 6.4× throughput** on prefix-heavy workloads.
    - Prompt structure: **static first, dynamic last**: system prompt → tool definitions → few-shot examples → retrieved documents → user question. Any change early in the prompt invalidates everything after it.
    - Same principle for provider prompt caching (SV-14).

    **Sources:** [SGLang, arXiv:2312.07104](https://arxiv.org/abs/2312.07104) · [vLLM — Automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)

### SV-08 · Chunked prefill 🟠

**Q:** What problem does chunked prefill solve?

??? success "Answer"
    - A long prompt's prefill monopolises the GPU for one iteration, so all running decodes stall → **inter-token latency spikes** for other users.
    - Chunked prefill splits the prompt into chunks (token budget per iteration) and batches them **together with decode tokens** ("stall-free" scheduling).
    - Sarathi-Serve reported **2.6–3.7× higher serving capacity** under latency SLOs. Enabled by default in recent vLLM versions.

    **Sources:** [Sarathi-Serve, arXiv:2403.02310](https://arxiv.org/abs/2403.02310) · [vLLM — Optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization.html)

### SV-09 · Speculative decoding 🟠

**Q:** Explain speculative decoding, why output quality is unchanged, and when it does not help.

??? success "Answer"
    - A cheap **draft** proposes k tokens (small model, n-gram/prompt lookup, EAGLE or Medusa heads); the target model scores all k in **one forward pass**.
    - Each draft token is accepted with probability min(1, p_target/p_draft); on rejection, resample from the adjusted distribution → output distribution **identical** to the target model.
    - Speedup depends on the **acceptance rate** and draft cost: large on predictable text (code, extraction, RAG answers copying context).
    - Little or no gain at **large batch sizes** (GPU already compute-saturated) or with high-temperature creative sampling.
    - Evidence: 2–3× on T5-XXL (Leviathan et al.); 2–2.5× on Chinchilla 70B (Chen et al.).

    **Sources:** [Leviathan et al., arXiv:2211.17192](https://arxiv.org/abs/2211.17192) · [Chen et al., arXiv:2302.01318](https://arxiv.org/abs/2302.01318) · [EAGLE, arXiv:2401.15077](https://arxiv.org/abs/2401.15077)

### SV-10 · FlashAttention 🟠

**Q:** What does FlashAttention change, and is it an approximation?

??? success "Answer"
    - Standard attention materialises the N×N score matrix in HBM → O(N²) memory traffic.
    - FlashAttention **tiles** Q, K, V into SRAM and uses an **online softmax**, never writing the N×N matrix to HBM; backward recomputes instead of storing.
    - It is **exact** (not an approximation); memory becomes linear in sequence length.
    - Versions: FA1 (up to 3× on GPT-2), FA2 (about 2× over FA1, better parallelism), FA3 for Hopper (async + FP8; up to about 740 TFLOPS in FP16, 75% of H100 peak).

    **Sources:** [FlashAttention, arXiv:2205.14135](https://arxiv.org/abs/2205.14135) · [FlashAttention-2, arXiv:2307.08691](https://arxiv.org/abs/2307.08691) · [FlashAttention-3, arXiv:2407.08608](https://arxiv.org/abs/2407.08608) · [GPU Optimization (this handbook)](../GenAI%20Engineering/Chapter%201%20-%20Deep%20Learning/GPU%20Optimization.md)

### SV-11 · Quantisation methods 🟠

**Q:** Compare GPTQ, AWQ, SmoothQuant, FP8 and GGUF.

??? success "Answer"
    | Method | What is quantised | Idea | Typical use |
    |---|---|---|---|
    | **GPTQ** | Weights (3–4 bit) | Layer-wise second-order (Hessian) error compensation | GPU serving, 4-bit |
    | **AWQ** | Weights (4 bit) | Protect ~1% salient channels chosen by activation magnitude | GPU serving, robust 4-bit |
    | **SmoothQuant** | Weights + activations (W8A8) | Move activation outliers into weights by scaling | INT8 throughput |
    | **FP8** (E4M3) | Weights + activations (+ KV) | Hardware-native on Hopper/Blackwell | Default near-lossless on recent GPUs |
    | **GGUF** (k-quants) | Weights (2–8 bit) | Block-wise quantisation for llama.cpp | CPU / Apple / local |
    - Evidence: GPTQ 175B at 3–4 bits, 3.25× speedup on A100; LLM.int8() showed outlier features from about 6.7B parameters break naive int8.
    - Always validate on **task evals** (reasoning, long context, multilingual), not only perplexity.

    **Sources:** [GPTQ, arXiv:2210.17323](https://arxiv.org/abs/2210.17323) · [AWQ, arXiv:2306.00978](https://arxiv.org/abs/2306.00978) · [SmoothQuant, arXiv:2211.10438](https://arxiv.org/abs/2211.10438) · [LLM.int8(), arXiv:2208.07339](https://arxiv.org/abs/2208.07339) · [Quantization (this handbook)](../GenAI%20Engineering/Chapter%203%20-%20Fine%20Tuning%20techniques/Quantization.md)

### SV-12 · Parallelism for inference 🔴

**Q:** Compare tensor, pipeline, expert and data parallelism for serving.

??? success "Answer"
    - **Tensor parallelism (TP):** splits each layer's matrices across GPUs (column- then row-parallel); 2 all-reduces per transformer layer → needs **NVLink**, keep within one node; reduces latency and per-GPU memory.
    - **Pipeline parallelism (PP):** splits layers across GPUs/nodes; only activations passed between stages; pipeline bubbles; used across nodes for very large models.
    - **Expert parallelism (EP):** MoE experts placed on different GPUs; all-to-all token routing.
    - **Data parallelism (replicas):** independent copies behind a load balancer → scales **throughput**, not single-request latency.
    - Typical: TP within a node to fit the model / meet latency, replicas for throughput.

    **Sources:** [Megatron-LM, arXiv:1909.08053](https://arxiv.org/abs/1909.08053) · [Ultra-Scale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) · [vLLM — Parallelism and scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html)

### SV-13 · Latency metrics 🟢

**Q:** Define TTFT, TPOT/ITL, end-to-end latency, throughput and goodput.

??? success "Answer"
    - **TTFT:** request arrival → first output token (queueing + prefill).
    - **TPOT / ITL:** average time between output tokens (decode).
    - **E2E latency** = TTFT + TPOT × (output tokens − 1).
    - **Throughput:** output (or total) tokens/s, or requests/s, across all users.
    - **Goodput:** requests/s that **meet the SLO** (e.g. TTFT < 1 s and TPOT < 50 ms); the metric to optimise.
    - Report percentiles (p50/p95/p99), not averages.

    **Sources:** [LLM inference benchmarking — fundamental concepts (NVIDIA)](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/) · [DistServe, arXiv:2401.09670](https://arxiv.org/abs/2401.09670)

### SV-14 · Prefill/decode disaggregation 🔴

**Q:** What is disaggregated serving and what does it buy?

??? success "Answer"
    - Prefill and decode run on **separate GPU pools**; KV cache is transferred from prefill to decode workers (NVLink/RDMA).
    - Each phase is scaled and parallelised independently (prefill: compute-optimised; decode: memory/batch-optimised); no prefill/decode interference.
    - Evidence: DistServe up to **7.4× more requests** or **12.6× tighter SLOs**; Splitwise (Microsoft) reported higher throughput at lower cost/power.
    - Implementations: NVIDIA Dynamo, llm-d, vLLM and SGLang disaggregated modes. Worth it at large scale with long prompts; overkill for a few GPUs.

    **Sources:** [DistServe, arXiv:2401.09670](https://arxiv.org/abs/2401.09670) · [Splitwise, arXiv:2311.18677](https://arxiv.org/abs/2311.18677) · [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) · [llm-d](https://llm-d.ai/)

### SV-15 · Provider prompt caching and batch pricing 🟢

**Q:** How do provider prompt caching and batch APIs reduce cost, with numbers?

??? success "Answer"
    - **Anthropic:** explicit `cache_control` breakpoints; cache writes cost 1.25× base input (5-min TTL) or 2× (1-hour TTL), cache reads **0.1×**; Anthropic reports up to 90% cost and 85% latency reduction on long prompts.
    - **OpenAI:** automatic caching for prompts ≥ 1,024 tokens with discounted cached input tokens.
    - **Batch APIs** (OpenAI, Anthropic, others): about **50% discount**, results within 24 h.
    - Requirement: stable, byte-identical prefix (no timestamps or request IDs at the top of the prompt).

    **Sources:** [Anthropic — Prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching) · [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) · [Anthropic — Batch processing](https://docs.claude.com/en/docs/build-with-claude/batch-processing)

### SV-16 · Serving engines 🟢

**Q:** Name the main LLM serving engines and when you would pick each.

??? success "Answer"
    - **vLLM:** de-facto open-source default; PagedAttention, broad model/hardware support, OpenAI-compatible server, LoRA multi-adapter.
    - **SGLang:** RadixAttention, fast structured outputs, strong for prefix-heavy and agentic workloads.
    - **TensorRT-LLM** (with Triton / Dynamo): best raw NVIDIA performance, more build complexity.
    - **llama.cpp / Ollama:** CPU, Apple Silicon, local and edge, GGUF.
    - **Managed:** Bedrock, Vertex AI, Azure AI Foundry, or dedicated endpoints when you do not want to run GPUs.

    **Sources:** [vLLM](https://docs.vllm.ai/) · [SGLang](https://docs.sglang.ai/) · [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [Serving Frameworks (this handbook)](../GenAI%20Engineering/Chapter%207%20-%20LLMOps/Serving%20Frameworks.md)

### SV-17 · LLM serving on Kubernetes 🔴

**Q:** What is specific about running LLM inference on Kubernetes?

??? success "Answer"
    - **GPU scheduling:** NVIDIA GPU Operator / device plugin, node pools per GPU type, MIG or time-slicing for small models.
    - **Cold starts:** weights of tens to hundreds of GB → pre-pull images, cache weights on local NVMe or shared volumes, keep warm replicas.
    - **Autoscaling** on queue depth, pending requests or KV cache usage (KEDA / custom metrics), **not CPU**.
    - **Routing:** KV-cache- and prefix-aware load balancing (Gateway API Inference Extension, llm-d, vLLM production-stack) instead of round-robin.
    - Model serving layers: KServe, vLLM production-stack, NVIDIA Dynamo.

    **Sources:** [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) · [vLLM production-stack](https://github.com/vllm-project/production-stack) · [KServe](https://kserve.github.io/website/) · [llm-d](https://llm-d.ai/)

## 5. LLM architecture

### AR-01 · Modern decoder block 🟢

**Q:** List the components of a modern decoder-only transformer block (Llama/Qwen/Mistral family) and why each replaced the original.

??? success "Answer"
    - **Pre-norm with RMSNorm** (instead of post-LayerNorm): more stable training, cheaper (no mean subtraction).
    - **Causal self-attention with RoPE** (instead of absolute position embeddings): relative position, better extrapolation.
    - **GQA** (instead of MHA): smaller KV cache (SV-04).
    - **SwiGLU MLP** (instead of ReLU/GELU FFN): better quality at equal compute.
    - Residual connections around attention and MLP; no biases in most linear layers; tied or untied embeddings.

    **Sources:** [Llama 2, arXiv:2307.09288](https://arxiv.org/abs/2307.09288) · [The Big LLM Architecture Comparison (Raschka)](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) · [GLU variants (Shazeer), arXiv:2002.05202](https://arxiv.org/abs/2002.05202) · [LLM Architecture (this handbook)](../GenAI%20Engineering/Chapter%202%20-%20LLM%20Pipeline/LLM%20Architecture.md)

### AR-02 · Scaled dot-product attention 🟢

**Q:** Write the attention formula. Why divide by √d_k? What is the complexity?

??? success "Answer"
    - `Attention(Q, K, V) = softmax(QKᵀ / √d_k + M) · V`, with M the causal mask (−∞ above the diagonal).
    - For components with unit variance, the dot product q·k has variance d_k; dividing by √d_k keeps it about 1 so softmax does not saturate (vanishing gradients).
    - Complexity: O(n² · d) compute and O(n²) score memory for sequence length n (memory linear with FlashAttention); with KV cache, each decode step is O(n · d).

    **Sources:** [Attention Is All You Need, arXiv:1706.03762](https://arxiv.org/abs/1706.03762) · [Transformers (this handbook)](../GenAI%20Engineering/Chapter%202%20-%20LLM%20Pipeline/Transformers.md)

### AR-03 · Encoder, decoder, encoder-decoder 🟢

**Q:** When do you use an encoder-only, decoder-only or encoder-decoder model?

??? success "Answer"
    - **Encoder-only** (BERT, ModernBERT): bidirectional attention, masked-LM pre-training → classification, NER, embeddings, reranking.
    - **Decoder-only** (GPT, Llama, Mistral): causal attention, next-token prediction → generation, chat, reasoning, tool use; dominant for general-purpose LLMs because one objective scales to everything.
    - **Encoder-decoder** (T5, BART, Whisper): encoder reads input, decoder generates with cross-attention → translation, summarisation, speech-to-text.

    **Sources:** [BERT, arXiv:1810.04805](https://arxiv.org/abs/1810.04805) · [T5, arXiv:1910.10683](https://arxiv.org/abs/1910.10683) · [Encoding and Decoding (this handbook)](../GenAI%20Engineering/Chapter%202%20-%20LLM%20Pipeline/Encoding%20and%20Decoding.md)

### AR-04 · RoPE and context extension 🟠

**Q:** How does RoPE encode position, and how are context windows extended after pre-training?

??? success "Answer"
    - RoPE rotates each pair of dimensions of q and k by angle `m·θᵢ`, with m the position and `θᵢ = base^(−2i/d)` (base 10,000; Llama 3 uses 500,000). The dot product q·k then depends only on the **relative** position.
    - Extension methods:
        - **Position Interpolation:** scale positions down to fit the trained range.
        - **NTK-aware scaling:** increase the base, preserving high-frequency dimensions.
        - **YaRN:** per-frequency interpolation + attention temperature; reported needing 10× fewer tokens and 2.5× fewer steps than prior methods.
    - All require some fine-tuning on long sequences for best results.

    **Sources:** [RoFormer, arXiv:2104.09864](https://arxiv.org/abs/2104.09864) · [Position Interpolation, arXiv:2306.15595](https://arxiv.org/abs/2306.15595) · [YaRN, arXiv:2309.00071](https://arxiv.org/abs/2309.00071) · [Positional embeddings repo](https://github.com/harrisonpim/positional-embeddings)

### AR-05 · Tokenisation 🟢

**Q:** How does BPE work, and what are the practical consequences of vocabulary size?

??? success "Answer"
    - **BPE:** start from bytes/characters, repeatedly merge the most frequent adjacent pair into a new token until the target vocabulary size. Byte-level BPE has no out-of-vocabulary tokens.
    - Larger vocabulary → fewer tokens per text (cheaper, longer effective context) but larger embedding/LM-head matrices. Llama 3 moved from 32K to **128K** tokens and reported up to **15% fewer tokens** than Llama 2.
    - Tokenisers are uneven across languages: the same text can need **up to 15× more tokens** in some languages → cost, latency and context differences.
    - Tokenisation explains classic failures (counting letters, arithmetic on numbers split oddly).

    **Sources:** [Sennrich et al. (BPE), arXiv:1508.07909](https://arxiv.org/abs/1508.07909) · [Introducing Llama 3 (Meta)](https://ai.meta.com/blog/meta-llama-3/) · [Petrov et al., arXiv:2305.15425](https://arxiv.org/abs/2305.15425) · [Tokenizers (this handbook)](../GenAI%20Engineering/Chapter%202%20-%20LLM%20Pipeline/Tokenizers.md)

### AR-06 · Sampling parameters 🟢

**Q:** Define temperature, top-k, top-p and min-p.

??? success "Answer"
    - **Temperature T:** logits divided by T before softmax; T < 1 sharpens, T > 1 flattens; T → 0 ≈ greedy.
    - **Top-k:** keep the k most probable tokens.
    - **Top-p (nucleus):** keep the smallest set of tokens whose cumulative probability ≥ p; adapts to the distribution shape.
    - **Min-p:** keep tokens with probability ≥ min_p × p_max; scales with the model's confidence, stays coherent at higher temperatures.
    - Order in most engines: penalties → temperature → truncation (top-k/top-p/min-p) → sample.

    **Sources:** [Holtzman et al. (nucleus sampling), arXiv:1904.09751](https://arxiv.org/abs/1904.09751) · [Min-p sampling, arXiv:2407.01082](https://arxiv.org/abs/2407.01082)

### AR-07 · Mixture of Experts 🟠

**Q:** How does a MoE layer work, what problem does load balancing solve, and what does MoE change for memory vs compute?

??? success "Answer"
    - The MLP is replaced by E experts plus a **router** selecting the top-k experts per token (Switch: k=1; Mixtral: 2 of 8; DeepSeek-V3: 8 of 256 routed + 1 shared).
    - **Load balancing:** without it, the router collapses onto a few experts. Fixes: auxiliary balancing loss (Switch), or auxiliary-loss-free bias adjustment (DeepSeek-V3); capacity factors.
    - **Compute** scales with active parameters; **memory** scales with total parameters.
    - Numbers: Mixtral 8x7B: 46.7B total / 12.9B active, matching or beating Llama 2 70B on most benchmarks; DeepSeek-V3: 671B total / 37B active.

    **Sources:** [Switch Transformers, arXiv:2101.03961](https://arxiv.org/abs/2101.03961) · [Mixtral, arXiv:2401.04088](https://arxiv.org/abs/2401.04088) · [DeepSeek-V3, arXiv:2412.19437](https://arxiv.org/abs/2412.19437) · [A Visual Guide to MoE (Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)

### AR-08 · Scaling laws 🟠

**Q:** What does Chinchilla say, and why are small models now trained far beyond it?

??? success "Answer"
    - **Chinchilla:** for a fixed training compute budget, parameters and tokens should scale equally, about **20 tokens per parameter**. Chinchilla (70B, 1.4T tokens) outperformed Gopher (280B) at the same compute.
    - It optimises **training** compute only. In production, **inference** cost dominates, so a smaller model trained on many more tokens is better overall: Llama 3 8B used about 15T tokens (about 1,900 tokens/parameter).
    - Third axis: **test-time compute** (AR-12).

    **Sources:** [Chinchilla, arXiv:2203.15556](https://arxiv.org/abs/2203.15556) · [Beyond Chinchilla-Optimal (inference-aware), arXiv:2401.00448](https://arxiv.org/abs/2401.00448) · [Introducing Llama 3 (Meta)](https://ai.meta.com/blog/meta-llama-3/)

### AR-09 · Long-context limits 🟠

**Q:** What does the evidence say about using very long context windows?

??? success "Answer"
    - **Lost in the Middle:** accuracy is U-shaped, best when relevant information is at the beginning or end, worst in the middle.
    - **RULER:** among models claiming ≥ 32K context, only about half kept satisfactory performance at 32K on tasks harder than needle retrieval.
    - **Context rot** (Chroma, 2025): performance degrades as input length grows, even on simple tasks.
    - Consequences: put key instructions/evidence at the start or end, retrieve and rerank instead of stuffing, test at the lengths you actually use.

    **Sources:** [Lost in the Middle, arXiv:2307.03172](https://arxiv.org/abs/2307.03172) · [RULER, arXiv:2404.06654](https://arxiv.org/abs/2404.06654) · [Context Rot (Chroma)](https://research.trychroma.com/context-rot)

### AR-10 · Numerical formats 🟠

**Q:** Give the bit layout and use of fp16, bf16, FP8 E4M3 and E5M2.

??? success "Answer"
    | Format | Sign-Exponent-Mantissa | Max value | Use |
    |---|---|---|---|
    | fp16 | 1-5-10 | 65,504 | Inference; training needs loss scaling |
    | bf16 | 1-8-7 | ≈ 3.4e38 (as fp32) | Default for training, no loss scaling |
    | FP8 E4M3 | 1-4-3 | 448 | Weights/activations (forward) |
    | FP8 E5M2 | 1-5-2 | 57,344 | Gradients (backward, wider range) |
    - FP8 needs per-tensor or per-block scaling factors; Blackwell adds micro-scaled MXFP8/MXFP4/NVFP4.

    **Sources:** [FP8 Formats for Deep Learning, arXiv:2209.05433](https://arxiv.org/abs/2209.05433) · [OCP Microscaling formats spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) · [Numerical representations (this handbook)](../GenAI%20Engineering/Chapter%201%20-%20Deep%20Learning/Numerical%20representations.md)

### AR-11 · Non-determinism at temperature 0 🔴

**Q:** Why can the same prompt at temperature 0 give different outputs on an inference server?

??? success "Answer"
    - Floating-point addition is **not associative**: (a+b)+c ≠ a+(b+c).
    - Kernels (matmul, attention, RMSNorm reductions) change their reduction strategy with **batch size**, which depends on server load → the same request gets different numerics → a different argmax eventually.
    - Other sources: MoE routing with capacity limits, different GPU types or kernel versions across replicas, silent provider model updates.
    - Evidence: Thinking Machines got **80 unique completions out of 1,000** identical requests; **batch-invariant kernels** made them identical, at a performance cost (about 20% in this handbook's summary).

    **Sources:** [Defeating Nondeterminism in LLM Inference (Thinking Machines)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) · [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) · [LLM Architecture (this handbook)](../GenAI%20Engineering/Chapter%202%20-%20LLM%20Pipeline/LLM%20Architecture.md)

### AR-12 · Reasoning models and test-time compute 🟠

**Q:** What changes with reasoning models, and when are they worth the cost?

??? success "Answer"
    - Trained (mostly with RL on verifiable rewards) to produce a long chain of thought before answering; accuracy improves with more thinking tokens (test-time compute).
    - Evidence: compute-optimal test-time scaling let a smaller model **outperform a 14× larger model** in FLOPs-matched comparisons on suitable problems (Snell et al.).
    - Worth it: maths, code, multi-step planning, complex extraction with rules. Not worth it: simple classification, retrieval-heavy Q&A, latency-critical paths.
    - Cost/latency: thinking tokens are billed as output; control with reasoning effort / thinking budget parameters.

    **Sources:** [Learning to reason with LLMs (OpenAI)](https://openai.com/index/learning-to-reason-with-llms/) · [Snell et al., arXiv:2408.03314](https://arxiv.org/abs/2408.03314) · [DeepSeek-R1, arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

## 6. AI engineering systems and solutions

### SD-01 · Chunking strategies 🟢

**Q:** List the main chunking strategies and typical parameters.

??? success "Answer"
    1. **Fixed-size** with overlap (e.g. 256–1,024 tokens, 10–20% overlap): simple baseline.
    2. **Recursive** splitting on separators (paragraphs → sentences → words).
    3. **Structure-aware**: by headings, sections, tables, code blocks (Markdown/HTML/PDF layout).
    4. **Semantic**: split where embedding similarity between sentences drops.
    5. **Late chunking**: embed the full document with a long-context model, then pool per chunk.
    6. **Contextual chunks**: prepend an LLM-generated summary of where the chunk sits in the document (Anthropic Contextual Retrieval).
    - Choose by evaluating retrieval recall on your own questions; Chroma's study found strategy choice changes recall by several points.

    **Sources:** [Evaluating Chunking Strategies (Chroma)](https://research.trychroma.com/evaluating-chunking) · [Late Chunking, arXiv:2409.04701](https://arxiv.org/abs/2409.04701) · [Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval) · [Chunking strategies (this handbook)](../GenAI%20Engineering/Chapter%204%20-%20LLM%20based%20solutions/6.3%20Chunking%20strategies.md)

### SD-02 · Hybrid retrieval and rank fusion 🟢

**Q:** What is hybrid retrieval and how do you merge result lists?

??? success "Answer"
    - Combine **sparse lexical** (BM25: exact terms, IDs, rare words) with **dense** embeddings (semantic similarity); optionally **learned sparse** (SPLADE).
    - Merge with **Reciprocal Rank Fusion**: `score(d) = Σᵢ 1 / (k + rankᵢ(d))`, k = 60 typically; uses ranks, so no score normalisation needed. Alternative: weighted sum of normalised scores.
    - Evidence: Anthropic measured a **35%** reduction in retrieval failures with contextual embeddings alone, and **49%** when contextual BM25 was added (hybrid).

    **Sources:** [Cormack et al., RRF (SIGIR 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) · [SPLADE, arXiv:2107.05720](https://arxiv.org/abs/2107.05720) · [Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval) · [RAG Optimization techniques (this handbook)](../GenAI%20Engineering/Chapter%204%20-%20LLM%20based%20solutions/6.1%20RAG%20Optimization%20techniques.md)

### SD-03 · Bi-encoder, cross-encoder, late interaction 🟠

**Q:** Compare bi-encoders, cross-encoders and ColBERT-style late interaction.

??? success "Answer"
    - **Bi-encoder:** query and document embedded separately; documents pre-computed; cosine similarity + ANN index → fast, scalable, lower precision. First-stage retrieval.
    - **Cross-encoder:** query and document concatenated and scored jointly → most accurate, but one forward pass per pair → only for reranking the top 20–100.
    - **Late interaction (ColBERT):** token-level embeddings stored per document; score = Σ over query tokens of max similarity (MaxSim) → near cross-encoder quality at bi-encoder-like speed, larger index.
    - Evidence: adding a reranker took Anthropic's retrieval failure reduction to **67%**.

    **Sources:** [Sentence-BERT, arXiv:1908.10084](https://arxiv.org/abs/1908.10084) · [ColBERT, arXiv:2004.12832](https://arxiv.org/abs/2004.12832) · [Retriever and re-ranker (this handbook)](../GenAI%20Engineering/Chapter%204%20-%20LLM%20based%20solutions/6.5%20Retriever%20and%20re-ranker.md)

### SD-04 · ANN index types 🟠

**Q:** Compare HNSW, IVF, product quantisation and DiskANN, with their key parameters.

??? success "Answer"
    | Index | Principle | Key parameters | Trade-off |
    |---|---|---|---|
    | **HNSW** | Multi-layer proximity graph | `M`, `efConstruction`, `efSearch` | High recall and speed, high RAM, slow builds |
    | **IVF** | k-means clusters, search nearest lists | `nlist`, `nprobe` | Less RAM, recall depends on `nprobe` |
    | **PQ** (often IVF-PQ) | Compress vectors into sub-quantiser codes | sub-vectors `m`, bits | 10–100× compression, recall loss |
    | **DiskANN** | Graph on SSD, compressed vectors in RAM | graph degree, beam width | Billion-scale on one machine |
    - Exact (flat) search is fine below about 100k–1M vectors; pgvector supports HNSW and IVFFlat.

    **Sources:** [HNSW, arXiv:1603.09320](https://arxiv.org/abs/1603.09320) · [FAISS documentation](https://github.com/facebookresearch/faiss/wiki) · [DiskANN (NeurIPS 2019)](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node) · [Vector Similarity Search Methods (this handbook)](../GenAI%20Engineering/Chapter%204%20-%20LLM%20based%20solutions/6.4%20Vector%20Similarity%20Search%20Methods.md)

### SD-05 · Choosing an embedding model 🟢

**Q:** Which criteria do you use to choose an embedding model?

??? success "Answer"
    - Retrieval quality on **your** data (build a small labelled set: recall@k, nDCG@10); MTEB leaderboard only as a pre-filter (contamination, domain mismatch).
    - Languages covered, max input tokens, **dimension** and storage cost; **Matryoshka** models allow truncating dimensions (up to 14× smaller at equal accuracy in the original paper).
    - Latency/cost, hosting (API vs self-hosted), licence, data residency.
    - Changing model = **re-embedding the whole corpus**: version the index with the model.

    **Sources:** [MTEB, arXiv:2210.07316](https://arxiv.org/abs/2210.07316) · [Matryoshka Representation Learning, arXiv:2205.13147](https://arxiv.org/abs/2205.13147) · [Overview of embedding model (this handbook)](../GenAI%20Engineering/Chapter%202%20-%20LLM%20Pipeline/Overview%20of%20embedding%20model.md)

### SD-06 · Query transformation 🟠

**Q:** Name four query transformation techniques for RAG and when to use each.

??? success "Answer"
    - **Rewriting / condensation:** turn a follow-up in a conversation into a standalone query.
    - **Multi-query:** generate several paraphrases, retrieve for each, fuse with RRF → better recall for ambiguous queries.
    - **Decomposition:** split multi-hop questions into sub-questions answered sequentially.
    - **HyDE:** generate a hypothetical answer and embed it instead of the question → helps when queries and documents are phrased very differently.
    - **Step-back prompting:** ask a more general question first to retrieve background principles.
    - Each adds an LLM call: measure recall gain vs latency.

    **Sources:** [HyDE, arXiv:2212.10496](https://arxiv.org/abs/2212.10496) · [Step-Back Prompting, arXiv:2310.06117](https://arxiv.org/abs/2310.06117) · [RAG Overview (this handbook)](../GenAI%20Engineering/Chapter%204%20-%20LLM%20based%20solutions/6.0%20RAG%20Overview.md)

### SD-07 · Access control in RAG 🟠

**Q:** How do you enforce document permissions in a RAG system?

??? success "Answer"
    - Store ACL metadata (groups, tenant, classification) with each chunk, synchronised from the source system.
    - **Pre-filter at query time** in the vector store using the caller's identity/groups (metadata filter inside the ANN search).
    - Alternatives: one index/namespace per tenant (strong isolation), or post-filtering (risk of empty results and leaks if misapplied).
    - Never rely on the LLM to hide content it received; handle ACL changes and deletions (re-index, tombstones).
    - Listed by OWASP as LLM08 "Vector and Embedding Weaknesses".

    **Sources:** [OWASP Top 10 for LLM Applications 2025 — LLM08](https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/) · [LLMSECOps (this handbook)](../GenAI%20Engineering/Chapter%207%20-%20LLMOps/LLMSECOps.md)

### SD-08 · Document parsing 🟢

**Q:** What are the options to parse complex PDFs (tables, multi-column, scans) for RAG?

??? success "Answer"
    1. **Text extraction** (PyMuPDF, pdfplumber): fast, fine for digital single-column text; loses tables/layout.
    2. **Layout-aware parsers** (Docling, Unstructured, Marker): detect reading order, headings, tables → Markdown/JSON.
    3. **OCR** (Tesseract, PaddleOCR, cloud OCR) for scanned documents.
    4. **Vision-language models** (page images to Markdown, or ColPali-style retrieval directly on page images): best on complex layouts, highest cost.
    - Evaluate on a sample of your documents; tables usually need dedicated handling (keep them whole, add a summary).

    **Sources:** [Docling, arXiv:2408.09869](https://arxiv.org/abs/2408.09869) · [ColPali, arXiv:2407.01449](https://arxiv.org/abs/2407.01449) · [Parser methods (this handbook)](../GenAI%20Engineering/Chapter%204%20-%20LLM%20based%20solutions/6.2%20Parser%20methods.md)

### SD-09 · GraphRAG and knowledge graphs 🟠

**Q:** When does GraphRAG beat vector RAG, and what does it cost?

??? success "Answer"
    - Microsoft GraphRAG: an LLM extracts entities and relations, builds a graph, clusters it into communities (Leiden) and pre-summarises each community.
    - Wins on **global sense-making questions** ("What are the main themes across all claims?"): beat naive RAG on comprehensiveness and diversity in the paper's evaluation.
    - Also useful for multi-hop questions over explicit relations (customer → contract → product).
    - Cost: indexing requires many LLM calls over the whole corpus; updates are expensive. Lighter variants: LightRAG, LazyGraphRAG.
    - Vector RAG remains better and cheaper for local, fact-lookup questions.

    **Sources:** [GraphRAG, arXiv:2404.16130](https://arxiv.org/abs/2404.16130) · [LazyGraphRAG (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) · [LightRAG, arXiv:2410.05779](https://arxiv.org/abs/2410.05779)

### SD-10 · Semantic caching 🟠

**Q:** How does a semantic cache work and what are its risks?

??? success "Answer"
    - Embed the incoming query, search past queries; if similarity ≥ threshold, return the cached answer without calling the LLM.
    - Risks: **false hits** (similar wording, different meaning: "cancel my order" vs "don't cancel my order"), stale answers, leaking answers across users/tenants.
    - Mitigations: tune the threshold on labelled pairs, scope the cache per tenant/permission, TTL and invalidation when sources change, cache only deterministic/FAQ-type flows.
    - Distinct from **exact** caching and provider **prompt** caching (which caches computation, not answers).

    **Sources:** [GPTCache](https://github.com/zilliztech/GPTCache) · [Redis — semantic caching](https://redis.io/docs/latest/develop/ai/redisvl/user_guide/llmcache/)

### SD-11 · Model routing and cascades 🟠

**Q:** What are routing and cascades, and what savings have been measured?

??? success "Answer"
    - **Router:** a classifier picks the cheapest model likely to answer well **before** calling.
    - **Cascade:** call a cheap model first; escalate to a stronger one if a scorer/confidence check fails.
    - Evidence: **FrugalGPT** matched GPT-4 performance with up to **98% cost reduction**; **RouteLLM** reduced cost by over **85% on MT-Bench** while keeping 95% of GPT-4's performance.
    - Needs an eval set per route and monitoring of escalation rates.

    **Sources:** [FrugalGPT, arXiv:2305.05176](https://arxiv.org/abs/2305.05176) · [RouteLLM, arXiv:2406.18665](https://arxiv.org/abs/2406.18665)

### SD-12 · LLM gateway capabilities 🟢

**Q:** What capabilities does an enterprise LLM gateway provide, and which tools exist?

??? success "Answer"
    - Unified API across providers; authentication and **virtual keys** per team; quotas, rate limits, **budgets** and cost attribution; routing, fallback and retries; caching; PII redaction and guardrail hooks; logging/tracing; approved-model catalogue; regional routing for data residency.
    - Tools: **LiteLLM**, **Portkey**, Kong AI Gateway, Envoy AI Gateway, cloud-native (Azure API Management AI gateway, AWS Bedrock, Cloudflare AI Gateway).

    **Sources:** [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) · [Portkey AI Gateway](https://github.com/Portkey-AI/gateway) · [Envoy AI Gateway](https://aigateway.envoyproxy.io/)

### SD-13 · Guardrails 🟢

**Q:** Where do you place guardrails in an LLM application and which tools implement them?

??? success "Answer"
    | Stage | Checks | Example tools |
    |---|---|---|
    | Input | Prompt injection/jailbreak, PII, topic/scope | Llama Prompt Guard, Lakera, Presidio |
    | Retrieval | ACL filtering, untrusted-content tagging | Vector store filters |
    | Output | Toxicity/safety, PII leakage, groundedness, schema | Llama Guard, NeMo Guardrails, Guardrails AI |
    | Tools | Allowlists, argument validation, authorisation | Deterministic code in the tool |
    - Run checks in parallel with generation where possible; prefer small classifiers over LLM calls for latency; track false-positive rate.

    **Sources:** [Llama Guard, arXiv:2312.06674](https://arxiv.org/abs/2312.06674) · [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) · [Guardrails AI](https://www.guardrailsai.com/docs) · [Guardrails (this handbook)](../GenAI%20Engineering/Chapter%207%20-%20LLMOps/Guardrails.md)

### SD-14 · PII detection options 🟢

**Q:** What are the approaches to detect PII in text and their trade-offs?

??? success "Answer"
    1. **Regex + checksums** (IBAN, credit cards with Luhn, emails, phone numbers): precise, cheap, structured identifiers only.
    2. **NER models** (spaCy, transformers, GLiNER-based PII models): names, addresses, organisations; CPU-friendly; zero-shot labels with GLiNER.
    3. **LLM-based detection:** flexible and context-aware, but slower, costlier, and itself a data exposure.
    - Frameworks combining them: **Microsoft Presidio** (recognisers + anonymisers). Measure recall per entity type: missing PII is usually worse than over-masking.

    **Sources:** [Microsoft Presidio](https://microsoft.github.io/presidio/) · [GLiNER, arXiv:2311.08526](https://arxiv.org/abs/2311.08526)

### SD-15 · Self-hosting cost per token 🔴

**Q:** How do you estimate the cost per million tokens of a self-hosted model and compare it with an API?

??? success "Answer"
    - `cost per 1M tokens = GPU $/hour ÷ (aggregate tokens/s × 3,600 × utilisation) × 10⁶`.
    - Example: H100 at $2.50/h, 2,000 output tokens/s aggregate with batching, 50% average utilisation → 2.50 ÷ (2,000 × 3,600 × 0.5) × 10⁶ ≈ **$0.69 per 1M tokens**.
    - Add: engineering/ops headcount, redundancy (N+1 replicas), idle capacity off-peak, storage, model upgrades.
    - Self-hosting wins with **high, steady utilisation** or hard constraints (sovereignty, offline, fine-tuned small model); APIs win for spiky or low volume.
    - Benchmark the throughput yourself (`vllm bench serve`) with your token distributions.

    **Sources:** [vLLM — Benchmark CLI](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) · [How to Scale Your Model — Inference](https://jax-ml.github.io/scaling-book/inference/)

### SD-16 · What to monitor 🟢

**Q:** List the production metrics for an LLM feature, by category.

??? success "Answer"
    - **System:** request rate, error rate (by type: 429, 5xx, timeouts), TTFT and E2E latency percentiles, provider availability.
    - **Usage/cost:** input/output/cached tokens, cost per request, per user, per tenant, per feature.
    - **Quality:** online eval scores on sampled traces (groundedness, relevance), refusal rate, user feedback (thumbs, regenerations, copy events), escalation-to-human rate.
    - **Drift:** input topic/language distribution, retrieval hit rate and scores, provider model version changes.
    - **Safety:** guardrail trigger rates, injection attempts, PII detections.

    **Sources:** [Observability — LLMOps (this handbook)](../GenAI%20Engineering/Chapter%207%20-%20LLMOps/Observability.md) · [Tracing Solutions 2026 (this handbook)](../AI%20Agents%20Engineering/5.%20Agent%20Ops/tracing_solutions_2026.md)

## 7. AI agents: patterns, architecture, practices and enterprise readiness

### AG-01 · Workflows vs agents 🟢

**Q:** How does Anthropic define workflows vs agents, and what are the five workflow patterns?

??? success "Answer"
    - **Workflow:** LLMs and tools orchestrated through **predefined code paths**. **Agent:** the LLM **dynamically directs** its own process and tool use in a loop.
    - Patterns:
        1. **Prompt chaining:** sequential steps with checks between them.
        2. **Routing:** classify input, send to specialised handler/model.
        3. **Parallelisation:** sectioning (independent subtasks) or voting (same task several times).
        4. **Orchestrator-workers:** a central LLM decomposes dynamically and delegates.
        5. **Evaluator-optimiser:** one LLM generates, another critiques in a loop.
    - Guidance: start with the simplest solution; use agents only when steps cannot be predicted.

    **Sources:** [Building effective agents (Anthropic)](https://www.anthropic.com/research/building-effective-agents) · [Agent Architecture Patterns (this handbook)](../AI%20Agents%20Engineering/2.%20Architecture%20and%20Design/agent-architecture-patterns.md)

### AG-02 · ReAct 🟢

**Q:** Describe the ReAct loop and its reported results.

??? success "Answer"
    - Interleaves **Thought** (reasoning) → **Action** (tool call) → **Observation** (tool result), repeated until a final answer. Today implemented with native tool calling.
    - Results: on ALFWorld and WebShop, ReAct beat imitation and RL baselines by **34% and 10% absolute success rate**; on HotpotQA/FEVER it reduced hallucination by grounding reasoning in retrieved facts.
    - Limits: myopic step-by-step decisions, error accumulation, cost per step.

    **Sources:** [ReAct, arXiv:2210.03629](https://arxiv.org/abs/2210.03629) · [Agent Architecture (this handbook)](../AI%20Agents%20Engineering/2.%20Architecture%20and%20Design/agent-architecture.md)

### AG-03 · Planning and reflection patterns 🟠

**Q:** Compare plan-and-execute, Reflexion and Tree of Thoughts, with results.

??? success "Answer"
    - **Plan-and-execute** (also ReWOO): a planner writes the full plan, executors run steps, replanning on failure → fewer LLM calls, easier to audit, and more robust to injection (plan fixed before reading untrusted data).
    - **Reflexion:** after a failure, the agent writes a verbal self-critique stored in memory and retries; reported **91% pass@1 on HumanEval** vs 80% for GPT-4 at the time.
    - **Tree of Thoughts:** explores several reasoning branches with evaluation and backtracking; Game of 24 success **74% vs 4%** with chain-of-thought (GPT-4), at much higher cost.

    **Sources:** [ReWOO, arXiv:2305.18323](https://arxiv.org/abs/2305.18323) · [Reflexion, arXiv:2303.11366](https://arxiv.org/abs/2303.11366) · [Tree of Thoughts, arXiv:2305.10601](https://arxiv.org/abs/2305.10601) · [Task Decomposition (this handbook)](../AI%20Agents%20Engineering/2.%20Architecture%20and%20Design/task-decomposition-and-conflict-resolution.md)

### AG-04 · Multi-agent topologies 🟠

**Q:** Name the multi-agent topologies and the evidence for and against multi-agent systems.

??? success "Answer"
    - Topologies: **supervisor/orchestrator–workers**, **hierarchical** (supervisors of supervisors), **peer-to-peer/network**, **handoffs/swarm** (control passed between agents), **sequential pipeline**.
    - **For:** Anthropic's research system (Opus lead + Sonnet subagents) beat single-agent Opus by **90.2%** on their research eval; token usage explained about 80% of variance on BrowseComp.
    - **Against/costs:** about **15× the tokens of a chat** (single agent about 4×); Cognition argues tightly coupled tasks (coding) suffer from conflicting implicit decisions; the MAST study catalogued **14 failure modes** in 3 categories (specification, inter-agent misalignment, verification).
    - Use multi-agent for breadth-first, parallelisable, loosely coupled tasks.

    **Sources:** [How we built our multi-agent research system (Anthropic)](https://www.anthropic.com/engineering/multi-agent-research-system) · [Don't Build Multi-Agents (Cognition)](https://cognition.ai/blog/dont-build-multi-agents) · [Why Do Multi-Agent LLM Systems Fail?, arXiv:2503.13657](https://arxiv.org/abs/2503.13657) · [Agent Hierarchy (this handbook)](../AI%20Agents%20Engineering/2.%20Architecture%20and%20Design/agent-hierarchy.md)

### AG-05 · Compounding error 🟢

**Q:** An agent's steps each succeed with 95% probability. What is the success rate of a 20-step task, and what do you do about it?

??? success "Answer"
    - 0.95²⁰ ≈ **0.358** (36%), assuming independent steps. At 99% per step: 0.99²⁰ ≈ 82%.
    - Mitigations: fewer steps (deterministic code for known paths), **verification** after critical steps (tests, schema checks, evaluators), checkpoints and retries per step, human approval at high-risk points, decomposition into shorter independent tasks.

    **Sources:** [Agent Architecture Patterns — compounding probabilistic error (this handbook)](../AI%20Agents%20Engineering/2.%20Architecture%20and%20Design/agent-architecture-patterns.md)

### AG-06 · Tool design principles 🟠

**Q:** What are the principles for designing tools that agents use well?

??? success "Answer"
    - **Task-level tools** over thin API wrappers (e.g. `schedule_event` instead of `list_users` + `list_events` + `create_event`).
    - **Namespacing** of tool names by service/resource; unambiguous parameter names (`user_id`, not `user`).
    - Return **meaningful, token-efficient** context: human-readable identifiers, pagination/filtering, concise vs detailed modes (Claude Code caps tool responses at 25,000 tokens by default).
    - **Actionable error messages** that tell the model how to fix the call.
    - Descriptions written like docs for a new hire; evaluate tools with real tasks and iterate.

    **Sources:** [Writing effective tools for agents (Anthropic)](https://www.anthropic.com/engineering/writing-tools-for-agents) · [Evaluating AI Agents — tool layer (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/evaluation.md)

### AG-07 · MCP primitives and transports 🟠

**Q:** What are the MCP server and client primitives, and the transports?

??? success "Answer"
    - **Server primitives:** **tools** (model-invoked actions), **resources** (application-provided context, URIs), **prompts** (user-invoked templates).
    - **Client primitives:** **sampling** (server asks the client's LLM for a completion), **roots** (filesystem/URI boundaries), **elicitation** (server asks the user for input).
    - **Transports:** **stdio** (local subprocess) and **Streamable HTTP** (remote; replaced the older HTTP+SSE transport in the 2025-03-26 spec).
    - Messages are **JSON-RPC 2.0**; remote servers use **OAuth 2.1** authorisation.

    **Sources:** [MCP specification](https://modelcontextprotocol.io/specification/latest) · [MCP — Transports](https://modelcontextprotocol.io/specification/latest/basic/transports) · [MCP — Authorization](https://modelcontextprotocol.io/specification/latest/basic/authorization)

### AG-08 · Function calling, MCP and A2A 🟢

**Q:** What is the difference between function calling, MCP and A2A?

??? success "Answer"
    - **Function calling:** model API feature; the model emits a structured call for a tool defined in the request; the application executes it.
    - **MCP:** protocol between an agent/host and **tool/context servers**, so tools are reusable across hosts (Claude, ChatGPT, IDEs).
    - **A2A:** protocol between **agents**; each agent publishes an **Agent Card** (capabilities, endpoint, auth); interactions are **tasks** with a lifecycle, messages made of parts, and **artifacts** as outputs; JSON-RPC over HTTP with streaming.
    - They stack: an agent uses function calling internally, MCP for tools, A2A to delegate to remote agents.

    **Sources:** [A2A protocol specification](https://a2a-protocol.org/latest/specification/) · [MCP specification](https://modelcontextprotocol.io/specification/latest) · [Agent Interaction Patterns (this handbook)](../AI%20Agents%20Engineering/2.%20Architecture%20and%20Design/agent-interaction-patterns.md)

### AG-09 · Agent memory types 🟠

**Q:** Which types of memory does an agent need, and how is each implemented?

??? success "Answer"
    | Type | Content | Implementation |
    |---|---|---|
    | Working | Current task context | Context window, scratchpad |
    | Episodic | Past interactions/events | Event log + vector search, summaries |
    | Semantic | Facts about user/domain | Knowledge base, profile store, graph |
    | Procedural | How to do things | System prompt, skills, tool definitions |
    - Each needs **write** (what to store, extraction), **retrieval** (when, how much) and **forgetting** (TTL, consolidation) policies.
    - Evidence: Mem0 reported +26% (LLM-as-judge) over OpenAI's memory on LOCOMO, with 91% lower p95 latency and over 90% token savings vs full-context.
    - Risk: memory poisoning (OWASP ASI06).

    **Sources:** [Memory (this handbook)](../AI%20Agents%20Engineering/3.%20Runtime%20Engineering/memory.md) · [MemGPT, arXiv:2310.08560](https://arxiv.org/abs/2310.08560) · [Mem0, arXiv:2504.19413](https://arxiv.org/abs/2504.19413) · [CoALA, arXiv:2309.02427](https://arxiv.org/abs/2309.02427)

### AG-10 · Context engineering techniques 🟠

**Q:** Which techniques keep an agent's context effective over long tasks?

??? success "Answer"
    - **Compaction:** summarise history near the limit, keep decisions and open issues, drop raw tool outputs.
    - **Tool-result clearing:** remove old tool outputs that were already used.
    - **Structured note-taking:** agent writes notes/todo files outside the context and re-reads them.
    - **Sub-agents:** explore in an isolated context, return a condensed summary (1–2k tokens).
    - **Just-in-time retrieval:** keep references (paths, IDs) and load content when needed instead of preloading.
    - Keep the system prompt at the "right altitude" (clear principles, not brittle if-else rules) and a minimal tool set.

    **Sources:** [Effective context engineering for AI agents (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) · [Context Manager (this handbook)](../AI%20Agents%20Engineering/3.%20Runtime%20Engineering/context-manager.md)

### AG-11 · Durable execution 🟠

**Q:** How do you make a long-running agent survive crashes and pauses for human approval?

??? success "Answer"
    - **Checkpointing** state after each step to a durable store; resume from the last checkpoint with the same thread/run ID.
    - **LangGraph:** checkpointer (Postgres, Redis…) keyed by `thread_id`; `interrupt()` pauses the graph, `Command(resume=...)` continues.
    - **Temporal / DBOS / Restate:** durable workflows; LLM and tool calls as activities with retries; deterministic replay.
    - Side-effecting tools must be **idempotent** (SE-03) because a step may run again after a crash.

    **Sources:** [LangGraph — Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) · [LangGraph — Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) · [Temporal](https://docs.temporal.io/) · [Agent Orchestration (this handbook)](../AI%20Agents%20Engineering/5.%20Agent%20Ops/agents-orchestration.md)

### AG-12 · Human-in-the-loop patterns 🟢

**Q:** List the human-in-the-loop patterns for agents.

??? success "Answer"
    - **Approve / edit / reject** a proposed tool call before execution (interrupt before action).
    - **Review output** before it is sent (draft mode).
    - **Escalation** on low confidence, policy triggers or high amounts.
    - **Clarification**: agent asks the user for missing information (MCP elicitation).
    - **Asynchronous approval** via ticket/Slack for long-running processes.
    - **Post-hoc review** of sampled runs feeding evals.
    - Measure approval/override rates to detect rubber-stamping (automation bias).

    **Sources:** [Human-in-the-Loop & Feedback Loops (this handbook)](../AI%20Agents%20Engineering/3.%20Runtime%20Engineering/human%20and%20feedback%20loops.md) · [LangGraph — Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)

### AG-13 · Agent identity and delegation 🔴

**Q:** How should an agent authenticate to downstream systems when acting for a user?

??? success "Answer"
    - Give the agent its **own workload identity** (e.g. SPIFFE ID, managed identity, service principal) separate from the user.
    - **Delegation (on-behalf-of):** exchange the user's token for a **short-lived, down-scoped** token via OAuth 2.0 Token Exchange (RFC 8693), so downstream sees both user and agent (`act` claim).
    - Scopes per tool and per task, short TTLs, no long-lived shared API keys, no reuse of user passwords/sessions.
    - Audit every call with user + agent + tool + decision; revocation path for a misbehaving agent.

    **Sources:** [RFC 8693 — OAuth 2.0 Token Exchange](https://www.rfc-editor.org/rfc/rfc8693) · [SPIFFE](https://spiffe.io/) · [Agent Taxonomy, Granularity & Identity (this handbook)](../AI%20Agents%20Engineering/1.%20Foundations/agent-identity-taxonomy.md)

### AG-14 · Defending agents against prompt injection 🔴

**Q:** What are the design patterns that make agents robust to indirect prompt injection?

??? success "Answer"
    - Break the **lethal trifecta** (private data + untrusted content + exfiltration channel).
    - Six patterns (Beurer-Kellner et al., 2025):
        1. **Action-selector:** LLM only picks from fixed actions, never sees tool outputs.
        2. **Plan-then-execute:** plan fixed before reading untrusted data.
        3. **LLM map-reduce:** isolated sub-agents process each untrusted item; only constrained results aggregated.
        4. **Dual LLM:** privileged LLM never sees untrusted text; quarantined LLM processes it, results passed as symbolic variables.
        5. **Code-then-execute:** LLM writes a program; data flows through the program, not the prompt.
        6. **Context-minimisation:** remove the user prompt/untrusted text before later steps.
    - **CaMeL** (Google DeepMind): capability-based data-flow control; solved about two-thirds of AgentDojo tasks with provable security.
    - Detection classifiers help but are not a security boundary.

    **Sources:** [Design Patterns for Securing LLM Agents, arXiv:2506.08837](https://arxiv.org/abs/2506.08837) · [CaMeL, arXiv:2503.18813](https://arxiv.org/abs/2503.18813) · [The lethal trifecta (Simon Willison)](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)

### AG-15 · OWASP Top 10 for Agentic Applications 🟠

**Q:** Name the ten risks of the OWASP Top 10 for Agentic Applications 2026.

??? success "Answer"
    - ASI01 Agent Goal Hijack
    - ASI02 Tool Misuse and Exploitation
    - ASI03 Identity and Privilege Abuse
    - ASI04 Agentic Supply Chain Vulnerabilities
    - ASI05 Unexpected Code Execution (RCE)
    - ASI06 Memory and Context Poisoning
    - ASI07 Insecure Inter-Agent Communication
    - ASI08 Cascading Failures
    - ASI09 Human-Agent Trust Exploitation
    - ASI10 Rogue Agents

    **Sources:** [OWASP Top 10 for Agentic Applications 2026 (this handbook)](../AI%20Agents%20Engineering/5.%20Agent%20Ops/vulnerabilities.md) · [OWASP GenAI Security Project](https://genai.owasp.org/)

### AG-16 · Agent release artifacts 🟠

**Q:** What makes up an agent release, and how do you roll it out?

??? success "Answer"
    - Artifacts: **code**, **system prompt(s)**, **model ID + version snapshot**, **tool schemas**, **config** (temperature, limits, routing), **retrieval index/embedding version**, **guardrail policies**, eval dataset version.
    - Release them as one **versioned bundle** (or with pinned references); any artifact can change behaviour.
    - Rollout: eval gate in CI → **shadow** on production traffic → **canary** (1–5%) with quality, cost and safety metrics → progressive rollout; feature flags for instant rollback.
    - Keep tool schemas and memory formats backward compatible for in-flight sessions.

    **Sources:** [Agent Artifact Taxonomy and Release Lifecycle (this handbook)](../AI%20Agents%20Engineering/5.%20Agent%20Ops/agent-artifacts-release.md) · [Deployment Strategies (this handbook)](../AI%20Agents%20Engineering/5.%20Agent%20Ops/agent-deployment-strategies.md)

### AG-17 · Agent frameworks and runtimes 🟢

**Q:** Give the defining feature of each: LangGraph, OpenAI Agents SDK, Claude Agent SDK, Pydantic AI, CrewAI, Google ADK, Microsoft Agent Framework, and managed runtimes.

??? success "Answer"
    - **LangGraph:** agents as stateful graphs; checkpointers, interrupts, time travel; low-level control.
    - **OpenAI Agents SDK:** agents, **handoffs**, guardrails, sessions, built-in tracing; lightweight.
    - **Claude Agent SDK:** the Claude Code harness as a library: file/bash tools, subagents, hooks, MCP, automatic compaction.
    - **Pydantic AI:** type-safe agents, dependency injection, validated structured outputs, multi-provider.
    - **CrewAI:** role-based "crews" plus event-driven "flows"; fast prototyping.
    - **Google ADK:** code-first multi-agent toolkit, Gemini/Vertex integration, A2A support.
    - **Microsoft Agent Framework:** successor that unifies Semantic Kernel and AutoGen (.NET and Python).
    - **Managed runtimes:** AWS Bedrock AgentCore, Vertex AI Agent Engine, Azure AI Foundry Agent Service: hosting, identity, memory, observability.

    **Sources:** [LangGraph](https://langchain-ai.github.io/langgraph/) · [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) · [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) · [Pydantic AI](https://ai.pydantic.dev/) · [CrewAI](https://docs.crewai.com/) · [Google ADK](https://google.github.io/adk-docs/) · [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) · [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/)

### AG-18 · Regulatory and standards landscape 🔴

**Q:** Which regulations and standards apply to enterprise AI agents in Europe, with key dates?

??? success "Answer"
    - **EU AI Act** (in force 1 Aug 2024): prohibited practices from 2 Feb 2025; general-purpose AI model obligations from 2 Aug 2025; most remaining obligations incl. Annex III high-risk systems from 2 Aug 2026; Annex I (product) high-risk from 2 Aug 2027. The Commission's late-2025 Digital Omnibus proposed delaying high-risk deadlines: check the current status.
    - **GDPR**: lawful basis, minimisation, Article 22 (automated decisions with legal effects).
    - **DORA** (EU Digital Operational Resilience Act, financial sector, from 17 Jan 2025): ICT risk and third-party (provider) management.
    - **ISO/IEC 42001:2023**: certifiable AI management system.
    - **NIST AI RMF** (Govern, Map, Measure, Manage) and its **GenAI profile NIST AI 600-1**.

    **Sources:** [EU AI Act implementation timeline](https://artificialintelligenceact.eu/implementation-timeline/) · [ISO/IEC 42001](https://www.iso.org/standard/42001) · [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) · [NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) · [Responsible AI (this handbook)](../GenAI%20Engineering/Chapter%205%20-%20Evaluation/Responsible%20AI.md)

### AG-19 · Agent cost controls 🟢

**Q:** Which controls bound the cost of an agent in production?

??? success "Answer"
    - Hard limits: **max steps/turns**, max tokens per run, wall-clock timeout, per-user/tenant budgets with alerts.
    - Model tiering: small model for routing/sub-tasks, large model only for hard steps.
    - Prompt caching of the stable prefix (system prompt + tools), compaction, trimming tool outputs.
    - Loop detection (same tool + same arguments repeated) and a kill switch.
    - Cost attribution per run in traces; review the most expensive runs weekly.

    **Sources:** [Effective context engineering (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) · [Agent Architecture Patterns — unbounded recursion (this handbook)](../AI%20Agents%20Engineering/2.%20Architecture%20and%20Design/agent-architecture-patterns.md)

## 8. Evaluation of GenAI and agentic AI for the enterprise

### EV-01 · Types of graders 🟢

**Q:** What types of graders exist for LLM evals, and when do you use each?

??? success "Answer"
    | Grader | Examples | Use when | Limits |
    |---|---|---|---|
    | **Code-based** | Exact match, regex, JSON schema, unit tests, SQL result check, state check | Objective, verifiable outputs | Cannot judge open-ended quality |
    | **Model-based** (LLM-as-judge) | Rubric scoring, pairwise comparison, groundedness | Open-ended text, scale | Biases, needs validation against humans |
    | **Human** | SME review, annotation | Ground truth, judge calibration, high-stakes | Slow, expensive, inconsistent |
    - Prefer code-based wherever possible; use LLM judges for the rest, calibrated on human labels.

    **Sources:** [Your AI Product Needs Evals (Hamel Husain)](https://hamel.dev/blog/posts/evals/) · [OpenAI — Evaluation best practices](https://platform.openai.com/docs/guides/evaluation-best-practices) · [Evaluation Foundations (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/foundations.md)

### EV-02 · Offline, online and continuous evaluation 🟢

**Q:** Define offline, online and continuous evaluation and what each catches.

??? success "Answer"
    - **Offline:** fixed dataset before release (CI, model/prompt comparison) → catches regressions before users see them.
    - **Online:** on live traffic (sampled traces scored by judges, user feedback, A/B tests, shadow mode) → catches real-world distribution and behaviour.
    - **Continuous:** scheduled re-runs and drift monitoring; production failures added to the offline set → catches silent degradation (provider model updates, data drift).

    **Sources:** [Evaluation Temporality (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/evaluation-temporality.md) · [Evaluating AI Agents (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/evaluation.md)

### EV-03 · Error analysis 🟠

**Q:** Describe the error-analysis process that should precede choosing eval metrics.

??? success "Answer"
    1. Collect about 50–100 diverse real (or realistic synthetic) traces.
    2. **Open coding:** a domain expert writes free-text notes on the first failure in each trace.
    3. **Axial coding:** group notes into a **failure taxonomy** (e.g. wrong tool, missing citation, ignored constraint).
    4. Count frequency per category and prioritise.
    5. Write evaluators (code or judge) **for the top failure modes**, not generic metrics.
    6. Repeat after each change; criteria evolve as you look at outputs ("criteria drift").

    **Sources:** [LLM Evals FAQ (Husain & Shankar)](https://hamel.dev/blog/posts/evals-faq/) · [Who Validates the Validators? (criteria drift), arXiv:2404.12272](https://arxiv.org/abs/2404.12272)

### EV-04 · Building an eval dataset 🟢

**Q:** Where do eval examples come from, and what must the dataset contain?

??? success "Answer"
    - Sources: production logs (anonymised), SME-written cases, support tickets, synthetic generation along **dimensions** (persona × intent × difficulty), adversarial/red-team cases.
    - Content: input, expected output or **grading criteria**, metadata (segment, language, difficulty, source), version.
    - Coverage: happy paths, edge cases, out-of-scope requests (expected refusal), multi-turn, each language/segment that matters.
    - Keep a **held-out** set not used during prompt iteration; version the dataset with the agent release.

    **Sources:** [Data strategy — Agent Evaluation (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/agent_specifics.md) · [LLM Evals FAQ (Husain & Shankar)](https://hamel.dev/blog/posts/evals-faq/)

### EV-05 · Sample size and error bars 🟠

**Q:** Your eval has 100 examples and a 80% pass rate. What is the 95% confidence interval, and how many examples for ±4 points?

??? success "Answer"
    - Standard error of a proportion: `SE = √(p(1−p)/n)` = √(0.8 × 0.2 / 100) = 0.04 → 95% CI ≈ ±1.96 × 0.04 = **±7.8 points** (72%–88%).
    - For ±4 points: n = 1.96² × 0.16 / 0.04² ≈ **384** (≈ 400).
    - Use **clustered standard errors** when questions come in groups (same document), and **average several runs** per item to reduce sampling variance.
    - Many "improvements" of 2–3 points on 100-item sets are noise.

    **Sources:** [Adding Error Bars to Evals (Miller), arXiv:2411.00640](https://arxiv.org/abs/2411.00640)

### EV-06 · Comparing two versions 🔴

**Q:** How do you decide whether version B is better than version A on the same eval set?

??? success "Answer"
    - Use a **paired** comparison (same items): compute per-item score differences; paired SE is smaller than comparing two independent means.
    - Binary outcomes: **McNemar's test** on discordant pairs; continuous scores: paired t-test or **paired bootstrap**.
    - Decide the **minimum detectable effect** and sample size before running (power analysis).
    - Check per-segment results (a global +2% can hide −10% on a key segment) and cost/latency deltas.
    - For pairwise LLM-judge comparisons, randomise/swap order.

    **Sources:** [Adding Error Bars to Evals (Miller), arXiv:2411.00640](https://arxiv.org/abs/2411.00640) · [McNemar's test (NIST/SEMATECH e-Handbook)](https://www.itl.nist.gov/div898/handbook/)

### EV-07 · Designing an LLM judge 🟠

**Q:** What are the design rules for a reliable LLM-as-judge?

??? success "Answer"
    - **One criterion per judge**, with a **binary pass/fail** verdict (more consistent than 1–10 scales).
    - Explicit rubric with definitions and **few-shot examples** of pass and fail.
    - Ask for the **reasoning before the verdict**; structured output (`{"reasoning": ..., "pass": true}`).
    - Provide a **reference answer** or source context when available.
    - Pairwise comparisons: evaluate both orders.
    - Pin the judge model version; ideally a different model family than the system under test.
    - Validate against human labels before trusting it (EV-08).

    **Sources:** [Judging LLM-as-a-Judge (MT-Bench), arXiv:2306.05685](https://arxiv.org/abs/2306.05685) · [G-Eval, arXiv:2303.16634](https://arxiv.org/abs/2303.16634) · [Using LLM-as-a-Judge (Hamel Husain)](https://hamel.dev/blog/posts/llm-judge/)

### EV-08 · Validating a judge 🔴

**Q:** How do you measure whether an LLM judge agrees with humans, and which biases do you check?

??? success "Answer"
    - Label 100–200 examples by domain experts; split into train (for prompt iteration) / dev / test.
    - Metrics: **TPR and TNR** (recall on pass and on fail), and/or **Cohen's kappa** `κ = (p_o − p_e) / (1 − p_e)` (agreement corrected for chance; > 0.6 substantial).
    - With known TPR/TNR you can correct the observed pass rate to estimate the true rate.
    - Biases (MT-Bench study): **position**, **verbosity**, **self-enhancement** (judge prefers its own family). Strong judges reached **> 80% agreement with humans**, about the human–human level.

    **Sources:** [Judging LLM-as-a-Judge, arXiv:2306.05685](https://arxiv.org/abs/2306.05685) · [Using LLM-as-a-Judge (Hamel Husain)](https://hamel.dev/blog/posts/llm-judge/) · [Cohen's kappa (Wikipedia)](https://en.wikipedia.org/wiki/Cohen%27s_kappa)

### EV-09 · RAG evaluation metrics 🟢

**Q:** Which metrics evaluate the retrieval and the generation parts of RAG?

??? success "Answer"
    - **Retrieval** (needs relevance labels): recall@k, precision@k, **MRR** (1/rank of first relevant), **nDCG@k** (graded relevance, position-discounted), hit rate.
    - **Generation** (RAGAS definitions):
        - **Faithfulness:** share of answer claims supported by the retrieved context.
        - **Answer relevancy:** how well the answer addresses the question.
        - **Context precision:** relevant chunks ranked high.
        - **Context recall:** share of ground-truth claims found in the retrieved context.
    - TruLens "RAG triad": context relevance, groundedness, answer relevance.
    - Evaluate retrieval and generation **separately** to locate failures.

    **Sources:** [RAGAS, arXiv:2309.15217](https://arxiv.org/abs/2309.15217) · [TruLens RAG triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/) · [Evaluation for NLP and RAG (this handbook)](../GenAI%20Engineering/Chapter%205%20-%20Evaluation/Evaluation%20for%20NLP.md)

### EV-10 · Hallucination detection 🟠

**Q:** What are the methods to detect hallucinations automatically?

??? success "Answer"
    1. **Groundedness vs source:** NLI/entailment model or LLM judge checks each claim against retrieved context (e.g. Vectara HHEM, RAGAS faithfulness).
    2. **Citation verification:** check that cited passages exist and support the claim.
    3. **Sampling consistency (SelfCheckGPT):** sample several answers; claims that are inconsistent across samples are likely hallucinated; no external source needed.
    4. **Token-level uncertainty** (log-probs/entropy) where available.
    - In enterprise RAG, (1) and (2) are the most practical; (3) is costly (multiple generations).

    **Sources:** [SelfCheckGPT, arXiv:2303.08896](https://arxiv.org/abs/2303.08896) · [Vectara Hallucination Leaderboard / HHEM](https://github.com/vectara/hallucination-leaderboard) · [RAGAS, arXiv:2309.15217](https://arxiv.org/abs/2309.15217)

### EV-11 · What to evaluate in an agent 🟠

**Q:** Which dimensions do you evaluate for an agent, and how is trajectory matching done?

??? success "Answer"
    - **Outcome:** task success checked on the **final state** (database row created, ticket status) rather than the final message.
    - **Trajectory:** right tools, right arguments, right order, no forbidden actions. Matching modes (agentevals): **strict** (same calls, same order), **unordered**, **subset** (no extra calls), **superset** (all required calls present); or an LLM judge on the trajectory.
    - **Efficiency:** steps, tokens, cost, latency per task.
    - **Safety/policy:** violations, unnecessary data access, HITL bypass.
    - **Tool layer:** tool-selection accuracy, argument validity, error recovery.

    **Sources:** [agentevals (LangChain)](https://github.com/langchain-ai/agentevals) · [Evaluating AI Agents — three layers (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/evaluation.md) · [Agent-Specific Evaluation (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/agent_specifics.md)

### EV-12 · pass@k vs pass^k 🟠

**Q:** Define pass@k and pass^k, give the unbiased estimators, and say which matters for production agents.

??? success "Answer"
    - **pass@k:** probability that **at least one** of k attempts succeeds. Unbiased estimator from n samples with c successes: `1 − C(n−c, k) / C(n, k)` (Codex paper).
    - **pass^k:** probability that **all** k attempts succeed (τ-bench): estimated by `C(c, k) / C(n, k)`.
    - pass@k fits "generate many, pick one with a verifier" (coding); **pass^k measures reliability**, which matters for customer-facing agents.
    - Evidence: in τ-bench retail, the best function-calling agent at the time dropped from about 61% pass^1 to **under 25% pass^8**.

    **Sources:** [Codex / HumanEval, arXiv:2107.03374](https://arxiv.org/abs/2107.03374) · [τ-bench, arXiv:2406.12045](https://arxiv.org/abs/2406.12045)

### EV-13 · Agent benchmarks 🟠

**Q:** Name the main agent benchmarks, what they test, and a reference result.

??? success "Answer"
    | Benchmark | Tests | Reference result |
    |---|---|---|
    | **τ-bench / τ²-bench** | Tool-using customer-service agents with simulated users and policies | Low pass^k consistency (EV-12) |
    | **GAIA** | 466 real-world assistant questions needing browsing, tools, reasoning | Humans 92% vs GPT-4 with plugins 15% at release |
    | **WebArena** | Web tasks on realistic self-hosted sites | GPT-4 agent 14.4% vs humans 78.2% at release |
    | **OSWorld** | Computer-use tasks on real OS apps | Best model 12.2% vs humans 72.4% at release |
    | **SWE-bench Verified** | Resolving real GitHub issues | See AC-10 |
    | **AgentDojo** | Prompt-injection robustness of tool agents | 97 tasks, 629 security test cases |
    | **BrowseComp** | Hard-to-find information on the web | Used in Anthropic multi-agent study |
    - Public results move fast; use them to select candidates, then evaluate on your own tasks.

    **Sources:** [τ²-bench, arXiv:2506.07982](https://arxiv.org/abs/2506.07982) · [GAIA, arXiv:2311.12983](https://arxiv.org/abs/2311.12983) · [WebArena, arXiv:2307.13854](https://arxiv.org/abs/2307.13854) · [OSWorld, arXiv:2404.07972](https://arxiv.org/abs/2404.07972) · [AgentDojo, arXiv:2406.13352](https://arxiv.org/abs/2406.13352) · [BrowseComp (OpenAI)](https://openai.com/index/browsecomp/)

### EV-14 · Simulated users and environments 🔴

**Q:** How do you evaluate a conversational agent end to end before production?

??? success "Answer"
    - **User simulator:** an LLM playing a persona with a goal and hidden information, following a scenario script; vary personas (impatient, ambiguous, adversarial).
    - **Sandboxed environment:** mock or staging versions of tools/APIs with a resettable state (database snapshot), so outcomes are checked on the final state.
    - **Graders:** state check (code) + policy compliance and tone (judge) + trajectory checks.
    - Run each scenario several times (pass^k), track cost and turns.
    - Limits: simulators are more cooperative than real users; calibrate with real transcripts.

    **Sources:** [τ-bench, arXiv:2406.12045](https://arxiv.org/abs/2406.12045) · [τ²-bench, arXiv:2506.07982](https://arxiv.org/abs/2506.07982) · [Worked example — travel booking agent (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/evaluation.md)

### EV-15 · Evals in CI/CD 🟠

**Q:** How do you integrate evals into CI/CD?

??? success "Answer"
    - **Tiered suites:** fast smoke subset (30–50 cases) on every PR touching prompts/tools/models; full suite nightly and before release; red-team suite before major releases.
    - **Thresholds per metric** stored in versioned config (e.g. `evals/configs/thresholds.yaml`); block merge on regression beyond the noise margin (EV-05).
    - Results logged as experiments (MLflow, Langfuse, LangSmith, Braintrust) linked to the git SHA and artifact versions.
    - Pin judge model and dataset versions so scores are comparable over time.
    - Tools: promptfoo, DeepEval (pytest-style), Inspect AI, provider eval APIs.

    **Sources:** [CI/CD pipeline — Agent Evaluation (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/agent_specifics.md) · [promptfoo — CI/CD](https://www.promptfoo.dev/docs/integrations/ci-cd/) · [DeepEval](https://deepeval.com/docs/getting-started) · [Inspect AI (UK AISI)](https://inspect.aisi.org.uk/)

### EV-16 · Online evaluation signals 🟢

**Q:** Which signals do you use to evaluate a GenAI feature in production?

??? success "Answer"
    - **Explicit feedback:** thumbs up/down, ratings, free-text comments (low coverage, biased to extremes).
    - **Implicit feedback:** regenerate clicks, copy/accept events, edits to drafts, abandonment, follow-up rephrasing, escalation to a human, task completion.
    - **Automated scoring** of a traffic sample with the offline judges (groundedness, policy).
    - **Business KPIs:** resolution rate, handling time, conversion; via A/B tests with guardrail metrics.
    - Route failing traces to an annotation queue and then into the offline dataset.

    **Sources:** [Online evaluation — Evaluation Temporality (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/evaluation-temporality.md) · [Langfuse — Evaluation overview](https://langfuse.com/docs/evaluation/overview)

### EV-17 · Red teaming tools 🟠

**Q:** How do you red-team a GenAI application, and which tools exist?

??? success "Answer"
    - Scope attack categories: direct jailbreaks, **indirect prompt injection** (documents, web, tool outputs), data exfiltration, PII leakage, harmful content, excessive agency (unauthorised actions), denial of wallet.
    - Combine **manual** expert testing with **automated** generators and scorers.
    - Tools: **PyRIT** (Microsoft), **garak** (NVIDIA, vulnerability scanner), **promptfoo** red-team mode, **AgentDojo** (agents), **HarmBench** (standardised harmful-behaviour eval).
    - Turn findings into regression tests in the eval suite.

    **Sources:** [PyRIT](https://github.com/Azure/PyRIT) · [garak](https://github.com/NVIDIA/garak) · [promptfoo red teaming](https://www.promptfoo.dev/docs/red-team/) · [HarmBench, arXiv:2402.04249](https://arxiv.org/abs/2402.04249) · [Red Teaming (this handbook)](../GenAI%20Engineering/Chapter%205%20-%20Evaluation/Red%20Teaming.md)

### EV-18 · Benchmark contamination 🟢

**Q:** Why do public benchmark scores overestimate real performance, and what do you do instead?

??? success "Answer"
    - **Contamination:** test items present in pre-training data (memorisation); a 2024 survey of closed models found widespread indirect leakage of benchmark data through API usage.
    - **Saturation and overfitting:** labs optimise for popular benchmarks; harness differences change scores.
    - **Distribution mismatch:** your domain, languages, document formats and policies are absent.
    - Instead: private held-out sets from your data, date-filtered benchmarks (LiveBench, LiveCodeBench), and periodic refresh of your eval sets.

    **Sources:** [Leak, Cheat, Repeat (Balloccu et al.), arXiv:2402.03927](https://arxiv.org/abs/2402.03927) · [LiveBench, arXiv:2406.19314](https://arxiv.org/abs/2406.19314)

### EV-19 · Eval governance in the enterprise 🔴

**Q:** Who owns evaluation in an enterprise, and what has to be documented for audit and regulation?

??? success "Answer"
    - **Ownership:** product/business owners and SMEs own the criteria and golden sets; AI engineers own the harness and judges; risk/compliance approve thresholds for high-risk use cases.
    - **Documentation:** intended use, eval datasets and versions, metrics and thresholds, results per release, known limitations, human oversight design (model/system cards).
    - **Regulatory hooks:** EU AI Act high-risk systems require appropriate levels of **accuracy, robustness and cybersecurity** with declared metrics (Article 15) and post-market monitoring; NIST AI RMF "Measure" function; ISO/IEC 42001 performance evaluation.
    - **Re-evaluation triggers:** provider model update, prompt/tool change, new data source, drift alert, incident.

    **Sources:** [EU AI Act — Article 15](https://artificialintelligenceact.eu/article/15/) · [NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) · [Model Cards, arXiv:1810.03993](https://arxiv.org/abs/1810.03993) · [Agent Registry & promotion (this handbook)](../AI%20Agents%20Engineering/4.%20Evaluation/agent_specifics.md)

### EV-20 · Eval and tracing tooling 🟢

**Q:** Name the main evaluation/observability platforms and their positioning.

??? success "Answer"
    - **Langfuse:** open-source (MIT core), tracing, prompt management, datasets, LLM-as-judge; self-hostable.
    - **Arize Phoenix:** open-source (Elastic License 2.0), OpenTelemetry/OpenInference-native tracing and evals.
    - **MLflow:** open-source, tracing + GenAI evaluation + prompt registry, integrated with Databricks.
    - **LangSmith:** managed (LangChain), tracing, datasets, online evals, annotation queues.
    - **Braintrust:** managed eval-first platform, experiments and scoring.
    - **Libraries:** RAGAS, DeepEval, promptfoo, Inspect AI.

    **Sources:** [Tracing Solutions 2026 (this handbook)](../AI%20Agents%20Engineering/5.%20Agent%20Ops/tracing_solutions_2026.md) · [Langfuse](https://langfuse.com/docs) · [Arize Phoenix](https://arize.com/docs/phoenix) · [MLflow GenAI](https://mlflow.org/docs/latest/genai/) · [LangSmith](https://docs.smith.langchain.com/)

## 9. General resources

### Interview question banks

- [60 GenAI interview questions (awesome-generative-ai-guide)](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/interview_prep/60_gen_ai_questions.md)
- [ML system design (alirezadir/Machine-Learning-Interviews)](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md)
- [Machine learning interview (khangich)](https://github.com/khangich/machine-learning-interview)
- [Data Science / Deep Learning Q&A (youssefHosni)](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers)
- [Devinterview-io question repositories](https://github.com/Devinterview-io?tab=repositories)
- [Deep-ML exercises](https://www.deep-ml.com/)
- [50 LLM and AI engineer interview questions for 2026](https://letsdatascience.com/blog/50-llm-and-ai-engineer-interview-questions-for-2026)
- [What applied-AI engineer interviews test in 2026](https://www.techinterview.org/post/3233476824/ai-engineer-interview-rag-agents-evals/)
- [AI inference infrastructure interview prep 2026](https://www.techinterview.org/post/3233475331/ai-inference-infrastructure-interview-prep-2026-vllm-triton/)

### Books

- Chip Huyen, *AI Engineering* (O'Reilly, 2025) and *Designing Machine Learning Systems* (2022)
- Sebastian Raschka, *Build a Large Language Model (From Scratch)* (2024) and [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [How to Scale Your Model (Google DeepMind)](https://jax-ml.github.io/scaling-book/)
- [Ultra-Scale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
