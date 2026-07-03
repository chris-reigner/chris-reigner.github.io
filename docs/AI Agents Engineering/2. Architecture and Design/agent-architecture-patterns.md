# Agent Architecture Patterns

> Based on Anthropic's production research and enterprise deployments. Diagrams and patterns sourced from [*Building Effective AI Agents: Architecture Patterns and Implementation Frameworks*](https://resources.anthropic.com/hubfs/Building%20Effective%20AI%20Agents-%20Architecture%20Patterns%20and%20Implementation%20Frameworks.pdf) — Anthropic, 2025.

---

## Design Principles

Before examining patterns, Anthropic's guidance from production systems establishes four foundational principles that apply regardless of architecture complexity:

**Start simple, scale intelligently.** Begin with single-purpose agents that do one thing well, then evolve into more sophisticated systems as requirements emerge. Simple systems cost less to run, are easier to debug, and produce metrics that tie directly to business outcomes.

**Choose the right model for the job.** Balance capabilities, speed, and cost. Complex financial analysis warrants a frontier model; high-volume classification tasks do not. At scale, mismatched model selection compounds in cost and latency.

**Practice modular design.** Prompts in centralized config, tools as discrete reusable modules, agents assembled from only the tools needed for their assigned task. This composition pattern enables evolution without system-wide refactoring.

**Build observable systems.** AI agents are non-deterministic. When an agent fails, a stack trace is insufficient — you need visibility into prompt chains, model decision paths, retrieval contexts, token consumption, and reasoning flow.

---

## Pattern 1 — Single Agent

A single LLM operates in a continuous loop: perceive the environment, decide next steps, act using tools, observe results, adjust approach based on feedback.

![Single Agent architecture diagram — Anthropic](../../assets/agent-patterns/single-agent.png)
*Source: Anthropic — Building Effective AI Agents (2025)*

**Core components:**

- An AI model as the reasoning engine
- A prompt defining role and capabilities
- A toolkit of integrations for external systems
- Skills for specialized domain knowledge loaded on demand
- Memory for state persistence across steps

**When to use:**

- Open-ended problems where the path forward is not predetermined
- Tasks fitting in a single context window with sequential tool calls
- Well-defined domains where added agent coordination overhead is unjustified
- Customer service, document Q&A, code review, routine analysis and reporting

**When to avoid:**

- When you need perfect answers 100% of the time on the first attempt
- When tasks require simultaneous pursuit of multiple independent directions
- When specialized expertise across two or more distinct domains is required — research shows single agents fall off sharply when facing two or more distractor domains

**Complexity cost:** Low. The cheapest and most debuggable architecture. Exhaust single-agent options before scaling to multi-agent.

---

## Pattern 2 — Hierarchical / Supervisory

A central supervisor agent coordinates multiple specialized subagents through intelligent task delegation. The supervisor analyzes requests, routes them to appropriate specialists, and synthesizes responses. Subagents are treated as tools — the supervisor uses a tool-calling model to decide which agent to invoke.

![Multi-agent hierarchical workflow — Anthropic](../../assets/agent-patterns/hierarchical-workflow.png)
*Source: Anthropic — Building Effective AI Agents (2025)*

**Key structural properties:**

- Supervisor owns the plan and final synthesis; it does not execute domain logic
- Subagents receive scoped context — not the full conversation history
- Subagents can themselves have subagents, abstracted from the top supervisor
- The supervisor only interacts with a subagent team leader, unaware of further delegation
- Clear chain of responsibility mirrors effective human organizational structures

**Implementation variations:**

| Variant | Description | Use case |
|---|---|---|
| Full orchestration | Supervisor maintains complete control over user interactions and task execution | Complex pipelines requiring end-to-end coordination |
| Routing-focused | Supervisor specializes in delegation decisions, handing off user communication to specialists | Intent routing in enterprise assistants |
| Hybrid coordination | Supervisor selectively involved based on task complexity | Mixed workloads with simple and complex tasks |

**Key challenge — Context management:**

The orchestrator faces a fundamental problem: context grows too complex for one agent to manage effectively. Mitigations:
- Context editing: automatically clear stale tool calls when approaching token limits
- Memory tools: store and retrieve information outside the context window via file-based or external stores
- Tool pagination and truncation: cap tool responses at manageable sizes (~25,000 tokens)

**When to use:**

- Tasks decomposable into 3+ independent subtasks requiring specialized expertise
- Scenarios where different sub-domains need different models, tools, or data access
- Workflows requiring a clear audit chain and centralized governance
- Moderate control requirements: customer support, content creation, data analysis

**Enterprise example:** Marketing campaign — a Marketing Director agent supervises Market Research, Creative Design, Copywriting, and Media Planning agents. Each specialist reports back to the supervisor, which handles integration and final delivery.

---

## Pattern 3 — Collaborative / Peer-to-Peer

Multiple specialized agents work together in real-time through direct peer-to-peer communication. Coordination emerges from agent interactions rather than being imposed by a central authority. Agents negotiate roles dynamically and collectively solve problems through distributed intelligence.

![Multi-agent collaborative workflow — Anthropic](../../assets/agent-patterns/collaborative-workflow.png)
*Source: Anthropic — Building Effective AI Agents (2025)*

**Coordination mechanisms:**

- **Group chat orchestration:** Multiple agents participate in a shared conversation thread, solving problems through discussion and collaborative reasoning
- **Event-driven coordination:** Events as a shared language — structured updates agents use to interpret instructions, share context, and coordinate tasks
- **Blackboard architecture:** A shared knowledge repository all agents read from and write to, acting as collective memory

**Key challenge — Emergent behavior unpredictability:**

Small changes can unpredictably affect how agents behave. Multi-agent systems exhibit emergent behaviors that arise without specific programming. Prevention requires:
- Frameworks defining division of labor, problem-solving approaches, and effort budgets — not strict instructions
- Mechanisms to prevent agents from bouncing tasks indefinitely
- Explicit conflict resolution logic when agents produce contradictory results

**When to use:**

- Low control requirements: research, brainstorming, complex analysis with diverse perspectives
- Problems requiring simultaneous cross-domain intelligence synthesis
- Scenarios where unpredictability is a feature — exploring possibilities, not executing defined processes

!!! warning "Enterprise caution"
    Collaborative patterns are the hardest to govern and debug. Emergent behaviors are difficult to anticipate. Reserve this pattern for exploration tasks where unpredictability is tolerated. Do not use for financial transactions, compliance workflows, or any scenario requiring deterministic audit trails.

**Enterprise example:** Competitive intelligence — Pricing, Product, Marketing, Financial, Social Media, and Strategic Intelligence agents work in parallel, cross-referencing findings in real-time. No central coordinator; a Report Agent synthesizes the collective output.

---

## Pattern 4 — Sequential Workflow

A predetermined control flow with defined execution paths. Agents hand off work sequentially, each adding specific value that the next stage depends on. Provides predictable, auditable behavior ideal for regulated environments.

![Multi-agent sequential workflow — Anthropic](../../assets/agent-patterns/sequential-workflow.png)
*Source: Anthropic — Building Effective AI Agents (2025)*

**Key properties:**

- Deterministic execution path — the full process flow can be mapped in advance
- Supports both software-defined decision points (conditional logic on outcomes) and AI-driven routing (model decides next step based on intermediate results)
- Clear audit trail: each stage is inspectable independently
- Main trade-off: accuracy gains per step vs. cumulative latency

**When to use:**

- Multi-stage processes with clear linear dependencies
- Data transformation pipelines where each stage adds value the next depends on
- Progressive refinement workflows: draft → review → polish
- Compliance checks requiring traceability and process consistency
- Regulatory environments where every step must be explainable

**When to avoid:**

- When a single agent can handle the task in a few steps
- When agents need to collaborate rather than sequentially hand off
- When the workflow requires backtracking or iteration between stages

**Enterprise example:** Automated data science insights — Scoping Agent → Data Engineering Agent → Analysis Agent → Review/Escalation → Deliver Insights. Each step consumes and builds on the structured output of the prior one.

---

## Pattern 5 — Parallel Workflow

Independent tasks distributed across multiple agents simultaneously, with results merged afterward. Resembles the fan-out/fan-in cloud design pattern. Enables significant speed improvements and diverse perspective coverage.

![Multi-agent parallel workflow — Anthropic](../../assets/agent-patterns/parallel-workflow.png)
*Source: Anthropic — Building Effective AI Agents (2025)*

**Two sub-patterns:**

| Sub-pattern | Mechanism | Example |
|---|---|---|
| **Sectioning** | Different agents handle different independent parts of the problem simultaneously | Guardrails: one model processes queries, another screens for inappropriate content |
| **Voting** | Multiple agents independently evaluate the same input; results aggregated for higher confidence | Code vulnerability review: several agents with different prompts, voting threshold balances false positives/negatives |

**When to use:**

- Subtasks that can execute concurrently without dependencies
- Multiple independent perspectives improve confidence in the result
- Speed matters more than coordination overhead
- Risk assessment requiring diverse, simultaneous viewpoints

**When to avoid:**

- When agents need to build on each other's work with cumulative context
- When a specific deterministic order of operations is required
- When agents cannot reliably coordinate changes to shared state
- When there is no clear conflict resolution strategy for contradictory results
- When result aggregation logic is too complex to be reliable

**Enterprise example:** Financial risk assessment — Credit Risk, Market Risk, Operational Risk, and Regulatory Compliance agents all run simultaneously on the same loan application. A Risk Aggregation Engine synthesizes all outputs into a unified recommendation.

---

## Pattern 6 — Evaluator-Optimizer

Two AI systems in iterative cycles: a **generator** creates content, an **evaluator** assesses it against predefined criteria and provides actionable feedback, the generator incorporates feedback and produces a revised version. Repeats until quality standards are met.

![Multi-agent evaluator workflow — Anthropic](../../assets/agent-patterns/evaluator-optimizer.png)
*Source: Anthropic — Building Effective AI Agents (2025)*

**Key properties:**

- Resembles writer-editor collaboration with structured feedback loops
- Generator and evaluator can use different models (e.g., fast generator + precise evaluator)
- Quality improvement is measurable; typically 2–4 cycles per task
- Higher token cost than single-pass generation — justify with quality requirement

**When to use:**

- Clear evaluation criteria exist and iterative refinement delivers measurable value
- Content creation requiring nuance: literary translation, professional communications, tone-sensitive writing
- Code generation with security or correctness requirements
- Research tasks needing multi-step reasoning with validation

**When to avoid:**

- First-attempt quality already meets requirements
- Evaluation criteria are subjective or poorly defined
- Real-time applications requiring immediate responses
- Simple classification or routine tasks
- Resource-constrained environments with strict token budgets

**Enterprise example:** API documentation — Generator Agent analyzes codebase and drafts docs; Technical Evaluator Agent validates accuracy against actual code implementation; Generator refines based on evaluator feedback. Typically 2–4 cycles; final output auto-published.

---

## Hybrid Patterns

Production systems rarely use a single pattern. Anthropic's guidance explicitly endorses combining patterns:

**Hierarchical + Parallel:** A supervisor delegates to specialists; each specialist runs parallel analyses within its domain. Example: a financial risk supervisor delegates to Credit, Market, and Operational Risk agents, each running parallel sub-analyses.

**Sequential + Dynamic Routing:** Linear processes that invoke different agent types based on intermediate results. A customer service workflow classifies the issue, then routes to either a simple resolution agent or a complex multi-agent research team based on complexity.

**Single Agent + Multi-Agent Escalation:** Simple agents handle routine tasks but automatically trigger a multi-agent system on edge cases. Optimizes cost while maintaining capability for complex scenarios.

---

## Anti-Patterns

### 1. Over-Engineering from Day One

**What it looks like:** Building hierarchical multi-agent systems before proving a single agent cannot solve the problem.

**Why it fails:** Multi-agent systems consume 10–15× more tokens than single agents. The performance gain only justifies the cost for genuinely complex, multi-domain tasks. Most enterprise use cases — customer service, document processing, routine analysis — are single-agent problems.

**Fix:** Start with a single agent. Add Skills to extend capability. Escalate to multi-agent only when single-agent performance hits a measurable ceiling.

---

### 2. Mega-Orchestrator

**What it looks like:** A single orchestrator with 20+ tools and a massive system prompt trying to handle all enterprise use cases.

**Why it fails:** LLMs struggle to correctly select among overlapping tools above ~15–20. Context saturation causes late-prompt instructions to be ignored. Routing accuracy degrades non-linearly with tool count.

**Fix:** Decompose into hub-and-spoke or hierarchical patterns. The orchestrator routes; domain specialists execute. Each agent handles ≤10 tools with non-overlapping descriptions.

---

### 3. Prompt-Only Governance

**What it looks like:** Security constraints, access controls, and compliance rules expressed entirely in the system prompt. No infrastructure enforcement.

**Why it fails:** Prompts can be bypassed via injection or simply ignored under context pressure. 39% of companies reported agents accessing unauthorized systems in 2025 (Shakudo research). Prompts produce no enforceable audit trail.

**Fix:** Enforce access control at the tool/MCP layer with IAM or scoped credentials. The contract between agent reasoning and external systems must be in code, not prose.

---

### 4. Missing Context Budget

**What it looks like:** Orchestrators that reconstruct full workflow state on every step by re-reading all prior agent outputs. No external state store.

**Why it fails:** Context window usage grows with every step. On long workflows the orchestrator loses track of early decisions — compounding errors and failed coordination.

**Fix:** Persist workflow state externally (Redis, DynamoDB). Pass only the minimal context slice relevant to the agent's next action. The orchestrator holds a pointer to state, not the state itself.

---

### 5. Agent Sprawl

**What it looks like:** Teams deploy independent agents for every new task with no registry, no ownership model, no central audit. 94% of organizations report sprawl increasing technical debt and security risk (Gartner, 2025).

**Why it fails:** Duplicate capabilities across agents, inconsistent behavior for the same task, shadow agents outside IT governance, no centralized observability.

**Fix:** Maintain an agent catalog. Define capability ownership. Require security review before production deployment. Apply hub-and-spoke or hierarchical consolidation.

---

### 6. Unbounded Recursion

**What it looks like:** Agents allowed to spawn sub-agents without depth limits or authorization checks.

**Why it fails:** Recursive agent trees grow unboundedly, consuming context, compute, and API quota. A potential vector for prompt injection attacks that escalate privileges through the recursion depth.

**Fix:** Set a maximum recursion depth (2–3 levels in practice). Require explicit authorization for sub-agent spawning. Sub-agents should not be able to spawn further sub-agents unless the root orchestrator explicitly permits it.

---

### 7. Missing Human-in-the-Loop for Irreversible Actions

**What it looks like:** Agents autonomously execute high-impact or irreversible actions — delete, send, pay, deploy — without a human checkpoint.

**Why it fails:** Production data loss, incorrect financial transactions, unwanted external communications that cannot be recalled.

**Fix:** Classify actions by reversibility. Require human approval for irreversible actions above a defined risk threshold. Implement stopping conditions: `"pause here for human review"` as an explicit agent output state.

---

### 8. Compounding Probabilistic Error

**What it looks like:** Long sequential pipelines with 10+ agent steps, each operating at ~95% accuracy.

**Why it fails:** Reliability compounds multiplicatively. A 10-step pipeline at 95% per-step accuracy produces ~60% end-to-end reliability — failing roughly 4 times out of 10. This is invisible in demos; it surfaces at production scale.

**Fix:** Minimize chain depth. Introduce deterministic validation gates between probabilistic steps. Replace agent steps with deterministic code where the logic is rule-based. The agent proposes in typed JSON; a deterministic runtime enforces execution.

---

## Pattern Selection at a Glance

| Control requirement | Problem complexity | Recommended pattern |
|---|---|---|
| High (regulated, financial, safety-critical) | Single domain | Single Agent |
| High | Multi-domain, predictable | Sequential Workflow |
| Moderate | Multi-domain with specialization | Hierarchical / Supervisory |
| Moderate | Independent parallel analyses | Parallel Workflow |
| Moderate | Iterative quality improvement | Evaluator-Optimizer |
| Low | Open-ended research, complex analysis | Collaborative |

**Real-world evolution path (e-commerce example):**

1. Single agent for customer inquiries (prove value)
2. Routing pattern separating order status, product questions, complaints
3. Specialized agents per category with shared context
4. Multi-agent system with inventory, payment, and shipping coordination
5. Evaluator agents for quality assurance and continuous improvement

The key: architecture should evolve with requirements. Start simple, measure everything, add complexity only when it delivers measurable value.

---

## References

- [Anthropic — Building Effective AI Agents: Architecture Patterns and Implementation Frameworks (PDF)](https://resources.anthropic.com/hubfs/Building%20Effective%20AI%20Agents-%20Architecture%20Patterns%20and%20Implementation%20Frameworks.pdf)
- [Anthropic — How We Built Our Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Anthropic — Building Effective AI Agents (guide)](https://resources.anthropic.com/building-effective-ai-agents)
- [O'Reilly — The Missing Layer in Agentic AI](https://www.oreilly.com/radar/the-missing-layer-in-agentic-ai/)
- [Shakudo — Why 80% of Enterprise AI Agents Fail in Production](https://www.shakudo.io/blog/enterprise-ai-agent-production-failures)
- [Gartner — Managing AI Agent Sprawl](https://cxotoday.com/editors-picks/how-to-manage-ai-agent-sprawl-a-six-step-framework-by-gartner/)
