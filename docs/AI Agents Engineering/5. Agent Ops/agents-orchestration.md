# Agent Orchestration: Patterns, Durability, and Resilience

Orchestrating AI agents is fundamentally different from orchestrating traditional software tasks. An agent invocation is non-deterministic, potentially long-running, dependent on external tools and sub-agents, and expensive to re-execute from scratch. The orchestration layer must account for these properties — or accept silent failures, lost work, and unpredictable costs.

---

## What Does "Orchestrating Agents" Mean?

Agent orchestration is the coordination of one or more autonomous agents executing tasks that may involve:

- Sequential or parallel tool calls
- Delegation to sub-agents
- Human-in-the-loop approval gates
- Long waits for external events (API callbacks, user input, scheduled triggers)
- Stateful multi-turn conversations spanning hours or days

The orchestrator's job is to **ensure the workflow progresses to completion** despite failures at any point in the chain.

---

## How Agent Orchestration Differs from Other Domains

The word "orchestration" is overloaded. It means very different things depending on whether you are coordinating software deployments, data pipelines, ML training, or autonomous agents. Understanding these differences is essential to avoid applying the wrong tool or pattern to agent workflows.

### Comparison Matrix

| Dimension | Software Lifecycle (CI/CD) | Data Engineering (ETL) | ML Workflows | Agent Orchestration |
|---|---|---|---|---|
| **Unit of work** | Build, test, deploy a deterministic artifact | Move/transform a data batch | Train, evaluate, register a model | Execute a non-deterministic reasoning chain |
| **Determinism** | Fully deterministic (same code → same binary) | Deterministic transforms, variable input data | Training is stochastic but bounded | Fundamentally non-deterministic (same input → different output) |
| **Duration** | Minutes (build + deploy) | Minutes to hours (batch window) | Hours to days (training) | Seconds to days (depends on human gates, tool latency) |
| **Failure mode** | Binary: build passes or fails | Data quality issues, schema drift, timeout | Training divergence, OOM, metric regression | Partial completion, hallucination, tool failure, infinite loops |
| **Cost of re-execution** | Low (compute is cheap, artifact is cached) | Medium (reprocess data, but idempotent) | High (GPU hours) | High (LLM tokens) and non-reproducible |
| **State requirements** | Stateless (artifact is the state) | Stateless tasks, state in data store | Checkpoints for long training | Full execution history needed for resume |
| **Retry semantics** | Retry the whole stage | Retry the failed task (idempotent) | Resume from checkpoint | Retry the activity, not the whole workflow |
| **Human involvement** | Approval gates (PR review, deploy approval) | Rare (alerting on failure) | Experiment review, model approval | Frequent (approval, feedback, clarification mid-workflow) |
| **Output validation** | Tests pass / tests fail | Schema validation, row counts | Metric thresholds (accuracy, loss) | No ground truth — requires LLM-as-judge or human eval |
| **Scheduling model** | Event-driven (git push, PR merge) | Time-driven (cron, batch windows) | Event or time (new data, weekly retrain) | Event-driven + long-lived (start on request, run for days) |

### Key Differences Explained

**1. Non-determinism changes everything about retries.**

In CI/CD, retrying a flaky test is safe — the expected output is fixed. In data engineering, replaying a transform on the same input produces the same output. In agent orchestration, retrying a failed step may produce a completely different (and potentially worse) result. This means:

- You must persist successful intermediate results, not just retry from scratch
- Retry policies need to account for output quality, not just success/failure
- Idempotency is not natural — you need explicit idempotency keys for side effects

**2. Duration unpredictability requires durable execution.**

CI/CD pipelines have predictable durations (minutes). ETL jobs have bounded batch windows. ML training is long but estimable. Agent workflows have **unbounded duration** — an agent waiting for human approval might pause for 5 minutes or 5 days. This rules out:

- Worker-thread-per-task models (thread exhaustion)
- Timeout-based scheduling (you cannot set a meaningful timeout on "wait for user")
- In-memory state (process will restart before the workflow completes)

**3. Partial completion is the norm, not the exception.**

A CI/CD pipeline either produces an artifact or doesn't. An ETL job either loads all rows or fails. An agent workflow routinely completes 7 of 10 steps before hitting a tool failure or a quality gate. The orchestrator must:

- Track per-step completion (not just workflow-level pass/fail)
- Support compensation for completed steps when later steps fail permanently
- Allow resumption from the last successful step, not from the beginning

**4. Cost structure demands step-level persistence.**

| Domain | Cost of full re-execution |
|---|---|
| CI/CD | ~$0.01–$0.10 (build minutes) |
| ETL | ~$0.10–$10 (compute + data transfer) |
| ML Training | ~$10–$10,000 (GPU hours) |
| Agent workflow (10 steps) | ~$0.50–$50 (LLM tokens + tool calls) |

Losing the output of step 8 in a 10-step agent workflow and restarting from step 1 means paying for steps 1–8 again — with no guarantee of the same results. Durable execution (persisting each step's output) is not a luxury; it is cost control.

**5. Validation is qualitative, not binary.**

CI/CD has tests. ETL has schema checks. ML has metric thresholds. Agent outputs have... nothing deterministic. You cannot `assert agent_output == expected` because the output is natural language. This means:

- Quality gates require LLM-as-judge or human review
- The orchestrator must support "pause and wait for evaluation" as a first-class primitive
- Rollback criteria are fuzzy ("output quality degraded") rather than binary ("test failed")

### Implications for Tool Selection

| If you're orchestrating... | You need... | Typical tools |
|---|---|---|
| Software builds/deploys | Event triggers, parallelism, artifact caching | GitHub Actions, GitLab CI, Jenkins, ArgoCD |
| Data pipelines | Scheduling, dependency graphs, data lineage | Airflow, Dagster, Prefect, dbt |
| ML workflows | Experiment tracking, checkpointing, model registry | Kubeflow, MLflow, ZenML, Flyte |
| Agent workflows | Durable execution, long timers, compensation, step-level persistence | Temporal, Restate, Inngest, AWS Step Functions |

The mistake teams make most often: using Airflow (a scheduler) to orchestrate agents, then discovering it cannot handle human-in-the-loop waits, has no step-level state persistence, and wastes resources on sensor polling. The orchestration requirements of agents are closer to **transaction coordination in distributed systems** than to **batch job scheduling**.

---

## Orchestration Paradigms

There are three fundamental paradigms for orchestrating agent workflows. They are not mutually exclusive — production systems often combine them.

### 1. Scheduled / DAG-Based Orchestration

The workflow is a directed acyclic graph of tasks, triggered on a schedule or by an event. Each task is stateless and independent.

**Characteristics:**

- Tasks are defined declaratively (DAG structure)
- Execution is time-triggered (cron) or event-triggered
- No persistent state between tasks — data flows via external storage
- Retries are per-task, with configurable count and delay

**When it fits agents:**

- Batch evaluation runs (run agent against a dataset nightly)
- Periodic retraining or knowledge base refresh
- Pipelines where each step is idempotent and short-lived

**Limitations for agents:**

- Cannot natively handle workflows that pause for days waiting on human input
- No built-in mechanism to resume mid-workflow after a crash
- Long waits consume worker resources (sensor polling)

---

### 2. Durable Execution

The workflow is written as imperative code. The platform persists every step's outcome in an event history. If the process crashes, it replays the history to rebuild state and resumes exactly where it left off.

**Characteristics:**

- Workflows are code (functions), not DAG definitions
- Full execution history is persisted — deterministic replay on failure
- Native support for long sleeps (days/months) without consuming resources
- Activities (side effects) are retried automatically and indefinitely until success or explicit timeout
- Compensation logic (Sagas) for rollback on partial failure

**When it fits agents:**

- Multi-step agent workflows with tool calls that may fail transiently
- Human-in-the-loop patterns (agent pauses, waits for approval, resumes)
- Long-running agent sessions (customer onboarding, multi-day research tasks)
- Multi-agent coordination where one agent's output feeds another

**Limitations:**

- Requires deterministic workflow code (no random, no system clock in workflow logic)
- Heavier infrastructure (server + persistence layer)
- Learning curve for the replay/determinism model

---

### 3. Event-Driven / Reactive Orchestration

The workflow is a set of event handlers. Each event triggers a function; state is reconstructed from the event stream.

**Characteristics:**

- Loosely coupled — components communicate via events/messages
- State is derived from event log (event sourcing)
- Naturally scales horizontally
- No central scheduler — execution is reactive

**When it fits agents:**

- Real-time agent responses to user actions (chat, notifications)
- Fan-out patterns (one event triggers multiple agents in parallel)
- Systems where agents are independently deployed microservices

**Limitations:**

- Ordering and exactly-once delivery are hard without additional infrastructure
- Debugging distributed event chains is complex
- No built-in "workflow completion" guarantee without layering durability on top

---

## Why Durability, Resilience, and Retry Mechanisms Are Critical

Agent workflows are uniquely vulnerable to failures that traditional orchestration was never designed to handle. Here is why these properties are non-negotiable for production agent systems.

### 1. Agent Calls Are Expensive and Non-Idempotent

A single agent invocation may cost $0.01–$5.00 in LLM tokens, take 10–120 seconds, and produce a unique output each time. Losing the result of a completed step because the orchestrator crashed means:

- **Wasted cost** — you pay again for the same work
- **Inconsistent state** — downstream steps may have already consumed the (now lost) output
- **User frustration** — a 30-minute research workflow restarting from zero

**Durability solves this**: every completed step is persisted. A crash at step 7 of 10 resumes at step 8, not step 1.

### 2. External Dependencies Fail Constantly

Agents call tools: APIs, databases, search engines, other agents. These fail transiently — rate limits, timeouts, network blips, cold starts. Without automatic retries with backoff:

- A single 429 from an API kills the entire workflow
- The operator must manually re-trigger (if they even notice)
- At scale (thousands of agent runs), manual intervention is impossible

**Retry mechanisms solve this**: transient failures are absorbed transparently. The workflow author writes happy-path code; the platform handles the unhappy path.

### 3. Agent Workflows Are Long-Running

Unlike a 200ms API call, agent workflows can span:

- Minutes (multi-step reasoning with tool calls)
- Hours (research tasks with human review gates)
- Days (onboarding flows, approval chains)

A scheduler that holds a worker thread open for the duration is wasteful. A system that loses state if the process restarts during a 3-day wait is broken.

**Durable timers solve this**: the workflow sleeps without consuming resources and wakes reliably, even if the infrastructure was restarted in between.

### 4. Partial Failures Require Compensation, Not Just Retries

Consider: Agent books a flight (step 1) → Agent books a hotel (step 2, fails permanently). You cannot just retry step 2 forever — you need to **cancel the flight** (compensate step 1). This is the Saga pattern.

Without compensation logic:

- You accumulate orphaned side effects (charges, reservations, notifications sent)
- Manual cleanup is error-prone and doesn't scale

**Saga/compensation support solves this**: each activity can declare its undo operation. On permanent failure, the orchestrator runs compensations in reverse order.

### 5. Observability Requires Execution History

When an agent produces a bad output, you need to answer: *what happened?* Which tools were called, in what order, with what inputs, and what failed along the way?

- Without persisted history, you have only logs (if you're lucky)
- With durable execution, the full event history is the audit trail — every activity invocation, every retry, every timer, every signal

---

## Choosing an Orchestration Strategy for Agents

| Requirement | DAG/Scheduled | Durable Execution | Event-Driven |
|---|---|---|---|
| Batch agent evaluation | ✅ Best fit | ✅ Works | ⚠️ Overkill |
| Multi-step tool-calling agent | ⚠️ Fragile | ✅ Best fit | ⚠️ Complex |
| Human-in-the-loop (wait hours/days) | ❌ Poor fit | ✅ Best fit | ⚠️ Needs state layer |
| Real-time chat agent | ⚠️ Too slow | ⚠️ Overhead | ✅ Best fit |
| Multi-agent coordination | ⚠️ Limited | ✅ Best fit | ✅ Works |
| Long-running research workflow | ❌ Worker exhaustion | ✅ Best fit | ⚠️ Needs durability |
| Simple scheduled pipeline | ✅ Best fit | ⚠️ Overkill | ⚠️ Overkill |

### Decision Framework

```
Is the workflow short-lived (<5 min) and scheduled?
  → DAG-based orchestration is sufficient

Does the workflow wait for external events or humans?
  → Durable execution is required

Does the workflow involve multi-step tool calls that may fail?
  → Durable execution with automatic retries

Is the workflow purely reactive (respond to events as they arrive)?
  → Event-driven, but layer durability if completion matters

Are you coordinating multiple agents with dependencies?
  → Durable execution for the coordination layer
```

---

## Architectural Patterns for Agent Orchestration

### Pattern 1: Orchestrator as Durable Workflow

The orchestrator owns the control flow. Each agent call is an "activity" with retry policies and timeouts.

```
Orchestrator (durable)
  ├── Activity: call Agent A (retry 3x, timeout 60s)
  ├── Activity: call Agent B with A's output
  ├── Timer: wait for human approval (up to 7 days)
  └── Activity: call Agent C to finalize
```

**Pros**: Full visibility, deterministic replay, compensation support.
**Cons**: Orchestrator must be deterministic; adds infrastructure.

### Pattern 2: Choreography with Event Sourcing

Agents publish events; other agents subscribe and react. An event store provides durability.

```
Agent A completes → publishes "research_done" event
Agent B subscribes → picks up event → executes → publishes "summary_ready"
Agent C subscribes → picks up event → finalizes
```

**Pros**: Loose coupling, independent scaling, no single point of failure.
**Cons**: Hard to reason about end-to-end flow; needs correlation IDs and dead-letter handling.

### Pattern 3: Hybrid — Durable Orchestrator + Event Bus

The orchestrator handles the happy path and compensation. Agents communicate results via events, but the orchestrator tracks overall progress.

```
Orchestrator (durable)
  ├── Dispatch task to Agent A via event bus
  ├── Wait for "agent_a_done" signal (with timeout)
  ├── Dispatch task to Agent B
  ├── Wait for "agent_b_done" signal
  └── On timeout → compensate → alert
```

**Pros**: Best of both worlds — loose coupling with completion guarantees.
**Cons**: Most complex to implement and operate.

---

## Anti-Patterns

| Anti-Pattern | Why It Fails |
|---|---|
| **Fire-and-forget agent calls** | No confirmation of completion; lost results are invisible |
| **Retry without idempotency keys** | Duplicate side effects (double bookings, duplicate messages) |
| **Unbounded retries without circuit breaker** | Runaway costs when an LLM endpoint is down |
| **State in memory only** | Process restart = workflow restart from zero |
| **Synchronous waits for async agents** | Thread/worker exhaustion under load |
| **No timeout on agent calls** | Hung agents block the entire pipeline indefinitely |

---

## Resources

### Foundational Concepts

- Temporal Technologies, "What is Durable Execution?," 2024. <https://temporal.io/blog/what-is-durable-execution>
- Bernd Ruecker, "Practical Process Automation," O'Reilly, 2021. (Saga pattern, compensation, orchestration vs. choreography)
- Martin Kleppmann, "Designing Data-Intensive Applications," O'Reilly, 2017. (Event sourcing, exactly-once semantics, distributed systems fundamentals)

### Agent-Specific Orchestration

- Anthropic, "Building Effective Agents," 2024. <https://www.anthropic.com/research/building-effective-agents>
- Andrew Ng, "Agentic Design Patterns," DeepLearning.AI, 2024. <https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/>
- LangGraph Documentation, "Persistence and Human-in-the-Loop." <https://langchain-ai.github.io/langgraph/concepts/persistence/>

### Durable Execution Platforms

- Temporal. <https://temporal.io>
- Restate. <https://restate.dev>
- Inngest. <https://www.inngest.com>
- Azure Durable Functions. <https://learn.microsoft.com/en-us/azure/azure-functions/durable/durable-functions-overview>
- AWS Step Functions. <https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html>

### DAG-Based / Scheduled Orchestration

- Apache Airflow. <https://airflow.apache.org>
- Prefect. <https://www.prefect.io>
- Dagster. <https://dagster.io>

### Comparison Articles

- ZenML, "Temporal vs Airflow: Which Orchestrator Fits Your Workflows?," 2025. <https://www.zenml.io/blog/temporal-vs-airflow>
- ZenML, "Temporal Alternatives: 9 Tools ML and Data Teams Prefer," 2025. <https://www.zenml.io/blog/temporal-alternatives>
- Akka, "The 10 Best Temporal Alternatives for Enterprise Teams," 2025. <https://akka.io/blog/temporal-alternatives>
