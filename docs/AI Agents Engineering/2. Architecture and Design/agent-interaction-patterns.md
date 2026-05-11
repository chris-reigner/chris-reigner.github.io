# Agent Interaction Patterns

How agents communicate with each other and with external systems. The choice of pattern determines latency, observability, coupling, and failure behavior of the entire system.

---

## 1. Request-Response (Synchronous)

The simplest and most common pattern. A calling agent sends a request and waits for the result before continuing.

```
Agent A ──── request ────► Agent B
Agent A ◄─── response ──── Agent B
```

**Properties:**

- Tight coupling — caller blocks until callee responds
- Easy to trace and debug — one request, one response, clear causality
- Latency accumulates sequentially across all hops
- Failure is immediate and visible — if Agent B is unavailable, Agent A fails at that step

**When to use:** Most orchestrator-to-worker calls within a single workflow execution. Any scenario where the caller needs the result before proceeding.

**Protocol:** HTTP REST, gRPC, direct function call within a framework.

---

## 2. Asynchronous Message Passing (Queue-Based)

The caller submits a task to a queue and continues. A worker picks up the task, processes it, and posts the result to a response queue or calls a callback.

```
Agent A ──── enqueue ────► Queue ◄──── poll ─── Agent B
Agent A ◄── callback ──────────────────────────── Agent B
```

**Properties:**

- Loose coupling — caller does not wait; can do other work in parallel
- Durable — queue persists tasks across restarts; no work is lost if a worker crashes
- Requires idempotency — task may be redelivered; worker must handle duplicates safely
- Harder to trace — a single user request produces multiple queue events across time

**When to use:** Long-running subtasks (> 30 seconds), fire-and-forget sub-delegation, batch processing pipelines, any case where the result is not needed immediately.

**Implementations:** Apache Kafka, AWS SQS/EventBridge, Azure Service Bus, Redis Streams.

---

## 3. Publish-Subscribe / Event-Driven

Agents publish events to a shared bus. Other agents subscribe to the event types they care about and react independently. No agent knows who is listening.

```
Agent A ──── publish(event: doc.ingested) ────► Bus
                                                  │
                              ┌───────────────────┤
                              ▼                   ▼
                         Agent B             Agent C
                     (subscribed to      (subscribed to
                      doc.ingested)       doc.ingested)
```

**Properties:**

- Maximum decoupling — producers and consumers have no knowledge of each other
- Scales horizontally — add subscribers without modifying the publisher
- Emergent behavior risk — hard to predict what fires and in what order
- Debugging requires distributed tracing with correlation IDs across all events
- Idempotency is mandatory — events may be replayed

**When to use:** Long-horizon asynchronous pipelines, notification and trigger workflows, scenarios where multiple independent systems must react to the same event. Avoid for workflows requiring strict ordering or guaranteed handoff.

**Implementations:** Kafka (enterprise standard), AWS EventBridge, NATS, Google Pub/Sub.

---

## 4. Streaming (Server-Sent Events / SSE)

An agent opens a persistent connection and receives incremental updates as the callee generates them. The standard mechanism in A2A for long-running tasks.

```
Agent A ──── subscribe(task_id) ────► Agent B
Agent A ◄─── event: step_1 ───────── Agent B
Agent A ◄─── event: step_2 ───────── Agent B
Agent A ◄─── event: completed ─────── Agent B
```

**Properties:**

- Real-time progress visibility — useful for long tasks where intermediate results matter
- One-directional — server pushes to client (not bidirectional like WebSockets)
- Allows early termination — caller can cancel based on intermediate results
- Lower perceived latency for end users even when total compute time is the same

**When to use:** Any interaction where task duration > a few seconds and progress is meaningful to the caller. Standard for agent-to-agent calls in the A2A protocol.

---

## 5. Blackboard / Shared Memory

All agents read from and write to a shared, structured knowledge store. No direct agent-to-agent communication — coordination happens through the shared state.

```
          ┌─────────────────────────────────┐
          │         Shared Blackboard        │
          │  { findings: [...],              │
          │    status: { A: done, B: running }│
          │    conflicts: [...] }             │
          └─────────────────────────────────┘
               ▲ read/write    ▲ read/write
               │               │
          Agent A           Agent B
```

**Properties:**

- Implicit coordination — no direct messaging needed
- Natural conflict detection — agents can see each other's contributions and flag contradictions
- Concurrency hazards — multiple agents writing simultaneously requires locking or conflict resolution logic
- Full audit trail if the blackboard is append-only
- Used in collaborative multi-agent patterns (Competitive Intelligence, Research Synthesis)

**When to use:** Collaborative agent systems where multiple agents contribute to a shared artifact. Research synthesis, multi-perspective analysis, any scenario where agents must build on each other's findings without strict sequencing.

---

## 6. Protocol Comparison

| Protocol | Coupling | Latency | Ordering | Traceability | Best for |
|---|---|---|---|---|---|
| **A2A** | Loose | Medium | Not guaranteed | Trace ID per task | Cross-platform agent interop |
| **MCP** | Tight | Low | Synchronous | Per-tool call | Tool access within a single agent |
| **HTTP REST** | Loose | Low | Per-request | Request ID | Simple agent-to-service calls |
| **gRPC** | Tight | Very low | Per-stream | Per-call | High-throughput agent-to-service |
| **Kafka** | Very loose | Medium | Partition-ordered | Correlation ID | Async pipelines, event-driven |
| **SSE** | Loose (push) | Low perceived | Event stream | Event ID | Long-running task progress |

## 7. The A2A Protocol in Detail

**Agent2Agent (A2A)** — published by Google in April 2025, now a multi-vendor open standard — is the first purpose-built protocol for inter-agent communication across different platforms and vendors.

**Core concepts:**

- **Agent Card:** A JSON document each agent publishes describing its capabilities, supported input/output formats, authentication requirements, and endpoint URL. Agents discover each other via Agent Cards — no hardcoded connections.
- **Task object:** The unit of work exchanged between agents. Has a stable ID, status lifecycle (submitted → working → completed/failed), and supports streaming updates via SSE.
- **Push notifications:** For truly long-running tasks, the server agent can push status updates to a webhook URL provided by the caller — no polling required.
- **Authentication:** OAuth 2.0, OpenID Connect, and mTLS — plugs into existing enterprise identity infrastructure (Entra, Okta, etc.).

**A2A + MCP:** These protocols are complementary, not competing. MCP connects an agent to tools and data sources. A2A connects agents to each other. A production architecture uses both:

```
User
 │
 ▼
Orchestrator Agent
 │ (A2A)                 (MCP)
 ├──► Specialist Agent A ──► Database Tool
 │                       ──► Search Tool
 │ (A2A)
 └──► Specialist Agent B ──► (MCP) API Tool
```

---

## References

- [Google — Agent2Agent Protocol (A2A)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [Anthropic — Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol)
- [Microsoft Research — AutoGen Multi-Agent Framework](https://arxiv.org/abs/2308.08155)
- [Anthropic — Building Effective AI Agents](https://resources.anthropic.com/building-effective-ai-agents)
