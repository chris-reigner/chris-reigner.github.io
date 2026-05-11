# Architecting AI Agents

## Multi-tenant isolation

Stronger isolation is one collection or indices per tenant or one namespace per tenant.
This gives hard infrastructure and storage boundaries.

Softer boundary is to have different indices in the same workspace/environment/account.
More risky as exposed to misconfiguration.

tenant-scoped AES-256 at rest and in-transit TLS per namespace. (encryption at rest and in transit for full security)


## Cross-region data residency

Data physically resides in that region's cluster and is never replicated cross-border


## Synchronous vs. Asynchronous Communication#

Microservices rely on inter-service communication for coherence. The two primary communication methods are:

Synchronous Communication: Services interact in real-time, often using HTTP or gRPC. While intuitive, this introduces latency and tight coupling.
Asynchronous Communication: Services communicate via message brokers like Kafka, enabling scalability and fault tolerance. The saga pattern aligns well with asynchronous methods, reducing dependency bottlenecks.


## Fault tolerance

What makes a system fault-tolerant? Although it may seem like magic from the end user experience, developers know that fault tolerance comes from strategic work and smart choices. Here are the main strategies:

- Redundancy: Have backups in place so if one component fails, another can take over.
- Replication: Copy and synchronize data across different nodes so nothing is lost if one goes down.
- Failover Mechanisms: Automatically reroute traffic to healthy instances when something breaks.
- Graceful Degradation: Instead of crashing entirely, the system continues to work with limited functionality until the issue is fixed.

Best Practices for Designing Fault-Tolerant Systems
Every system is different, but some best practices apply across the board. You should always:

- Eliminate single points of failure.
- Use redundancy and replication at multiple levels.
- Implement automatic retries and timeouts.
- Ensure data consistency through smart synchronization.
- Enable graceful recovery so users aren’t affected during failures.
- Monitor everything in real time and set up alerts for fast response.

## SAGA pattern

In usual system we search for ACID transactions (Atomicity, Consistency, Isolation, Durability) ensure that operations either complete entirely or leave the system unchanged. This is feasible in monolithic systems with centralized databases. However, in distributed systems, achieving ACID compliance becomes impractical due to decentralized data, asynchronous communication, and varying failure modes.
A saga is a sequence of distributed transactions where each step updates the system. If a step fails, compensating actions are triggered to revert changes. It’s like booking a vacation and having something go wrong: if the flight reservation fails, hotel bookings and car rentals must be canceled to maintain consistency.

Sagas manage distributed transactions through two main approaches:

Choreography: Decentralized; each service listens for events and independently triggers subsequent actions.
Orchestration: Centralized; a single orchestrator manages the transaction flow, invoking services and handling compensations when needed. For more, read about saga compensating transactions.


https://temporal.io/blog/mastering-saga-patterns-for-distributed-transactions-in-microservices


## Split between deterministic and probabilistic flow



Each agent in a chain introduces probabilistic variability that compounds at every handoff. If each agent operates at 95% accuracy, a 10-step workflow produces overall system reliability of 59.9% — failing roughly four out of ten times. A deterministic rule at the same step would execute at 100% consistency for the same input, with no compounding loss.

The agent interprets context and proposes intent, but the actual execution requires a privileged deterministic boundary — the Decision Intelligence Runtime (DIR) — which sits between agent reasoning and external APIs, maintaining a context store as an immutable record ensuring the runtime holds the single source of truth, while agents operate only on snapshots
https://www.oreilly.com/radar/the-missing-layer-in-agentic-ai/

The agent proposes — in typed JSON — and the orchestration disposes — in deterministic code. The contract between them is enforced by the runtime, not by a prompt, which means it cannot be hallucinated around

## Idempotency


Idempotency means performing an operation once produces exactly the same result as performing it multiple times. The goal is safety under uncertainty: when you can't be sure whether a previous attempt succeeded, you can retry freely without fear of duplicate side effects.
In conventional software, idempotency is primarily a network and infrastructure problem. The uncertainty is physical — did the packet arrive? Did the database commit before the connection dropped? The solution is well-understood: assign a client-generated key to each request, store it server-side, and return the cached result if the same key arrives again. Stripe's payment API is the canonical example. The key derives from stable, known inputs: SHA256(user_id + order_id + amount). The system knows exactly what it tried to do; the only question is whether the attempt landed.

For AI agents the problem is structurally different in three ways.
The agent may not know what it proposed. A conventional system generates a deterministic request and then wonders whether it was received. An agent generates a probabilistic output and then wonders both whether it was received and whether the output was correct. On retry it may produce a subtly different proposal — different phrasing, different parameter — that looks like a new action to downstream systems but represents the same intent. The idempotency key must therefore be derived from canonical action parameters, not from the agent's raw output text.
Context changes between attempts. In conventional software, a $100 payment that fails and retries is still $100. In a trading agent scenario, if you include the market snapshot in the idempotency key, the retry fires at a new price, hashes to a different key, and bypasses the duplicate check entirely. The correct design hashes only the frozen action parameters (instrument, quantity, direction) and explicitly excludes the context snapshot: same intent = same key, regardless of what the world looks like when the retry fires.
The agent may retry because it is confused, not because of a network failure. A conventional client retries because it received no acknowledgment. An agent may retry because it hallucinated that a prior step failed, or because an orchestrator re-invoked it without passing back the prior result. Idempotency must therefore be enforced at the deterministic execution boundary, not inside the agent itself — because the agent cannot be trusted to remember that it already acted.
The practical upshot: for conventional software, idempotency is a courtesy the client extends to the server to enable safe retries. For AI agents, it is a hard constraint the orchestration layer enforces against the agent to prevent real-world actions from firing twice — regardless of what the agent thinks it did.


## Regulation

Under DORA Article 11 (ICT operational resilience), the deterministic orchestration layer must operate independently if the LLM inference service goes down. Hybrid AI architectures that combine LLM flexibility with a deterministic control layer are the only architecture that can guarantee this.

## DSL

## Semantic layer and RAG?