# Agent Taxonomy, Granularity & Identity

Three foundational engineering decisions that every enterprise AI deployment must address before writing a single line of agent code. They are closely related: how you classify agents determines their scope, their scope determines their identity requirements, and identity must be tied to a verifiable, auditable credential.

---

## 1. Agent Taxonomy

Taxonomy is the act of classifying agents into meaningful categories. Without a shared classification system, organizations cannot reason about what they have deployed, who is responsible for it, or what governance applies to it.

### 1.1 Classification by Function

The most practical taxonomy for enterprise teams. What does the agent *do*?

| Type | Role | Typical scope |
|---|---|---|
| **Orchestrator / Supervisor** | Decomposes goals, delegates to other agents, synthesizes results | Cross-domain, workflow-level |
| **Worker / Specialist** | Executes a narrow domain task delegated by an orchestrator | Single domain, task-level |
| **Router / Classifier** | Classifies intent and routes requests to the right agent or workflow | Entry point, no domain logic |
| **Evaluator** | Reviews, scores, or validates the output of another agent | Quality gate, compliance checker |
| **Memory Agent** | Manages retrieval, storage, and maintenance of shared knowledge | Infrastructure layer |

### 1.2 Classification by Autonomy Level

The Knight First Amendment Institute's five-level framework (2025) classifies agents by how much independent decision-making they exercise:

| Level | Human role | Agent behavior | Enterprise examples |
|---|---|---|---|
| **L1 — Assisted** | Operator — human executes every action; agent provides suggestions only | Recommends, does not act | Copilot suggesting email replies |
| **L2 — Collaborative** | Collaborator — human and agent share decision-making | Acts on defined task types; refers novel situations | Code review agent auto-approving style fixes; flagging logic changes |
| **L3 — Supervised** | Consultant — agent acts; human reviews before irreversible effects | Executes freely in reversible domain; pauses for approval on consequential actions | Expense processing agent that files < $500 automatically, escalates above |
| **L4 — Delegated** | Approver — human sets policy; agent acts within it independently | Full autonomy within a defined authorization envelope | Contract agent that signs within pre-approved templates, no signature above |
| **L5 — Autonomous** | Observer — human sets high-level objectives only | Fully independent; self-directs to achieve goal | Research agents with sustained multi-day autonomous operation |

!!! warning "L5 in enterprise"
    Fully autonomous agents (L5) carry the highest liability and the most complex governance requirements. Anthropic's internal research uses supervised parallel sub-agents, not fully autonomous single agents, even for complex open-ended research tasks. Default to L3 or L4 in regulated environments.

### 1.3 Classification by Scope

| Scope | Boundary | Characteristics |
|---|---|---|
| **Task-scoped** | One specific function (e.g., extract invoice fields) | Short-lived, stateless, narrow tool access |
| **Domain-scoped** | One business domain (e.g., all legal review workflows) | Medium-lived, domain-specific memory, scoped data access |
| **Process-scoped** | One end-to-end business process (e.g., full customer onboarding) | Long-lived, cross-system access, stateful |
| **Enterprise-scoped** | Broad organizational mandate (e.g., internal assistant for all employees) | Persistent, multi-domain tool access, requires strongest governance |

---

## 2. Agent Granularity Principles

Granularity defines how much responsibility a single agent carries. Getting this wrong is the most common cause of poor agent performance and unmanageable systems.

### 2.1 The Single Responsibility Principle

Borrowed from software engineering: **each agent should do exactly one thing well.**

> "Every agent has one clearly defined responsibility, a defined input/output contract typed against a JSON Schema, and a clear handoff boundary."

A Salesforce (2025) benchmark found:

- Generic single-prompt agents succeeded **58%** of the time
- Performance dropped to **35%** in multi-prompt scenarios requiring maintained context across domains

Specialist agents with narrow scope consistently outperform generalist agents at scale.

### 2.2 Signs an Agent Is Too Broad

| Symptom | Root cause | Effect |
|---|---|---|
| Agent uses the wrong tool for a task | Too many tools with overlapping descriptions (> 15–20) | Routing accuracy degrades non-linearly |
| Agent loses track of earlier decisions mid-task | Context window fills with irrelevant information | Context overflow → instruction-following failures |
| Agent behaves inconsistently across similar requests | Too many domains competing for attention | Non-deterministic routing |
| Hard to debug which step failed | Too much work in one agent | No clear accountability boundary |
| Cannot upgrade one capability without risking others | Tightly coupled responsibilities | Brittle architecture |

### 2.3 Signs an Agent Is Too Narrow

| Symptom | Root cause | Effect |
|---|---|---|
| Trivial tasks spawn 5+ agent calls | Sub-agents created for single tool calls | 10–15× unnecessary token cost |
| Orchestrator overwhelmed managing dozens of micro-agents | Over-decomposition | Coordination overhead exceeds value |
| Context reconstructed from scratch on every step | No shared state between over-split agents | Compounding cost; coherence failures |

### 2.4 Granularity Decision Rules

**Define the agent by its output contract, not by its inputs.**

A well-scoped agent can answer: *"What structured artifact do I produce, and what is the precise set of inputs I consume?"*

```
✓ CORRECT scope: "I receive a raw contract PDF and produce a
  structured risk-assessment JSON with five defined fields."

✗ TOO BROAD: "I handle all legal and compliance tasks
  for the organization."

✗ TOO NARROW: "I extract the date from a contract."
  (this is a tool, not an agent)
```

**Tool vs. agent boundary:**

- If the work requires a single API call → it is a **tool**
- If the work requires multiple tool calls, intermediate reasoning, and produces a structured result → it is an **agent**

**The 10-tool ceiling:** An agent with more than 10 tools begins to show measurable routing degradation. When you reach this limit, split the agent rather than expanding the tool list.

**Granularity and governance are linked:** Narrower agents have narrower tool access, which reduces blast radius if the agent is compromised or behaves unexpectedly. Organizations using tiered, granular authorization models experience **76% fewer agent safety incidents** than those using binary autonomous/non-autonomous models.

### 2.5 Scaling Granularity with Autonomy

```
Low autonomy (L1–L2) → Broader agents acceptable
  ↓ Human oversight compensates for imprecise scoping

High autonomy (L4–L5) → Narrow agents required
  ↓ No human backstop; scope must be the primary safety control
```

As autonomy increases, granularity must tighten proportionally. The combination of broad scope + high autonomy is the highest-risk configuration in enterprise agent architecture.

---

## 3. Agent Identity

### 3.1 What Is Agent Identity and Why It Matters

Agent identity is the cryptographically verifiable proof of *who* or *what* is acting in a system. It answers: "When this action was taken against our database, what entity took it — a human, a service account, or an AI agent — and can we verify that claim?"

Without distinct agent identity:

- You cannot separate actions taken by AI agents from actions taken by employees or traditional software in your audit logs
- You cannot apply differentiated security policies to agent traffic
- You cannot detect when an agent is acting outside its authorized scope
- You cannot attribute a harmful action to the specific agent that caused it

**The scale problem:** As of Q1 2026, Non-Human Identities (NHIs) — agents, service accounts, bots — outnumber human employees at a ratio of **144:1** in large enterprises. Traditional IAM systems designed for human users and long-lived application identities cannot manage this volume or the dynamic lifecycle of modern agents.

### 3.2 Agent Identity vs. Other Identity Types

| Dimension | Human identity | Application/service identity | Agent identity |
|---|---|---|---|
| **Bound to** | A named person | A long-lived application | A specific agent runtime instance |
| **Lifetime** | Employment duration | Application lifecycle (months–years) | Task duration (seconds to days) |
| **Authentication** | Passwords, MFA, passkeys | API keys, client secrets, certificates | Federated credentials, SPIFFE SVIDs, X.509 certificates |
| **Creation rate** | Low (HR-driven) | Low-medium (release-driven) | High — agents created and destroyed thousands of times per day |
| **Can impersonate user** | N/A — is the user | Explicitly prohibited | Controlled delegation only, with scope attenuation |
| **Enterprise product** | Microsoft Entra ID, Okta | Service principals, IAM roles | Microsoft Entra Agent ID, Google Agent Identity, AWS Bedrock task roles |

### 3.3 Core Identity Properties Every Agent Must Have

**Unique identifier:** A globally unique, stable ID assigned at agent creation. Not a name, not a role — a machine-readable identifier that persists for the agent's lifetime and is referenced in every audit log entry.

**Cryptographic credential:** The agent authenticates using short-lived cryptographic proofs (X.509 certificates, SPIFFE SVIDs, JWT tokens), not long-lived secrets like API keys or passwords. Short-lived credentials expire automatically; compromised credentials are self-healing.

**Defined owner / sponsor:** Every agent identity is linked to a human accountable for its behavior — the person or team who deployed it. This is the "sponsor" in Microsoft's model. If the agent causes harm, the sponsor is accountable.

**Scoped permissions:** The agent's identity carries its authorization scope — what systems it can access and with what rights. Permissions are attached to the identity, not embedded in the agent's prompt.

**Audit trail linkage:** Every action the agent takes is logged against its unique identifier, creating an immutable record that distinguishes agent actions from human actions.

### 3.4 How Big Tech Implements Agent Identity

#### Microsoft — Entra Agent ID

Microsoft launched **Entra Agent ID** (Preview, 2025) as a first-class identity construct purpose-built for AI agents, separate from user identities and service principals.

**Key design decisions:**

- Agents do not use passwords, SMS, or MFA — they authenticate exclusively via **Federated Identity Credentials (FIC)** issued by an agent blueprint
- **Agent Blueprint:** A template that holds the actual credentials and mints short-lived tokens on behalf of agent instances. Separates credential management from agent runtime
- **Agent Registry:** A centralized metadata repository giving a unified view of all agents deployed in the tenant — who created them, what they access, their sponsor, and their activity log
- **Agents' User Account:** For scenarios where backward compatibility requires a human-like identity (e.g., a legacy system that only accepts user tokens), an agent can be paired with a special user account in a 1:1 relationship — but it remains clearly tagged as AI-driven in audit logs
- **Lifecycle design:** Built for ephemerality — agents created in bulk, policies applied consistently, retired without orphaned credentials

**Delegation model:** Agents can act with:

1. **Autonomous access** — permissions granted directly to the agent identity (no human in the loop)
2. **Delegated access** — acting on behalf of a specific human user, using only the rights that user has granted

#### Google Cloud — Agent Identity

Google Cloud launched **Agent Identity** (2025) built on the **SPIFFE standard**, providing cryptographic identity for agentic workloads.

**Key design decisions:**

- Every agent receives a unique **SPIFFE ID** — a URI of the form `spiffe://agents.global.org-123456789012.system.id.goog/resources/aiplatform/...`
- **X.509 certificates** auto-provisioned at deployment, **24-hour validity**, auto-renewed — no developer intervention required
- Identities are **not shared** between agents by default, **cannot be impersonated**, and **do not permit long-lived key generation** (unlike service accounts)
- **Token binding:** Access tokens are cryptographically bound to the agent's X.509 certificate — stolen tokens cannot be used from a different context
- **Agent Identity Auth Manager:** A credential vault for third-party tool access — agents retrieve credentials for external APIs from the auth manager at call time, never storing them at rest
- Full **IAM integration**: SPIFFE IDs appear as principals in IAM allow/deny policies and Principal Access Boundary policies
- **Context-Aware Access:** Default policies enforce certificate binding and mutual TLS for all agent traffic

#### AWS — Bedrock AgentCore Task Roles

AWS manages agent identity through **IAM roles** attached to Bedrock agent runtimes, following the same workload identity model used for Lambda and ECS tasks.

**Key design decisions:**

- Each Bedrock agent is assigned an **IAM execution role** scoped to the specific tools and data sources it needs
- **Task-scoped credentials:** Agent receives short-lived credentials (STS tokens) at invocation time, not persistent API keys
- **Resource-based policies** on tools (S3 buckets, DynamoDB tables, Lambda functions) explicitly authorize which agent roles may call them — defense in depth
- **VPC isolation:** Agents can be deployed in private VPCs with PrivateLink, so no traffic traverses the public internet
- **Bedrock AgentCore Identity:** Supports environment-agnostic workload identities that can simultaneously hold multiple authentication credentials for different downstream systems
- **CloudTrail integration:** All agent actions logged with the agent's IAM role ARN as the principal — auditors can distinguish agent traffic from human or service traffic

---

## 4. Identity Delegation

Delegation is how an agent acts **on behalf of** a human or another system, using authority that was temporarily granted to it — not authority it permanently owns.

### 4.1 Delegation vs. Impersonation

This distinction is critical for security and compliance:

| | **Delegation** | **Impersonation** |
|---|---|---|
| **Definition** | Agent acts with a subset of a user's permissions, clearly marked as agent-originated | Agent acts as if it were the user — indistinguishable from the user in audit logs |
| **Audit trail** | Shows: "Agent X acted on behalf of User Y with scope Z" | Shows: "User Y did this" — the agent is invisible |
| **Permission scope** | Always ≤ what the delegating user holds; can be further restricted | Equals the full user permission set |
| **Revocability** | User can revoke the delegation at any time | User cannot easily revoke; they have given away their identity |
| **Enterprise recommendation** | Required practice | Prohibited practice |

> *"The OpenID Foundation's October 2025 whitepaper identifies user impersonation by agents as the industry's most urgent unsolved problem — and proposes it must be replaced everywhere by delegated authority."*

### 4.2 The Delegation Chain

In multi-agent systems, delegation flows through a chain. Each hop must:

1. Carry the full chain history (who authorized what, at what scope)
2. Attenuate scope — a sub-agent can only receive permissions the delegating agent already holds
3. Be cryptographically verifiable at each step

```
Human User (full permissions)
    │
    │ grants scope: [read:contracts, write:drafts]
    ▼
Orchestrator Agent (identity: agent-001)
    │
    │ sub-delegates scope: [read:contracts]  ← can only give what it has
    ▼
Legal Review Sub-Agent (identity: agent-002)
    │
    │ calls Legal DB API
    │ token carries: {user: "alice", delegated_via: "agent-001", scope: "read:contracts"}
    ▼
Legal Database (verifies full chain before granting access)
```

**Key rule:** The sub-agent's effective permissions = `min(delegator's permissions, sub-agent's own granted permissions)`. Scope can only narrow at each hop, never expand.

### 4.3 Technical Standards for Delegation

**OAuth 2.0 On-Behalf-Of (RFC 8693 Token Exchange)**
The current standard for agent delegation in enterprise environments. The orchestrator presents its own access token and exchanges it for a new token scoped to a sub-agent's needs. The resulting token carries the identity of the original authorizing user plus the delegation chain.

**WIMSE (Workload Identity in Multi-Service Environments)**
An IETF working group (active 2025) standardizing how workload identities — including AI agents — authenticate and delegate across service boundaries. Defines the intersection of SPIFFE (cryptographic identity) and OAuth 2.0 (authorization delegation).

**SPIFFE SVIDs (SPIFFE Verifiable Identity Documents)**
Cryptographic identity documents (X.509 certificates or JWT tokens) issued to each agent at runtime by a SPIRE server. SVIDs carry the agent's SPIFFE ID and are short-lived (hours). When combined with OAuth token exchange, they provide both *who the agent is* (SVID) and *what it is permitted to do* (OAuth scope).

**OIDC-A (OpenID Connect for Agents)**
A proposed extension (2025) to OpenID Connect 1.0 that adds agent-specific claims to identity tokens: agent type, autonomy level, delegating principal, tool access scope, and session lineage. Enables resource servers to make fine-grained authorization decisions based on the full agentic context.

### 4.4 Enterprise Delegation Recommendations

**Never grant delegation authority by default.** An agent that can re-delegate its permissions to sub-agents is a privilege escalation vulnerability. Delegation authority must be explicitly granted and scoped to a maximum depth.

**Minimum delegation scope.** When an orchestrator delegates to a sub-agent, it should pass the minimum permissions the sub-agent needs for its specific subtask — not its own full scope.

**Time-bound delegations.** All delegated credentials should carry an explicit expiry. The agent's ability to act on behalf of a user expires when the task ends or at a maximum TTL, whichever is shorter.

**Preserve the chain in audit logs.** Every privileged action must log the full delegation lineage: initiating human → orchestrator → sub-agent → action → resource. This is the only way to reconstruct causality during an incident.

**Validate the chain at the resource.** The resource server (database, API, service) should verify the full delegation chain before granting access — not just check that the caller has a valid token.

---

## 5. The OWASP Top Identity Risks for Agents

The **OWASP Top 10 for Agentic Applications (2026)** identifies identity as the most concentrated risk surface in multi-agent systems.

### ASI03 — Identity and Privilege Abuse

An agent's effective identity is the union of everything it can touch: its own credentials, every delegated permission, every cached token, and every MCP server scope it holds. A single compromised agent gives an attacker all of those simultaneously.

**Attack vectors:**

- Exploiting inherited or cached credentials from a previous session
- Forging agent-to-agent messages (which often lack authentication) to impersonate a trusted orchestrator
- Injecting into the delegation chain to escalate a sub-agent's permissions
- Exploiting long-lived secrets stored in agent memory or system prompts

**Mitigations:**

- Short-lived credentials only (hours, not months)
- Cryptographic verification at every agent-to-agent boundary
- Separate identity per agent, per deployment, per task (no shared credentials)
- Immutable audit trail that captures the full delegation lineage
- Regular rotation and revocation of agent credentials as part of the agent lifecycle

### The Prompt Injection → Privilege Escalation Chain

The most dangerous attack path in multi-agent enterprise systems:

```
1. Attacker embeds malicious instruction in a document the agent will read
2. Agent reads document → follows injected instruction
3. Injected instruction tells agent to spawn a sub-agent with elevated permissions
4. Sub-agent (now attacker-controlled) calls production APIs with agent's full scope
5. No human in the loop; audit log shows "agent action" not "attacker action"
```

**Prevention:** Prompt injection defense + bounded delegation authority + human review for any agent that requests permission expansion.

---

## References

- [Microsoft — What are agent identities? (Entra Agent ID)](https://learn.microsoft.com/en-us/entra/agent-id/identity-platform/what-is-agent-id)
- [Microsoft — Announcing Microsoft Entra Agent ID](https://techcommunity.microsoft.com/blog/microsoft-entra-blog/announcing-microsoft-entra-agent-id-secure-and-manage-your-ai-agents/3827392)
- [Microsoft — Three tiers of Agentic AI](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/three-tiers-of-agentic-ai---and-when-to-use-none-of-them/4510377)
- [Microsoft — Perspective on agentic identity standards](https://techcommunity.microsoft.com/blog/microsoft-entra-blog/microsoft%E2%80%99s-perspective-on-agentic-identity-standards/2111910)
- [Google Cloud — Agent Identity overview](https://docs.cloud.google.com/iam/docs/agent-identity-overview)
- [AWS — Amazon Bedrock AgentCore](https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-agentcore-is-now-generally-available/)
- [OpenID Foundation — Identity Management for Agentic AI (whitepaper)](https://openid.net/wp-content/uploads/2025/10/Identity-Management-for-Agentic-AI.pdf)
- [OWASP — Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/)
- [Knight First Amendment Institute — Levels of Autonomy for AI Agents](https://knightcolumbia.org/content/levels-of-autonomy-for-ai-agents-1)
- [HashiCorp — SPIFFE: Securing the identity of agentic AI](https://www.hashicorp.com/en/blog/spiffe-securing-the-identity-of-agentic-ai-and-non-human-actors)
- [IETF WIMSE Working Group](https://datatracker.ietf.org/wg/wimse/about/)
- [CoSAI — Technical patterns for agent authentication and authorization](https://riptides.io/blog-post/spiffe-meets-oauth2-current-landscape-for-secure-workload-identity-in-the-agentic-ai-era/)
- [Aembit — Why Traditional IAM Is No Match for Agentic AI](https://aembit.io/blog/iam-agentic-ai/)
