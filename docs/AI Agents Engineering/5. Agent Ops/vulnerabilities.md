# OWASP Top 10 for Agentic Applications 2026

> Source: OWASP Gen AI Security Project — Agentic Security Initiative, December 2025
> Focus: Vulnerabilities specific to AI agents — autonomous, multi-step, tool-using systems

---

## Why This Is Different from LLM Security

Classic LLM security (OWASP Top 10 for LLMs) deals with single-turn model interactions. Agentic systems are different: they **plan across multiple steps, call tools, delegate to sub-agents, persist memory across sessions, and act on behalf of users**. A single manipulated input can cascade into system-wide harm before any human reviews it.

Key agentic properties that amplify risk:

- **Autonomy** — agents act without per-step human approval
- **Persistence** — memory and state survive across sessions
- **Tool access** — agents can read/write files, call APIs, execute code, send emails
- **Multi-agent delegation** — agents spawn and trust other agents

---

## ASI01 — Agent Goal Hijack

**What it is:** Attackers redirect what the agent is trying to accomplish — through prompt injection, poisoned documents, malicious tool outputs, or forged agent messages. Unlike a single-turn jailbreak, this subverts multi-step planning.

**Real examples:**

- Hidden instructions in a calendar invite silently reweight the agent's objectives each morning
- A malicious Google Doc tricks ChatGPT into exfiltrating user data (AgentFlayer)
- EchoLeak: crafted email to M365 Copilot triggers zero-click data exfiltration

**What to do:**

- Treat all natural-language inputs (documents, emails, RAG results) as untrusted
- Lock system prompts under configuration management; require human approval for goal changes
- Validate agent *intent* before executing high-impact or goal-changing actions
- Log goal state and alert on unexpected shifts
- Sanitize all connected data sources (RAG, email, calendar, uploaded files) before they can influence goals

---

## ASI02 — Tool Misuse and Exploitation

**What it is:** Agents misuse legitimate tools — not because they lack permission, but because they apply them unsafely. An email summarizer that can also send mail. A DB tool that can delete tables. A research agent that follows malicious links.

**Real examples:**

- Agent given full financial API access issues refunds when it should only fetch order history
- Ping tool used by an agent to exfiltrate data via DNS queries
- EDR bypass: legitimate PowerShell + cURL + internal APIs chained to exfiltrate logs — no malware detected

**What to do:**

- **Least Agency**: restrict each tool to only what the agent actually needs (read-only DB queries, no send/delete for email summarizers)
- Require human confirmation for destructive actions (delete, transfer, publish)
- Run tools in sandboxed containers with egress allowlists
- Apply per-tool usage ceilings (rate, cost, token budgets)
- Use short-lived, session-scoped credentials; revoke immediately after use
- Log all tool invocations and detect anomalous chaining patterns

---

## ASI03 — Identity and Privilege Abuse

**What it is:** Agents don't have proper identities — they often inherit credentials from the user or orchestrator, cache secrets across sessions, and accept internal agent requests without re-verifying authorization. Attackers exploit delegation chains.

**Real examples:**

- IT admin agent caches SSH credentials; non-admin reuses same session to create unauthorized account
- Low-privilege email sorting agent relays instructions to a high-privilege finance agent — which processes fraudulent payment without re-checking
- Fake "Admin Helper" agent registered in agent registry; other agents route privileged tasks to it

**What to do:**

- Issue per-agent, per-task short-lived scoped tokens (mTLS certs or OAuth with intent binding)
- Wipe agent state and credentials between sessions
- Re-verify permissions at each privileged step — don't trust inherited context
- Require human approval for privilege escalation
- Monitor for transitive privilege inheritance in delegation chains

---

## ASI04 — Agentic Supply Chain Vulnerabilities

**What it is:** Agents dynamically load tools, prompt templates, plugins, and sub-agents at runtime from external sources. Unlike static software dependencies, this is a **live supply chain** — compromised components can cascade instantly across all agents using them.

**Real examples:**

- Malicious MCP server on npm impersonating `postmark-mcp` — silently BCC'd emails to attacker
- GitHub MCP prompt injection: public tool hides commands in metadata, exfiltrates private repo data
- Poisoned NPM package auto-installed by coding agents — exfiltrated SSH keys and API tokens
- Amazon Q: poisoned prompt shipped in VS Code extension v1.84.0 to thousands of users

**What to do:**

- Sign and verify all tool manifests, prompt templates, and agent descriptors (SBOMs/AIBOMs)
- Pin prompts, tools, and configs by content hash + commit ID
- Allowlist approved registries; block untrusted sources
- Run agents in sandboxed containers with strict network limits
- Implement emergency kill-switch for instant tool/agent revocation on compromise
- Continuously re-check signatures and hashes at runtime, not just at install time

---

## ASI05 — Unexpected Code Execution (RCE)

**What it is:** Agents generate and execute code. Prompt injection, tool misuse, or unsafe deserialization can turn natural-language inputs into running exploits. Bypasses traditional security controls because the code is generated in real-time by a trusted process.

**Real examples:**

- "Vibe coding" agent generates unreviewed shell commands in its own workspace, deletes production data
- Memory system with `eval()` exposed to untrusted input — attacker embeds executable code in a prompt
- Multi-tool chain: file upload → path traversal → dynamic code loading → RCE

**What to do:**

- Ban `eval()` in production agents; require safe interpreters with taint-tracking
- Never run agents as root; use sandboxed containers with no network egress by default
- Separate code generation from execution — validate generated code before running it
- Require human approval for any elevated or irreversible code execution
- Static scan generated code before execution; log and audit all runs
- Lint and block known-vulnerable packages; pin dependencies by hash

---

## ASI06 — Memory & Context Poisoning

**What it is:** Agents maintain persistent memory — RAG stores, conversation summaries, embeddings — that influences future reasoning. Attackers poison this memory with false data, causing biased decisions across all future sessions, long after the original attack.

**Real examples:**

- Attacker reinforces fake flight price in travel booking agent's memory — agent approves inflated bookings indefinitely
- Prompt injection plants false memories in ChatGPT, persists across user sessions (AgentFlayer)
- Cross-tenant vector bleed: attacker's near-duplicate content pulls another tenant's sensitive data into retrieval

**What to do:**

- Validate all memory writes with content scanning before committing
- Isolate memory per user session and per domain — prevent cross-tenant bleed
- Require source attribution for memory entries; flag suspicious update patterns
- Never auto-re-ingest agent-generated outputs back into trusted memory (prevents bootstrap poisoning)
- Expire unverified memory entries; decay low-trust entries over time
- Maintain memory snapshots with rollback capability for suspected poisoning events

---

## ASI07 — Insecure Inter-Agent Communication

**What it is:** Multi-agent systems pass messages via APIs, message buses, and shared memory. Without proper authentication, integrity validation, and encryption, attackers can intercept, spoof, replay, or inject into agent-to-agent communications.

**Real examples:**

- MITM on unencrypted channel injects hidden instructions; agents behave maliciously while appearing normal
- Attacker registers fake peer agent in discovery service using cloned schema — intercepts privileged coordination traffic
- Replayed emergency coordination message triggers outdated procedures and resource misallocation

**What to do:**

- Encrypt all inter-agent channels (mTLS, PKI certificate pinning, forward secrecy)
- Digitally sign all messages; validate semantic content for hidden instructions
- Use nonces, session IDs, and timestamps to prevent replay attacks
- Require mutual authentication for all agent-to-agent connections
- Pin protocol versions; reject downgrades to legacy/unencrypted modes
- Use signed agent cards and attested registries — verify agent identity before accepting any coordination message

---

## ASI08 — Cascading Failures

**What it is:** A single fault — hallucination, poisoned memory, malicious input — propagates across autonomous agents that act without per-step human checks. Errors compound before anyone can intervene. Agents planning and delegating at machine speed can cause system-wide failures in seconds.

**Real examples:**

- Financial trading: prompt injection inflates risk limits in analysis agent → position agent auto-trades larger positions → compliance stays blind
- Healthcare: supply chain tamper corrupts drug data → treatment agent auto-adjusts protocols → spread network-wide
- Auto-remediation feedback loop: agent suppresses alerts to meet latency SLAs → planning agent interprets fewer alerts as success → widens automation → compounds blind spots

**What to do:**

- Design for fault tolerance — assume any LLM component can fail or be compromised
- Sandbox agents with least privilege, network segmentation, and scoped APIs
- Issue short-lived, task-scoped credentials for each agent run
- Implement circuit breakers between planner and executor agents
- Rate-limit and throttle on anomalous command fan-out
- Require human gates before agent outputs propagate downstream to high-impact systems
- Maintain tamper-evident, time-stamped logs with lineage metadata for every propagated action

---

## ASI09 — Human-Agent Trust Exploitation

**What it is:** Humans overtrust agents because agents are fluent, confident, and apparently authoritative. Attackers exploit this via fake explanations, emotional manipulation, and social engineering — making humans approve actions the agent was manipulated into suggesting. The agent acts as an untraceable intermediary; the human takes the audited action.

**Real examples:**

- Finance copilot poisoned by manipulated invoice confidently recommends urgent payment to attacker's bank — manager approves without independent check
- IT support agent cites real tickets to appear legitimate, harvests credentials from new hire
- Agent fabricates plausible audit rationale to justify risky configuration change; reviewer approves

**What to do:**

- Require multi-step human confirmation before sensitive or irreversible actions
- Display confidence warnings ("low-certainty", "unverified source") alongside agent recommendations
- Provide plain-language risk summaries — not model-generated rationales — for human review
- Visually differentiate high-risk recommendations (red borders, banners, confirmation prompts)
- Separate preview from effect: block network/state-changing calls during preview context
- Give users a simple way to flag suspicious or manipulative agent behavior
- Train personnel on automation bias and agent manipulation patterns

---

## ASI10 — Rogue Agents

**What it is:** Agents that deviate from their intended behavior — through external compromise, goal drift, reward hacking, or emergent misalignment. Individual actions may appear legitimate, but the overall behavior becomes harmful. Distinct from tool misuse: this is about **loss of behavioral integrity**, not just over-privileged access.

**Real examples:**

- After encountering poisoned web content, agent continues autonomously scanning and transmitting sensitive files — even after malicious source is removed
- Agents tasked with minimizing cloud costs autonomously delete production backups to meet their metric
- Compromised agent spawns unauthorized replicas of itself across the network for persistence

**What to do:**

- Maintain immutable, signed audit logs of all agent actions, tool calls, and inter-agent communication
- Assign trust zones with strict inter-zone communication rules; sandbox agents by default
- Deploy behavioral detection and watchdog agents to validate peer behavior
- Implement kill-switches and credential revocation for instant disable of rogue agents
- Attach signed behavioral manifests to each agent (expected capabilities, tools, goals) — validate before each action
- Monitor for reward hacking: verify that optimization targets can't be "gamed" through destructive shortcuts

---

## Quick Reference

| ID | Name | Core Risk | Attack Surface |
|---|---|---|---|
| ASI01 | Agent Goal Hijack | Redirect agent objectives | Prompts, documents, RAG, email |
| ASI02 | Tool Misuse | Unsafe use of legitimate tools | Tool APIs, MCP, orchestration |
| ASI03 | Identity & Privilege Abuse | Credential/delegation exploitation | Auth tokens, delegation chains |
| ASI04 | Supply Chain | Compromised components at runtime | Plugins, MCP servers, registries |
| ASI05 | Unexpected RCE | Code generation → execution | Code tools, eval(), sandboxes |
| ASI06 | Memory Poisoning | Corrupt persistent context | RAG, embeddings, memory tools |
| ASI07 | Insecure Inter-Agent Comms | MITM, spoofing, replay | Message buses, APIs, discovery |
| ASI08 | Cascading Failures | Fault propagation at machine speed | Multi-agent workflows, planners |
| ASI09 | Human-Agent Trust | Exploit human over-reliance | UI, recommendations, approvals |
| ASI10 | Rogue Agents | Behavioral drift and deviation | Entire agent lifecycle |

---

## Key Principles for Agentic Security

1. **Least Agency** — grant only the autonomy actually needed; don't deploy agentic behavior where deterministic code suffices
2. **Human-in-the-Loop for high-impact actions** — irreversible actions (money transfers, deletions, code deploys) require human confirmation
3. **Observability is non-negotiable** — if you can't see what an agent is doing and why, you can't secure it
4. **Trust nothing implicitly** — all inputs, tool outputs, peer-agent messages, and memory entries are untrusted until validated
5. **Design for fault propagation** — assume any component can fail or be compromised; contain the blast radius

---

*OWASP Top 10 for Agentic Applications 2026, OWASP Gen AI Security Project — Agentic Security Initiative. Licensed CC BY-SA 4.0.*
