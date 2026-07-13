# Human-in-the-Loop (HITL) & Feedback Loops

Human-in-the-loop is not a safety net bolted onto agents after the fact — it is an architectural primitive that determines the effective autonomy level of the system.

---

## 1. Canonical HITL Patterns

### Pattern 1 — Pre-Execution Approval Gate

The agent proposes an action and pauses. A human reviews the proposal and either approves, modifies, or rejects it before execution.

```
Agent: "I'm about to send this contract amendment to the client.
        Proposed email: [draft]. Approve / Edit / Reject?"
```

**When to use:** Any irreversible or high-impact action — sending external communications, executing financial transactions, deleting records, deploying to production.

**Implementation:** The agent outputs a structured approval request with the proposed action, its justification, and explicit options. The calling layer (workflow engine or UI) routes this to a human interface and suspends the task until a response is received.

**Timeout handling:** Always define a timeout. If no response is received within the window:

- Default approve (for low-risk, time-sensitive operations)
- Default reject (for high-risk operations — safer but may block workflows)
- Escalate to a secondary reviewer

**Claude Code — task and subtask gating:** Claude Code implements the approval gate at the tool-call level. Before executing any action category (file write, bash command, MCP tool call), the agent presents the proposed action and waits for explicit user approval. This maps directly to pre-execution gating at the granularity of individual tool calls.

For multi-step plans, Claude Code exposes the planned task list before execution begins. A human can review the full list of planned tasks and subtasks — removing, reordering, or modifying individual items — before the agent takes any action. In multi-agent setups where an orchestrator spawns subagents, the orchestrator's action plan is gated at the orchestrator level; subagents inherit the permission scope defined for them and cannot escalate beyond it. This makes it possible to gate at three granularities in a single workflow: the full plan, individual task groups, and individual tool calls within each task.

---

### Pattern 2 — Interrupt and Resume

The agent executes autonomously until it encounters an uncertainty, ambiguity, or a threshold breach. It then emits an interrupt signal, suspends state, and resumes from the exact point of interruption once a human responds.

```
Execution:  ──────────────── pause ──────────────── resume ──────────────►
                              │                       ▲
                              ▼                       │
                   Human Review Interface ────────────┘
                   (Agent state preserved during pause)
```

The key difference from Pattern 1 is **mid-execution suspension** — the agent has already done work and needs clarification before continuing, not before starting.

**LangGraph implementation:**

```python
from langgraph.types import interrupt

def review_node(state: State):
    # Agent reaches a decision point mid-execution
    human_decision = interrupt({
        "question": "Found conflicting data. Which source should I trust?",
        "option_a": state["source_internal"],
        "option_b": state["source_external"],
        "context": state["findings_so_far"]
    })
    # Execution resumes here with human_decision populated
    state["trusted_source"] = human_decision
    return state
```

**State preservation requirements:** The agent's full context (variables, tool call history, intermediate results) must be serialized to an external store before suspending. In-memory state is lost on suspension.

**When to use:** Long-running analytical tasks where mid-course correction is cheaper than restarting. Document review, research synthesis, code generation with ambiguous requirements.

---

### Pattern 3 — Asynchronous Review Queue

The agent completes its work and posts the output to a review queue. Execution continues downstream (or a result is returned to the user as "pending review"). A human reviewer processes the queue independently on their own schedule.

```
Agent ──► [Output] ──► Review Queue ──► Human Reviewer ──► Approved Output
                                                          └──► Rejected (→ retry)
```

**Properties:**

- Non-blocking — the agent does not wait for the human
- High throughput — reviewers process batches, not synchronous requests
- Latency — the human review step adds calendar time (hours to days)
- Requires a defined SLA for review turnaround

**When to use:** Content moderation, compliance sign-off on generated documents, financial report approval. Any scenario where human review is mandatory but not time-critical for the immediate user experience.

---

### Pattern 4 — Feedback Injection

A human does not block the agent's current task but injects corrections, preferences, or context that modify the agent's behavior in subsequent steps or future sessions. This is the primary mechanism for iterative improvement without retraining.

```
Run 1:  Agent ──► Output A ──────────────────────────────────────────────────►
                                │
                                ▼
                         Human: "Too formal. Avoid legal jargon."
                                │
Run 2:  Agent ──► Output B ─────┘ (adjusted style)                          ►
```

**Forms of feedback injection:**

| Feedback type | Mechanism | Persistence |
|---|---|---|
| **Preference correction** | Natural language instruction added to memory | Across sessions (long-term memory) |
| **Output correction** | Human edits the output; delta stored as preference | Episodic memory |
| **Explicit rating** | Thumbs up/down, 1–5 scale | Used for offline evaluation pipeline |
| **Structured label** | Human annotates output with category or error type | Training dataset candidate |

**When to use:** Any system where personalization matters, where the agent's defaults are not aligned with the specific user or team, or where quality must improve over time without full retraining.

---

### Pattern 5 — HITL as a Tool

Human expertise is exposed to the agent as a named tool. The agent calls `ask_human(question)` the same way it calls `search_web()` or `query_db()`. The agent decides when human input is needed — it is not pre-programmed at specific checkpoints.

```python
tools = [
    search_web,
    query_database,
    ask_human  # Human expertise as a callable tool
]

# Agent reasoning:
# "This contract clause has an unusual jurisdiction. I don't have
#  enough context to assess risk confidently. I'll call ask_human."
result = ask_human("What is the legal precedent for this clause in French law?")
```

**Strengths:** Maximum flexibility — the agent applies human oversight exactly when it determines it is needed, not at hardcoded checkpoints.
**Weaknesses:** Requires well-calibrated agents. An over-relying agent creates excessive human interruptions. An under-relying agent misses cases where human input was critical.

**Guardrail:** Implement a rate limit on `ask_human` calls per task. An agent calling `ask_human` more than 3 times on a single task likely has a planning or context problem that should be addressed at the architecture level.

---

## 2. When to Trigger HITL: Decision Framework

Not every action requires human review. Over-triggering HITL is a design failure — it creates alert fatigue and negates the value of automation. Under-triggering is a safety failure.

**Two-axis decision matrix:**

| | **Reversible** | **Irreversible** |
|---|---|---|
| **Low impact** | No HITL required | Optional notification |
| **High impact** | Post-execution review | Pre-execution approval gate (mandatory) |

**Trigger conditions (any one sufficient):**

1. **Action is irreversible** — cannot be undone within the system (send email, execute payment, delete data, deploy)
2. **Confidence below threshold** — agent's confidence score on its output < configured threshold (e.g., < 0.75)
3. **Conflict unresolved** — all automated conflict resolution strategies failed to reach consensus
4. **Policy flag** — action type is in the organization's mandatory-review list regardless of confidence
5. **Novelty detected** — task matches no historical precedent in memory; agent is operating outside trained distribution
6. **Cost threshold exceeded** — proposed action exceeds a financial or resource threshold (e.g., >$10k transaction, >1,000 records modified)
7. **Governance gate** — the action requires institutional sign-off to satisfy a compliance, audit, or accountability requirement. This is distinct from condition 4 (policy flag on action type): governance gates apply because *someone must be accountable for the decision*, not because the action is risky by classification. Examples: a regulatory submission that requires a named authorized signatory, a credit decision above a regulatory threshold that mandates human judgment, an AI-generated medical recommendation that must be countersigned by a licensed professional. Governance gates exist to lower the institutional and legal risk of autonomous agent behavior, not only to prevent mistakes — even a highly confident, low-risk action may require a human signature for liability reasons.
8. **Random sampling for behavioral testing** — even when no trigger condition applies, a configured percentage of tasks (e.g., 2–5%) are routed to human review at random. This is not a safety gate for individual tasks but a continuous quality signal: it lets you verify that the agent's autonomous behavior stays within acceptable bounds without reviewing everything. As sampling builds confidence, the autonomy budget expands. This technique allows you to give an agent more autonomy in practice while maintaining a statistical guarantee that deviations will be detected. Track the sample-reviewed tasks in the same evaluation pipeline as flagged tasks — the distribution of reviewer verdicts on sampled tasks is one of the strongest indicators of agent health.

**Calibration guidance:**

```
Trigger too often → Human fatigue → Reviewers approve without reading → System is unsafe
Trigger too rarely → Agent acts autonomously in high-risk cases → Trust erosion after first incident
```

Track the HITL trigger rate as a KPI. A rate above 20% suggests the agent's task scope is too broad or confidence calibration is poor. A rate of 0% on a new deployment is a red flag — the system is likely under-triggering.

---

## 3. Feedback Loops

Feedback loops are how a deployed agent system improves after release. Without them, agent quality degrades as the world changes while the agent stays fixed. With them, the system adapts to user preferences, domain drift, and new failure modes without requiring full retraining.

### 3.1 Online vs. Offline Feedback

| Dimension | **Online (Real-time)** | **Offline (Batch)** |
|---|---|---|
| **Timing** | Immediate — feedback applied during or after a single session | Periodic — feedback collected, analyzed, and applied in batches |
| **Mechanism** | Memory write, prompt adjustment, in-context correction | Evaluation pipeline, dataset curation, fine-tuning, system prompt update |
| **Latency** | Seconds to minutes | Hours to weeks |
| **Scope** | Affects this user or this session | Affects all users after deployment |
| **Risk** | Can introduce inconsistency if feedback is noisy | Requires data volume; slow to respond to sudden domain shifts |
| **Examples** | User corrects agent output → preference stored in memory | Weekly eval run identifies degraded accuracy → prompt is updated |

### 3.2 Memory-Based Adaptation (Online)

The most immediate feedback loop: the agent stores human corrections and preferences in its memory system and retrieves them in future sessions.

```
Session 1:
  Agent output: [too verbose, jargon-heavy]
  Human correction: "Keep reports to 3 bullet points max. No legal terms."
  → Memory write: {user_id: "...", preference: "3 bullets, plain language"}

Session 2:
  Memory retrieval: preference injected into system prompt context
  Agent output: [3 concise bullets, plain language]
```

**Memory architecture for feedback:**

| Memory type | What to store | Retrieval |
|---|---|---|
| **Semantic memory** | Preferences, recurring corrections, domain-specific rules | Embedding similarity at prompt time |
| **Episodic memory** | Past task outcomes with ratings | Retrieved for similar future tasks |
| **Procedural memory** | Corrected workflows and approved templates | Matched by task type |

**Anti-pattern:** Storing every interaction in memory without curation. Memory poisoning — storing incorrect feedback from a confused or adversarial user — can degrade all future sessions. Always tag memory writes with source, confidence, and recency. Implement a memory review process for enterprise deployments.

### 3.3 Evaluation Pipelines (Offline)

The offline feedback loop treats the deployed agent as a system under continuous quality monitoring.

```
Production traffic
        │
        ▼
    Logging layer (all inputs/outputs, tool calls, latency, cost)
        │
        ▼
    Evaluation pipeline (runs on sampled traffic)
    ├── Automated checks (schema, factual grounding, format compliance)
    ├── LLM-as-judge (quality, relevance, hallucination rate)
    └── Human annotation (random sample, flagged cases)
        │
        ▼
    Metrics dashboard (accuracy, latency, cost, HITL trigger rate, failure modes)
        │
        ▼
    Improvement actions:
    ├── System prompt update (most common, fastest)
    ├── Tool update or replacement
    ├── RAG corpus refresh
    ├── Memory rule addition
    └── Fine-tuning (least common, highest cost)
```

**Key metrics to track in the evaluation pipeline:**

| Metric | What it measures | Alert threshold |
|---|---|---|
| **Task completion rate** | % of tasks finished without human escalation or failure | Drop > 5% week-over-week |
| **HITL trigger rate** | % of tasks requiring human intervention | > 20% or < 1% (on new deployments) |
| **Hallucination rate** | % of responses containing factually unsupported claims | > 2% on verified evaluation set |
| **Tool failure rate** | % of tool calls that return errors | > 5% on any individual tool |
| **Context utilization** | Avg tokens used / context window budget | > 85% consistently |
| **Latency P95** | 95th percentile end-to-end response time | Exceeds SLA |
| **Cost per task** | Token cost + tool API cost per completed task | > budget threshold |

### 3.4 How to Use Feedback

Feedback from HITL loops and evaluation pipelines can be applied at different levels of the stack, each with different cost, latency, and scope. The question of whether RLHF, memory tuning, or RL belongs in a production feedback loop depends entirely on the type of error you are correcting and the data you have.

**Decision rule: match the technique to the error type.**

---

#### Tier 1 — No-gradient techniques (immediate, zero training cost)

These address the majority of production issues without touching model weights.

- **System prompt update:** Fix tone, scope, constraints, persona. Applies globally to all users immediately. Best for systematic instruction-following errors ("agent is too verbose", "agent uses deprecated API names").
- **Memory rule addition:** Add a persistent rule or preference to the shared memory store. Available to all sessions that retrieve it. Best for domain-specific corrections that apply across users or recurring preferences within a user.
- **RAG corpus refresh:** Replace outdated documents, add new knowledge, correct factual errors in the retrieval base. Improves grounding without any model change. Best for knowledge drift — when the world changes but the model's training data is stale.
- **Tool update:** Fix a failing API integration, improve a search tool's query formulation, replace a deprecated service. Best for tool-failure errors that show up in evaluation metrics.

**80% of quality issues in production agent systems are addressable at this tier.** Exhaust these options before moving to any gradient-based technique.

---

#### Tier 2 — Memory tuning (in-context learning from corrected episodes)

Memory tuning is a practical middle ground between prompt engineering and fine-tuning. Corrected task episodes — human-edited outputs, preferred completions, annotated failures — are stored in episodic memory and retrieved as few-shot examples at inference time.

```
Human corrects agent output on task T
→ Store: {task_type: T, input: ..., preferred_output: ..., correction_note: ...}

Future task similar to T:
→ Retrieve corrected episode
→ Inject as few-shot example in system context
→ Agent behavior aligns with correction without any weight update
```

**No gradient update is applied.** The model's weights do not change. But its effective behavior changes durably for tasks similar to the corrected episode — as long as the memory store is maintained. This is the right choice when you have sparse but high-quality corrections that are expensive to collect in bulk (e.g., domain expert edits). The limitation: the correction only applies when the episode is retrieved, and retrieval degrades on tasks that are semantically distant from the stored examples.

---

#### Tier 3 — Preference-based fine-tuning (RLHF and DPO)

When enough human feedback has accumulated (typically 1k–10k preference examples), you can train the base model to internalize the corrections as weight updates. Two approaches:

**RLHF (Reinforcement Learning from Human Feedback):**
Collect human rankings or preference judgments on pairs of agent outputs → train a reward model on these preferences → fine-tune the base LLM with PPO to maximize the reward model's score. RLHF is the classic technique (InstructGPT, early Claude training). For deployed agents, this applies when you have a large volume of human ratings and the errors are systematic but too subtle for prompt engineering to fix.

**Cost:** High. Requires a reward model, RL training infrastructure (PPO is unstable and expensive), and careful calibration to avoid reward hacking. Most teams doing "RLHF" in production are actually doing DPO.

**DPO (Direct Preference Optimization):**
Collect preference pairs (output A preferred over output B for task T) → fine-tune directly using a contrastive loss, with no separate reward model. DPO produces comparable alignment quality to RLHF with substantially lower variance and simpler infrastructure. It is the practical default for teams with preference data who need behavioral corrections that cannot be addressed at the prompt layer.

**When RLHF/DPO is warranted:**

- Systematic quality degradation that persists after prompt and memory fixes
- Enough preference data (>1k pairs) collected from real user interactions
- The error pattern is consistent and general, not user-specific
- You have fine-tuning infrastructure and can run evaluation before deployment

**When it is not warranted:** sparse feedback, user-specific preferences (use memory instead), issues that are actually tool or RAG failures (fix the tool/corpus instead).

---

#### Tier 4 — RL from environment reward (no human labels)

For agents that perform verifiable tasks — coding, structured data extraction, math, test execution — the environment itself provides a reward signal without requiring human annotation.

```
Agent generates code → compiler runs → tests pass/fail → reward: +1 / -1
Agent fills form → validator checks schema → reward: pass/fail
Agent makes trade → P&L realized → reward: outcome
```

Pure RL without human labels requires a **ground-truth evaluator**: a function that can score the agent's output automatically. Where one exists, this is often the highest-leverage improvement path because the feedback loop is fast, cheap, and scales with compute rather than human annotator time.

**DeepSeek-R1 (2025)** demonstrated that pure RL without any supervised fine-tuning or human labels can produce o1-level reasoning capability — emergent self-verification and strategy adaptation through environment reward alone. For tool-using agents on verifiable tasks, this pattern is increasingly practical.

**Limitations:** RL alone does not work for subjective tasks (writing quality, tone, helpfulness judgments). It also risks reward hacking — the agent learns to maximize the reward signal in ways that do not correspond to genuine task quality. Always evaluate RL-trained agents on held-out benchmarks before deployment, and pair with a human sampling program (see trigger condition 8) to catch distributional drift post-deployment.

---

**Summary: when to use which technique**

| Error type | Technique |
|---|---|
| Instruction-following, tone, scope | System prompt update |
| User-specific or domain-specific preference | Memory write |
| Factual / knowledge staleness | RAG corpus refresh |
| Broken tool or API | Tool update |
| Sparse high-quality corrections | Memory tuning (episodic retrieval) |
| Systematic behavioral error, enough labeled data | DPO |
| Requires nuanced human ranking at scale | RLHF |
| Verifiable task with automatable success criterion | RL from environment reward |

---

## References

- [LangGraph — Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [Anthropic — Building Effective AI Agents](https://resources.anthropic.com/building-effective-ai-agents)
- [OWASP — Top 10 for Agentic Applications 2026](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [DPO — Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [DeepSeek-R1 — Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948)
