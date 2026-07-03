## Table of Contents

1. [Introduction](#1-introduction)
2. [Why evaluation is critical](#2-why-evaluation-is-critical)
3. [Anatomy of an agent](#3-anatomy-of-an-agent)
4. [What are we trying to evaluate?](#4-what-are-we-trying-to-evaluate)
5. [How we evaluate](#5-how-we-evaluate)

## 1. Introduction

### The problem

An AI agent is not a classic microservice. It's a **non-deterministic**, **multi-step** system that orchestrates LLM calls, tools, memory, and potentially other agents.

Entities develop agents via an SDK. The goal is that this development is accompanied by good AgentOps practices, notably:

- Evaluation driven best practices sharing
- A common **evaluation standard**, integrated into the agent lifecycle
- A **measured and automated** dev → staging → production transition (gates, CI/CD)
- **Proactive detection** of regressions in production (not via user complaints)
- Systematic **experiment tracking** to compare versions and make data-driven decisions

### The objective

Provide a **complete and actionable blueprint** to:

1. Evaluate an agent at each stage of its lifecycle
2. Automate this evaluation in a CI/CD pipeline
3. Monitor and maintain an agent in production
4. [To be defined] Build a complementary evaluation library to an SDK
5. Manage the multi-framework reality (Langfuse, MLflow, LangSmith...) of the group

```
┌────────────────────────────────────────────────────────────────┐
│                      LIFECYCLE OVERVIEW                        │
│                                                                │
│  ┌───────┐    ┌─────────┐    ┌──────────────────────────────┐  │
│  │  DEV  │───▶│ STAGING │───▶│         PRODUCTION           │  │
│  └───┬───┘    └────┬────┘    └──────────────┬───────────────┘  │
│      │             │                        │                  │
│  Eval offline  Eval offline          Eval online (continuous)  │
│  Exp tracking  QA datasets           Feedback loop             │
│  Unit tests    Gate checks           Batch eval (scheduled)    │
│  Dataset CI    Promotion gate        Alerting & drift detect.  │
│                                      Registry (versions)       │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Why evaluation is critical

### 2.1 AI agents: inherently non-deterministic systems

An AI agent is an autonomous system that orchestrates LLM calls, tools, memory, and potentially other agents. Its fundamental characteristics make evaluation indispensable:

- **Non-deterministic**: the same input can produce different outputs on each execution (LLM temperature, memory state, tool results)
- **Multi-step**: the agent takes a chain of decisions (routing, tool selection, parameters, synthesis) where each can be a source of error
- **Difficult to test**: classic unit tests only cover a fraction of behavior; regressions are subtle and often silent
- **Dependent on external components**: LLMs (whose versions change without notice), third-party tools/APIs, conversational memory
- **High impact**: an agent error is not an isolated technical error: it's a hallucination, bad advice, an incorrect action performed on behalf of the user

### 2.2 Risks without evaluation

- **Silent regressions**: a prompt change degrades responses without anyone seeing it
- **Production hallucinations**: the agent invents data, calls the wrong tool
- **Uncontrolled cost**: a routing change increases consumed tokens
- **Loss of trust**: users stop using the agent
- **Compliance**: in an insurance company, an agent giving bad financial or HR information = legal risk

### 2.3 Evaluation: non-negotiable criterion for production

No agent should be deployed to production without having passed a formalized evaluation process. This is a structural requirement, same as security tests:

- **Before production**: the agent must demonstrate, through objective and reproducible metrics, that it reaches quality thresholds defined with the business (correctness, tool usage, latency, safety)
- **In production**: the agent must be continuously evaluated via periodic batch evaluations and user feedback to detect any regression
- **For each change**: any modification (prompt, tool, model, configuration) must go back through the evaluation pipeline before reaching production

Without this framework, entities expose themselves to the risks described in §2.2: silent regressions, undetected hallucinations, and loss of user trust.

---

## 3. Anatomy of an agent

### 3.1 Decomposition of an agentic workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                         AGENT WORKFLOW                           │
│                                                                  │
│  ┌────────┐   ┌───────────┐   ┌──────────┐   ┌───────────────┐   │
│  │ INPUT  │──▶│  ROUTING  │──▶│ PLANNING │──▶│  EXECUTION    │   │
│  │ (query)│   │  (intent  │   │ (steps,  │   │  (tool calls) │   │
│  │        │   │  detect.) │   │ strategy)│   │               │   │
│  └────────┘   └───────────┘   └──────────┘   └──────┬────────┘   │
│                                                      │           │
│       ┌──────────────────────────────────────────────┘           │
│       ▼                                                          │
│  ┌──────────┐   ┌────────────┐   ┌────────────┐                  │
│  │  TOOL    │──▶│ SYNTHESIS  │──▶│  OUTPUT    │                  │
│  │ RESULTS  │   │ (reasoning │   │ (response  │                  │
│  │          │   │  assembly) │   │  to user)  │                  │
│  └──────────┘   └────────────┘   └────────────┘                  │
│                                                                  │
│  ════════════════════════════════════════════════════════════    │
│  EVALUATION POINTS:                                              │
│  [E1] Input parsing & intent classification accuracy             │
│  [E2] Route selection correctness                                │
│  [E3] Plan quality & step ordering                               │
│  [E4] Tool selection accuracy (right tool for the task)          │
│  [E5] Tool call correctness (right parameters)                   │
│  [E6] Tool result interpretation                                 │
│  [E7] Synthesis quality (faithfulness, completeness)             │
│  [E8] Final output quality (relevance, format, tone)             │
│  [E9] End-to-end trajectory correctness                          │
│  [E10] Latency, cost tokens, number of steps                     │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Evaluation ↔ step mapping

| Point | Step            | Evaluation type                      | Recommended method                |
|-------|------------------|----------------------------------------|-------------------------------------|
| E1    | Input parsing    | Classification accuracy                | Exact match, F1 on intents         |
| E2    | Routing          | Route correctness                      | Exact match vs golden route          |
| E3    | Planning         | Plan quality                           | LLM-as-Judge, structure match        |
| E4    | Tool selection   | Tool recall / precision                | Exact match vs expected tool set     |
| E5    | Tool call params | Parameter correctness                  | JSON diff vs expected params         |
| E6    | Tool results     | Result interpretation                  | LLM-as-Judge, factual consistency    |
| E7    | Synthesis        | Faithfulness, completeness             | RAGAS-style, LLM-as-Judge           |
| E8    | Output           | Relevance, format, tone                | LLM-as-Judge, regex, custom scorers  |
| E9    | Trajectory       | End-to-end correctness                 | Trajectory match, outcome match      |
| E10   | Performance      | Latency, tokens, steps                 | Numerical thresholds                |

> **Note**: we're used to evaluating only E8 (the final response). This is insufficient. An agent can give a good response by accident (the right result for the wrong reasons). Evaluating the **trajectory** (E1→E9) is what distinguishes a mature evaluation from a naive one.

### 3.3 Multi-agent: additional complexity

When an agent orchestrates other agents (inter-agent communication), evaluation must cover:

```
  Agent Orchestrator
       │
       ├── Correct delegation? (which sub-agent, with what context?)
       ├── Agent A
       │     └── Standard evaluation E1-E10
       ├── Agent B
       │     └── Standard evaluation E1-E10
       └── Result aggregation
             └── Final result coherent?
```

This multiplies the number of evaluation points. The recommended strategy is to evaluate each agent **independently** (agent unit tests) then the **orchestration** as an integration test.

---

## 4. What are we trying to evaluate?

### 4.1 Taxonomy of evaluation dimensions

```
                    ┌──────────────────────┐
                    │   EVALUATION DIMS    │
                    └──────────┬───────────┘
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
   ┌───────────────┐  ┌────────────┐  ┌───────────────┐
   │  FUNCTIONAL   │  │ OPERATIONAL│  │   BUSINESS    │
   │               │  │            │  │               │
   │ • Correctness │  │ • Latency  │  │ • User sat.   │
   │ • Tool usage  │  │ • Cost ($) │  │ • Task comp.  │
   │ • Trajectory  │  │ • Tokens   │  │ • Conversion  │
   │ • Faithfulness│  │ • Error    │  │ • Retention   │
   │ • Relevance   │  │   rate     │  │ • Revenue     │
   │ • Safety      │  │ • Uptime   │  │ • NPS         │
   │               │  │ • Thruput  │  │               │
   └───────────────┘  └────────────┘  └───────────────┘
```

> **Note**: Non-exhaustive list.

### 4.2 Functional metrics

#### 4.2.1 Final response correctness

- **Exact match**: is the response identical to the reference? (rarely applicable to agents)
- **Semantic similarity**: embedding cosine similarity between response and reference
- **LLM-as-Judge**: a model evaluates quality with explicit criteria

#### 4.2.2 Tool selection correctness

Did the agent select the right tool(s) for the task?

- **Evaluation type**: **Deterministic**. Compares `tools_called` vs `expected_tools` by name: `|correct_tools_selected| / |tools_selected|`. Optionally, when the list of all `available_tools` is provided, an LLM judge additionally evaluates whether the selection was optimal; the final score becomes `min(deterministic_score, llm_score)`.

> *Example — Motor claim*: The policyholder says *"I had a car accident, I want to file a claim."*
>
> - **Expected tools**: `authenticate_policyholder`, `lookup_motor_policy`, `check_coverage("accident")`, `create_motor_claim`
> - **Agent called**: `authenticate_policyholder`, `lookup_motor_policy`, `check_coverage("accident")`, `create_motor_claim`, `get_weather`
> - **Score**: 4 correct / 5 called = **0.80** — `get_weather` is superfluous.

#### 4.2.3 Tool argument correctness

Were the arguments passed to each tool correct?

- **Evaluation type**: **LLM-as-a-Judge** (referenceless). An LLM judge verifies that the arguments are logically consistent with the user's input — no golden reference arguments are needed.

> *Example — Motor claim*: The policyholder says *"My policy number is MOT-2026-5678, the accident happened on April 5th in Lyon."*
>
> - **Agent calls**: `create_motor_claim(policy_id="MOT-2026-5678", incident_date="2026-04-05", location="Lyon")`
> - If the agent had passed `incident_date="2026-05-04"` (wrong date format interpretation), the judge would flag it.
> - **Score**: 1.0 if all parameters match the input context, lower otherwise.

#### 4.2.4 Task completion

Did the agent fully complete the requested task?

- **Evaluation type**: **LLM-as-a-Judge** (referenceless). The judge evaluates whether the outcome aligns with the task by analysing the full agent trace. Score of 1 means complete fulfilment; lower scores indicate partial or failed completion.

> *Example — Motor claim*: The policyholder asks *"I want to declare a motor claim for a rear-end collision and get my claim number."*
>
> - **Agent output**: *"Your motor claim has been registered under reference SIN-MOT-2026-0412. A claims adjuster will contact you within 48h."*
> - **Score**: 1.0 — task fully accomplished. If the agent had only authenticated the user but never created the claim, the score would be low.

#### 4.2.5 Task adherence

Did the agent stay within the scope of the task without performing unrelated or forbidden actions?

- **Evaluation type**: **Deterministic**. Calculated as `1 - (|off-task_actions| / |total_actions|)`. Flags any tool call present in a `forbidden_tools` list or absent from the expected trajectory.

> *Example — Motor claim*: The policyholder asks *"File a claim for my car accident."*
>
> - **Agent actions**: `authenticate_policyholder`, `lookup_motor_policy`, `check_coverage`, `create_motor_claim`, `cancel_policy`
> - `cancel_policy` is a **forbidden action** — the user never asked to cancel anything.
> - **Score**: `1 - (1 forbidden / 5 total)` = **0.80**

#### 4.2.6 Steps efficiency

Did the agent complete the task in a reasonable number of steps?

- **Evaluation type**: **LLM-as-a-Judge**. The judge analyses the full execution trace and penalises redundant tool calls, unnecessary reasoning loops, and any actions not strictly required to complete the task.

> *Example — Motor claim*: The policyholder asks *"Declare a motor claim for a windshield break."*
>
> - **Expected steps**: 4 (authenticate → lookup policy → check coverage → create claim)
> - **Agent actual steps**: 7 (authenticate → lookup policy → lookup policy again → check coverage → get weather → check coverage again → create claim)
> - **Score**: low — 3 redundant steps (duplicate lookup, unnecessary weather call, duplicate coverage check).

#### 4.2.7 Plan adherence

Did the agent follow the expected plan or trajectory?

- **Evaluation type**: **LLM-as-a-Judge**. The judge compares the actual execution steps against the agent's stated plan and evaluates how faithfully it was followed.

> *Example — Motor claim*: The agent's plan is: *(1) authenticate → (2) lookup motor policy → (3) check accident coverage → (4) create claim → (5) send confirmation.*
>
> - **Actual execution**: authenticate → check accident coverage → lookup motor policy → create claim → send confirmation
> - Steps 2 and 3 are **swapped** — the agent checked coverage before looking up the policy, which could lead to errors if the policy doesn't exist.
> - **Score**: ~0.7 — mostly followed but deviated on ordering.

### 4.3 Operational metrics

Operational metrics are normally managed by **application tracing** via OpenTelemetry and integrated into the observability platform. Frameworks like Langfuse natively capture these metrics from traces (latency, tokens, cost). They therefore don't require specific implementation in the evaluation system, but must be considered in acceptance thresholds.

| Metric            | Target (exemple) | Critical if (example)  |
|-------------------|------------------|------------------------|
| Latency P50       | < 2s             | > 5s                   |
| Latency P99       | < 10s            | > 30s                  |
| Tokens/request    | < 4000           | > 10000                |
| Cost/request      | < 0.05$          | > 0.20$                |
| Error rate        | < 1%             | > 5%                   |
| Tool failure rate | < 0.5%           | > 2%                   |
| Tool calls/request| < 6              | > 15                   |
| Agent calls/request| < 3             | > 8                    |
| Steps/request     | < 5              | > 10                   |

### 4.4 Business metrics

These metrics are **specific to the use case** and must be defined with the business:

- **Task completion rate**: % of tasks resolved without human escalation
- **User satisfaction**: average feedback score (thumbs up/down, CSAT)
- **First-contact resolution**: agent resolves on first interaction
- **Deflection rate**: % of queries handled by agent vs human
- **Time-to-resolution**: average time to resolve a query

> **Note**: a metric is only useful if it helps you make a decision. A high satisfaction score on its own can be misleading: 90% satisfaction is not a success if 50% of users stop using the agent. Always read business metrics together with functional metrics.

---

## 5. How we evaluate

### 5.1 The three families of evaluation functions

```
┌──────────────────────────────────────────────────────────────────┐
│                    EVALUATION FUNCTIONS                          │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│  │  HEURISTIC      │  │  MODEL-BASED     │  │  HUMAN         │   │
│  │                 │  │                  │  │                │   │
│  │ Exact match     │  │ LLM-as-Judge     │  │ Thumbs up/down │   │
│  │ Regex           │  │ Embedding sim.   │  │ Likert scale   │   │
│  │ JSON diff       │  │ NLI (entailment) │  │ A/B preference │   │
│  │ Contains check  │  │ RAGAS metrics    │  │ Expert review  │   │
│  │ Levenshtein     │  │ Custom LLM eval  │  │ Annotation     │   │
│  │ Set operations  │  │ Classifier       │  │                │   │
│  │                 │  │                  │  │                │   │
│  │ COST: ~0        │  │ COST: ~tokens    │  │ COST: ~human   │   │
│  └─────────────────┘  └──────────────────┘  └────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Heuristic functions (code-based)

Heuristic functions are pure calculations that don't require LLM calls. They include:

- **Tool selection metrics**: Recall, precision, accuracy of tool choices
- **Parameter validation**: JSON schema validation, parameter matching
- **Sequence analysis**: Tool order correctness, step sequence validation
- **Performance thresholds**: Latency, token count, cost limits
- **Safety checks**: Forbidden tool detection, PII leakage detection

These functions are fast, cost-effective (~0), and highly reproducible. They form the foundation of evaluation and can be run frequently in CI/CD pipelines.

### 5.3 Model-based functions (LLM-as-Judge)

#### Why LLM-as-Judge?

To evaluate **qualitative** dimensions (relevance, tone, completeness, faithfulness), heuristics are not enough. We use an LLM as judge.

#### Judge architecture

```
┌────────────────────────────────────────────────────┐
│                  LLM-AS-JUDGE                      │
│                                                    │
│  INPUT:                                            │
│    • Original question                             │
│    • Agent response                                │
│    • Reference response (optional)                 │
│    • Context / source documents (optional)         │
│    • Evaluation criteria (rubric)                  │
│                                                    │
│  PROMPT TEMPLATE:                                  │
│    "Evaluate the following response on criteria:   │
│     1. Relevance (True/False)                      │
│     2. Completeness (True/False)                   │
│     3. Faithfulness to sources (True/False)        │
│     4. Clarity (True/False)                        │
│     Return JSON with binary scores and             │
│     justifications"                                │
│                                                    │
│  OUTPUT:                                           │
│    {"relevance": 4, "completeness": 3, ...}        │
│    + reasoning for each dimension                  │
└────────────────────────────────────────────────────┘
```

#### LLM-as-Judge best practices

| Practice                          | Why                                                                                                                                                             |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Use a stronger or specialized model | Claude 4.5 Sonnet / GPT-5.4-mini as judge, or a model fine-tuned for evaluation (ex: Prometheus, flow-judge). Don't use the same model as the agent             |
| Explicit and detailed rubrics  | Reduce judge subjectivity                                                                                                                                       |
| Structured output (JSON)          | Reliable score parsing                                                                                                                                          |
| Few-shot examples in the prompt  | Calibrate judge on known cases                                                                                                                                  |
| Double evaluation + agreement     | Detect judge inconsistency                                                                                                                                      |
| Temperature = 0                   | Maximize reproducibility                                                                                                                                        |
| Separate dimensions            | One prompt per dimension if necessary                                                                                                                           |
| **Control cost**             | An LLM-Judge costs money per evaluation. On 150 cases × 3 dimensions = 450 calls/run. Plan budget (cf. §19.2), use cache, and prefer heuristics when sufficient. |

> **Note**: LLM-as-Judge is powerful but has known biases (preference for long responses, position bias, self-complacency if same model). Always **calibrate** the judge against human annotations on a sample. The judge is only reliable if it correlates with human judgment.

### 5.4 Traces: foundation of evaluation

**Without traces, no evaluation.** Agent evaluation relies entirely on **traces** captured during execution. Tracing can be instrumented via native framework SDKs (Langfuse `@observe`, MLflow `mlflow.trace`) or via an **OpenTelemetry collector** that exports to the chosen observability platform. A trace is the complete record of everything the agent did during an interaction:

```
┌──────────────────────────────────────────────────────────────────┐
│                    ANATOMY OF A TRACE                            │
│                                                                  │
│  Trace ID: tr-2026-04-02-abc123                                  │
│  Agent: claims-assistant v1.4.2                                  │
│  Input: "Status of my auto claim of March 15"                    │
│                                                                  │
│  ┌─── Step 1: LLM Call (routing) ──────────────────────────── ┐  │
│  │  Model: gpt-5.4                                            │  │
│  │  Input: system_prompt + user_query                         │  │
│  │  Output: "intent=claim_inquiry, route=claims_agent"        │  │
│  │  Tokens: 342 in / 28 out   Latency: 1.2s                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌─── Step 2: Tool Call ───────────────────────────────────── ┐  │
│  │  Tool: authenticate_policyholder                           │  │
│  │  Input: {"user_id": "u-123"}                               │  │
│  │  Output: {"status": "authenticated", "name": "Jean D."}    │  │
│  │  Latency: 0.3s                                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌─── Step 3: Tool Call ───────────────────────────────────── ┐  │
│  │  Tool: lookup_claim                                        │  │
│  │  Input: {"type": "auto", "date": "2026-03-15"}             │  │
│  │  Output: {"claim_id": "SIN-2026-03-0456", "status": ".."}  │  │
│  │  Latency: 0.5s                                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌─── Step 4: LLM Call (response generation) ──────────────── ┐  │
│  │  Model: gpt-5.4                                            │  │
│  │  Input: context + tool_results                             │  │
│  │  Output: "Your claim SIN-2026-03-0456 is in process."      │  │
│  │  Tokens: 512 in / 45 out   Latency: 0.8s                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Final Output: "Your claim SIN-2026-03-0456 is in process."      │
│  Total Latency: 2.8s   Total Tokens: 927   Total Cost: $0.012    │
└──────────────────────────────────────────────────────────────────┘
```

It's this trace that allows calculating **all evaluation metrics**:

| Metric | How to calculate from trace | Example |
|----------|-------------------------------------|---------|
| **Tool selection correctness** | Tools correctly called / total tools called (compare `tools_called` vs `expected_tools` by name) | 2 correct / 2 called = 1.0 ✅ |
| **Tool argument correctness** | LLM judge verifies each tool call's arguments are consistent with the user input | `type=auto, date=2026-03-15` ✅ |
| **Task completion** | LLM judge evaluates whether the final outcome aligns with the requested task | Claim status returned → 1.0 ✅ |
| **Task adherence** | Check no forbidden tool appears in trace + all actions are within scope: `1 - (off-task / total)` | 0 forbidden / 4 steps = 1.0 ✅ |
| **Steps efficiency** | LLM judge analyses trace for redundant or unnecessary steps | 4 steps, no redundancy ✅ |
| **Plan adherence** | LLM judge compares actual execution order vs the agent's stated plan | Steps in expected order ✅ |
| **Latency** | Sum of latencies in trace | 2.8s |
| **Token usage** | Sum of tokens in trace | 927 |
| **Tool calls** | Count of tool call spans in trace | 2 |
| **Agent calls** | Count of agent/sub-agent spans in trace | 1 |

### 5.5 The complete flow: dataset → agent → trace → evaluation

Here's the concrete flow, as it works with **Langfuse**:

```
┌──────────────────────────────────────────────────────────────────┐
│           CONCRETE EVALUATION FLOW (Langfuse example)            │
│                                                                  │
│  1. DATASET                                                      │
│     ┌──────────────────────────────────────┐                     │
│     │ { input: "Status auto claim?",       │                     │
│     │   expected_output: "SIN-..in proc.", │                     │
│     │   expected_tools: [auth, lookup],    │                     │
│     │   expected_sub_agents: [],           │                     │
│     │   forbidden_tools: [cancel_policy] } │                     │
│     └───────────────┬──────────────────────┘                     │
│                     │                                            │
│  2. INVOKE AGENT    ▼                                            │
│     The runner sends input to the agent.                         │
│     The agent is instrumented with Langfuse SDK (or MLflow).     │
│     Each step is automatically captured as a trace.              │
│                     │                                            │
│  3. CAPTURED TRACE  ▼                                            │
│     ┌──────────────────────────────────────┐                     │
│     │ Trace tr-abc123:                     │                     │
│     │  Step 1: LLM call (routing)          │                     │
│     │  Step 2: Tool: authenticate_holder   │                     │
│     │  Step 3: Tool: lookup_claim          │                     │
│     │  Step 4: LLM call (response gen)     │                     │
│     │  Output: "Your claim SIN-..."        │                     │
│     └───────────────┬──────────────────────┘                     │
│                     │                                            │
│  4. EVALUATION      ▼                                            │
│     The evaluator compares TRACE to DATASET:                     │
│     • trace.tools vs dataset.expected_tools → tool_recall        │
│     • trace.output vs dataset.expected_output → LLM-Judge        │
│     • trace.tools vs dataset.forbidden_tools → safety check      │
│                     │                                            │
│  5. SCORES          ▼                                            │
│     Scores are posted to the trace in Langfuse.                  │
│     → Visible in native Langfuse dashboard.                      │
│     → Aggregated by version tag for experiment tracking.         │
└──────────────────────────────────────────────────────────────────┘
```

> **Key point**: it's the **trace** that's at the center of evaluation. Without trace, we can only evaluate the final response. With trace, we can evaluate each step: which tool was called, with what parameters, which sub-agent was invoked, etc.

---
