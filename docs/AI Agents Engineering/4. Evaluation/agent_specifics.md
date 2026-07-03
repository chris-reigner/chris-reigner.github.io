# Agent Evaluation

---

## Table of Contents

1. [Data strategy](#1-data-strategy)
2. [Experiment tracking](#2-experiment-tracking)
3. [Evaluation modes](#3-evaluation-modes)
4. [GitOps & branching strategy](#4-gitops--branching-strategy)
5. [CI/CD pipeline](#5-cicd-pipeline)
6. [Agent Registry & promotion](#6-agent-registry--promotion)
7. [Appendix — Resources](#7-appendix--resources)

---

## 1. Data strategy

### 1.1 Ideal data for evaluating an agent

An agent's evaluation dataset is **fundamentally different** from a classic NLP dataset. It's not just question/answer pairs.

#### Structure of a complete eval record

```json
{
  "id": "eval-001",

  "dataset_metadata": {
    "version": "v1.2.0",
    "language": "en",
    "author": "qa_team",
    "created_at": "2026-03-15",
    "last_updated": "2026-04-01",
    "tags": ["insurance", "claim", "auto", "follow-up"],
    "category": "business_workflow"
  },

  "agent_metadata": {
    "version": "v2.1.0",
    "commit_hash": "a1b2c3d4",
    "repository_url": "https://github.company.com/claims-agent",
    "model_name": "gpt-5.4",
    "temperature": 0.0,
    "prompt_version": "v3.1"
  },

  "evaluation_metadata": {
    "judge_model": "claude-4.5-sonnet",
    "judge_version": "v1.0",
    "evaluation_date": "2026-04-02",
    "framework": "langfuse",
    "evaluator_versions": {
      "tool_usage": "v1.0",
      "trajectory": "v1.1",
      "safety": "v1.0"
    }
  },

  "input": {
    "query": "I would like to know the status of my auto claim of March 15",
    "conversation_history": [],
    "user_context": {"user_id": "u-123", "role": "insured", "locale": "fr-FR"}
  },

  "expected_trajectory": {
    "steps": [
      {
        "order": 1,
        "tool": "authenticate_policyholder",
        "params": {"user_id": "u-123"},
        "expected_result_contains": ["authenticated"],
        "optional": false
      },
      {
        "order": 2,
        "tool": "lookup_claim",
        "params": {"type": "auto", "date": "2026-03-15"},
        "expected_result_contains": ["claim_id", "status"],
        "optional": false
      }
    ],
    "forbidden_tools": ["cancel_policy", "process_payment", "admin_panel"],
    "max_steps": 4,
    "required_tools": ["authenticate_policyholder", "lookup_claim"],
    "tool_order_matters": true
  },

  "expected_output": {
    "reference_answer": "Your auto claim of March 15 (ref. SIN-2026-03-0456) is in process. An expert has been assigned.",
    "must_contain": ["claim", "auto", "process"],
    "must_not_contain": ["error", "impossible"],
    "tone": "professional",
    "language": "fr"
  },

  "thresholds": {
    "min_relevance_score": 4,
    "min_tool_recall": 1.0,
    "max_latency_ms": 5000,
    "max_tokens": 3000
  }
}
```

### 1.2 Dataset types

Three dataset types serve distinct purposes in the evaluation lifecycle. Each has a different owner, cadence, and role in the pipeline.

#### 1.2.1 Golden dataset

The golden dataset is the **single source of truth** for agent evaluation. It contains business-validated inputs paired with ground truth for every metric you wish to evaluate (expected output, expected tools, expected trajectory, forbidden tools, thresholds).

**Purpose**: bridge the gap between technical metrics and business expectations. A golden dataset item doesn't just say "the answer should be X" — it defines what correct tool usage, task completion, and adherence look like for a real business scenario.

**Composition**:

- A **mix of representative cases** (happy paths that cover the most frequent user intents) and **edge cases** (ambiguous queries, missing data, multi-step scenarios)
- Well curated: each item is reviewed, not bulk-generated
- Minimum ~30 items per agent to be statistically meaningful, ideally 50–100+ covering all supported intents

**Who creates it**: Business/PM defines the scenarios and expected outcomes. Data Scientists and AI Engineers formalise them into the evaluation schema (expected tools, trajectories, thresholds). Both parties validate together.

**When to update**: new feature, new tool, changed business rule, or when production feedback reveals a gap.

```json
{
  "input": "I had a car accident yesterday on the A6 motorway, I want to file a claim",
  "expected_output": "Your motor claim has been registered under reference SIN-MOT-...",
  "expected_tools": ["authenticate_policyholder", "lookup_motor_policy", "check_coverage", "create_motor_claim"],
  "expected_trajectory": [
    {"step": 1, "action": "authenticate_policyholder"},
    {"step": 2, "action": "lookup_motor_policy"},
    {"step": 3, "action": "check_coverage", "params": {"peril": "accident"}},
    {"step": 4, "action": "create_motor_claim", "params": {"type": "accident", "location": "A6"}}
  ],
  "forbidden_tools": ["cancel_policy", "modify_coverage"],
  "max_steps": 6,
  "category": "motor_claim",
  "difficulty": "medium"
}
```

#### 1.2.2 Synthetic dataset

The synthetic dataset is **generated from the golden dataset** using an LLM to produce variations at scale. It allows running offline and experiment evaluations in a broader manner, surfacing potential issues the business may not have anticipated.

**Purpose**: volume and diversity. Where the golden dataset has 50 curated items, the synthetic dataset can have 500+ variations — different phrasings, typos, ambiguous formulations, multilingual inputs, unusual parameter combinations — all derived from the same golden scenarios.

**How it works**:

1. Take each golden dataset item as a seed
2. Use an LLM to generate N variations (rephrasings, edge-case twists, adversarial formulations)
3. Inherit the same expected tools, trajectory, and thresholds from the seed (adjusted where the variation changes the expected behaviour)
4. Optionally include adversarial cases: prompt injections, out-of-scope requests, attempts to trigger forbidden tools

**Who creates it**: AI Engineers / Data Scientists, using automated generation pipelines. Business reviews a sample to validate realism.

**When to regenerate**: each evaluation cycle, when the golden dataset changes, or when exploring a new model/prompt version.

> **Important**: synthetic datasets are useful for coverage but **don't replace** the golden dataset. LLM-generated cases tend to be "too clean" and miss the chaotic reality of real user queries. Always use synthetic as a complement, never as the sole evaluation source.

#### 1.2.3 Regression dataset

The regression dataset has a fundamentally different goal: **ensure that fixed bugs stay fixed and that existing behaviour is not altered** by new changes.

**Purpose**: quality gate for QA and development. Every bug fix, edge case discovery, or production incident becomes a new regression test case. This dataset grows monotonically — items are added, never removed.

**Composition**:

- Each fixed bug → a new test case with the input that triggered the bug and the now-correct expected behaviour
- Edge cases discovered during development or QA
- Cases from production incidents (anonymised)
- Behavioural expectations: not just "correct output" but also "must not call forbidden tool X" or "must complete in ≤ N steps"

**Who creates it**: Developers and QA engineers, as part of the bug-fix workflow. When closing a bug ticket, adding a regression case is mandatory.

**When to update**: every bug fix, every production incident, every QA finding.

| Dataset    | Owner                    | Size       | Cadence          | Role in pipeline                  |
|------------|--------------------------|------------|------------------|-----------------------------------|
| Golden     | Business + PM + DS + AI  | 50–100+    | Feature/rule change | Acceptance gate (staging → prod) |
| Synthetic  | AI Engineer / DS         | 500+       | Each eval cycle   | Broad offline evaluation          |
| Regression | Dev + QA                 | Growing    | Each bug fix      | CI gate (no regressions)          |

### 1.3 Dataset storage and versioning

The storage and versioning of evaluation datasets is an **open question** that must be decided based on the entity's existing infrastructure.

**Three main options:**

| Approach | Advantages | Disadvantages |
|----------|-----------|---------------|
| **Git** (datasets in agent repo) | Native versioning, diff, PR review, CI/CD trigger | Limited for large datasets, no native link with traces |
| **Object storage** (S3, Azure Blob) | Scalable, suitable for large datasets, no PII in Git | Manual versioning to manage |
| **Langfuse/MLflow** (native datasets) | Direct integration with traces, built-in versioning, evaluation UI integration | Framework dependency, storage limits |

### 1.4 Dataset creation procedure

How to create an evaluation dataset? Here's the recommended procedure:

```
┌──────────────────────────────────────────────────────────────────┐
│               DATASET CREATION PROCEDURE                         │
│                                                                  │
│  STEP 1: Understand the agent                                    │
│  ├── What are its main use cases?                                │
│  ├── What tools does it use?                                     │
│  ├── What sub-agents does it call?                               │
│  └── What are the edge cases?                                    │
│                                                                  │
│  STEP 2: Create golden set (manual, sized by category)           │
│  ├── Write nominal cases (happy paths)                           │
│  ├── For each case, define:                                      │
│  │   ├── input (query + user context)                            │
│  │   ├── expected_output (expected response)                     │
│  │   ├── expected_tools (tools to call, with params)             │
│  │   ├── expected_sub_agents (sub-agents to invoke)              │
│  │   ├── forbidden_tools (tools that MUST NOT be called)         │
│  │   └── thresholds (case-specific thresholds)                   │
│  └── Team review (pair review of dataset)                        │
│                                                                  │
│  STEP 3: Enrich with production traces (if available)            │
│  ├── Sample real traces from Langfuse/MLflow                     │
│  ├── Anonymize data (remove PII)                                 │
│  ├── Manually annotate expected outputs                          │
│  └── Add to regression set                                       │
│                                                                  │
│  STEP 4: Synthetic generation (optional, for volume)             │
│  ├── Use an LLM to generate variations                           │
│  ├── Vary formulations (formal, informal, ambiguous)             │
│  ├── Mandatory human review of generated cases                   │
│  └── Add to synthetic set                                        │
│                                                                  │
│  STEP 5: Version and synchronize                                 │
│  ├── Version dataset (tag or version identifier)                 │
│  ├── Sync to evaluation framework for execution                  │
│  └── Update configuration if necessary                           │
└──────────────────────────────────────────────────────────────────┘
```

**Exemple structure of an eval record (with all fields for trace ↔ dataset link)** :

```json
{
  "id": "eval-042",
  "metadata": {
    "category": "claim_inquiry",
    "difficulty": "medium",
    "source": "manual",
    "author": "qa_team",
    "created_at": "2026-03-15"
  },
  "input": {
    "query": "I would like an auto insurance certificate for my vehicle AB-123-CD",
    "conversation_history": [],
    "user_context": {"user_id": "u-456", "locale": "fr-FR"}
  },
  "expected_tools": [
    {"name": "authenticate_policyholder", "params": {"user_id": "u-456"}, "order": 1, "optional": false},
    {"name": "lookup_vehicle_policy", "params": {"plate": "AB-123-CD"}, "order": 2, "optional": false},
    {"name": "generate_certificate", "params": {"type": "attestation_auto"}, "order": 3, "optional": false}
  ],
  "expected_sub_agents": [],
  "forbidden_tools": ["cancel_policy", "modify_coverage", "process_payment"],
  "expected_output": {
    "reference_answer": "Your auto insurance certificate for vehicle AB-123-CD has been generated. You will find it in your customer space.",
    "must_contain": ["certificate", "auto", "AB-123-CD"],
    "must_not_contain": ["error", "impossible"],
    "language": "fr"
  },
  "thresholds": {
    "min_tool_recall": 1.0,
    "max_latency_ms": 5000,
    "max_steps": 5
  }
}
```

---

## 2. Experiment tracking

### 2.1 Why experiment tracking?

An agent constantly evolves: prompt change, tool addition/removal, routing modification, LLM model change. Each change is an **experiment** whose impact must be measured.

```
┌─────────────────────────────────────────────────────────────────┐
│                   EXPERIMENT TRACKING FLOW                      │
│                                                                 │
│  ┌────────────┐   ┌──────────────┐   ┌───────────┐              │
│  │ HYPOTHESIS │──▶│  EXPERIMENT  │──▶│  RESULTS  │              │
│  │            │   │              │   │           │              │
│  │ "If I      │   │ • Prompt v3  │   │ Scores:   │              │
│  │  change    │   │ • Model: 5.4 │   │  +3% rel. │──▶ DECISION  │
│  │  the       │   │ • Tools: +1  │   │  -1% cost │   (adopt /   │
│  │  prompt    │   │ • Dataset: G │   │  +5% tool │    reject)   │
│  │  routing"  │   │              │   │           │              │
│  └────────────┘   └──────────────┘   └───────────┘              │
│                                                                 │
│  TRACKED PARAMETERS:             TRACKED METRICS:               │
│  • agent_version_tag (v1.4.2)    • All eval scores              │
│  • git_tag (source of truth)     • Latency percentiles          │
│  • prompt_version                • Token usage                  │
│  • model_name + version          • Cost per request             │
│  • tool_configuration            • Error rate                   │
│  • temperature                   • Pass/fail per test case      │
│  • system_prompt_hash            • Composite score              │
│  • dataset_version                                              │
│  • eval_config_version                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Comparison between experiments

The major interest is to **compare** runs:

```
┌─────────────────────────────────────────────────────────────────┐
│              EXPERIMENT COMPARISON DASHBOARD                    │
│                                                                 │
│  Run ID       Version  Prompt  Model          Composite Tool R. │
│  ───────────  ───────  ──────  ────────────── ───────── ─────── │
│  run-0401     v1.3.1   v3.1    gpt-5-mini     0.821     0.88    │
│  run-0402     v1.4.0   v3.2    gpt-5.4        0.847 ▲   0.92 ▲  │
│  run-0402b    v1.4.0   v3.2    claude-4.5-s   0.862 ▲   0.94 ▲  │
│  run-0403     v1.4.1   v3.3    gpt-4o-mini    0.831 ▼   0.90    │
│                                                                 │
│  ▲ = improvement vs baseline    ▼ = regression vs baseline      │
│                                                                 │
│  DECISION: v3.2 with Claude is the best candidate.              │
│  Warning: v3.3 regresses on composite despite tool recall ok    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Evaluation modes

### 3.1 Offline evaluation

Offline evaluation is the process of assessing an agent's performance in a **controlled, non-production environment** using pre-collected or curated datasets (golden, synthetic, regression — see §1.2) rather than live user traffic. The agent runs against a fixed set of inputs, its outputs and traces are collected, and scorers compute metrics across multiple dimensions (correctness, tool usage, adherence, efficiency, etc.).

This is what was described in the dataset and experiment tracking sections: a known dataset with **ground truth** is the prerequisite. Without expected outputs, expected tools, and expected trajectories, offline evaluation cannot produce meaningful scores.

**What it enables**:

- Systematically measure quality and compare agent versions
- Detect regressions before deployment
- Validate prompt, model, or tool changes in isolation
- Feed guardrail validators with specific scenarios to build project-specific guardrails and enhance their objective functions

**Who uses it**: Data Scientists, AI Engineers, QA Engineers, Business Owners — each tracking their evaluations (dataset, code, agent configuration, results) to ensure reusability and auditability over time.

### 3.2 Online evaluation

Online evaluation applies the **same metrics as offline evaluation but on new, unseen data** — production traces and logs from real user interactions. There is no pre-defined ground truth; instead, quality is inferred from the traces themselves and progressively enriched with human feedback (thumbs up/down, expert review, escalation signals).

**What it detects**:

- **Behavioural drift**: the agent starts responding differently due to prompt changes, model updates, or tool modifications
- **Data drift**: user queries evolve (new intents, new vocabulary, seasonal patterns) or upstream services change their responses
- **Silent regressions**: degradations that only surface with real-world traffic patterns

**Evaluation methods**:

- **Deterministic metrics**: agent workflow completion, tool call counts, latency, token usage — computed directly from traces
- **LLM-as-a-Judge**: applied on sampled traces to evaluate qualitative dimensions (relevance, completeness, adherence)

> **FinOps consideration**: LLM-as-a-Judge on every production trace is cost-prohibitive. Evaluate a **statistically significant sample** of traces (e.g. 5–10% with stratified sampling by intent) rather than exhaustive evaluation. Prioritise deterministic metrics for 100% coverage and reserve LLM-judge for sampled or flagged traces.

### 3.3 Continuous evaluation

Continuous evaluation is the practice of **systematically and repeatedly** assessing an agent's performance over time — combining both offline and online — to detect quality degradation, catch regressions, and drive ongoing improvement as the agent, its environment, and user behaviour evolve.

Online evaluation can be performed through **guardrails**: validators act directly on traces in real time, blocking or flagging problematic responses. This creates a tight feedback loop between evaluation and enforcement.

In a target implementation, the data used for offline evaluation is **continuously enriched** with new data — including human feedback (corrections, escalations) and agent self-assessments. Achieving this in a fully autonomous manner is challenging, but the goal is a virtuous cycle:

```
┌─────────────────────────────────────────────────────────────────┐
│                  CONTINUOUS EVALUATION LOOP                     │
│                                                                 │
│  ┌───────────────────┐    ┌────────────────────┐                │
│  │ Offline eval      │    │ Online eval        │                │
│  │ (golden/synthetic │◄──►│ (production        │                │
│  │  /regression)     │    │  traces + feedback)│                │
│  └────────┬──────────┘    └────────┬───────────┘                │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌──────────────────────────────────────────┐                   │
│  │         Feedback & enrichment             │                  │
│  │  • Human-in-the-loop corrections          │                  │
│  │  • Automated alerting & quality gates     │                  │
│  │  • Dataset evolution (new cases added)    │                  │
│  │  • Version comparison & promotion         │                  │
│  └──────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

**Continuous evaluation includes**:

| Component                          | Mode     | Description                                                        |
|------------------------------------|----------|--------------------------------------------------------------------|
| Scheduled offline evaluation       | Offline  | Periodic re-runs against golden + regression datasets (CI/CD)      |
| Online production monitoring       | Online   | Deterministic metrics on 100% of traces, LLM-judge on samples      |
| Human-in-the-loop feedback loops   | Online   | Expert review, user feedback enriching datasets over time           |
| Automated alerting and gating      | Both     | Quality gates in CI/CD (offline) + drift alerts in production       |
| Dataset evolution                  | Offline  | New cases from production incidents, feedback, and edge discoveries |
| Version comparison                 | Offline  | A/B experiment tracking between agent versions                      |

> **Key distinction**: offline evaluation requires **ground truth** (expected outputs, tools, trajectories) and is run against curated datasets. Online evaluation operates on **new data without ground truth**, but is progressively enriched with human feedback to close the loop. Together, they form a complete evaluation strategy.

---

## 4. GitOps & branching strategy

### 4.1 Technical proposal: Trunk-Based + Tag-based promotion

The proposed model is **trunk-based development** with version tag promotion. Environments (dev, staging, prod) are not branches but **deployments** of a specific tag.

```
  main (trunk)
    │
    ├── feat/new-tool ──▶ PR ──▶ Smoke Eval ──▶ Merge to main
    │
    │   [main = dev environment by default]
    │
    ├── Promotion tag: staging-v1.4.2
    │     └── Staging deployment + Full Eval + QA
    │
    ├── Promotion tag: prod-v1.4.2
    │     └── Prod deployment + Registry update
    │
    └── Hotfix direct on main if urgent
```

**Technical choices:**

- **Single flow**: no drift between long branches, no merge conflicts
- **Tiered evaluation**: smoke eval (heuristics, fast) on each PR, full eval (LLM-Judge) only at promotion
- **Tags = source of truth**: Git tag determines what is deployed and evaluated in each environment
- **Rollback**: re-deploy tag N-1 if regression detected in staging or prod

### 4.2 Branch ↔ evaluation mapping

| Git event              | Triggered evaluation      | Thresholds (exemple)            | Blocking? |
|---------------------------|-----------------------------|---------------------------------|------------|
| Push on feature branch   | Smoke tests (5-10 cases)      | No crash, basic sanity          | No        |
| PR to dev/main          | Full eval (golden + regression)| Composite ≥ 0.80, no regression | Yes        |
| Promotion → staging       | Full eval + adversarial      | Composite ≥ 0.85, safety pass   | Yes        |
| Promotion → prod          | Full eval | Composite ≥ 0.90, all gates     | Yes        |
| Scheduled (daily)         | Batch eval production sample | Alerting if drift               | N/A        |

---

## 5. CI/CD pipeline

### 5.1 Global pipeline architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  CI/CD PIPELINE - COMPLETE VIEW                      │
│                                                                      │
│  ┌─────────────┐                                                     │
│  │ DEVELOPER   │                                                     │
│  │ git push    │                                                     │
│  └──────┬──────┘                                                     │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ STAGE 1: BUILD & LINT                                │            │
│  │ • Python lint (ruff/black)                           │            │
│  │ • Type checking (mypy)                               │            │
│  │ • Unit tests (pytest), mock LLM calls                │            │
│  │ • Dependency check                                   │            │
│  └──────────────────────────────┬───────────────────────┘            │
│                                 │ PASS                               │
│                                 ▼                                    │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ STAGE 2: SMOKE EVAL                                  │            │
│  │ • 10-15 cases from golden set                        │            │
│  │ • Heuristics only (no LLM judge)                     │            │
│  │ • Check: tool calls OK, no crash, format OK          │            │
│  │ • Threshold: 100% pass rate on smoke set             │            │
│  └──────────────────────────────┬───────────────────────┘            │
│                                 │ PASS                               │
│                                 ▼                                    │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ STAGE 3: FULL EVAL (on PR only)                      │            │
│  │ • Multiple dataset sets (golden, regression, biz)    │            │
│  │ • Heuristics + LLM-as-Judge                          │            │
│  │ • Comparison vs baseline (last main run)             │            │
│  │ • Configurable thresholds in thresholds.yaml         │            │
│  │ • Results logged in MLflow/Langfuse                  │            │
│  └──────────────────────────────┬───────────────────────┘            │
│                                 │ PASS                               │
│                                 ▼                                    │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ STAGE 4: GATE DECISION                               │            │
│  │ • Check composite_score >= threshold                 │            │
│  │ • Check zero regressions on business_critical set    │            │
│  │ • Check safety tests all pass                        │            │
│  │ • Post results as PR comment                         │            │
│  │ • Block or approve merge                             │            │
│  └──────────────────────────────┬───────────────────────┘            │
│                                 │ APPROVED                           │
│                                 ▼                                    │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ STAGE 5: DEPLOY + REGISTER (if promotion)            │            │
│  │ • Deploy to target environment                       │            │
│  │ • Register in Agent Registry                         │            │
│  │ • Update agent card                                  │            │
│  └──────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Threshold configuration

Evaluation thresholds must be **defined with the business**, not only by the technical team. Each threshold translates a concrete business requirement:

```yaml
# evals/configs/thresholds.yaml
# Single set of thresholds per agent, the agent must reach these levels
# before any production deployment.

thresholds:
  # -- Functional metrics --
  composite_score: 0.85          # Weighted global score, threshold discussed with PO
  tool_recall: 0.90              # Agent correctly calls all expected tools
  tool_precision: 0.85           # Agent doesn't call superfluous tools
  pass_rate: 0.90                # % of dataset cases that pass all checks
  safety_pass_rate: 1.0          # No safety failure tolerated (PII, injection, forbidden tools)
  business_critical_pass_rate: 1.0  # Compliance/regulatory cases pass at 100%

  # -- Operational metrics --
  max_latency_p99_ms: 10000      # Aligned with agent SLA
  max_regressions: 0             # Zero regression vs baseline

# Composite score weighting, to be adjusted per agent according to business priorities
weights:
  tool_recall: 0.20
  tool_precision: 0.15
  output_relevance: 0.25         # Response relevance (LLM-Judge)
  output_completeness: 0.15      # Does response cover all points?
  output_faithfulness: 0.15      # Is response faithful to sources?
  latency: 0.05
  step_efficiency: 0.05
```

> Thresholds are not an isolated technical exercise. A `tool_recall` of 0.90 means that in 10% of cases, the agent forgets to call a critical tool, for example, not checking coverage before creating a claim. It's up to the PO to decide if this level is acceptable for their use case.

---

## 6. Agent Registry & promotion

Evaluation results could be integrated into **agent cards** deployed in the Agent Registry. For each agent version, relevant evaluation information would be: composite score, link to experiment run (Langfuse/MLflow), gate status (pass/fail), and date of last batch eval.

These metadata would enable data-driven promotion decisions and facilitate rollback in case of regression detected in production.

---

## 7. Appendix — Resources

**Evaluation frameworks:**

1. **Langfuse Evaluation**: scores, dataset runs, online evaluation : <https://langfuse.com/docs/scores/overview>
2. **Langfuse Online Evaluation**: automatic asynchronous trace evaluation : <https://langfuse.com/docs/scores/online-evaluation>
3. **MLflow LLM Evaluation**: mlflow.evaluate() for LLMs : <https://mlflow.org/docs/latest/llms/llm-evaluate>
4. **LangSmith Evaluation**: datasets & evaluators : <https://docs.smith.langchain.com/evaluation>
5. **RAGAS**: evaluation framework for RAG pipelines (applicable to agents) : <https://docs.ragas.io>
6. **Braintrust**: eval framework : <https://braintrust.dev>
7. **DeepEval**: open-source LLM evaluation framework : <https://docs.confident-ai.com>
8. **OpenAI Evals**: open-source evaluation framework : <https://github.com/openai/evals>

**Pricing references:**
9. **OpenAI Pricing Calculator**: <https://invertedstone.com/calculators/openai-pricing>
