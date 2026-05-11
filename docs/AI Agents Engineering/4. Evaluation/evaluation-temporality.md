# Evaluation Temporality: From Offline to Continuous Evaluation  (Under review)

> *"Offline evaluation tells you what your agent can do. Online evaluation tells you what it actually does. Continuous evaluation tells you when it stops doing it."*

Evaluation is not a binary event that happens before launch. It is a practice that deepens as your agent matures — and three things change simultaneously as it does: **when** you evaluate, **what data** you evaluate against, and **what signal** you use to score it. These three dimensions are inseparable. Treating them as separate decisions leads to evaluation strategies that are either too expensive, too unreliable, or too disconnected from what the business actually needs to know.

This document maps all three dimensions across the full maturity arc.

---

## The Three Dimensions

Before going into each maturity stage, it is worth naming the three dimensions explicitly. Every decision in evaluation can be traced back to one of them.

```
┌─────────────────────┬──────────────────────────────────────────────────────┐
│  WHEN               │  Before deploy (offline) → after deploy (online)     │
│                     │  → always-on with closed loop (continuous)           │
├─────────────────────┼──────────────────────────────────────────────────────┤
│  WHAT DATA          │  Golden dataset → synthetic → real traces → mix      │
├─────────────────────┼──────────────────────────────────────────────────────┤
│  WHAT SIGNAL        │  Ground truth → LLM judge → human feedback → RLHF   │
└─────────────────────┴──────────────────────────────────────────────────────┘
```

These dimensions do not evolve independently. As your maturity grows, all three shift together. An organization in Stage 1 has no real traces and limited signal options. An organization in Stage 3 has all signal types available and must decide how to weight them against each other.

---

## What Are You Actually Evaluating?

This is the question most teams skip. Before choosing a dataset or a scoring method, you need to decide what part of your agent's behavior you are putting under scrutiny.

There are two fundamental objects of evaluation:

### Final Output (Black-Box)

You observe only what the agent returned to the user. The internal steps — the reasoning, the tool calls, the intermediate decisions — are invisible to the evaluator.

**Advantages:** Simple to set up. Works with any agent, regardless of instrumentation.

**Limitation:** You cannot distinguish a correct answer reached correctly from a correct answer reached by accident. You cannot identify *where* a failure occurred. A score on final output tells you that something went wrong; it does not tell you why.

### Full Trace (Glass-Box)

You observe the complete execution history: every reasoning step, every tool call and its arguments, every intermediate output, and the final response. This is the unit of evaluation that makes agent-specific metrics possible.

**What you annotate in a trace:**

| Trace element | What you evaluate |
|---|---|
| Tool call sequence | Was the right set of tools called, in the right order? |
| Tool arguments | Were parameters correct, well-formed, within valid ranges? |
| Intermediate outputs | Were tool results interpreted correctly before the next step? |
| Reasoning steps | Was the chain of thought coherent? Were conclusions valid? |
| Final response | Did it address the user's intent? Is it faithful to tool outputs? |

**Limitation:** Requires instrumentation. You need to log every span. If your agent is not traced, glass-box evaluation is not possible.

> **Practical rule:** Always instrument from day one. The cost of adding tracing after the fact is much higher than the cost of adding it upfront. Every trace you don't log is an evaluation opportunity permanently lost.

---

## Signal Types: What Are You Comparing Against?

The signal is what you use to score an output or trace. Four types exist, ordered from most structured to most dynamic:

---

### Ground Truth

A human — usually a domain expert or the product team — defined in advance what "correct" looks like. The agent's output is compared directly against this reference.

**Forms:**
- Expected final output: `"Confirmation number XY4821"` — exact or semantic match
- Expected tool trajectory: `[search_flights, check_seat_availability, book_flight]`
- Expected output fields: response must contain `confirmation_id`, `flight_id`, `price`
- Rubric-based: a scoring guide with defined criteria (not a single expected string)

**Strengths:** Most reliable signal. Fully reproducible. No model dependency. Works at temperature=0 for deterministic checks.

**Limitations:** Ground truth only exists for scenarios you anticipated. It ages out when your domain changes. Defining "correct" for open-ended tasks (summarization, reasoning) requires rubrics that are expensive to write and calibrate. And for agentic trajectories, defining the *one* correct sequence of tool calls is often artificial — many valid paths exist.

**When it breaks down:** A retrieval agent that finds the right answer via a different tool sequence than the one in your golden dataset will score 0 on tool correctness even though it completed the task correctly. Ground truth for trajectories requires careful design to avoid penalizing valid alternative paths.

---

### LLM-as-a-Judge Score

A separate LLM evaluates the trace or output against a rubric. See the [How to Run These Metrics](./evaluation.md#how-to-run-these-metrics) section for implementation details and documented biases.

**Forms:**
- Binary: did the agent complete the task? (yes/no)
- Graded: 0–1 or 0–10 scale with rubric
- Comparative: is output A better than output B? (pairwise)

**Strengths:** Scalable. Works without ground truth. Handles open-ended outputs. Can evaluate criteria that are impossible to express as rules (tone, coherence, intent alignment).

**Limitations:** Non-deterministic even at temperature=0. Known biases (verbosity, position, self-enhancement). Agreement with human experts drops to 60–68% in specialized domains. Produces a proxy for quality, not a direct measure. Adds cost and latency to the evaluation pipeline.

**Honest calibration requirement:** Never trust an LLM judge you haven't validated against human labels. Before deploying a judge in production, measure its agreement with a human-annotated gold set. Target Cohen's κ > 0.7. If you can't reach this, your rubric is ambiguous or your judge model is not capable enough for the domain.

---

### Human Feedback

Real humans — users, annotators, domain experts — provide signal about the agent's output after the fact.

**Two kinds:**

**Explicit feedback** — the user directly rates the agent:
- Thumbs up / thumbs down (binary)
- Star ratings (ordinal)
- Structured annotation forms (detailed rubric per response)
- Correction: the user provides the answer that should have been given

**Implicit feedback** — inferred from behavior without asking:
- User re-ran the same task immediately after → likely a failure
- User escalated to a human agent → task not completed satisfactorily
- User copied the response and used it → likely a success
- User rephrased the same question → response was not understood
- Session abandoned after the agent's response → negative signal

**Important reality check:** Explicit feedback is rare. [Research](https://winder.ai/user-feedback-llm-powered-applications/) consistently finds that fewer than 1% of interactions generate an explicit rating. You cannot run an evaluation strategy on thumbs-down signals alone. Implicit feedback, while noisier, is orders of magnitude more abundant.

[Microsoft's approach](https://medium.com/data-science-at-microsoft/beyond-thumbs-up-and-thumbs-down-a-human-centered-approach-to-evaluation-design-for-llm-products-d2df5c821da5) argues for designing feedback collection as part of the product UX — not as an afterthought — with specific feedback prompts triggered by contextual events (e.g., a follow-up question immediately after a response) rather than a generic thumbs-down button.

**Strengths:** Reflects real user preferences. Catches failure modes that automated evaluation misses. Ground truth for preference learning.

**Limitations:** Slow. Expensive for expert annotation. Selection bias — users who give feedback are not representative of all users. Thumbs-down captures dissatisfaction, not necessarily the *reason* for it.

---

### RLHF / Preference Signal

Reinforcement Learning from Human Feedback uses pairwise comparisons — "is response A better than response B?" — to train a reward model that can then score outputs at scale without per-query human annotation. The reward model is used to guide the agent's behavior via reinforcement learning.

**How it differs from the others:**

| | Ground truth | LLM judge | Human feedback | RLHF |
|---|---|---|---|---|
| **Who scores** | Rule / exact match | LLM | Human | Reward model (trained on human preferences) |
| **Scales to production** | Yes | Yes | No | Yes |
| **Learns from real preferences** | No | No | Yes | Yes |
| **Feeds back to model** | No | No | No | Yes |
| **Cost per label** | Near zero | ~$0.01 | $1–$10 | High upfront, near zero after |

**When it makes sense:** RLHF is the most powerful signal because it closes the loop from evaluation all the way back to model or policy training. But it requires:
- Volume: thousands of pairwise comparisons to train a reliable reward model
- Infrastructure: a training pipeline for continuous fine-tuning or policy updates
- Stability: a clear, stable definition of what "better" means in your domain

Modern variants reduce the human annotation burden significantly. **RLAIF** (Reinforcement Learning from AI Feedback) uses an LLM to generate preference labels instead of humans — at a cost below $0.01 per data point vs $1+ for human annotation — with comparable alignment performance for many domains. **Targeted RLHF** combines LLM-based initial alignment with selective human corrections, achieving full-annotation-level performance with only 6–7% of the human annotation effort ([CMU, 2025](https://blog.ml.cmu.edu/2025/06/01/rlhf-101-a-technical-tutorial-on-reinforcement-learning-from-human-feedback/)).

**Honest limitation:** Reward hacking is a real and documented risk. When models are optimized for a reward signal, they find ways to maximize that signal without actually improving the underlying behavior. This is particularly dangerous for agent systems with tool access, where reward hacking can manifest as unexpected tool usage patterns that score well but behave poorly in practice.

---

## Dataset Types

The signal defines *how* you score. The dataset defines *what* you score against. Four types, each with a different role in the evaluation lifecycle.

---

### Golden Dataset

A curated, versioned collection of inputs and their verified expected outputs — the source of truth for measuring quality. Hand-labeled by domain experts. Every item has been reviewed and agreed upon.

**What it contains:**
- Input: the user request or agent task
- Expected output (optional for open-ended tasks): exact string, required fields, or rubric
- Expected trajectory (optional): the tool call sequence considered correct
- Metadata: scenario tags, difficulty level, data source, annotator, date, version

**Size guidelines** ([Maxim AI](https://www.getmaxim.ai/articles/building-a-golden-dataset-for-ai-evaluation-a-step-by-step-guide/)):
- 50–100 items: minimum viable — catches obvious failures
- 200–500 items: production-ready — covers major use cases and known edge cases
- 1,000+: mature system — statistically meaningful per scenario slice

For a 95% confidence interval with ±5% margin of error at 80% pass rate, you need approximately **246 samples per scenario**. Smaller slices will produce confidence intervals too wide to be actionable.

**Maintenance:** Treat the golden dataset as a living artifact with version control. Map dataset versions to model versions and prompt versions. Update it when you discover new failure modes in production. Decontaminate it from training data to avoid inflated metrics from memorization. Refresh compliance-related cases when regulations change.

**The decontamination problem:** If items in your golden dataset overlapped with training data, scores will be artificially inflated. Run embedding similarity checks between your golden dataset and training corpora. Exact match is not enough — the model may have seen paraphrased versions of your evaluation inputs.

---

### Synthetic Dataset

LLM-generated test cases that expand coverage beyond what can be manually curated — particularly useful for edge cases, rare failure modes, and adversarial inputs that don't naturally appear in production early on.

**When to reach for synthetic data:**

| Situation | Why synthetic helps |
|---|---|
| Cold start (no real users yet) | You have no production data; synthetic fills the gap |
| Rare but high-risk scenarios | Real production may never surface these; you need to test them |
| Adversarial / red-teaming | Systematically generate inputs designed to break the agent |
| Multi-turn conversation coverage | Generating long realistic dialogues manually is expensive |
| Scale without annotation cost | Generate 10,000 variants of a scenario at near-zero cost |

**Generation approach** ([Confident AI](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms)):
1. Start with a small seed set of real or hand-crafted examples
2. Use a strong model (e.g., GPT-4o, Claude Opus) to generate variants
3. Evolve the data in three directions: *in-depth* (more complex reasoning), *in-breadth* (more diverse), *elimination* (remove low-quality outputs)
4. Filter for quality: clarity, self-containment, relevance, consistency
5. Human review a sample before treating it as evaluation-grade

**Critical limitation:** Synthetic data reflects the generator model's blind spots, not yours. If GPT-4o generates your test cases, GPT-4o will also tend to find them easy. Synthetic data augments a golden dataset — it does not replace it. Always validate a sample of synthetic cases against real human judgment before adding them to your evaluation pipeline.

The quality of synthetic data improves substantially when generated from real production documents, domain-specific knowledge bases, or actual user queries — not from generic prompts. Ground the generator in your context.

---

### Production Traces

Raw execution logs from the live agent. Not curated, not labeled. The most realistic data available, and also the noisiest.

**What production traces give you that other datasets cannot:**
- The real distribution of user inputs — not what you anticipated, but what users actually do
- Failure modes you didn't design test cases for
- Latency and token consumption under real load
- Edge cases that only appear after thousands of interactions

**The labeling problem:** Production traces have no ground truth by default. You know what the agent did; you don't know what it *should* have done. To turn a production trace into evaluation data, you need to either:
- Apply LLM-as-judge scoring (fast, scalable, imperfect)
- Collect implicit feedback signals (escalations, retries, session abandonment)
- Route flagged traces to human annotators for labeling
- Use business rule checks (Layer 0) that flag policy violations automatically

**Sampling strategy for labeling:** You cannot label every trace. A practical approach:
- Always run deterministic checks (Layer 0 + Layer 2 tool correctness) on 100%
- Sample 5–15% for LLM-as-judge scoring
- Prioritize: traces that ended in error, escalation, or retry; traces from new input patterns; traces near metric thresholds

---

### Mixed Dataset

The practical reality for any mature evaluation pipeline is a mix of all three. The ratio shifts with maturity:

```
Early (pre-launch)
  └── 70% golden (hand-crafted)
  └── 30% synthetic (coverage gaps, adversarial)
  └──  0% production traces (none yet)

Growth (post-launch, 1–6 months)
  └── 40% golden (maintained, growing)
  └── 20% synthetic (edge cases, red-teaming)
  └── 40% production traces (labeled subset)

Mature (> 6 months in production)
  └── 30% golden (core regression tests)
  └── 10% synthetic (adversarial, new scenarios)
  └── 60% production traces (labeled, curated failures)
```

As production data accumulates, it progressively dominates the dataset — because it reflects the real distribution better than anything you can construct artificially. The golden dataset does not shrink; it shifts roles from *primary benchmark* to *regression test suite*.

---

## Stage 1 — Offline Evaluation

**Maturity level:** Low
**When:** During development, before every release, in CI/CD
**Dataset:** Golden + synthetic
**Signal:** Ground truth + LLM-as-judge
**Human involvement:** High

---

### What it looks like

You have built an agent. No real users have interacted with it yet. You have a golden dataset of 50–200 curated test cases covering the scenarios you designed for. You run the agent against every case and score the results. You do this on every significant change to the model, prompt, or tool configuration.

This is your **regression harness**. It catches failures before users see them.

### What you evaluate

| What | How | Signal |
|---|---|---|
| Tool correctness | Compare actual vs. expected tool sequence | Deterministic — ground truth |
| Argument correctness | Field-by-field comparison | Deterministic — ground truth |
| Final output | Does it contain required fields/content? | Deterministic (regex/schema) + LLM judge |
| Task completion | Did the agent accomplish the scenario? | LLM-as-judge with rubric |
| Trajectory quality | Was the path coherent and efficient? | LLM-as-judge |
| Business rules (Layer 0) | No PII, no forbidden content, format valid | Deterministic — rule-based |

### Role of synthetic data here

You use synthetic data to fill **coverage gaps** in your golden dataset. The golden dataset covers scenarios you designed; synthetic data covers scenarios you might have missed. Specific uses at this stage:

- Generate 20–30 adversarial variants of each golden case (malformed inputs, boundary values, ambiguous phrasing)
- Generate multi-turn conversation scenarios that are expensive to hand-craft
- Red-team: generate inputs specifically designed to trigger known agent failure modes (e.g., inputs that historically cause tool hallucination)

### What ground truth looks like for traces

Defining ground truth for a trajectory is harder than for a final output. Three practical approaches:

1. **Exact trajectory**: specify the exact sequence of tools expected. Strict, but fails when multiple valid paths exist.
2. **Required tools set**: specify which tools *must* be called, without enforcing order. More permissive, better for tasks with valid alternative paths.
3. **Forbidden tools set**: specify which tools *must not* be called for this task. Catches unnecessary tool calls (e.g., `get_weather` on a pure booking task) without constraining valid paths.

Most production teams use a combination of (2) and (3): required tools to ensure completeness, forbidden tools to enforce discipline.

### The release gate

Every metric has a threshold. The agent ships when all thresholds are met. No subjective debates.

Before you start building, write down the thresholds. If you wait until after your first eval run to decide what "good enough" means, you will rationalize whatever score you got.

### Honest limitation

Your golden dataset is an approximation of reality. It reflects the scenarios *you imagined*. Real users will behave differently. Treat offline scores as a floor — they tell you the agent is ready to face users, not that it will perform well when it does.

---

## Stage 2 — Online Evaluation

**Maturity level:** Medium
**When:** Continuously, after deployment, asynchronously
**Dataset:** Live production traces (sampled + scored)
**Signal:** LLM-as-judge (sampled) + implicit human feedback + deterministic checks (all traces)
**Human involvement:** Medium — triage alerts, review flagged traces

---

### What it looks like

Your agent is live. Real users are sending real requests. You cannot control the inputs. You instrument every trace, sample a fraction for LLM-as-judge scoring, and watch for metric degradation over time. When a threshold is breached, an alert fires and a human investigates.

This is your **reality check**. It tells you what the agent actually does in the wild, not what it did on your golden dataset.

### The instrumentation prerequisite

Online evaluation is impossible without tracing. Every span must be logged: each LLM call, each tool invocation with its arguments and result, each sub-agent handoff, each error and retry. Every trace must carry a unique ID and metadata: timestamp, user ID (or session ID if anonymous), model version, prompt version, agent version.

If your agent is not traced, you cannot run online evaluation. There is no workaround.

### What you evaluate and how you score it

| What | Coverage | Signal | Timing |
|---|---|---|---|
| Layer 0 (business rules) | 100% of traces | Deterministic — synchronous in-process | Before response is returned |
| Tool call presence / format | 100% of traces | Deterministic — post-hoc | After trace completes |
| Task completion | 5–15% sample | LLM-as-judge | Async, minutes later |
| Trajectory quality | 5–15% sample | LLM-as-judge | Async |
| Escalations / retries | 100% (captured as events) | Implicit human feedback | Real-time |
| User thumbs-down | < 1% (rare) | Explicit human feedback | Real-time |

### The "no ground truth" problem

For most production traces, you do not know what the agent *should* have done. You only know what it *did*. This means you cannot compute tool correctness (which requires a ground truth trajectory) on production traces without additional labeling.

This is the central challenge of online evaluation. Two ways to partially address it:

1. **Behavioral contracts:** for well-defined task types, you can define invariants — rules that must always hold regardless of which path the agent took. E.g., "for any booking task, `book_flight` must be in the trace." This is a weaker form of ground truth but works at scale.
2. **Production labeling queue:** route a sample of traces to human annotators who provide after-the-fact labels. Expensive but produces high-quality signal that feeds back into the golden dataset.

### What human feedback actually tells you

**Escalation:** the user gave up on the agent and asked a human. This is a strong failure signal but tells you *nothing* about where in the trace the failure occurred. The agent may have failed on step 1 or step 7.

**Retry:** the user submitted the same or a similar request again. Strong evidence the first response was unsatisfactory. Still does not tell you why.

**Thumbs-down:** the user explicitly rated the response negatively. Rare (< 1%) and selection-biased toward extreme dissatisfaction. Valuable when it occurs; not sufficient as a primary signal.

**Implicit positive:** the user accepted the response, moved on, and completed their task. Absent signal — which is the most common case. The absence of negative feedback is weak evidence of success, not strong evidence.

### Sampling strategy

| Trace type | Sample rate | Reason |
|---|---|---|
| Random traces | 5–15% | Representative quality baseline |
| Traces with any error or retry | 100% | Failure investigation |
| Traces flagged by Layer 0 violations | 100% | Policy compliance |
| Traces with explicit negative feedback | 100% | High-signal failure cases |
| Traces from new input patterns (OOD detection) | 100% | Distribution shift monitoring |

### Honest limitation

Online evaluation is noisier than offline. Traffic seasonality, upstream API degradation, UX changes, and user behavior shifts all affect metrics independently of agent quality. A 5% drop in task completion could mean your agent regressed — or it could mean a new user cohort with different query patterns started using the product. You need 2–4 weeks of baseline data before online trends become statistically actionable as release decisions.

---

## Stage 3 — Continuous Evaluation

**Maturity level:** High
**When:** Always on — triggered automatically by metric thresholds, not by human decision
**Dataset:** Living mix — golden (core regression) + labeled production failures + synthetic (new edge cases)
**Signal:** All of the above + RLHF / preference learning when scale is available
**Human involvement:** Low — review findings, approve significant changes

---

### What it looks like

Evaluation no longer requires a human to initiate it. Metric dashboards watch for threshold breaches. When one fires, an automated workflow activates: it exports the failing traces, clusters them by failure type, runs targeted evaluation, and surfaces a structured root cause report. A human reviews the report and decides whether to act — but the investigation has already been done.

At the highest maturity level, the loop closes all the way back to the model: preference signals from A/B experiments and labeled production failures are fed into a reward model, which guides continuous fine-tuning or policy updates.

### The feedback loop

```
Production failure detected (metric breach)
          │
          ▼
Automated failure cluster analysis
  ├── Export traces from past 24h matching failure pattern
  ├── Run LLM-as-judge on cluster
  ├── Compare against golden dataset — is this a regression or new?
  └── Generate structured report: what failed, where, since when

          │
          ▼
Human reviews report (low effort — findings already structured)
          │
    ┌─────┴──────────────────────────────────┐
    │ Fix: prompt update, tool schema fix,   │
    │      new test case added to golden     │
    └─────┬──────────────────────────────────┘
          │
          ▼
Golden dataset grows  ←── Every production failure becomes a regression test
```

This is the self-reinforcing property that defines mature evaluation: the system gets better at finding its own bugs over time.

### The role of RLHF at this stage

RLHF enters the picture when three conditions are met:
1. You have sufficient volume of pairwise preference data (thousands of comparisons, not dozens)
2. You have a stable, clear definition of "better" — not "higher task completion" in the abstract, but concrete, validated preference criteria for your domain
3. You have the infrastructure to run iterative fine-tuning or policy updates without destabilizing the production agent

When these conditions hold, RLHF closes the loop in a way that no other signal type can: it changes *how the model behaves*, not just how you measure it.

**RLAIF as a practical on-ramp:** If you cannot afford the annotation cost of human pairwise comparisons, RLAIF (AI feedback) is a viable alternative. A strong LLM generates preference labels at < $0.01 per data point vs $1+ for human annotation, with comparable alignment in non-specialized domains. Use RLAIF to build initial preference data; supplement with human review for high-stakes cases.

**Reward hacking — the main risk at this stage:** Optimizing a reward model shifts the agent's behavior toward maximizing the reward, not toward being genuinely better. The reward model is a proxy for what you want — and like all proxies, it can be gamed. Symptoms: the agent learns to produce outputs that score well on the reward model but fail in ways not captured by it. Mitigations: diversify reward signal sources, monitor for unexpected behavioral shifts, run red-teaming against the reward-optimized agent.

### Living dataset management

At this stage, the dataset is no longer something you manually maintain. It has a lifecycle:

| Event | Dataset action |
|---|---|
| Production failure triaged | Add to golden dataset as regression test |
| New scenario type detected (OOD) | Generate synthetic variants and add to dataset |
| Old scenario type disappears from traffic | Archive (don't delete — regression value remains) |
| Domain or policy changes | Update golden cases; re-label affected synthetic cases |
| Model or prompt version change | Re-run affected golden cases; check for drift |

The dataset version is tracked alongside the model version, the prompt version, and the agent code version. An evaluation run is meaningful only in the context of all four.

---

## The Consolidating Matrix

| Dimension | **Stage 1 — Offline** | **Stage 2 — Online** | **Stage 3 — Continuous** |
|---|---|---|---|
| **When** | Pre-deploy, on every change | Post-deploy, always async | Always-on + threshold-triggered |
| **Dataset** | Golden + synthetic | Live production traces | Living mix: golden + production + synthetic |
| **Ground truth** | Yes — pre-defined | Partial — behavioral contracts only | Mixed — growing golden + reward model |
| **Signal** | Ground truth + LLM judge | LLM judge (sampled) + implicit feedback | All signals + RLHF when scale allows |
| **Human involvement** | High — design, label, review | Medium — triage alerts, annotate samples | Low — review automated findings |
| **What you evaluate** | Final output + full trace | Full trace (sampled) | Full trace (all) + aggregate trends |
| **Main value** | Catch regressions before users | Catch drift and novel failures after deploy | Identify root cause automatically; close loop to model |
| **Main risk** | Dataset doesn't reflect real distribution | Noise; no ground truth for most traces | Reward hacking; automation overconfidence |
| **Prerequisite** | A curated dataset | Distributed tracing instrumented | Stable online eval + enough preference data |

---

## One Honest Closing Point

These three stages are not sequential checkpoints you move through once. A mature organization runs all three simultaneously. Offline evaluation never stops — you still run regression tests on every deploy. Online evaluation continues forever — the real distribution never stops changing. Continuous evaluation builds on both and closes the loop.

The maturity is not in which stage you are in. It is in how much of the pipeline is automated, how tight the feedback loop is, and how consistently evaluation decisions are grounded in business KPIs rather than benchmark scores.

An organization with a small but well-curated golden dataset, clean tracing instrumentation, and a single automated alert on task completion degradation is more mature than one with a sprawling benchmark suite that no one checks before deploying.

---

## References

- [Building a Golden Dataset for AI Evaluation — Maxim AI](https://www.getmaxim.ai/articles/building-a-golden-dataset-for-ai-evaluation-a-step-by-step-guide/)
- [Golden Dataset: Role in Custom LLM Evals — Arize AI](https://arize.com/resource/golden-dataset/)
- [Golden Datasets: Evaluating Fine-Tuned LLMs — Sigma AI](https://sigma.ai/golden-datasets/)
- [Golden Datasets: Creating Evaluation Standards — Statsig](https://www.statsig.com/perspectives/golden-datasets-evaluation-standards)
- [The Definitive Guide to Synthetic Data Generation Using LLMs — Confident AI](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms)
- [How to Create LLM Test Datasets with Synthetic Data — Evidently AI](https://www.evidentlyai.com/llm-guide/llm-test-dataset-synthetic-data)
- [Generate, Evaluate, Iterate: Synthetic Data for Human-in-the-Loop Refinement of LLM Judges — arXiv 2511.04478](https://arxiv.org/abs/2511.04478)
- [Beyond Thumbs Up and Thumbs Down: A Human-Centered Approach to Evaluation Design — Microsoft Data Science Blog](https://medium.com/data-science-at-microsoft/beyond-thumbs-up-and-thumbs-down-a-human-centered-approach-to-evaluation-design-for-llm-products-d2df5c821da5)
- [User Feedback in LLM-Powered Applications — Winder AI](https://winder.ai/user-feedback-llm-powered-applications/)
- [RLHF 101: A Technical Tutorial — CMU ML Blog](https://blog.ml.cmu.edu/2025/06/01/rlhf-101-a-technical-tutorial-on-reinforcement-learning-from-human-feedback/)
- [RLHF Deciphered: A Critical Analysis — ACM Computing Surveys](https://dl.acm.org/doi/10.1145/3743127)
- [Reinforcement Learning from Human Feedback — arXiv 2504.12501](https://arxiv.org/html/2504.12501v3)
- [How to Evaluate Your Agent with Trajectory Evaluations — LangSmith Docs](https://docs.langchain.com/langsmith/trajectory-evals)
- [Evaluating Production Traces — MLflow](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/)
- [NeurIPS Traxgen: Ground-Truth Trajectory Generation for AI Agent Evaluation](https://neurips.cc/virtual/2025/127975)
- [Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents — arXiv 2510.02837](https://arxiv.org/abs/2510.02837)
- [From First Eval to Autonomous AI Ops: A Maturity Model — Arize AI](https://arize.com/blog/from-first-eval-to-autonomous-ai-ops-a-maturity-model-for-ai-evaluation/)
- [Offline vs Online AI Evaluation: When to Use Each — Label Studio](https://labelstud.io/learningcenter/offline-evaluation-vs-online-evaluation-when-to-use-each/)
- [How to A/B Test AI Agents with a Bayesian Model — Parloa Labs](https://www.parloa.com/labs/research/ai-agent-testing/)
- [A Practical Guide for Evaluating LLMs and LLM-Reliant Systems — arXiv 2506.13023](https://arxiv.org/html/2506.13023v1)
