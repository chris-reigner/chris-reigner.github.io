# Evaluating AI Agents (Under review)

Evaluating AI agents is fundamentally harder than evaluating LLMs.
An agent produces a **trajectory** — a sequence of decisions, tool calls, sub-agent handoffs, and reasoning steps — before you ever see a final response. Any step in that chain can fail silently, and the final output may look correct even when the path to it was wrong.

We cover here some evaluation strategy an how to think about agent evaluation systematically: the three evaluation layers, the metrics that matter at each layer, the tools available, and a concrete worked example showing how to run these metrics against a real agent.
Experience shows setting up the evaluation strategy from the beginning helps save days of work later on.

---

## Define an Evaluation Strategy

Before picking metrics or frameworks, start with a simple question: **what is this agent supposed to do for the business?**

In an enterprise context, AI agents are not research projects — they are deployed to move a specific needle: reduce operational cost, increase throughput, improve customer satisfaction, cut resolution time. Every evaluation strategy should start from those goals and work downward to technical metrics, not the other way around.

If you start with technical metrics ("let's track faithfulness and task completion"), you may end up with an agent that scores well on a benchmark but doesn't actually move the KPI the business cares about.

---

### Step 1 — Name the Business Goal

Write it in one sentence. Be specific.

> ❌ "Improve our customer support with AI"
> ✓ "Reduce tier-1 support ticket resolution time from 8 minutes to under 3 minutes"
>
> ❌ "Automate procurement"
> ✓ "Process 80% of purchase orders under €10,000 without human review"
>
> ❌ "Use AI for onboarding"
> ✓ "Reduce time-to-first-commit for new engineers from 5 days to 2 days"

The more specific the business goal, the more obvious the KPI becomes.

---

### Step 2 — Define the Business KPI

This is the number the business will report. It lives in a dashboard that a manager or stakeholder looks at. It has nothing to do with LLMs — it's a business metric.

| Business goal | Business KPI |
|---|---|
| Reduce support resolution time | Average handle time (AHT) per ticket |
| Automate procurement | % of POs processed without human intervention |
| Improve onboarding | Time-to-first-commit (days) |
| Reduce code review backlog | PR review turnaround time (hours) |
| Increase sales conversion | % of leads contacted within 1 hour of signup |

---

### Step 3 — Map Business KPIs to Technical KPIs

The business KPI tells you *what* to move. Technical KPIs tell you *why* it's not moving — or *how* to verify the agent is on the right trajectory. This is the bridge.

For each business KPI, define 2–4 technical KPIs that are necessary (but not sufficient) conditions for it.

**Example: Customer Support Agent**

| Business KPI | Technical KPI | Why it's linked |
|---|---|---|
| Average handle time < 3 min | Task completion rate > 95% | Agent that can't complete tasks forces human escalation, blowing the AHT |
| Average handle time < 3 min | Agent latency P95 < 8s | Slow responses directly inflate handle time |
| Average handle time < 3 min | Tool correctness > 90% | Wrong tool calls cause retries and loops, adding time |
| Escalation rate < 10% | Intent resolution > 90% | Misunderstood requests always escalate |

**Example: Procurement Agent**

| Business KPI | Technical KPI | Why it's linked |
|---|---|---|
| 80% of POs processed without review | Task completion rate > 85% | Incomplete runs go to human queue |
| 80% of POs processed without review | Argument correctness > 95% | Bad field values (amounts, vendors) trigger compliance holds |
| Zero unauthorized approvals | Faithfulness > 99% | Agent must only act on verified data — no hallucinated approvals |
| Zero unauthorized approvals | Hallucination rate < 0.1% | Single hallucinated vendor ID could approve a fraudulent PO |

---

### Step 4 — Set Thresholds, Not Just Directions

A metric without a threshold is an observation, not a criterion. Define the minimum acceptable value for each technical KPI before you build — not after.

```
✓  Task completion rate  ≥ 90%   (below this, SLA is broken)
✓  Tool correctness      ≥ 88%   (below this, error rate spikes)
✓  P95 latency           ≤ 10s   (above this, UX is unacceptable)
✗  "Task completion should be high"
```

This also gives you a release gate: the agent ships when all thresholds are met. No subjective debates.

---

## The Three Evaluation Layers

Agent evaluation organizes naturally into three layers, each answering a different question:

```
┌─────────────────────────────────────────────────────┐
│  Layer 3 — AGENTIC                                  │
│  Did the agent accomplish the user's goal?          │
│  Was the plan coherent? Was the path efficient?     │
├─────────────────────────────────────────────────────┤
│  Layer 2 — TOOL                                     │
│  Were the right tools called?                       │
│  Were inputs correct? In the right order?           │
├─────────────────────────────────────────────────────┤
│  Layer 1 — MODEL                                    │
│  Is the underlying LLM performing well?             │
│  Faithfulness, reasoning quality, hallucination     │
└─────────────────────────────────────────────────────┘
```

These layers are not redundant — a failure at Layer 1 (hallucinated tool arguments) looks different from a failure at Layer 2 (right tool, wrong order) which looks different from a failure at Layer 3 (right path, wrong goal understood). You need all three to diagnose agent failures accurately.

---

## Layer 1 — Model Evaluation

This layer evaluates the **underlying LLM** powering the agent: its reasoning quality, faithfulness to retrieved context, and tendency to hallucinate. These are pre-agentic metrics — they apply to the model regardless of how it is orchestrated.

### Why it matters for agents

A weak base model that hallucinates tool arguments or misreads retrieval results will fail silently inside an agent pipeline. Model-level issues are the hardest to debug from agent-level metrics alone.

### Metrics

#### Faithfulness

Measures whether the model's response is grounded in the provided context — retrieved documents, tool outputs, or memory — without fabricating information.

- **Input**: retrieved context + model response
- **Score**: fraction of claims in the response that are supported by the context (0–1)
- **Failure signal**: response includes facts not present in any retrieved source

#### Answer Relevancy

Measures whether the response actually addresses the user's input rather than drifting to tangential content.

- **Input**: user query + model response
- **Score**: semantic alignment between query and response (0–1, via LLM-as-judge or embedding cosine similarity)
- **Failure signal**: response is coherent but answers a different question

#### Hallucination Rate

The fraction of responses that contain at least one fabricated factual claim. Operationalized through FactScore — decompose the response into atomic claims, then verify each against a source.

- **Failure signal**: agent states a flight price, booking ID, or entity that doesn't exist in any tool output

#### Reasoning Quality

Evaluates whether the model's reasoning trace (if visible via CoT or scratchpad) is logically sound: premises are valid, conclusions follow, no contradictions.

- **Input**: reasoning trace
- **Score**: LLM-as-judge rubric (0–10 or categorical: SOUND / FLAWED / CIRCULAR)

#### Noise Robustness

Measures whether model performance degrades when irrelevant context is added to the prompt. Relevant for agents with large tool outputs injected into context.

---

## Layer 2 — Tool Evaluation

This layer evaluates the **action layer**: how the agent selects, calls, and sequences tools. Tool failures are often the most impactful because they block downstream steps and can cascade into complete task failure.

### Tool Evaluation Metrics

#### Tool Correctness

Validates that the agent called the **right tools** for the task — no more, no less. Measured by comparing the actual tool call sequence to a ground-truth reference set.

Evaluation dimensions:

- **Tool name match**: was the correct tool selected?
- **Parameter correctness**: were inputs well-formed and accurate?
- **Ordering**: for tasks where sequence matters, were tools called in the right order?

```
ground_truth_tools = ["search_flights", "check_seat_availability", "book_flight"]
actual_tools       = ["search_flights", "check_seat_availability", "book_flight"]  ✓

ground_truth_tools = ["search_flights", "book_flight"]
actual_tools       = ["search_hotels", "book_flight"]                               ✗  (wrong tool)
```

Implementations (e.g. DeepEval's `ToolCorrectnessMetric`) support configurable strictness: exact ordering vs. order-independent matching, and exact count vs. frequency tolerance.

#### Argument Correctness

Even when the right tool is called, the arguments may be wrong. This metric evaluates whether the tool inputs match expected values or fall within valid ranges.

```json
// Expected
{"destination": "London", "date": "2025-06-15", "cabin_class": "economy"}

// Actual (hallucinated date)
{"destination": "London", "date": "2025-02-30", "cabin_class": "economy"}  ✗
```

- **Score**: exact match for enumerables; semantic similarity for free-text fields; range check for numeric fields

#### Tool Efficiency

Measures whether the agent took the **most direct path** via tool calls — no unnecessary calls, no redundant lookups, no loops.

- **Input**: tool call sequence + task description + list of available tools
- **Evaluation**: LLM-as-judge asks "was this the minimal effective sequence of tool calls for this task?"
- **Failure signal**: agent called `search_flights` three times with identical parameters, or called `get_weather` for a booking task where weather is irrelevant

#### Tool Failure Recovery

Assesses how the agent handles a tool returning an error, timeout, or unexpected output.

- Does it retry intelligently (with modified parameters)?
- Does it fall back to an alternative tool?
- Does it surface the failure to the user rather than hallucinating a result?

---

## Layer 3 — Agentic Evaluation

This is the top layer — evaluating the agent as a whole system. It answers whether the agent understood the user's goal, pursued it coherently, and completed it successfully. These metrics require observing the **full execution trace**, not just individual steps.

### Agentic Evaluation Metrics

#### Task Completion

Binary or graded measure of whether the agent accomplished the user's stated goal by the end of the run.

- **Binary**: did the agent complete the task? (yes/no, LLM-as-judge)
- **Graded**: to what degree was the task completed? (0–1 scale, useful when tasks are partially completable)
- **Input**: user request + full execution trace (all steps, tool calls, outputs)
- **Failure signal**: agent ran out of steps, got stuck in a loop, or produced output that doesn't satisfy the original request

#### Task Adherence / Intent Resolution

Evaluates whether the agent correctly understood the user's **underlying intent** — not just the surface request — and oriented its plan accordingly.

Critical for catching failures where the agent appears to complete a task but addresses the wrong objective (e.g. user asks "find me cheap flights" and the agent books first-class).

- **Input**: user request + first N agent actions
- **Score**: alignment between inferred goal and actual plan (LLM-as-judge)

#### Trajectory Quality

Evaluates the entire sequence of reasoning and action steps for coherence, efficiency, and correctness — independent of the final output. This is the "glass-box" metric.

Sub-dimensions:

- **Coherence**: do consecutive steps follow logically from each other?
- **Loop detection**: does the agent ever repeat the same action without new information?
- **Recovery quality**: when a step fails, does the agent adapt appropriately?
- **Progress rate**: at each step, is the agent closer to the goal? (AgentBoard metric)

#### Plan Quality

For planning agents that generate an explicit plan before executing, this metric scores the plan itself: is it logically complete, correctly ordered, and achievable with available tools?

- **Score**: LLM-as-judge rubric covering completeness, feasibility, and ordering (0–10)
- **Evaluated**: before execution begins — catches plan-level failures before they cause step-level failures

#### Plan Adherence

Did the agent follow its own plan? An agent that generates a good plan but then deviates from it is a reliability risk.

- **Score**: fraction of plan steps actually executed in the order planned
- **Failure signal**: agent generated a 5-step plan, executed steps 1–2, then skipped to step 5

#### Step Efficiency

Measures the ratio of useful steps to total steps. Steps that don't advance the goal (redundant lookups, excessive self-questioning, repeated summarization) waste tokens and add latency.

```
efficiency = goal_advancing_steps / total_steps
```

#### Handoff Accuracy (Multi-Agent)

For systems where agents delegate to sub-agents, this metric evaluates whether the orchestrator correctly identified when to hand off, to which agent, and with what context.

---

## Metrics Summary

| Layer | Metric | Type | How scored |
|---|---|---|---|
| **Model** | Faithfulness | Continuous (0–1) | Claims vs. context |
| **Model** | Answer Relevancy | Continuous (0–1) | Semantic similarity |
| **Model** | Hallucination Rate | Rate | FactScore (atomic claim verification) |
| **Model** | Reasoning Quality | Categorical / 0–10 | LLM-as-judge on CoT trace |
| **Tool** | Tool Correctness | Binary / Continuous | Set comparison vs. ground truth |
| **Tool** | Argument Correctness | Continuous | Field-level match vs. expected |
| **Tool** | Tool Efficiency | Continuous (0–1) | LLM-as-judge on call sequence |
| **Tool** | Failure Recovery | Categorical | Pattern matching + LLM-as-judge |
| **Agentic** | Task Completion | Binary / Graded | LLM-as-judge on full trace |
| **Agentic** | Intent Resolution | Continuous (0–1) | LLM-as-judge on early steps vs. goal |
| **Agentic** | Trajectory Quality | Multi-dimensional | LLM-as-judge (coherence, loops, recovery) |
| **Agentic** | Plan Quality | 0–10 | LLM-as-judge on plan |
| **Agentic** | Plan Adherence | Rate | Plan steps executed / total planned |
| **Agentic** | Step Efficiency | Rate | Useful steps / total steps |

---

## How to Run These Metrics

There are two fundamentally different ways to compute evaluation metrics, and the choice between them has real consequences for cost, speed, reliability, and what you can actually measure.

---

### Method 1 — Deterministic (Rule-Based)

Deterministic metrics are computed by code, not by a model. Given the same input, they always produce the same output. They are fast (microseconds to milliseconds), cheap (no API calls), and fully reproducible.

**What can be measured deterministically:**

| Metric | How it's computed |
|---|---|
| Tool correctness | Set intersection: `actual_tools == expected_tools` |
| Argument correctness | Field-level comparison: exact match, regex, range check |
| Step count / efficiency | `len(actual_steps)` vs `len(expected_steps)` |
| Tool call count | Count occurrences of each tool name in the trace |
| Response format validity | JSON schema validation, regex pattern match |
| Latency | Wall-clock time per step and end-to-end |
| Token count | `len(tokenizer.encode(text))` |
| Required field presence | Check output contains `confirmation_id`, `status`, etc. |
| PII presence | Regex or NER scan for emails, phone numbers, SSNs |
| Forbidden content | String match against a blocklist |

These checks are the **cheapest signal you will ever get**. They should run on every trace, in CI, before you ever invoke an LLM judge. A tool correctness check that catches a bug 100% of the time costs nothing compared to an LLM judge that catches it 85% of the time and costs $0.02 per call.

**Practical example — tool correctness check in Python:**

```python
def check_tool_correctness(actual_calls: list[str], expected_calls: list[str]) -> float:
    actual_set   = set(actual_calls)
    expected_set = set(expected_calls)

    true_positives  = len(actual_set & expected_set)   # called and expected
    false_positives = len(actual_set - expected_set)   # called but not expected
    false_negatives = len(expected_set - actual_set)   # expected but not called

    precision = true_positives / len(actual_set)   if actual_set   else 0.0
    recall    = true_positives / len(expected_set) if expected_set else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return f1

# Example
actual   = ["search_flights", "get_weather", "check_seat_availability", "book_flight"]
expected = ["search_flights", "check_seat_availability", "book_flight"]

score = check_tool_correctness(actual, expected)
# precision = 3/4 = 0.75, recall = 3/3 = 1.0, F1 = 0.857
```

**When deterministic is not enough:**

Deterministic checks fail as soon as the "correct" answer has more than one valid form. Did the agent correctly summarize the issue? Was the tone appropriate? Did the response address the user's underlying intent? These cannot be reduced to a rule. That's where LLM-as-judge comes in.

---

### Method 2 — LLM-as-a-Judge

LLM-as-a-judge means using a separate LLM — not the agent itself — to evaluate a trace, output, or step against a rubric. The judge receives a carefully crafted prompt containing the evaluation criteria and the content to assess, and returns a score, a label, or a verdict with reasoning.

**Basic pattern:**

```python
def llm_judge(trace: str, criteria: str, model: str = "gpt-4o") -> dict:
    prompt = f"""You are an objective evaluator of AI agent behavior.

Evaluation criteria:
{criteria}

Agent trace to evaluate:
{trace}

Think step by step, then provide:
- score: float between 0.0 and 1.0
- reasoning: one sentence explaining your score

Respond in valid JSON only."""

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,          # determinism matters here
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

LLM-as-judge is the right tool when the evaluation requires genuine language understanding: coherence, intent alignment, tone appropriateness, trajectory quality, reasoning soundness. It scales to any criteria you can express in natural language.

**But be honest about what it is.** LLM-as-judge is a *proxy* for human judgment — not a replacement for it.

#### The Real Biases You Need to Know

Research ([arXiv 2410.02736](https://arxiv.org/abs/2410.02736), [arXiv 2406.07791](https://arxiv.org/abs/2406.07791)) has identified and quantified several systematic biases:

| Bias | What it does | Impact |
|---|---|---|
| **Verbosity bias** | Prefers longer responses regardless of quality | An agent that adds irrelevant sentences scores higher |
| **Position bias** | Favors content placed first or last in a prompt | Simply reordering options shifts accuracy > 10% |
| **Self-enhancement bias** | Prefers output from the same model family as the judge | GPT-4o judging GPT-4o output is not neutral |
| **Stochastic variance** | Same input, different run → different score | Scores shift ±0.1 even at temperature=0 across different days |
| **Domain expertise gap** | Agreement with human experts drops to 60–68% in specialized domains (law, medicine, dietetics) | Do not use generic judges for compliance-sensitive domains without calibration |

These are not edge cases — they are documented, reproducible phenomena. An evaluation pipeline that ignores them will generate misleading results.

#### Mitigations

- **Set temperature=0** on the judge. It doesn't eliminate variance but reduces it.
- **Ask for reasoning before the score** (chain-of-thought). This measurably improves agreement with human labels.
- **Use odd-numbered scales** (1–5, 1–10) to avoid even-scale central tendency.
- **Calibrate against a human-labeled gold set** before trusting the judge. Target Cohen's κ > 0.7.
- **Use a different model family as judge** than the one running the agent. A Claude judge evaluating a GPT-4o agent is more neutral than GPT-4o judging itself.
- **Ensemble multiple judges** and average — reduces variance, surfaces disagreement.

---

### Choosing Between the Two

| Signal you want | Use |
|---|---|
| Did the agent call the right tools? | Deterministic |
| Were the tool arguments correct? | Deterministic |
| Did the response contain required fields? | Deterministic |
| Was PII present in the output? | Deterministic |
| Did the agent stay under token budget? | Deterministic |
| Did the agent understand the user's intent? | LLM-as-judge |
| Was the reasoning coherent? | LLM-as-judge |
| Was the response tone appropriate? | LLM-as-judge |
| Was the trajectory efficient? | LLM-as-judge |
| Was the task genuinely completed? | LLM-as-judge |

**The pragmatic rule:** reach for deterministic first. If you can write a rule that correctly classifies > 95% of your cases, use it. Only use LLM-as-judge for the residual evaluation surface that rules genuinely cannot cover. Every LLM judge call has cost, latency, and uncertainty attached to it.

A good production pipeline runs deterministic checks on every trace (zero added latency, happens in-process), then samples a fraction of traces for LLM-as-judge scoring asynchronously. You do not need to judge every call in real time.

---

### Layer 0 — Business Rules (The Floor)

The three evaluation layers covered earlier (Model, Tool, Agentic) are about *quality*. There is a layer below them that is not about quality — it is about **compliance, safety, and business policy**. These are pass/fail, non-negotiable rules that must hold before quality even matters.

Call it **Layer 0: Business Rules**.

These are always deterministic. They check whether the agent's output is *permissible*, not whether it is *good*.

**Examples by enterprise context:**

| Domain | Business rule | Check |
|---|---|---|
| **Finance** | Agent must never recommend a specific stock | Output does not contain ticker symbols + buy/sell verbs |
| **Finance** | All monetary amounts must reference the source tool output | Every `€X` in response traceable to a tool result |
| **Legal** | Agent must not give legal advice | Output does not contain "you should sign", "you are liable", etc. |
| **Healthcare** | No medical diagnosis or drug recommendations | Output scanned against diagnosis and drug name lists |
| **HR** | Agent must not comment on protected characteristics | PII filter: age, gender, ethnicity not referenced in output |
| **Any** | No PII in logs or responses | Regex / NER scan for emails, SSNs, phone numbers, credit card numbers |
| **Any** | Response must stay within defined scope (no off-topic) | Intent classifier: is the response within the agent's defined domain? |
| **Any** | Agent must not impersonate a human | Output does not claim "I am a human" or deny being an AI |

These checks are typically implemented as a **guardrail pipeline** that runs synchronously on every output before it is returned to the user — not as part of an offline evaluation batch.

**How they integrate with evaluation:**

```
Agent output
     │
     ▼
Layer 0: Business Rules   ← deterministic, sync, blocks output if failed
     │ pass
     ▼
Layer 1: Model quality    ← faithfulness, hallucination
     │
     ▼
Layer 2: Tool quality     ← tool correctness, argument correctness
     │
     ▼
Layer 3: Agentic quality  ← task completion, trajectory quality
```

A Layer 0 failure is **not a quality issue** — it is a policy violation. It should be logged separately, trigger a different alert, and potentially block deployment. Mixing it with quality metrics obscures both.

**Practical implementation:**

```python
FORBIDDEN_PATTERNS = [
    r"\b(buy|sell|short)\s+\$?[A-Z]{1,5}\b",   # stock recommendations
    r"\b\d{3}-\d{2}-\d{4}\b",                   # SSN
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
]

def check_business_rules(response: str) -> list[dict]:
    violations = []
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, response):
            violations.append({"rule": pattern, "status": "VIOLATED"})
    return violations

# In your eval pipeline
violations = check_business_rules(agent_output)
if violations:
    log_policy_violation(violations)
    return block_response()   # or substitute a safe fallback
```

> **Critical note:** Business rules are not a substitute for good agent design. If your agent regularly violates Layer 0 rules, the fix is in the system prompt, tool design, or model fine-tuning — not in tighter filters. Filters catch what slips through; they do not fix the underlying behavior.

---

## Worked Example: Evaluating a Travel Booking Agent

The following example walks through a complete evaluation of a travel booking agent that has four tools: `search_flights`, `check_seat_availability`, `get_weather`, and `book_flight`.

### The Test Case

**User input:**
> "Book me the cheapest economy flight from Paris to New York on June 15th."

**Ground truth (expected behavior):**

| Step | Tool | Purpose |
|---|---|---|
| 1 | `search_flights` | Find available economy flights CDG → JFK on June 15 |
| 2 | `check_seat_availability` | Verify seats on the cheapest result |
| 3 | `book_flight` | Confirm the booking |

**What the agent actually did (execution trace):**

```
Step 1 — Reasoning:
  "I need to search for flights from Paris to New York on June 15."
  Tool:    search_flights(origin="CDG", destination="NYC", date="2025-06-15", cabin="economy")
  Result:  [{"id": "AF001", "price": 480}, {"id": "AA202", "price": 610}, ...]

Step 2 — Reasoning:
  "Let me also check the weather in New York to give the user useful context."
  Tool:    get_weather(city="New York", date="2025-06-15")
  Result:  {"temp": 24, "condition": "sunny"}

Step 3 — Reasoning:
  "AF001 is cheapest at €480. I should check seat availability before booking."
  Tool:    check_seat_availability(flight_id="AF001")
  Result:  {"available": true, "seats_left": 4}

Step 4 — Reasoning:
  "Seats available. Proceeding to book."
  Tool:    book_flight(flight_id="AF001")
  Result:  {"confirmation": "XY4821", "status": "confirmed"}

Final response:
  "I've booked flight AF001 (Air France, Paris CDG → New York JFK) on June 15th
   for €480 economy. Confirmation number: XY4821.
   The weather in New York that day will be sunny and 24°C — enjoy your trip!"
```

---

### Layer 1 — Model Scores

**Faithfulness: 1.0**

Every factual claim in the response is traceable to a tool output:

| Claim | Source |
|---|---|
| "AF001", "Air France" | `search_flights` result |
| "Paris CDG → New York JFK" | Derived from origin/destination inputs |
| "€480 economy" | `search_flights` result |
| "XY4821" | `book_flight` result |
| "sunny and 24°C" | `get_weather` result |

No hallucination. Score: 1.0.

**Answer Relevancy: 0.90**

The response directly answers the booking request with confirmation details. Minor deduction: weather information was not requested and represents scope creep. The agent added value it wasn't asked for. Score: 0.90.

**Reasoning Quality: 7/10**

Steps 1, 3, and 4 show clean causal reasoning. Step 2 is the weak point: "to give the user useful context" is a rationalization, not a logical necessity of the task. The reasoning is sound overall but reveals imprecise goal representation. Score: 7/10.

---

### Layer 2 — Tool Scores

**Tool Correctness: 0.75**

```
Expected:  [search_flights, check_seat_availability, book_flight]
Actual:    [search_flights, get_weather, check_seat_availability, book_flight]
```

All 3 required tools were called. 1 extra tool (`get_weather`) was called that is not part of the expected set. Scoring with order-independent matching: 3 correct / 4 called = 0.75.

**Argument Correctness: 0.85**

Examining `search_flights` arguments field by field:

| Parameter | Expected | Actual | Match |
|---|---|---|---|
| `origin` | `"CDG"` | `"CDG"` | ✓ |
| `destination` | `"JFK"` | `"NYC"` | ✗ (city code vs. airport code) |
| `date` | `"2025-06-15"` | `"2025-06-15"` | ✓ |
| `cabin` | `"economy"` | `"economy"` | ✓ |

`"NYC"` vs. `"JFK"` is a real bug — many flight APIs require IATA airport codes. The tool may have silently accepted it, or searched all NYC-area airports. Averaging argument correctness across all four calls: ~0.85.

**Tool Efficiency: 0.60**

LLM-as-judge prompt:

```
Task: "Book the cheapest economy flight from Paris to New York on June 15."
Available tools: search_flights, check_seat_availability, get_weather, book_flight
Tools called: search_flights → get_weather → check_seat_availability → book_flight

Was this the most efficient sequence to complete the task? Score 0–10.
```

Judge output:
> "The minimum effective sequence for this task is: search_flights →
> check_seat_availability → book_flight. The get_weather call serves no
> booking purpose — it adds latency and tokens without advancing the goal.
> Score: 6/10."

Normalized: 0.60.

**Tool Failure Recovery: N/A**

No tool failures occurred in this run.

---

### Layer 3 — Agentic Scores

**Task Completion: 1.0**

LLM-as-judge:
> "User requested booking the cheapest economy flight from Paris to New York on
> June 15. The agent identified the cheapest option (AF001, €480), verified
> availability, and completed the booking (XY4821). Task fully accomplished."

Score: 1.0.

**Intent Resolution: 0.90**

The agent correctly inferred: book (not just search), economy class, cheapest available, June 15. Minor deduction for the weather detour, which suggests the agent's goal representation included "be helpful broadly" beyond the stated task. Score: 0.90.

**Trajectory Quality:**

| Dimension | Score | Notes |
|---|---|---|
| Coherence | 0.90 | Steps 1, 3, 4 follow directly; Step 2 is a mild deviation |
| Loop detection | 1.00 | No repeated tool calls |
| Recovery quality | N/A | No failures in this run |
| Progress rate | 0.75 | Step 2 (weather) makes zero progress toward the booking goal |

**Step Efficiency: 0.75**

```
Total steps:          4
Goal-advancing steps: 3  (search, check_availability, book)
Off-task steps:       1  (get_weather)

Efficiency = 3 / 4 = 0.75
```

---

### Full Evaluation Dashboard

| Layer | Metric | Score | Threshold | Status |
|---|---|---|---|---|
| Model | Faithfulness | 1.00 | > 0.80 | ✓ Pass |
| Model | Answer Relevancy | 0.90 | > 0.80 | ✓ Pass |
| Model | Reasoning Quality | 7/10 | > 6/10 | ✓ Pass |
| Tool | Tool Correctness | 0.75 | > 0.90 | ✗ Fail |
| Tool | Argument Correctness | 0.85 | > 0.90 | ⚠ Near miss |
| Tool | Tool Efficiency | 0.60 | > 0.80 | ✗ Fail |
| Agentic | Task Completion | 1.00 | > 0.90 | ✓ Pass |
| Agentic | Intent Resolution | 0.90 | > 0.85 | ✓ Pass |
| Agentic | Progress Rate | 0.75 | > 0.85 | ⚠ Near miss |
| Agentic | Step Efficiency | 0.75 | > 0.85 | ✗ Fail |

**Diagnosis:** The agent completed the task correctly and produced a faithful response — it would look fine in a demo. The real problems are at Layer 2:

1. **`get_weather` on a booking task** — systematic tool selection indiscipline. The agent is optimizing for "helpful response" over "task efficiency." Root cause: system prompt or tool descriptions encourage broad helpfulness rather than goal-focused action.
2. **`destination="NYC"` instead of `"JFK"`** — argument quality bug. The tool description likely doesn't specify that IATA airport codes are required, causing the model to use a city name.

Both failures are invisible from the final output alone. Only trajectory evaluation surfaces them.

**Fixes:**

1. Tighten system prompt: *"Call only tools required to complete the user's stated task. Do not call auxiliary tools unless explicitly requested."*
2. Update `search_flights` tool description to specify: *"destination: IATA airport code (e.g. 'JFK', not 'NYC')"*
3. Add regression test that asserts `get_weather` is NOT called for any pure booking task

---

### Running This in Code

**With [DeepEval](https://deepeval.com):**

```python
from deepeval import evaluate
from deepeval.metrics import (
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

test_case = LLMTestCase(
    input="Book me the cheapest economy flight from Paris to New York on June 15th.",
    actual_output=(
        "I've booked flight AF001 (Air France, Paris CDG → New York JFK) on June 15th "
        "for €480 economy. Confirmation: XY4821. Weather in New York: sunny, 24°C."
    ),
    # Context = everything the model had available (tool outputs)
    retrieval_context=[
        '{"id": "AF001", "price": 480, "cabin": "economy"}',
        '{"confirmation": "XY4821", "status": "confirmed"}',
        '{"temp": 24, "condition": "sunny"}',
    ],
    # What the agent actually called
    tools_called=[
        ToolCall(name="search_flights",
                 input={"origin": "CDG", "destination": "NYC",
                        "date": "2025-06-15", "cabin": "economy"}),
        ToolCall(name="get_weather",
                 input={"city": "New York", "date": "2025-06-15"}),
        ToolCall(name="check_seat_availability",
                 input={"flight_id": "AF001"}),
        ToolCall(name="book_flight",
                 input={"flight_id": "AF001"}),
    ],
    # Ground truth: what should have been called
    expected_tools=[
        ToolCall(name="search_flights"),
        ToolCall(name="check_seat_availability"),
        ToolCall(name="book_flight"),
    ],
)

metrics = [
    FaithfulnessMetric(threshold=0.8),
    ToolCorrectnessMetric(
        evaluation_params=[ToolCallParams.TOOL, ToolCallParams.INPUT_PARAMETERS],
        should_consider_ordering=True,
    ),
    TaskCompletionMetric(threshold=0.9, model="gpt-4o", include_reason=True),
]

evaluate(test_cases=[test_case], metrics=metrics)
```

**With [Langfuse](https://langfuse.com) (tracing + dataset eval):**

```python
from langfuse import get_client

langfuse = get_client()

# 1. Create evaluation dataset
dataset = langfuse.create_dataset(name="travel-agent-v1")
langfuse.create_dataset_item(
    dataset_name="travel-agent-v1",
    input={"query": "Book me the cheapest economy flight from Paris to New York on June 15th."},
    expected_output={
        "tools_required": ["search_flights", "check_seat_availability", "book_flight"],
        "must_contain": ["confirmation"],
    },
)

# 2. Define agent wrapper that auto-traces
def run_agent(input_data):
    with langfuse.start_as_current_span(name="travel-agent-run"):
        result = your_travel_agent(input_data["query"])
        return result

# 3. Run experiment — traces every run, stores results for comparison
result = dataset.run_experiment(
    name="travel-agent-gpt4o-temp0",
    task=run_agent,
)

# 4. Score traces with LLM-as-judge
for item in result.items:
    langfuse.score(
        trace_id=item.trace_id,
        name="task_completion",
        value=your_llm_judge(item.output, item.expected_output),
    )
```

**LLM-as-judge scorer (reusable pattern):**

{% raw %}

```python
from anthropic import Anthropic

client = Anthropic()

def score_task_completion(agent_trace: str, user_request: str) -> dict:
    prompt = f"""You are evaluating an AI agent's task execution.

User request: {user_request}

Agent execution trace:
{agent_trace}

Score the agent's task completion on a scale of 0.0 to 1.0.
Rules:
- 1.0: Task fully completed, all user requirements met
- 0.7–0.9: Task mostly completed, minor omissions
- 0.4–0.6: Task partially completed, significant gaps
- 0.0–0.3: Task not completed or wrong objective pursued

Respond in JSON:
{{"score": <float>, "reasoning": "<one sentence explaining the score>"}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    import json
    return json.loads(response.content[0].text)
```

{% endraw %}

---

## Evaluation Tools and Frameworks

| Tool | Best for |
|---|---|
| [DeepEval](https://deepeval.com) | Full agent eval suite: ToolCorrectnessMetric, TaskCompletionMetric, FaithfulnessMetric, G-Eval; Python-native, CI/CD integration |
| [Langfuse](https://langfuse.com) | Tracing + dataset management + experiment comparison; framework-agnostic; open-source |
| [LangSmith](https://smith.langchain.com) | Native LangChain/LangGraph tracing; dataset eval, human annotation workflows |
| [RAGAS](https://docs.ragas.io) | RAG-layer metrics within agents: faithfulness, answer relevancy, context recall |
| [Azure AI Evaluation SDK](https://learn.microsoft.com/azure/ai-studio/how-to/evaluate-generative-ai-app) | Task Adherence, Tool Call Accuracy, Intent Resolution; integrates with Azure AI Foundry |
| [Inspect AI](https://inspect.ai-safety-institute.org.uk/) | Safety-focused agent evaluation (UK AISI); custom solvers and scorers |
| [AgentBench](https://github.com/THUDM/AgentBench) | Multi-environment benchmark: OS, DB, web, code tasks |

---

## Evaluation Temporality: From Offline to Continuous

Evaluation is not a binary switch you flip before launch. It is a practice that deepens as your agent matures. Most teams start offline because it requires no live traffic and has zero user impact. As the agent reaches production, online evaluation becomes possible. The most mature organizations close the loop entirely — evaluation triggers remediation automatically, without human initiation.

Each stage is a genuine prerequisite for the next. You cannot run meaningful online evaluation without a baseline from offline. You cannot build continuous evaluation without the instrumentation that online evaluation requires.

```
Low maturity    ──────────────────────────────────►  High maturity

  OFFLINE              ONLINE                CONTINUOUS
(pre-deploy)         (post-deploy)           (always-on)
fixed dataset        live traffic            automated loop
manual trigger       async scoring           self-healing
```

---

### Stage 1 — Offline Evaluation

**What it is:** Running your agent against a fixed, curated dataset before any user sees it. No live traffic. No deployment risk. You control every input.

**When it happens:** During development, before every release, and as a regression gate in CI/CD.

**What you build:**

A **golden dataset** — a set of test cases with known inputs and expected outputs (or expected tool trajectories). Start small: 50–100 cases covering happy paths, known edge cases, and representative failure modes. Grow it systematically by adding every production failure you encounter.

```
golden_dataset/
├── happy_path/          # standard tasks that must always work
├── edge_cases/          # ambiguous inputs, unusual formats, boundary values
├── adversarial/         # tool failures, contradictory instructions, injection attempts
└── regressions/         # cases that previously failed — must stay fixed
```

**What you run:**

| Check | When | Method |
|---|---|---|
| Deterministic (tool correctness, argument correctness, format) | Every commit | Fast, in-process, < 1s total |
| Layer 3 quality (task completion, trajectory quality) | Every PR merge / daily | LLM-as-judge, batched |
| Full adversarial suite | Pre-release | LLM-as-judge + human review |

**The release gate:** Define minimum thresholds per metric before you start building. The agent ships when all thresholds are met. This prevents the common failure mode of shipping an agent that "looks good" on the demo cases while silently failing on edge cases.

**Honest limitation:** Offline evaluation is necessary but not sufficient. A curated dataset is always a simplified approximation of reality. The distribution of real user inputs will surprise you. Users will phrase things in ways you never anticipated, chain tasks you didn't design for, and surface failure modes your golden dataset doesn't cover. Treat offline scores as a *floor*, not a ceiling.

---

### Stage 2 — Online Evaluation

**What it is:** Scoring your agent's behavior against real production traffic, asynchronously, after deployment.

**When it happens:** After launch, on a continuous sample of live interactions.

**The key constraint:** Online evaluation must never block the user. Every scoring call runs asynchronously, after the response has been returned. You are observing what happened — not gating it.

**What you instrument:**

Distributed tracing across every span: each LLM call, each tool call, each sub-agent handoff. Every trace gets a unique ID. Metadata is attached: user ID, session ID, timestamp, model version, prompt version. This is the raw material for everything that follows.

```python
# Pseudo-code: async evaluation on every Nth trace
import random

def on_agent_trace_complete(trace: AgentTrace):
    # Don't score everything — sample intelligently
    if random.random() < SAMPLE_RATE:          # e.g. 0.10 = 10% of traces
        eval_queue.enqueue(trace)              # non-blocking

# Evaluation worker (runs separately)
def evaluate_worker():
    while True:
        trace = eval_queue.dequeue()
        scores = {
            "tool_correctness":  check_tool_correctness(trace),   # deterministic
            "task_completion":   llm_judge_task_completion(trace), # LLM-as-judge
            "business_rules":    check_business_rules(trace),     # deterministic
        }
        metrics_store.write(trace.id, scores)
        if any_threshold_violated(scores):
            alerting.fire(trace, scores)
```

**Sampling strategy:** You cannot and should not score every trace with LLM-as-judge — it's too expensive. A practical approach:

- **Always:** run deterministic checks (Layer 0 + Layer 2) on 100% of traces in-process
- **Sample 5–15%:** run LLM-as-judge on a random sample for quality trend monitoring
- **Always:** score traces that triggered an error, retry, or escalation
- **Always:** score traces flagged by users (thumbs-down, explicit complaints)

**What you watch:**

| Signal | What it means |
|---|---|
| Task completion rate drops | The agent is failing on an emerging class of inputs |
| Tool error rate spikes | A downstream API changed or is degrading |
| Average step count increases | The agent is getting less efficient — possibly looping |
| Business rule violations appear | A prompt regression or adversarial input pattern |
| Score variance increases | Non-determinism is growing — model or prompt instability |

**Honest limitation:** Online evaluation is noisier than offline. Real traffic has confounders: seasonality, UX changes, upstream API performance, adversarial inputs. A metric drop does not automatically mean the agent regressed — it might mean traffic patterns shifted. You need enough baseline data (typically 2–4 weeks) before online trends become statistically actionable.

---

### Stage 3 — Continuous Evaluation

**What it is:** Evaluation that not only observes but also **closes the loop** — automatically identifying failure modes, triggering investigation, and in some cases proposing or applying fixes without human initiation.

**What separates it from online evaluation:** Online evaluation tells you *that* something is wrong. Continuous evaluation tells you *what* is wrong and *starts doing something about it*.

**The architecture:** A monitor watches metric trends. When a threshold is breached, a webhook fires. An evaluation agent (or a human-on-call workflow) activates, exports the relevant traces, runs targeted analysis, clusters the failure modes, and surfaces structured findings.

```
Production metrics dashboard
          │
          │ threshold breached (e.g. task_completion < 0.85)
          ▼
    Alerting webhook
          │
          ▼
    Evaluation agent
    ├── Export failing traces from past 24h
    ├── Run targeted LLM-as-judge on failure cluster
    ├── Identify: which intent type / tool / scenario is failing?
    ├── Compare against offline golden dataset — is this a regression?
    └── Surface structured report:
           "Tool correctness on flight booking tasks dropped from 0.91 to 0.74
            after the search_flights API update at 14:32 UTC.
            23 affected traces. Recommended: update tool schema in system prompt."
          │
          ▼
    Human review / automated fix
```

**The feedback loop that matters:** Every production failure that gets triaged should feed back into the offline golden dataset. The system is self-reinforcing: production failures become regression tests, regression tests prevent the same failure from recurrence, the golden dataset grows to cover the real distribution of user inputs.

```
Production failure
      │
      ▼
Triage: is this a new failure mode?
      │ yes
      ▼
Add to golden dataset as regression test
      │
      ▼
Offline eval suite now catches this class of failure automatically
```

---

### Making Scores Statistically Meaningful

A score without uncertainty is just a number. At any maturity level, you need to know whether a metric change is **signal or noise** before acting on it.

**The basic problem:** If your task completion rate drops from 0.91 to 0.88, is that a regression or random variance? It depends entirely on how many traces you scored.

**Confidence intervals for evaluation scores:**

For a proportion metric (like task completion, which is 0 or 1 per trace):

```python
import numpy as np
from scipy import stats

def eval_confidence_interval(scores: list[float], confidence: float = 0.95) -> dict:
    n    = len(scores)
    mean = np.mean(scores)
    se   = stats.sem(scores)                     # standard error of the mean
    ci   = stats.t.interval(confidence, df=n-1, loc=mean, scale=se)

    return {"mean": round(mean, 4), "ci_low": round(ci[0], 4),
            "ci_high": round(ci[1], 4), "n": n}

# Example
scores = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0]   # 10 traces, 7 completions
result = eval_confidence_interval(scores)
# {"mean": 0.7, "ci_low": 0.407, "ci_high": 0.993, "n": 10}
```

With only 10 traces, the 95% CI spans from 0.41 to 0.99. That "0.7 task completion rate" is essentially meaningless as a decision signal. **You need at least 100 traces per metric to get a CI narrow enough to be actionable.** For detecting a 5% regression with 80% power, you need ~400 traces.

**Bayesian A/B testing for model comparisons:**

When comparing two versions of your agent (e.g., after a prompt change or model upgrade), frequentist A/B testing requires waiting until a fixed sample size is reached before drawing conclusions. Bayesian testing lets you update continuously as data arrives and express results as probabilities rather than p-values.

A hierarchical Bayesian model ([Parloa Labs](https://www.parloa.com/labs/research/ai-agent-testing/)) can combine deterministic binary metrics and LLM-judge scores into a single framework:

```
P(agent_B > agent_A | data) = 0.93

Credible interval on difference in task completion: [+0.021, +0.043]
```

This is more useful than "p=0.04" because it directly answers the deployment question: "How confident are we that B is better?" And it supports partial pooling by scenario type — you can see that B is better on booking tasks but equivalent on search tasks.

**Weighted composite score:**

When you have multiple metrics, you may want a single decision signal. Assign weights based on business importance:

```python
METRIC_WEIGHTS = {
    "task_completion":    0.40,   # highest weight — directly maps to AHT
    "tool_correctness":   0.25,
    "intent_resolution":  0.20,
    "step_efficiency":    0.15,
}

def composite_score(metrics: dict[str, float]) -> float:
    return sum(metrics[k] * w for k, w in METRIC_WEIGHTS.items()
               if k in metrics)

# composite_score({"task_completion": 0.91, "tool_correctness": 0.75,
#                  "intent_resolution": 0.90, "step_efficiency": 0.75})
# → 0.40*0.91 + 0.25*0.75 + 0.20*0.90 + 0.15*0.75 = 0.847
```

Weights are not universal — they must reflect your business priorities. An agent where a wrong tool call has compliance consequences (e.g. finance, healthcare) should weight tool correctness much higher.

> **Critical caveat:** A composite score hides information. A score of 0.85 could mean everything is good, or it could mean task completion is 1.0 and tool efficiency is 0.5. Always report individual metric scores alongside composites. Use the composite only as a quick-glance signal, never as the sole decision criterion.

---

### Maturity Summary

| | **Offline** | **Online** | **Continuous** |
|---|---|---|---|
| **Data source** | Curated golden dataset | Live production traffic | Live traffic + feedback loop |
| **Trigger** | Manual / CI commit | Always-on async | Metric threshold breach |
| **Scoring method** | Deterministic + LLM-judge batched | Deterministic (100%) + LLM-judge (sampled) | Automated analysis agent |
| **Latency** | Minutes to hours | Milliseconds (async) | Minutes from breach to report |
| **Human involvement** | High | Medium (triage alerts) | Low (review findings only) |
| **Main value** | Catch regressions before users see them | Catch drift after deployment | Identify root cause automatically |
| **Main risk** | Dataset doesn't reflect real distribution | Noise from production variance | False automation confidence |
| **When to adopt** | From day one | After first production deployment | When online eval is stable and trusted |

---

## References

- [Definitive AI Agent Evaluation Guide — Confident AI](https://www.confident-ai.com/blog/definitive-ai-agent-evaluation-guide)
- [LLM Agent Evaluation: Tool Use, Task Completion, Agentic Reasoning — Confident AI](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)
- [AI Agent Evaluation Guide — DeepEval](https://deepeval.com/guides/guides-ai-agent-evaluation)
- [AI Agent Evaluation Metrics — DeepEval](https://deepeval.com/guides/guides-ai-agent-evaluation-metrics)
- [Evaluating Agentic AI Systems: A Deep Dive into Agentic Metrics — Microsoft Azure AI Foundry](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/evaluating-agentic-ai-systems-a-deep-dive-into-agentic-metrics/4403923)
- [Evaluating Agentic AI Systems: Frameworks, Metrics, and Best Practices — Maxim AI](https://www.getmaxim.ai/articles/evaluating-agentic-ai-systems-frameworks-metrics-and-best-practices/)
- [Agent Evaluation: How to Evaluate LLM Agents — Langfuse](https://langfuse.com/guides/cookbook/example_pydantic_ai_mcp_agent_evaluation)
- [LLM Agent Evaluation: Metrics, Methods & Real-World Use Cases — DeepChecks](https://deepchecks.com/llm-agent-evaluation/)
- [Evaluating AI Agents: Real-World Lessons from Amazon — AWS Blog](https://aws.amazon.com/blogs/machine-learning/evaluating-ai-agents-real-world-lessons-from-building-agentic-systems-at-amazon/)
- [Beyond Task Completion: An Assessment Framework for Evaluating Agentic AI Systems — arXiv 2512.12791](https://arxiv.org/html/2512.12791v2)
- [Evaluation and Benchmarking of LLM Agents: A Survey — arXiv 2507.21504](https://arxiv.org/html/2507.21504v1)
- [Evaluations for the Agentic World — McKinsey QuantumBlack](https://medium.com/quantumblack/evaluations-for-the-agentic-world-c3c150f0dd5a)
- [How to Evaluate Agentic AI Pipelines — Stack AI](https://www.stackai.com/blog/how-to-evaluate-agentic-ai-pipelines-metrics-frameworks-and-real-world-examples)
- [LLM-as-a-Judge: Complete Guide — Evidently AI](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [LLM Evaluation Metrics Guide — Braintrust](https://www.braintrust.dev/articles/llm-evaluation-metrics-guide)
- [Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge — arXiv 2410.02736](https://arxiv.org/abs/2410.02736)
- [Judging the Judges: Position Bias in LLM-as-a-Judge — arXiv 2406.07791](https://arxiv.org/abs/2406.07791)
- [A Survey on LLM-as-a-Judge — arXiv 2411.15594](https://arxiv.org/abs/2411.15594)
- [AI Agent Guardrails: Production Guide 2026 — Authority Partners](https://authoritypartners.com/insights/ai-agent-guardrails-production-guide-for-2026/)
- [From First Eval to Autonomous AI Ops: A Maturity Model for AI Evaluation — Arize AI](https://arize.com/blog/from-first-eval-to-autonomous-ai-ops-a-maturity-model-for-ai-evaluation/)
- [Offline vs Online AI Evaluation: When to Use Each — Label Studio](https://labelstud.io/learningcenter/offline-evaluation-vs-online-evaluation-when-to-use-each/)
- [Online vs Offline LLM Evaluation — Milestone](https://mstone.ai/question/difference-between-online-and-offline-llm-evaluation/)
- [How to A/B Test AI Agents with a Bayesian Model — Parloa Labs](https://www.parloa.com/labs/research/ai-agent-testing/)
- [Applying Statistics to LLM Evaluations — Cameron Wolfe](https://cameronrwolfe.substack.com/p/stats-llm-evals)
- [State of Agent Engineering — LangChain](https://www.langchain.com/state-of-agent-engineering)
