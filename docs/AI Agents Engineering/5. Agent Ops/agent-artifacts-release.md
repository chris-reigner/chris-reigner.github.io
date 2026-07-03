# AI Agent Artifact Taxonomy and Release Lifecycle (Under review)

An AI agent is not a monolithic artifact. It is a composition of heterogeneous components, each evolving at its own pace, with its own versioning contract, and its own risk profile when changed. Understanding this composition matters when you think about how to package, promote, and update agents in production.

---

## What Is an Agent Made Of?

Every production agent is a combination of at least eight artifact types. Most failures — and most unnecessary release cycles — come from conflating them.

### Artifact Taxonomy

| Artifact | Description | Change Frequency | Change Risk |
|---|---|---|---|
| **Core Logic** | Business workflow orchestration: routing, branching, state machine | Low | High |
| **System Prompt** | Instructions shaping agent behavior, tone, constraints | High | Medium |
| **LLM Configuration** | Model ID, temperature, top-p, max tokens, stop sequences | Medium | High |
| **Skills** | Packaged, reusable capabilities combining a prompt + task logic; invocable by the orchestration layer | Medium | High |
| **Sub-Agents** | Other agents called as delegates, each with their own artifact lifecycle | Medium | High |
| **Knowledge Base** | RAG corpus, vector store contents, document versions | High | Low–Medium |
| **Memory Store** | Persistent user or session context across runs | Continuous | Medium |
| **Evaluation Dataset** | Golden inputs/outputs used to test the agent | Low–Medium | Low |

---

## Why This Matters for Release Lifecycle

Traditional software promotes a single artifact: a compiled binary or a container image. The version is the deployment unit.

An agent cannot follow this model cleanly. If you must cut a full release and go through your entire CI/CD pipeline every time a prompt is tuned or a document corpus is updated, you will either:

- Move too slowly (blocking prompt iteration behind a 2-week sprint cycle), or
- Move dangerously (skipping validation to ship faster).

The solution is to **decouple artifact lifecycles** while keeping coherence guarantees. Each artifact type gets its own versioning contract, its own promotion gate, and its own update path.

---

## The Agent Release Lifecycle

### Standard Stages

```
Dev → Staging → Production
```

When the full agent (all artifacts) is promoted together — typically on initial launch or after a core logic change — every artifact goes through all stages. But most ongoing updates touch only one layer.

| Stage | Purpose | Gate Criteria |
|---|---|---|
| **Dev** | Iterate freely | No gate |
| **Staging** | Offline evaluation | Golden dataset pass rate ≥ threshold |
| **Production** | Full traffic | Staging validated, rollback ready |

### Versioning Rules

Apply semantic versioning (`MAJOR.MINOR.PATCH`) at the agent level, with clear rules for what triggers each increment:

| Change Type | Version Bump | Rationale |
|---|---|---|
| LLM model swap (e.g., `claude-3` → `claude-4`) | MAJOR | Behavioral contract change |
| New sub-agent added or removed | MAJOR | Capability surface change |
| Skill added, removed, or interface changed | MAJOR | Capability surface change |
| Prompt refactor changing behavior | MINOR | Output distribution shifts |
| Prompt typo / formatting fix | PATCH | No behavioral change |
| Document corpus update | No agent bump | KCV has its own version |
| Temperature adjustment | MINOR | Output distribution shifts |

---

## Independent Artifact Updates

The key insight: most production updates are **single-layer changes**. Design your system so they can be deployed without touching the others.

### Prompt Hot-Swap (CLV Updates)

**Pattern**: Store prompts outside your codebase. The agent loads its prompt at runtime from a registry, not from a hardcoded string.

```python
# Without prompt registry — prompt is frozen in code
SYSTEM_PROMPT = "You are a helpful assistant that..."

# With prompt registry — prompt is live-reloadable
from prompt_registry import get_prompt

system_prompt = get_prompt(
    name="customer-support-agent",
    version="production",   # or a specific semver tag
    fallback="v1.2.0"
)
```

**What this enables**:

- A/B test two prompt variants in production without a code deploy
- Roll back a bad prompt change in minutes by pointing the `production` alias to a previous version
- Different environments (`staging`, `production`) each resolve to their own prompt version

```python
# LangSmith example
from langsmith import Client

client = Client()
prompt = client.pull_prompt("customer-support-agent:production")
```

### Model Version Swap (MHV Updates)

**Pattern**: Abstract the model identifier and configuration behind a config layer. Never hardcode `model="claude-3-5-sonnet-20241022"` in business logic.

```yaml
# agent_config.yaml — versioned separately, loaded at runtime
model:
  provider: anthropic
  model_id: claude-sonnet-4-6
  temperature: 0.3
  max_tokens: 4096
```

```python
# Agent code loads config, not model name
config = load_agent_config(env="production")
response = client.messages.create(
    model=config.model.model_id,
    temperature=config.model.temperature,
    ...
)
```

**Gate before promoting an MHV change**: run your offline golden dataset against both the old and new model, compute delta on key metrics (faithfulness, task completion, latency, cost). Require the delta to be positive or within acceptable regression bounds before promotion.

**Platforms**: MLflow 3.0 (model registry with native agent artifact support), AWS SageMaker Model Registry, Vertex AI Model Registry.

### Sub-Agent Versioning

Sub-agents have their own artifact lifecycles. A parent agent calling a sub-agent faces the same dependency management problem as any service calling another service.

**Agent Cards** (from the A2A protocol) are a natural versioning mechanism for sub-agents. An Agent Card is a JSON descriptor — typically served at `/.well-known/agent.json` — that declares the sub-agent's capabilities, skills, endpoint, and a `version` field. A parent agent can:

1. Discover the sub-agent via its card at registration time
2. Pin the card version it depends on
3. Detect incompatible changes when the card version bumps

```yaml
# parent-agent-config.yaml
sub_agents:
  - name: flight-search-agent
    card_url: https://flight-agent.internal/.well-known/agent.json
    pinned_version: "2.1.0"
  - name: hotel-search-agent
    card_url: https://hotel-agent.internal/.well-known/agent.json
    pinned_version: "1.4.2"
```

**Promotion rule**: When a sub-agent ships a new MAJOR version (breaking interface change), the parent agent must explicitly opt in and re-validate. MINOR and PATCH updates can auto-promote if backward-compatible contracts are declared in the card.

---

## Feature Flags for Agent Artifacts

Feature flags decouple deployment from release: you deploy the artifact to production infrastructure but control which traffic sees it.

```python
# LaunchDarkly AI Configs pattern
from ldclient import get as ld_get

ld = ld_get()
prompt_variant = ld.variation(
    "customer-support-prompt-variant",
    user_context,
    default="control"
)

system_prompt = get_prompt(
    name="customer-support-agent",
    variant=prompt_variant   # "control" or "treatment"
)
```

**What this enables**:

- **Gradual rollout**: 5% → 20% → 50% → 100% without a redeployment
- **User-segment targeting**: new prompt for premium users, old for free tier
- **Instant kill switch**: flag off a bad artifact in seconds
- **Multivariate testing**: compare prompt or skill variants simultaneously

---

## Capability Declaration vs. Capability Binding

### Definitions

**Capability declaration** is the contract an agent publishes about what it can do. It is part of the agent's identity — expressed in its Agent Card, its system prompt, or its interface spec. Declaring a capability says: *this agent can perform this class of task, under these input/output constraints*. Changing a declared capability is a version change, because consumers depend on it.

**Capability binding** is the runtime wiring that connects a declared capability to a specific implementation. It answers: *which sub-agent, which endpoint, which model, which prompt version* is currently fulfilling this capability. A binding is configuration — it can change without touching the capability declaration, provided the implementation still satisfies the declared contract.

| | Capability Declaration | Capability Binding |
|---|---|---|
| **What it is** | The contract: what the agent can do | The wiring: what fulfills it at runtime |
| **Where it lives** | Agent Card, interface spec, system prompt | Config file, environment, runtime registry |
| **Who depends on it** | Consumers of your agent | Only your agent's own orchestration |
| **Version impact** | Adding/removing = MAJOR bump | Changing binding = config change, not code version bump |
| **Example** | "This agent can search flights" | "Flight search is handled by sub-agent dummy v2.1 at endpoint X" |

### Principles

**1. Declaration is a promise; binding is an implementation detail.**
Your consumers depend on your declarations, not on your bindings. Swapping the sub-agent that fulfills "search flights" for an equivalent one is a binding change. Removing "search flights" from the agent's declared capabilities is a declaration change. Only the latter breaks consumers.

**2. Bindings must be recorded in lineage, not in versions.**
Every run must log the full set of active bindings: which sub-agent version, which model, which prompt version, which corpus snapshot. This is not version bumping — it is execution provenance. Without it, the same agent version can produce different outputs on different days with no traceability.

**3. A declared capability with an unverified binding is a liability.**
If your agent declares it can call sub-agent dummy but you have no mechanism to validate that dummy still behaves as expected, the declaration is a fiction. Capability declarations require contract enforcement at the binding layer.

**4. Bindings to external agents you don't control are the highest-risk configuration.**
When the entity implementing your declared capability can change without your knowledge, your declaration can silently become false. This is the core problem that the architecture patterns below address.

---

## Resources

Additional references:

- AWS, "Prompt, Agents and Model lifecycle management," <https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-serverless/prompt-agent-and-model.html>
- Langfuse, "Prompt Management," 2024. <https://langfuse.com/docs/prompts/get-started>
- Anthropic, "Building Effective Agents," 2024. <https://www.anthropic.com/research/building-effective-agents>
