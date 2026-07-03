# Agent Deployment Strategies

An agent is not a single binary. It is a **bundle of artifacts**: code (orchestration logic), model (provider + config), tools (integrations), prompts (system + skill prompts), sub-agents (declared dependencies), and services (external APIs, vector stores, memory backends). How you package and deploy those artifacts is a first-class architectural decision.

This document defines the principal deployment strategies, their trade-offs, and how to manage them across multiple environments.

---

## The Core Tension

Every deployment strategy is a point on a spectrum between two extremes:

| | All-in-One | All-External |
|---|---|---|
| **What you ship** | A single self-contained image | A thin launcher that pulls everything at runtime |
| **Update cost** | Rebuild + redeploy for any change | Update the artifact in its registry; restart or hot-reload |
| **Blast radius of a bad deploy** | Everything changes at once | Only the changed artifact is affected |
| **Operational complexity** | Low | High |
| **Best for** | Early-stage, infrequent updates | Production, frequent prompt/model iteration |

Most real deployments sit somewhere in the middle. The right choice depends on which artifacts change most often and how much downtime a bad update costs.

---

## Strategy 1 — Monolithic Image

Everything is baked into a single Docker image: code, prompts (inlined), model identifier, tool definitions, and the list of sub-agents.

```
┌─────────────────────────────────┐
│         Docker Image            │
│  ┌──────────────────────────┐   │
│  │  Code (orchestration)    │   │
│  │  Prompts (hardcoded)     │   │
│  │  Model config (hardcoded)│   │
│  │  Tool definitions        │   │
│  │  Sub-agent references    │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
```

**How it works:** Build once, push to a registry, deploy everywhere. A Dockerfile installs dependencies and copies all artifacts into the image.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .                          # code + prompts + config all in one
CMD ["python", "-m", "agent.main"]
```

**When to use:**
- POC or early-stage agents with infrequent updates
- When operational simplicity is the priority
- When all artifacts version together (model + prompt + code are co-owned by one team)

**When to avoid:**
- When prompts are tuned frequently — every tweak requires a full image rebuild and redeploy
- When different environments need different model configurations — you end up building one image per environment

**Update cost:** High. Every artifact change triggers a full CI pipeline.

---

## Strategy 2 — Code Image + External Configuration

The Docker image contains only code. Every other artifact — model config, prompts, tool definitions, sub-agent endpoints — is externalized as configuration loaded at startup.

```
┌─────────────────────┐     ┌──────────────────────────┐
│    Docker Image     │     │     Config (external)    │
│  ┌───────────────┐  │     │  model_id: claude-...    │
│  │  Code only    │──┼────▶│  temperature: 0.3        │
│  └───────────────┘  │     │  prompt_version: v2.1    │
└─────────────────────┘     │  tools: [search, calc]   │
                             │  sub_agents: [agent-b]   │
                             └──────────────────────────┘
```

In Kubernetes, configuration is injected via ConfigMaps and Secrets. The agent code reads its config at startup:

```python
# agent/config.py
import os, yaml

def load_config():
    config_path = os.getenv("AGENT_CONFIG_PATH", "/etc/agent/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()
MODEL_ID     = config["model"]["model_id"]
TEMPERATURE  = config["model"]["temperature"]
PROMPT_PATH  = config["prompts"]["system_prompt_file"]
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
data:
  config.yaml: |
    model:
      model_id: claude-sonnet-4-6
      temperature: 0.3
      max_tokens: 4096
    prompts:
      system_prompt_file: /etc/prompts/system.txt
    sub_agents:
      - name: search-agent
        url: http://search-agent-svc:8080
    tools:
      - name: web_search
        enabled: true
```

**When to use:**
- Production agents where model or prompt updates are frequent
- When the same image is promoted across dev → staging → prod with only the config changing
- When you want to roll back a bad prompt without a full redeploy

**Update cost:** Medium. Code changes require a new image. Configuration changes (model, prompts) only need a ConfigMap update + pod restart.

---

## Strategy 3 — Code Image + Runtime Artifact Fetch

The image contains only code. Each artifact type is pulled from a dedicated registry at runtime, not at startup from a flat config file. This enables hot-swapping individual artifacts without any restart.

```
                        ┌──────────────────┐
                        │  Prompt Registry │ (LangSmith / Langfuse / S3)
                        └────────┬─────────┘
┌──────────────────┐             │
│   Docker Image   │     ┌───────▼──────────┐
│  ┌────────────┐  │     │  Model Registry  │ (MLflow / Bedrock / Vertex)
│  │ Code only  │──┼────▶│                  │
│  └────────────┘  │     └───────┬──────────┘
└──────────────────┘             │
                        ┌────────▼──────────┐
                        │   Tool Registry   │ (internal API / MCP server)
                        └───────────────────┘
```

```python
# agent/main.py
from prompt_registry import get_prompt
from model_registry import get_model_config
from tool_registry import load_tools

# Artifacts resolved at runtime from their registries
prompt       = get_prompt("customer-support", env="production")
model_config = get_model_config("customer-support-agent", env="production")
tools        = load_tools(["web_search", "crm_lookup"], env="production")
```

**When to use:**
- High-velocity teams where prompt engineers iterate daily
- When A/B testing prompt or model variants against live traffic
- When artifact updates must not cause any service restart

**Update cost:** Low for config artifacts (push to registry → agent picks it up). New image only needed for code changes.

**Operational cost:** High. Requires registries to be reliable — if the prompt registry is down at startup, the agent fails to initialize.

---

## Strategy 4 — Multi-Container (Agent + Sidecar Tools)

Tools are deployed as separate containers alongside the agent. The agent calls them over a local network interface (e.g., an MCP server or HTTP). Each tool can be updated, scaled, and versioned independently.

```
┌────────────────────────────────────────────────┐
│                      Pod                       │
│  ┌─────────────────┐     ┌──────────────────┐  │
│  │  Agent Container│────▶│ Tool Sidecar      │  │
│  │  (orchestration)│     │ (MCP server)      │  │
│  └────────────────-┘     └──────────────────┘  │
└────────────────────────────────────────────────┘
```

```yaml
# k8s/deployment.yaml (simplified)
spec:
  containers:
  - name: agent
    image: agent:v2.3.0
    env:
    - name: TOOLS_ENDPOINT
      value: "http://localhost:8081"

  - name: tools-mcp
    image: tools-mcp-server:v1.5.0  # versioned independently
    ports:
    - containerPort: 8081
```

**When to use:**
- When tools are heavy or have independent scaling needs
- When multiple agents share the same tool implementations
- When tools are owned by a different team than the agent

**Update cost:** Low per component. Update the tool image without touching the agent, or vice versa. Requires coordination on interface contracts (MCP schema).

---

## Multi-Environment Management with Kustomize

Across all four strategies, the same agent will run in multiple environments — dev, staging, production — with different model versions, different prompt pins, and different resource limits. Kustomize manages this through a **base configuration** plus **per-environment overlays**.

### Directory Layout

```
k8s/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml       # shared structure
│   └── configmap.yaml        # default config (dev defaults)
│
└── overlays/
    ├── dev/
    │   ├── kustomization.yaml
    │   └── config-patch.yaml  # dev-specific overrides
    ├── staging/
    │   ├── kustomization.yaml
    │   └── config-patch.yaml
    └── prod/
        ├── kustomization.yaml
        ├── config-patch.yaml
        └── hpa.yaml           # prod-only: horizontal pod autoscaler
```

### Base Configuration

```yaml
# k8s/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
data:
  config.yaml: |
    model:
      model_id: claude-haiku-4-5-20251001   # cheapest model as base default
      temperature: 0.5
      max_tokens: 2048
    prompts:
      version: latest
    sub_agents:
      search:
        url: http://search-agent-svc:8080
```

```yaml
# k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - configmap.yaml
```

### Staging Overlay

```yaml
# k8s/overlays/staging/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
bases:
  - ../../base
patches:
  - path: config-patch.yaml
    target:
      kind: ConfigMap
      name: agent-config
```

```yaml
# k8s/overlays/staging/config-patch.yaml
- op: replace
  path: /data/config.yaml
  value: |
    model:
      model_id: claude-sonnet-4-6          # test with production model
      temperature: 0.3
      max_tokens: 4096
    prompts:
      version: staging                     # staging prompt pin
    sub_agents:
      search:
        url: http://search-agent-svc:8080
```

### Production Overlay

```yaml
# k8s/overlays/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
bases:
  - ../../base
patches:
  - path: config-patch.yaml
    target:
      kind: ConfigMap
      name: agent-config
resources:
  - hpa.yaml                               # prod-only autoscaler
```

```yaml
# k8s/overlays/prod/config-patch.yaml
- op: replace
  path: /data/config.yaml
  value: |
    model:
      model_id: claude-sonnet-4-6
      temperature: 0.3
      max_tokens: 4096
    prompts:
      version: v2.1.0                      # pinned, stable version
    sub_agents:
      search:
        url: http://search-agent-svc.prod:8080
```

### Deploying Per Environment

```bash
# Deploy to staging
kubectl apply -k k8s/overlays/staging

# Deploy to production
kubectl apply -k k8s/overlays/prod

# Preview what production will look like (dry run)
kubectl kustomize k8s/overlays/prod
```

---

## Strategy Comparison

| | Monolithic Image | Code + Config | Code + Runtime Fetch | Multi-Container |
|---|---|---|---|---|
| **Update a prompt** | Rebuild image | Update ConfigMap + restart | Push to registry (no restart) | N/A |
| **Update the model** | Rebuild image | Update ConfigMap + restart | Push to model registry | N/A |
| **Update a tool** | Rebuild image | Rebuild image | Update tool registry | Update tool container |
| **Multi-env management** | Build one image per env | Kustomize overlays | Registry env tags | Kustomize overlays |
| **Operational complexity** | Low | Medium | High | High |
| **Best for** | POC, stable agents | Most production agents | High-velocity teams | Tool-heavy agents |

---

## Choosing a Strategy

Start with **Strategy 2** (Code + Config) for any production agent. It gives you environment isolation via Kustomize with low operational overhead. Evolve to **Strategy 3** only when prompt or model iteration velocity demands it. Use **Strategy 4** when tools are heavy enough to justify their own lifecycle.

The key invariant across all strategies: **never let the deployment unit dictate your update cadence**. If updating a prompt requires a full CI pipeline run, your prompt iteration loop is broken.

---

## Zero-Downtime Version Promotion

When you are in Strategy 3 or 4 (external config + prompt registry), you can update the agent without touching the running instance. The core principle: **deploy first, route later**. The new version runs alongside the old one and receives no traffic until you explicitly decide to send it.

### The Prompt Registry Alias Pattern

The prompt registry is the key enabler. Instead of pointing your agent directly at a version string, you point it at a **named alias** (`production`, `canary`). The alias is a mutable pointer you control independently of any deployment.

```
Prompt Registry
┌─────────────────────────────────┐
│  "production" ──────► v2.1.0   │  ← current running agent reads this
│  "canary"     ──────► v2.2.0   │  ← new agent reads this
└─────────────────────────────────┘
```

When you want to promote: update the `production` alias to point to `v2.2.0`. No deployment needed. Every running pod picks it up on the next request (or next restart, depending on how you cache prompts).

```python
# Agent reads alias, not version directly
prompt = get_prompt("customer-support", env=os.getenv("PROMPT_ENV", "production"))
# PROMPT_ENV=production  → resolves to v2.1.0 (current)
# PROMPT_ENV=canary      → resolves to v2.2.0 (next)
```

---

### Blue/Green Deployment

Blue/Green runs two complete, identical deployments simultaneously. Blue is live. Green is the new version. A Kubernetes Service switches traffic between them atomically.

```
                         ┌──────────────────────────┐
                         │  Kubernetes Service       │
                         │  selector: slot=blue      │──► Blue (v2.1, prompt: production)
                         └──────────────────────────┘
                                                        ── Green (v2.2, prompt: canary) [no traffic]
```

**Step-by-step:**

**1. Deploy green alongside blue (zero impact on running traffic):**

```yaml
# k8s/overlays/prod-green/kustomization.yaml
bases:
  - ../../base
nameSuffix: -green          # all resources get -green suffix
patches:
  - path: config-patch.yaml # points to PROMPT_ENV=canary, new image tag
```

```bash
kubectl apply -k k8s/overlays/prod-green
# agent-green pods start, but Service still routes to blue
```

**2. Validate green (smoke tests, eval suite, manual check):**

```bash
# Test green directly by port-forwarding — no production traffic affected
kubectl port-forward svc/agent-green 8080:80
curl http://localhost:8080/health
```

**3. Switch traffic atomically:**

```bash
kubectl patch service agent \
  -p '{"spec":{"selector":{"slot":"green"}}}'
# All traffic now goes to green. Blue still running as instant rollback.
```

**4. Rollback if needed (instant):**

```bash
kubectl patch service agent \
  -p '{"spec":{"selector":{"slot":"blue"}}}'
```

**5. Tear down blue once green is stable:**

```bash
kubectl delete -k k8s/overlays/prod-blue
```

**Trade-off:** Requires 2× the resources during the transition window. The switch is atomic — all traffic moves at once, so there is no gradual validation on real users.

---

### Canary Deployment

Canary shifts traffic gradually: 5% → 20% → 50% → 100%. The old version serves the majority while the new version is validated on a small slice of real traffic.

```
100% traffic ──► Blue (v2.1, prompt: production)
                        │
                        └──5%──► Green (v2.2, prompt: canary)
```

With **Argo Rollouts** (recommended for production):

```yaml
# k8s/rollout.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: agent
spec:
  strategy:
    canary:
      steps:
      - setWeight: 5        # 5% to canary
      - pause: {duration: 10m}
      - setWeight: 20
      - pause: {duration: 10m}
      - setWeight: 50
      - pause: {}           # manual approval gate before full rollout
  template:
    spec:
      containers:
      - name: agent
        image: agent:v2.2.0
        env:
        - name: PROMPT_ENV
          value: canary
```

```bash
# Promote manually after each pause
kubectl argo rollouts promote agent

# Abort and roll back at any point
kubectl argo rollouts abort agent
```

**Trade-off:** More complex to set up than Blue/Green, but exposes real user traffic to the new version before full rollout. A bad prompt change only affects 5% of users, not 100%.

---

### Prompt-Only Update (No Code Change)

If only the prompt changes — the most common case in production — you do not need to deploy a new image at all.

```
1. Push v2.2.0 to prompt registry
2. Point "canary" alias → v2.2.0
3. Deploy a single canary pod (same image, PROMPT_ENV=canary)
4. Validate canary pod behavior
5. Point "production" alias → v2.2.0
6. All existing pods pick up new prompt on next request
7. Remove canary pod
```

```bash
# Step 2: update alias in registry (LangSmith example)
langsmith prompts update-alias customer-support \
  --alias canary --version v2.2.0

# Step 5: promote — zero redeployment, all pods pick it up
langsmith prompts update-alias customer-support \
  --alias production --version v2.2.0
```

This is the fastest path: no image build, no Kubernetes rollout, no downtime.

---

### Decision Tree

```
Do you need to change the agent code?
│
├── No → Prompt-only update via registry alias
│         (push new prompt version, update alias, done)
│
└── Yes → Do you want gradual validation on real traffic?
          │
          ├── No  → Blue/Green (atomic switch, instant rollback)
          │
          └── Yes → Canary (Argo Rollouts, progressive traffic shift)
```
