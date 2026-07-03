# AI Agent Terms — Supplement

Focused on terms from the [HuggingFace Agent Glossary](https://huggingface.co/blog/agent-glossary) — concepts complementing the main glossary.

---

## Three Concepts People Confuse: Scaffolding, Context Engineering, Harness

**Scaffolding** — *the content*
What is currently placed in the model's context window: system prompt, tool descriptions, skill instructions, retrieved memories, conversation history. A snapshot — it describes what the model sees at a given moment. Static configuration, not running code.

**Context Engineering** — *the discipline*
The practice of deciding *what* goes into the context window, *when*, and *how much*. Which memories to retrieve and inject? When to load a new skill? When to prune stale tool results? When to compress? Context engineering is a capability — something a harness does well or poorly.

**Harness** — *the running system*
The code that drives the agent loop: calls the model, routes tool calls, updates the scaffolding between steps (this is context engineering in practice), enforces stopping conditions, runs hooks. The harness is not the machine it runs on — it is the application-level orchestration logic.

| | Scaffolding | Context Engineering | Harness |
|---|---|---|---|
| **What it is** | Content in context at a given moment | Practice of curating that content | Running system that implements the loop |
| **Static or dynamic** | Static snapshot | Dynamic process | Dynamic execution |
| **Lives where** | Context window | In the harness's logic | Inside the runtime |
| **Example** | System prompt + tool descriptions + Skill X | "At step 3, retrieve episodic memory and inject; prune tool results older than 2 steps" | Claude Code CLI, LangGraph, custom loop, managed harness |

---

## Enterprise Deployment Architecture

### Diagram 1 — Execution Architecture

Runtime and Gateway are platform services that both sit on infrastructure. The agent runs *inside* the runtime. Tool calls flow: Harness → Runtime → Gateway → External.

> **Interactive diagram:** [AI Agents Execution Diagram](../../img/AI%20Agents%20Execution%20Diagram.html)

<iframe src="/img/AI%20Agents%20Execution%20Diagram.html" width="100%" height="620" frameborder="0" style="border-radius:8px;border:1px solid #ccc;"></iframe>

### Diagram 2 — Supporting Services

These services are not in the hot path of every request but are required for production operation.

```
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
  │  OBSERVABILITY   │  │     MEMORY       │  │   EVALUATION     │  │       REGISTRIES         │
  │                  │  │                  │  │                  │  │                          │
  │  traces per step │  │  working         │  │  eval harness    │  │  model registry          │
  │  logs            │  │  (context win.)  │  │  golden datasets │  │  tool / MCP catalog      │
  │  token metrics   │  │  episodic        │  │  rubric scoring  │  │  agent registry          │
  │  cost alerts     │  │  semantic        │  │  A/B test results│  │  prompt registry         │
  │  dashboards      │  │  procedural      │  │  regression runs │  │  skill catalog           │
  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  └────────────┬─────────────┘
           │                     │                      │                         │
           └─────────────────────┴──────────────────────┴─────────────────────────┘
                                          feeds / consumed by
                              Harness · Runtime · Gateway · CI/CD pipeline
```

| Service | What it provides | Who consumes it |
|---|---|---|
| **Observability** | Traces, logs, token counts, latency, cost per rollout | Engineers debugging; ops dashboards; cost alerts |
| **Memory** | Persistent storage for all four memory layers across sessions | Harness injects retrieved memories into scaffolding (context engineering) |
| **Evaluation** | Offline scoring against golden datasets before any promotion | CI/CD gate before promoting a new prompt, model, or agent version |
| **Registries** | Versioned catalog of models, tools, prompts, agents, skills | Harness fetches the right artifact version at startup; gateway discovers available tools |

---

## Behavior Terms

**Policy** — The mapping from what the agent perceives to what it does next. Partly in model weights (trained), partly in scaffolding (prompted). Changing the system prompt changes the policy without retraining.

**Rollout** *(Trajectory, Trace)* — One complete agent run. In RL training: raw data for rewards. In production: the execution trace stored in observability.

**Reward** — Score assigned to a rollout. *Verifiable*: tests pass/fail. *Learned*: human ratings or LLM-as-judge.

**Trainer** — Runs many rollouts, scores them, updates model weights. Training time only — distinct from the harness (inference time).

---

## Example: Claude Code

Claude Code is a concrete, observable example of every term in the architecture above.

| Term | What it looks like in Claude Code |
|---|---|
| **Model** | Claude Sonnet 4.6 — the LLM that reasons about code, decides which files to read, generates edits |
| **Scaffolding** | The system prompt embedded in the CLI (*"you are Claude Code... prefer editing existing files... never push without confirmation"*) + descriptions of all available tools (Bash, Read, Write, Edit, Grep, Glob) + any skill instructions loaded for the session |
| **Context Engineering** | Between each model call: injects the current file tree if needed, appends the latest tool result, prunes bash output older than a few steps, compacts the full conversation when the context window approaches its limit, re-injects the original task after compaction |
| **Skills** | `/python-review` — a packaged multi-step runbook: run linting → run security scan → check style → format findings. Loaded into scaffolding on demand. `ecc:code-reviewer` is another skill: a structured code review workflow the model reads and follows step by step |
| **Tools** | `Bash(cmd)` — run a shell command; `Read(path)` — read a file; `Write(path, content)` — create a file; `Edit(file, old, new)` — targeted string replacement; `Grep(pattern)` — search code; `Glob(pattern)` — find files by pattern. Each is a single atomic function. |
| **Harness** | The Claude Code CLI itself — the process that: calls the Anthropic API, receives tool call responses, routes each tool call to the right local function, feeds results back into context, runs hooks at `PreToolUse` / `PostToolUse` / `Stop`, decides when the task is done |
| **Gateway** | Not present in the local deployment — tools execute directly in the local process. In a CI/CD or enterprise deployment, a gateway would sit between the harness and external services (e.g., GitHub API, internal registries) to handle auth and audit |
| **Runtime** | Local: the terminal process or IDE extension. In CI: a container. The runtime provides the compute and filesystem — the harness runs inside it |
| **Infrastructure** | Developer laptop, CI runner (GitHub Actions, Jenkins), or a cloud VM. Not agent-aware — just provides compute and network |
| **Observability** | Token counts and tool latency printed in the terminal after each session; full session transcripts stored in `~/.claude/projects/`; hooks can emit structured logs at each tool boundary |
| **Memory** | *Procedural*: `CLAUDE.md` — the project conventions and forbidden commands the agent always reads. *Episodic*: session transcripts (what happened in past runs). *Semantic*: user preference files in `~/.claude/memory/` (e.g., "user prefers TypeScript strict mode"). *Working*: the active context window |
| **Evaluation** | After every code edit: the harness can run `pytest` and treat pass/fail as an in-loop verifiable reward. `/ultrareview` spawns a multi-agent cloud review of the current branch — an offline evaluation step before merging |
| **Registries** | *Model*: Anthropic API with a pinned model ID in config. *Prompt*: `CLAUDE.md` and session memory files. *Skills*: `.claude/plugins/` directory. *Agent*: not applicable for a single-agent CLI — relevant when Claude Code spawns sub-agents |
| **Policy** | Read before editing. Prefer targeted edits over full rewrites. Ask before destructive operations. Confirm before any git push. Never skip pre-commit hooks. |
| **Rollout** | One complete session: user says "fix the login bug" → agent reads `auth.py` → reads tests → identifies missing token expiry check → edits `auth.py` → runs `pytest` → 12 tests pass → reports done. Everything from first prompt to final stop = one rollout |
| **Reward** | *Verifiable*: `pytest` passes after the edit. *Learned*: user accepts the change without reverting it; user explicitly confirms ("yes, exactly") |

## Sources

- [HuggingFace Agent Glossary](https://huggingface.co/blog/agent-glossary)
- [Agent Deployment Architecture — Harness vs Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-vs-runtime.html)
- [Gateway concepts for agentic systems](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html)
- [AI Agent Memory Systems: 2026 Engineering Guide](https://jobsbyculture.com/blog/ai-agent-memory-systems-guide-2026)
