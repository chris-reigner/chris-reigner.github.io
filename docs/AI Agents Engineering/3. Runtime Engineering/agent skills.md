# Agent Skills

Skills are a lightweight, open format for giving AI agents specialized knowledge and procedural expertise on demand. Rather than stuffing everything into a system prompt, skills let an agent discover and load domain-specific instructions only when they become relevant.

The format was originally developed by Anthropic, released as an open standard, and has since been adopted by nearly every major AI coding agent and platform — Claude Code, GitHub Copilot, Cursor, OpenCode, VS Code, Gemini CLI, and more.

> In its minimal form a skill is a folder containing a `SKILL.md` file. That's it.

---

## How Skills Work

Skills use **progressive disclosure** to manage context efficiently. Loading everything upfront is wasteful and crowds out useful context. Instead, skills follow a three-phase lifecycle:

1. **Discovery** — At startup, the agent scans available skills and loads only the `name` and `description` from each `SKILL.md` (~100 tokens total). This is enough to know what each skill does without paying the full cost of its instructions.

2. **Activation** — When a user request matches a skill's description, the agent reads the full `SKILL.md` instructions into context (typically under 5k tokens). No irrelevant skills are loaded.

3. **Execution** — The agent follows the instructions. If the skill references external scripts, templates, or files, those are loaded on demand as needed.

This keeps agents fast and focused: many skills can remain available without consuming unnecessary context window space.

---

## Skills vs. Related Concepts

Skills are often confused with prompts, MCP servers, subagents, and project knowledge. The distinction matters because using the wrong abstraction leads to brittle agents.

| Concept | What it does | When to use it |
|---|---|---|
| **Skills** | Teach procedural knowledge that loads dynamically when relevant | Repeatable workflows, domain expertise, org-specific procedures |
| **System Prompt / Instructions** | Always-on context baked into every request | Core persona, hard constraints, fundamental capabilities |
| **Prompts** | One-time instructions within a single conversation turn | Ad-hoc tasks, clarifications, short-lived context |
| **Project Knowledge** | Persistent background knowledge for an entire workspace | Codebase conventions, team norms, persistent reference material |
| **MCP Servers** | Connect the agent to external data sources and tools via a protocol | Fetching live data, calling APIs, reading databases |
| **Subagents** | Independent agents that handle parallel or delegated subtasks | Long-running tasks, tasks requiring different permissions, parallelization |

## Anatomy of a Skill

Every skill lives in a folder with this structure:

```
my-skill/
├── SKILL.md          # Required: frontmatter + instructions
├── scripts/          # Optional: executable code the agent can run
├── references/       # Optional: docs, specs, lookup tables
└── assets/           # Optional: templates, output examples
```

The `SKILL.md` file contains a YAML frontmatter block followed by Markdown instructions:

```yaml
---
name: pdf-processing
description: Extract PDF text, fill forms, merge files. Use when handling PDFs.
---

# PDF Processing

## When to use this skill
Use this skill when the user needs to work with PDF files...

## How to extract text
1. Use pdfplumber for text extraction
2. Handle scanned PDFs with OCR fallback via pytesseract...

## How to fill forms
...
```

**Required frontmatter fields:**

- `name` — short identifier, kebab-case by convention
- `description` — written for the agent, not a human reader. It's the text the agent uses to decide whether to activate the skill. Make it precise and task-oriented: *"Use when handling PDFs"* is better than *"PDF utilities"*.

The Markdown body has no structural restrictions. Write it as you would a clear runbook: numbered steps, decision trees, examples, edge cases.

---

## Evaluating a Skill

A skill is only as good as its activation reliability and output consistency. Evaluation happens at two levels: **routing** (does the skill activate when it should?) and **execution** (does the agent follow the instructions correctly?).

### Layer 1 — Routing Evaluation

Routing is the most common failure mode. The agent either activates the skill on irrelevant tasks, misses it on relevant ones, or picks the wrong skill when several are available.

**Build a routing test set.** For each skill, write at least five examples in each category:

| Category | What to test | Expected outcome |
|---|---|---|
| **True positives** | Tasks clearly within the skill's domain | Skill activates |
| **True negatives** | Unrelated tasks | Skill does not activate |
| **Hard negatives** | Adjacent tasks that look similar but fall outside the skill | Skill does not activate |
| **Ambiguous** | Tasks that could match multiple skills | Correct skill activates, not a competitor |

**Diagnose routing failures by category:**

- High false negatives → description is too narrow or uses uncommon trigger words; broaden or add synonyms
- High false positives → description is too vague; add negative examples ("do not use for X")
- Wrong skill chosen → competing skill descriptions overlap; sharpen the boundary in both

### Layer 2 — Execution Evaluation

Once a skill activates correctly, evaluate whether the agent follows the instructions faithfully and produces correct output.

**Define expected outputs.** For each test case, specify what a correct execution looks like. This can be:

- A reference output to compare against (exact or semantic match)
- A checklist of required elements (steps completed, files touched, constraints respected)
- A rubric scored by a judge model (useful for open-ended skills)

**Execution metrics to track:**

| Metric | What it measures |
|---|---|
| **Step completion rate** | Fraction of required steps the agent completes |
| **Constraint adherence** | Whether the agent respects stated rules (naming conventions, format, order) |
| **Error recovery** | Whether the agent handles branches and fallbacks described in the skill |
| **Output quality** | Semantic correctness of the final result (human or LLM judge) |

### Evaluation Workflow

```
1. Write test cases (true positives, true negatives, hard negatives)
         ↓
2. Run routing tests — does the skill activate on the right inputs?
         ↓
3. For activating cases, run execution tests — does the agent follow the instructions?
         ↓
4. Identify failure patterns — routing problem or instruction problem?
         ↓
5. Fix: tweak description (routing) or rewrite instructions (execution)
         ↓
6. Re-run and track metrics over time
```

### Failure Taxonomy

Most skill failures fall into one of these buckets:

- **Silent miss** — skill never activates; agent improvises without skill guidance. Fix: rewrite the description.
- **Over-activation** — skill activates on unrelated tasks, wasting context. Fix: add negative conditions to the description.
- **Partial execution** — skill activates but the agent skips steps or ignores constraints. Fix: restructure instructions as numbered steps with explicit checkpoints.
- **Instruction conflict** — skill instructions contradict the system prompt or another skill. Fix: audit for overlap, use explicit precedence rules.
- **Stale skill** — underlying workflow changed but the skill wasn't updated. Fix: version-control skills alongside the code they describe; review on every relevant code change.

### Automating Evaluation

For teams running skills at scale, evaluation can be automated with a judge model:

```python
ROUTING_PROMPT = """
Given this user request:
<request>{request}</request>

And this skill description:
<description>{description}</description>

Should this skill activate? Answer YES or NO with one sentence of reasoning.
"""

EXECUTION_PROMPT = """
Given this skill instruction and the agent's output, did the agent follow the skill correctly?
Score on a scale of 1–5 and list any steps that were missed or violated.

Skill instructions: {instructions}
Agent output: {output}
"""
```

Run both prompts against your test set, track scores over time, and treat regressions as bugs. A skill that degrades silently is worse than one that never worked — the agent appears to be following a process it has quietly abandoned.

---

## Authoring Best Practices

- **Write descriptions for the agent, not the user.** The description is a routing signal. Think: *under what task conditions should this activate?*
- **Keep SKILL.md focused.** One skill per domain. Don't create a single mega-skill for everything.
- **Prefer steps over paragraphs.** Numbered lists produce more consistent agent behavior than prose.
- **Version-control your skills.** Skills are just files — commit them to git alongside your codebase.
- **Test activation.** Give the agent a task that should trigger the skill and verify it loads. Then give a task that should *not* trigger it and verify it doesn't.
- **Iterate on the description first.** Most activation failures are description problems, not instruction problems.

---

### Specification and Docs

| Resource | Description |
|---|---|
| [agentskills.io](https://agentskills.io) | Open standard home page, full specification, and community |
| [agentskills.io/specification](https://agentskills.io/specification) | Complete SKILL.md format specification |
| [Anthropic — Agent Skills Overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) | Official Claude platform documentation |
| [Anthropic — Best Practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices) | Authoring guidance from Anthropic |
| [claude.com/blog/skills-explained](https://claude.com/blog/skills-explained) | Conceptual overview of how skills work in Claude |
