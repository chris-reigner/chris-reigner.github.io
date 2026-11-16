---
hide:
  - navigation
  - toc
---

# Tracing Solutions (2026)

Benchmark of LLM/agent tracing and evaluation platforms. Start with the summary matrix, then open a solution tab for the detailed, sourced findings.

## Summary matrix

✅ available · ⚠️ partial, preview/alpha or manual · ❌ not available · ❔ unknown / to be confirmed · SH = self hosted

| Feature | Langfuse (SH) | MLflow Databricks | MLflow (SH) | MLflow SageMaker | LangSmith (SH) | LangSmith SaaS | Dynatrace SaaS | Phoenix (SH) | Arize AX SaaS |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Archi/technical details** | | | | | | | | | |
| OpenTelemetry | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| User usage and tracking | ✅ | ⚠️ | ⚠️ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Python SDK | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Features** | | | | | | | | | |
| Framework agnostic | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Datasets storage and lineage | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Experiments | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Evaluate prompt against Dataset (Server side) | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| Evaluate prompt against Dataset (Client side) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| Continuous Evaluation in Prod | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Prompt playground | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| User feedback | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Annotation queue | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ |
| Multi-modality | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cost visibility | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CI Integration | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ | ❔ | ✅ | ⚠️ |
| MCP Server | ✅ | ✅ | ✅ | ❌ | ❔ | ❔ | ❔ | ✅ | ⚠️ |
| **Roles/Authentication** | | | | | | | | | |
| Agents support | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Role based on project | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| RBAC | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Authentication OpenID Connect | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Token (member_of) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Prompt Registry | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ❔ | ✅ | ✅ |
| AI Gateway | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| Conversation Simulation | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| LLM as a judge & scorers | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| Prompt optimization | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ⚠️ | ✅ |
| Admin UI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Traces lifecycle automation (archive) | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Issue detections automation | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| Multi-turn Evaluation | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | ⚠️ | ✅ |
| Model registry | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Sandbox | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Security Requirements** | | | | | | | | | |
| PII masking | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Audit logs | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Data export & interoperability | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ✅ |
| SLI / SLO | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Apdex | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Regression detection | ⚠️ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Resilience (patterns + feature flipping) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Automated provisioning (dashboards, alerts) | ❌ | ⚠️ | ❌ | ❌ | ⚠️ | ⚠️ | ✅ | ⚠️ | ⚠️ |

??? info "Feature definitions"

    | Feature | Definition |
    |---------|------------|
    | Hosting options | Available deployment models (SaaS, self-hosted, hybrid, managed). |
    | Architecture components | Key infrastructure and software components that make up the platform's architecture. |
    | Consulting support | Available professional services, support plans, and community resources. |
    | Scalability | Ability to handle increasing workloads through auto-scaling or infrastructure expansion. |
    | Warm-up time | Time required for initial provisioning or cold-start before the service becomes operational. |
    | Latency / Parallelization | Trace ingestion latency overhead and support for parallel/batch processing of evaluations. |
    | OpenTelemetry | Native support for OpenTelemetry (OTel) protocol for ingesting or exporting traces via OTLP. |
    | User usage and tracking | Built-in analytics for monitoring platform usage, token consumption, and cost per user/model/session. |
    | Python SDK | Availability of a Python SDK for programmatic instrumentation, tracing, and evaluation. |
    | Logical isolation | Separation of data and access at the application level (projects, workspaces, namespaces) without separate infrastructure. |
    | Hard isolation | Physical or account-level separation requiring distinct deployments or infrastructure for complete tenant isolation. |
    | Storage isolation | Separation of stored data (traces, artifacts) between tenants, projects, or environments. |
    | Network isolation | Network-level separation (VPC, PrivateLink, firewalls) to restrict traffic between tenants or from the internet. |
    | Framework agnostic | Ability to trace and instrument any LLM framework or custom application code without being locked to a specific library. |
    | Datasets storage and lineage | Built-in support for managing curated collections of input/output pairs used for evaluation, testing, and benchmarking LLM applications. |
    | Experiments | Capability to run, track, and compare multiple iterations of an LLM application (different prompts, models, parameters) with recorded metrics and artifacts. |
    | Evaluate prompt against Dataset (Server side) | Running evaluation scorers (including LLM-as-judge) on the platform's server infrastructure, without requiring the user's local compute. |
    | Evaluate prompt against Dataset (Client side) | Running evaluation scorers from the user's local machine or CI/CD pipeline, with results sent back to the platform for storage and visualization. |
    | Continuous Evaluation in Prod | Automated, ongoing scoring and monitoring of live production traces to detect quality degradation without manual intervention. |
    | Observability fields | The set of structured data captured per trace/span (e.g., inputs, outputs, latency, tokens, cost, metadata, tool calls, errors). |
    | Dashboard capabilities | UI capabilities for slicing and querying traces by dimensions such as time range, user, model, score, tags, cost, or latency. |
    | Prompt playground | An interactive UI for testing and iterating on prompts against different models, comparing outputs side-by-side without writing code. |
    | User feedback | Built-in mechanism for end-users to rate or comment on LLM outputs (e.g., thumbs up/down), with feedback linked to the corresponding trace. |
    | Annotation queue | A workflow UI where human reviewers can systematically label, review, and annotate traces for quality assessment or training data curation. |
    | Multi-modality | Support for tracing and rendering non-text content (images, audio, files) within trace spans, not just text-based inputs/outputs. |
    | Cost visibility | Built-in dashboards or metrics showing token usage and associated costs per model, user, session, or time period. |
    | CI Integration | Native support for running evaluations as part of a CI/CD pipeline (e.g., pytest plugin, GitHub Actions integration) with automated pass/fail gating. |
    | Agents support | Ability to trace multi-step agentic workflows (tool calls, reasoning chains, sub-agent delegation) with full visibility into each decision step. |
    | Role based on project | Ability to assign different roles and permissions scoped to specific projects or workspaces within the platform. |
    | RBAC | Role-Based Access Control — restricting system access based on assigned roles with granular permissions at resource level. |
    | Authentication OpenID Connect | Support for federated authentication via OpenID Connect (OIDC) or SAML protocols, enabling SSO with enterprise identity providers. |
    | Token (member_of) | Token-based authentication (API keys, PATs, OAuth tokens) with scoped permissions tied to user/group membership. |
    | Prompt Registry | A versioned store for managing, deploying, and A/B testing prompt templates across environments with SDK access. |
    | AI Gateway | A centralized proxy layer for LLM API calls that provides rate limiting, cost controls, model routing/fallback, and governance across providers. |
    | Conversation Simulation | Generating synthetic multi-turn conversations to stress-test agents and chatbots against diverse user scenarios without real users. |
    | LLM as a judge & scorers | Using an LLM to automatically evaluate the quality of another LLM's outputs (correctness, relevance, safety) alongside rule-based code scorers. |
    | Prompt optimization | Algorithmic techniques that automatically improve prompt wording based on evaluation scores, beyond manual trial-and-error iteration. |
    | Admin UI | A web interface for platform administrators to manage users, roles, projects, API keys, data retention, and system configuration. |
    | Traces lifecycle automation (archive) | Automated retention policies that archive or delete old traces after a configurable TTL, reducing storage costs and enforcing data governance. |
    | Issue detections automation | AI-powered or rule-based system that automatically identifies, clusters, and surfaces recurring problems from production traces (errors, quality drops, anomalies). |
    | Multi-turn Evaluation | Evaluating the quality of entire conversation sessions (not just single turns), assessing coherence, context retention, and goal completion across multiple exchanges. |
    | Model registry | A versioned catalog for registering, staging, and promoting ML/LLM models through lifecycle stages (dev ? staging ? production). |
    | Sandbox | An isolated, secure execution environment for running untrusted or LLM-generated code safely during evaluation or testing. |
    | RBAC & SSO | Role-Based Access Control (granular permissions per user/team) combined with Single Sign-On integration (SAML/OIDC) for enterprise identity management. |
    | Encryption mechanisms | Methods used to protect data at rest and in transit (e.g., AES-256, TLS, customer-managed keys). |
    | PII masking | Ability to detect and redact personally identifiable information from trace data before or after storage. |
    | Audit logs | Immutable records of user and system actions (who did what, when) for security investigation and compliance. |
    | Data export & interoperability | Ability to export trace data to external systems (S3, APIs, OTel collectors) and integrate with other observability/analytics tools. |
    | SLI / SLO | Service Level Indicators (measurable metrics like latency, error rate) and Service Level Objectives (target thresholds with error budgets) for reliability management. |
    | Apdex | Application Performance Index — a standardized satisfaction score (0–1) based on response time thresholds (Satisfied/Tolerating/Frustrated). |
    | Regression detection | Automated identification of quality or performance degradation when deploying new versions, comparing current metrics against established baselines. |
    | Resilience (patterns + feature flipping) | Built-in support for fault-tolerance patterns (retries, circuit breakers, fallbacks) and feature flags to safely roll out or kill-switch LLM features. |
    | Automated provisioning (dashboards, alerts) | Ability to programmatically create and manage monitoring resources (dashboards, alerts, SLOs) via APIs, Terraform, or configuration-as-code. |

## Details per solution

=== "Langfuse (SH)"

    **Langfuse (self hosted)**

    | Feature | Detail |
    |---------|--------|
    | **Infos** | |
    | Website | <https://langfuse.com> |
    | Licence | Cloud/Enterprise tiers proprietary.  [src: https://langfuse.com/pricing-self-host] |
    | Pricing | Infrastructure cost - see infra cost tab<br>Annual license - 150kEUR (TBD) for 0.5B units (aka traces)<br>Up to 5B/year of unit (aka traces) |
    | **Archi/technical details** | |
    | Hosting options | Cloud SaaS (US/EU) OR self-hosted (Docker, K8s, Terraform for AWS/GCP/Azure)  [src: https://langfuse.com/self-hosting] |
    | Architecture components | ClickHouse (analytics), PostgreSQL (metadata), Redis (cache), S3/Blob (media), Web UI, API server.  [src: https://langfuse.com/self-hosting] |
    | Consulting support | Community (GitHub/Discord). Enterprise: dedicated support, SLA (1h P1), Shakudo/GAO partners.  [src: https://langfuse.com/enterprise/enterprise] |
    | Scalability | Depends on target infrastructure |
    | Warm-up time | NA |
    | Latency / Parallelization | Async SDK ingestion (batched).  Parallel eval pipelines.  [src: https://langfuse.com/docs/observability/overview] |
    | OpenTelemetry | Yes - OTLP endpoint at /api/public/otel. Receives OTel traces natively.<br>[src: https://langfuse.com/integrations/native/opentelemetry] |
    | User usage and tracking | Built-in: cost/latency/token dashboards per user, model, prompt. Data export.  [src: https://langfuse.com/docs/metrics/overview] |
    | Python SDK | Yes - langfuse Python SDK. Also JS/TS SDK.  [src: https://langfuse.com/docs/observability/sdk/overview] |
    | Logical isolation | Project level: RBAC + project-id filtering |
    | Hard isolation | Requires separate deployments |
    | Storage isolation | Project level |
    | Network isolation | User-managed |
    | **Features** | |
    | Framework agnostic | Yes - OTel backend + native SDKs + decorator tracing. Any framework.  [src: https://langfuse.com/docs/opentelemetry/] |
    | Datasets storage and lineage | Yes - built-in dataset UI. Create from traces or upload. Dataset versioning available<br>Note  [src: https://langfuse.com/docs/evaluation/overview] |
    | Experiments | Yes - run evaluations across datasets, compare results.  [src: https://langfuse.com/docs/evaluation/overview] |
    | Evaluate prompt against Dataset (Server side) | Yes - server-side eval via API. Built-in + custom evaluators. LLM-as-judge.  [src: https://langfuse.com/docs/evaluation/overview] |
    | Evaluate prompt against Dataset (Client side) | Yes - Python SDK evaluation. CI/CD or notebook triggered.  [src: https://langfuse.com/docs/scores/external-evaluation-pipelines] |
    | Continuous Evaluation in Prod | Yes - webhook/Slack alerts on thresholds. Online eval via external pipelines on prod traces.  [src: https://langfuse.com/blog/2025-11-12-evals] |
    | Observability fields | Inputs, outputs, steps, model, tokens, cost, latency, scores, user, session, metadata, tags. Multi-modal.  [src: https://langfuse.com/docs/observability/features/multi-modality] |
    | Dashboard capabilities | Rich UI: time, user, session, model, score, cost, latency, tags, metadata.  [src: https://langfuse.com/docs/observability/overview] |
    | Prompt playground | Yes - built-in Playground. Test prompts, compare models, run against datasets.  [src: https://langfuse.com/docs/prompt-management/features/playground] |
    | User feedback | Yes - built-in (thumbs up/down, numeric, text). SDK + UI. Linked to traces.  [src: https://langfuse.com/docs/observability/features/user-feedback] |
    | Annotation queue | Yes - built-in annotation queue UI. Human labeling of traces.  [src: https://langfuse.com/docs/evaluation/overview] |
    | Multi-modality | Yes - images, audio, files in traces. Rendered inline.  [src: https://langfuse.com/docs/observability/features/multi-modality] |
    | Cost visibility | Yes - aggregation by model, scores, traces, users, session |
    | CI Integration | Manual using SDK |
    | MCP Server | Yes - enables AI agents/llm to securely access and manage project data, prompts and observability ressources |
    | **Roles/Authentication** | |
    | Agents support | Yes - LangGraph, CrewAI, OpenAI Agents, Pydantic AI, n8n. Multi-turn.  [src: https://langfuse.com/blog/2024-07-ai-agent-observability-with-langfuse] |
    | Role based on project | Yes - Orgs ? Projects. Roles: Owner, Admin, Member, Viewer.  [src: https://langfuse.com/docs/rbac] |
    | RBAC | Yes - built-in RBAC (Org+Project). Enterprise: SCIM, audit logs.  [src: https://langfuse.com/enterprise/enterprise] |
    | Authentication OpenID Connect | Yes - SSO: Google, GitHub, Azure AD, Okta, Auth0, Cognito, Keycloak. Enterprise enforced.  [src: https://langfuse.com/enterprise/enterprise] |
    | Token (member_of) | Yes - API keys (public/secret) per project. Scoped access.  [src: https://langfuse.com/docs/api-and-data-platform/overview] |
    | Prompt Registry | Yes - built-in: versioning, labels, environments, A/B, SDK deploy.  [src: https://langfuse.com/docs/prompt-management/overview] |
    | AI Gateway | No native gateway - recommends LiteLLM for rate limiting/routing.  [src: https://langfuse.com/integrations/gateways/litellm] |
    | Conversation Simulation | No - not a built-in feature. Can simulate via external scripts using SDK.  [src: https://langfuse.com/guides/cookbook/example_simulated_multi_turn_conversations] |
    | LLM as a judge & scorers | Yes - built-in LLM-as-judge evaluators + custom evaluators. Supports any LLM via configured connections.  [src: https://langfuse.com/docs/evaluation/overview] |
    | Prompt optimization | No - no built-in algorithmic optimization. Manual iteration via Playground + versioning. Agent skill-assisted improvement possible (e.g. Claude Code reads annotations).  [src: https://langfuse.com/blog/2026-02-16-prompt-improvement-claude-skills] |
    | Admin UI | Yes - Web UI for org/project/user management, settings, API keys, data retention configuration [src: https://langfuse.com/self-hosting/administration] |
    | Traces lifecycle automation (archive) | Yes - built-in Data Retention with configurable TTL policies. Project-level retention via ClickHouse TTL on traces table. Self-hosted: manage via ClickHouse TTL configs + S3 lifecycle rules on blob storage. Cloud: automatic cleanup.  [src: https://langfuse.com/self-hosting/administration + https://langfuse.com/docs/administration/data-retention] |
    | Issue detections automation | No built-in automatic issue detection. Supports manual error analysis via annotation queues, external evaluation pipelines, and threshold-based alerts (webhook/Slack). No native AI-powered anomaly clustering.  [src: https://langfuse.com/blog/2025-11-12-evals + https://langfuse.com/guides/cookbook/error-analysis-llm-applications] |
    | Multi-turn Evaluation | Yes - multi-turn evaluation via session-based traces. N+1 evaluation pattern: evaluate each turn in context of prior conversation. Simulated multi-turn conversations for testing. Supports custom and LLM-as-judge evaluators across conversation sessions.  [src: https://langfuse.com/blog/2025-10-09-evaluating-multi-turn-conversations + https://langfuse.com/guides/cookbook/example_evaluating_multi_turn_conversations] |
    | Model registry | No |
    | Sandbox | No - no built-in sandbox for code execution [src: https://langfuse.com/docs] |
    | RBAC & SSO | Project RBAC + projectId filtering  Multi-tenant SSO via SAML,OIDC |
    | **Security Requirements** | |
    | Encryption mechanisms | AES-256 via AWS KMS (Cloud). Self-hosted: ENCRYPTION_KEY env var (256-bit hex) encrypts secrets (API keys, LLM connections). TLS enforced (Cloud) or LANGFUSE_CSP_ENFORCE_HTTPS (self-hosted).  [src: https://langfuse.com/self-hosting/configuration/encryption + https://langfuse.com/security/security-faq] |
    | PII masking | Yes - client-side data masking functions (regex, custom rules) before data leaves host. beforeSend hooks in SDK. Server-side masking (Enterprise).  [src: https://langfuse.com/self-hosting/security/data-masking] |
    | Audit logs | Yes (Enterprise) - comprehensive activity tracking: who, what, when, changes made. Queryable via UI.  [src: https://langfuse.com/docs/audit-logs] |
    | Data export & interoperability | Yes - Blob Storage Export (S3/Azure/GCS) with scheduled exports. Full REST API + CLI. OTel receiver (inbound OTLP). PostHog/Mixpanel integration. Migration cookbook.  [src: https://langfuse.com/docs/api-and-data-platform/features/export-to-blob-storage + https://langfuse.com/docs/integrations/api] |
    | SLI / SLO | No native SLI/SLO engine. Provides raw metrics (latency, cost, scores per trace) that can feed external SLO tooling. Threshold-based spend alerts only. You must build SLO tracking via exported metrics to Prometheus/Grafana or Datadog.  [src: https://langfuse.com/docs/administration/usage-alerts + https://langfuse.com/academy/monitoring] |
    | Apdex | No built-in Apdex. Latency data available per trace/span — could compute Apdex externally from exported data, but no native Apdex score in UI.  [src: https://langfuse.com/docs/analytics/custom-dashboards] |
    | Regression detection | Versioning (tag releases, compare metrics between versions). External eval pipelines can trigger alerts on threshold breaches. No AI-powered clustering.  [src: https://langfuse.com/docs/observability/features/releases-and-versioning + https://langfuse.com/blog/2025-11-12-evals] |
    | Resilience (patterns + feature flipping) | No built-in resilience or feature flag support. Langfuse is an observability layer — resilience patterns (retries, circuit breakers, fallbacks) must be implemented in your application code. No integration with feature flag services.  [src: https://langfuse.com/docs/observability/overview] |
    | Automated provisioning (dashboards, alerts) | No - dashboards can be customizable but not automatically |
    | Vendor certifications | SOC 2 Type II, ISO 27001:2022, GDPR. No ISO 42001 announced. Self-hosted = your responsibility. <br> [src: https://langfuse.com/security/soc2 + https://langfuse.com/security/iso27001] |

=== "MLflow Databricks"

    **MLflow on Databricks (Managed)**

    | Feature | Detail |
    |---------|--------|
    | **Infos** | |
    | Website | <https://databricks.com/product/managed-mlflow> |
    | Licence | Databricks service   <https://docs.databricks.com/aws/en/mlflow3/genai/overview/oss-managed-diff> |
    | Pricing | Included in Databricks workspace (Premium/Enterprise tiers). DBU-based: $0.07–$0.65+/DBU depending on workload type + cloud infra. No separate MLflow license cost. Standard tier retired Oct 2026.    <https://pre.mammoth.io/databricks-pricing/> + <https://apis.io/plans/azure-databricks/> |
    | **Archi/technical details** | |
    | Hosting options | Fully managed SaaS on Databricks (AWS, Azure, GCP).  <https://docs.databricks.com/aws/en/mlflow3/genai/overview/oss-managed-diff><br>Databricks vs OSS features: <https://www.databricks.com/product/managed-mlflow> |
    | Architecture components | MLflow Tracking Server (managed), Unity Catalog (governance), Lakehouse (storage), AI Gateway (model routing), Model Serving, Review App, Databricks SQL for analytics.   <https://docs.databricks.com/aws/en/mlflow3/genai/> |
    | Consulting support | Databricks Professional Services + Partner ecosystem. Enterprise support with SLA on Premium/Enterprise tiers. Community forum + Knowledge Base.    <https://help.databricks.com> + <https://community.databricks.com> |
    | Scalability | Fully managed auto-scaling servers. Resource limits documented. Designed for enterprise multi-team usage.   <https://docs.databricks.com/aws/en/mlflow3/genai/overview/oss-managed-diff> |
    | Warm-up time | No |
    | Latency / Parallelization | Production-scale trace ingestion with async logging. Batch evaluation with configurable concurrency. Token-based scoring reduces latency vs full-text.    <https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing> |
    | OpenTelemetry | Yes - full OTLP support since MLflow 3.6.0. Accepts OTel spans directly. Compatible with any OTel-instrumented app (Python, Java, Go, Rust). Unified traces across OTel + MLflow instrumentation.   <http://mlflow.org/blog/opentelemetry-tracing-support/> |
    | User usage and tracking | System tables provide usage and audit logs. Databricks SQL + AI/BI dashboards for custom analytics on traces/metrics.   <https://docs.databricks.com/aws/en/mlflow3/genai/overview/oss-managed-diff> |
    | Python SDK | Yes - mlflow Python SDK (pip install mlflow[databricks]). Also TypeScript SDK available.    <https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk/> |
    | Logical isolation | Unity Catalog based |
    | Hard isolation | Workspace isolation + workspace-catalog binding |
    | Storage isolation | Unity catalog/schema isolation |
    | Network isolation | Databricks VPC / Private Connectivity |
    | **Features** | |
    | Framework agnostic | Yes - supports any framework via manual tracing (@mlflow.trace decorator), OTel ingestion, or custom instrumentation. Not locked to a single framework.    <https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/> |
    | Datasets storage and lineage | Yes - MLflow Evaluation Datasets with versioning, lineage, and Unity Catalog integration. Can create from traces or external data. JSON, pandas DataFrames supported.   <https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/eval-examples> |
    | Experiments | Yes - core feature. Experiment tracking with runs, metrics, parameters, artifacts. Compare versions side-by-side.   <https://docs.databricks.com/aws/mlflow/mlflow-3-install> |
    | Evaluate prompt against Dataset (Server side) | Yes - mlflow.genai.evaluate() runs evaluation server-side on Databricks. Built-in + custom scorers applied automatically. Results stored in experiment.    <https://docs.databricks.com/aws/mlflow3/genai/eval-monitor/concepts/eval-harness> |
    | Evaluate prompt against Dataset (Client side) | Yes - can run mlflow.genai.evaluate() from local/client environment pointing to Databricks tracking server. SDK handles execution locally.   <https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/> |
    | Continuous Evaluation in Prod | Yes - Production Monitoring service. Schedule scorers against an experiment, automatically evaluates sample of incoming traces. Results attached as feedback.   <https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/production-monitoring> |
    | Observability fields | Inputs, outputs, intermediate steps (tool calls, retriever steps, LLM calls), latency, token counts, metadata, cost, error status, custom attributes. Multimodal content (images, audio, files).    <https://mlflow.org/docs/latest/genai/tracing/> + <https://mlflow.org/blog/multimodal-tracing/> |
    | Dashboard capabilities | Via Databricks SQL + AI/BI dashboards on trace tables in Unity Catalog. Filter by experiment, time range, scorer results, tags, status, etc.   <https://docs.databricks.com/aws/en/mlflow3/genai/overview/oss-managed-diff> |
    | Prompt playground | Yes - Databricks AI Playground. Chat-based interface to test, prompt, and compare LLMs side-by-side. Supports tool-calling agents prototyping.   <https://docs.databricks.com/sap/en/ai-playground> |
    | User feedback | Yes - End-user feedback collection (thumbs up/down, text). Developer annotations on traces. Both SDK (mlflow.genai.log_feedback) and UI-based.    <https://docs.databricks.com/aws/mlflow3/genai/getting-started/human-feedback> |
    | Annotation queue | Yes - Review App with Labeling Sessions. Domain experts review and label traces. Structured annotation UI (no code required). Supports custom schemas.    <https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/expert-feedback/label-existing-traces> |
    | Multi-modality | Yes - since MLflow 3.x. Multimodal tracing for images, audio, and files. Binary content automatically extracted, stored as artifacts, rendered inline in UI.    <http://mlflow.org/blog/multimodal-tracing/> |
    | Cost visibility | Yes - token usage tracking per trace. Cost analytics via Databricks SQL + AI/BI dashboards on trace tables. Per-model and per-experiment aggregation.  [src: https://docs.databricks.com/aws/en/mlflow3/genai/overview/oss-managed-diff] |
    | CI Integration | Manual using SDK |
    | MCP Server | Yes - enables AI agents/llm to securely access and manage project data, prompts and observability ressources |
    | **Roles/Authentication** | |
    | Agents support | Yes - first-class. Tracing captures multi-step agentic workflows, tool calls, reasoning chains. Auto-tracing for LangGraph, AutoGen, CrewAI, etc. Conversation simulation for agent testing.   <https://docs.databricks.com/aws/en/mlflow3/genai/> + <https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/conversation-simulation> |
    | Role based on project | Yes - via Unity Catalog + Workspace permissions. Experiments scoped to projects/workspaces. Access control at experiment, model, and asset level.   <https://docs.databricks.com/aws/en/mlflow3/genai/overview/oss-managed-diff> |
    | RBAC | Yes - inherited from Databricks platform. Workspace ACLs for experiments. Unity Catalog privileges for models/assets. Fine-grained permission control.    <https://docs.databricks.com/aws/security/auth/access-control/> |
    | Authentication OpenID Connect | Yes - Databricks supports SAML 2.0 and OpenID Connect (OIDC) for SSO. Unified login across workspaces. Any OIDC-compliant IdP (Okta, Azure AD, etc.).    <https://docs.databricks.com/aws/security/auth/> |
    | Token (member_of) | Yes - PAT (Personal Access Tokens), OAuth M2M tokens, service principals. Token-based API access. Scoped permissions via groups/roles.   <https://docs.databricks.com/aws/security/auth/> |
    | Prompt Registry | Yes - centralized prompt management with versioning, A/B testing, Unity Catalog integration. Share prompts across org with governance.   <https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/> |
    | AI Gateway | Yes - unified endpoint for managing model access, cost controls, rate limiting, routing across providers. Governance layer for LLM calls.   <https://docs.databricks.com/aws/en/ai-gateway/> |
    | Conversation Simulation | Yes - generate synthetic multi-turn conversations for testing agents. Define scenarios, MLflow simulates realistic user interactions.   <https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/conversation-simulation> |
    | LLM as a judge & scorers | Yes - mlflow.genai scorers (built-in + custom). LLM judges fully supported. Built-in scorers for correctness, relevance, safety, groundedness, etc.  [src: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-harness] |
    | Prompt optimization | Yes - GEPA (ICLR 2026) + DSPy MIPROv2 algorithms. Custom optimizers via BasePromptOptimizer. Uses mlflow.genai.evaluate() scorers as objective.  [src: https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/ + https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tutorials/examples/prompt-optimization-quickstart] |
    | Admin UI | Yes - Databricks Workspace Admin Console for users, permissions, clusters, Unity Catalog settings [src: https://docs.databricks.com/aws/security/auth/access-control/] |
    | Traces lifecycle automation (archive) | Yes - traces stored in Delta tables via Unity Catalog with long-term retention. Standard Delta table lifecycle management (VACUUM, OPTIMIZE, retention policies). Production Monitoring provides durable storage with automated governance.  [src: https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/trace-unity-catalog] |
    | Issue detections automation | Yes - same as MLflow self-hosted: AI-powered Automatic Issue Detection with CLEARS framework. Available in Databricks-hosted MLflow UI. Clusters issues from traces, assigns severity, tracks resolution.  [src: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/] |
    | Multi-turn Evaluation | Yes - first-class multi-turn evaluation + Conversation Simulation. Session-level scorers assess entire conversation sessions (frustration, completeness, coherence). ConversationSimulator generates synthetic multi-turn conversations from test scenarios. Automatic evaluation on multi-turn traces.  [src: https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/evaluate-conversations + https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/conversation-simulation] |
    | Model registry | Yes |
    | Sandbox | Yes - Databricks secure code evaluation with isolated environments for LLM-generated code testing [src: https://www.databricks.com/blog/secure-code-evaluation] |
    | RBAC & SSO | Unity Catalog + Workspace ACLs. SAML 2.0/OIDC SSO.  [src: https://docs.databricks.com/aws/security/auth/] |
    | **Security Requirements** | |
    | Encryption mechanisms | Databricks-managed AES-256 encryption at rest. Unity Catalog encrypted storage. Customer-managed keys (CMEK) available. TLS enforced.  [src: https://docs.databricks.com/aws/security/] |
    | PII masking | Yes - client-side span processors + server-side AI Gateway guardrails for input/output filtering.  [src: http://mlflow.org/docs/latest/genai/tracing/observe-with-traces/masking/ + https://docs.databricks.com/aws/en/ai-gateway/] |
    | Audit logs | Yes - system tables for audit logs. Unity Catalog + CloudTrail integration. Full activity tracking.  [src: https://docs.databricks.com/aws/security/auth/access-control/] |
    | Data export & interoperability | Yes - Delta tables in Unity Catalog (SQL queryable). Dual OTel export. REST API + Databricks CLI. Delta Sharing for cross-org export.  [src: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/integrations/open-telemetry] |
    | SLI / SLO | No built-in SLI/SLO engine in MLflow layer. Relies on Databricks Lakehouse Monitoring + SQL alerts on trace tables to define custom SLIs. Can build SLO dashboards via AI/BI on trace metrics. No native error budget tracking.  [src: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing/ + https://docs.databricks.com/gcp/mlflow3/genai/] |
    | Apdex | No. MLflow is not an APM tool. No Apdex computation. Would need to use Databricks SQL on trace latency data to compute Apdex manually.  [src: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing/] |
    | Regression detection | Yes - AI-powered Automatic Issue Detection (CLEARS: Correctness, Latency, Execution, Adherence, Relevance, Safety). Clusters traces into named issues, tracks severity over time. Production Monitoring compares scorer results across versions. Evaluation harness gates releases.  [src: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/ + https://docs.databricks.com/gcp/mlflow3/genai/] |
    | Resilience (patterns + feature flipping) | No built-in resilience/feature flag in MLflow. Databricks AI Gateway provides rate limiting and fallback routing between model providers. Feature flags via external services. Evaluation harness can serve as a release gate (fail deploy if quality drops).  [src: https://docs.databricks.com/aws/en/ai-gateway/ + https://docs.databricks.com/gcp/mlflow3/genai/] |
    | Automated provisioning (dashboards, alerts) | Partially - Databricks Terraform Provider provisions workspaces, clusters, jobs, Unity Catalog, permissions. No MLflow-specific dashboard/alert Terraform resources. SQL alerts and AI/BI dashboards are created via UI or Databricks API/CLI. Webhooks for model registry events.  [src: https://docs.databricks.com/dev-tools/terraform/ + https://docs.databricks.com/en/mlflow/model-registry-webhooks.html] |
    | Vendor certifications | SOC 2 Type II, ISO 27001:2022, GDPR. No ISO 42001 announced. <br> [src: https://databricks.com/trust/soc-2-type-ii + https://databricks.com/trust/iso-27001] |

=== "MLflow (SH)"

    **MLFlow self hosted**

    | Feature | Detail |
    |---------|--------|
    | **Infos** | |
    | Website | <https://mlflow.org> [src: https://mlflow.org] |
    | Licence | ? |
    | Pricing | Infrastructure costs only (compute, DB, S3) - see Infra cost tab [src: https://mlflow.org/docs/latest/self-hosting/] |
    | **Archi/technical details** | |
    | Hosting options | Self-hosted on any infrastructure (bare metal, Docker, K8s, cloud VMs). Helm chart available since 3.13 [src: https://mlflow.org/docs/latest/self-hosting/kubernetes-helm] |
    | Architecture components | MLflow Tracking Server, SQL backend (PostgreSQL/MySQL), S3/GCS/Azure Blob (artifacts), Web UI, AI Gateway [src: https://mlflow.org/docs/latest/self-hosting/architecture/overview/] |
    | Consulting support | Community (GitHub issues, Slack). No commercial support [src: https://mlflow.org/docs/latest/community/] |
    | Scalability | Depends on target infrastructure |
    | Warm-up time | NA |
    | Latency / Parallelization | Async trace logging. Batch evaluation. Production-scale with OTLP [src: https://mlflow.org/docs/latest/genai/tracing/] |
    | OpenTelemetry | Yes - full OTLP support since MLflow 3.6.0. Accepts OTel spans directly [src: http://mlflow.org/blog/opentelemetry-tracing-support/] |
    | User usage and tracking | MLflow UI traces + metrics. No built-in cost dashboards [src: https://mlflow.org/docs/latest/genai/tracing/] |
    | Python SDK | Yes - mlflow Python SDK (pip install mlflow). Also TypeScript SDK [src: https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/] |
    | Logical isolation | Isolation at workspace level (3.10+)<br>Usage of MLflow OIDC feature<br>Split up to "experiment layer" |
    | Hard isolation | Requires separate deployments |
    | Storage isolation | per workspace |
    | Network isolation | User-managed |
    | **Features** | |
    | Framework agnostic | Yes - @mlflow.trace decorator + OTLP ingestion. Any framework [src: https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/] |
    | Datasets storage and lineage | Yes - MLflow Evaluation Datasets with versions as metadata. JSON/DataFrames [src: https://mlflow.org/docs/latest/genai/eval-monitor/] |
    | Experiments | Yes - core MLflow feature. Runs, metrics, params, artifacts [src: https://mlflow.org/docs/latest/genai/tracing/] |
    | Evaluate prompt against Dataset (Server side) | Yes - mlflow.genai.evaluate() with built-in + custom scorers. LLM judges supported [src: https://mlflow.org/docs/latest/genai/eval-monitor/] |
    | Evaluate prompt against Dataset (Client side) | Yes - mlflow.evaluate() from client with all scorers [src: https://mlflow.org/docs/latest/genai/eval-monitor/] |
    | Continuous Evaluation in Prod | Yes - assessments API + feedback collection on production traces [src: http://www.mlflow.org/docs/latest/genai/assessments/feedback/] |
    | Observability fields | Inputs, outputs, intermediate steps, tool calls, latency, tokens, metadata, tags, status. Multimodal [src: https://mlflow.org/docs/latest/genai/tracing/] |
    | Dashboard capabilities | MLflow UI: filter by tags, status, time, environment [src: https://mlflow.org/docs/latest/genai/tracing/] |
    | Prompt playground | No - not available in OSS MLflow [src: https://mlflow.org/docs/latest/genai/] |
    | User feedback | Yes - mlflow.genai Feedback API. Structured feedback on traces (thumbs up/down, scores, text) [src: http://www.mlflow.org/docs/latest/genai/assessments/feedback/] |
    | Annotation queue | No - not available in OSS MLflow (Databricks-only Review App) [src: https://mlflow.org/docs/latest/genai/eval-monitor/] |
    | Multi-modality | Yes - MLflow 3.x multimodal tracing (images, audio, files) [src: https://mlflow.org/blog/multimodal-tracing/] |
    | Cost visibility | No built-in cost dashboards [src: https://mlflow.org/docs/latest/genai/tracing/] |
    | CI Integration | Manual using SDK |
    | MCP Server | Yes - enables AI agents/llm to securely access and manage project data, prompts and observability ressources |
    | **Roles/Authentication** | |
    | Agents support | Yes - LangGraph, AutoGen, CrewAI auto-tracing. Multi-step workflows [src: https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/] |
    | Role based on project | Yes - via Workspaces (v3.10+). Workspace-scoped roles and resources [src: https://mlflow.org/docs/latest/self-hosting/workspaces/] |
    | RBAC | Yes - full RBAC since v3.13. Custom roles, workspace-scoped permissions. Admin UI [src: https://mlflow.org/docs/latest/self-hosting/security/role-based-access-control/] |
    | Authentication OpenID Connect | Yes - via mlflow-oidc-auth plugin (Okta, Google, Cognito, Azure Entra ID) [src: https://mlflow.org/docs/latest/self-hosting/security/sso/] |
    | Token (member_of) | Basic auth tokens + OIDC access tokens. Per-user role assignments [src: https://mlflow.org/docs/latest/self-hosting/security/sso/] |
    | Prompt Registry | Yes - MLflow Prompt Registry. Versioning, diff views, aliases, evaluation integration [src: https://mlflow.org/prompt-registry/] |
    | AI Gateway | Yes - MLflow AI Gateway. Centralized control plane for model access, rate limiting, cost controls [src: https://mlflow.org/blog/mlflow-ai-gateway] |
    | Conversation Simulation | Yes -  generate synthetic multi-turn conversations for testing your conversational AI agents. [src: https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/conversation-simulation/] |
    | LLM as a judge & scorers | Yes - mlflow.genai scorers (built-in + custom). LLM judges fully supported [src: https://mlflow.org/docs/latest/genai/eval-monitor/] |
    | Prompt optimization | Yes - built-in GEPA (state-of-the-art) + DSPy MIPROv2 algorithms. Custom optimizers via BasePromptOptimizer class. Iterations tracked as MLflow runs.  [src: https://mlflow.org/docs/latest/genai/prompt-version-mgmt/optimize-prompts/ + https://mlflow.org/blog/mlflow-prompt-optimization] |
    | Admin UI | Yes - Admin UI at /admin since v3.13. Manage users, roles, permissions, workspace grants [src: https://mlflow.org/docs/latest/self-hosting/security/role-based-access-control/] |
    | Traces lifecycle automation (archive) | Yes - built-in trace archival (since v3.13). Configurable retention at server, workspace, and experiment level. Scheduler moves span payloads to archive artifact storage (S3/GCS/Azure Blob). Supports archive-now and long_retention_allowlist. Also `delete_traces` API and `mlflow gc` for permanent purge.  [src: https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/archive-traces] |
    | Issue detections automation | Yes - AI-powered Automatic Issue Detection (since v3.12). Multi-stage LLM analysis pipeline clusters traces into named issues across 6 dimensions (CLEARS: Correctness, Latency, Execution, Adherence, Relevance, Safety). Severity rating, affected trace linking, issue status management (pending/resolved/rejected).  [src: https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/detect-issues + https://mlflow.org/blog/issue-detection] |
    | Multi-turn Evaluation | Yes - first-class multi-turn evaluation (since v3.8/3.10). Session-level conversational scorers evaluate entire sessions (frustration, coherence, completeness). ConversationSimulator generates synthetic multi-turn conversations for agent testing. Automatic evaluation runs on multi-turn traces as they arrive.  [src: https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/multi-turn + https://mlflow.org/blog/multiturn-evaluation] |
    | Model registry | Yes |
    | Sandbox | No - no built-in sandbox [src: https://mlflow.org/docs/latest/genai/] |
    | RBAC & SSO | Workspace RBAC + SSO with OIDC |
    | **Security Requirements** | |
    | Encryption mechanisms | Infrastructure-level only (S3 SSE, RDS encryption). No app-level encryption of trace content. Security middleware v3.5.0+ for HTTPS/CORS/DNS rebinding.  [src: https://mlflow.org/docs/latest/self-hosting/security/network/] |
    | PII masking | Yes - span processors for PII masking (TracingConfig). Filter/redact inputs, outputs, attributes before backend storage. Client-side only.  [src: http://mlflow.org/docs/latest/genai/tracing/observe-with-traces/masking/] |
    | Audit logs | No built-in audit logs. Relies on DB access logs and application-level logging.  [src: https://mlflow.org/docs/latest/self-hosting/security/] |
    | Data export & interoperability | Yes - Dual OTel export (MLflow + any Collector simultaneously). REST API + CLI. mlflow.search_traces() for programmatic access. GenAI Semantic Conventions (gen_ai.*) supported.  [src: http://mlflow.org/docs/latest/genai/tracing/opentelemetry/export/ + https://mlflow.org/docs/latest/tracing/tracing-schema] |
    | SLI / SLO | No built-in SLI/SLO. Provides production metrics (latency, errors, token usage, scorer results) but no SLO engine or error budget tracking. Must export metrics to external SLO tooling (Grafana, Datadog, Prometheus).  [src: https://mlflow.org/docs/latest/genai/tracing/prod-tracing/] |
    | Apdex | No. Not an APM tool. No Apdex computation. Latency data available in traces but no satisfaction scoring.  [src: https://mlflow.org/docs/latest/genai/tracing/prod-tracing/] |
    | Regression detection | No automated regression detection. Supports manual comparison via Releases |
    | Resilience (patterns + feature flipping) | No built-in resilience/feature flags. MLflow AI Gateway provides rate limiting and model routing/fallback. Application-level resilience (retries, circuit breakers) is your responsibility. Feature flags via external services.  [src: https://mlflow.org/blog/mlflow-ai-gateway + https://mlflow.org/docs/latest/genai/tracing/prod-tracing/] |
    | Automated provisioning (dashboards, alerts) | No  [src: https://mlflow.org/docs/latest/genai/tracing/prod-tracing/] |
    | Vendor certifications | NA |

=== "MLflow SageMaker"

    **MLflow on AWS SageMaker (Serverless)**

    | Feature | Detail |
    |---------|--------|
    | **Infos** | |
    | Website | <https://aws.amazon.com/sagemaker-ai/experiments/> |
    | Licence | AWS service |
    | Pricing | Sagemaker mlflow serverlss service  S3 storage cost  AWS infrastructure required by each environment |
    | **Archi/technical details** | |
    | Hosting options | Fully managed serverless on AWS.  [src: https://aws.amazon.com/blogs/aws/accelerate-ai-development-using-amazon-sagemaker-ai-with-serverless-mlflow/] |
    | Architecture components | MLflow Tracking Server (serverless), S3 (artifacts), SageMaker Studio UI, IAM  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Consulting support | AWS Professional Services, Support plans (Business/Enterprise), re:Post, Partners.  [src: https://aws.amazon.com/sagemaker/pricing/] |
    | Scalability | Auto-scaling serverless.  [src: https://aws.amazon.com/blogs/machine-learning/scaling-mlflow-for-enterprise-ai-whats-new-in-sagemaker-ai-with-mlflow/] |
    | Warm-up time | initial provision - 25 mns |
    | Latency / Parallelization | Async trace logging. Batch evaluation.  [src: https://aws.amazon.com/blogs/aws/accelerate-ai-development-using-amazon-sagemaker-ai-with-serverless-mlflow/] |
    | OpenTelemetry | Yes - MLflow 3.0 OTLP ingestion support.  [src: https://aws.amazon.com/blogs/machine-learning/accelerating-generative-ai-development-with-fully-managed-mlflow-3-0-on-amazon-sagemaker-ai/] |
    | User usage and tracking | CloudWatch metrics, CloudTrail audit. No built-in user analytics.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Python SDK | Yes - mlflow + sagemaker-mlflow packages.  [src: https://aws.amazon.com/blogs/machine-learning/accelerating-generative-ai-development-with-fully-managed-mlflow-3-0-on-amazon-sagemaker-ai/] |
    | Logical isolation | SageMaker Domain/Space level + IAM per project. MLflow App-level isolation.  [src: https://aws.amazon.com/blogs/machine-learning/scaling-mlflow-for-enterprise-ai-whats-new-in-sagemaker-ai-with-mlflow/] |
    | Hard isolation | AWS account or Sagemaker domain level or App level isolation (S3 bucket + IAM) |
    | Storage isolation | AWS account, domain or S3 isolation |
    | Network isolation | VPC + PrivateLink |
    | **Features** | |
    | Framework agnostic | Yes - @mlflow.trace decorator + OTLP ingestion.  [src: https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/] |
    | Datasets storage and lineage | Yes - MLflow Evaluation Datasets with versions as metadata.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Experiments | Yes - core MLflow feature. Runs, metrics, params, artifacts.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Evaluate prompt against Dataset (Server side) | Partially - code-based scorers work. LLM judges NOT supported.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-troubleshooting.html] |
    | Evaluate prompt against Dataset (Client side) | Yes - mlflow.evaluate() from client. Code scorers only. LLM judges must run externally.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-troubleshooting.html] |
    | Continuous Evaluation in Prod | Not natively. Must build custom pipeline (SageMaker Pipelines + FMEval).  [src: https://aws.amazon.com/blogs/machine-learning/track-llm-model-evaluation-using-amazon-sagemaker-managed-mlflow-and-fmeval/] |
    | Observability fields | Inputs, outputs, steps, tool calls, latency, tokens, metadata, tags, status.  [src: https://aws.amazon.com/blogs/machine-learning/accelerating-generative-ai-development-with-fully-managed-mlflow-3-0-on-amazon-sagemaker-ai/] |
    | Dashboard capabilities | MLflow UI: filter by tags, status, time, environment.  [src: https://aws.amazon.com/blogs/machine-learning/accelerating-generative-ai-development-with-fully-managed-mlflow-3-0-on-amazon-sagemaker-ai/] |
    | Prompt playground | No - not available on SageMaker MLflow. Use SageMaker AI Playground separately.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-troubleshooting.html] |
    | User feedback | Not built-in. Must implement custom.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Annotation queue | No - not available.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Multi-modality | Yes - MLflow 3.x multimodal tracing (images, audio, files).  [src: https://mlflow.org/blog/multimodal-tracing/] |
    | Cost visibility | No built-in cost dashboards. Use CloudWatch + S3 storage metrics for infrastructure cost tracking.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | CI Integration | Manual using SDK |
    | MCP Server | No AWS-managed MLflow MCP server |
    | **Roles/Authentication** | |
    | Agents support | Yes - LangGraph, Strands Agents, AutoGen. Multi-step workflows. MCP.  [src: https://aws.amazon.com/blogs/machine-learning/build-a-serverless-conversational-ai-agent-using-claude-with-langgraph-and-managed-mlflow-on-amazon-sagemaker-ai/] |
    | Role based on project | Yes - SageMaker Domains/Spaces + IAM per project/team.  [src: https://aws.amazon.com/blogs/machine-learning/scaling-mlflow-for-enterprise-ai-whats-new-in-sagemaker-ai-with-mlflow/] |
    | RBAC | Yes - IAM policies + SageMaker execution roles. Resource-level.  [src: https://aws.amazon.com/blogs/machine-learning/securing-mlflow-in-aws-fine-grained-access-control-with-aws-native-services/] |
    | Authentication OpenID Connect | Yes - AWS IAM Identity Center (OIDC/SAML). Any IdP.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Token (member_of) | Yes - IAM roles, STS credentials, service roles.  [src: https://aws.amazon.com/blogs/machine-learning/securing-mlflow-in-aws-fine-grained-access-control-with-aws-native-services/] |
    | Prompt Registry | Yes - TBC  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-troubleshooting.html] |
    | AI Gateway | No - NOT available. Use Amazon Bedrock Guardrails separately.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-troubleshooting.html] |
    | Conversation Simulation | No - not available on SageMaker MLflow.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-troubleshooting.html] |
    | LLM as a judge & scorers | No - LLM judges NOT supported on SageMaker MLflow. Code-based scorers only.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-troubleshooting.html] |
    | Prompt optimization | No - not available.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-troubleshooting.html] |
    | Admin UI | Yes - SageMaker Studio console UI for MLflow Apps management (create, delete, configure) [src: https://aws.amazon.com/blogs/aws/accelerate-ai-development-using-amazon-sagemaker-ai-with-serverless-mlflow/] |
    | Traces lifecycle automation (archive) | No built-in retention automation in SageMaker MLflow. Must use S3 Lifecycle Policies on artifact buckets to expire/archive objects. No server-side trace archival or TTL.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Issue detections automation | No - no built-in anomaly or issue detection for traces. Must build custom monitoring.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Multi-turn Evaluation | Partially - mlflow.genai.evaluate() supports multi-turn via conversation scorers (same as OSS MLflow). LLM judges NOT supported on SageMaker, limiting full multi-turn evaluation capability.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-troubleshooting.html + https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/multi-turn] |
    | Model registry | Yes |
    | Sandbox | No - no built-in sandbox. Use SageMaker Studio notebooks for isolated experimentation [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | RBAC & SSO | IAM policies + SageMaker execution roles. SSO via IAM Identity Center (OIDC/SAML).  [src: https://aws.amazon.com/blogs/machine-learning/securing-mlflow-in-aws-fine-grained-access-control-with-aws-native-services/] |
    | **Security Requirements** | |
    | Encryption mechanisms | AWS-managed encryption (S3 SSE, RDS encryption at rest, TLS in transit). No app-level encryption in MLflow itself.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | PII masking | No built-in PII masking in SageMaker MLflow. Use AWS Glue or custom preprocessing.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Audit logs | Yes - via AWS CloudTrail (API-level audit). SageMaker domain activity logs.  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Data export & interoperability | Artifacts in S3 (exportable). REST API. No OTel export (inbound only via OTLP).  [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | SLI / SLO | No built-in SLI/SLO management. Must use CloudWatch + custom metrics to define SLIs (latency, error rate, token usage) and set SLO targets externally. No error budget tracking. [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Apdex | No. Not an APM tool. No Apdex computation. [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Regression detection | No built-in regression detection. Must build custom pipelines comparing metrics across deployments. [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Resilience (patterns + feature flipping) | No built-in resilience or feature flag support. Resilience depends on AWS infrastructure (Lambda retries, Step Functions, SQS DLQ). Feature flags via external services (CloudWatch Evidently deprecated ? use LaunchDarkly/Unleash). [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Automated provisioning (dashboards, alerts) | No built-in provisioning API for monitoring resources within MLflow. Must use AWS APIs separately [src: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html] |
    | Vendor certifications | SOC 2 Type II, ISO 27001, ISO 42001, HIPAA, FedRAMP, PCI DSS (AWS platform-level — SageMaker in scope for all). <br> [src: https://aws.amazon.com/compliance/iso-42001-faqs/ + https://aws.amazon.com/compliance/iso-27001-faqs/] |

=== "LangSmith (SH)"

    **Langsmith (self hosted)**

    | Feature | Detail |
    |---------|--------|
    | **Infos** | |
    | Website | <https://docs.langchain.com/langsmith> [src: https://docs.langchain.com/langsmith] |
    | Licence | Proprietary. Enterprise license required for self-hosting [src: https://support.langchain.com/articles/7011309930-how-do-i-obtain-a-self-hosted-langsmith-license-key] |
    | Pricing | Commitment fee with up to 15M traces/year = 75k$  Unlimited seats  0,005$ per trace above  + Infrastructure costs |
    | **Archi/technical details** | |
    | Hosting options | Self-hosted on K8s (AWS EKS, Azure AKS, GCP GKE). Full or hybrid mode [src: https://docs.smith.langchain.com/self_hosting/installation/kubernetes] |
    | Architecture components | LangSmith UI and APIs, backend services (queue, playground, ACE), PostgreSQL, Redis, ClickHouse, optional blob storage [src: https://docs.smith.langchain.com/self_hosting/installation/kubernetes] |
    | Consulting support | Enterprise support with SLA. LangChain Professional Services [src: https://docs.langchain.com/langsmith/enterprise] |
    | Scalability | Depends on target infrastructure |
    | Warm-up time | NA |
    | Latency / Parallelization | Async SDK ingestion. Stateless horizontal scaling [src: https://docs.langchain.com/langsmith/scalability-and-resilience] |
    | OpenTelemetry | Yes - OTLP ingestion. Native OTel SDKs for Python, TypeScript, Go, Java [src: https://blog.langchain.com/end-to-end-opentelemetry-langsmith/] |
    | User usage and tracking | Built-in: cost/latency/token dashboards. Custom cost data. Alerts via webhooks/PagerDuty [src: https://docs.langchain.com/langsmith/cost-tracking] |
    | Python SDK | Yes - Python, TypeScript, Go, Java SDKs [src: https://www.langchain.com/langsmith/observability] |
    | Logical isolation | Workspace based project separation |
    | Hard isolation | Separate instance |
    | Storage isolation | per workspace |
    | Network isolation | User managed |
    | **Features** | |
    | Framework agnostic | Yes - OTel integration + native SDKs. Any framework [src: https://blog.langchain.com/end-to-end-opentelemetry-langsmith/] |
    | Datasets storage and lineage | Yes - built-in dataset UI. Create from traces or upload. Versioning available as metadata [src: https://docs.langchain.com/langsmith/manage-datasets-in-application] |
    | Experiments | Yes - run evaluations across datasets, compare results side-by-side [src: https://docs.langchain.com/langsmith/evaluation] |
    | Evaluate prompt against Dataset (Server side) | Yes - server-side eval via API. Built-in + custom evaluators. LLM-as-judge [src: https://docs.langchain.com/langsmith/evaluation] |
    | Evaluate prompt against Dataset (Client side) | Yes - Python SDK evaluation. CI/CD or notebook triggered [src: https://docs.langchain.com/langsmith/evaluation-quickstart] |
    | Continuous Evaluation in Prod | Yes - online evaluators on live traffic. LangSmith Engine auto-detects failures [src: https://www.langchain.com/blog/introducing-langsmith-engine] |
    | Observability fields | Inputs, outputs, steps, model, tokens, cost, latency, scores, user, session, metadata, tags. Multi-modal [src: https://www.langchain.com/langsmith/observability] |
    | Dashboard capabilities | Rich UI: time, user, session, model, score, cost, latency, tags, metadata [src: https://docs.langchain.com/langsmith/cost-tracking] |
    | Prompt playground | Yes - built-in Playground. Replay production runs, edit prompts, compare models [src: https://docs.langchain.com/langsmith/prompt-engineering-quickstart] |
    | User feedback | Yes - built-in (thumbs up/down, numeric, text). SDK + UI. Linked to traces [src: https://docs.langchain.com/langsmith/annotation-queues] |
    | Annotation queue | Yes - annotation queues with rubrics, reviewer progress tracking [src: https://docs.langchain.com/langsmith/annotation-queues] |
    | Multi-modality | Yes - images, audio, files in traces. Rendered inline [src: https://www.langchain.com/langsmith/observability] |
    | Cost visibility | Yes - automatic token usage + cost recording for major providers. Custom cost data [src: https://docs.langchain.com/langsmith/cost-tracking] |
    | CI Integration | Native with pytest runner |
    | MCP Server | ? |
    | **Roles/Authentication** | |
    | Agents support | Yes - LangGraph, CrewAI, OpenAI Agents, Pydantic AI. Multi-turn. OTel for any agent [src: https://www.langchain.com/langsmith/observability] |
    | Role based on project | Yes - Organization ? Workspace. RBAC + ABAC per workspace [src: https://docs.langchain.com/langsmith/workload-isolation] |
    | RBAC | Yes - Workspace RBAC + ABAC. Custom roles (Enterprise) [src: https://docs.langchain.com/langsmith/rbac] |
    | Authentication OpenID Connect | Yes - SAML SSO. SCIM [src: https://docs.langchain.com/langsmith/enterprise] |
    | Token (member_of) | Personal Access Tokens, API keys, service keys per workspace [src: https://docs.langchain.com/langsmith/manage-organization-by-api] |
    | Prompt Registry | Yes - built-in: versioning, environments, compare, SDK deploy [src: https://docs.langchain.com/langsmith/prompt-engineering-quickstart] |
    | AI Gateway | No native gateway [src: https://docs.langchain.com/langsmith/enterprise] |
    | Conversation Simulation | No [src: https://docs.langchain.com/langsmith/evaluation] |
    | LLM as a judge & scorers | Yes - LLM-as-judge (openevals). Custom evaluators. Pairwise comparisons [src: https://docs.langchain.com/langsmith/llm-as-judge-sdk] |
    | Prompt optimization | No - manual prompt engineering via Playground only.  [src: https://docs.langchain.com/langsmith/prompt-engineering-quickstart] |
    | Admin UI | Yes - Organization settings UI for users, workspaces, roles, SSO, billing [src: https://docs.langchain.com/langsmith/administration-overview] |
    | Traces lifecycle automation (archive) | Yes - configurable TTL per project: base (14 days) or extended (up to 400 days). Workspace-level retention customization (Enterprise). API-based trace deletion (by ID or metadata). Self-hosted: blob storage lifecycle rules per TTL prefix.  [src: https://docs.langchain.com/langsmith/data-purging-compliance + https://docs.langchain.com/langsmith/self-host-ttl] |
    | Issue detections automation | Yes - LangSmith Engine: autonomous AI agent that monitors production traces, clusters failures into named issues, diagnoses root causes against code, proposes fixes and evaluators. Detects errors, evaluator failures, latency spikes, negative feedback, and unusual behavior. Can open GitHub PRs with fixes.  [src: https://docs.langchain.com/langsmith/engine + https://www.langchain.com/blog/introducing-langsmith-engine] |
    | Multi-turn Evaluation | Yes - Multi-turn Evals: evaluate entire conversation threads (not just individual turns). Multi-turn online evaluators run on live production threads. Multi-turn simulation for offline testing. Evaluators can assess context retention, goal completion, and coherence across turns.  [src: https://blog.langchain.com/insights-agent-multiturn-evals-langsmith + https://docs.langchain.com/langsmith/online-evaluations-multi-turn] |
    | Model registry | No |
    | Sandbox | Yes - LangSmith Sandboxes (GA). Hardware-virtualized microVMs for secure execution of model-generated code. Snapshot/restore, fork, resource control [src: https://www.langchain.com/blog/langsmith-sandboxes-generally-available] |
    | RBAC & SSO | Workspace RBAC + ABAC   Enterprise SSO (SAML/OIDC) |
    | **Security Requirements** | |
    | Encryption mechanisms | TLS in transit. Encryption at rest (managed infrastructure). Customer-managed encryption keys available (Enterprise).  [src: https://docs.langchain.com/langsmith/data-storage-and-privacy] |
    | PII masking | Yes - input/output filtering rules. PII detection and auto-redaction via LLM Gateway (private beta).  [src: https://docs.langchain.com/langsmith/llm-gateway-redaction] |
    | Audit logs | Yes - audit logs for organization activity. Enterprise feature.  [src: https://docs.langchain.com/langsmith/audit-logs] |
    | Data export & interoperability | Yes - REST API. OTel OTLP ingestion. Trace data exportable via API. No bulk blob export.  [src: https://docs.langchain.com/langsmith/manage-trace] |
    | SLI / SLO | No native SLI/SLO engine. Provides production metrics + alerts on feedback scores, latency, error counts with configurable thresholds (webhook/PagerDuty/Slack). No error budget concept. Closest: threshold-based alerting on traces as an SLI proxy.  [src: https://docs.langchain.com/langsmith/alerts-webhook + https://www.langchain.com/blog/langsmith-alerts] |
    | Apdex | No built-in Apdex. Latency metrics and dashboards available but no user-satisfaction index computation.  [src: https://docs.langchain.com/langsmith/alerts-webhook] |
    | Regression detection | Yes - LangSmith Engine autonomously detects regressions: clusters failures into named issues, diagnoses root causes against code, proposes fixes + evaluators to prevent recurrence. Also: threshold-based alerts on feedback scores detect quality dips. Eval-based regression tests (offline).  [src: https://docs.langchain.com/langsmith/engine + https://www.langchain.com/blog/langsmith-alerts] |
    | Resilience (patterns + feature flipping) | No built-in resilience patterns or feature flags. LangSmith is an observability/eval platform. Resilience must be in application code. However, LangSmith Deployment (LangGraph Platform) supports rollback to previous agent versions. Alerts can trigger external automation (webhook ? feature flag off).  [src: https://docs.langchain.com/langsmith/alerts-webhook + https://docs.langchain.com/langgraph-platform/api-ref-control-plane] |
    | Automated provisioning (dashboards, alerts) | Partially - REST API to create workspaces, projects, deployments programmatically. Control Plane API for LangGraph deployments. Alerts configured via UI or API. No Terraform provider. No dashboard-as-code — dashboards are project-scoped and auto-generated.  [src: https://docs.langchain.com/langsmith/smith-api/workspaces/create-workspace + https://docs.langchain.com/langgraph-platform/api-ref-control-plane] |
    | Vendor certifications | Native with pytest runner |

=== "LangSmith SaaS"

    **Langsmith SaaS**

    | Feature | Detail |
    |---------|--------|
    | **Infos** | |
    | Website | <https://smith.langchain.com> [src: https://smith.langchain.com] |
    | Licence | Proprietary SaaS. Free tier (5K traces) / Plus ($39/seat) / Enterprise [src: https://pecollective.com/blog/langsmith-pricing/] |
    | Pricing | Developer: free (5K traces/mo, 1 user). Plus: $39/seat/mo (100K traces). Enterprise: custom. Overage: $0.50/1K base traces [src: https://pecollective.com/blog/langsmith-pricing/] |
    | **Archi/technical details** | |
    | Hosting options | Fully managed SaaS (US/EU regions) [src: https://docs.langchain.com/langsmith/enterprise] |
    | Architecture components | Fully managed (same components as self-hosted, LangChain-operated) [src: https://docs.langchain.com/langsmith/scalability-and-resilience] |
    | Consulting support | Enterprise support with SLA. LangChain Professional Services [src: https://docs.langchain.com/langsmith/enterprise] |
    | Scalability | Fully managed auto-scaling. Stateless horizontal scaling [src: https://docs.langchain.com/langsmith/scalability-and-resilience] |
    | Warm-up time | No |
    | Latency / Parallelization | Async SDK ingestion. Managed auto-scaling [src: https://docs.langchain.com/langsmith/scalability-and-resilience] |
    | OpenTelemetry | Yes - OTLP ingestion. Native OTel SDKs for Python, TypeScript, Go, Java [src: https://blog.langchain.com/end-to-end-opentelemetry-langsmith/] |
    | User usage and tracking | Built-in: cost/latency/token dashboards. Custom cost data. Alerts via webhooks/PagerDuty [src: https://docs.langchain.com/langsmith/cost-tracking] |
    | Python SDK | Yes - Python, TypeScript, Go, Java SDKs [src: https://www.langchain.com/langsmith/observability] |
    | Logical isolation | Workspace based project separation [src: https://docs.langchain.com/langsmith/workload-isolation] |
    | Hard isolation | ? |
    | Storage isolation | ? |
    | Network isolation | ? |
    | **Features** | |
    | Framework agnostic | Yes - OTel integration + native SDKs. Any framework [src: https://blog.langchain.com/end-to-end-opentelemetry-langsmith/] |
    | Datasets storage and lineage | Yes - built-in dataset UI. Create from traces or upload. Versioning available as metadata [src: https://docs.langchain.com/langsmith/manage-datasets-in-application] |
    | Experiments | Yes - run evaluations across datasets, compare results side-by-side [src: https://docs.langchain.com/langsmith/evaluation] |
    | Evaluate prompt against Dataset (Server side) | Yes - server-side eval via API. Built-in + custom evaluators. LLM-as-judge [src: https://docs.langchain.com/langsmith/evaluation] |
    | Evaluate prompt against Dataset (Client side) | Yes - Python SDK evaluation. CI/CD or notebook triggered [src: https://docs.langchain.com/langsmith/evaluation-quickstart] |
    | Continuous Evaluation in Prod | Yes - online evaluators on live traffic. LangSmith Engine auto-detects failures [src: https://www.langchain.com/blog/introducing-langsmith-engine] |
    | Observability fields | Inputs, outputs, steps, model, tokens, cost, latency, scores, user, session, metadata, tags. Multi-modal [src: https://www.langchain.com/langsmith/observability] |
    | Dashboard capabilities | Rich UI: time, user, session, model, score, cost, latency, tags, metadata [src: https://docs.langchain.com/langsmith/cost-tracking] |
    | Prompt playground | Yes - built-in Playground. Replay runs, edit prompts, compare models. AI Agent Polly for auto-improvement [src: https://www.langchain.com/langsmith/evaluation] |
    | User feedback | Yes - built-in (thumbs up/down, numeric, text). SDK + UI. Linked to traces [src: https://docs.langchain.com/langsmith/annotation-queues] |
    | Annotation queue | Yes - annotation queues with rubrics, reviewer progress tracking [src: https://docs.langchain.com/langsmith/annotation-queues] |
    | Multi-modality | Yes - images, audio, files in traces. Rendered inline [src: https://www.langchain.com/langsmith/observability] |
    | Cost visibility | Yes - automatic token usage + cost recording for major providers. Custom cost data [src: https://docs.langchain.com/langsmith/cost-tracking] |
    | CI Integration | Native with pytest runner |
    | MCP Server | ? |
    | **Roles/Authentication** | |
    | Agents support | Yes - LangGraph, CrewAI, OpenAI Agents, Pydantic AI. Multi-turn. OTel for any agent [src: https://www.langchain.com/langsmith/observability] |
    | Role based on project | Yes - Organization ? Workspace. RBAC + ABAC per workspace [src: https://docs.langchain.com/langsmith/workload-isolation] |
    | RBAC | Yes - Workspace RBAC + ABAC. Custom roles (Enterprise) [src: https://docs.langchain.com/langsmith/rbac] |
    | Authentication OpenID Connect | Yes - SAML SSO. SCIM [src: https://docs.langchain.com/langsmith/enterprise] |
    | Token (member_of) | Personal Access Tokens, API keys, service keys per workspace [src: https://docs.langchain.com/langsmith/manage-organization-by-api] |
    | Prompt Registry | Yes - built-in: versioning, environments, compare, SDK deploy [src: https://docs.langchain.com/langsmith/prompt-engineering-quickstart] |
    | AI Gateway | No native gateway [src: https://docs.langchain.com/langsmith/enterprise] |
    | Conversation Simulation | No [src: https://docs.langchain.com/langsmith/evaluation] |
    | LLM as a judge & scorers | Yes - LLM-as-judge (openevals). Custom evaluators. Pairwise comparisons [src: https://docs.langchain.com/langsmith/llm-as-judge-sdk] |
    | Prompt optimization | Yes - AI Agent Polly auto-improves prompts based on evaluation results.  [src: https://www.langchain.com/langsmith/evaluation] |
    | Admin UI | Yes - Organization settings UI for users, workspaces, roles, SSO, billing [src: https://docs.langchain.com/langsmith/administration-overview] |
    | Traces lifecycle automation (archive) | Yes - configurable TTL per project: base (14 days) or extended (up to 400 days). Workspace-level retention customization (Enterprise). API-based trace deletion (by ID or metadata). Automatic purge on schedule.  [src: https://docs.langchain.com/langsmith/data-purging-compliance] |
    | Issue detections automation | Yes - LangSmith Engine: autonomous AI agent that monitors production traces, clusters failures into named issues, diagnoses root causes against code, proposes fixes and evaluators. Detects errors, evaluator failures, latency spikes, negative feedback, and unusual behavior. Can open GitHub PRs with fixes.  [src: https://docs.langchain.com/langsmith/engine + https://www.langchain.com/blog/introducing-langsmith-engine] |
    | Multi-turn Evaluation | Yes - Multi-turn Evals: evaluate entire conversation threads (not just individual turns). Multi-turn online evaluators run on live production threads. Multi-turn simulation for offline testing. Evaluators can assess context retention, goal completion, and coherence across turns.  [src: https://blog.langchain.com/insights-agent-multiturn-evals-langsmith + https://docs.langchain.com/langsmith/online-evaluations-multi-turn] |
    | Model registry | No |
    | Sandbox | Yes - LangSmith Sandboxes (GA). Hardware-virtualized microVMs for secure execution of model-generated code. Snapshot/restore, fork, resource control [src: https://www.langchain.com/blog/langsmith-sandboxes-generally-available] |
    | RBAC & SSO | Workspace RBAC + ABAC  Enterprise SSO (SAML/OIDC) [src: https://docs.langchain.com/langsmith/rbac] |
    | **Security Requirements** | |
    | Encryption mechanisms | TLS in transit. Encryption at rest (managed infrastructure).  [src: https://docs.langchain.com/langsmith/data-storage-and-privacy] |
    | PII masking | Yes - input/output filtering rules. PII detection and auto-redaction via LLM Gateway (private beta).  [src: https://docs.langchain.com/langsmith/llm-gateway-redaction] |
    | Audit logs | Yes - audit logs for organization activity. Enterprise feature.  [src: https://docs.langchain.com/langsmith/audit-logs] |
    | Data export & interoperability | Yes - REST API. OTel OTLP ingestion. Trace data exportable via API. No bulk blob export.  [src: https://docs.langchain.com/langsmith/manage-trace] |
    | SLI / SLO | No native SLI/SLO engine. Same as self-hosted: threshold alerts on feedback scores, latency, error counts (webhook/PagerDuty/Slack). No error budget concept.  [src: https://docs.langchain.com/langsmith/alerts-webhook + https://www.langchain.com/blog/langsmith-alerts] |
    | Apdex | No built-in Apdex. Same as self-hosted.  [src: https://docs.langchain.com/langsmith/alerts-webhook] |
    | Regression detection | Yes - LangSmith Engine autonomously detects regressions: clusters failures into named issues, diagnoses root causes against code, proposes fixes + evaluators to prevent recurrence. Also: threshold-based alerts on feedback scores detect quality dips. Eval-based regression tests (offline).  [src: https://docs.langchain.com/langsmith/engine + https://www.langchain.com/blog/langsmith-alerts] |
    | Resilience (patterns + feature flipping) | No built-in resilience patterns or feature flags. Same as self-hosted: observability/eval only. LangGraph Platform supports version rollback. Alerts trigger external automation.  [src: https://docs.langchain.com/langsmith/alerts-webhook + https://docs.langchain.com/langgraph-platform/api-ref-control-plane] |
    | Automated provisioning (dashboards, alerts) | Partially - same REST API as self-hosted for workspace/project/deployment creation. Alerts via API. No Terraform provider. Dashboards auto-generated per project.  [src: https://docs.langchain.com/langsmith/smith-api/workspaces/create-workspace + https://docs.smith.langchain.com/administration/how_to_guides/organization_management/manage_organization_by_api] |
    | Vendor certifications | SOC 2 Type II, GDPR, HIPAA. No ISO 27001 publicly confirmed. No ISO 42001. <br> [src: https://changelog.langchain.com/announcements/langsmith-is-now-soc-2-type-ii-compliant] |

=== "Dynatrace SaaS"

    **Dynatrace SaaS**

    | Feature | Detail |
    |---------|--------|
    | **Infos** | |
    | Website | <https://dynatrace.com> |
    | Licence | Proprietary commercial SaaS.  [src: https://www.dynatrace.com/pricing/] |
    | Pricing | ? |
    | **Archi/technical details** | |
    | Hosting options | SaaS (Dynatrace-hosted)  [src: https://www.dynatrace.com/platform/saas-upgrade/] |
    | Architecture components | OneAgent, ActiveGate, Grail (data lakehouse), Davis AI (causal AI), Smartscape, AppEngine, Cluster.  [src: https://devopsschool.com/blog/dynatrace-architecture-and-components-a-comprehensive-guide] |
    | Consulting support | Dynatrace ONE (premium), Professional Services, partner network, 24/7 support.  [src: https://www.dynatrace.com/pricing/] |
    | Scalability | Depends on deployment infrastructure |
    | Warm-up time | No |
    | Latency / Parallelization | Real-time sub-second ingestion. OneAgent <1% overhead. Native distributed tracing.  [src: https://docs.dynatrace.com/docs/observe-and-explore/dynatrace-for-ai-observability] |
    | OpenTelemetry | Yes - native OTel. OTLP API endpoints. Full GenAI semantic conventions.  [src: https://docs.dynatrace.com/docs/shortlink/ai-ml-opentelemetry] |
    | User usage and tracking | Yes. Session replay. User journey analytics. Cost per AI service.  [src: https://dynatrace.com/hub/detail/ai-and-llm-observability] |
    | Python SDK | Yes - OneAgent auto-injection + OTel SDK for custom instrumentation.  [src: https://docs.dynatrace.com/docs/shortlink/ai-ml-opentelemetry] |
    | Logical isolation | Account ? Environment ? Management Zone |
    | Hard isolation | Separate environments (each gets dedicated storage) |
    | Storage isolation | Per-environment + management zones |
    | Network isolation | ActiveGate proxy- VPC-scoped agents |
    | **Features** | |
    | Framework agnostic | Yes - OTel-native. Auto-discovers any instrumented service.  [src: https://docs.dynatrace.com/docs/shortlink/ai-ml-opentelemetry] |
    | Datasets storage and lineage | No - not a dataset management tool.  <br>Dynatrace allows to filter on datasets metadata but it does not handle the versioning <br>Roadmap announces new features but it is unclear at mid-june 2026[src: https://www.dynatrace.com/news/blog/evaluate-llm-and-agent-quality-in-dynatrace-ai-observability/)] |
    | Experiments | No - A/B testing via traffic split is available, no offline experiments. Roadmap is unclear on this topic  [src: https://docs.dynatrace.com/docs/analyze-explore-automate/dynatrace-for-ai-observability/get-started/sample-use-cases/ab-model-testing] |
    | Evaluate prompt against Dataset (Server side) | Yes (alpha release) with the cli dt-evals. Offline and online evaluation is possible [src: hhttps://github.com/dynatrace-oss/dt-evals] |
    | Evaluate prompt against Dataset (Client side) | Yes (alpha release) with the cli dt-evals. Offline and online evaluation is possible [src: hhttps://github.com/dynatrace-oss/dt-evals] |
    | Continuous Evaluation in Prod | Yes - real-time monitoring (latency, errors, cost, drift). Davis AI anomaly detection. Alpha release of dt-evals.  [src: https://www.dynatrace.com/news/blog/dynatrace-accelerates-business-transformation-with-new-ai-observability-solution/] |
    | Observability fields | Full-stack: infra metrics + app traces + LLM spans (I/O, tokens, cost, latency, model, error). Browser to GPU.  [src: https://dynatrace.com/hub/detail/ai-and-llm-observability] |
    | Dashboard capabilities | Advanced and customizable: DQL queries, topology views, AI-powered filtering. Any dimension.  [src: https://docs.dynatrace.com/docs/analyze-explore-automate/dynatrace-for-ai-observability/ai-observability-app] |
    | Prompt playground | No - not a prompt engineering tool.  [src: https://docs.dynatrace.com/docs/observe-and-explore/dynatrace-for-ai-observability] |
    | User feedback | No - monitors UX metrics (RUM) but no LLM feedback mechanism.  [src: https://docs.dynatrace.com/docs/observe-and-explore/dynatrace-for-ai-observability] |
    | Annotation queue | No.  [src: https://docs.dynatrace.com/docs/observe-and-explore/dynatrace-for-ai-observability] |
    | Multi-modality | Yes - traces multi-modal API calls. Captures metadata, no inline rendering.  [src: https://www.dynatrace.com/news/blog/ai-agent-observability-amazon-bedrock-agents-monitoring/] |
    | Cost visibility | Yes - full cost tracking for AI/LLM services via Davis AI. Predicts and forecasts cost changes. Per-model, per-service cost attribution.  [src: https://www.dynatrace.com/news/press-release/dynatrace-advances-ai-observability-to-support-generative-ai-initiatives/] |
    | CI Integration | ? |
    | MCP Server | ? |
    | **Roles/Authentication** | |
    | Agents support | Yes - Bedrock AgentCore, LangGraph, MCP/A2A. Auto-discovery of agent topology.  [src: https://www.dynatrace.com/news/blog/announcing-amazon-bedrock-agentcore-agent-observability/] |
    | Role based on project | Yes - Management Zones + environments. Granular per-team.  [src: https://docs.dynatrace.com/docs/manage/account-management] |
    | RBAC | Yes - comprehensive. Policies, groups, roles. Environment + Management Zone level.  [src: https://docs.dynatrace.com/docs/manage/account-management] |
    | Authentication OpenID Connect | Yes - SAML 2.0 + OIDC. Any compliant IdP. MFA.  [src: https://docs.dynatrace.com/docs/manage/account-management] |
    | Token (member_of) | Yes - API tokens, OAuth clients, PATs. Group-based policies.  [src: https://docs.dynatrace.com/docs/manage/account-management] |
    | Prompt Registry | To be confirmed. Dynatrace has prompt versioning and lineage ability. Does it have the ability to use alias as a registry can offer it to be confirmed.  [https://docs.dynatrace.com/docs/observe/dynatrace-for-ai-observability/ai-observability-app] |
    | AI Gateway | No dedicated gateway - monitors any gateway via distributed tracing.  [src: https://docs.dynatrace.com/docs/observe-and-explore/dynatrace-for-ai-observability] |
    | Conversation Simulation | No - not an evaluation platform.  [src: https://docs.dynatrace.com/docs/observe-and-explore/dynatrace-for-ai-observability] |
    | LLM as a judge & scorers | yes available in preview on mid-june 2026  [src: https://docs.dynatrace.com/docs/observe-and-explore/dynatrace-for-ai-observability] |
    | Prompt optimization | No - not a prompt engineering tool.  [src: https://docs.dynatrace.com/docs/observe-and-explore/dynatrace-for-ai-observability] |
    | Admin UI | Yes - full Settings app for environment/account management. IAM, policies, groups, all configurable from web UI [src: https://docs.dynatrace.com/docs/manage/settings] |
    | Traces lifecycle automation (archive) | Yes - Grail bucket-level retention from 10 days to 10 years. Configurable per bucket (per team/use-case). Adaptive Data Retention auto-adjusts when storage quota exceeded. Automated lifecycle — no manual cleanup needed.  [src: https://docs.dynatrace.com/docs/analyze-explore-automate/distributed-tracing/data-retention] |
    | Issue detections automation | Yes - Davis AI provides out-of-the-box anomaly detection on any numeric time series (latency, error rates, token costs). Predictive AI detects drift in LLM usage patterns. Custom anomaly detectors configurable on Grail data. Automatic root cause analysis. No LLM-judge-based quality detection.  [src: https://dynatrace.com/hub/detail/davis-anomaly-detection + https://dynatrace.com/news/blog/create-a-davis-ai-anomaly-detector-on-grail] |
    | Multi-turn Evaluation | Yes (in preview) you can evaluate trajectory and multi turns agent.  [src: https://docs.dynatrace.com/docs/observe-and-explore/dynatrace-for-ai-observability] |
    | Model registry | No |
    | Sandbox | No - not an evaluation/execution platform [src: https://docs.dynatrace.com/docs/observe-and-explore/dynatrace-for-ai-observability] |
    | RBAC & SSO | Comprehensive RBAC (policies, groups, roles) + SAML 2.0/OIDC SSO.  [src: https://docs.dynatrace.com/docs/manage/account-management] |
    | **Security Requirements** | |
    | Encryption mechanisms | AES-256 encryption at rest (Grail storage). TLS 1.2+ in transit. FIPS 140-2 validated cryptographic modules.  [src: https://docs.dynatrace.com/docs/manage/data-privacy-and-security] |
    | PII masking | Yes - data masking rules configurable per environment. Management Zone-level data access restrictions.  [src: https://docs.dynatrace.com/docs/manage/data-privacy-and-security] |
    | Audit logs | Yes - full audit trail. Davis AI activity tracking. Environment-level audit logs. Configurable retention.  [src: https://docs.dynatrace.com/docs/manage/account-management] |
    | Data export & interoperability | Yes - DQL API for data export. Grail data accessible via API. OTel-native traces exportable. Metrics/logs export via API.  [src: https://docs.dynatrace.com/docs/analyze-explore-automate/dynatrace-api] |
    | SLI / SLO | Yes - first-class SLO management. Define SLIs (availability, latency, error rate) and SLOs with error budgets. Davis AI predicts SLO violations proactively and pinpoints root causes. Burn-rate alerting. SLO-based release validation via Site Reliability Guardian.  [src: https://docs.dynatrace.com/docs/shortlink/slo-overview + https://www.dynatrace.com/news-next/blog/dynatrace-ai-predicts-slo-violations-and-pinpoints-root-causes-proactively/] |
    | Apdex | Yes - native Apdex. Configurable thresholds per service/application (Satisfactory/Tolerable/Frustrating). Automatic Apdex scoring on web, mobile, and custom services. Can be applied to AI service endpoints via RUM or server-side metrics.  [src: https://www.dynatrace.com/support/help/how-to-use-dynatrace/real-user-monitoring/setup-and-configuration/configure-apdex + https://www.dynatrace.com/knowledge-base/apdex/] |
    | Regression detection | Yes - Davis AI anomaly detection on any numeric metric (latency, error rate, token cost). Automatic baseline learning and regression detection. Site Reliability Guardian validates releases against SLOs — blocks deployments if regression detected.  [src: https://docs.dynatrace.com/docs/deliver/release-validation-automated + https://dynatrace.com/hub/detail/davis-anomaly-detection] |
    | Resilience (patterns + feature flipping) | Yes - full-stack resilience observability + progressive delivery. DevCycle by Dynatrace (acquired): native feature flag management with progressive rollouts, A/B experiments, kill switches. Site Reliability Guardian validates feature impact against SLOs. Automated rollback via Workflows when SLO violated. Real-time topology-aware impact analysis.  [src: https://www.dynatrace.com/hub/detail/devcycle-by-dt/ + https://docs.dynatrace.com/docs/deliver/release-validation-automated + https://www.dynatrace.com/news/blog/progressive-delivery-done-right/] |
    | Automated provisioning (dashboards, alerts) | Yes - full Configuration as Code (CaC). Dynatrace Terraform Provider covers all resources: SLOs, dashboards, alerting profiles, anomaly detectors, management zones, synthetic monitors. Monaco CLI for YAML-based config management. Automated provisioning on environment creation. Self-service model for dev teams.  [src: https://docs.dynatrace.com/docs/shortlink/configuration-as-code + https://github.com/dynatrace-oss/terraform-provider-dynatrace] |
    | Vendor certifications | SOC 2 Type II, SOC 1 Type II, ISO 27001:2022, FedRAMP Moderate, HIPAA, TISAX, CSA STAR 2, IRAP, ENS. No ISO 42001 announced. Strongest compliance portfolio for regulated European industries. <br> [src: https://www.dynatrace.com/company/trust-center/] |

=== "Phoenix (SH)"

    **Arize Phoenix (self hosted)**

    | Feature | Detail |
    |---------|--------|
    | **Infos** | |
    | Website | <https://arize.com/docs/phoenix> [src: https://github.com/Arize-ai/phoenix] |
    | Licence | Source-available (not OSI open source) - Elastic License 2.0 (ELv2). Cannot be offered to third parties as a hosted/managed service. No license fees, no usage limits, no feature gates when self-hosting [src: https://github.com/Arize-ai/phoenix + https://arize.com/docs/phoenix/self-hosting] |
    | Pricing | Infrastructure costs only (compute, PostgreSQL) - see Infra cost tab [src: https://arize.com/docs/phoenix/self-hosting] |
    | **Archi/technical details** | |
    | Hosting options | Self-hosted only: Docker / Docker Compose, Kubernetes, Helm, AWS (CloudFormation), Google Cloud Run, Azure (ARM templates), Railway, Render. Air-gapped deployments supported [src: https://arize.com/docs/phoenix/self-hosting] |
    | Architecture components | Single container: Web UI, trace collector (OTLP), SQL backend (SQLite for dev, PostgreSQL 14+ for prod, optional read replica) [src: https://arize.com/docs/phoenix/self-hosting/architecture + https://arize.com/docs/phoenix/release-notes] |
    | Consulting support | Community (GitHub issues). No SLA for OSS [src: https://github.com/Arize-ai/phoenix] |
    | Scalability | Depends on target infrastructure. Horizontal scaling: multiple instances sharing one PostgreSQL database behind a load balancer [src: https://arize.com/docs/phoenix/self-hosting/architecture] |
    | Warm-up time | NA |
    | Latency / Parallelization | OTel-based SDKs (gRPC and HTTP exporters). Read queries can be routed to a PostgreSQL read replica [src: https://github.com/Arize-ai/phoenix + https://arize.com/docs/phoenix/release-notes] |
    | OpenTelemetry | Yes - built on OpenTelemetry + OpenInference. OTel GenAI semantic conventions auto-converted to OpenInference at ingest (May 2026). OTLP project routing via x-project-name header [src: https://github.com/Arize-ai/phoenix + https://arize.com/docs/phoenix/release-notes] |
    | User usage and tracking | Built-in: cost, latency, tokens per trace, session, project and experiment. Metric charts above tables [src: https://arize.com/docs/phoenix/tracing/how-to-tracing/cost-tracking + https://arize.com/docs/phoenix/release-notes] |
    | Python SDK | Yes - Python, TypeScript, Java, Go. Also px CLI [src: https://github.com/Arize-ai/phoenix] |
    | Logical isolation | Project level. One instance = one tenant: all projects visible according to instance-wide roles. Resource tags for finer access planned later in 2026 [src: https://arize.com/docs/phoenix/self-hosting/architecture] |
    | Hard isolation | Requires separate deployments (separate instances with independent databases) [src: https://arize.com/docs/phoenix/self-hosting/architecture] |
    | Storage isolation | Per instance, or one PostgreSQL schema per instance (PHOENIX_SQL_DATABASE_SCHEMA) [src: https://arize.com/docs/phoenix/self-hosting/architecture] |
    | Network isolation | User-managed. Outbound access controls, SSRF and CSRF protections [src: https://arize.com/docs/phoenix/self-hosting] |
    | **Features** | |
    | Framework agnostic | Yes - OTel/OpenInference. 50+ framework and LLM provider integrations [src: https://github.com/Arize-ai/phoenix] |
    | Datasets storage and lineage | Yes - versioned datasets. Create from traces (span associations kept), upload CSV/JSONL, dataset splits. Edits saved as new version [src: https://arize.com/docs/phoenix/datasets-and-experiments/overview-datasets + https://arize.com/docs/phoenix/release-notes] |
    | Experiments | Yes - run experiments on datasets, compare runs, metric charts [src: https://arize.com/docs/phoenix/datasets-and-experiments/overview-datasets + https://arize.com/docs/phoenix/release-notes] |
    | Evaluate prompt against Dataset (Server side) | Yes - dataset evaluators run server-side on Playground experiments: LLM judges + Python/TypeScript code evaluators in sandbox [src: https://arize.com/docs/phoenix/evaluation/server-evals/overview] |
    | Evaluate prompt against Dataset (Client side) | Yes - phoenix-evals library + Python/TS clients. CI/CD or notebook triggered [src: https://github.com/Arize-ai/phoenix + https://arize.com/docs/phoenix/evaluation/integrations/pytest] |
    | Continuous Evaluation in Prod | Not natively - server-side evaluators are tied to datasets/experiments, not continuous monitoring of production traces. Must build external pipeline [src: https://arize.com/docs/phoenix/evaluation/server-evals/overview] |
    | Observability fields | Inputs, outputs, spans (LLM, tool, agent, retriever, decision), model, tokens, cost, latency, session, user, metadata, tags, annotations, notes. Images [src: https://arize.com/docs/phoenix/release-notes + https://arize.com/docs/phoenix/tracing/how-to-tracing/advanced/multimodal-tracing] |
    | Dashboard capabilities | Trace/session filter expressions (plain-English AI query), time range picker, metric charts above tables, read-only analytics SQL (via MCP). No custom dashboards [src: https://arize.com/docs/phoenix/release-notes] |
    | Prompt playground | Yes - built-in Playground. Compare models, replay traced calls, run against datasets, provider-native tools [src: https://github.com/Arize-ai/phoenix + https://arize.com/docs/phoenix/release-notes] |
    | User feedback | Yes - feedback captured via SDK/API as annotations linked to spans. Session feedback toolbar (thumbs up/down) in UI [src: https://arize.com/docs/phoenix/tracing/how-to-tracing/feedback-and-annotations/capture-feedback + https://arize.com/docs/phoenix/release-notes] |
    | Annotation queue | No dedicated annotation queue documented in OSS. Manual annotation of spans/traces/sessions via UI, SDK, CLI [src: https://arize.com/docs/phoenix/tracing/llm-traces/how-to-annotate-traces + https://arize.com/docs/phoenix/release-notes] |
    | Multi-modality | Yes - images (base64 or URL) rendered inline in traces [src: https://arize.com/docs/phoenix/tracing/how-to-tracing/advanced/multimodal-tracing] |
    | Cost visibility | Yes - automatic from token counts + built-in pricing table (auto-updated daily). Custom pricing. Aggregation by trace, session, project, experiment [src: https://arize.com/docs/phoenix/tracing/how-to-tracing/cost-tracking] |
    | CI Integration | Native with pytest plugin (@pytest.mark.phoenix) + Vitest. Pass/fail gating via exit code [src: https://arize.com/docs/phoenix/evaluation/integrations/pytest] |
    | MCP Server | Yes - @arizeai/phoenix-mcp (npx) + remote MCP endpoint /mcp (beta, July 2026): projects, spans, annotations, prompts, datasets, experiments [src: https://arize.com/docs/phoenix/integrations/phoenix-mcp-server + https://arize.com/docs/phoenix/release-notes] |
    | **Roles/Authentication** | |
    | Agents support | Yes - OpenInference instrumentors (LangChain, LlamaIndex, CrewAI, Claude Agent SDK, Atomic Agents...). Agent/tool/decision spans. Tool selection and invocation evaluators [src: https://github.com/Arize-ai/phoenix + https://arize.com/docs/phoenix/release-notes] |
    | Role based on project | No - roles are instance-wide. Resource tags for fine-grained access planned later in 2026 [src: https://arize.com/docs/phoenix/self-hosting/architecture] |
    | RBAC | Yes - 3 built-in roles: Admin, Member, Viewer. Viewers read-only on REST API [src: https://arize.com/docs/phoenix/self-hosting/features/authentication] |
    | Authentication OpenID Connect | Yes - OAuth2/OIDC (Google, AWS Cognito, Microsoft Entra ID, any OIDC provider). Group-based access + role mapping. LDAP also supported [src: https://arize.com/docs/phoenix/self-hosting/features/authentication] |
    | Token (member_of) | System keys (admin-created) + user keys inheriting user permissions. Bearer token [src: https://arize.com/docs/phoenix/self-hosting/features/authentication] |
    | Prompt Registry | Yes - prompt management: versioning, tags, version diff view, SDK/REST access [src: https://arize.com/docs/phoenix/prompt-engineering/overview-prompts/prompt-management + https://arize.com/docs/phoenix/release-notes] |
    | AI Gateway | Partially - OpenAI-compatible /v1/chat/completions proxy to any provider (Aug 2026), custom providers. No documented rate limiting or cost controls [src: https://arize.com/docs/phoenix/release-notes] |
    | Conversation Simulation | No - not documented. ATIF trajectory upload and Harbor plugin for agent benchmarks [src: https://arize.com/docs/phoenix/release-notes] |
    | LLM as a judge & scorers | Yes - built-in LLM evaluators (correctness, tool selection/invocation, PII detection, user friction, completeness, retrieval relevance) + custom + code evaluators [src: https://arize.com/docs/phoenix/evaluation/server-evals/overview + https://arize.com/docs/phoenix/release-notes] |
    | Prompt optimization | Partially - via separate Prompt Learning SDK (meta-prompting: an LLM rewrites the prompt from outputs + feedback) [src: https://arize.com/docs/phoenix/prompt-engineering/tutorial/optimize-prompts-automatically] |
    | Admin UI | Yes - Settings UI for users, roles, API keys, secrets, retention policies [src: https://arize.com/docs/phoenix/self-hosting/features/authentication + https://arize.com/docs/phoenix/settings/data-retention + https://arize.com/docs/phoenix/release-notes] |
    | Traces lifecycle automation (archive) | Yes - retention policies per project (max days and/or trace count), CRON-scheduled purge (weekly by default). Default policy = indefinite (PHOENIX_DEFAULT_RETENTION_POLICY_DAYS). Purge only, no archive tier [src: https://arize.com/docs/phoenix/settings/data-retention] |
    | Issue detections automation | Partially - PXI agent (since June 2026) performs on-demand error analysis on traces and can file GitHub issues with approval. Not documented as scheduled detection [src: https://arize.com/docs/phoenix/pxi + https://arize.com/docs/phoenix/release-notes] |
    | Multi-turn Evaluation | Partially - sessions, session-level annotations, session turns API (get_session_turns) to build multi-turn evals. Session-level server evaluators not documented [src: https://arize.com/docs/phoenix/tracing/llm-traces/sessions + https://arize.com/docs/phoenix/release-notes] |
    | Model registry | No |
    | Sandbox | Yes - code evaluators run in sandboxes (WebAssembly, Deno, E2B, Daytona, Vercel, Modal, Docker) [src: https://arize.com/docs/phoenix/release-notes] |
    | RBAC & SSO | Instance RBAC (Admin/Member/Viewer) + OAuth2/OIDC SSO + LDAP [src: https://arize.com/docs/phoenix/self-hosting/features/authentication] |
    | **Security Requirements** | |
    | Encryption mechanisms | TLS natively (PHOENIX_TLS_ENABLED) or at load balancer / service mesh. At rest: infrastructure-level (managed PostgreSQL or disk encryption). PHOENIX_SECRET master key: passwords PBKDF2-HMAC-SHA256, provider credentials Fernet (AES-128-CBC + HMAC-SHA256) [src: https://arize.com/docs/phoenix/self-hosting/advanced/encryption] |
    | PII masking | Yes - client-side: OpenInference env vars (hide inputs/outputs) + custom span processors. PII Detection evaluator (Sep 2026) [src: https://arize.com/docs/phoenix/tracing/how-to-tracing/advanced/masking-span-attributes + https://arize.com/docs/phoenix/release-notes] |
    | Audit logs | No built-in audit logs documented [src: https://arize.com/docs/phoenix/self-hosting] |
    | Data export & interoperability | Yes - REST API, Python/TS clients (get_traces, spans), px CLI. Download spans/traces as OTLP JSON or JSONL. Server logs export over OTLP [src: https://arize.com/docs/phoenix/tracing/how-to-tracing/importing-and-exporting-traces + https://arize.com/docs/phoenix/release-notes] |
    | SLI / SLO | No native SLI/SLO engine. No alerting documented. Metrics must be exported to external tooling [src: https://arize.com/docs/phoenix/release-notes] |
    | Apdex | No. Not an APM tool |
    | Regression detection | No automated regression detection. Experiment comparison + pytest/Vitest gating in CI [src: https://arize.com/docs/phoenix/evaluation/integrations/pytest] |
    | Resilience (patterns + feature flipping) | No built-in resilience or feature flag support. Observability/eval layer only [src: https://github.com/Arize-ai/phoenix] |
    | Automated provisioning (dashboards, alerts) | Partially - configuration via env vars, Helm chart, REST API (projects, datasets, prompts, sessions). No dashboard/alert-as-code [src: https://arize.com/docs/phoenix/self-hosting/deployment-options/kubernetes-helm + https://arize.com/docs/phoenix/release-notes] |
    | Vendor certifications | NA |

=== "Arize AX SaaS"

    **Arize AX SaaS**

    | Feature | Detail |
    |---------|--------|
    | **Infos** | |
    | Website | <https://arize.com> [src: https://arize.com/docs/ax] |
    | Licence | Proprietary SaaS. AX Free / AX Pro / AX Enterprise tiers. Self-hosted option on Enterprise only [src: https://arize.com/pricing/] |
    | Pricing | AX Free: free (25K spans/mo, 1 GB/mo, 15 days retention). AX Pro: $50/mo (50K spans/mo, 10 GB/mo, 30 days retention). Enterprise: custom. Unlimited users on all tiers. No overage pricing published [src: https://arize.com/pricing/] |
    | **Archi/technical details** | |
    | Hosting options | SaaS in US, EU and Canada regions. Enterprise: self-hosted in customer VPC / on-prem on Kubernetes (GKE, EKS, AKS, IBM Cloud, OpenShift, Rancher bare metal), self-managed or managed by Arize [src: https://arize.com/docs/ax/security-and-settings/compliance/data-regions + https://arize.com/docs/ax/selfhosting/info + https://arize.com/docs/ax/selfhosting/installation] |
    | Architecture components | Fully managed by Arize. Self-hosted: control plane + data plane deployed on Kubernetes, object storage buckets (S3, GCS, Azure Blob or S3-compatible) [src: https://arize.com/docs/ax/selfhosting/architecture/platform-options + https://arize.com/docs/ax/selfhosting/getting-started/prerequisites] |
    | Consulting support | Pro: standard SLA. Enterprise: custom SLAs. Self-hosted: premier support by the Arize engineering team [src: https://arize.com/pricing/ + https://arize.com/docs/ax/selfhosting/info] |
    | Scalability | Fully managed. Span/storage quotas on Free and Pro, custom on Enterprise [src: https://arize.com/pricing/] |
    | Warm-up time | No |
    | Latency / Parallelization | OTLP gRPC and HTTP exporters. Online evals with configurable sampling (1-100%) [src: https://arize.com/docs/ax/integrations/opentelemetry/overview + https://arize.com/docs/ax/evaluate/run-evals-on-traces] |
    | OpenTelemetry | Yes - OTLP gRPC + HTTP exporters with OpenInference semantic conventions. OpenLLMetry and OpenLIT also supported [src: https://arize.com/docs/ax/integrations/opentelemetry/overview] |
    | User usage and tracking | Built-in: dashboards (templates: Tracing Project Overview, Token Tracking and Latency). Cost per model, user, session [src: https://arize.com/docs/ax/observe/dashboards + https://arize.com/docs/ax/instrument/track-costs] |
    | Python SDK | Yes - Python SDK (v8), TypeScript, Go SDKs + CLI. REST and GraphQL APIs [src: https://arize.com/docs/api-clients/python/version-8/overview + https://arize.com/docs/api-reference/overview] |
    | Logical isolation | Account > Organization > Space > Project hierarchy with roles at each level [src: https://arize.com/docs/ax/security-and-settings/sso-and-rbac/account-and-space-roles] |
    | Hard isolation | Enterprise self-hosted: data remains in customer VPC [src: https://arize.com/docs/ax/selfhosting/info] |
    | Storage isolation | Data region choice (US/EU/Canada). Self-hosted: customer-owned buckets [src: https://arize.com/docs/ax/security-and-settings/compliance/data-regions + https://arize.com/docs/ax/selfhosting/architecture/platform-options] |
    | Network isolation | Arize Private Connect + IP whitelisting [src: https://arize.com/docs/ax/security-and-settings/arize-private-connect + https://arize.com/docs/ax/security-and-settings/whitelisting] |
    | **Features** | |
    | Framework agnostic | Yes - OTel/OpenInference, OpenLLMetry, OpenLIT [src: https://arize.com/docs/ax/integrations/opentelemetry/overview] |
    | Datasets storage and lineage | Yes - datasets curated from traces, used for experiments and Playground [src: https://arize.com/docs/ax/improve/build-a-dataset] |
    | Experiments | Yes - compare prompt, model or agent changes against datasets. Remote agent experiments [src: https://arize.com/docs/ax + https://arize.com/docs/ax/improve/remote-agent-experiments] |
    | Evaluate prompt against Dataset (Server side) | Yes - evaluator tasks run server-side: LLM-as-judge, code, agent-as-a-judge, remote evaluators [src: https://arize.com/docs/ax/evaluate/run-evals-on-traces] |
    | Evaluate prompt against Dataset (Client side) | Yes - export spans, run Phoenix Evals locally, log results back via Python SDK. client.experiments.run() [src: https://arize.com/docs/ax/evaluate/run-evals-on-traces + https://arize.com/docs/ax/improve/ci-cd-for-automated-experiments] |
    | Continuous Evaluation in Prod | Yes - online eval tasks attached to live projects, sampling 1-100%, historical backfill. Monitors alert on eval labels [src: https://arize.com/docs/ax/evaluate/run-evals-on-traces + https://arize.com/docs/ax/observe/production-monitoring] |
    | Observability fields | Inputs, outputs, tools, costs, latency, status, token counts, eval labels, custom metadata, sessions [src: https://arize.com/docs/ax + https://arize.com/docs/ax/observe/production-monitoring] |
    | Dashboard capabilities | Custom dashboards: time series, statistic, distribution, pivot table, scatter plot, eval score widgets. Global + per-widget filters, group-by. Alyx can build dashboards [src: https://arize.com/docs/ax/observe/dashboards] |
    | Prompt playground | Yes - run prompts across datasets, Compare Models, side-by-side variants with diff, evaluators on every row [src: https://arize.com/docs/ax/improve/prompt-playground] |
    | User feedback | Yes - annotations (categorical, score, text) via UI, Python SDK or Alyx, on traces, spans, sessions [src: https://arize.com/docs/ax/evaluate/human-review] |
    | Annotation queue | Yes - Labeling Queues: assignment, review instructions, reassignment, completion tracking, dedicated RBAC roles [src: https://arize.com/docs/ax/evaluate/human-review + https://arize.com/blog/arize-releases-labeling-queues-expand-collapse-rows-in-trace-table/] |
    | Multi-modality | Yes - audio and image tracing and evaluation [src: https://arize.com/docs/ax/cookbooks/evaluate/tracing-and-evaluating-audio + https://arize.com/docs/ax/cookbooks/evaluate/evaluate-receipt-agents-with-image-judge] |
    | Cost visibility | Yes - automatic cost per LLM call and trace. Custom cost configs. Aggregation by model, user, session [src: https://arize.com/docs/ax/instrument/track-costs] |
    | CI Integration | Manual using SDK (experiment script exits non-zero below threshold). Guides for GitHub Actions, GitLab, Azure DevOps, Jenkins, Harness [src: https://arize.com/docs/ax/improve/ci-cd-for-automated-experiments] |
    | MCP Server | Yes - hosted MCP server (alpha) over REST API v2: query traces, run experiments, manage evaluators and prompts, annotate [src: https://arize.com/docs/api-clients/mcp/overview] |
    | **Roles/Authentication** | |
    | Agents support | Yes - agent tracing (inputs, outputs, tools, costs). Agent-as-a-judge evaluators, session evals [src: https://arize.com/docs/ax + https://arize.com/docs/ax/evaluate/agent-as-a-judge] |
    | Role based on project | Yes - project-level roles (Viewer, Editor, Admin) on top of space roles [src: https://arize.com/docs/ax/security-and-settings/sso-and-rbac/account-and-space-roles] |
    | RBAC | Yes - managed roles at account, organization, space, project level + custom space roles (21 permission categories). RBAC REST API [src: https://arize.com/docs/ax/security-and-settings/sso-and-rbac/account-and-space-roles + https://arize.com/docs/ax/security-and-settings/sso-and-rbac/rbac-rest-api] |
    | Authentication OpenID Connect | Yes - SAML SSO (e.g. Okta) with SAML role mapping. Enterprise plan [src: https://arize.com/docs/ax/security-and-settings/sso-and-rbac/saml-configuration + https://arize.com/pricing/] |
    | Token (member_of) | API keys + service keys. Member - UI Only role has no API/service key permissions [src: https://arize.com/docs/ax/security-and-settings/api-keys + https://arize.com/docs/ax/security-and-settings/sso-and-rbac/account-and-space-roles] |
    | Prompt Registry | Yes - Prompt Hub: immutable versions with commit messages, tags (e.g. production) resolved at runtime, SDK loading, CI/CD [src: https://arize.com/docs/ax/improve/prompt-playground + https://arize.com/docs/ax/concepts/prompts/prompt-hub] |
    | AI Gateway | No native gateway - integrates with external gateways (e.g. Portkey) [src: https://portkey.ai/docs/aigw/integrations/tracing-providers/arize] |
    | Conversation Simulation | No - not documented |
    | LLM as a judge & scorers | Yes - LLM-as-a-judge, agent-as-a-judge, decision judge, code evaluators. Align evals to human feedback [src: https://arize.com/docs/ax/evaluate/evaluators/llm-as-a-judge + https://arize.com/docs/ax/evaluate/align-evals-to-human-feedback] |
    | Prompt optimization | Yes - Prompt Learning (meta-prompting) as SDK or UI task + conversational optimization with Alyx. Results saved as new Prompt Hub versions [src: https://arize.com/docs/ax/concepts/prompts/optimizing-prompts] |
    | Admin UI | Yes - Settings UI for members, roles, SSO, API/service keys, cost configs [src: https://arize.com/docs/ax/security-and-settings/sso-and-rbac/invite-and-manage-members + https://arize.com/docs/ax/instrument/track-costs] |
    | Traces lifecycle automation (archive) | Yes - plan-based retention (Free 15 days, Pro 30 days, Enterprise custom), automatic deletion per retention policy. On-demand deletion of sensitive traces [src: https://arize.com/pricing/ + https://arize.com/docs/ax/security-and-settings/compliance/delete-traces-with-sensitive-data] |
    | Issue detections automation | Yes - Signal: scheduled scans group recurring failures into issues with investigation (trace evidence + suggested fix). Claude Code as default harness. Free 10 / Pro 25 / Enterprise unlimited issues per month; fix PRs Enterprise only [src: https://arize.com/docs/ax/observe/signal] |
    | Multi-turn Evaluation | Yes - session evals score every trace sharing a session.id (conversation or turn-data mode). Run as online tasks. Results filterable per session [src: https://arize.com/docs/ax/evaluate/trace-and-session-evals] |
    | Model registry | No |
    | Sandbox | No - not documented |
    | RBAC & SSO | Account/Org/Space/Project RBAC + custom roles. SAML SSO (Enterprise) [src: https://arize.com/docs/ax/security-and-settings/sso-and-rbac/account-and-space-roles + https://arize.com/pricing/] |
    | **Security Requirements** | |
    | Encryption mechanisms | Not detailed in public docs - see Arize Trust Center [src: https://arize.com/docs/ax/security-and-settings/compliance + https://arize.com/trust-center/] |
    | PII masking | Yes - client-side only: OpenInference TraceConfig/env vars + span processors (regex, Microsoft Presidio) [src: https://arize.com/docs/ax/instrument/mask-and-redact-data] |
    | Audit logs | Yes (Enterprise) - logins (success/failed), user operations, data exports. Queried via GraphQL API, Account Admin role required [src: https://arize.com/docs/ax/security-and-settings/compliance/arize-audit-log + https://arize.com/pricing/] |
    | Data export & interoperability | Yes - Data Fabric (Enterprise): sync tracing data to own bucket (S3/GCS/Azure Blob) as Iceberg or Delta Lake, readable from BigQuery, Databricks Unity Catalog, Snowflake. REST + GraphQL APIs [src: https://arize.com/docs/ax/security-and-settings/data-fabric] |
    | SLI / SLO | No native SLI/SLO engine. Monitors with automatic (historical) or static thresholds on span attributes, eval labels, custom metrics. Alerts: Email, Slack, PagerDuty, OpsGenie, Teams, Webhooks [src: https://arize.com/docs/ax/observe/production-monitoring] |
    | Apdex | No. Not documented [src: https://arize.com/docs/ax/observe/production-monitoring] |
    | Regression detection | Yes - monitors with automatic thresholds from historical data + Signal recurring issues. CI experiment gating [src: https://arize.com/docs/ax/observe/production-monitoring + https://arize.com/docs/ax/observe/signal + https://arize.com/docs/ax/improve/ci-cd-for-automated-experiments] |
    | Resilience (patterns + feature flipping) | No built-in resilience or feature flag support. Alerts via webhooks can trigger external automation [src: https://arize.com/docs/ax/observe/production-monitoring] |
    | Automated provisioning (dashboards, alerts) | Partially - REST API v2, GraphQL, RBAC REST API, custom metrics API. Terraform/Kubernetes config for self-hosted. No Terraform provider for dashboards/monitors documented [src: https://arize.com/docs/ax/rest-reference/overview + https://arize.com/docs/ax/selfhosting/info] |
    | Vendor certifications | SOC 2 Type II, ISO/IEC 27001, PCI DSS 4.0, HIPAA, GDPR, CSA STAR Level 1. No ISO 42001 disclosed. <br> [src: https://arize.com/docs/ax/security-and-settings/compliance + https://arize.com/blog/arize-ai-achieves-iso-iec-27001-certification + https://arize.com/trust-center/] |

---

## Options

### Key elements

- **Security & vendor risk**: Handling secrets and confidential data requires a vendor risk assessment for each solution. For example, the current Dynatrace risk assessment is approved for Confidential data. SaaS solutions
may impose stricter data residency and access constraints than self-hosted alternatives.

- **Rapid iteration across the market**: All solutions in this benchmark evolve very quickly — some features may only be available in alpha, beta, or pre-release. Planning should distinguish between a short-term target (6–9
months) and a longer-term target to account for upcoming capabilities.

- **Convergence of observability and evaluation**: Over time, native observability platforms (e.g., Dynatrace) are adding evaluation features to their roadmaps. While their approach remains primarily log- and metrics-driven,
they already offer highly customizable dashboards that could partially cover evaluation use cases in the future.

### Recommended Tool Combinations

- **Dynatrace alone** evolves quickly and has evaluation features but **in preview** . It does not have data oriented features. It has strong dashboarding features.

- **Dynatrace + Other solution** Dynatrace could be combined with a complementary solution (MLflow, Langfuse, or LangSmith) at least for the short term while some features are lacking in GA mode.

- **Managed MLflow (Databricks or SageMaker)** eliminates deployment and infrastructure management. The trade-off is vendor lock-in to the provider's feature release lifecycle and additional service charges (DBU-based
pricing, platform fees).

- **Self-hosted solutions (MLflow / Langfuse / LangSmith)** share similar ingestion overhead (all async, negligible app-side latency) and core features (tracing, evaluation, prompt management). Key differences:
  - MLflow offers ML-specific capabilities not found in the others: model registry, experiment tracking, and prompt optimization.
  - Langfuse and LangSmith rely on ClickHouse (columnar OLAP), providing faster analytical query performance at scale (e.g., computing p99 latency across 10M production traces). MLflow uses a SQL backend (PostgreSQL/MySQL),
which is slower for large-scale analytics but results in a lighter and cheaper infrastructure to maintain (no ClickHouse + Redis + S3 blob store to operate).
