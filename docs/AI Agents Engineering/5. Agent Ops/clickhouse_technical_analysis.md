# ClickHouse — Technical Analysis

**Date:** June 2026
**Context:** ClickHouse is the analytical database backend used by Langfuse (LLM observability platform) for traces, observations, and scores storage. This analysis evaluates ClickHouse's suitability, risks, and trade-offs for enterprise deployments.

---

## 1. What Is ClickHouse?

ClickHouse is an **open-source, column-oriented OLAP database** developed originally at Yandex (2009), open-sourced in 2016, and now maintained by ClickHouse Inc. (Linux Foundation Data & AI project since 2024).

**Core characteristics:**
- Column-oriented storage (reads only needed columns)
- Vectorized query execution (SIMD CPU instructions, processes data in batches)
- Aggressive data compression (LZ4, ZSTD — typically 5-10× compression ratios)
- MergeTree engine family (LSM-tree-inspired, append-only with background merges)
- Designed for **analytical aggregation queries** over billions of rows, not transactional workloads

**License:** Apache 2.0 (fully open source)

**Company:** ClickHouse Inc. — $400M Series D (2026), $250M+ ARR, 4,000+ customers. Acquired Langfuse in 2026.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                  ClickHouse Server                    │
├──────────┬──────────┬───────────┬───────────────────┤
│  Parser  │ Planner  │ Executor  │  Storage Engine    │
│  (SQL)   │ (Query)  │(Vectorized)│ (MergeTree)       │
└──────────┴──────────┴───────────┴───────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────────┐
│  Coordination   │          │   Local Filesystem   │
│  (Keeper/ZK)    │          │   (Columnar Parts)   │
└─────────────────┘          └─────────────────────┘
```

**Key architectural components:**
- **MergeTree Engine:** Data written as immutable "parts" (sorted by primary key). Background merges consolidate parts. This is the core — all production tables use ReplicatedMergeTree.
- **ClickHouse Keeper (or ZooKeeper):** Coordination service for replication metadata (not data). Tracks which parts exist, coordinates merges, handles leader election.
- **Distributed Tables:** Query fan-out across shards for horizontal scaling.

---

## 3. Dependencies & Operational Requirements

### Required Components (self-hosted)

| Component | Purpose | Minimum |
|-----------|---------|---------|
| ClickHouse Server | Query engine + storage | 1 node (dev), 3+ (prod) |
| ClickHouse Keeper (or ZooKeeper) | Replication coordination | 3 nodes (quorum) |
| Local NVMe/SSD storage | Data parts | High IOPS, low latency |
| Load Balancer (HAProxy/nginx) | Query distribution | 1 (prod) |

### Additional Dependencies for Langfuse

| Component | Purpose |
|-----------|---------|
| PostgreSQL | Transactional metadata (users, projects, configs) |
| Redis | Event queue (BullMQ) + caching (API keys, prompts) |
| S3 / Blob Storage | Raw ingestion events, multi-modal attachments |

**Total stack for Langfuse self-hosted:** ClickHouse cluster (3+ nodes) + Keeper (3 nodes) + PostgreSQL + Redis + S3 + Application containers = **minimum 8-10 processes/containers**.

### Operational Complexity Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Installation | ⚠️ Medium | Single node trivial; cluster setup requires Keeper config, shard/replica topology |
| Day-to-day operations | ⚠️ Medium-High | Requires monitoring part counts, merge queues, memory pressure, ZooKeeper health |
| Schema changes | ⚠️ Medium | ALTER is async (mutations queue); large table mutations can block merges for days |
| Upgrades | ⚠️ Medium | Rolling restarts possible but require replica coordination |
| Backup/Restore | ⚠️ Medium | Native backup to S3 available, but large clusters take hours |
| Kubernetes deployment | 🔴 High | IO-bound, merge-heavy workloads conflict with K8s disposable pod assumptions |

---

## 4. Scalability

### Proven Scale (Real-World)

| Company | Scale | Source |
|---------|-------|--------|
| Cloudflare | Quadrillion rows, hundreds of millions of rows/sec | [clickhouse.com/blog/cloudflare](https://clickhouse.com/blog/cloudflare) |
| ClickHouse (internal observability) | 100 PB uncompressed, 500 trillion rows | [clickhouse.com/blog/scaling-observability-beyond-100pb](https://clickhouse.com/blog/scaling-observability-beyond-100pb-wide-events-replacing-otel) |
| Replo | 100+ billion events, 3,000-5,000 events/sec ingestion | [clickhouse.com/blog/replo](https://clickhouse.com/blog/replo) |
| Langfuse | Migrated from PostgreSQL; "minutes → near real-time" query latency | [clickhouse.com/blog/langfuse-llm-analytics](https://clickhouse.com/blog/langfuse-llm-analytics) |

### Scaling Mechanisms

| Mechanism | How | Limit |
|-----------|-----|-------|
| Vertical scaling | More RAM, CPU, NVMe per node | Single node: billions of rows, TBs of compressed data |
| Horizontal sharding | Distribute data across nodes by shard key | Linear throughput scaling |
| Replication | ReplicatedMergeTree across replicas for HA | Keeper becomes bottleneck at very high replica count |
| Read replicas | Parallel replicas for query fan-out | Near-linear read scaling |

### Performance Benchmarks

- Sub-second queries over **billions of rows** on single node with proper schema design
- 10-100× throughput vs PostgreSQL/MySQL for aggregation queries
- ClickHouse vs Elasticsearch (billion-row benchmark): ClickHouse "vastly outperforms" for count(*) aggregations
- ClickHouse vs Databricks/Snowflake: "faster and cheaper at every scale, from 721M to 7.2B rows" (join benchmark)
- Production target: **100K QPS with sub-second latency** on 3 data nodes + 2 Keeper nodes + load balancer

**Sources:** [clickhouse.com/benchmarks](https://clickhouse.com/benchmarks), [clickhouse.com/blog/join-me-if-you-can](https://clickhouse.com/blog/join-me-if-you-can-clickhouse-vs-databricks-snowflake-join-performance)

---

## 5. Limitations & Weaknesses

### Fundamental Design Trade-offs

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **No ACID transactions** | Cannot guarantee consistency for concurrent writes to same rows. No rollback. | Design for append-only patterns. Use ReplacingMergeTree for eventual deduplication |
| **UPDATE/DELETE historically weak** | Mutations are async, rewrite entire parts. Can queue for days on large tables | New lightweight updates (2025-2026) with "patch parts" — 2,400× faster. Still not OLTP-grade |
| **JOINs are slow** | Rule-based planner (not cost-based), no data shuffling, memory limits | Denormalize, use dictionaries (6.6× faster than joins), pre-aggregate |
| **No full-text search** | No BM25, fuzzy matching, relevance scoring | Use ClickHouse for analytics, Elasticsearch for search. Or use inverted indexes (experimental) |
| **High-frequency small inserts** | Each insert creates a new part; too many parts causes "too many parts" errors | Batch inserts (10K+ rows minimum). Use Buffer tables or async insert mode |
| **Memory pressure** | Aggregations can consume all RAM → OOM killer invokes | Set `max_memory_usage`, monitor carefully, size nodes with headroom |
| **Schema rigidity** | Columnar format makes frequent schema changes expensive (mutations) | Plan schema upfront. Use JSON columns for flexible fields |

### Operational Risks (Real-World Incidents)

| Incident | What Happened | Root Cause | Source |
|----------|---------------|------------|--------|
| Cloudflare outage (2025) | ChatGPT knocked offline for hours, 20% global traffic affected | ClickHouse query produced duplicate data → Bot Management cascade | [cybersecurefox.com](https://cybersecurefox.com/en/cloudflare-outage-clickhouse-bot-management-configuration-failure/) |
| Trigger.dev (Nov-Dec 2025) | Intermittent OTel ingestion failures for 3 days | "Too many parts" error — ingestion outpaced merges | [trigger.dev/blog/clickhouse-too-many-parts-postmortem](https://trigger.dev/blog/clickhouse-too-many-parts-postmortem) |
| PostHog (Sep 2024) | ClickHouse cluster OOM issues | Memory pressure from concurrent queries exceeding allocated RAM | [isdown.app/status/posthog-us](https://isdown.app/status/posthog-us/incidents/319398-clickhouse-memory-issues) |
| Generic pattern | Background merges stall, parts explode to 85,000, query performance degrades | Queued mutations (7 over 14 days) each rewriting 40% of 2TB table | [medium.com/@rakesh.therani](https://medium.com/@rakesh.therani/the-70-of-clickhouse-metrics-youre-not-monitoring-a-deep-dive-into-hidden-operational-risks-c313463ae6d1) |

**Common failure modes:**
1. **Part explosion** — too many small inserts without batching
2. **OOM kills** — queries or mutations consuming all memory
3. **Keeper/ZooKeeper failures** — replication halts, writes may block
4. **Mutation backlogs** — ALTER TABLE UPDATE/DELETE queues growing indefinitely on large tables

---

## 6. Security Assessment

### Known CVEs (2024-2025)

| CVE | Severity | Description | Fixed In |
|-----|----------|-------------|----------|
| CVE-2025-1385 | Critical | Failed input validation in clickhouse-library-bridge API → RCE | Patched |
| CVE-2024-6873 | High | Heap buffer overflow in native interface → DoS/execution redirect (unauthenticated) | Patched |
| CVE-2024-41436 | High | Buffer overflow in evaluateConstantExpressionImpl | Patched |
| CVE-2024-23689 | Medium | Client certificate password exposure in exception logs | Patched |

**Sources:** [clickhouse.com/docs/whats-new/security-changelog](https://clickhouse.com/docs/whats-new/security-changelog), [nvd.nist.gov](https://nvd.nist.gov/vuln/detail/cve-2024-6873), [github.com/ClickHouse/ClickHouse/security](https://github.com/ClickHouse/ClickHouse/security/advisories/GHSA-5phv-x8x4-83x5)

### Security Features

| Feature | Status |
|---------|--------|
| Authentication | Username/password, LDAP, Kerberos, X.509 certificates |
| Authorization | RBAC (roles, row-level policies, column-level access) |
| Encryption at rest | Not native — relies on filesystem/disk encryption (LUKS, cloud KMS) |
| Encryption in transit | TLS for client connections and inter-node replication |
| Network isolation | Configurable listen addresses, IP allowlists |
| Audit logging | Query log tables (system.query_log, system.text_log) |

### Security Assessment Summary

- **Active CVE history** — several critical/high CVEs in 2024-2025 including one unauthenticated RCE path. ClickHouse responds with patches but the attack surface is broad (native protocol, HTTP, library-bridge).
- **No native encryption at rest** — you must rely on disk-level or cloud-provider encryption. This is a gap for compliance-sensitive environments.
- **Rapidly improving** — RBAC has matured significantly. Row-level security, quotas, and profiles are production-ready.

---

## 7. ClickHouse vs Mainstream Alternatives

### For Observability/Analytics (Langfuse Use Case)

| Criteria | ClickHouse | PostgreSQL | Elasticsearch | TimescaleDB |
|----------|-----------|------------|---------------|-------------|
| **Query speed (aggregations)** | ⭐⭐⭐⭐⭐ Sub-second on billions of rows | ⭐⭐ Minutes on 100M+ rows | ⭐⭐⭐ Fast for text, slow for aggregations | ⭐⭐⭐ Good for time-series, limited for general OLAP |
| **Ingestion throughput** | ⭐⭐⭐⭐⭐ Millions of rows/sec (batched) | ⭐⭐ Thousands of rows/sec | ⭐⭐⭐⭐ High (bulk indexing) | ⭐⭐⭐ Good |
| **Compression** | ⭐⭐⭐⭐⭐ 5-10× typical (columnar) | ⭐⭐ TOAST + pg_compress (2-3×) | ⭐⭐⭐ Moderate (inverted index overhead) | ⭐⭐⭐ Good (columnar chunks) |
| **Operational simplicity** | ⭐⭐ Complex cluster management | ⭐⭐⭐⭐⭐ Battle-tested, simple | ⭐⭐ JVM tuning, cluster management | ⭐⭐⭐⭐ PostgreSQL extension |
| **ACID/Transactions** | ❌ No | ✅ Full | ❌ Eventually consistent | ✅ Full (PostgreSQL) |
| **UPDATE/DELETE** | ⚠️ Improved but not OLTP-grade | ✅ Native | ⚠️ Immutable segments + merge | ✅ Native |
| **Full-text search** | ❌ Limited | ⚠️ Basic (tsvector) | ✅ Excellent | ⚠️ Basic |
| **Cost at scale** | ⭐⭐⭐⭐⭐ Low (compression + commodity HW) | ⭐⭐ Expensive at TB+ scale | ⭐⭐ Expensive (RAM-heavy, JVM) | ⭐⭐⭐ Moderate |
| **Ecosystem maturity** | ⭐⭐⭐ Growing rapidly (4,000 customers) | ⭐⭐⭐⭐⭐ 30+ years, universal | ⭐⭐⭐⭐ Very mature for search/logs | ⭐⭐⭐ Niche |

### Why Langfuse Chose ClickHouse Over PostgreSQL

Langfuse's migration rationale (from their engineering blog + ClickHouse case study):

1. **Query latency:** PostgreSQL dashboard queries took **minutes** at scale → ClickHouse delivers **sub-second**
2. **Ingestion volume:** LLM observability generates massive trace volumes that PostgreSQL couldn't keep up with
3. **Compression:** Traces are repetitive (same schema, similar content) → columnar compression reduces storage 5-10×
4. **Analytical workload pattern:** Langfuse queries are aggregations (count, avg, percentiles) over time ranges — exactly what ClickHouse optimizes for

**Source:** [clickhouse.com/blog/langfuse-llm-analytics](https://clickhouse.com/blog/langfuse-llm-analytics)

---

## 8. ClickHouse Cloud (Managed) vs Self-Hosted

| Aspect | Self-Hosted | ClickHouse Cloud |
|--------|-------------|------------------|
| **Pricing** | Infrastructure cost only | From $66/mo (Basic) to $499/mo+ (Scale). Compute: $0.22-0.75/unit-hour. Storage: ~$25/TB-month |
| **Operational burden** | High — Keeper cluster, backups, upgrades, monitoring, scaling | Zero — fully managed |
| **Availability** | User responsibility (3+ replicas recommended) | SLA-backed (99.95%+) |
| **Scaling** | Manual shard/replica management | Auto-scaling (compute scales to zero when idle) |
| **Security** | User responsibility | SOC2 Type II, ISO 27001, encryption at rest managed |
| **Regions** | Anywhere | AWS, GCP, Azure (major regions) |

**Source:** [clickhouse.com/pricing](https://clickhouse.com/pricing), [clickhouse.com/docs/cloud/manage/billing](https://clickhouse.com/docs/cloud/manage/billing/overview)

---

## 9. Critical Assessment & Risks for Enterprise Adoption

### Strengths

1. **Unmatched analytical performance** — no other open-source database comes close for aggregation queries at scale
2. **Cost efficiency** — compression + commodity hardware = significantly lower TCO than Elasticsearch or cloud DWH
3. **Active development** — rapid release cadence, strong investment ($400M), growing ecosystem
4. **Proven at hyperscale** — Cloudflare, Uber, eBay, Spotify, Deutsche Bank use it in production
5. **Good fit for observability** — trace data is append-heavy, query-heavy, time-bounded — perfect for ClickHouse

### Risks & Concerns

1. **Operational complexity** — not a "set and forget" database. Requires ClickHouse-specific expertise (part management, merge tuning, Keeper health). Failure modes are non-obvious (part explosion, mutation backlogs).

2. **Security posture** — multiple critical CVEs in recent years including unauthenticated RCE paths. No native encryption at rest. Active attack surface on native protocol. Requires defense-in-depth (network isolation, frequent patching).

3. **Vendor concentration risk** — ClickHouse Inc. acquired Langfuse (2026). If you self-host Langfuse with ClickHouse, your entire LLM observability stack depends on one company's ecosystem. Consider: what if ClickHouse Inc. changes licensing, pricing, or strategic direction?

4. **Not ACID** — if your use case ever requires transactional guarantees on trace data (regulatory, audit trail integrity), ClickHouse cannot provide them. Eventual consistency is acceptable for observability but may not be for compliance.

5. **Kubernetes friction** — ClickHouse's IO-bound, stateful, merge-heavy nature conflicts with K8s ephemeral pod assumptions. Production K8s deployments require significant tuning (dedicated nodes, local storage, anti-affinity rules).

6. **Talent scarcity** — fewer engineers have ClickHouse production experience compared to PostgreSQL/MySQL. Hiring and onboarding is a factor.

7. **Dependency chain** — Langfuse self-hosted requires ClickHouse + PostgreSQL + Redis + S3. That's 4 stateful services to operate. Compare with solutions that use only PostgreSQL (simpler but slower at scale).

### Recommendation

**ClickHouse is the right choice for Langfuse's use case** (high-volume trace analytics), but enterprises should:

- **Prefer ClickHouse Cloud** over self-hosting unless compliance/data residency mandates self-hosting
- **Patch aggressively** — subscribe to security changelog, apply updates within days of critical CVEs
- **Monitor proactively** — part counts, merge queue depth, memory usage, Keeper latency are critical metrics
- **Plan for operational expertise** — budget for ClickHouse-specific training or Altinity/ClickHouse support contracts
- **Maintain exit strategy** — ensure data is exportable (blob storage exports, API access) in case of ecosystem changes

---

## 10. References

| Source | URL |
|--------|-----|
| ClickHouse Architecture Overview | https://clickhouse.com/docs/en/development/architecture/ |
| ClickHouse Security Changelog | https://clickhouse.com/docs/whats-new/security-changelog |
| CVE-2024-6873 (DoS/RCE) | https://nvd.nist.gov/vuln/detail/cve-2024-6873 |
| CVE-2025-1385 (RCE via library-bridge) | https://github.com/ClickHouse/ClickHouse/security/advisories/GHSA-5phv-x8x4-83x5 |
| Langfuse + ClickHouse Case Study | https://clickhouse.com/blog/langfuse-llm-analytics |
| Cloudflare Quadrillion-Row Scale | https://clickhouse.com/blog/cloudflare |
| Cloudflare Outage (ClickHouse root cause) | https://cybersecurefox.com/en/cloudflare-outage-clickhouse-bot-management-configuration-failure/ |
| Trigger.dev "Too Many Parts" Post-Mortem | https://trigger.dev/blog/clickhouse-too-many-parts-postmortem |
| PostHog ClickHouse OOM Incident | https://isdown.app/status/posthog-us/incidents/319398-clickhouse-memory-issues |
| ClickHouse Benchmarks | https://clickhouse.com/benchmarks |
| ClickHouse vs Databricks/Snowflake | https://clickhouse.com/blog/join-me-if-you-can-clickhouse-vs-databricks-snowflake-join-performance |
| ClickHouse vs Elasticsearch (Billion Row) | https://clickhouse.com/blog/clickhouse_vs_elasticsearch_the_billion_row_matchup |
| When Not to Use ClickHouse | https://chistadata.com/when-not-to-use-clickhouse/ |
| ClickHouse JOINs Limitations | https://www.glassflow.dev/blog/clickhouse-limitations-joins |
| ClickHouse Keeper Explained | https://oneuptime.com/blog/post/2026-03-31-clickhouse-keeper-explained/view |
| Scaling to Billions of Rows | https://oneuptime.com/blog/post/2026-03-31-clickhouse-scale-billions-rows/view |
| ClickHouse Pricing | https://clickhouse.com/pricing |
| ClickHouse Series D + Langfuse Acquisition | https://clickhouse.com/blog/clickhouse-raises-400-million-series-d-acquires-langfuse-launches-postgres |
| Langfuse Architecture Handbook | https://langfuse.com/handbook/product-engineering/architecture |
| ClickHouse Production Architecture | https://markaicode.com/architecture/clickhouse-production-system-design-architecture/ |
| ClickHouse Kubernetes Pitfalls | https://pulse.support/kb/clickhouse-kubernetes-operator |
| ClickHouse ACID Limitations | https://www.singlestore.com/blog/the-acid-dilemma-clickhouse-falls-short-singlestore-delivers |
| ClickHouse UPDATE Support (2026) | https://dataanalyticsguide.substack.com/p/clickhouse-update-support |
