# PROPOSAL: Migrate MCP transport from SSE → Streamable HTTP

## Problem

progress-rag MCP server at `http://localhost:8765/mcp` implements **SSE transport**
(GET handshake → session_id → POST with `?session_id=xxx`).

OpenCode's Oh My Pi MCP client configured with `"type": "http"` speaks
**Streamable HTTP** — the newer MCP transport. It sends a single POST with
`Content-Type: application/json` and expects a direct JSON-RPC response.
No session handshake.

Result: `400: session_id is required`.

Current workaround: `"type": "sse"` in `.omp/mcp.json`. Works, but the SDK
itself recommends `"type": "http"` for new configs ("Legacy SSE transport kept
for compatibility. Prefer http for new configs.").

## Solution

Replace `SseServerTransport` + `_MCPSSEApp` with `StreamableHTTPSessionManager`
from the same MCP SDK (v1.28.1 already installed at
`~/.local/lib/python3.10/site-packages/mcp/server/streamable_http_manager.py`).

## Changes

### File: `~/Work/graphrag/src/http_api.py`

| Change | What | Why |
|--------|------|-----|
| 1. Import | Add `from mcp.server.streamable_http_manager import StreamableHTTPSessionManager` | SDK class for Streamable HTTP |
| 2. New class | `_MCPStreamableHTTPApp` — forwards ASGI scope → `session_manager.handle_request()` | Thin ASGI wrapper instead of `_MCPSSEApp` |
| 3. `_mount_mcp` | Create `StreamableHTTPSessionManager(app=server, json_response=True, stateless=True)`, mount it, return it | Stateless = no session tracking, each POST standalone |
| 4. `lifespan` | Wrap `async with manager.run():` around the yield | Manager needs a task group; lifespan is the natural owner |
| 5. Remove | Delete `_MCPSSEApp` class, `SseServerTransport` import | Dead code |

### File: `~/Work/.omp/mcp.json`

| Change | What | Why |
|--------|------|-----|
| 6. Type | `"sse"` → `"http"` | Match the new transport |

### File: `~/Work/graphrag/src/mcp_server.py`

No changes needed. `_make_mcp_server()` already returns a `Server` instance
that both SSE and Streamable HTTP transports accept identically.

## Transport lifecycle (stateless mode)

```
OpenCode                          FastAPI (port 8765)
   │                                    │
   │  POST /mcp                          │
   │  Content-Type: application/json     │
   │  {"jsonrpc":"2.0","id":1,           │
   │   "method":"tools/list","params":{}}│
   │ ──────────────────────────────────► │
   │                                    ├─ _MCPStreamableHTTPApp.__call__
   │                                    ├─ StreamableHTTPSessionManager
   │                                    │  .handle_request(scope, receive, send)
   │                                    ├─ StreamableHTTPServerTransport
   │                                    │  (created per-request, mcp_session_id=None)
   │                                    ├─ _task_group.start(run_stateless_server)
   │                                    │     └─ http_transport.connect()
   │                                    │     └─ app.run(stateless=True)
   │                                    ├─ http_transport.handle_request()
   │                                    │  → parses JSON-RPC
   │                                    │  → forwards to Server.call_tool
   │                                    │  → JSON-RPC response
   │                                    ├─ http_transport.terminate()
   │                                    │
   │  HTTP 200                           │
   │  Content-Type: application/json     │
   │  {"jsonrpc":"2.0","id":1,           │
   │   "result":{"tools":[...]}}         │
   │ ◄────────────────────────────────── │
```

## Edge case: Accept header

`StreamableHTTPServerTransport._validate_accept_header` checks that the client
sends `Accept: application/json` (when `json_response=True`). If OpenCode omits
the Accept header, `has_json` is `False` → 406 error.

Quick fix in `_handle_stateless_request` if needed: skip validation or treat
absent Accept as `*/*`. The SDK code is at
`streamable_http.py:449-469` — we can either patch the SDK or handle
it in our wrapper.

## Rollback

Revert `.omp/mcp.json` to `"type": "sse"`. The SSE handler code is deleted,
but the `StreamableHTTPSessionManager` is backward-compatible with SSE clients
only if `json_response=False` (enables SSE streaming). For pure rollback,
restore `SseServerTransport` from git.


# PROPOSAL: Semantic Reasoning Layer over the Knowledge Graph

## Problem

Current RAG answers questions by searching for semantically similar chunks.
This fails on questions that require **logical traversal of relationships**
rather than semantic proximity:

- «Кто вызывает auth-service?» — needs graph walk, not text search
- «Что сломается если убрать этот метод?» — needs transitive closure
- «Этот метод переопределён где-то?» — needs inference across IMPLEMENTS + EXTENDS

The graph store already has raw edges (`IMPLEMENTS`, `EXTENDS`, `CALLS`,
`DEPENDS_ON`), but there is no way to derive **new facts** from existing ones
or to verify them against code.

Hybrid search + graph BFS covers adjacency, but not **transitive, compositional,
or conditional relationships** — the kind that require `JOIN`-like reasoning:

```
IMPLEMENTS(Child, M) ∧ EXTENDS(Child, Parent) ∧ IMPLEMENTS(Parent, M)
    → OVERRIDES(Child, M)
```

## Solution

Add a **semantic reasoning layer** with three components:

1. **Ontology** — declarative, fixed, extensible definition of entity types,
   relation types, and their allowed domains/ranges
2. **Inference engine** — Datalog-like rules that derive new edges from
   existing graph facts
3. **LLM co-pilot** — proposes new relation types and rules when the ontology
   falls short; verifies inferred edges against source code

### Architecture

```
                         ┌──────────────────────────────┐
                         │       Ontology (YAML)          │
                         │  entity types + relation       │
                         │  signatures + rules            │
                         └──────────┬───────────────────┘
                                    │ loads
┌──────────────┐     ┌──────────────▼───────────┐     ┌─────────────────┐
│  Graph Store │────▶│     Inference Engine       │────▶│  Enriched Graph  │
│  (raw edges) │     │  forward-chaining Datalog  │     │  (raw+inferred)  │
└──────────────┘     │  on NetworkX + Python      │     └────────┬────────┘
                     └──────────────┬───────────┘              │
                                    │                          │
                     ┌──────────────▼───────────┐              │
                     │     LLM Co-pilot           │◀─────────────┘
                     │  - propose relation types │
                     │  - propose rules          │
                     │  - verify inferred edges  │
                     │  - resolve conflicts      │
                     └──────────────────────────┘
```

### Layer 1: Ontology (declarative, YAML)

```yaml
# src/rag_data/ontology.yaml
entity_types:
  Class:
    description: "OOP class or interface"
  Method:
    description: "Class method, interface method, or standalone function"
  Service:
    description: "Microservice / deployable unit"
  Endpoint:
    description: "HTTP/gRPC endpoint exposed by a service"
  Module:
    description: "Package, namespace, or directory"

relations:
  IMPLEMENTS:
    domain: [Class, Interface]
    range: [Method]
    transitive: false

  EXTENDS:
    domain: [Class]
    range: [Class]
    transitive: true

  CALLS:
    domain: [Method, Service]
    range: [Method, Endpoint]
    transitive: false

  DEPENDS_ON:
    domain: [Module, Service]
    range: [Module, Service]
    transitive: true

# Inferred relations (not present in raw graph, derived by rules)
inferred_relations:
  OVERRIDES:
    description: "Method overrides a parent class method"
  TRANSITIVE_DEPENDS:
    description: "Transitive closure of DEPENDS_ON"
  SERVICE_CALLS:
    description: "Cross-service call derived from endpoint exposure"
  DIAMOND_DEPENDS:
    description: "Module depends on two modules that share a common dependency"
```

### Layer 2: Inference Engine (Python + NetworkX, forward-chaining)

Rules are Python functions decorated with `@rule`. The engine applies them
in fixed-point iteration until no new edges are produced:

```python
from graphrag.reasoner import rule, Inferrer

@rule(produces="OVERRIDES")
def infer_overrides(g: nx.DiGraph) -> list[tuple[str, str, str]]:
    """OVERRIDES(Child, M) :-
        IMPLEMENTS(Child, M),
        EXTENDS(Child, Parent),
        IMPLEMENTS(Parent, M)."""
    new = []
    for child, parent, _ in edges_of_type(g, "EXTENDS"):
        child_methods = {m for _, m, _ in edges_of_type(g, "IMPLEMENTS")
                         if _ == child}
        parent_methods = {m for _, m, _ in edges_of_type(g, "IMPLEMENTS")
                          if _ == parent}
        for m in child_methods & parent_methods:
            new.append((child, m, "OVERRIDES"))
    return new

@rule(produces="TRANSITIVE_DEPENDS")
def infer_transitive(g: nx.DiGraph) -> list[tuple[str, str, str]]:
    """Transitive closure of DEPENDS_ON."""
    sub = g.edge_subgraph(
        (u, v) for u, v, d in g.edges(data=True)
        if d.get("relation") == "DEPENDS_ON"
    )
    closure = nx.transitive_closure(sub)
    existing = set((u, v) for u, v, _ in edges_of_type(g, "DEPENDS_ON"))
    return [
        (u, v, "TRANSITIVE_DEPENDS")
        for u, v in closure.edges()
        if (u, v) not in existing
    ]
```

**Engine guarantees:**
- Fixed-point semantics: rules fire until no new edges
- Cycle-safe: rules with cycles are detected and skipped
- Incremental: only changed facts re-trigger dependent rules (Rete-like)
- Relation metadata (weight, provenance) preserved

### Layer 3: LLM Co-pilot

Three operations, all opt-in via MCP tools:

#### 3a. Propose relation types
When the graph contains patterns the ontology doesn't capture, the LLM
proposes new `relation` entries. Example trigger: the graph has
`ISSUES_EVENT(Kafka, topic)` edges but no `PUBLISHES` relation defined.

#### 3b. Propose inference rules
Given a new relation type and examples of related facts, the LLM proposes
Datalog-like rules.

#### 3c. Verify inferred edges
When an inference produces an edge with low confidence (e.g., ambiguous
names, conflicting rules), the LLM inspects source code to verify:

```
Inferred: OVERRIDES(UserService.authenticate, AuthService.authenticate)
Evidence: IMPLEMENTS + EXTENDS chain
Confidence: 0.6 (class names don't match exactly)

LLM: Check source code → "Confirms: UserService extends BasicAuth,
      both implement authenticate() → OVERRIDES confirmed."
```

### New MCP Tools

| Tool | Signature | Description |
|------|-----------|-------------|
| `rag_infer` | `rules: list[str] \| null` | Run inference engine. No args → all rules. Returns `{inferred: N, edges: [...]}` |
| `rag_query_rule` | `query: str` | Ad-hoc Datalog query over the fact base. Example: `"SERVICE_CALLS(X, Y)"` |
| `rag_get_ontology` | — | Return current ontology (entity types, relations, rules) |
| `rag_propose_relations` | — | LLM scans graph for undefined edge types, proposes additions |
| `rag_propose_rules` | `relation: str` | LLM proposes inference rules for a given relation |
| `rag_verify_edge` | `source: str, target: str, relation: str` | LLM verifies an inferred edge against source code |
| `rag_accept_ontology_patch` | `patch: object` | Apply ontology/rules changes proposed by LLM (staged) |

### Files

| File | What |
|------|------|
| `src/ontology.py` | Ontology loader/validator from YAML |
| `src/reasoner.py` | Inference engine: rule registry, fixed-point executor, NetworkX integration |
| `src/ontology_extender.py` | LLM co-pilot: propose relations, propose rules, verify edges |
| `src/rag_data/ontology.yaml` | Default ontology definition |
| `src/rag_data/rules/` | Per-rule Python modules (auto-discovered) |

### Integration with Graph Store

Inferred edges are stored alongside raw edges in the graph store (`_inferred`
metadata flag). Queries can:
- Include inferred edges: `rag_get_related(node_id, include_inferred=True)` (default)
- Exclude them: `rag_get_related(node_id, include_inferred=False)`
- Filter by provenance: `rag_get_related(node_id, provenance="OVERRIDES")`

Invalidation: when raw edges are added/deleted, the inference cache for
affected rules is invalidated. `rag_infer` re-runs incrementally.

### Example Scenario

**Question:** «Какие методы переопределены в UserService?»

1. `rag_get_related("UserService", relation="EXTENDS")` → `["BasicAuth"]`
2. `rag_infer(rules=["OVERRIDES"])` → `[(UserService.authenticate, OVERRIDES), (UserService.authorize, OVERRIDES)]`
3. `rag_query_rule("OVERRIDES(UserService, M)")` → список методов
4. Ответ: «authenticate, authorize — оба переопределяют методы из BasicAuth»

Without reasoning: RAG поискал бы «UserService методы» и вернул общий список всех методов класса.

### Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Rules produce false positives | LLM verification step; confidence threshold |
| Rule explosion (combinatorial) | Fixed-point with max iterations; incremental Rete-like network |
| Ontology too rigid | LLM-driven extension proposals; user review before apply |
| Performance on large graphs | Limit inference depth; cache transitive closures; souffle migration path |

### Migration Path to Souffle

The Python forward-chaining prototype is a stepping stone. If the rule set
grows beyond ~20 rules or the graph exceeds ~100K edges, migrate to
[Souffle](https://souffle-lang.github.io/) — a Datalog engine compiled to C++:

```prolog
// rules.dl — same semantics, 100x faster
.decl OVERRIDES(child: symbol, method: symbol)
OVERRIDES(C, M) :-
    IMPLEMENTS(C, M),
    EXTENDS(C, P),
    IMPLEMENTS(P, M).
```

Souffle runs as a subprocess; the Python `reasoner.py` shell becomes a thin
wrapper that serializes facts to CSV, invokes `souffle`, and loads results
back into NetworkX.


# PROPOSAL: Confidence Decay Model for Knowledge Freshness

## Problem

Not all documents in RAG are equally trustworthy. Some are actively maintained,
frequently consulted cornerstones of the system. Others are stale, deprecated,
or describe code that no longer exists. Yet the current system treats them as
equal — a chunk from a document last updated six months ago ranks the same as
one verified yesterday.

We need a **freshness signal** that:
- Decays naturally over time
- Slows its decay when a document is actively read (sign of continued relevance)
- Resets when a document is updated (sign of active maintenance)
- Never deletes anything — just annotates low-confidence content
- Lets the consuming agent decide: trust, or verify from source

This is not cache eviction. Nothing is ever removed. Confidence is metadata
that travels with every search result and graph traversal, giving the LLM
consumer a signal: «this answer is based on well-maintained knowledge» vs
«this is from a rarely-visited corner of the docs — double-check it».

## Related Work

**Google Knowledge Vault (2014):** every extracted fact carries a confidence
score derived from source reliability, extraction freshness, and corroboration
count. Facts never disappear — their confidence drifts.

**Wikidata:** every statement has `rank` (preferred / normal / deprecated).
Deprecated statements persist indefinitely with a «no longer current» marker.

**Adaptive TTL (RFC 8767):** DNS resolvers may extend a record's lifetime
beyond its TTL if the authoritative server is unreachable, relying on the
record's historical stability as a trust proxy.

**Truth Discovery (Li et al., 2016):** iterative models that jointly estimate
source reliability and fact confidence, with recency-weighted evidence.

**LRU cache eviction (inverted):** instead of evicting cold items, we keep
them but lower their confidence. Frequent access acts as a soft refresh —
the opposite of LRU's «unused → evict».

Our model differs from all of these in one key aspect: **decay slows with
use, not just with recency of update**. A document that is read daily decays
slower than one that was updated last week but never consulted. This turns
read traffic into a first-class freshness signal.

## Solution

### Core Model

Every node and edge in the knowledge graph carries a `Confidence` record:

```python
@dataclass
class Confidence:
    value: float            # [0.0, 1.0]
    last_access: float      # unix timestamp
    last_update: float      # unix timestamp
    access_count: int       # total explicit reads
    half_life_days: float   # configurable per document type
```

### Decay Function

Exponential decay with access-based dampening:

```
decay(conf, now) = conf.value * 0.5 ^ (age_days / effective_half_life)

where:
    age_days           = (now - conf.last_update) / 86400
    access_rate        = conf.access_count / max(age_days, 1)
    effective_half_life = conf.half_life_days * (1 + access_rate * DAMPENING_FACTOR)
```

- `half_life_days` — base half-life. Default: 30 days for code docs, 90 for ADRs, 180 for concept docs.
- `DAMPENING_FACTOR` — how strongly frequent access slows decay. Default: 7 (so daily access roughly doubles the half-life).
- Decay is computed **lazily on read**, not via a background job. Thread-safe via atomic timestamp updates.

**Why exponential decay with access dampening:**
- Exponential models real-world knowledge staleness better than linear — a document is most likely to go stale shortly after its last update, then stabilizes
- Access dampening reflects the «wisdom of the crowd»: if many agents consult this document and nobody edits it, it's probably still correct
- No background sweep needed — confidence is always current when read

### Access Counting

| Event | Effect on confidence |
|---|---|
| `rag_get_document(doc_id)` — explicit full read | `access_count += 1`, `last_access = now` |
| `rag_get_related(node_id)` — graph traversal | `access_count += 1` for the center node, `+= 0.5` for each direct neighbor |
| `rag_search*` — search hit | `access_count += 0.01` per hit (1/100th of a read). BM25/semantics found it → weak validity signal |
| `rag_add_document` / `rag_add_relation` — creation | `value = 1.0`, `last_update = now`, `last_access = now` |
| `rag_update_document` — content change | `value = 1.0`, `last_update = now`, `access_count` preserved |
| `rag_update_document` — metadata-only change | `value += 0.3` (capped at 1.0), `last_update = now` |
| `rag_add_relation` — new edge | Edge `value = 1.0`. Node `value` unchanged |
| `rag_delete_document` / `rag_delete_relation` — removal | Record removed (no confidence to track) |

**Why search hits count as 1/100:** a search match means the document is
semantically or lexically relevant to a real query. This is a weak but
genuine signal that the content is still «in play». The 1/100 ratio
prevents popular but shallow documents from dominating: a document that
appears in 100 search results but is never opened shouldn't outrank a
document read 5 times explicitly.

**Noise floor:** `access_count` saturates at `SATURATION_LIMIT` (default: 1000)
to prevent ancient documents with massive historical traffic from never
decaying.

### Confidence Tiers

For human and LLM consumption, raw `value` is mapped to a tier:

| Tier | Range | Label | Meaning |
|------|-------|-------|---------|
| `fresh` | ≥ 0.8 | «актуально» | Recently updated or heavily consulted. Trust directly. |
| `stable` | 0.5–0.8 | «стабильно» | No recent changes but regularly read. Likely correct. |
| `decaying` | 0.2–0.5 | «устаревает» | Not updated or read in a while. Cross-check with source. |
| `stale` | < 0.2 | «возможно неактуально» | Abandoned corner of the docs. Verify from code before relying. |

### Isolation Model (v1)

Confidence is **per-node, per-edge, independent**. A node's confidence does
not propagate to its neighbors in v1. Rationale:

- Simpler reasoning: a document is stale, not its entire subgraph
- Prevents cascading decay: one abandoned doc doesn't drag down everything
  that links to it
- The consuming LLM can combine signals itself: «this answer comes from a
  stable doc that references a stale doc → I should verify the stale part»

**Future (v2): confidence diffusion.** A node linked from many high-confidence
nodes gets a partial decay slowdown. An edge between two fresh nodes decays
slower than an edge to a stale node. Requires solving the feedback loop
problem (read traffic on high-confidence nodes shouldn't inflate their
neighbors indefinitely).

### Stability Tiers (Metaplasticity)

The base decay model assumes all documents decay at the same rate.
This fails for documents that are **structurally important but rarely read**
— ADRs, architecture overviews, core interface definitions. These change
less often by definition, so they're consulted less often, yet they're the
most trustworthy content in the system. Without correction, they decay
faster than a transient proposal that everyone reads for a week then forgets.

**Metaplasticity** — borrowed from neuroscience — solves this: a synapse
that has been strong for a long time becomes structurally resistant to
weakening. Even if signal temporarily stops, the connection degrades slowly.

Applied to documents:
- A document that **held high confidence for an extended period** earns
  a lower decay rate — it has proven its stability
- A new document decays at the standard rate until it earns stability
- A human or LLM can explicitly mark a document as structurally important,
  bypassing the waiting period

#### Colour Tiers

Every document carries a `colour` field — a stability tier that acts as a
multiplier on the effective half-life:

| Colour | Multiplier | Meaning | How earned |
|---|---|---|---|
| `blue` | ×1.0 | New document, standard decay | Default for all new documents |
| `yellow` | ×2.0 | Established: proven stability | 30 consecutive days with confidence > 0.5 |
| `green` | ×5.0 | Core: structural knowledge | 60 consecutive days with confidence > 0.8, OR manual assignment |

#### Transitions

```
                    ┌──────────────────────┐
                    │        blue          │
                    │   (standard decay)   │
                    └──────────┬───────────┘
                               │ 30 days confidence > 0.5
                               ▼
                    ┌──────────────────────┐
                    │       yellow         │
                    │   (2× slower decay)  │
                    └──────────┬───────────┘
                               │ 60 days confidence > 0.8
                               ▼
                    ┌──────────────────────┐
                    │        green         │
                    │   (5× slower decay)  │
                    └──────────────────────┘

  Demotions (fast downward adjustments):

  green  → yellow  when confidence drops below 0.3
  yellow → blue    when confidence drops below 0.15
```

Promotion is **slow** (30/60 days of sustained confidence). Demotion is
**fast** (value drops below threshold → immediate downgrade). This
asymmetry mirrors biology: building structural stability takes time;
losing it happens quickly when the underlying content is genuinely stale.

#### Manual Override

```python
rag_set_color(doc_id="auth-service/ADR-003", color="green")
rag_set_color(doc_id="auth-service/ADR-003", color="auto")  # return to automatic
```

Use cases:
- **LLM during indexing:** detects «this is an ADR» → marks `green`
- **Human curator:** marks known-trustworthy documents
- **System analyst cron:** promotes documents that survived multiple
  verification cycles

#### Updated Decay Function

```python
def effective_half_life(conf: Confidence) -> float:
    base = conf.half_life_days
    access_bonus = 1 + min(conf.access_count, SATURATION_LIMIT) / max(age_days, 1) * DAMPENING_FACTOR
    color_mult = {"blue": 1.0, "yellow": 2.0, "green": 5.0}[conf.color]
    return base * access_bonus * color_mult
```

#### Effect on ADR Scenario

| Scenario | Without colour | With green ADR |
|---|---|---|
| Half-life | 90 days | 90 × 5.0 = 450 days |
| No reads for 6 months | confidence → 0.25 (decaying) | confidence → 0.75 (stable) |
| No reads for 2 years | confidence → 0.004 (stale) | confidence → 0.32 (decaying) |

A green ADR survives two years without attention before reaching the
decaying tier — long enough that a system analyst or repo sync will
have refreshed it.

#### Confidence Response Includes Colour

```
{
  "doc_id": "auth-service/ADR-003",
  "confidence": {
    "value": 0.94,
    "tier": "fresh",
    "colour": "green",
    "trend": "stable"
  }
}
```

The LLM consumer sees both the point-in-time confidence and the structural
stability tier — «this document is trustworthy now AND has been trustworthy
for a long time.»

#### Integration with Colours

| Document type | Recommended colour | Rationale |
|---|---|---|
| ADR | `green` (manual) | Changes rarely; foundational |
| Architecture overview | `green` (manual) | Same |
| API reference (extracted) | `yellow` → auto-promote | Tracked from code; updates with repo sync |
| Design proposal | `blue` (default) | Transient by nature |
| Meeting notes | `blue` (default) | Decay quickly unless promoted |
| Inferred edges | Min(colour) of premises | An edge is only as stable as its weakest source fact |

#### Persistence

```sql
ALTER TABLE documents ADD COLUMN confidence_colour TEXT DEFAULT 'blue';
ALTER TABLE documents ADD COLUMN confidence_colour_since REAL;          -- unix timestamp
ALTER TABLE documents ADD COLUMN confidence_colour_manual INTEGER DEFAULT 0;  -- 1 = set by rag_set_color
```

### Authority Score (Cumulative Reputation)

Confidence is a point-in-time snapshot — it jumps to 1.0 on every update
and decays between reads. This makes it a poor tiebreaker for contradictions:
a freshly updated proposal (confidence 1.0) would beat a stable ADR
(confidence 0.7) in a head-to-head comparison, even though the ADR has
months of accumulated trust.

**Authority** is a separate, monotonic score that grows over time as a
function of confidence. It represents **cumulative reputation** — how
much the system has trusted this document over its lifetime.

```
authority(t) = ∫ confidence(τ) × colour_multiplier dτ
```

In discrete form, authority is updated lazily on every read:

```python
def update_authority(conf: Confidence, now: float) -> float:
    """Add confidence-weighted time delta to authority."""
    delta_t = (now - conf.authority_last_updated) / 86400  # days
    if delta_t <= 0:
        return conf.authority
    colour_mult = {"blue": 1.0, "yellow": 2.0, "green": 5.0}[conf.colour]
    conf.authority += conf.value * colour_mult * delta_t
    conf.authority_last_updated = now
    return conf.authority
```

#### Key Properties

| Property | Confidence | Authority |
|---|---|---|
| Direction | Up and down | Monotonic — only grows |
| Meaning | «How fresh is this right now?» | «How much has the system trusted this over time?» |
| Reacts to | Updates, reads, time | Integral of confidence over time |
| Jumps on | Document edit → 1.0 | Nothing — authority is smooth |
| Used for | Tier classification, search ranking | Contradiction resolution, core/periphery classification |

#### Effect

```
Document A (ADR, green, confidence 0.85, 180 days old):
  authority ≈ 0.85 × 5.0 × 180 ≈ 765

Document B (proposal, blue, confidence 1.0, 2 days old):
  authority ≈ 1.0 × 1.0 × 2 ≈ 2

Contradiction: A vs B. Confidence says B wins (1.0 > 0.85).
Authority says A wins (765 >> 2). System picks A.
```

Authority captures what confidence cannot: **structural, accumulated
trustworthiness that survives momentary freshness spikes.**

#### Manual Seeding

```python
rag_set_authority(doc_id="auth-service/ADR-003", authority=500)
```

Use case: ADRs, architecture docs, and verified reference material
get seeded authority during indexing — no need to wait months for
the integral to accumulate.

#### Contradiction Resolution via Authority

Replaces the confidence-based tiebreaker in Proposal #4:

```
For each contradiction (claim_A, claim_B):
    winner = argmax(authority(claim) for claim in [claim_A, claim_B])
    loser  = argmin(...)

    If authority(winner) / authority(loser) > DOMINANCE_RATIO (default: 3.0):
        → winner marked preferred, loser suppressed
    Else:
        → ratio too close, flagged for LLM review
```

Using authority instead of confidence for contradiction resolution
fixes the core problem: a new document with confidence 1.0 can no
longer shout down a proven document with confidence 0.7.

#### MCP Tools (additions)

| Tool | Signature | Description |
|---|---|---|
| `rag_get_authority` | `doc_id: str` | Return `{authority, colour, last_updated}` |
| `rag_set_authority` | `doc_id: str, authority: float` | Seed or override authority score |
| `rag_list_authoritative` | `limit?: int, colour?: str` | Top documents by authority, optionally filtered by colour |

Confidence responses include authority:

```json
{
  "doc_id": "auth-service/ADR-003",
  "confidence": {"value": 0.85, "tier": "fresh", "colour": "green"},
  "authority": 765.2
}
```

#### Persistence

```sql
ALTER TABLE documents ADD COLUMN authority REAL DEFAULT 0.0;
ALTER TABLE documents ADD COLUMN authority_last_updated REAL;
ALTER TABLE edges ADD COLUMN authority REAL DEFAULT 0.0;
ALTER TABLE edges ADD COLUMN authority_last_updated REAL;
```

### MCP Tools

| Tool | Signature | Description |
|------|-----------|-------------|
| `rag_get_confidence` | `doc_id: str` | Return `{value, tier, colour, last_access, last_update, half_life_days, trend}` for a document |
| `rag_get_edge_confidence` | `source_id: str, target_id: str, relation: str` | Return confidence for a specific edge |
| `rag_list_stale` | `threshold?: float, limit?: int` | List documents below confidence threshold, sorted by most stale first |
| `rag_set_colour` | `doc_id: str, colour: "green" \| "yellow" \| "blue" \| "auto"` | Manually set stability tier. `"auto"` returns to automatic promotion/demotion |
| `rag_confidence_stats` | — | Distribution: `{fresh: N, stable: N, decaying: N, stale: N, green: N, yellow: N, blue: N}` |

Search results and graph traversals automatically include `confidence` in
every returned item:

```json
{
  "doc_id": "abc123",
  "text": "...",
  "score": 0.87,
  "confidence": {
    "value": 0.72,
    "tier": "stable",
    "colour": "yellow",
    "trend": "decaying"
  }
```

### Integration with Semantic Reasoning

Inferred edges (from Proposal #2) inherit confidence from their source facts:

```
OVERRIDES(Child, M) confidence = min(
    IMPLEMENTS(Child, M).confidence,
    EXTENDS(Child, Parent).confidence,
    IMPLEMENTS(Parent, M).confidence
)
```

An inferred edge is only as trustworthy as its weakest premise. If the
underlying IMPLEMENTS edge is stale, the OVERRIDES conclusion carries
that uncertainty forward.

When `rag_verify_edge` is called (LLM verification, Proposal #2, layer 3c),
a successful verification sets `value = 0.95` and resets `last_update`.

### Persistence

Confidence fields are stored in the SQLite `documents` table:

```sql
ALTER TABLE documents ADD COLUMN confidence_value REAL DEFAULT 1.0;
ALTER TABLE documents ADD COLUMN confidence_last_access REAL;
ALTER TABLE documents ADD COLUMN confidence_last_update REAL;
ALTER TABLE documents ADD COLUMN confidence_access_count INTEGER DEFAULT 0;
ALTER TABLE documents ADD COLUMN confidence_half_life_days REAL DEFAULT 30.0;
```

And in the `edges` table:

```sql
ALTER TABLE edges ADD COLUMN confidence_value REAL DEFAULT 1.0;
ALTER TABLE edges ADD COLUMN confidence_last_access REAL;
ALTER TABLE edges ADD COLUMN confidence_last_update REAL;
ALTER TABLE edges ADD COLUMN confidence_access_count INTEGER DEFAULT 0;
```

### Configuration

```bash
# Default half-life per document type (days)
RAG_CONFIDENCE_CODE_HALF_LIFE=30
RAG_CONFIDENCE_ADR_HALF_LIFE=90
RAG_CONFIDENCE_CONCEPT_HALF_LIFE=180
RAG_CONFIDENCE_DEFAULT_HALF_LIFE=60

# Access dampening factor (higher = access slows decay more)
RAG_CONFIDENCE_DAMPENING_FACTOR=7

# Access count saturation limit (prevents immortal docs)
RAG_CONFIDENCE_SATURATION_LIMIT=1000
```

### Example Scenario

**Question:** «Как работает аутентификация?»

Search returns two documents:

| Doc | Score | Confidence | Tier |
|-----|-------|-----------|------|
| `auth-architecture` (ADR) | 0.89 | 0.94 | fresh |
| `old-auth-proposal` (design doc) | 0.81 | 0.31 | decaying |

Both are semantically relevant, but `old-auth-proposal` hasn't been read in
3 months and was last updated 5 months ago. The LLM consumer sees the tier
labels and decides:
- Base answer on `auth-architecture`
- Optionally mention `old-auth-proposal` with a caveat: «вот старая версия
  дизайна, возможно устарела — сверьтесь с кодом»

Without confidence decay, both documents would appear equally authoritative.

### Files

| File | What |
|------|------|
| `src/confidence.py` | `Confidence` dataclass, decay function, tier classifier, lazy evaluation |
| `src/document_store.py` | Schema migration (new columns), access/update hooks |
| `src/graph_store.py` | Edge confidence columns, access hooks for `get_related` |
| `src/mcp_server.py` | New tools: `rag_get_confidence`, `rag_list_stale`, `rag_confidence_stats` |
| `src/rag.py` | Inject confidence into search results and graph traversals |
| `tests/test_confidence.py` | Decay math, saturation, search-hit fractional counting |

### Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Feedback loop: high confidence → more reads → even higher confidence | Access only slows decay, never raises confidence above the post-update value. Only updates reset to 1.0 |
| Feedback loop: high confidence → more search hits → more access points | Search hit contribution is 1/100 vs 1 for explicit read. Marginal impact |
| Ancient docs with huge historical traffic never decay | `access_count` saturates at `SATURATION_LIMIT` (1000). Beyond that, additional access doesn't slow decay further |
| Half-life choice is arbitrary | Per-type defaults (code=30d, ADR=90d, concept=180d), overridable via env. Empirical tuning via `rag_confidence_stats` histogram |
| Write amplification on every read | Confidence recomputed lazily on read, not on access. The access counter is an atomic increment; decay is computed only when `value` is requested |
| Low-value documents are never cleaned up | This is a feature, not a bug. `rag_list_stale(threshold=0.1)` lets an analyst or cron job decide what to archive. Nothing is ever auto-deleted |


# Synthesis: How the Three Proposals Compose

Each proposal solves one weakness of classical RAG. Together they produce
**emergent self-organization** — the system calibrates itself from usage
without a human curator.

## Emergent Layering

As documents accumulate, the system spontaneously stratifies into three
bands driven entirely by confidence decay and access patterns:

```
                    ┌───────────────────────────┐
                    │  Core (confidence > 0.8)    │
                    │  ADR, architecture docs     │
                    │  Frequently read, rarely    │
                    │  updated — high consensus   │
                    └─────────────┬───────────────┘
                                  │ linked from
                    ┌─────────────▼───────────────┐
                    │  Periphery (0.5–0.8)         │
                    │  Active development docs     │
                    │  Frequently updated,         │
                    │  actively consulted          │
                    └─────────────┬───────────────┘
                                  │ rarely linked
                    ┌─────────────▼───────────────┐
                    │  Archive (< 0.5)             │
                    │  Old proposals, deprecated   │
                    │  modules. No one reads       │
                    │  them — they quietly fade    │
                    └─────────────────────────────┘
```

This structure is **not curated** — it emerges from:
- The reasoning layer linking documents by structural dependencies
- Confidence decay demoting abandoned content
- Read traffic slowing decay on frequently-consulted documents

## Scaling Properties

| Property | Without proposals | With all three |
|---|---|---|
| Search quality at 10K+ docs | Degrades: LLM must disambiguate stale from fresh by content alone | Stable: every result carries a confidence tier; LLM filters on it |
| Core vs noise | No distinction; all documents are equal peers in the index | Core is self-identifying: high-confidence, high-access, well-linked |
| Adding new documents | Raises noise floor; new doc indistinguishable from old | New doc starts at 1.0, earns its place through reads — or decays to periphery |
| Reasoning across services | Impossible: graph edges exist but no inference | Transitive closures, override detection, affected-components analysis |
| Curator dependency | High: someone must manually tag/archive/update | Near zero: update events, access traffic, and inference keep the graph alive |
| Archive access | Binary: either deleted or forever present with full authority | Stale content is always reachable, always annotated with «verify before trusting» |

## How They Interact

1. **Confidence decay tells the reasoning layer which facts are reliable.**
   An inferred `OVERRIDES` edge whose premises come from stale documents gets
   a proportionally lowered confidence. The LLM sees «this override was
   inferred, but one premise is from a decaying document — verify».

2. **Reasoning feeds confidence.** A document linked from many core documents
   gets indirect access traffic through graph traversals (`rag_get_related`),
   slowing its decay. Being in the structural center of the system becomes a
   persistence advantage — structurally important docs naturally live longer.

3. **LLM co-pilot closes the feedback loop.** When `rag_verify_edge` confirms
   an inferred edge against source code, it resets that edge's confidence to
   0.95. The next agent that traverses this edge sees it as verified, not
   inferred. Verification is a one-time operation whose result persists.

## Effect on LLM Consumer

Before: LLM receives a flat list of chunks and must guess which ones are
trustworthy. After:

```
Search results for "authentication flow":

✓ [fresh 0.94  ● green]  auth-service/ADR-003-auth-flow.md
  "Authentication uses JWT tokens issued by auth-service..."

✓ [stable 0.72  ● yellow]  auth-service/src/auth/handler.py
  "@app.post('/login') → returns JWT access + refresh tokens"

△ [decaying 0.38  ● blue]  proposals/old-oauth-proposal.md
  "Proposed OAuth2 migration path. Was last updated 4 months ago,
   no reads in 6 weeks. Consider verifying against current code."
```

The LLM can now construct an answer that:
- Builds on fresh and stable sources
- Mentions the old proposal with an explicit caveat
- Knows to verify before presenting decaying content as fact

## Summary

| Layer | Problem solved | Mechanism |
|---|---|---|
| Reasoning | Questions beyond semantic search | Datalog inference, transitive closure |
| Confidence | Staleness invisible to LLM | Exponential decay + access dampening |
| **Together** | **System self-calibrates from usage** | Core emerges, archive settles, LLM sees freshness signal in every result |


# Naming: Hebbian Knowledge Graph (HKG)
# Нейросимволическая база знаний с хеббовской пластичностью (НБЗХП)

The composition of all three proposals produces a system that is qualitatively
different from classical RAG. It is not merely a retrieval engine — it is a
**self-regulating, neuro-symbolic knowledge base** with the following
emergent properties:

| Property | Mechanism |
|---|---|
| Memory | Confidence decay — documents remember their usage history |
| Forgetting | Exponential decay — unused content fades, never deleted |
| Consolidation | Stability tiers — structurally important knowledge resists decay |
| Associative reasoning | Inference engine — new facts derived from existing ones |
| Meta-cognition | LLM co-pilot — verifies inferences against ground truth, proposes rules |
| Self-organization | Core/periphery/archive emerges without a curator |
| Homeostasis | System trends toward equilibrium: new content cools, core consolidates, noise fades into background |

The name **Hebbian Knowledge Graph** reflects the core design principle:

> *Neurons that fire together, wire together. Cells that don't, fade.*

- **Hebbian** — the system strengthens knowledge that is used and weakens
  knowledge that is not, mirroring Hebbian plasticity in biological neural
  networks. Read traffic is the signal; exponential decay is the forgetting
  curve; stability tiers are metaplasticity — the structural resistance to
  weakening that strong synapses acquire over time.
- **Knowledge Graph** — entities and typed relations form the structural
  backbone. Inference derives new facts. The ontology declares what kinds
  of things exist and how they relate.

Русское название — **Нейросимволическая база знаний с хеббовской
пластичностью (НБЗХП)** — подчёркивает двойственную природу системы:
символический слой (онтология, граф, Datalog-правила) и нейро-слой
(confidence decay, metaplasticity, LLM-верификация). Это не просто
RAG с графом — это архитектура, в которой символьный вывод и
статистическая саморегуляция работают в одном цикле.

An HKG is to a knowledge base what long-term potentiation is to a synapse:
it is not a static store — it is a living structure that calibrates itself
through use.


# PROPOSAL: Automatic Contradiction Detection (Cognitive Dissonance)

## Problem

A knowledge graph that grows from multiple sources — code extraction, ADRs,
proposals, LLM-generated summaries — will inevitably accumulate contradictions:

- Two documents state opposite facts about the same entity
- An inferred edge conflicts with a stated edge
- A document describes an API that the extracted code graph says doesn't exist

Classical RAG has no way to detect these. The LLM consumer receives all
chunks and may not notice the conflict, or notices it but can't resolve
it without a separate verification round-trip.

With inference and confidence decay already in place, the system has
everything needed to **detect contradictions automatically** and **resolve
them by freshness** — the graph equivalent of cognitive dissonance.

## Solution

### Contradiction Types

| Type | Pattern | Example |
|---|---|---|
| **Type clash** | Same entity, conflicting types | `UserService` is both `Class` and `Module` |
| **Edge conflict** | Opposite relations between same nodes | `CALLS(A, B)` + `AVOIDS(A, B)` — A depends on B but explicitly avoids it |
| **Inferred vs stated** | Inference derives X, graph explicitly states ¬X | Rule infers `OVERRIDES(Child, M)`, but graph has `REMOVES(Child, M)` |
| **Factual contradiction** | Two documents claim different values for same property | Doc A: «auth uses JWT», Doc B: «auth uses sessions» |
| **Structural impossibility** | Graph topology violates ontology constraints | Cycle in a DAG-only relation; service depends on itself transitively |

### Detection: Contradiction Rules

Contradiction rules are inference rules with a `CONTRADICTS` output:

```python
@contradiction_rule(
    name="type_clash",
    description="Same entity cannot have two different types"
)
def detect_type_clashes(g: nx.DiGraph) -> list[Contradiction]:
    """For each entity, check that all incoming HAS_TYPE edges agree."""
    clashes = []
    for node in g.nodes:
        types = {d["value"] for _, _, d in edges_of_type(g, "HAS_TYPE")
                 if d["target"] == node}
        if len(types) > 1:
            clashes.append(Contradiction(
                kind="type_clash",
                subjects=[node],
                claims=[f"{node} is {t}" for t in types],
            ))
    return clashes

@contradiction_rule(
    name="inferred_vs_stated",
    description="Inferred edge conflicts with an explicitly stated edge"
)
def detect_inferred_vs_stated(g: nx.DiGraph) -> list[Contradiction]:
    """OVERRIDES(Child, M) inferred, but REMOVES(Child, M) stated."""
    # Inferred edges carry _inferred=True metadata
    # Stated edges carry _inferred=False
    ...
```

### Resolution: Highlight, Don't Resolve

Contradiction resolution is not a solvable problem for RAG or HKG —
and it doesn't need to be. The system's job is not to decide who is
right. Its job is to **make the conflict visible** to the entity that
can decide: the human or LLM adding the document.

HKG never auto-suppresses a claim. When a contradiction is detected,
the system produces a **contradiction report** — a structured
side-by-side comparison of the conflicting claims with their full
authority context:

```
For each contradiction (claim_A, claim_B):
    Emit: {
        claim_A: { text, source, authority, confidence, colour },
        claim_B: { text, source, authority, confidence, colour },
        authority_ratio: authority(A) / authority(B) if both > 0 else ∞,
        recommendation: null  // system never recommends a winner
    }

    The consumer (LLM or human) decides:
    - Accept new claim → rag_accept_claim(contradiction_id, winner)
    - Keep old claim  → rag_reject_claim(contradiction_id)
    - Defer           → contradiction persists, both claims coexist
```

The authority and colour signals make the decision informed, not
automatic. «ADR-003, green, authority 765» vs «new proposal, blue,
authority 0» — the LLM sees this and understands the weight of what
it's about to override. But the system never makes the call.

### LLM Review (Cognitive Dissonance Resolution)

When the LLM consumer requests knowledge that touches a contradiction,
the contradiction report is included in the response — the LLM sees
both claims and their authority context inline:

```
Query: "How does AuthService authenticate?"

Relevant documents:

✓ [fresh 0.85 · ● green · authority 765] ADR-003:
  "AuthService issues JWT tokens via /login endpoint"

△ [stable 0.68 · ● blue · authority 12] auth/README.md:
  "AuthService uses session cookies"
  ⚠ CONTRADICTS ADR-003 (authority ratio 63.8)
     This document claims session cookies; ADR-003 claims JWT.
     ADR-003 is green-tier with 180 days of accumulated trust.
     Resolution: unresolved — decide which claim to accept.

LLM response:
  "According to ADR-003 (authoritative, last verified 2 days ago),
   AuthService uses JWT tokens. However, auth/README.md mentions
   session cookies. These claims contradict each other. I recommend
   verifying against auth-service/src/auth/handler.py."
```

The LLM never receives a pre-resolved answer. It receives the full
dissonance with all the signals it needs to reason about it. The HKG's
role ends at «here is what I know, and here is what doesn't add up.»

#### Manual Resolution

The LLM or human can explicitly resolve a contradiction:

```
rag_accept_claim(contradiction_id, winner_id)
  → winner's confidence → 0.95 (verified by acceptance)
  → loser marked as contradicted, ALL relations severed
  → loser moved to archive (excluded from search, inference, graph traversals)
  → loser eligible for deletion by cleanup cron
  → CONTRADICTS edge stored as audit trail pointing winner → loser

rag_reject_claim(contradiction_id)
  → both claims remain active, contradiction closed as «irreconcilable»
  → both documents carry a warning annotation
```

When a claim loses, it is **fully ejected** from the active knowledge
graph — not suppressed, not deprecated, but removed from circulation.
It exists only in the archive as an audit record. A cleanup cron can
purge archived contradicted documents after a retention period (default:
90 days). This keeps the active graph clean: defeated claims do not
linger, do not accumulate authority, and do not pollute future queries.

Unresolved contradictions persist indefinitely. A document involved
in an unresolved contradiction shows a warning badge in search results.
This is intentional: the HKG refuses to hide unresolved conflict.

### New MCP Tools

| Tool | Signature | Description |
|---|---|---|
| `rag_detect_contradictions` | `scope?: "all" \| "entity" \| "relation", entity_id?: str` | Run contradiction rules, return list of conflicts with authority context |
| `rag_get_contradictions` | `entity_id: str` | All unresolved contradictions involving this entity |
| `rag_accept_claim` | `contradiction_id: str, winner_id: str` | Accept winner; loser is ejected to archive, all relations severed, eligible for deletion |
| `rag_reject_claim` | `contradiction_id: str` | Close contradiction as irreconcilable; both claims coexist with warning |

Search results include contradiction annotations:

```json
{
  "doc_id": "auth/README.md",
  "text": "AuthService uses session cookies...",
  "confidence": {"value": 0.68, "tier": "stable"},
  "authority": 12.3,
  "contradiction": {
    "status": "unresolved",
    "conflicts_with": "ADR-003-auth-flow.md",
    "conflict_authority": 765.0,
    "conflict_colour": "green"
  }
}

### Integration with Existing Layers

**Authority feeds contradiction awareness:**
When a contradiction is detected, both claims are annotated with each
other's authority context. Time may widen the authority gap, making
the decision easier — but the system never makes the decision. The
contradiction report evolves (authority numbers change) but resolution
always requires explicit human or LLM action.

**Inference engine provides contradiction rules:**
Contradiction rules are first-class citizens in the rule registry,
alongside inference rules. Both run in the same fixed-point loop.
An inferred edge can trigger a contradiction check. A resolved
contradiction feeds back into inference: deprecated claims are
excluded from future inference premises.

**LLM co-pilot verifies, not resolves:**
The `rag_verify_edge` tool from Proposal #2 verifies a claim against
source code. Successful verification boosts confidence to 0.95 and
updates authority — but does NOT auto-resolve the contradiction.
Resolution remains a separate, explicit step.

### Files

| File | What |
|---|---|
| `src/contradiction.py` | Contradiction detector: type-clash, edge-conflict, inferred-vs-stated rules |
| `src/contradiction_resolver.py` | Freshness tiebreaker, LLM review dispatcher, suppression logic |
| `src/rag_data/rules/contradictions/` | Per-rule contradiction modules (auto-discovered) |

### Example Scenario

```
1. Repo sync extracts: "AuthService" HAS_TYPE "Service"
2. ADR-003 ingested: "AuthService" HAS_TYPE "Module"
3. Contradiction rule fires: type_clash(AuthService, Service, Module)
4. Contradiction report emitted:
     ADR-003 authority = 765 · green · confidence 0.85
     Extracted fact authority = 1.7 · blue · confidence 1.0
     Authority ratio = 450
     → Unresolved. Both claims coexist with warning.
5. LLM consumer queries «What is AuthService?» — sees both claims
   with authority context, notes the contradiction, and either:
   a) Inspects source code and calls rag_accept_claim(winner=...)
   b) Flags for human review
6. No automatic suppression. The conflict is visible, not hidden.
```

Without contradiction detection: the LLM receives both facts and
presents «AuthService is both a Service and a Module» — confusing
and wrong. With it: the LLM sees the conflict, sees the authority
gap, and is equipped to resolve it deliberately.

### Risks and Mitigations

| Risk | Mitigation |
|---|---|
| False contradictions: legitimate synonyms flagged as conflicts | Ontology-defined synonym relations; `EQUIVALENT_TO` edges prevent clash detection |
| Over-eager suppression: new correct fact suppressed by old high-authority fact | No auto-suppression exists. Both claims coexist with warning until explicitly resolved. Manual `rag_set_authority` can bootstrap critical new documents |
| Contradiction cascade: resolving one contradiction triggers more | Single-pass resolution per cycle; max resolution depth |
| LLM review cost: every contradiction invokes LLM | Contradiction report is cheap (structural rules). LLM is only invoked when consumer queries touch a contradiction — not on every detection |

### New Document Contradiction Flow

When an LLM agent adds a document via `rag_add_document`, the document
starts with `authority = 0` and `confidence = 1.0`. This is by design:
a new document has no accumulated trust, regardless of its momentary
confidence. But a new document with authority 0 that contradicts an
existing document with authority 765 creates a mathematical edge case —
the authority ratio is infinite, and automatic resolution is impossible.

This is not a bug. It is a **forcing function**: the system MUST surface
the contradiction to the LLM that added the document, because the LLM
is the only entity qualified to decide whether it is fixing a mistake
or making one.

#### Flow

```
LLM calls rag_add_document(content, resolve_contradictions=True)
  │
  ▼
System indexes document
  confidence = 1.0, authority = 0, colour = blue
  │
  ▼
System runs contradiction rules scoped to the new document
  │
  ▼
Returns structured contradiction report directly to LLM:

  ⚠ New document «auth-v2-proposal» introduces 2 contradictions:

  ┌─ Contradiction #1: type_clash ─────────────────────────────┐
  │ New:  "AuthService" IS_A "Module"                          │
  │       authority 0 · confidence 1.0 · colour blue           │
  │                                                            │
  │ Old:  "AuthService" IS_A "Service"                         │
  │       authority 765 · confidence 0.85 · colour green       │
  │       Source: ADR-003 (last read 2 days ago)               │
  │                                                            │
  │ ⚡ Ratio ∞ — old document is structurally authoritative    │
  │    ADR-003 is a green-tier document with 180 days of trust │
  └────────────────────────────────────────────────────────────┘

  ┌─ Contradiction #2: factual_conflict ───────────────────────┐
  │ New:  "AuthService uses OAuth2 tokens"                     │
  │       authority 0 · confidence 1.0 · colour blue           │
  │                                                            │
  │ Old:  "AuthService issues JWT tokens"                      │
  │       authority 120 · confidence 0.72 · colour yellow      │
  Resolve by calling:
    rag_accept_claim(id=<id>, winner_id="<new|old>")  — pick winner
    rag_reject_claim(id=<id>)                         — keep both, mark irreconcilable
  (Unresolved contradictions persist and appear in all future queries touching these documents.)
  └────────────────────────────────────────────────────────────┘

#### Why This Matters

**LLM is smarter than the system, but it's also fallible.** The HKG's
job is not to be the final authority — it's to ensure the LLM never
overrides authoritative knowledge **by accident**. The contradiction
report is a cognitive speed bump:
- LLM sees «ADR-003, green, authority 765» and pauses
- If the LLM is correcting a genuine mistake → `rag_accept_claim(id, winner_id=new)` — conscious override
- If the LLM hallucinated → `rag_accept_claim(id, winner_id=old)` or deletes its document
- The signal is the **colour** and **authority gap** — LLM understands
  «green ADR» means «think twice» far better than a numeric threshold

This is the HKG's core value proposition for LLM consumers: **it tells
you what it knows, how well it knows it, and whether your new claim
contradicts old claims that have proven trustworthy over time.**

#### When Authority Is Non-Zero

If the new document is added by a repo sync (not LLM) and already has
some authority from prior existence, the same flow applies. The
contradiction report is always emitted — the LLM sees it as context
in the response. No automatic resolution occurs regardless of ratio.

#### Acceptance as a Deliberate Operation

Overriding an authoritative claim is not a simple write — it is an
**acceptance ceremony**. The HKG treats knowledge replacement as a
high-friction operation by design:
- Old claims with high authority cannot be silently overwritten
- The contradiction report is always emitted — never auto-resolved
- LLM must explicitly call `rag_accept_claim(contradiction_id, winner_id)` —
  a conscious act, not a side effect of adding a document
- The loser is fully ejected: all relations severed, moved to archive,
  eligible for deletion after retention period
- The CONTRADICTS edge persists in the graph as an audit trail:
  who overrode what, when, and why

This asymmetry — easy to add, hard to override — mirrors how human
teams treat authoritative documents. Anyone can write a proposal.
Overriding an ADR requires explicit acknowledgment that you have
read the ADR, understood it, and are consciously replacing it.


# Known Limitations & Open Problems

Every layer introduces its own failure modes. Below is a systematic
audit of what can go wrong and how to mitigate it — both in v1 and
as future research directions.

## 1. Inference Engine

### Rule Explosion

Transitive closure on a dense graph produces O(n²) edges. At 10K
nodes with `DEPENDS_ON`, this is 100M inferred edges. The fixed-point
loop never terminates within practical limits.

**Mitigation:** maximum inference depth (≤ 3 hops). Per-query edge
limit: at most 10 inferred edges are returned. The LLM consumer
receives the top-10 highest-confidence inferred edges and can request
more via pagination (`rag_infer(offset=10)`). This prevents graph
saturation while keeping the most valuable inferences immediately
available. Inferred edges inherit `min(confidence)` of all premises —
garbage premises produce garbage conclusions with low confidence,
which rank below the top-10 threshold.

**Open:** lazy transitive closure — compute on demand per query rather
than materialising all pairs ahead of time. LLM-guided edge curation:
the LLM itself decides which 10 edges are most important to keep,
pruning the rest.

### Garbage In, Garbage Out

One false positive edge from code extraction triggers a cascade of
false inferred edges. The system has no way to distinguish «inferred
from solid premises» vs «inferred from extraction noise».

**Mitigation:** inferred edges carry provenance — which premises were
used. If a premise is later deleted or contradicted, dependent
inferred edges are invalidated. Edge confidence = min(premise
confidences), so noisy edges yield low-confidence conclusions.

### Missing Rules

Inference only answers questions for which someone wrote a rule.
«Which services use the Circuit Breaker pattern?» — no rule → no
answer. Coverage depends on manual rule authoring.

**Mitigation:** LLM-propose-rules (Proposal #2, layer 3b) covers
discovery. **Open:** pattern mining on the graph — cluster
substructures, detect recurring motifs, propose rules for them
without LLM involvement.

### Ontology Drift

The ontology is fixed: Class, Method, Service, Endpoint, Module.
When the codebase introduces new concepts — MessageQueue, CronJob,
FeatureFlag — extraction produces entities with unknown or missing
types. These entities are invisible to inference: no rule matches
them, no contradiction rule checks them, no reasoning applies.
The system progressively goes blind to new parts of the codebase.

**Mitigation:** periodic ontology gap scan. Entities without a type
or with low inference coverage (few incident edges matching known
relations) are flagged. The LLM co-pilot (Proposal #2, layer 3a)
proposes new entity types and relation types for them.
`rag_propose_ontology_extensions()` — batch proposal for all
untyped entities detected in the last scan.

**Open:** semi-automatic ontology extension. If the LLM proposes
a new type `FeatureFlag` and a human approves it, can we mine the
graph for existing entities that match the new type's signature
and reclassify them automatically?

## 2. Confidence Decay

### Cold Start

A small team with 5 queries per week generates near-zero traffic.
All documents decay at the same rate → no stratification → tiers
are meaningless. The system needs a minimum pulse to breathe.

**Mitigation:** global decay floor. If total system traffic over
30 days is below `MIN_TRAFFIC_THRESHOLD`, decay is suspended or
slowed globally. A dormant system should not rot.

### Burst Traffic

A production incident drives 50 reads of one file in an hour.
`access_count` saturates at `SATURATION_LIMIT`. After the incident,
the document remains artificially alive for months — it was read
during a crisis, not because it's authoritative.

**Mitigation:** burst detection. If read frequency exceeds N standard
deviations from the document's historical mean, apply diminishing
returns — access count increases sublinearly (e.g., log scale) for the
duration of the burst. Rate limiter on access count accumulation.

### No Negative Feedback

Current model: reading a document always slows its decay. But «read
and found wrong» should have the opposite effect. There is no signal
for distrust.

**Mitigation:** negative access. A `QualityCard` with 👎 or an
explicit `rag_mark_incorrect(doc_id)` call accelerates decay —
subtracts from effective access count. Human distrust overrides
automatic heuristics.

**Open:** implicit negative feedback. If a user reads document A,
then immediately reads document B on the same topic — was A
unsatisfactory? Can we infer distrust from reading patterns?

### Edge Confidence Blind Spot

Confidence and authority models focus on nodes. Edges have their own
aging dynamics that are currently unaddressed:

- `CALLS(A, B)` — A's code changed, no longer calls B. Node A was
  updated (confidence 1.0), but the edge retains old confidence.
- Edges are rarely accessed directly — they're traversed via
  `rag_get_related`. Their access counts are always lower than nodes'.
- An edge between two green nodes may be false, but inherits inflated
  trust from its incident nodes.

**Mitigation:** edge-specific decay. When either incident node is
updated, the edge's decay accelerates (half-life halved for 7 days).
This reflects the structural reality: if A changed, its outgoing
edges are suspect until re-verified. `rag_verify_edge` (Proposal #2)
extends to edges — LLM checks whether the edge still exists in code
and resets confidence to 0.95 if confirmed.

## 3. Stability Tiers

### Green Inflation

After six months of operation, 80% of documents may be green.
The tier loses its discriminatory power — it no longer signals
«think twice before overriding.»

**Mitigation:** green quota. At most N% of documents (default: 20%)
may be green simultaneously. When the quota is exceeded, the
lowest-authority green documents are demoted to yellow. This creates
healthy competition: only the most structurally proven documents
retain green status.

### Wrongful Promotion

A document with an error is heavily read (people are debugging it).
High access rate → high confidence → promoted to yellow → green.
The erroneous document becomes structurally protected from decay.

**Mitigation:** negative feedback overrides tiers. A single 👎
resets colour to blue, regardless of confidence history. Human
judgment trumps automatic promotion.

### Demotion Instability

A document drops below the demotion threshold for one day due to
a calculation glitch and immediately loses green. Re-promotion
takes another 60 days.

**Mitigation:** sustained breach requirement. Confidence must be
below the demotion threshold for N consecutive days (default: 7)
before colour changes downward. Protection against transient
fluctuations.

## 4. Authority

### Absolute Authority Makes Old Documents Unassailable

An ADR with authority 5000 will never lose to a new document in a
contradiction report. The LLM will always see it as the dominant
claim. But ADRs do become obsolete — architecture changes. The system
relies on the LLM to consciously override, which requires the LLM
to have independent knowledge that the ADR is wrong.

**Mitigation:** slow authority decay — e.g., 1% per month. Not
as aggressive as confidence decay, but enough that a 5-year-old
ADR without reads eventually becomes contestable. Combined with
relative authority (percentile, not absolute value), the system
can detect «this was once authoritative but is no longer.» The
contradiction report shows both absolute and relative authority.

### Authority Ratio on Noise

Document A: authority 0.003. Document B: authority 0.001.
Ratio = 3.0. Both are noise — neither has meaningful authority.
The contradiction report is still emitted, but the LLM sees both
authorities are negligible and treats both claims as unproven.

**Mitigation:** authority tier annotation. Documents below
`MIN_MEANINGFUL_AUTHORITY` (default: 10.0) are annotated as
«low authority — limited trust history» in the report. The LLM
can see that neither side has accumulated meaningful reputation.

### Authority Inflation

Authority only grows. After a year, the mean authority across
all documents shifts from 10 to 500. The absolute difference
between «authoritative» and «not» compresses — like monetary
inflation.

**Mitigation:** percentile-based authority display. «Top 5% by
authority» is stable over time, even as absolute values inflate.
Absolute authority is retained for computation; percentile is
used for presentation in contradiction reports.

### Archive vs Deletion Trade-off

Ejected documents remain in the archive as audit trail. A cleanup cron
purges archived contradicted documents after a configurable retention
period (default: 90 days). The trade-off: too short → loses audit
history; too long → archive bloat.

**Mitigation:** configurable retention. Critical namespaces (ADRs)
may have longer retention. Archived documents are excluded from all
queries, search, and inference — they consume only disk.

### Authority Opacity

Authority is a single number — the integral of confidence over time.
For an LLM consumer, «authority 765» is opaque. Is 765 high? Low?
Expected for this document type? The LLM cannot calibrate its trust
without understanding how the number was computed.

**Mitigation:** `rag_explain_authority(doc_id)` — decomposes authority
into components:

```
{
  "authority": 765.2,
  "percentile": "top 3%",
  "components": {
    "colour": "green (×5.0 multiplier)",
    "age_days": 180,
    "access_count": 42,
    "avg_confidence": 0.85,
    "confidence_trend": "stable — above 0.8 for 90 consecutive days"
  },
  "verdict": "Structurally authoritative. Green-tier ADR with sustained
              high confidence and consistent access. Appropriate for
              this document type."
}
```

The LLM receives both the number and the story behind it. «Authority
765 because it's a green ADR, 180 days old, read 42 times, with
confidence above 0.8 for 3 months» is actionable. «Authority 765» is not.

## 5. Contradiction Detection

### False Negatives on Semantic Contradictions

Structural contradictions (type clash, edge conflict) are caught
by rules. Semantic contradictions («auth uses JWT» vs «auth uses
OAuth») require comparing the *meaning* of two text chunks — out
of scope for Datalog rules. Most real-world contradictions are
semantic, not structural.

**Mitigation:** periodic semantic scan. A nightly cron selects
pairs of documents with high embedding similarity but different
sources, and asks the LLM: «Do these documents contradict each
other?» Expensive but acceptable at daily cadence.

**Open:** contradiction embedding. Can we train a lightweight
classifier that detects factual contradiction between two chunks
without invoking the LLM?

### Archive Cascade

Document D is ejected to archive via `rag_accept_claim`. Fifty inferred
edges had D as a premise. Those edges are now dangling — their premise
is no longer in the active graph.

**Mitigation:** lazy re-evaluation. Inferred edges are recalculated on
next access. If a premise is archived, the edge's confidence drops to
the min of remaining premises. If no premises remain, the edge is
deleted. No permanent damage — inference is always computed from the
current active graph.
### Irreversible Resolutions

A contradiction is resolved via `rag_accept_claim`, a winner is chosen,
the loser is ejected to archive. Six months later, the winner itself
is found wrong. The loser is still in archive, unreachable by normal
queries. The system has no mechanism to restore it if the resolution
was mistaken.

**Mitigation:** reversible resolutions with TTL. Each resolution
stores a snapshot of both claims' authority at resolution time.
If the winner is itself contradicted, the resolution is reopened.
The loser can be restored from archive via `rag_restore_claim(doc_id)`.
Archive retention ensures the document still exists on disk.
Resolution is a soft commitment, not a permanent judgment.

### Mass Contradiction (Contradiction Storm)

A bad repo sync misclassifies 50 entities → 50 contradiction reports
fired simultaneously. The LLM is flooded. Individual review of each
contradiction is impractical — the consumer needs to see the pattern,
not the list.

**Mitigation:** contradiction aggregation. When N > AGGREGATION_THRESHOLD
(default: 5) contradictions share the same source and structural pattern,
they are collapsed into a single aggregated report:

```
⚠ Repo sync «auth-service» introduced 47 type_clash contradictions.
  Pattern: extracted type → «Module», existing claim → «Service».
  Affected entities: AuthService, UserService, TokenService, ...
  Source authority: extracted facts avg 2.1, existing claims avg 340.
  Recommendation: review the extraction config for auth-service.
  [Expand to see all 47] [Accept all old] [Accept all new]
```

The LLM sees one pattern, not 47 individual conflicts. Batch
resolution is supported: `rag_accept_claim(contradiction_ids=[...], winner=...)`.

### Batch Recovery from Wrong Resolutions

A series of `rag_accept_claim` calls turned out to be wrong — the
winning documents were themselves later contradicted. Restoring
each loser individually is impractical at scale.

**Mitigation:** batch rollback. `rag_rollback_resolutions(source=X,
since=<date>)` — restores all losers that were ejected due to
contradictions where the winner came from source X within the given
time window. The losers are restored from archive, their relations
re-established. Useful when an entire extraction run or agent session
produced bad resolutions.

## 6. Global Risks

### Adversarial Use

A buggy agent or malicious cron adds 1000 junk documents → graph
saturation → contradiction detection overwhelmed → LLM budget
exhausted on garbage review.

**Mitigation:** per-agent rate limiting on document creation.
Documents that trigger > N contradictions on insertion enter a
sandbox queue — not visible to search until reviewed. Anomaly
detection: if a document's contradiction count exceeds K standard
deviations from the mean, it is quarantined.

### Feedback Loop Between Layers

High authority → more reads → slower decay → higher confidence →
even higher authority. The system could enter a «rich get richer»
regime where early-established documents dominate forever.

**Mitigation:** confidence is bounded (cannot exceed 1.0). Authority
growth rate, not absolute authority, is affected by reads. Slow
authority decay (1%/month) provides a counter-force. Green quota
prevents tier monopolisation.

### Loss of Novelty

If authority and stability tiers work too well, new documents
struggle to break into visibility. The system becomes conservative
— it prefers old, proven knowledge over new, possibly better
knowledge.

**Mitigation:** new documents start at confidence 1.0 and are
always included in search results. The contradiction report, not
search suppression, is the check against error. Novelty bonus:
new documents get a temporary search-ranking boost (first 7 days)
to ensure they are seen and can begin accumulating authority.

### Cold Start: Bootstrapping

An empty HKG has no authority, no confidence history, no
stratification. All documents start equal — blue, authority 0.
For the first 30 days, the HKG is indistinguishable from a
plain RAG. Seeding initial authority and colour is necessary
but currently undefined.

**Mitigation:** bootstrap policy — declarative rules for initial
document classification on first import:

| Document source | Initial colour | Initial authority | Rationale |
|---|---|---|---|
| ADR (architectural decision record) | `green` | 500 | Structural by definition |
| Architecture overview | `green` | 300 | Same |
| Code extraction (stable module) | `blue` | 10 | Needs time to prove itself |
| Design proposal | `blue` | 0 | Transient by nature |
| Meeting notes | `blue` | 0 | Decay quickly |
| API reference (extracted) | `yellow` | 50 | Tied to code; semi-stable |

Bootstrap policy is applied once on first import. After that,
automatic promotion/demotion takes over. The policy is
configurable per project via `bootstrap.yaml`. Documents that
don't match any rule start at blue, authority 0.

**Open:** can the bootstrap policy itself be learned? After 6
months of operation, the system knows which document types
tend to become green. Could it propose bootstrap policy updates
for new projects based on historical patterns?