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

### Resolution: Authority Wins

When a contradiction is detected, the system does NOT delete either fact.
Instead it applies the **authority tiebreaker** — cumulative reputation
decides, not the momentary confidence snapshot:

```
For each contradiction (claim_A, claim_B):
    winner = argmax(authority(claim) for claim in [claim_A, claim_B])
    loser  = argmin(...)

    If authority(winner) / authority(loser) > DOMINANCE_RATIO (default: 3.0):
        → winner marked as "preferred", loser suppressed in search
    Else:
        → ratio too close, flagged for LLM review
```

Authority is used instead of confidence because it is **immune to freshness
spikes**. A newly updated proposal with confidence 1.0 and authority 2 does
not beat a six-month-old ADR with confidence 0.7 and authority 765. The
Hebbian principle still holds — the document that has accumulated more
trust over its lifetime wins — but the metric is cumulative, not point-in-time.

### LLM Review (Cognitive Dissonance Resolution)

When authority ratio is too close to call or the contradiction is high-impact
(e.g., architecture-level), the LLM co-pilot is invoked:

```
Contradiction detected:
  Claim A [authority 765, confidence 0.72]: "AuthService issues JWT tokens"
    Source: ADR-003, last updated 2026-06-15, colour green
  Claim B [authority 12, confidence 0.68]: "AuthService uses session cookies"
    Source: auth/README.md, last updated 2026-06-20, colour blue

Resolution: authority ratio 765 / 12 = 63.8 → clear winner (>> 3.0).
→ Claim A automatically preferred. Claim B suppressed.
→ No LLM review needed.

(If ratio were < 3.0, LLM would inspect source code to break the tie.)

### New MCP Tools

| Tool | Signature | Description |
|---|---|---|
| `rag_detect_contradictions` | `scope?: "all" \| "entity" \| "relation", entity_id?: str` | Run contradiction rules, return list of detected conflicts with confidence deltas |
| `rag_get_contradictions` | `entity_id: str` | All unresolved contradictions involving this entity |
| `rag_resolve_contradiction` | `contradiction_id: str, winner: str` | Manually resolve: pick winning claim |

Search results include contradiction annotations:

```json
{
  "doc_id": "auth/README.md",
  "text": "AuthService uses session cookies...",
  "confidence": {"value": 0.68, "tier": "stable"},
  "authority": 12.3,
  "contradiction": {
    "status": "suppressed",
    "winner": "ADR-003-auth-flow.md",
    "winner_authority": 765.0,
    "ratio": 62.2,
    "reason": "Authority ratio 62.2 >> 3.0 — automatic resolution"
  }
}
```

### Integration with Existing Layers

**Authority feeds contradiction resolution:**
When a contradiction is detected, the system records it but doesn't
immediately resolve. Time works for the system — the winning fact's
authority continues to accumulate, widening the gap against the loser.
After a grace period (default: 7 days), unresolved contradictions
are re-evaluated: the authority ratio may have crossed the 3.0 threshold
for automatic resolution.

**Inference engine provides contradiction rules:**
Contradiction rules are first-class citizens in the rule registry,
alongside inference rules. Both run in the same fixed-point loop.
An inferred edge can trigger a contradiction check; a resolved
contradiction can suppress further inference on dubious premises.

**LLM co-pilot closes the loop:**
The same `rag_verify_edge` tool from Proposal #2 is used to verify
the winning claim against source code. Successful verification
boosts confidence to 0.95, marks the contradiction resolved, and
updates authority with the verified confidence weight.

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
4. Authority check:
     ADR-003 authority = 765 (green, 180 days of accumulated trust)
     Extracted fact authority = 1.7 (blue, just synced 2 days ago)
   Ratio = 765 / 1.7 = 450 >> 3.0 → resolved automatically
5. "Module" claim suppressed. ADR-003 confirmed as authoritative.
6. No LLM review needed — authority gap is decisive.
```

Without contradiction detection: the LLM consumer receives both facts
and presents «AuthService is both a Service and a Module» — confusing
and wrong. With it: the conflict is resolved before the consumer sees it.

### Risks and Mitigations

| Risk | Mitigation |
|---|---|
| False contradictions: legitimate synonyms flagged as conflicts | Ontology-defined synonym relations; `EQUIVALENT_TO` edges prevent clash detection |
| Over-eager suppression: new correct fact suppressed by old high-authority fact | New facts start with authority 0; grace period (7 days) gives them time to accumulate before ratio comparison. Manual `rag_set_authority` can bootstrap critical new docs |
| Contradiction cascade: resolving one contradiction triggers more | Single-pass resolution per cycle; max resolution depth |
| LLM review cost: every contradiction invokes LLM | Only close authority ratios (< 3.0) or high-impact contradictions trigger LLM. Clear winners (ratio > 3.0) resolve automatically |

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
  │       Source: auth/README.md (last read 5 days ago)        │
  │                                                            │
  │ ⚡ Ratio ∞ — old document has 120 days of accumulated trust│
  └────────────────────────────────────────────────────────────┘

  Resolve by calling:
    rag_resolve_contradiction(id=<id>, winner="new"|"old")
```

#### Why This Matters

**LLM is smarter than the system, but it's also fallible.** The HKG's
job is not to be the final authority — it's to ensure the LLM never
overrides authoritative knowledge **by accident**. The contradiction
report is a cognitive speed bump:

- LLM sees «ADR-003, green, authority 765» and pauses
- If the LLM is correcting a genuine mistake → `rag_resolve(id, winner=new)` — conscious override
- If the LLM hallucinated → `rag_resolve(id, winner=old)` or deletes its document
- The signal is the **colour** and **authority gap** — LLM understands
  «green ADR» means «think twice» far better than a numeric threshold

This is the HKG's core value proposition for LLM consumers: **it tells
you what it knows, how well it knows it, and whether your new claim
contradicts old claims that have proven trustworthy over time.**

#### When Authority Is Non-Zero

If the new document is added by a repo sync (not LLM) and already has
some authority from previous existence, the same flow applies but the
ratio is finite and automatic resolution may fire if ratio > 3.0.
The contradiction report is always emitted — the LLM sees it as context
in the response, even if resolution was automatic.