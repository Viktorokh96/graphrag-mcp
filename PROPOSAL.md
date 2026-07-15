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
