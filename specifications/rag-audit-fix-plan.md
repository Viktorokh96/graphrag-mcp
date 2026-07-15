# Implementation Plan: RAG Audit Fixes (D1-D10)

## Overview

Comprehensive fix for 10 defects (D1-D10) identified in `specifications/rag-audit-fixes.md`.
Organized by priority: P0 (critical), P1 (serious), P2/P3 (improvements).

**Status:** ✅ All code changes implemented, tests written, all passing.
**Date:** 2026-07-06
**Updated:** 2026-07-15 — документация обновлена, `bm25_index.py` удалён (BM25 в Qdrant)

### Legend
- ✅ Implemented
- ⚠️ Known limitation / deferred
- ⏳ Not started

## Changes per file

> **⚠️ NOTE (2026-07-15):** `src/bm25_index.py` удалён — BM25 теперь в Qdrant sparse vectors.
> Ссылки на `bm25_index.py` ниже — исторические.

### 1. `src/bm25_index.py` ✅ (DELETED — BM25 in Qdrant now)

#### D9 — Stop words filtering ✅
- `STOP_WORDS` constant added: standard English + basic Russian stop words
- `_tokenize()`: accepts `remove_stopwords=True` param, filters out stop words
- Backward compatible (default on)

#### D1 — Consistency helpers ✅
- `get_all_doc_ids() -> set[str]` method added

---

### 2. `src/graph_store.py` ✅

#### D3 — Bidirectional BFS in `get_related()` ✅
- BFS traverses BOTH outgoing edges (`source == current`) and incoming edges (`target == current`)
- `direction` field in each result tuple: `"out"` for source→target, `"in"` for target→source
- Return format: `list[tuple[str, str, str, float, str]]` (5-tuple, last element is direction)
- Parameter: `direction: str = "both"` ("out" | "in" | "both")

#### D2 — Graph validation helpers ✅
- `get_all_node_ids() -> set[str]` 
- `remove_phantom_edges(valid_ids: set[str]) -> int`
- `remove_phantom_nodes(valid_ids: set[str]) -> int`

#### D8 — Auto-extract relations from text ⏳
- Deferred; D3 fixes the most critical graph issues

---

### 3. `src/vector_store.py` ✅

#### D10 — Metadata normalization ✅
- `get_by_id()`: `meta` always `{}` not `None`
- `list_documents()`: `meta` always `{}` not `None`
- `search()`: `meta` always `{}` not `None`
- `get_all()`: `meta` always `{}` not `None`
- Extra: `rag.py` `search()` and `bm25_search()` also normalize metadata

#### D1 — Consistency helpers ✅
- `get_all_ids() -> set[str]` method added

---

### 4. `src/rag.py` — Main orchestrator changes ✅

#### D7 — RRF instead of linear combination in `search_hybrid()` ✅
RRF implemented with `RRF_K=20`, alpha weighting, candidate expansion.
Key design detail: documents found by only one channel get full reciprocal rank (no dilution by alpha).
See `src/rag.py:216-318`.

#### D5 — `min_content_length` guard ✅
`MIN_CONTENT_LENGTH = 40`, `ValueError` in `add_document()`.

#### D6 — Dedup via content hash ✅
SHA256 of normalized text (lowercased/stripped/whitespace-collapsed).
`_content_hashes` dict maintained, rebuilt on `_sync_stores()` and `reindex()`.

#### D1/D2 — Store synchronization on init ✅
`_sync_stores()` called from `__init__`, uses vector store as source of truth.

#### D10 — Metadata normalization in RAGSystem ✅
Normalized in `get_document()`, `list_documents()`, `search()`, `bm25_search()`.

---

### 5. `src/mcp_server.py` ✅

#### D7 — Update `rag_search_hybrid` description ✅
Description now reflects RRF formula.

#### D10 — Meta normalization in `_fmt()` ✅
`metadata: r[3] if isinstance(r[3], dict) else {}` ensures always dict.

---

### 6. Test Data Cleanup (D5 part 2) ✅

4 "full check" documents deleted via `rag_delete_document` MCP calls:
- `0d4985b7...` — «full check: empty string meta»
- `3643812b...` — «full check: null meta»
- `c9d5a1ec...` — «full check: dict meta»
- `c1941e5a...` — «full check: json string meta»

---

## Test verification plan ✅

### Tests implemented:

1. `tests/test_rag_v2.py` ✅:
   - `test_add_document_too_short` — D5: verify ValueError for short docs
   - `test_add_document_duplicate` — D6: verify same hash = same doc_id
   - `test_rrf_no_ties` — D7: RRF eliminates ties in top-3
   - `test_store_sync_on_init` — D1: phantom docs cleaned on init
   - `test_metadata_always_dict_in_get_document` — D10: metadata always dict
   - `test_metadata_always_dict_in_list_documents` — D10: metadata always dict
   - `test_metadata_always_dict_in_search` — D10: metadata always dict across all search methods

2. `tests/test_graph_store.py` ✅:
   - `test_get_related_bidirectional` — D3: bidirectional BFS finds both in/out edges
   - `test_get_related_default_is_both` — D3: default direction is "both"
   - `test_get_related_depth2_bidirectional` — D3: BFS depth 2 across directions
   - `test_get_related_unknown_node_returns_empty` — regression guard
   - `test_get_related_single_node_no_edges` — edge case guard

3. `tests/test_bm25.py` ✅:
   - `test_stop_words_filtered_from_query` — D9: all-stop-word query returns []
   - `test_stop_words_filtered_from_documents` — D9: stop words not indexed
   - `test_stop_words_disabled` — D9: remove_stopwords=False works
   - `test_stop_words_contains_russian` — D9: Russian stop words included

### Test fixes:

4. `tests/test_search_quality.py` ✅:
   - `test_hybrid_beats_pure_bm25_on_cross_lingual` — fixed: uses `rag.bm25_search()` for pure BM25 comparison
   - `test_hybrid_ndcg_beats_pure_semantic_on_full_query_set` — fixed: uses `rag.bm25_search()` for pure BM25 NDCG

---

## D3 Return format change ✅

`get_related()` now returns 5-element tuples:
`[(source, target, relation, weight, direction), ...]`

All layers updated: `GraphKnowledgeBase.get_related()`, `RAGSystem.get_related()`, `mcp_server.py` handler.
