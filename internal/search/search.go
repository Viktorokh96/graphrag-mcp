// Package search implements the search orchestrator — the central coordinator
// that wires together all RAG components (document store, vector store, graph,
// embedding, reranking, query expansion, graph extraction, community detection)
// into a single Service type used by both the MCP server and HTTP API.
//
// The Service struct fulfills all top-level RAG operations:
//   - Indexing: AddDocument, AddFile, AddStructured
//   - Search (semantic, BM25/sparse, hybrid)
//   - Document CRUD: Get, List, Update, Delete
//   - Graph operations: AddRelation, DeleteRelation, GetRelated, GraphStats
//   - Community detection: FindCommunities, GetCommunities, SetCommunityNames
//
// Hybrid search uses Reciprocal Rank Fusion (RRF) with configurable alpha
// weighting based on query language (Cyrillic vs. Latin). Query expansion is
// integrated by searching each variant independently and RRF-merging the results.
// An optional reranking stage improves precision on the top candidates.
package search

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// RRF_K is the smoothing constant used in Reciprocal Rank Fusion.
const RRF_K = 20

// expandedK returns the larger of k*3 and 20 to ensure each search channel
// produces enough candidates for fusion.
func expandedK(k int) int {
	n := k * 3
	if n < 20 {
		return 20
	}
	return n
}

// ── Service ────────────────────────────────────────────────────────────────

// Service orchestrates all RAG operations by delegating to the individual
// store and provider interfaces. Every public method is safe for concurrent
// use only if the underlying stores are concurrency-safe.
type Service struct {
	docs    ragtypes.DocumentStore
	vec     ragtypes.VectorStore
	graph   ragtypes.GraphStore
	embed   ragtypes.EmbeddingProvider
	rerank  ragtypes.Reranker
	expand  ragtypes.QueryExpander
	extract ragtypes.GraphExtractor
	comm    ragtypes.CommunityDetector
	cfg     *config.RAGConfig
}

// New creates a fully wired search Service.
func New(
	docStore ragtypes.DocumentStore,
	vecStore ragtypes.VectorStore,
	graphStore ragtypes.GraphStore,
	embedProv ragtypes.EmbeddingProvider,
	reranker ragtypes.Reranker,
	expander ragtypes.QueryExpander,
	extractor ragtypes.GraphExtractor,
	commDetector ragtypes.CommunityDetector,
	cfg *config.RAGConfig,
) *Service {
	return &Service{
		docs:    docStore,
		vec:     vecStore,
		graph:   graphStore,
		embed:   embedProv,
		rerank:  reranker,
		expand:  expander,
		extract: extractor,
		comm:    commDetector,
		cfg:     cfg,
	}
}

// ── Document indexing ──────────────────────────────────────────────────────

// AddDocument indexes a text document: deduplication, storage, embedding,
// vector indexing, and optional graph extraction.
func (s *Service) AddDocument(text string, meta map[string]any) (*ragtypes.IndexResult, error) {
	// Deduplication — return existing doc ID if text is a duplicate.
	if existingID, dup := s.docs.IsDuplicate(text); dup {
		return &ragtypes.IndexResult{DocID: existingID, Duplicate: true}, nil
	}

	// Step 1: persist text + metadata.
	docID, err := s.docs.Add(text, meta)
	if err != nil {
		return nil, fmt.Errorf("search: add document to store: %w", err)
	}

	// Step 2: generate dense embedding.
	vec, err := s.embed.Embed(text)
	if err != nil {
		return nil, fmt.Errorf("search: embed document %q: %w", docID, err)
	}

	// Step 3: index the vector.
	if err := s.vec.Add(docID, text, vec, nil); err != nil {
		return nil, fmt.Errorf("search: add vector for %q: %w", docID, err)
	}

	// Step 4: optional graph extraction.
	if s.extract != nil {
		if err := s.extract.ExtractAndLink(docID, text, s.cfg.GraphExtractMode); err != nil {
			return nil, fmt.Errorf("search: extract graph from %q: %w", docID, err)
		}
	}

	return &ragtypes.IndexResult{DocID: docID, Duplicate: false}, nil
}

// AddFile reads a file from disk and indexes it via AddDocument.
func (s *Service) AddFile(filePath string) (*ragtypes.IndexResult, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("search: read file %q: %w", filePath, err)
	}

	meta := map[string]any{
		"source":    "file",
		"file_path": filePath,
		"file_name": filepath.Base(filePath),
	}
	return s.AddDocument(string(data), meta)
}

// ── Search ─────────────────────────────────────────────────────────────────

// Search dispatches to the appropriate search strategy based on mode.
func (s *Service) Search(mode ragtypes.SearchMode, query string, k int, filter map[string]any) ([]ragtypes.SearchResult, error) {
	switch mode {
	case ragtypes.SearchSemantic:
		return s.searchSemantic(query, k, filter)
	case ragtypes.SearchBM25:
		return s.searchBM25(query, k, filter)
	case ragtypes.SearchHybrid:
		return s.searchHybrid(query, k, filter)
	default:
		return s.searchHybrid(query, k, filter)
	}
}

func (s *Service) searchSemantic(query string, k int, filter map[string]any) ([]ragtypes.SearchResult, error) {
	vec, err := s.embed.Embed(query)
	if err != nil {
		return nil, fmt.Errorf("search: embed query: %w", err)
	}
	results, err := s.vec.SearchDense(vec, k, filter)
	if err != nil {
		return nil, fmt.Errorf("search: dense search: %w", err)
	}
	return s.enrichWithLinks(results), nil
}

func (s *Service) searchBM25(query string, k int, filter map[string]any) ([]ragtypes.SearchResult, error) {
	results, err := s.vec.SearchSparse(query, k, filter)
	if err != nil {
		return nil, fmt.Errorf("search: sparse search: %w", err)
	}
	return s.enrichWithLinks(results), nil
}

// searchHybrid performs hybrid search combining dense and sparse vectors with
// optional query expansion and reranking.
//
// Flow:
//  1. Language detection → alpha (blend factor).
//  2. Optional query expansion — each variant is independently searched and
//     the results are RRF-merged.
//  3. RRF fusion of dense and sparse candidates (expandedK from each channel).
//  4. Optional reranking via the configured reranker.
//  5. Graph-link enrichment.
func (s *Service) searchHybrid(query string, k int, filter map[string]any) ([]ragtypes.SearchResult, error) {
	// ── 1. Build query variants ─────────────────────────────────────────
	variants := []string{query}
	if s.expand != nil && s.cfg.QueryExpansionEnabled {
		expanded, err := s.expand.Expand(query)
		if err != nil {
			return nil, fmt.Errorf("search: expand query: %w", err)
		}
		variants = append(variants, expanded...)
	}

	// ── 2. Search each variant ──────────────────────────────────────────
	expK := expandedK(k)
	var variantResultSets [][]ragtypes.SearchResult

	for _, v := range variants {
		// Embed the variant for dense search.
		vec, err := s.embed.Embed(v)
		if err != nil {
			return nil, fmt.Errorf("search: embed variant %q: %w", v, err)
		}

		dense, err := s.vec.SearchDense(vec, expK, filter)
		if err != nil {
			return nil, fmt.Errorf("search: dense search for variant %q: %w", v, err)
		}

		sparse, err := s.vec.SearchSparse(v, expK, filter)
		if err != nil {
			return nil, fmt.Errorf("search: sparse search for variant %q: %w", v, err)
		}

		merged := rrfMerge([][]ragtypes.SearchResult{dense, sparse}, expK)
		variantResultSets = append(variantResultSets, merged)
	}

	// ── 3. RRF-fuse all variant result sets ─────────────────────────────
	// Use expK rather than k to give the reranker a larger pool.
	merged := rrfMerge(variantResultSets, expK)

	// ── 4. Optional reranking ───────────────────────────────────────────
	if s.rerank != nil && s.cfg.RerankerEnabled {
		if s.cfg.RerankerTopKMultiplier > 0 {
			rerankK := k * s.cfg.RerankerTopKMultiplier
			if rerankK > len(merged) {
				rerankK = len(merged)
			}
			merged = merged[:rerankK]
		}
		var err error
		merged, err = s.rerank.Rerank(query, merged, k)
		if err != nil {
			return nil, fmt.Errorf("search: rerank: %w", err)
		}
	} else if len(merged) > k {
		merged = merged[:k]
	}

	// ── 5. Enrich with graph links ──────────────────────────────────────
	return s.enrichWithLinks(merged), nil
}

// ── Graph operations ────────────────────────────────────────────────────────

// AddRelation creates a directed, typed edge in the knowledge graph.
func (s *Service) AddRelation(source, target, relation string, weight float64) error {
	return s.graph.AddEdge(source, target, relation, weight)
}

// DeleteRelation removes a matching edge from the knowledge graph.
func (s *Service) DeleteRelation(source, target, relation string) (bool, error) {
	return s.graph.DeleteEdge(source, target, relation)
}

// GetRelated returns edges from the graph for a given node up to maxDepth.
func (s *Service) GetRelated(nodeID ragtypes.DocID, maxDepth int, filter map[string]any) ([]ragtypes.Edge, error) {
	return s.graph.GetRelated(nodeID, maxDepth, filter)
}

// GraphStats returns aggregate graph statistics.
func (s *Service) GraphStats() ragtypes.GraphStats {
	return s.graph.Stats()
}

// ExtractGraph runs graph extraction (LLM or NER) on a standalone text,
// linking extracted entities to the given document.
func (s *Service) ExtractGraph(docID ragtypes.DocID, text string, mode string) error {
	if s.extract == nil {
		return fmt.Errorf("search: graph extractor not configured")
	}
	return s.extract.ExtractAndLink(docID, text, mode)
}

// ── Graph data export ─────────────────────────────────────────────────────

// GraphData returns all edges and all node IDs for visualization or export.
func (s *Service) GraphData() ([]ragtypes.Edge, []ragtypes.DocID, error) {
	edges := s.graph.AllEdges()
	nodes := s.graph.AllNodes()
	return edges, nodes, nil
}

// ── Document CRUD ──────────────────────────────────────────────────────────

// ListDocuments returns a paginated, optionally filtered list of documents.
func (s *Service) ListDocuments(limit, offset int, filter map[string]any) ([]*ragtypes.Document, int, error) {
	docs, total, err := s.docs.List(limit, offset, filter)
	if err != nil {
		return nil, 0, err
	}
	s.enrichDocs(docs)
	return docs, total, nil
}

func (s *Service) GetDocument(docID ragtypes.DocID) (*ragtypes.Document, error) {
	doc, err := s.docs.Get(docID)
	if err != nil || doc == nil {
		return doc, err
	}
	s.enrichDocs([]*ragtypes.Document{doc})
	return doc, nil
}

// UpdateDocument updates text and/or metadata for an existing document.
func (s *Service) UpdateDocument(docID ragtypes.DocID, text *string, metadata map[string]any) error {
	return s.docs.Update(docID, text, metadata)
}

// DeleteDocument removes a document from all stores — doc store, vector store,
// and graph store (orphan edges / node).
func (s *Service) DeleteDocument(docID ragtypes.DocID) (bool, error) {
	removed, err := s.docs.Delete(docID)
	if err != nil {
		return false, fmt.Errorf("search: delete doc %q: %w", docID, err)
	}
	if !removed {
		return false, nil
	}

	// Best-effort cleanup of vector and graph entries.
	if err := s.vec.Delete(docID); err != nil {
		return false, fmt.Errorf("search: delete vec for %q: %w", docID, err)
	}
	if err := s.graph.DeleteNode(docID); err != nil {
		return false, fmt.Errorf("search: delete graph node %q: %w", docID, err)
	}
	return true, nil
}

// ── Health ─────────────────────────────────────────────────────────────────

// Health returns a lightweight status map for each subsystem.
func (s *Service) Health() map[string]any {
	h := map[string]any{
		"docstore":   "ok",
		"vecstore":   "ok",
		"graphstore": "ok",
		"embedding":  "ok",
		"documents":  s.docs.Count(),
		"dimension":  s.vec.Dimension(),
	}
	gs := s.graph.Stats()
	h["graph_nodes"] = gs.TotalNodes
	h["graph_edges"] = gs.TotalEdges
	return h
}

// ── Aggregated stats ───────────────────────────────────────────────────────

// Stats combines document, vector, and graph statistics into a single summary.
func (s *Service) Stats() ragtypes.StoreStats {
	gs := s.graph.Stats()
	return ragtypes.StoreStats{
		TotalDocuments: s.docs.Count(),
		StorePath:      s.cfg.StorePath,
		Dimension:      s.vec.Dimension(),
		TotalNodes:     gs.TotalNodes,
		TotalEdges:     gs.TotalEdges,
		RelationTypes:  gs.RelationTypes,
	}
}

// Clear removes all data from every store.
func (s *Service) Clear() error {
	if err := s.graph.Close(); err != nil {
		return fmt.Errorf("search: close graph: %w", err)
	}
	if err := s.vec.Clear(); err != nil {
		return fmt.Errorf("search: clear vectors: %w", err)
	}

	// Drop and re-initialise: list all documents and delete in bulk.
	// For large stores this is expensive; a truncate method on DocumentStore
	// would be more appropriate, so we accept the O(n) cost.
	all, _, err := s.docs.List(1<<30, 0, nil)
	if err != nil {
		return fmt.Errorf("search: list docs for clear: %w", err)
	}
	for _, d := range all {
		if _, err := s.docs.Delete(d.ID); err != nil {
			return fmt.Errorf("search: clear delete %q: %w", d.ID, err)
		}
	}
	return nil
}
// Reindex rebuilds all Qdrant vectors from the document store.
func (s *Service) Reindex() (int, error) {
	docs, count, err := s.docs.List(100000, 0, nil)
	if err != nil {
		return 0, fmt.Errorf("reindex: list docs: %w", err)
	}
	if count == 0 {
		return 0, nil
	}

	batchSize := 8
	denseVecs := make(map[ragtypes.DocID][]float32, count)
	sparseVecs := make(map[ragtypes.DocID]map[string]float32, count)

	for start := 0; start < len(docs); start += batchSize {
		end := start + batchSize
		if end > len(docs) {
			end = len(docs)
		}
		batch := docs[start:end]
		texts := make([]string, len(batch))
		for i, d := range batch {
			texts[i] = d.Text
		}
		embeddings, err := s.embed.EmbedBatch(texts)
		if err != nil {
			return 0, fmt.Errorf("reindex: embed batch %d-%d: %w", start, end, err)
		}
		if len(embeddings) != len(batch) {
			return 0, fmt.Errorf("reindex: embed returned %d vectors for %d texts", len(embeddings), len(batch))
		}
		for i, d := range batch {
			denseVecs[d.ID] = embeddings[i]
			sparseVecs[d.ID] = buildSparseVector(d.Text)
		}
	}

	if err := s.vec.Reindex(docs, denseVecs, sparseVecs); err != nil {
		return 0, fmt.Errorf("reindex: vecstore: %w", err)
	}
	return count, nil
}

// ── Community detection ────────────────────────────────────────────────────

// FindCommunities runs Leiden community detection with the given parameters.
func (s *Service) FindCommunities(resolution float64, kNN int) ([]ragtypes.Community, error) {
	return s.comm.Find(resolution, kNN)
}

// SetCommunityNames assigns human-readable names to communities.
func (s *Service) SetCommunityNames(names map[int]string) error {
	return s.comm.SetNames(names)
}

// GetCommunities returns the current list of detected communities.
func (s *Service) GetCommunities() ([]ragtypes.Community, error) {
	return s.comm.Get()
}

// ── Structured indexing ────────────────────────────────────────────────────

// AddStructured indexes a Repomix-style JSON payload read from r.
// The expected format is a JSON object with a "files" array, each element
// having at least "path" and "content" fields.
func (s *Service) AddStructured(r io.Reader) (*ragtypes.StructuredResult, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("search: read structured input: %w", err)
	}

	var payload struct {
		StructureDocID string `json:"structure_doc_id,omitempty"`
		Files          []struct {
			Path    string `json:"path"`
			Content string `json:"content"`
		} `json:"files"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, fmt.Errorf("search: unmarshal structured input: %w", err)
	}

	res := &ragtypes.StructuredResult{
		Status:         "ok",
		StructureDocID: payload.StructureDocID,
		FilesCount:     len(payload.Files),
	}

	for _, f := range payload.Files {
		meta := map[string]any{
			"source":    "structured",
			"file_path": f.Path,
		}
		ir, err := s.AddDocument(f.Content, meta)
		if err != nil {
			res.Errors++
			res.ErrorDetails = append(res.ErrorDetails, ragtypes.FileIndexError{
				Path:  f.Path,
				Error: err.Error(),
			})
			continue
		}
		res.FileDocIDs = append(res.FileDocIDs, ir.DocID)
	}

	if res.Errors > 0 {
		res.Status = "partial"
	}
	return res, nil
}

// ── Internal helpers ───────────────────────────────────────────────────────

// hasCyrillic returns true when s contains any Cyrillic character (Unicode
// block U+0400–U+04FF).
func hasCyrillic(s string) bool {
	for _, r := range s {
		if unicode.In(r, unicode.Cyrillic) {
			return true
		}
	}
	return false
}

// detectAlpha returns the blending factor used internally by hybrid search.
// Cyrillic-heavy queries use a higher alpha (favour dense vectors), controlled
// via configuration.
func (s *Service) detectAlpha(query string) float64 {
	if hasCyrillic(query) {
		return s.cfg.CyrillicAlpha
	}
	return s.cfg.DefaultAlpha
}

// rrfMerge combines multiple ranked result lists using Reciprocal Rank Fusion.
// Lists may have different lengths. The returned slice is sorted by fused score
// descending and limited to k items.
func rrfMerge(sets [][]ragtypes.SearchResult, k int) []ragtypes.SearchResult {
	if len(sets) == 0 {
		return nil
	}
	if len(sets) == 1 {
		results := sets[0]
		if len(results) > k {
			results = results[:k]
		}
		return results
	}

	type entry struct {
		docID ragtypes.DocID
		text  string
		meta  map[string]any
	}

	scores := make(map[ragtypes.DocID]float64)
	items := make(map[ragtypes.DocID]entry)

	for _, results := range sets {
		for rank, r := range results {
			scores[r.DocID] += 1.0 / (float64(RRF_K) + float64(rank+1))
			if _, seen := items[r.DocID]; !seen {
				items[r.DocID] = entry{
					docID: r.DocID,
					text:  r.Text,
					meta:  r.Metadata,
				}
			}
		}
	}

	// Build slice, sort by score descending.
	merged := make([]ragtypes.SearchResult, 0, len(scores))
	for docID := range scores {
		e := items[docID]
		merged = append(merged, ragtypes.SearchResult{
			DocID:    e.docID,
			Text:     e.text,
			Score:    scores[docID],
			Metadata: e.meta,
		})
	}

	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Score > merged[j].Score
	})

	if len(merged) > k {
		merged = merged[:k]
	}
	return merged
}

// enrichWithLinks attaches graph edges to each search result by batch-fetching
// the links for all result doc IDs in one round-trip.
func (s *Service) enrichWithLinks(results []ragtypes.SearchResult) []ragtypes.SearchResult {
	if len(results) == 0 {
		return results
	}

	ids := make([]ragtypes.DocID, len(results))
	for i, r := range results {
		ids[i] = r.DocID
	}

	links := s.graph.GetEdgesBatch(ids)
	for i := range results {
		if ls, ok := links[results[i].DocID]; ok {
			results[i].Links = ls
		} else {
			results[i].Links = []ragtypes.Link{}
		}
	}
	return results
}

// enrichDocs attaches graph links to Document structs.
func (s *Service) enrichDocs(docs []*ragtypes.Document) {
	ids := make([]ragtypes.DocID, len(docs))
	for i, d := range docs {
		ids[i] = d.ID
	}
	linksMap := s.graph.GetEdgesBatch(ids)
	for _, d := range docs {
		d.Links = linksMap[d.ID]
	}
}

// roundTo rounds f to the given number of decimal places.
func roundTo(f float64, places int) float64 {
	shift := math.Pow(10, float64(places))
	return math.Round(f*shift) / shift
}

// Close releases resources held by all underlying stores and providers.
func (s *Service) Close() error {
	var errs []error
	if err := s.docs.Close(); err != nil {
		errs = append(errs, err)
	}
	if err := s.vec.Close(); err != nil {
		errs = append(errs, err)
	}
	if err := s.graph.Close(); err != nil {
		errs = append(errs, err)
	}
	if err := s.embed.Close(); err != nil {
		errs = append(errs, err)
	}
	if err := s.rerank.Close(); err != nil {
		errs = append(errs, err)
	}
	if err := s.expand.Close(); err != nil {
		errs = append(errs, err)
	}
	if len(errs) > 0 {
		return fmt.Errorf("search: close errors: %v", errs)
	}
	return nil
}

// buildSparseVector creates a simple word-frequency sparse vector.
func buildSparseVector(text string) map[string]float32 {
	words := strings.Fields(strings.ToLower(text))
	vec := make(map[string]float32, len(words))
	for _, w := range words {
		vec[w]++
	}
	var sum float32
	for _, v := range vec {
		sum += v * v
	}
	if sum > 0 {
		norm := float32(math.Sqrt(float64(sum)))
		for k := range vec {
			vec[k] /= norm
		}
	}
	return vec
}
