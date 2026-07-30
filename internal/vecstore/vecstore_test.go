// Package vecstore tests for QdrantVecStore.
//
// These tests use a Qdrant gRPC endpoint. Set QDRANT_TEST_URL env (default
// "127.0.0.1:6334") to point at a running Qdrant instance. Tests are skipped
// when the endpoint is unreachable.
package vecstore

import (
	"context"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

func testConfig(t *testing.T) *config.RAGConfig {
	t.Helper()
	url := os.Getenv("QDRANT_TEST_URL")
	if url == "" {
		url = "127.0.0.1:6334"
	}
	return &config.RAGConfig{
		StorePath:    t.TempDir(),
		QdrantURL:    url,
		EmbeddingDim: 128,
	}
}

func newTestStore(t *testing.T) *QdrantVecStore {
	t.Helper()
	cfg := testConfig(t)
	store, err := NewQdrantVecStore(cfg)
	if err != nil {
		t.Skipf("Qdrant not available at %s: %v", cfg.QdrantURL, err)
	}
	t.Cleanup(func() { store.Close() })
	t.Cleanup(func() { store.Clear() })
	return store.(*QdrantVecStore)
}

func TestAddAndSearchDense(t *testing.T) {
	s := newTestStore(t)
	_ = s.Clear()

	docID := "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
	err := s.Add(docID, "hello world", genVec(128, 1.0), nil)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	results, err := s.SearchDense(genVec(128, 1.0), 10, nil)
	if err != nil {
		t.Fatalf("SearchDense: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least 1 result")
	}
	if results[0].DocID != docID {
		t.Errorf("expected doc_id=%q, got %q", docID, results[0].DocID)
	}
	if results[0].Text != "hello world" {
		t.Errorf("expected text=%q, got %q", "hello world", results[0].Text)
	}
	if results[0].Score <= 0 {
		t.Errorf("expected positive score, got %f", results[0].Score)
	}
}

func TestAddWithSparse(t *testing.T) {
	s := newTestStore(t)
	_ = s.Clear()

	docID := "b2c3d4e5-f6a7-8901-bcde-f12345678901"
	sparse := map[string]float32{"foo": 0.8, "bar": 0.5, "baz": 0.3}

	err := s.Add(docID, "sparse test", genVec(128, 0.5), sparse)
	if err != nil {
		t.Fatalf("Add with sparse: %v", err)
	}

	// SearchSparse with matching query terms.
	results, err := s.SearchSparse("foo bar", 10, nil)
	if err != nil {
		t.Fatalf("SearchSparse: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least 1 sparse result")
	}
}

func TestSearchWithFilter(t *testing.T) {
	s := newTestStore(t)
	_ = s.Clear()

	ids := []string{
		"c3d4e5f6-a7b8-9012-cdef-123456789012",
		"d4e5f6a7-b8c9-0123-defa-234567890123",
	}

	for i, id := range ids {
		err := s.Add(id, "doc "+strconv.Itoa(i), genVec(128, float32(i+1)), nil)
		if err != nil {
			t.Fatalf("Add %s: %v", id, err)
		}
	}

	// Filter by doc_id.
	results, err := s.SearchDense(genVec(128, 1.0), 10, map[string]any{"doc_id": ids[0]})
	if err != nil {
		t.Fatalf("SearchDense with filter: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 filtered result, got %d", len(results))
	}
	if results[0].DocID != ids[0] {
		t.Errorf("expected %q, got %q", ids[0], results[0].DocID)
	}
}

func TestDelete(t *testing.T) {
	s := newTestStore(t)
	_ = s.Clear()

	docID := "e5f6a7b8-c9d0-1234-efab-345678901234"
	if err := s.Add(docID, "delete me", genVec(128, 0.2), nil); err != nil {
		t.Fatalf("Add: %v", err)
	}

	if err := s.Delete(docID); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	results, err := s.SearchDense(genVec(128, 0.2), 10, nil)
	if err != nil {
		t.Fatalf("SearchDense after delete: %v", err)
	}
	for _, r := range results {
		if r.DocID == docID {
			t.Errorf("deleted point %q still returned", docID)
		}
	}
}

func TestDimension(t *testing.T) {
	s := newTestStore(t)
	if d := s.Dimension(); d != 128 {
		t.Errorf("expected 128, got %d", d)
	}
}

func TestClear(t *testing.T) {
	s := newTestStore(t)
	_ = s.Clear()

	docID := "f6a7b8c9-d0e1-2345-fabc-456789012345"
	if err := s.Add(docID, "will be cleared", genVec(128, 0.9), nil); err != nil {
		t.Fatalf("Add: %v", err)
	}

	if err := s.Clear(); err != nil {
		t.Fatalf("Clear: %v", err)
	}

	results, err := s.SearchDense(genVec(128, 0.9), 10, nil)
	if err != nil {
		t.Fatalf("SearchDense after clear: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results after clear, got %d", len(results))
	}
}

func TestGetAllEmbeddings(t *testing.T) {
	s := newTestStore(t)
	_ = s.Clear()

	ids := []string{
		"a1b2c3d4-e5f6-7890-abcd-ef1234567890",
		"b2c3d4e5-f6a7-8901-bcde-f12345678901",
	}

	for i, id := range ids {
		if err := s.Add(id, "doc", genVec(128, float32(i+1)), nil); err != nil {
			t.Fatalf("Add %s: %v", id, err)
		}
	}

	time.Sleep(200 * time.Millisecond) // allow Qdrant indexing

	embs, err := s.GetAllEmbeddings()
	if err != nil {
		t.Fatalf("GetAllEmbeddings: %v", err)
	}
	if len(embs) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(embs))
	}
	for _, id := range ids {
		vec, ok := embs[id]
		if !ok {
			t.Errorf("missing embedding for %q", id)
			continue
		}
		if len(vec) != 128 {
			t.Errorf("expected 128-dim vector for %q, got %d", id, len(vec))
		}
	}
}

func TestReindex(t *testing.T) {
	s := newTestStore(t)
	_ = s.Clear()

	docs := []*ragtypes.Document{
		{ID: "c3d4e5f6-a7b8-9012-cdef-123456789012", Text: "doc a"},
		{ID: "d4e5f6a7-b8c9-0123-defa-234567890123", Text: "doc b"},
	}
	denseVecs := map[ragtypes.DocID][]float32{
		"c3d4e5f6-a7b8-9012-cdef-123456789012": genVec(128, 0.3),
		"d4e5f6a7-b8c9-0123-defa-234567890123": genVec(128, 0.7),
	}
	sparseVecs := map[ragtypes.DocID]map[string]float32{
		"c3d4e5f6-a7b8-9012-cdef-123456789012": {"test": 0.9},
	}

	if err := s.Reindex(docs, denseVecs, sparseVecs); err != nil {
		t.Fatalf("Reindex: %v", err)
	}

	time.Sleep(200 * time.Millisecond)

	embs, err := s.GetAllEmbeddings()
	if err != nil {
		t.Fatalf("GetAllEmbeddings after reindex: %v", err)
	}
	if len(embs) != 2 {
		t.Errorf("expected 2 embeddings after reindex, got %d", len(embs))
	}
}

func TestSearchSparseNoMatch(t *testing.T) {
	s := newTestStore(t)
	_ = s.Clear()

	// No points in collection — SearchSparse should return empty.
	results, err := s.SearchSparse("something entirely unique", 10, nil)
	if err != nil {
		t.Fatalf("SearchSparse on empty: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results on empty collection, got %d", len(results))
	}
}

func TestParseQdrantURL(t *testing.T) {
	tests := []struct {
		raw    string
		host   string
		port   int
		apiKey string
		tls    bool
		err    bool
	}{
		{"127.0.0.1:6334", "127.0.0.1", 6334, "", false, false},
		{"localhost", "localhost", 6334, "", false, false},
		{"https://qdrant.example.com:6333?api_key=secret", "qdrant.example.com", 6333, "secret", true, false},
		{"http://qdrant.example.com", "qdrant.example.com", 6334, "", false, false},
		{"", "localhost", 6334, "", false, false},
	}
	for _, tt := range tests {
		host, port, apiKey, tls, err := parseQdrantURL(tt.raw)
		if tt.err {
			if err == nil {
				t.Errorf("parseQdrantURL(%q): expected error", tt.raw)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseQdrantURL(%q): unexpected error: %v", tt.raw, err)
			continue
		}
		if host != tt.host {
			t.Errorf("parseQdrantURL(%q) host = %q, want %q", tt.raw, host, tt.host)
		}
		if port != tt.port {
			t.Errorf("parseQdrantURL(%q) port = %d, want %d", tt.raw, port, tt.port)
		}
		if apiKey != tt.apiKey {
			t.Errorf("parseQdrantURL(%q) apiKey = %q, want %q", tt.raw, apiKey, tt.apiKey)
		}
		if tls != tt.tls {
			t.Errorf("parseQdrantURL(%q) useTLS = %v, want %v", tt.raw, tls, tt.tls)
		}
	}
}

func TestSparseToIndicesValues(t *testing.T) {
	sparse := map[string]float32{"hello": 0.5, "world": 0.8}
	indices, values := sparseToIndicesValues(sparse)

	if len(indices) != 2 || len(values) != 2 {
		t.Fatalf("expected 2 entries, got %d indices, %d values", len(indices), len(values))
	}
	if !sort.SliceIsSorted(indices, func(i, j int) bool { return indices[i] < indices[j] }) {
		t.Error("indices not sorted")
	}
	// Values should still sum correctly and match expected.
	valSum := values[0] + values[1]
	if valSum < 1.2 || valSum > 1.4 {
		t.Errorf("unexpected value sum: %f", valSum)
	}
}

func TestTokenizeQuery(t *testing.T) {
	tokens := tokenizeQuery("Hello World! Hello again.")
	expected := []string{"hello", "world", "again"}
	if len(tokens) != len(expected) {
		t.Fatalf("expected %d tokens, got %d: %v", len(expected), len(tokens), tokens)
	}
	for i, tok := range tokens {
		if tok != expected[i] {
			t.Errorf("token[%d] = %q, want %q", i, tok, expected[i])
		}
	}
}

func TestBuildFilter(t *testing.T) {
	f := buildFilter(map[string]any{
		"type":  "article",
		"count": int64(5),
	})
	if f == nil {
		t.Fatal("filter is nil")
	}
	if len(f.Must) != 2 {
		t.Fatalf("expected 2 conditions, got %d", len(f.Must))
	}
}

func TestClose(t *testing.T) {
	s := newTestStore(t)
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Test helpers

// genVec produces a float32 slice of length dim where every element is val.
func genVec(dim int, val float32) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = val
	}
	return v
}

// Ensure context import is used.
var _ = context.Background
