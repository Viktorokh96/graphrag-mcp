package graphstore

import (
	"database/sql"
	"testing"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
	_ "modernc.org/sqlite"
)

// newTestDB opens an in-memory SQLite database for testing.
func newTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:?_journal_mode=WAL&cache=shared")
	if err != nil {
		t.Fatalf("open in-memory db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

// newTestStore creates a fresh SQLGraphStore from an in-memory SQLite database.
func newTestStore(t *testing.T) *SQLGraphStore {
	t.Helper()
	db := newTestDB(t)
	gs, err := NewGraphStore(db, nil)
	if err != nil {
		t.Fatalf("NewGraphStore: %v", err)
	}
	return gs.(*SQLGraphStore)
}

func TestAddEdge(t *testing.T) {
	gs := newTestStore(t)

	if err := gs.AddEdge("a", "b", "related", 1.0); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}

	if !gs.HasNode("a") {
		t.Error("expected node 'a' to exist")
	}
	if !gs.HasNode("b") {
		t.Error("expected node 'b' to exist")
	}

	edges := gs.AllEdges()
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(edges))
	}
	if edges[0].Source != "a" || edges[0].Target != "b" || edges[0].Relation != "related" || edges[0].Weight != 1.0 {
		t.Errorf("unexpected edge: %+v", edges[0])
	}
}

func TestAddEdgeDuplicateWeightUpdate(t *testing.T) {
	gs := newTestStore(t)

	_ = gs.AddEdge("a", "b", "knows", 1.0)
	_ = gs.AddEdge("a", "b", "knows", 2.5)

	edges := gs.AllEdges()
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge after upsert, got %d", len(edges))
	}
	if edges[0].Weight != 2.5 {
		t.Errorf("expected weight 2.5, got %f", edges[0].Weight)
	}
}

func TestAddEdgeEmptyFields(t *testing.T) {
	gs := newTestStore(t)

	if err := gs.AddEdge("", "b", "rel", 1.0); err == nil {
		t.Error("expected error for empty source")
	}
	if err := gs.AddEdge("a", "", "rel", 1.0); err == nil {
		t.Error("expected error for empty target")
	}
	if err := gs.AddEdge("a", "b", "", 1.0); err == nil {
		t.Error("expected error for empty relation")
	}
}

func TestDeleteEdge(t *testing.T) {
	gs := newTestStore(t)

	_ = gs.AddEdge("a", "b", "rel", 1.0)
	_ = gs.AddEdge("a", "c", "rel", 1.0)

	deleted, err := gs.DeleteEdge("a", "b", "rel")
	if err != nil {
		t.Fatalf("DeleteEdge: %v", err)
	}
	if !deleted {
		t.Error("expected edge to be deleted")
	}

	// second delete should return false
	deleted2, _ := gs.DeleteEdge("a", "b", "rel")
	if deleted2 {
		t.Error("expected second delete to return false")
	}

	edges := gs.AllEdges()
	if len(edges) != 1 {
		t.Fatalf("expected 1 remaining edge, got %d", len(edges))
	}
	if edges[0].Target != "c" {
		t.Errorf("expected remaining edge to target 'c', got %s", edges[0].Target)
	}
}

func TestDeleteNode(t *testing.T) {
	gs := newTestStore(t)

	_ = gs.AddEdge("a", "b", "rel", 1.0)
	_ = gs.AddEdge("b", "c", "rel", 1.0)
	_ = gs.AddEdge("c", "a", "rel", 1.0)

	if err := gs.DeleteNode("b"); err != nil {
		t.Fatalf("DeleteNode: %v", err)
	}

	if gs.HasNode("b") {
		t.Error("expected node 'b' to be removed")
	}

	edges := gs.AllEdges()
	for _, e := range edges {
		if e.Source == "b" || e.Target == "b" {
			t.Errorf("edge still references deleted node 'b': %+v", e)
		}
	}
}

func TestAllNodes(t *testing.T) {
	gs := newTestStore(t)

	nodes := gs.AllNodes()
	if len(nodes) != 0 {
		t.Errorf("expected 0 nodes on empty store, got %d", len(nodes))
	}

	_ = gs.AddEdge("a", "b", "rel", 1.0)
	_ = gs.AddEdge("c", "d", "rel", 1.0)

	nodes = gs.AllNodes()
	nodeSet := make(map[ragtypes.DocID]struct{}, len(nodes))
	for _, n := range nodes {
		nodeSet[n] = struct{}{}
	}

	for _, expected := range []ragtypes.DocID{"a", "b", "c", "d"} {
		if _, ok := nodeSet[expected]; !ok {
			t.Errorf("expected node %q in AllNodes", expected)
		}
	}
}

func TestHasNode(t *testing.T) {
	gs := newTestStore(t)

	if gs.HasNode("nonexistent") {
		t.Error("HasNode should return false for missing node")
	}

	_ = gs.AddEdge("x", "y", "rel", 1.0)
	if !gs.HasNode("x") {
		t.Error("HasNode should return true for existing source node")
	}
	if !gs.HasNode("y") {
		t.Error("HasNode should return true for existing target node")
	}
}

func TestGetRelatedDepth1(t *testing.T) {
	gs := newTestStore(t)

	_ = gs.AddEdge("a", "b", "knows", 1.0)
	_ = gs.AddEdge("a", "c", "knows", 1.0)
	_ = gs.AddEdge("b", "d", "knows", 1.0)

	edges, err := gs.GetRelated("a", 1, nil)
	if err != nil {
		t.Fatalf("GetRelated: %v", err)
	}

	if len(edges) != 2 {
		t.Fatalf("expected 2 edges at depth 1, got %d", len(edges))
	}
}

func TestGetRelatedDepth2(t *testing.T) {
	gs := newTestStore(t)

	_ = gs.AddEdge("a", "b", "knows", 1.0)
	_ = gs.AddEdge("b", "c", "knows", 1.0)
	_ = gs.AddEdge("c", "d", "knows", 1.0)

	edges, err := gs.GetRelated("a", 2, nil)
	if err != nil {
		t.Fatalf("GetRelated: %v", err)
	}

	// depth 1: a->b (1 edge)
	// depth 2: b->c (1 edge)
	// total: 2 edges
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges at depth 2, got %d", len(edges))
	}
}

func TestGetRelatedBidirectional(t *testing.T) {
	gs := newTestStore(t)

	// Incoming edge to 'a'
	_ = gs.AddEdge("b", "a", "reports_to", 1.0)
	// Outgoing edge from 'a'
	_ = gs.AddEdge("a", "c", "knows", 1.0)

	edges, err := gs.GetRelated("a", 1, nil)
	if err != nil {
		t.Fatalf("GetRelated: %v", err)
	}

	// Should find both b->a (incoming) and a->c (outgoing)
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges (in+out) at depth 1, got %d", len(edges))
	}
}

func TestGetRelatedMaxDepth0Becomes1(t *testing.T) {
	gs := newTestStore(t)

	_ = gs.AddEdge("a", "b", "knows", 1.0)
	_ = gs.AddEdge("b", "c", "knows", 1.0)

	edges, err := gs.GetRelated("a", 0, nil)
	if err != nil {
		t.Fatalf("GetRelated: %v", err)
	}

	// maxDepth 0 is clamped to 1
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge when maxDepth=0 is clamped to 1, got %d", len(edges))
	}
}

func TestGetRelatedWithMetadataFilter(t *testing.T) {
	db := newTestDB(t)

	meta := map[ragtypes.DocID]map[string]any{
		"b": {"type": "person"},
		"c": {"type": "location"},
		"d": {"type": "person"},
	}

	getMeta := func(id ragtypes.DocID) (map[string]any, bool) {
		m, ok := meta[id]
		return m, ok
	}

	gsRaw, err := NewGraphStore(db, getMeta)
	if err != nil {
		t.Fatalf("NewGraphStore: %v", err)
	}
	gs := gsRaw.(*SQLGraphStore)

	_ = gs.AddEdge("a", "b", "knows", 1.0)
	_ = gs.AddEdge("a", "c", "knows", 1.0)
	_ = gs.AddEdge("a", "d", "knows", 1.0)

	// Filter for "person" type nodes only
	edges, err := gs.GetRelated("a", 1, map[string]any{"type": "person"})
	if err != nil {
		t.Fatalf("GetRelated: %v", err)
	}

	if len(edges) != 2 {
		t.Fatalf("expected 2 edges (b, d are persons), got %d", len(edges))
	}
}

func TestGetRelatedWithFilterNoGetMeta(t *testing.T) {
	gs := newTestStore(t)

	_ = gs.AddEdge("a", "b", "knows", 1.0)

	// Filter without getMeta should return all edges (passesFilter returns true when getMeta is nil)
	edges, err := gs.GetRelated("a", 1, map[string]any{"type": "person"})
	if err != nil {
		t.Fatalf("GetRelated: %v", err)
	}
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge when getMeta is nil, got %d", len(edges))
	}
}

func TestGetEdgesBatch(t *testing.T) {
	gs := newTestStore(t)

	_ = gs.AddEdge("a", "b", "knows", 1.0)
	_ = gs.AddEdge("a", "c", "knows", 1.0)
	_ = gs.AddEdge("d", "a", "reports_to", 0.5)

	result := gs.GetEdgesBatch([]ragtypes.DocID{"a", "d"})

	// a has 2 outgoing + 1 incoming
	aLinks := result["a"]
	if len(aLinks) != 3 {
		t.Fatalf("expected 3 links for 'a', got %d", len(aLinks))
	}

	outCount, inCount := 0, 0
	for _, l := range aLinks {
		switch l.Direction {
		case "out":
			outCount++
		case "in":
			inCount++
		}
	}
	if outCount != 2 {
		t.Errorf("expected 2 outgoing links for 'a', got %d", outCount)
	}
	if inCount != 1 {
		t.Errorf("expected 1 incoming link for 'a', got %d", inCount)
	}

	// d has 1 outgoing
	dLinks := result["d"]
	if len(dLinks) != 1 {
		t.Fatalf("expected 1 link for 'd', got %d", len(dLinks))
	}
	if dLinks[0].Direction != "out" {
		t.Errorf("expected direction 'out', got %q", dLinks[0].Direction)
	}
}

func TestGetEdgesBatchEmpty(t *testing.T) {
	gs := newTestStore(t)
	result := gs.GetEdgesBatch(nil)
	if result != nil {
		t.Errorf("expected nil for empty input, got %v", result)
	}
	result = gs.GetEdgesBatch([]ragtypes.DocID{})
	if result != nil {
		t.Errorf("expected nil for empty input, got %v", result)
	}
}

func TestStatsEmpty(t *testing.T) {
	gs := newTestStore(t)
	stats := gs.Stats()

	if stats.TotalNodes != 0 {
		t.Errorf("expected 0 nodes, got %d", stats.TotalNodes)
	}
	if stats.TotalEdges != 0 {
		t.Errorf("expected 0 edges, got %d", stats.TotalEdges)
	}
	if len(stats.RelationTypes) != 0 {
		t.Errorf("expected 0 relation types, got %d", len(stats.RelationTypes))
	}
}

func TestStats(t *testing.T) {
	gs := newTestStore(t)

	_ = gs.AddEdge("a", "b", "knows", 1.0)
	_ = gs.AddEdge("a", "c", "knows", 1.0)
	_ = gs.AddEdge("b", "d", "reports_to", 1.0)
	_ = gs.AddEdge("c", "e", "knows", 1.0)

	stats := gs.Stats()
	if stats.TotalNodes != 5 {
		t.Errorf("expected 5 nodes, got %d", stats.TotalNodes)
	}
	if stats.TotalEdges != 4 {
		t.Errorf("expected 4 edges, got %d", stats.TotalEdges)
	}
	if stats.RelationTypes["knows"] != 3 {
		t.Errorf("expected 3 'knows' relations, got %d", stats.RelationTypes["knows"])
	}
	if stats.RelationTypes["reports_to"] != 1 {
		t.Errorf("expected 1 'reports_to' relation, got %d", stats.RelationTypes["reports_to"])
	}
}

func TestAllEdgesEmpty(t *testing.T) {
	gs := newTestStore(t)
	edges := gs.AllEdges()
	if len(edges) != 0 {
		t.Errorf("expected 0 edges, got %d", len(edges))
	}
}

func TestDoubleClose(t *testing.T) {
	gs := newTestStore(t)
	if err := gs.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	// second Close should not panic or error (sql.DB.Close is idempotent)
	if err := gs.Close(); err != nil {
		t.Fatalf("second Close should not error, got: %v", err)
	}
}

func TestNewGraphStoreNilDB(t *testing.T) {
	_, err := NewGraphStore(nil, nil)
	if err == nil {
		t.Error("expected error for nil database handle")
	}
}

func TestConcurrency(t *testing.T) {
	gs := newTestStore(t)

	done := make(chan struct{})
	go func() {
		for range 100 {
			_ = gs.AddEdge("a", "target", "concurrent", 1.0)
		}
		close(done)
	}()

	for range 100 {
		_ = gs.HasNode("target")
		_ = gs.Stats()
	}

	<-done

	stats := gs.Stats()
	if stats.TotalEdges == 0 {
		t.Error("expected some edges after concurrent operations")
	}
}
