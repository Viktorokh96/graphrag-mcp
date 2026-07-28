package docstore

import (
	"testing"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// storeForTest creates a fresh in-memory SQLiteDocStore for a single test.
func storeForTest(t *testing.T) ragtypes.DocumentStore {
	t.Helper()
	s, err := NewSQLiteDocStore(":memory:")
	if err != nil {
		t.Fatalf("NewSQLiteDocStore(:memory:) = %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestNewSQLiteDocStore_File(t *testing.T) {
	s, err := NewSQLiteDocStore("/tmp/graphrag_test_store.sqlite")
	if err != nil {
		t.Fatalf("NewSQLiteDocStore(file) = %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close = %v", err)
	}
}

func TestAddAndGet(t *testing.T) {
	s := storeForTest(t)
	docID, err := s.Add("hello world", map[string]any{"key": "val"})
	if err != nil {
		t.Fatalf("Add = %v", err)
	}
	if docID == "" {
		t.Fatal("Add returned empty docID")
	}

	doc, err := s.Get(docID)
	if err != nil {
		t.Fatalf("Get(%q) = %v", docID, err)
	}
	if doc == nil {
		t.Fatal("Get returned nil document")
	}
	if doc.Text != "hello world" {
		t.Errorf("Text = %q, want %q", doc.Text, "hello world")
	}
	if doc.Metadata["key"] != "val" {
		t.Errorf("Metadata[key] = %v, want %v", doc.Metadata["key"], "val")
	}
	if doc.CreatedAt.IsZero() {
		t.Error("CreatedAt is zero")
	}
}

func TestAddWithNilMetadata(t *testing.T) {
	s := storeForTest(t)
	docID, err := s.Add("no metadata", nil)
	if err != nil {
		t.Fatalf("Add = %v", err)
	}
	doc, err := s.Get(docID)
	if err != nil {
		t.Fatalf("Get = %v", err)
	}
	if doc.Metadata == nil {
		t.Error("Get returned nil metadata after Add with nil")
	}
}

func TestAddDuplicate(t *testing.T) {
	s := storeForTest(t)
	id1, err := s.Add("duplicate me", map[string]any{"a": 1})
	if err != nil {
		t.Fatalf("first Add = %v", err)
	}
	id2, err := s.Add("duplicate me", map[string]any{"b": 2})
	if err != nil {
		t.Fatalf("second Add = %v", err)
	}
	if id2 != id1 {
		t.Errorf("duplicate Add returned id %q, want %q", id2, id1)
	}

	c1 := s.Count()
	if c1 != 1 {
		t.Errorf("Count after duplicate Add = %d, want 1", c1)
	}
}

func TestAddWhitespaceNormalizedDuplicate(t *testing.T) {
	s := storeForTest(t)
	id1, err := s.Add("hello   world", nil)
	if err != nil {
		t.Fatalf("first Add = %v", err)
	}
	id2, err := s.Add("  hello world  ", nil)
	if err != nil {
		t.Fatalf("second Add = %v", err)
	}
	if id2 != id1 {
		t.Errorf("normalized duplicate returned id %q, want %q", id2, id1)
	}
}

func TestGetNotFound(t *testing.T) {
	s := storeForTest(t)
	doc, err := s.Get("nonexistent")
	if err != nil {
		t.Fatalf("Get(nonexistent) = %v", err)
	}
	if doc != nil {
		t.Fatal("Get(nonexistent) returned non-nil document")
	}
}

func TestUpdateText(t *testing.T) {
	s := storeForTest(t)
	docID, _ := s.Add("original", nil)
	newText := "updated"
	if err := s.Update(docID, &newText, nil); err != nil {
		t.Fatalf("Update = %v", err)
	}
	doc, _ := s.Get(docID)
	if doc.Text != "updated" {
		t.Errorf("text = %q, want %q", doc.Text, "updated")
	}
}

func TestUpdateMetadata(t *testing.T) {
	s := storeForTest(t)
	docID, _ := s.Add("data", map[string]any{"old": "v1"})
	newMeta := map[string]any{"new": "v2"}
	if err := s.Update(docID, nil, newMeta); err != nil {
		t.Fatalf("Update metadata = %v", err)
	}
	doc, _ := s.Get(docID)
	if doc.Metadata["new"] != "v2" {
		t.Errorf("Metadata[new] = %v, want v2", doc.Metadata["new"])
	}
	if v, ok := doc.Metadata["old"]; ok {
		t.Errorf("stale metadata key 'old' = %v, expected removed", v)
	}
}

func TestUpdateNotFound(t *testing.T) {
	s := storeForTest(t)
	text := "anything"
	err := s.Update("missing", &text, nil)
	if err == nil {
		t.Fatal("Update on missing doc should error")
	}
}

func TestDelete(t *testing.T) {
	s := storeForTest(t)
	docID, _ := s.Add("delete me", nil)
	deleted, err := s.Delete(docID)
	if err != nil {
		t.Fatalf("Delete = %v", err)
	}
	if !deleted {
		t.Error("Delete returned false, want true")
	}
	if s.Exists(docID) {
		t.Error("Exists returned true after Delete")
	}
}

func TestDeleteNotFound(t *testing.T) {
	s := storeForTest(t)
	deleted, err := s.Delete("nonexistent")
	if err != nil {
		t.Fatalf("Delete(nonexistent) = %v", err)
	}
	if deleted {
		t.Error("Delete(nonexistent) returned true, want false")
	}
}

func TestExists(t *testing.T) {
	s := storeForTest(t)
	if s.Exists("ghost") {
		t.Error("Exists before Add returned true")
	}
	docID, _ := s.Add("exists", nil)
	if !s.Exists(docID) {
		t.Error("Exists after Add returned false")
	}
}

func TestIsDuplicate(t *testing.T) {
	s := storeForTest(t)
	id, _ := s.Add("unique text", nil)

	dupID, ok := s.IsDuplicate("unique text")
	if !ok {
		t.Fatal("IsDuplicate returned false for existing text")
	}
	if dupID != id {
		t.Errorf("IsDuplicate id = %q, want %q", dupID, id)
	}

	_, ok = s.IsDuplicate("absent text")
	if ok {
		t.Fatal("IsDuplicate returned true for absent text")
	}
}

func TestIsDuplicateNormalized(t *testing.T) {
	s := storeForTest(t)
	id, _ := s.Add("multi   space", nil)
	dupID, ok := s.IsDuplicate("multi space")
	if !ok {
		t.Fatal("IsDuplicate returned false for whitespace-normalized match")
	}
	if dupID != id {
		t.Errorf("IsDuplicate id = %q, want %q", dupID, id)
	}
}

func TestCount(t *testing.T) {
	s := storeForTest(t)
	if n := s.Count(); n != 0 {
		t.Errorf("Count = %d, want 0", n)
	}
	s.Add("a", nil)
	s.Add("b", nil)
	if n := s.Count(); n != 2 {
		t.Errorf("Count = %d, want 2", n)
	}
}

func TestListAll(t *testing.T) {
	s := storeForTest(t)
	s.Add("first", map[string]any{"order": 1})
	s.Add("second", map[string]any{"order": 2})
	s.Add("third", map[string]any{"order": 3})

	docs, total, err := s.List(10, 0, nil)
	if err != nil {
		t.Fatalf("List = %v", err)
	}
	if total != 3 {
		t.Errorf("total = %d, want 3", total)
	}
	if len(docs) != 3 {
		t.Errorf("len(docs) = %d, want 3", len(docs))
	}
}

func TestListPagination(t *testing.T) {
	s := storeForTest(t)
	s.Add("a", nil)
	s.Add("b", nil)
	s.Add("c", nil)

	docs, total, err := s.List(2, 1, nil)
	if err != nil {
		t.Fatalf("List = %v", err)
	}
	if total != 3 {
		t.Errorf("total = %d, want 3", total)
	}
	if len(docs) != 2 {
		t.Errorf("len(docs) = %d, want 2", len(docs))
	}
}

func TestListFilter(t *testing.T) {
	s := storeForTest(t)
	s.Add("alpha", map[string]any{"kind": "letter", "n": 1})
	s.Add("beta", map[string]any{"kind": "letter", "n": 2})
	s.Add("one", map[string]any{"kind": "digit", "n": 1})

	docs, total, err := s.List(10, 0, map[string]any{"kind": "letter"})
	if err != nil {
		t.Fatalf("List filtered = %v", err)
	}
	if total != 2 {
		t.Errorf("total = %d, want 2", total)
	}
	if len(docs) != 2 {
		t.Errorf("len(docs) = %d, want 2", len(docs))
	}
}

func TestListFilterNoMatch(t *testing.T) {
	s := storeForTest(t)
	s.Add("x", map[string]any{"env": "dev"})
	docs, total, err := s.List(10, 0, map[string]any{"env": "prod"})
	if err != nil {
		t.Fatalf("List non-matching filter = %v", err)
	}
	if total != 0 {
		t.Errorf("total = %d, want 0", total)
	}
	if len(docs) != 0 {
		t.Errorf("len(docs) = %d, want 0", len(docs))
	}
}

func TestListMultipleFilterConditions(t *testing.T) {
	s := storeForTest(t)
	s.Add("a", map[string]any{"kind": "letter", "n": 1})
	s.Add("b", map[string]any{"kind": "letter", "n": 2})
	s.Add("one", map[string]any{"kind": "digit", "n": 1})

	docs, total, err := s.List(10, 0, map[string]any{"kind": "letter", "n": float64(1)})
	if err != nil {
		t.Fatalf("List multi-filter = %v", err)
	}
	if total != 1 {
		t.Errorf("total = %d, want 1", total)
	}
	if len(docs) != 1 {
		t.Errorf("len(docs) = %d, want 1", len(docs))
	}
	if docs[0].Text != "a" {
		t.Errorf("doc text = %q, want %q", docs[0].Text, "a")
	}
}

func TestClose(t *testing.T) {
	s := storeForTest(t)
	if err := s.Close(); err != nil {
		t.Fatalf("Close = %v", err)
	}
}
