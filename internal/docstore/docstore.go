// Package docstore implements ragtypes.DocumentStore backed by SQLite.
package docstore

import (
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
	_ "github.com/mattn/go-sqlite3" // SQLite driver registration
)

// whitespaceRe matches one or more whitespace characters for normalization.
var whitespaceRe = regexp.MustCompile(`\s+`)

// SQLiteDocStore implements ragtypes.DocumentStore using a local SQLite database.
type SQLiteDocStore struct {
	mu sync.RWMutex
	db *sql.DB
}

// NewSQLiteDocStore opens (or creates) a SQLite database at dbPath and returns
// a ragtypes.DocumentStore backed by it. Pass ":memory:" for an in-memory DB.
func NewSQLiteDocStore(dbPath string) (ragtypes.DocumentStore, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("docstore: open db: %w", err)
	}
	// Enable WAL mode for better concurrent read performance.
	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
		return nil, fmt.Errorf("docstore: enable WAL: %w", err)
	}
	s := &SQLiteDocStore{db: db}
	if err := s.initDB(); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

// initDB creates the documents table if it does not exist.
func (s *SQLiteDocStore) initDB() error {
	const schema = `
	CREATE TABLE IF NOT EXISTS documents (
		doc_id      TEXT PRIMARY KEY,
		text        TEXT NOT NULL,
		metadata    TEXT NOT NULL DEFAULT '{}',
		content_hash TEXT UNIQUE,
		created_at  TEXT NOT NULL
	);
	`
	if _, err := s.db.Exec(schema); err != nil {
		return fmt.Errorf("docstore: create table: %w", err)
	}
	return nil
}

// ── helpers ────────────────────────────────────────────────────────────────

// normalize trims leading/trailing whitespace and collapses all interior
// whitespace runs into a single space.
func normalize(text string) string {
	s := strings.TrimSpace(text)
	s = whitespaceRe.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

// contentHash returns the hex-encoded SHA-256 digest of the normalized text.
func contentHash(text string) string {
	h := sha256.Sum256([]byte(normalize(text)))
	return hex.EncodeToString(h[:])
}

// generateUUID returns a random UUIDv4 string (no hyphens).
func generateUUID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("docstore: generate uuid: %w", err)
	}
	// Set version 4 (0100 in the 4 most significant bits of byte 6).
	b[6] = (b[6] & 0x0f) | 0x40
	// Set variant bits (10 in the 2 most significant bits of byte 8).
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:]), nil
}

// now returns the current UTC time in ISO 8601 format.
func now() string {
	return time.Now().UTC().Format(time.RFC3339)
}

// ── DocumentStore implementation ───────────────────────────────────────────

// Add stores a new document. If the normalized text already exists (content_hash
// matches), it returns the existing DocID with no error.
func (s *SQLiteDocStore) Add(text string, meta map[string]any) (ragtypes.DocID, error) {
	if meta == nil {
		meta = make(map[string]any)
	}
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return "", fmt.Errorf("docstore: marshal metadata: %w", err)
	}

	hash := contentHash(text)
	docID, err := generateUUID()
	if err != nil {
		return "", err
	}
	createdAt := now()

	s.mu.Lock()
	defer s.mu.Unlock()

	// Check for duplicate before inserting.
	var existingID string
	err = s.db.QueryRow("SELECT doc_id FROM documents WHERE content_hash = ?", hash).Scan(&existingID)
	if err == nil {
		// Duplicate found — return the existing ID.
		return existingID, nil
	}
	if err != sql.ErrNoRows {
		return "", fmt.Errorf("docstore: check duplicate: %w", err)
	}

	_, err = s.db.Exec(
		"INSERT INTO documents (doc_id, text, metadata, content_hash, created_at) VALUES (?, ?, ?, ?, ?)",
		docID, text, string(metaJSON), hash, createdAt,
	)
	if err != nil {
		return "", fmt.Errorf("docstore: insert: %w", err)
	}
	return docID, nil
}

// Get retrieves a document by ID. Returns nil, nil if not found.
func (s *SQLiteDocStore) Get(docID ragtypes.DocID) (*ragtypes.Document, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var (
		text, metaJSON, createdAt string
	)
	err := s.db.QueryRow(
		"SELECT text, metadata, created_at FROM documents WHERE doc_id = ?", docID,
	).Scan(&text, &metaJSON, &createdAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("docstore: get %q: %w", docID, err)
	}
	return scanDoc(docID, text, metaJSON, createdAt)
}

// scanDoc builds a *ragtypes.Document from raw SQLite row values.
func scanDoc(docID, text, metaJSON, createdAt string) (*ragtypes.Document, error) {
	var meta map[string]any
	if metaJSON != "" {
		if err := json.Unmarshal([]byte(metaJSON), &meta); err != nil {
			return nil, fmt.Errorf("docstore: unmarshal metadata: %w", err)
		}
	}
	if meta == nil {
		meta = make(map[string]any)
	}
	createdAtTime, err := time.Parse(time.RFC3339, createdAt)
	if err != nil {
		// Fall back to current time if parsing fails.
		createdAtTime = time.Now().UTC()
	}
	return &ragtypes.Document{
		ID:        docID,
		Text:      text,
		Metadata:  meta,
		CreatedAt: createdAtTime,
	}, nil
}

// Update modifies the text and/or metadata of an existing document. A nil text
// pointer leaves the text unchanged; a nil metadata map leaves metadata
// unchanged.
func (s *SQLiteDocStore) Update(docID ragtypes.DocID, text *string, meta map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Fetch current values first.
	var curText, curMetaJSON, curHash string
	err := s.db.QueryRow(
		"SELECT text, metadata, content_hash FROM documents WHERE doc_id = ?", docID,
	).Scan(&curText, &curMetaJSON, &curHash)
	if err == sql.ErrNoRows {
		return fmt.Errorf("docstore: update %q: not found", docID)
	}
	if err != nil {
		return fmt.Errorf("docstore: update %q: fetch: %w", docID, err)
	}

	newText := curText
	newHash := curHash
	if text != nil {
		newText = *text
		newHash = contentHash(*text)
	}

	newMetaJSON := curMetaJSON
	if meta != nil {
		b, err := json.Marshal(meta)
		if err != nil {
			return fmt.Errorf("docstore: update %q: marshal metadata: %w", docID, err)
		}
		newMetaJSON = string(b)
	}

	_, err = s.db.Exec(
		"UPDATE documents SET text = ?, metadata = ?, content_hash = ? WHERE doc_id = ?",
		newText, newMetaJSON, newHash, docID,
	)
	if err != nil {
		return fmt.Errorf("docstore: update %q: %w", docID, err)
	}
	return nil
}

// Delete removes a document by ID. Returns true if a row was deleted.
func (s *SQLiteDocStore) Delete(docID ragtypes.DocID) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	res, err := s.db.Exec("DELETE FROM documents WHERE doc_id = ?", docID)
	if err != nil {
		return false, fmt.Errorf("docstore: delete %q: %w", docID, err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("docstore: delete %q: rows affected: %w", docID, err)
	}
	return n > 0, nil
}

// List returns up to limit documents starting at offset, optionally filtered by
// metadata fields. The second return value is the total count of matching
// documents (ignoring limit/offset).
//
// Filter keys are metadata field paths — the implementation uses
// json_extract(metadata, '$.<key>') = ? for each entry.
func (s *SQLiteDocStore) List(limit, offset int, filter map[string]any) ([]*ragtypes.Document, int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	where, args := buildFilter(filter)

	// Total count of matching documents.
	countQ := "SELECT COUNT(*) FROM documents" + where
	var total int
	if err := s.db.QueryRow(countQ, args...).Scan(&total); err != nil {
		return nil, 0, fmt.Errorf("docstore: list count: %w", err)
	}

	// Paginated query.
	rowsQ := "SELECT doc_id, text, metadata, created_at FROM documents" + where +
		" ORDER BY created_at DESC LIMIT ? OFFSET ?"
	rowsArgs := append(args, limit, offset)
	rows, err := s.db.Query(rowsQ, rowsArgs...)
	if err != nil {
		return nil, 0, fmt.Errorf("docstore: list query: %w", err)
	}
	defer rows.Close()

	var docs []*ragtypes.Document
	for rows.Next() {
		var docID, text, metaJSON, createdAt string
		if err := rows.Scan(&docID, &text, &metaJSON, &createdAt); err != nil {
			return nil, 0, fmt.Errorf("docstore: list scan: %w", err)
		}
		doc, err := scanDoc(docID, text, metaJSON, createdAt)
		if err != nil {
			return nil, 0, err
		}
		docs = append(docs, doc)
	}
	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("docstore: list rows: %w", err)
	}
	return docs, total, nil
}

// buildFilter constructs the WHERE clause and argument slice from a metadata
// filter map. Each key becomes a json_extract condition. Returns " WHERE ..."
// or an empty string when filter is empty.
func buildFilter(filter map[string]any) (string, []any) {
	if len(filter) == 0 {
		return "", nil
	}
	var clauses []string
	var args []any
	for key, val := range filter {
		clauses = append(clauses, fmt.Sprintf("json_extract(metadata, '$.%s') = ?", key))
		args = append(args, val)
	}
	return " WHERE " + strings.Join(clauses, " AND "), args
}

// Count returns the total number of documents.
func (s *SQLiteDocStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var n int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM documents").Scan(&n); err != nil {
		return 0
	}
	return n
}

// Exists returns true if a document with the given ID is present.
func (s *SQLiteDocStore) Exists(docID ragtypes.DocID) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var n int
	err := s.db.QueryRow("SELECT 1 FROM documents WHERE doc_id = ?", docID).Scan(&n)
	return err == nil
}

// IsDuplicate checks whether the normalized text already exists in the store.
// If a duplicate is found it returns (docID, true); otherwise ("", false).
func (s *SQLiteDocStore) IsDuplicate(text string) (ragtypes.DocID, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	hash := contentHash(text)
	var docID string
	err := s.db.QueryRow("SELECT doc_id FROM documents WHERE content_hash = ?", hash).Scan(&docID)
	if err == sql.ErrNoRows {
		return "", false
	}
	if err != nil {
		return "", false
	}
	return docID, true
}

// Close closes the underlying database connection.
func (s *SQLiteDocStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.db.Close()
}
