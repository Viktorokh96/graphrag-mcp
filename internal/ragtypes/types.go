// Package ragtypes defines the shared domain types and interfaces for the RAG system.
// Every sub-package depends on this one; it MUST NOT import any other internal package.
package ragtypes

import "time"

// ── Domain types ────────────────────────────────────────────────────────────

// DocID is a UUIDv4 string identifying a document.
type DocID = string

// Document represents a stored text document with metadata.
type Document struct {
	ID        DocID            `json:"doc_id"`
	Text      string           `json:"text"`
	Metadata  map[string]any   `json:"metadata,omitempty"`
	CreatedAt time.Time        `json:"created_at"`
}

// SearchResult is a single ranked hit from any search method.
type SearchResult struct {
	DocID    DocID          `json:"doc_id"`
	Text     string         `json:"text"`
	Score    float64        `json:"score"`
	Metadata map[string]any `json:"metadata,omitempty"`
	Links    []Link         `json:"links,omitempty"`
}

// Link is a graph edge attached to a search result.
type Link struct {
	DocID     DocID   `json:"doc_id"`
	Relation  string  `json:"relation"`
	Weight    float64 `json:"weight"`
	Direction string  `json:"direction"` // "out" | "in"
}

// Edge is a directed, typed, weighted graph edge.
type Edge struct {
	Source   DocID   `json:"source_id"`
	Target   DocID   `json:"target_id"`
	Relation string  `json:"relation"`
	Weight   float64 `json:"weight"`
}

// Community is a detected cluster of documents.
type Community struct {
	ID      int      `json:"id"`
	Name    string   `json:"name,omitempty"`
	Size    int      `json:"size"`
	Members []DocID  `json:"members"`
}

// GraphStats holds aggregate graph information.
type GraphStats struct {
	TotalNodes    int            `json:"total_nodes"`
	TotalEdges    int            `json:"total_edges"`
	RelationTypes map[string]int `json:"relation_types"`
}

// StoreStats holds aggregate storage information.
type StoreStats struct {
	TotalDocuments int    `json:"total_documents"`
	StorePath      string `json:"store_path"`
	Dimension      int    `json:"dimension"`
	TotalNodes     int    `json:"total_nodes"`
	TotalEdges     int    `json:"total_edges"`
	RelationTypes  map[string]int `json:"relation_types"`
}

// StructuredResult is the outcome of indexing a Repomix-style JSON payload.
type StructuredResult struct {
	Status           string              `json:"status"`
	StructureDocID   string              `json:"structure_doc_id,omitempty"`
	FileDocIDs       []DocID             `json:"file_doc_ids"`
	FilesCount       int                 `json:"files_count"`
	Errors           int                 `json:"errors"`
	ErrorDetails     []FileIndexError    `json:"error_details,omitempty"`
}

// FileIndexError captures a per-file indexing failure.
type FileIndexError struct {
	Path  string `json:"path"`
	Error string `json:"error"`
}

// IndexResult is returned when adding a document.
type IndexResult struct {
	DocID     DocID `json:"doc_id"`
	Duplicate bool  `json:"duplicate"`
}

// SearchMode selects the search strategy.
type SearchMode string

const (
	SearchSemantic SearchMode = "semantic"
	SearchBM25     SearchMode = "bm25"
	SearchHybrid   SearchMode = "hybrid"
)

// ── Store interfaces ───────────────────────────────────────────────────────

// DocumentStore is the source of truth for document text and metadata.
type DocumentStore interface {
	Add(text string, meta map[string]any) (DocID, error)
	Get(docID DocID) (*Document, error)
	Update(docID DocID, text *string, meta map[string]any) error
	Delete(docID DocID) (bool, error)
	List(limit, offset int, filter map[string]any) ([]*Document, int, error)
	Count() int
	Exists(docID DocID) bool
	IsDuplicate(text string) (DocID, bool)
	Close() error
}

// VectorStore handles dense + sparse vector indexing and search.
type VectorStore interface {
	Add(docID DocID, text string, dense []float32, sparse map[string]float32) error
	SearchDense(queryVec []float32, k int, filter map[string]any) ([]SearchResult, error)
	SearchSparse(query string, k int, filter map[string]any) ([]SearchResult, error)
	Delete(docID DocID) error
	Dimension() int
	Clear() error
	Reindex(docs []*Document, denseVecs map[DocID][]float32, sparseVecs map[DocID]map[string]float32) error
	GetAllEmbeddings() (map[DocID][]float32, error)
	Close() error
}

// GraphStore manages a knowledge graph: typed, weighted, directed edges.
type GraphStore interface {
	AddEdge(source, target DocID, relation string, weight float64) error
	DeleteEdge(source, target DocID, relation string) (bool, error)
	DeleteNode(docID DocID) error
	GetRelated(nodeID DocID, maxDepth int, filter map[string]any) ([]Edge, error)
	GetEdgesBatch(docIDs []DocID) map[DocID][]Link
	Stats() GraphStats
	AllEdges() []Edge
	AllNodes() []DocID
	HasNode(docID DocID) bool
	Close() error
}

// EmbeddingProvider generates dense vector embeddings for text.
type EmbeddingProvider interface {
	Embed(text string) ([]float32, error)
	EmbedBatch(texts []string) ([][]float32, error)
	Dimension() int
	Close() error
}

// Reranker re-ranks search candidates for higher precision.
type Reranker interface {
	Rerank(query string, candidates []SearchResult, k int) ([]SearchResult, error)
	Close() error
}

// QueryExpander generates alternative query phrasings.
type QueryExpander interface {
	Expand(query string) ([]string, error)
	Close() error
}

// GraphExtractor extracts entity-relation triples from text (LLM or NER).
type GraphExtractor interface {
	ExtractAndLink(docID DocID, text string, mode string) error
	Close() error
}

// CommunityDetector finds document clusters via Leiden on k-NN + graph edges.
type CommunityDetector interface {
	Find(resolution float64, kNN int) ([]Community, error)
	SetNames(names map[int]string) error
	Get() ([]Community, error)
}
