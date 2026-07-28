// Package migrate handles migration from legacy ChromaDB + graph_index.json
// to the new Qdrant + SQLite stores. The Go implementation provides the
// structure but delegates heavy lifting to the Python version for now.
package migrate

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

const oldGraphFile = "graph_index.json"

// OldGraph represents the legacy graph_index.json format.
type OldGraph struct {
	Nodes map[ragtypes.DocID]OldNode `json:"nodes"`
	Edges map[string]OldEdge         `json:"edges"`
}

// OldNode is a legacy graph node.
type OldNode struct {
	Text     string         `json:"text"`
	Metadata map[string]any `json:"metadata"`
}

// OldEdge is a legacy graph edge.
type OldEdge struct {
	Source   ragtypes.DocID `json:"source"`
	Target   ragtypes.DocID `json:"target"`
	Relation string         `json:"relation"`
	Weight   float64        `json:"weight"`
}

// HasOldData checks whether the store path contains legacy ChromaDB or graph data.
func HasOldData(storePath string) bool {
	chromaDir := filepath.Join(storePath, "chroma.sqlite3")
	if _, err := os.Stat(chromaDir); err == nil {
		return true
	}
	graphFile := filepath.Join(storePath, oldGraphFile)
	if _, err := os.Stat(graphFile); err == nil {
		return true
	}
	return false
}

// ReadOldGraph loads the legacy graph_index.json file.
func ReadOldGraph(storePath string) (*OldGraph, error) {
	path := filepath.Join(storePath, oldGraphFile)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &OldGraph{Nodes: map[ragtypes.DocID]OldNode{}, Edges: map[string]OldEdge{}}, nil
		}
		return nil, fmt.Errorf("migrate: cannot read %s: %w", path, err)
	}

	var g OldGraph
	if err := json.Unmarshal(data, &g); err != nil {
		return nil, fmt.Errorf("migrate: invalid %s: %w", path, err)
	}
	if g.Nodes == nil {
		g.Nodes = map[ragtypes.DocID]OldNode{}
	}
	if g.Edges == nil {
		g.Edges = map[string]OldEdge{}
	}
	return &g, nil
}

// RunStats holds migration result counters.
type RunStats struct {
	Documents        int `json:"documents"`
	Edges            int `json:"edges"`
	SkippedDocuments int `json:"skipped_documents"`
	SkippedEdges     int `json:"skipped_edges"`
}

// Run performs the full migration: reads legacy data, writes to new stores.
// For now, this is a stub — full ChromaDB reading requires the Python client.
func Run(storePath string, force bool) (*RunStats, error) {
	if !HasOldData(storePath) {
		return nil, fmt.Errorf("migrate: no legacy data found in %s", storePath)
	}

	// Read old graph
	oldGraph, err := ReadOldGraph(storePath)
	if err != nil {
		return nil, err
	}

	stats := &RunStats{
		Documents: len(oldGraph.Nodes),
		Edges:     len(oldGraph.Edges),
	}

	// TODO: Read ChromaDB documents (requires chromadb Go client or Python bridge)
	// For now, only graph edges can be migrated

	return stats, nil
}
