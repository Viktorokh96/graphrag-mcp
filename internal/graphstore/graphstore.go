// Package graphstore implements ragtypes.GraphStore backed by SQLite with an
// in-memory adjacency cache for fast lookups.
package graphstore

import (
	"database/sql"
	"fmt"
	"sync"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// SQLGraphStore implements ragtypes.GraphStore backed by a SQLite edges table
// and an in-memory adjacency index.
type SQLGraphStore struct {
	db      *sql.DB
	mu      sync.RWMutex
	edges   map[ragtypes.DocID][]ragtypes.Edge // source -> outgoing edges
	nodeSet map[ragtypes.DocID]struct{}         // fast node existence check
	getMeta func(ragtypes.DocID) (map[string]any, bool)
}

// ensure interface compliance at compile time.
var _ ragtypes.GraphStore = (*SQLGraphStore)(nil)

// NewGraphStore opens or creates the SQLite database at dbPath, initialises the
// edges schema, loads the in-memory cache, and returns a ragtypes.GraphStore.
// getMeta is optional; when non-nil it is used by GetRelated to apply metadata
// filters on neighbours.
func NewGraphStore(db *sql.DB, getMeta func(ragtypes.DocID) (map[string]any, bool)) (ragtypes.GraphStore, error) {
	if db == nil {
		return nil, fmt.Errorf("graphstore: nil database handle")
	}

	stmt := `CREATE TABLE IF NOT EXISTS edges (
		source   TEXT NOT NULL,
		target   TEXT NOT NULL,
		relation TEXT NOT NULL,
		weight   REAL NOT NULL DEFAULT 1.0,
		PRIMARY KEY (source, target, relation)
	)`
	if _, err := db.Exec(stmt); err != nil {
		return nil, fmt.Errorf("graphstore: create edges table: %w", err)
	}

	gs := &SQLGraphStore{
		db:      db,
		edges:   make(map[ragtypes.DocID][]ragtypes.Edge),
		nodeSet: make(map[ragtypes.DocID]struct{}),
		getMeta: getMeta,
	}

	if err := gs.loadCache(); err != nil {
		return nil, fmt.Errorf("graphstore: load cache: %w", err)
	}
	return gs, nil
}

// loadCache reads all edges from the database into the in-memory adjacency
// index and node set.
func (gs *SQLGraphStore) loadCache() error {
	gs.mu.Lock()
	defer gs.mu.Unlock()

	rows, err := gs.db.Query(`SELECT source, target, relation, weight FROM edges`)
	if err != nil {
		return err
	}
	defer rows.Close()

	clear(gs.edges)
	clear(gs.nodeSet)

	for rows.Next() {
		var e ragtypes.Edge
		if err := rows.Scan(&e.Source, &e.Target, &e.Relation, &e.Weight); err != nil {
			return err
		}
		gs.edges[e.Source] = append(gs.edges[e.Source], e)
		gs.nodeSet[e.Source] = struct{}{}
		gs.nodeSet[e.Target] = struct{}{}
	}
	return rows.Err()
}

// ── GraphStore implementation ──────────────────────────────────────────────

func (gs *SQLGraphStore) AddEdge(source, target ragtypes.DocID, relation string, weight float64) error {
	if source == "" || target == "" || relation == "" {
		return fmt.Errorf("graphstore: source, target and relation must be non-empty")
	}

	_, err := gs.db.Exec(
		`INSERT INTO edges (source, target, relation, weight) VALUES (?, ?, ?, ?)
		 ON CONFLICT(source, target, relation) DO UPDATE SET weight = ?`,
		source, target, relation, weight, weight,
	)
	if err != nil {
		return fmt.Errorf("graphstore: add edge: %w", err)
	}

	gs.mu.Lock()
	// upsert in cache — update weight if edge exists, else append
	edges := gs.edges[source]
	found := false
	for i := range edges {
		if edges[i].Target == target && edges[i].Relation == relation {
			edges[i].Weight = weight
			found = true
			break
		}
	}
	if !found {
		gs.edges[source] = append(gs.edges[source], ragtypes.Edge{
			Source: source, Target: target, Relation: relation, Weight: weight,
		})
	}
	gs.nodeSet[source] = struct{}{}
	gs.nodeSet[target] = struct{}{}
	gs.mu.Unlock()
	return nil
}

func (gs *SQLGraphStore) DeleteEdge(source, target ragtypes.DocID, relation string) (bool, error) {
	res, err := gs.db.Exec(
		`DELETE FROM edges WHERE source = ? AND target = ? AND relation = ?`,
		source, target, relation,
	)
	if err != nil {
		return false, fmt.Errorf("graphstore: delete edge: %w", err)
	}
	n, _ := res.RowsAffected()
	deleted := n > 0

	if deleted {
		gs.mu.Lock()
		gs.removeEdgeFromCache(source, target, relation)
		gs.mu.Unlock()
	}
	return deleted, nil
}

// removeEdgeFromCache removes a single edge from the in-memory adjacency
// index. Caller must hold gs.mu write lock.
func (gs *SQLGraphStore) removeEdgeFromCache(source, target, relation string) {
	edges := gs.edges[source]
	for i, e := range edges {
		if e.Source == source && e.Target == target && e.Relation == relation {
			gs.edges[source] = append(edges[:i], edges[i+1:]...)
			break
		}
	}
	// shrink nodeSet if source is now orphaned (no incoming/outgoing edges remain)
	if len(gs.edges[source]) == 0 {
		delete(gs.edges, source)
		gs.rebuildNodeSet()
	}
}

// rebuildNodeSet reconstructs the node set from edges. Caller must hold gs.mu
// write lock.
func (gs *SQLGraphStore) rebuildNodeSet() {
	clear(gs.nodeSet)
	for _, edgeList := range gs.edges {
		for _, e := range edgeList {
			gs.nodeSet[e.Source] = struct{}{}
			gs.nodeSet[e.Target] = struct{}{}
		}
	}
}

func (gs *SQLGraphStore) DeleteNode(docID ragtypes.DocID) error {
	_, err := gs.db.Exec(
		`DELETE FROM edges WHERE source = ? OR target = ?`,
		docID, docID,
	)
	if err != nil {
		return fmt.Errorf("graphstore: delete node: %w", err)
	}

	gs.mu.Lock()
	delete(gs.edges, docID)
	// remove all edges where docID is the target
	for src, edgeList := range gs.edges {
		filtered := edgeList[:0]
		for _, e := range edgeList {
			if e.Target != docID {
				filtered = append(filtered, e)
			}
		}
		if len(filtered) == 0 {
			delete(gs.edges, src)
		} else {
			gs.edges[src] = filtered
		}
	}
	gs.rebuildNodeSet()
	gs.mu.Unlock()
	return nil
}

func (gs *SQLGraphStore) GetRelated(nodeID ragtypes.DocID, maxDepth int, filter map[string]any) ([]ragtypes.Edge, error) {
	if maxDepth < 1 {
		// special case: request depth 0 means just edges of nodeID itself
		maxDepth = 1
	}

	gs.mu.RLock()
	defer gs.mu.RUnlock()

	visited := make(map[ragtypes.DocID]int) // nodeID -> depth reached
	queue := []struct {
		id    ragtypes.DocID
		depth int
	}{{nodeID, 0}}
	visited[nodeID] = 0

	var result []ragtypes.Edge

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if current.depth >= maxDepth {
			continue
		}

		nextDepth := current.depth + 1

		// outgoing edges from current node
		for _, e := range gs.edges[current.id] {
			// only include edge if neighbour passes the metadata filter
			if !gs.passesFilter(e.Target, filter) {
				continue
			}
			result = append(result, e)

			// queue target as next level if not visited
			if _, seen := visited[e.Target]; !seen {
				visited[e.Target] = nextDepth
				queue = append(queue, struct {
					id    ragtypes.DocID
					depth int
				}{e.Target, nextDepth})
			}
		}

		// incoming edges (bidirectional): scan all edges where target == current
		for src, edgeList := range gs.edges {
			if src == current.id {
				continue // already handled as outgoing
			}
			for _, e := range edgeList {
				if e.Target != current.id {
					continue
				}
				// only consider edges to unvisited sources — if src was already
				// visited, this edge was already captured as an outgoing edge
				// from src during its own processing round.
				if _, seen := visited[src]; seen {
					continue
				}
				// only include edge if neighbour (src) passes filter
				if !gs.passesFilter(src, filter) {
					continue
				}
				// add the edge as-is (source=src, target=current)
				result = append(result, e)
				visited[src] = nextDepth
				queue = append(queue, struct {
					id    ragtypes.DocID
					depth int
				}{src, nextDepth})
			}
		}
	}

	return result, nil
}

// passesFilter checks whether a node passes the metadata filter. If filter is
// nil or empty the node always passes. Otherwise getMeta is called and every
// key in filter must exist in the node's metadata with a matching value.
func (gs *SQLGraphStore) passesFilter(nodeID ragtypes.DocID, filter map[string]any) bool {
	if len(filter) == 0 {
		return true
	}
	if gs.getMeta == nil {
		return true // no way to check, let it through
	}
	meta, ok := gs.getMeta(nodeID)
	if !ok {
		return false
	}
	for k, v := range filter {
		mv, exists := meta[k]
		if !exists || mv != v {
			return false
		}
	}
	return true
}

func (gs *SQLGraphStore) GetEdgesBatch(docIDs []ragtypes.DocID) map[ragtypes.DocID][]ragtypes.Link {
	if len(docIDs) == 0 {
		return nil
	}

	// Build a set for fast lookup
	idSet := make(map[ragtypes.DocID]struct{}, len(docIDs))
	for _, id := range docIDs {
		idSet[id] = struct{}{}
	}

	result := make(map[ragtypes.DocID][]ragtypes.Link, len(docIDs))

	gs.mu.RLock()
	defer gs.mu.RUnlock()

	for _, src := range docIDs {
		// outgoing edges
		for _, e := range gs.edges[src] {
			result[src] = append(result[src], ragtypes.Link{
				DocID:     e.Target,
				Relation:  e.Relation,
				Weight:    e.Weight,
				Direction: "out",
			})
		}
	}

	// incoming edges: scan all edges where target matches any docID
	for src, edgeList := range gs.edges {
		for _, e := range edgeList {
			if _, requested := idSet[e.Target]; requested {
				result[e.Target] = append(result[e.Target], ragtypes.Link{
					DocID:     src,
					Relation:  e.Relation,
					Weight:    e.Weight,
					Direction: "in",
				})
			}
		}
	}

	return result
}

func (gs *SQLGraphStore) Stats() ragtypes.GraphStats {
	gs.mu.RLock()
	nodeCount := len(gs.nodeSet)
	edgeCount := 0
	relTypes := make(map[string]int)
	for _, edgeList := range gs.edges {
		for _, e := range edgeList {
			edgeCount++
			relTypes[e.Relation]++
		}
	}
	gs.mu.RUnlock()

	return ragtypes.GraphStats{
		TotalNodes:    nodeCount,
		TotalEdges:    edgeCount,
		RelationTypes: relTypes,
	}
}

func (gs *SQLGraphStore) AllEdges() []ragtypes.Edge {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	total := 0
	for _, edgeList := range gs.edges {
		total += len(edgeList)
	}

	result := make([]ragtypes.Edge, 0, total)
	for _, edgeList := range gs.edges {
		result = append(result, edgeList...)
	}
	return result
}

func (gs *SQLGraphStore) AllNodes() []ragtypes.DocID {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	nodes := make([]ragtypes.DocID, 0, len(gs.nodeSet))
	for id := range gs.nodeSet {
		nodes = append(nodes, id)
	}
	return nodes
}

func (gs *SQLGraphStore) HasNode(docID ragtypes.DocID) bool {
	gs.mu.RLock()
	defer gs.mu.RUnlock()
	_, ok := gs.nodeSet[docID]
	return ok
}

func (gs *SQLGraphStore) Close() error {
	gs.mu.Lock()
	defer gs.mu.Unlock()

	clear(gs.edges)
	clear(gs.nodeSet)
	return gs.db.Close()
}
