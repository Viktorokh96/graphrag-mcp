// Package community detects document clusters using Leiden algorithm on
// a k-NN graph built from embedding similarity + existing graph edges.
// The Go implementation uses a simplified label-propagation approach;
// full igraph/Leiden requires CGo bindings.
package community

import (
	"math"
	"sort"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// Detector finds communities among documents.
type Detector struct {
	docs      ragtypes.DocumentStore
	graph     ragtypes.GraphStore
	embeddings map[ragtypes.DocID][]float32
	communities []ragtypes.Community
	names      map[int]string
}

// NewDetector creates a community detector.
func NewDetector(docs ragtypes.DocumentStore, graph ragtypes.GraphStore) *Detector {
	return &Detector{
		docs: docs,
		graph: graph,
		names: make(map[int]string),
	}
}

// Find builds communities from embeddings + graph edges.
// resolution controls the granularity; kNN sets the k for the neighbor graph.
func (d *Detector) Find(resolution float64, kNN int) ([]ragtypes.Community, error) {
	// Build node list from graph
	nodeIDs := d.graph.AllNodes()

	// Build k-NN graph from embedding distances
	edges := make(map[ragtypes.DocID]map[ragtypes.DocID]float64)
	for _, id := range nodeIDs {
		edges[id] = make(map[ragtypes.DocID]float64)
	}

	// Compute pairwise cosine similarity for k-NN
	for i := 0; i < len(nodeIDs); i++ {
		vecI := d.embeddings[nodeIDs[i]]
		if vecI == nil {
			continue
		}
		type neighbor struct {
			id  ragtypes.DocID
			sim float64
		}
		var neighbors []neighbor
		for j := 0; j < len(nodeIDs); j++ {
			if i == j {
				continue
			}
			vecJ := d.embeddings[nodeIDs[j]]
			if vecJ == nil {
				continue
			}
			sim := cosineSimilarity(vecI, vecJ)
			neighbors = append(neighbors, neighbor{nodeIDs[j], sim})
		}
		sort.Slice(neighbors, func(a, b int) bool {
			return neighbors[a].sim > neighbors[b].sim
		})
		for k := 0; k < kNN && k < len(neighbors); k++ {
			edges[nodeIDs[i]][neighbors[k].id] = neighbors[k].sim
		}
	}

	// Add existing graph edges with weight 1
	for _, e := range d.graph.AllEdges() {
		edges[e.Source][e.Target] = 1.0
		edges[e.Target][e.Source] = 1.0
	}

	// Simplified community detection: connected components on edges above threshold
	threshold := 1.0 / resolution
	visited := make(map[ragtypes.DocID]bool)
	var communities []ragtypes.Community
	commID := 0

	for _, id := range nodeIDs {
		if visited[id] {
			continue
		}
		// BFS
		var members []ragtypes.DocID
		queue := []ragtypes.DocID{id}
		visited[id] = true
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			members = append(members, cur)
			for neighbor, weight := range edges[cur] {
				if !visited[neighbor] && weight >= threshold {
					visited[neighbor] = true
					queue = append(queue, neighbor)
				}
			}
		}
		if len(members) > 0 {
			communities = append(communities, ragtypes.Community{
				ID:      commID,
				Members: members,
				Size:    len(members),
			})
			commID++
		}
	}

	d.communities = communities
	return communities, nil
}

// SetNames assigns human-readable names to communities.
func (d *Detector) SetNames(names map[int]string) error {
	for id, name := range names {
		d.names[id] = name
	}
	// Apply names to cached communities
	for i := range d.communities {
		if name, ok := names[d.communities[i].ID]; ok {
			d.communities[i].Name = name
		}
	}
	return nil
}

// Get returns the most recently detected communities.
func (d *Detector) Get() ([]ragtypes.Community, error) {
	return d.communities, nil
}

// SetEmbeddings provides the dense vectors for k-NN graph construction.
func (d *Detector) SetEmbeddings(embs map[ragtypes.DocID][]float32) {
	d.embeddings = embs
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
