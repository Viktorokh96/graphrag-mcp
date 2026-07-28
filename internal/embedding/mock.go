package embedding

import (
	"math"
	"math/rand/v2"
	"sync"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// MockProvider returns random unit vectors for offline development.
// Always succeeds — no network calls.
type MockProvider struct {
	dim   int
	mu    sync.RWMutex
	cache map[string][]float32
}

// newMockProvider creates a new MockProvider from config.
func newMockProvider(cfg *config.RAGConfig) *MockProvider {
	return &MockProvider{
		dim:   cfg.EmbeddingDim,
		cache: make(map[string][]float32),
	}
}

// Embed generates a random unit vector for the given text.
// Deterministic per unique input string within a session.
func (p *MockProvider) Embed(text string) ([]float32, error) {
	p.mu.RLock()
	if vec, ok := p.cache[text]; ok {
		p.mu.RUnlock()
		return vec, nil
	}
	p.mu.RUnlock()

	vec := p.randomUnitVector(p.dim)

	p.mu.Lock()
	p.cache[text] = vec
	p.mu.Unlock()

	return vec, nil
}

// EmbedBatch generates random unit vectors for all texts.
func (p *MockProvider) EmbedBatch(texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))

	p.mu.RLock()
	missIdx := make([]int, 0, len(texts))
	missTexts := make([]string, 0, len(texts))
	for i, t := range texts {
		if vec, ok := p.cache[t]; ok {
			results[i] = vec
		} else {
			missIdx = append(missIdx, i)
			missTexts = append(missTexts, t)
		}
	}
	p.mu.RUnlock()

	if len(missTexts) > 0 {
		p.mu.Lock()
		for i, t := range missTexts {
			vec := p.randomUnitVector(p.dim)
			p.cache[t] = vec
			results[missIdx[i]] = vec
		}
		p.mu.Unlock()
	}

	return results, nil
}

// Dimension returns the configured embedding dimension.
func (p *MockProvider) Dimension() int {
	return p.dim
}

// Close is a no-op.
func (p *MockProvider) Close() error {
	return nil
}

// randomUnitVector generates an L2-normalized vector of length d.
func (p *MockProvider) randomUnitVector(d int) []float32 {
	if d <= 0 {
		return nil
	}
	vec := make([]float64, d)
	var sumSq float64
	for i := range vec {
		v := rand.NormFloat64()
		vec[i] = v
		sumSq += v * v
	}
	norm := math.Sqrt(sumSq)
	if norm == 0 {
		norm = 1
	}
	out := make([]float32, d)
	for i, v := range vec {
		out[i] = float32(v / norm)
	}
	return out
}

// compile-time interface check.
var _ ragtypes.EmbeddingProvider = (*MockProvider)(nil)
