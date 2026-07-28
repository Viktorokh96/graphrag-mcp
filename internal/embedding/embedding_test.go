package embedding

import (
	"math"
	"testing"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
)

func TestMockProvider_Embed(t *testing.T) {
	p := newMockProvider(&config.RAGConfig{EmbeddingDim: 128})

	vec, err := p.Embed("hello")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if len(vec) != 128 {
		t.Fatalf("expected dim 128, got %d", len(vec))
	}

	// Verify unit vector (L2 norm ≈ 1).
	var sumSq float64
	for _, v := range vec {
		sumSq += float64(v) * float64(v)
	}
	norm := math.Sqrt(sumSq)
	if norm < 0.99 || norm > 1.01 {
		t.Fatalf("expected unit norm ~1, got %f", norm)
	}

	// Cache: same input returns identical vector.
	vec2, _ := p.Embed("hello")
	if len(vec2) != len(vec) {
		t.Fatal("cached vector length mismatch")
	}
	for i := range vec {
		if vec[i] != vec2[i] {
			t.Fatal("cached vector differs")
		}
	}
}

func TestMockProvider_EmbedBatch(t *testing.T) {
	p := newMockProvider(&config.RAGConfig{EmbeddingDim: 64})

	texts := []string{"a", "b", "c"}
	vecs, err := p.EmbedBatch(texts)
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}
	if len(vecs) != 3 {
		t.Fatalf("expected 3 vectors, got %d", len(vecs))
	}
	for i, vec := range vecs {
		if len(vec) != 64 {
			t.Fatalf("vec[%d] dim = %d, want 64", i, len(vec))
		}
	}

	// Repeat — cache hit path.
	vecs2, err := p.EmbedBatch(texts)
	if err != nil {
		t.Fatalf("EmbedBatch (cached) failed: %v", err)
	}
	for i := range vecs {
		for j := range vecs[i] {
			if vecs[i][j] != vecs2[i][j] {
				t.Fatalf("cached batch vec[%d][%d] differs", i, j)
			}
		}
	}
}

func TestMockProvider_Dimension(t *testing.T) {
	p := newMockProvider(&config.RAGConfig{EmbeddingDim: 256})
	if d := p.Dimension(); d != 256 {
		t.Fatalf("expected dim 256, got %d", d)
	}
}

func TestMockProvider_Close(t *testing.T) {
	p := newMockProvider(&config.RAGConfig{EmbeddingDim: 1})
	if err := p.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestNewProvider_Mock(t *testing.T) {
	for _, kind := range []config.EmbeddingProviderKind{
		config.ProviderSentenceTransformer,
		config.ProviderAnthropic,
	} {
		cfg := &config.RAGConfig{EmbeddingProvider: kind, EmbeddingDim: 8}
		p, err := NewProvider(cfg)
		if err != nil {
			t.Fatalf("NewProvider(%q): %v", kind, err)
		}
		vec, err := p.Embed("test")
		if err != nil {
			t.Fatalf("Embed with %q: %v", kind, err)
		}
		if len(vec) != 8 {
			t.Fatalf("dim mismatch for %q", kind)
		}
	}
}

func TestNewProvider_Unknown(t *testing.T) {
	cfg := &config.RAGConfig{EmbeddingProvider: "nonexistent"}
	_, err := NewProvider(cfg)
	if err == nil {
		t.Fatal("expected error for unknown provider")
	}
}
