package embedding

import (
	"os"
	"testing"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
)

func TestNewProvider_Ollama(t *testing.T) {
	if os.Getenv("OLLAMA_TEST") == "" {
		t.Skip("OLLAMA_TEST not set — skipping integration test")
	}
	cfg := &config.RAGConfig{
		EmbeddingProvider: config.ProviderOllama,
		EmbeddingModel:    "nomic-embed-text:latest",
		EmbeddingBaseURL:  "http://localhost:11434",
		EmbeddingDim:      768,
	}
	p, err := NewProvider(cfg)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}
	defer p.Close()

	vec, err := p.Embed("test")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != 768 {
		t.Fatalf("expected dim 768, got %d", len(vec))
	}
}

func TestNewProvider_OpenAICompatible(t *testing.T) {
	baseURL := os.Getenv("OPENAI_EMBED_URL")
	if baseURL == "" {
		t.Skip("OPENAI_EMBED_URL not set — skipping integration test")
	}
	cfg := &config.RAGConfig{
		EmbeddingProvider: config.ProviderOpenAICompatible,
		EmbeddingModel:    os.Getenv("OPENAI_EMBED_MODEL"),
		EmbeddingBaseURL:  baseURL,
		EmbeddingAPIKey:   os.Getenv("OPENAI_EMBED_KEY"),
		EmbeddingDim:      1536,
	}
	p, err := NewProvider(cfg)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}
	defer p.Close()

	vec, err := p.Embed("test")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) == 0 {
		t.Fatal("got empty embedding vector")
	}
}

func TestNewProvider_Unknown(t *testing.T) {
	cfg := &config.RAGConfig{
		EmbeddingProvider: "xyz-unknown",
	}
	_, err := NewProvider(cfg)
	if err == nil {
		t.Fatal("expected error for unknown provider")
	}
}
