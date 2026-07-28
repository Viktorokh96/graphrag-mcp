// Package embedding provides text-to-vector embedding providers.
package embedding

import (
	"fmt"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// NewProvider constructs the appropriate EmbeddingProvider from config.
// Supported kinds: ollama, openai-compatible.
// sentence_transformer and anthropic fall back to MockProvider for offline dev.
func NewProvider(cfg *config.RAGConfig) (ragtypes.EmbeddingProvider, error) {
	switch cfg.EmbeddingProvider {
	case config.ProviderOllama:
		return newOllamaProvider(cfg), nil
	case config.ProviderOpenAICompatible:
		return newOpenAIProvider(cfg), nil
	case config.ProviderSentenceTransformer, config.ProviderAnthropic:
		return newMockProvider(cfg), nil
	default:
		return nil, fmt.Errorf("embedding: unknown provider %q", cfg.EmbeddingProvider)
	}
}
