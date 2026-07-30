// Package embedding provides text-to-vector embedding providers.
package embedding

import (
	"fmt"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// NewProvider constructs the appropriate EmbeddingProvider from config.
// Supported providers: ollama, openai-compatible.
func NewProvider(cfg *config.RAGConfig) (ragtypes.EmbeddingProvider, error) {
	switch cfg.EmbeddingProvider {
	case config.ProviderOllama:
		return newOllamaProvider(cfg), nil
	case config.ProviderOpenAICompatible:
		return newOpenAIProvider(cfg), nil
	default:
		return nil, fmt.Errorf("embedding: unsupported provider %q — use 'ollama' or 'openai-compatible'", cfg.EmbeddingProvider)
	}
}
