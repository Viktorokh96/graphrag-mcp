package embedding

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// ollamaEmbedRequest is the JSON body for POST /api/embed.
type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// ollamaEmbedResponse is the JSON response from POST /api/embed.
type ollamaEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// OllamaProvider embeds text via an Ollama server at /api/embed.
type OllamaProvider struct {
	baseURL string
	model   string
	client  *http.Client
	dim     int

	mu    sync.RWMutex
	cache map[string][]float32
}

// newOllamaProvider creates a new OllamaProvider from config.
func newOllamaProvider(cfg *config.RAGConfig) *OllamaProvider {
	baseURL := cfg.EmbeddingBaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	return &OllamaProvider{
		baseURL: baseURL,
		model:   cfg.EmbeddingModel,
		client:  &http.Client{Timeout: 30 * time.Second},
		dim:     cfg.EmbeddingDim,
		cache:   make(map[string][]float32),
	}
}

// Embed generates a single embedding vector via Ollama.
func (p *OllamaProvider) Embed(text string) ([]float32, error) {
	// Check cache first.
	p.mu.RLock()
	if vec, ok := p.cache[text]; ok {
		p.mu.RUnlock()
		return vec, nil
	}
	p.mu.RUnlock()

	vecs, err := p.EmbedBatch([]string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch generates embedding vectors for multiple texts in one request.
func (p *OllamaProvider) EmbedBatch(texts []string) ([][]float32, error) {
	// Resolve cache hits.
	p.mu.RLock()
	results := make([][]float32, len(texts))
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

	if len(missTexts) == 0 {
		return results, nil
	}

	// Build request.
	body := ollamaEmbedRequest{
		Model: p.model,
		Input: missTexts,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("embedding: ollama marshal: %w", err)
	}

	url := p.baseURL + "/api/embed"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("embedding: ollama request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedding: ollama post: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding: ollama status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var oResp ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&oResp); err != nil {
		return nil, fmt.Errorf("embedding: ollama decode: %w", err)
	}

	if len(oResp.Embeddings) != len(missTexts) {
		return nil, fmt.Errorf("embedding: ollama expected %d embeddings, got %d", len(missTexts), len(oResp.Embeddings))
	}

	// Store in cache and fill results.
	p.mu.Lock()
	for i, vec := range oResp.Embeddings {
		text := missTexts[i]
		p.cache[text] = vec
		results[missIdx[i]] = vec
	}
	p.mu.Unlock()

	return results, nil
}

// Dimension returns the embedding dimension (configured, or 0 if unknown).
func (p *OllamaProvider) Dimension() int {
	return p.dim
}

// Close is a no-op for the HTTP-backed provider.
func (p *OllamaProvider) Close() error {
	return nil
}

// compile-time interface check.
var _ ragtypes.EmbeddingProvider = (*OllamaProvider)(nil)
