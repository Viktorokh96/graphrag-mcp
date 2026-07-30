package embedding

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"
	"strings"
	"sync"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// openaiEmbedRequest is the JSON body for POST /v1/embeddings.
type openaiEmbedRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"` // string or []string
}

// openaiEmbedData is one element in the response data array.
type openaiEmbedData struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// openaiEmbedResponse is the JSON response from POST /v1/embeddings.
type openaiEmbedResponse struct {
	Data []openaiEmbedData `json:"data"`
}

// OpenAIProvider embeds text via an OpenAI-compatible API at /v1/embeddings.
type OpenAIProvider struct {
	baseURL string
	model   string
	apiKey  string
	client  *http.Client
	dim     int

	mu    sync.RWMutex
	cache map[string][]float32
}

// newOpenAIProvider creates a new OpenAIProvider from config.
func newOpenAIProvider(cfg *config.RAGConfig) *OpenAIProvider {
	baseURL := cfg.EmbeddingBaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}

	return &OpenAIProvider{
		baseURL: baseURL,
		model:   cfg.EmbeddingModel,
		apiKey:  cfg.EmbeddingAPIKey,
		client:  &http.Client{Timeout: 30 * time.Second},
		dim:     cfg.EmbeddingDim,
		cache:   make(map[string][]float32),
	}
}

// Embed generates a single embedding vector via the OpenAI-compatible API.
func (p *OpenAIProvider) Embed(text string) ([]float32, error) {
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
func (p *OpenAIProvider) EmbedBatch(texts []string) ([][]float32, error) {
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

	// Chunk into batches of 10 to stay under token limits.
	const batchSize = 10
	for start := 0; start < len(missTexts); start += batchSize {
		end := start + batchSize
		if end > len(missTexts) {
			end = len(missTexts)
		}
		chunk := missTexts[start:end]
		chunkIdx := missIdx[start:end]

		var input any = chunk
		if len(chunk) == 1 {
			input = chunk[0]
		}

		body := openaiEmbedRequest{Model: p.model, Input: input}
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("embedding: openai marshal: %w", err)
		}

		url := strings.TrimRight(p.baseURL, "/") + "/v1/embeddings"
		req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(raw))
		if err != nil {
			return nil, fmt.Errorf("embedding: openai request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if p.apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+p.apiKey)
		}

		resp, err := p.client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("embedding: openai post: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			return nil, fmt.Errorf("embedding: openai status %d: %s", resp.StatusCode, string(bodyBytes))
		}

		var oResp openaiEmbedResponse
		if err := json.NewDecoder(resp.Body).Decode(&oResp); err != nil {
			resp.Body.Close()
			return nil, fmt.Errorf("embedding: openai decode: %w", err)
		}
		resp.Body.Close()

		if len(oResp.Data) != len(chunk) {
			return nil, fmt.Errorf("embedding: openai expected %d embeddings, got %d", len(chunk), len(oResp.Data))
		}

		sort.Slice(oResp.Data, func(i, j int) bool { return oResp.Data[i].Index < oResp.Data[j].Index })

		p.mu.Lock()
		for i, d := range oResp.Data {
			p.cache[chunk[i]] = d.Embedding
			results[chunkIdx[i]] = d.Embedding
		}
		p.mu.Unlock()
	}

	return results, nil
}

// Dimension returns the embedding dimension (configured, or 0 if unknown).
func (p *OpenAIProvider) Dimension() int {
	return p.dim
}

// Close is a no-op for the HTTP-backed provider.
func (p *OpenAIProvider) Close() error {
	return nil
}

// compile-time interface check.
var _ ragtypes.EmbeddingProvider = (*OpenAIProvider)(nil)
