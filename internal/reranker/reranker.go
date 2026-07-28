// Package reranker provides Reranker implementations that re-rank search candidates.
//
// Two implementations:
//   - NoopReranker — passes candidates through unchanged, trimming to top-k.
//   - APIReranker — POSTs to a CrossEncoder-compatible HTTP endpoint.
package reranker

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// ── NoopReranker ─────────────────────────────────────────────────────────────

// NoopReranker returns candidates unchanged, respecting only the top-k limit.
type NoopReranker struct{}

// NewNoopReranker creates a NoopReranker.
func NewNoopReranker() ragtypes.Reranker {
	return &NoopReranker{}
}

// Rerank returns the input candidates sorted by their existing score (descending)
// and truncated to at most k items.
func (r *NoopReranker) Rerank(_ string, candidates []ragtypes.SearchResult, k int) ([]ragtypes.SearchResult, error) {
	if k <= 0 {
		return nil, nil
	}
	sorted := make([]ragtypes.SearchResult, len(candidates))
	copy(sorted, candidates)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Score > sorted[j].Score
	})
	if len(sorted) > k {
		sorted = sorted[:k]
	}
	return sorted, nil
}

// Close is a no-op.
func (r *NoopReranker) Close() error { return nil }

// ── APIReranker ───────────────────────────────────────────────────────────────

// apiRerankRequest is the JSON body sent to the CrossEncoder rerank endpoint.
type apiRerankRequest struct {
	Query    string   `json:"query"`
	Passages []string `json:"passages"`
	Model    string   `json:"model,omitempty"`
	APIKey   string   `json:"api_key,omitempty"`
}

// apiRerankResult is a single scored item in the API response.
type apiRerankResult struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

// apiRerankResponse is the expected JSON response from the rerank endpoint.
type apiRerankResponse struct {
	Results []apiRerankResult `json:"results"`
}

// APIReranker calls a remote CrossEncoder HTTP API to re-rank candidates.
type APIReranker struct {
	baseURL string
	model   string
	apiKey  string
	client  *http.Client
}

// NewAPIReranker creates an APIReranker that sends requests to {baseURL}/rerank.
// The model and apiKey fields are optional and may be empty strings.
func NewAPIReranker(baseURL, model, apiKey string) ragtypes.Reranker {
	return &APIReranker{
		baseURL: baseURL,
		model:   model,
		apiKey:  apiKey,
		client:  &http.Client{},
	}
}

// Rerank sends the query and candidate texts to the remote rerank endpoint,
// then re-orders and returns the top-k candidates according to the returned scores.
func (r *APIReranker) Rerank(query string, candidates []ragtypes.SearchResult, k int) ([]ragtypes.SearchResult, error) {
	if k <= 0 || len(candidates) == 0 {
		return nil, nil
	}

	passages := make([]string, len(candidates))
	for i, c := range candidates {
		passages[i] = c.Text
	}

	body := apiRerankRequest{
		Query:    query,
		Passages: passages,
		Model:    r.model,
		APIKey:   r.apiKey,
	}

	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("reranker: marshal request: %w", err)
	}

	url := r.baseURL + "/rerank"
	resp, err := r.client.Post(url, "application/json", bytes.NewReader(encoded))
	if err != nil {
		return nil, fmt.Errorf("reranker: post %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("reranker: %s returned %d: %s", url, resp.StatusCode, string(raw))
	}

	var apiResp apiRerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("reranker: decode response: %w", err)
	}

	// Build a map from original index to the new score.
	newScores := make(map[int]float64, len(apiResp.Results))
	for _, r := range apiResp.Results {
		if r.Index >= 0 && r.Index < len(candidates) {
			newScores[r.Index] = r.Score
		}
	}

	// Build results with updated scores.
	ranked := make([]ragtypes.SearchResult, len(candidates))
	for i, c := range candidates {
		ranked[i] = c
		if s, ok := newScores[i]; ok {
			ranked[i].Score = s
		}
	}

	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].Score > ranked[j].Score
	})

	if len(ranked) > k {
		ranked = ranked[:k]
	}
	return ranked, nil
}

// Close closes the underlying HTTP client's idle connections.
func (r *APIReranker) Close() error {
	r.client.CloseIdleConnections()
	return nil
}
