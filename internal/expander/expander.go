// Package expander provides QueryExpander implementations that generate
// alternative phrasings of a search query.
//
// Two implementations:
//   - NoopExpander — returns the original query unchanged.
//   - OllamaExpander — calls a local Ollama instance to produce N variants.
package expander

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// ── NoopExpander ─────────────────────────────────────────────────────────────

// NoopExpander returns a single-element slice containing the original query.
type NoopExpander struct{}

// NewNoopExpander creates a NoopExpander.
func NewNoopExpander() ragtypes.QueryExpander {
	return &NoopExpander{}
}

// Expand returns [query].
func (e *NoopExpander) Expand(query string) ([]string, error) {
	return []string{query}, nil
}

// Close is a no-op.
func (e *NoopExpander) Close() error { return nil }

// ── OllamaExpander ───────────────────────────────────────────────────────────

// ollamaGenerateRequest is the JSON body for the Ollama /api/generate endpoint.
type ollamaGenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

// ollamaGenerateResponse is the JSON response from Ollama /api/generate when stream=false.
type ollamaGenerateResponse struct {
	Response string `json:"response"`
}

// OllamaExpander calls a local Ollama instance to generate query variants.
type OllamaExpander struct {
	baseURL     string
	model       string
	numVariants int
	client      *http.Client
}

// NewOllamaExpander creates an OllamaExpander.
// baseURL is the Ollama server address (e.g. "http://localhost:11434").
// model is the model name (e.g. "llama3").
// numVariants is the number of alternative query phrasings to generate.
func NewOllamaExpander(baseURL, model string, numVariants int) ragtypes.QueryExpander {
	if numVariants < 1 {
		numVariants = 1
	}
	return &OllamaExpander{
		baseURL:     strings.TrimRight(baseURL, "/"),
		model:       model,
		numVariants: numVariants,
		client:      &http.Client{},
	}
}

// Expand sends the query to Ollama with a prompt asking for N alternative phrasings,
// then parses the newline-separated response into individual variants.
// The original query is always prepended as the first element.
func (e *OllamaExpander) Expand(query string) ([]string, error) {
	prompt := fmt.Sprintf(
		`Generate %d alternative phrasings of the following search query. Return one variant per line, no numbering or extra text.

Query: %s`,
		e.numVariants, query,
	)

	body := ollamaGenerateRequest{
		Model:  e.model,
		Prompt: prompt,
		Stream: false,
	}

	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("expander: marshal request: %w", err)
	}

	url := e.baseURL + "/api/generate"
	resp, err := e.client.Post(url, "application/json", bytes.NewReader(encoded))
	if err != nil {
		return nil, fmt.Errorf("expander: post %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("expander: %s returned %d: %s", url, resp.StatusCode, string(raw))
	}

	var apiResp ollamaGenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("expander: decode response: %w", err)
	}

	// Parse newline-separated variants. Filter empty lines and trim whitespace.
	rawLines := strings.Split(apiResp.Response, "\n")
	variants := make([]string, 0, len(rawLines))
	for _, line := range rawLines {
		line = strings.TrimSpace(line)
		if line != "" {
			variants = append(variants, line)
		}
	}

	// If Ollama returned nothing usable, fall back to the original query.
	if len(variants) == 0 {
		return []string{query}, nil
	}

	// Cap at the requested number of variants.
	if len(variants) > e.numVariants {
		variants = variants[:e.numVariants]
	}

	// Prepend the original query.
	result := make([]string, 0, 1+len(variants))
	result = append(result, query)
	result = append(result, variants...)

	return result, nil
}

// Close closes the underlying HTTP client's idle connections.
func (e *OllamaExpander) Close() error {
	e.client.CloseIdleConnections()
	return nil
}
