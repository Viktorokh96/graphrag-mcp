// Package extractor implements graph extraction from text — entity-relation
// triple discovery via LLM (Ollama) or NER (spaCy). The Go implementation
// currently only supports LLM mode via Ollama API.
package extractor

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// LLMExtractor extracts graph triples from text using an LLM via Ollama API.
type LLMExtractor struct {
	baseURL   string
	model     string
	client    *http.Client
	rag       DocStoreAdder
}

// DocStoreAdder is the subset of RAGSystem needed by the extractor.
type DocStoreAdder interface {
	AddDocument(text string, meta map[string]any, extractGraph bool) (ragtypes.DocID, error)
	AddRelation(source, target ragtypes.DocID, relation string, weight float64) error
}

// NewLLMExtractor creates an LLM-based graph extractor.
func NewLLMExtractor(baseURL, model string, rag DocStoreAdder) *LLMExtractor {
	return &LLMExtractor{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		client:  &http.Client{Timeout: 120 * time.Second},
		rag:     rag,
	}
}

// ExtractAndLink sends text to the LLM, parses triples (subject | relation | object),
// creates entity documents and edges.
func (e *LLMExtractor) ExtractAndLink(docID ragtypes.DocID, text string, mode string) error {
	if mode != "llm" {
		return fmt.Errorf("extractor: unsupported mode %q (Go supports only 'llm')", mode)
	}

	triples, err := e.queryLLM(text)
	if err != nil {
		return fmt.Errorf("extractor: LLM query failed: %w", err)
	}

	for _, t := range triples {
		if t.Subject == "" || t.Relation == "" || t.Object == "" {
			continue
		}

		// Create entity docs
		subjID, err := e.ensureEntity(t.Subject)
		if err != nil {
			return err
		}
		objID, err := e.ensureEntity(t.Object)
		if err != nil {
			return err
		}

		// Link entity ↔ source document
		if err := e.rag.AddRelation(subjID, docID, "appears_in", 1.0); err != nil {
			return err
		}
		if err := e.rag.AddRelation(objID, docID, "appears_in", 1.0); err != nil {
			return err
		}

		// Link entity ↔ entity
		if err := e.rag.AddRelation(subjID, objID, t.Relation, 1.0); err != nil {
			return err
		}
	}

	return nil
}

type triple struct {
	Subject  string `json:"subject"`
	Relation string `json:"relation"`
	Object   string `json:"object"`
}

func (e *LLMExtractor) queryLLM(text string) ([]triple, error) {
	// Truncate text for the prompt
	promptText := text
	if len(promptText) > 4000 {
		promptText = promptText[:4000]
	}

	prompt := fmt.Sprintf(
		`Extract entity-relation triples from the text below. 
Output as JSON array: [{"subject":"...","relation":"...","object":"..."},...]
Only include clear, factual relationships. Skip vague connections.

Text:
%s`, promptText)

	body := map[string]any{
		"model":  e.model,
		"prompt": prompt,
		"stream": false,
	}
	bodyJSON, _ := json.Marshal(body)

	resp, err := e.client.Post(
		e.baseURL+"/api/generate",
		"application/json",
		bytes.NewReader(bodyJSON),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	// Find the JSON array in the response
	jsonStart := strings.Index(result.Response, "[")
	jsonEnd := strings.LastIndex(result.Response, "]")
	if jsonStart == -1 || jsonEnd == -1 {
		return nil, nil // no triples found
	}

	var triples []triple
	if err := json.Unmarshal([]byte(result.Response[jsonStart:jsonEnd+1]), &triples); err != nil {
		return nil, fmt.Errorf("extractor: failed to parse LLM response: %w", err)
	}

	return triples, nil
}

func (e *LLMExtractor) ensureEntity(name string) (ragtypes.DocID, error) {
	// Simple: each entity is a minimal document with name as text.
	// In production, we'd check for existing entities first.
	docID, err := e.rag.AddDocument(name, map[string]any{"type": "entity", "entity_name": name}, false)
	if err != nil {
		return "", err
	}
	return docID, nil
}

// Close is a no-op for the LLM extractor.
func (e *LLMExtractor) Close() error { return nil }
