// Package indexer handles structured indexing: Repomix-style JSON payloads
// are split into chunks with sibling relationships between chunks of the same file.
package indexer

import (
	"encoding/json"
	"fmt"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// StructuredIndexer processes Repomix JSON and indexes file chunks.
type StructuredIndexer struct {
	rag DocStoreAdder
}

// DocStoreAdder is the subset of RAGSystem needed for indexing.
type DocStoreAdder interface {
	AddDocument(text string, meta map[string]any, extractGraph bool) (ragtypes.DocID, error)
	AddRelation(source, target ragtypes.DocID, relation string, weight float64) error
}

// New creates a structured indexer.
func New(rag DocStoreAdder) *StructuredIndexer {
	return &StructuredIndexer{rag: rag}
}

// repomixFile mirrors a single file entry in a Repomix JSON payload.
type repomixFile struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// repomixPayload mirrors the top-level Repomix JSON structure.
type repomixPayload struct {
	Repository string        `json:"repository"`
	Files      []repomixFile `json:"files"`
}

// Index parses a Repomix JSON string and indexes all files as chunked documents
// with sibling links between chunks of the same file.
func (si *StructuredIndexer) Index(content string, extractGraph bool) (ragtypes.StructuredResult, error) {
	var payload repomixPayload
	if err := json.Unmarshal([]byte(content), &payload); err != nil {
		return ragtypes.StructuredResult{}, fmt.Errorf("indexer: invalid repomix JSON: %w", err)
	}

	result := ragtypes.StructuredResult{
		Status:     "ok",
		FilesCount: len(payload.Files),
	}

	// Create a structure document for the repo
	repoMeta := map[string]any{
		"type":       "repository_structure",
		"repository": payload.Repository,
	}
	structureID, err := si.rag.AddDocument(
		fmt.Sprintf("Repository: %s\nFiles: %d", payload.Repository, len(payload.Files)),
		repoMeta,
		false,
	)
	if err != nil {
		return result, fmt.Errorf("indexer: structure doc: %w", err)
	}
	result.StructureDocID = structureID

	for _, file := range payload.Files {
		fileMeta := map[string]any{
			"type":     "source_file",
			"filepath": file.Path,
			"repo":     payload.Repository,
		}
		_ = fileMeta // reserved: file-level metadata for parent doc
		// Chunk large files
		chunks := chunkText(file.Content, 4000, 200)

		var chunkIDs []ragtypes.DocID
		for i, chunk := range chunks {
			chunkMeta := map[string]any{
				"type":       "source_chunk",
				"filepath":   file.Path,
				"repo":       payload.Repository,
				"chunk_index": i,
				"total_chunks": len(chunks),
			}
			docID, err := si.rag.AddDocument(chunk, chunkMeta, extractGraph)
			if err != nil {
				result.ErrorDetails = append(result.ErrorDetails, ragtypes.FileIndexError{
					Path:  file.Path,
					Error: err.Error(),
				})
				continue
			}
			chunkIDs = append(chunkIDs, docID)
			result.FileDocIDs = append(result.FileDocIDs, docID)
		}

		// Link chunks as siblings
		for i := 0; i < len(chunkIDs); i++ {
			for j := i + 1; j < len(chunkIDs); j++ {
				if err := si.rag.AddRelation(chunkIDs[i], chunkIDs[j], "sibling_chunk", 1.0); err != nil {
					// Non-fatal
					continue
				}
			}
		}
	}

	result.Errors = len(result.ErrorDetails)
	return result, nil
}

// chunkText splits text into overlapping chunks of maxSize with overlap bytes.
func chunkText(text string, maxSize, overlap int) []string {
	if len(text) <= maxSize {
		return []string{text}
	}
	var chunks []string
	start := 0
	for start < len(text) {
		end := start + maxSize
		if end > len(text) {
			end = len(text)
		}
		chunks = append(chunks, text[start:end])
		start += maxSize - overlap
	}
	return chunks
}
