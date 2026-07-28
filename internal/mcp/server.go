// Package mcp implements a stdio-based MCP server using the official
// modelcontextprotocol/go-sdk, exposing all RAG search and management tools.
package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
	"github.com/Viktorokh96/graphrag-mcp/internal/search"
)

// Server wraps the official MCP SDK server with RAG tool handlers.
type Server struct {
	sdk *sdkmcp.Server
	svc *search.Service
}

// New creates an MCP Server backed by the given search service.
// All 19 RAG tools are registered on creation.
func New(svc *search.Service) *Server {
	impl := &sdkmcp.Implementation{
		Name:    "graphrag-mcp",
		Title:   "GraphRAG Knowledge Base",
		Version: "0.4.0-go",
	}
	srv := &Server{
		sdk: sdkmcp.NewServer(impl, nil),
		svc: svc,
	}
	srv.registerTools()
	return srv
}

// Run starts the MCP server over stdio transport. Blocks until stdin closes.
func (s *Server) Run() error {
	log.SetOutput(os.Stderr)
	log.SetPrefix("[mcp] ")
	return s.sdk.Run(context.Background(), &sdkmcp.StdioTransport{})
}

// ── Tool registration ─────────────────────────────────────────────────────

func (s *Server) registerTools() {
	// Search tools
	s.add("rag_search", "Semantic search over the knowledge base using vector embeddings.",
		toolSchema{
			"query":   prop{Type: "string", Desc: "Search query text (natural language)."},
			"k":       prop{Type: "integer", Desc: "Max results. Default 5.", Default: 5},
			"max_chars": prop{Type: "integer", Desc: "Truncate text to N chars. Default 2000.", Default: 2000},
			"metadata_filter": prop{Desc: "Metadata filter dict.", Default: nil},
		}, s.callSearchSemantic)

	s.add("rag_bm25_search", "Keyword search using BM25.",
		toolSchema{
			"query":   prop{Type: "string", Desc: "Search keywords."},
			"k":       prop{Type: "integer", Desc: "Max results.", Default: 5},
			"max_chars": prop{Type: "integer", Default: nil},
			"metadata_filter": prop{Default: nil},
		}, s.callSearchBM25)

	s.add("rag_search_hybrid", "Hybrid search (RRF).",
		toolSchema{
			"query":           prop{Type: "string", Desc: "Search query."},
			"k":               prop{Type: "integer", Default: 5},
			"alpha":           prop{Type: "number", Default: nil},
			"max_chars":       prop{Type: "integer", Default: nil},
			"metadata_filter": prop{Default: nil},
			"rerank":          prop{Type: "boolean", Default: nil},
			"query_expansion": prop{Type: "boolean", Default: nil},
		}, s.callSearchHybrid)

	// Document tools
	s.add("rag_add_document", "Add a text document to the knowledge base.",
		toolSchema{
			"text":          prop{Type: "string", Desc: "Text content to index."},
			"meta":          prop{Default: nil},
			"extract_graph": prop{Type: "boolean", Default: false},
		}, s.callAddDocument)

	s.add("rag_add_file", "Read a file from disk and index it.",
		toolSchema{
			"filepath":      prop{Type: "string", Desc: "Path to file."},
			"meta":          prop{Default: nil},
			"extract_graph": prop{Type: "boolean", Default: false},
		}, s.callAddFile)

	s.add("rag_list_documents", "List documents with pagination.",
		toolSchema{
			"limit":           prop{Type: "integer", Default: 20},
			"offset":          prop{Type: "integer", Default: 0},
			"max_chars":       prop{Type: "integer", Default: nil},
			"metadata_filter": prop{Default: nil},
		}, s.callListDocuments)

	s.add("rag_get_document", "Get a document by ID.",
		toolSchema{
			"doc_id": prop{Type: "string", Desc: "Document ID."},
			"offset": prop{Type: "integer", Default: 0},
			"limit":  prop{Type: "integer", Default: nil},
		}, s.callGetDocument)

	s.add("rag_update_document", "Update document text or metadata.",
		toolSchema{
			"doc_id": prop{Type: "string", Desc: "Document ID."},
			"text":   prop{Type: "string", Default: nil},
			"meta":   prop{Default: nil},
		}, s.callUpdateDocument)

	s.add("rag_delete_document", "Delete a document (cascading).",
		toolSchema{
			"doc_id": prop{Type: "string", Desc: "Document ID."},
		}, s.callDeleteDocument)

	// Graph tools
	s.add("rag_add_relation", "Create a graph edge.",
		toolSchema{
			"source_id": prop{Type: "string", Desc: "Source doc_id."},
			"target_id": prop{Type: "string", Desc: "Target doc_id."},
			"relation":  prop{Type: "string", Desc: "Relation type."},
			"weight":    prop{Type: "number", Default: 1.0},
		}, s.callAddRelation)

	s.add("rag_delete_relation", "Delete a graph edge.",
		toolSchema{
			"source_id": prop{Type: "string"},
			"target_id": prop{Type: "string"},
			"relation":  prop{Type: "string"},
		}, s.callDeleteRelation)

	s.add("rag_get_related", "BFS traversal from a node.",
		toolSchema{
			"node_id":         prop{Type: "string", Desc: "Starting node."},
			"max_depth":       prop{Type: "integer", Default: 1},
			"metadata_filter": prop{Default: nil},
		}, s.callGetRelated)

	s.add("rag_graph_stats", "Graph statistics.", nil, s.callGraphStats)

	// Community tools
	s.add("rag_find_communities", "Find document clusters.",
		toolSchema{
			"resolution": prop{Type: "number", Default: 1.0},
			"k_nn":       prop{Type: "integer", Default: 15},
		}, s.callFindCommunities)

	s.add("rag_set_community_names", "Set community names.",
		toolSchema{
			"names": prop{Desc: "Map of community_id → name."},
		}, s.callSetCommunityNames)

	s.add("rag_get_communities", "Get cached communities.", nil, s.callGetCommunities)

	// Management
	s.add("rag_stats", "Store statistics.", nil, s.callStats)
	s.add("rag_clear", "Delete ALL data.", nil, s.callClear)
	s.add("rag_add_structured", "Index Repomix JSON.", toolSchema{
		"content":       prop{Type: "string", Desc: "Repomix JSON string."},
		"extract_graph": prop{Type: "boolean", Default: false},
	}, s.callAddStructured)
}

func (s *Server) add(name, desc string, schema toolSchema, h func(ctx context.Context, args map[string]any) (string, error)) {
	inputSchema := buildJSONSchema(schema)
	tool := &sdkmcp.Tool{
		Name:        name,
		Description: desc,
		InputSchema: inputSchema,
	}
	s.sdk.AddTool(tool, func(ctx context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		var args map[string]any
		if len(req.Params.Arguments) > 0 {
			if err := json.Unmarshal(req.Params.Arguments, &args); err != nil {
				return &sdkmcp.CallToolResult{IsError: true, Content: []sdkmcp.Content{textContent(fmt.Sprintf("invalid arguments: %v", err))}}, nil
			}
		}
		if args == nil {
			args = map[string]any{}
		}
		text, err := h(ctx, args)
		if err != nil {
			return &sdkmcp.CallToolResult{IsError: true, Content: []sdkmcp.Content{textContent(err.Error())}}, nil
		}
		return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{textContent(text)}}, nil
	})
}

func textContent(text string) sdkmcp.Content {
	return &sdkmcp.TextContent{Text: text}
}

// ── Tool handlers ─────────────────────────────────────────────────────────

func (s *Server) callSearchSemantic(_ context.Context, args map[string]any) (string, error) {
	query := getString(args, "query")
	k := getInt(args, "k", 5)
	filter := getMap(args, "metadata_filter")
	results, err := s.svc.Search(ragtypes.SearchSemantic, query, k, filter)
	if err != nil {
		return "", err
	}
	return toJSON(results), nil
}

func (s *Server) callSearchBM25(_ context.Context, args map[string]any) (string, error) {
	query := getString(args, "query")
	k := getInt(args, "k", 5)
	filter := getMap(args, "metadata_filter")
	results, err := s.svc.Search(ragtypes.SearchBM25, query, k, filter)
	if err != nil {
		return "", err
	}
	return toJSON(results), nil
}

func (s *Server) callSearchHybrid(_ context.Context, args map[string]any) (string, error) {
	query := getString(args, "query")
	k := getInt(args, "k", 5)
	filter := getMap(args, "metadata_filter")
	results, err := s.svc.Search(ragtypes.SearchHybrid, query, k, filter)
	if err != nil {
		return "", err
	}
	return toJSON(results), nil
}

func (s *Server) callAddDocument(_ context.Context, args map[string]any) (string, error) {
	text := getString(args, "text")
	meta := getMap(args, "meta")
	result, err := s.svc.AddDocument(text, meta)
	if err != nil {
		return "", err
	}
	return toJSON(result), nil
}

func (s *Server) callAddFile(_ context.Context, args map[string]any) (string, error) {
	path := getString(args, "filepath")
	result, err := s.svc.AddFile(path)
	if err != nil {
		return "", err
	}
	return toJSON(result), nil
}

func (s *Server) callAddRelation(_ context.Context, args map[string]any) (string, error) {
	source := getString(args, "source_id")
	target := getString(args, "target_id")
	relation := getString(args, "relation")
	weight := getFloat(args, "weight", 1.0)
	if err := s.svc.AddRelation(source, target, relation, weight); err != nil {
		return "", err
	}
	return `{"status":"ok"}`, nil
}

func (s *Server) callDeleteRelation(_ context.Context, args map[string]any) (string, error) {
	source := getString(args, "source_id")
	target := getString(args, "target_id")
	relation := getString(args, "relation")
	deleted, err := s.svc.DeleteRelation(source, target, relation)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(`{"deleted":%v}`, deleted), nil
}

func (s *Server) callListDocuments(_ context.Context, args map[string]any) (string, error) {
	limit := getInt(args, "limit", 20)
	offset := getInt(args, "offset", 0)
	filter := getMap(args, "metadata_filter")
	docs, total, err := s.svc.ListDocuments(limit, offset, filter)
	if err != nil {
		return "", err
	}
	return toJSON(map[string]any{"total": total, "documents": docs}), nil
}

func (s *Server) callGetDocument(_ context.Context, args map[string]any) (string, error) {
	id := getString(args, "doc_id")
	doc, err := s.svc.GetDocument(id)
	if err != nil {
		return "", err
	}
	if doc == nil {
		return "", fmt.Errorf("document not found: %s", id)
	}
	return toJSON(doc), nil
}

func (s *Server) callUpdateDocument(_ context.Context, args map[string]any) (string, error) {
	id := getString(args, "doc_id")
	var textPtr *string
	if t, ok := args["text"].(string); ok && t != "" {
		textPtr = &t
	}
	meta := getMap(args, "meta")
	if err := s.svc.UpdateDocument(id, textPtr, meta); err != nil {
		return "", err
	}
	return `{"status":"ok"}`, nil
}

func (s *Server) callDeleteDocument(_ context.Context, args map[string]any) (string, error) {
	id := getString(args, "doc_id")
	deleted, err := s.svc.DeleteDocument(id)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(`{"deleted":%v}`, deleted), nil
}

func (s *Server) callGetRelated(_ context.Context, args map[string]any) (string, error) {
	nodeID := getString(args, "node_id")
	depth := getInt(args, "max_depth", 1)
	filter := getMap(args, "metadata_filter")
	edges, err := s.svc.GetRelated(nodeID, depth, filter)
	if err != nil {
		return "", err
	}
	return toJSON(map[string]any{"node_id": nodeID, "relations": edges}), nil
}

func (s *Server) callGraphStats(_ context.Context, _ map[string]any) (string, error) {
	return toJSON(s.svc.GraphStats()), nil
}

func (s *Server) callStats(_ context.Context, _ map[string]any) (string, error) {
	return toJSON(s.svc.Stats()), nil
}

func (s *Server) callClear(_ context.Context, _ map[string]any) (string, error) {
	if err := s.svc.Clear(); err != nil {
		return "", err
	}
	return `{"status":"ok"}`, nil
}

func (s *Server) callFindCommunities(_ context.Context, args map[string]any) (string, error) {
	res := getFloat(args, "resolution", 1.0)
	knn := getInt(args, "k_nn", 15)
	comms, err := s.svc.FindCommunities(res, knn)
	if err != nil {
		return "", err
	}
	return toJSON(comms), nil
}

func (s *Server) callSetCommunityNames(_ context.Context, args map[string]any) (string, error) {
	raw, ok := args["names"]
	if !ok {
		return "", fmt.Errorf("missing required argument: names")
	}
	names := map[int]string{}
	switch v := raw.(type) {
	case map[string]any:
		for k, val := range v {
			var id int
			fmt.Sscanf(k, "%d", &id)
			names[id] = fmt.Sprint(val)
		}
	default:
		return "", fmt.Errorf("names must be a dict of id→name")
	}
	if err := s.svc.SetCommunityNames(names); err != nil {
		return "", err
	}
	return `{"status":"ok"}`, nil
}

func (s *Server) callGetCommunities(_ context.Context, _ map[string]any) (string, error) {
	comms, err := s.svc.GetCommunities()
	if err != nil {
		return "", err
	}
	return toJSON(comms), nil
}

func (s *Server) callAddStructured(_ context.Context, args map[string]any) (string, error) {
	content := getString(args, "content")
	result, err := s.svc.AddStructured(strings.NewReader(content))
	if err != nil {
		return "", err
	}
	return toJSON(result), nil
}

// ── JSON Schema builder (lightweight) ─────────────────────────────────────

type prop struct {
	Type    string
	Desc    string
	Default any
}

type toolSchema map[string]prop

func buildJSONSchema(s toolSchema) map[string]any {
	props := map[string]any{}
	required := []string{}
	for name, p := range s {
		propObj := map[string]any{}
		if p.Type != "" {
			propObj["type"] = p.Type
		}
		if p.Desc != "" {
			propObj["description"] = p.Desc
		}
		if p.Default != nil {
			propObj["default"] = p.Default
		}
		props[name] = propObj
		if p.Default == nil {
			required = append(required, name)
		}
	}
	schema := map[string]any{
		"type":       "object",
		"properties": props,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

// ── Argument helpers ──────────────────────────────────────────────────────

func getString(args map[string]any, key string) string {
	if v, ok := args[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func getInt(args map[string]any, key string, def int) int {
	if v, ok := args[key]; ok {
		switch n := v.(type) {
		case float64:
			return int(n)
		case int:
			return n
		}
	}
	return def
}

func getFloat(args map[string]any, key string, def float64) float64 {
	if v, ok := args[key]; ok {
		switch n := v.(type) {
		case float64:
			return n
		case int:
			return float64(n)
		}
	}
	return def
}

func getMap(args map[string]any, key string) map[string]any {
	if v, ok := args[key]; ok {
		if m, ok := v.(map[string]any); ok {
			return m
		}
	}
	return nil
}

func toJSON(v any) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return fmt.Sprintf(`{"error":%q}`, err.Error())
	}
	return strings.TrimSpace(buf.String())
}

// ensure io.Reader referenced
var _ io.Reader = (*strings.Reader)(nil)
