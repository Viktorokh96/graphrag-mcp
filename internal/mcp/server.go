// Package mcp implements a stdio-based MCP (Model Context Protocol) server
// following JSON-RPC 2.0, exposing all RAG search and management tools.
package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
	"github.com/Viktorokh96/graphrag-mcp/internal/search"
)

// ── JSON-RPC 2.0 types ───────────────────────────────────────────────────

type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      any         `json:"id"`
	Result  any         `json:"result,omitempty"`
	Error   *rpcError   `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

const (
	errCodeInvalidRequest = -32600
	errCodeMethodNotFound = -32601
	errCodeInvalidParams  = -32602
	errCodeInternal       = -32603
	errCodeToolError      = -32000
)

// ── MCP method constants ─────────────────────────────────────────────────

const (
	methodInitialize = "initialize"
	methodToolsList  = "tools/list"
	methodToolsCall  = "tools/call"
)

// ── Tool schema types ─────────────────────────────────────────────────────

// ToolSchema is the JSON Schema for a tool's input parameters.
type ToolSchema struct {
	Type       string                  `json:"type"`
	Properties map[string]ToolProperty `json:"properties,omitempty"`
	Required   []string                `json:"required,omitempty"`
}

// ToolProperty describes a single input parameter.
type ToolProperty struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// ToolDef is an MCP tool definition returned by tools/list.
type ToolDef struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	InputSchema ToolSchema `json:"inputSchema"`
}

// toolDefs is the static list of all RAG tools.
var toolDefs = []ToolDef{
	{
		Name:        "rag_search",
		Description: "Semantic (dense vector) search over indexed documents.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"query":  {Type: "string", Description: "Search query text"},
				"k":      {Type: "number", Description: "Max results (default 10)"},
				"filter": {Type: "object", Description: "Optional metadata filter"},
			},
			Required: []string{"query"},
		},
	},
	{
		Name:        "rag_bm25_search",
		Description: "BM25 (sparse keyword) search over indexed documents.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"query":  {Type: "string", Description: "Search query text"},
				"k":      {Type: "number", Description: "Max results (default 10)"},
				"filter": {Type: "object", Description: "Optional metadata filter"},
			},
			Required: []string{"query"},
		},
	},
	{
		Name:        "rag_search_hybrid",
		Description: "Hybrid search combining dense and sparse results. Uses RRF fusion with optional query expansion and reranking.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"query":  {Type: "string", Description: "Search query text"},
				"k":      {Type: "number", Description: "Max results (default 10)"},
				"filter": {Type: "object", Description: "Optional metadata filter"},
			},
			Required: []string{"query"},
		},
	},
	{
		Name:        "rag_add_document",
		Description: "Add a text document to the knowledge base.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"text":     {Type: "string", Description: "Document text content"},
				"metadata": {Type: "object", Description: "Optional metadata key-value pairs"},
			},
			Required: []string{"text"},
		},
	},
	{
		Name:        "rag_add_file",
		Description: "Add a file from the local filesystem to the knowledge base.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"file_path": {Type: "string", Description: "Path to the file on disk"},
			},
			Required: []string{"file_path"},
		},
	},
	{
		Name:        "rag_add_relation",
		Description: "Add a typed, weighted edge between two documents in the knowledge graph.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"source":   {Type: "string", Description: "Source document ID"},
				"target":   {Type: "string", Description: "Target document ID"},
				"relation": {Type: "string", Description: "Relation type label"},
				"weight":   {Type: "number", Description: "Edge weight (default 1.0)"},
			},
			Required: []string{"source", "target", "relation"},
		},
	},
	{
		Name:        "rag_delete_relation",
		Description: "Delete a typed edge from the knowledge graph.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"source":   {Type: "string", Description: "Source document ID"},
				"target":   {Type: "string", Description: "Target document ID"},
				"relation": {Type: "string", Description: "Relation type label"},
			},
			Required: []string{"source", "target", "relation"},
		},
	},
	{
		Name:        "rag_list_documents",
		Description: "List documents with pagination and optional metadata filter.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"limit":  {Type: "number", Description: "Max results (default 20)"},
				"offset": {Type: "number", Description: "Pagination offset (default 0)"},
				"filter": {Type: "object", Description: "Optional metadata filter"},
			},
		},
	},
	{
		Name:        "rag_get_document",
		Description: "Get a single document by its ID.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"doc_id": {Type: "string", Description: "Document ID"},
			},
			Required: []string{"doc_id"},
		},
	},
	{
		Name:        "rag_update_document",
		Description: "Update a document's text and/or metadata.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"doc_id":   {Type: "string", Description: "Document ID"},
				"text":     {Type: "string", Description: "New text (omit to keep unchanged)"},
				"metadata": {Type: "object", Description: "New metadata to merge"},
			},
			Required: []string{"doc_id"},
		},
	},
	{
		Name:        "rag_delete_document",
		Description: "Delete a document and its graph node.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"doc_id": {Type: "string", Description: "Document ID"},
			},
			Required: []string{"doc_id"},
		},
	},
	{
		Name:        "rag_get_related",
		Description: "Get graph edges reachable from a document node.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"node_id":   {Type: "string", Description: "Source document ID"},
				"max_depth": {Type: "number", Description: "Max traversal depth (default 1)"},
				"filter":    {Type: "object", Description: "Optional edge filter"},
			},
			Required: []string{"node_id"},
		},
	},
	{
		Name:        "rag_graph_stats",
		Description: "Get aggregate knowledge graph statistics.",
		InputSchema: ToolSchema{
			Type:       "object",
			Properties: map[string]ToolProperty{},
		},
	},
	{
		Name:        "rag_stats",
		Description: "Get aggregate storage statistics across all stores.",
		InputSchema: ToolSchema{
			Type:       "object",
			Properties: map[string]ToolProperty{},
		},
	},
	{
		Name:        "rag_clear",
		Description: "Clear all documents, vectors, and graph data.",
		InputSchema: ToolSchema{
			Type:       "object",
			Properties: map[string]ToolProperty{},
		},
	},
	{
		Name:        "rag_find_communities",
		Description: "Run community detection (Leiden) on the document graph.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"resolution": {Type: "number", Description: "Leiden resolution (default 1.0)"},
				"knn":        {Type: "number", Description: "k-NN graph size (default 5)"},
			},
		},
	},
	{
		Name:        "rag_set_community_names",
		Description: "Set human-readable names for communities by their integer ID.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"names": {Type: "object", Description: "Map of community ID to name, e.g. {\"1\": \"docs\"}"},
			},
			Required: []string{"names"},
		},
	},
	{
		Name:        "rag_get_communities",
		Description: "List all detected communities with members.",
		InputSchema: ToolSchema{
			Type:       "object",
			Properties: map[string]ToolProperty{},
		},
	},
	{
		Name:        "rag_add_structured",
		Description: "Index a Repomix-style structured JSON payload.",
		InputSchema: ToolSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"data": {Type: "object", Description: "Repomix-style JSON structure (will be serialized)"},
			},
			Required: []string{"data"},
		},
	},
}

// ── Server ────────────────────────────────────────────────────────────────

// Server is a stdio-based MCP JSON-RPC 2.0 server backed by a search.Service.
type Server struct {
	svc *search.Service
}

// New creates an MCP Server backed by the given search service.
func New(svc *search.Service) *Server {
	return &Server{svc: svc}
}

// Run reads JSON-RPC requests from stdin and writes responses to stdout.
// Blocks until stdin is closed or an unrecoverable error occurs.
func (s *Server) Run() error {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		resp := s.handleMessage([]byte(line))
		if resp != nil {
			s.writeResponse(resp)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("mcp: stdin read error: %w", err)
	}
	return nil
}

// handleMessage parses a JSON-RPC request and dispatches it.
func (s *Server) handleMessage(raw []byte) *jsonRPCResponse {
	var req jsonRPCRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return errorResponse(nil, errCodeInvalidRequest, "Parse error: "+err.Error())
	}
	if req.JSONRPC != "2.0" {
		return errorResponse(req.ID, errCodeInvalidRequest, "Only jsonrpc 2.0 is supported")
	}

	switch req.Method {
	case methodInitialize:
		return s.handleInitialize(req)
	case methodToolsList:
		return s.handleToolsList(req)
	case methodToolsCall:
		return s.handleToolsCall(req)
	default:
		return errorResponse(req.ID, errCodeMethodNotFound, "Method not found: "+req.Method)
	}
}

// writeResponse marshals and writes a JSON-RPC response to stdout.
func (s *Server) writeResponse(resp *jsonRPCResponse) {
	data, err := json.Marshal(resp)
	if err != nil {
		fallback, _ := json.Marshal(jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      nil,
			Error:   &rpcError{Code: errCodeInternal, Message: "marshal error"},
		})
		fmt.Fprintln(os.Stdout, string(fallback))
		return
	}
	fmt.Fprintln(os.Stdout, string(data))
}

// ── Response helpers ─────────────────────────────────────────────────────

func errorResponse(id json.RawMessage, code int, msg string) *jsonRPCResponse {
	var rid any
	if id != nil {
		_ = json.Unmarshal(id, &rid)
	}
	return &jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      rid,
		Error:   &rpcError{Code: code, Message: msg},
	}
}

func successResponse(id json.RawMessage, result any) *jsonRPCResponse {
	var rid any
	if id != nil {
		_ = json.Unmarshal(id, &rid)
	}
	return &jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      rid,
		Result:  result,
	}
}

// textContent builds an MCP text content block.
func textContent(text string) map[string]any {
	return map[string]any{"type": "text", "text": text}
}

// ── Handlers for initialize / tools/list ─────────────────────────────────

func (s *Server) handleInitialize(req jsonRPCRequest) *jsonRPCResponse {
	return successResponse(req.ID, map[string]any{
		"protocolVersion": "0.1.0",
		"capabilities": map[string]any{
			"tools": map[string]any{},
		},
		"serverInfo": map[string]string{
			"name":    "graphrag-mcp",
			"version": "0.1.0",
		},
	})
}

func (s *Server) handleToolsList(req jsonRPCRequest) *jsonRPCResponse {
	return successResponse(req.ID, map[string]any{
		"tools": toolDefs,
	})
}

// ── Argument parsing ─────────────────────────────────────────────────────

type toolArgs map[string]any

func (s *Server) parseArgs(req jsonRPCRequest) (toolArgs, *jsonRPCResponse) {
	if req.Params == nil {
		return nil, errorResponse(req.ID, errCodeInvalidParams, "Missing params")
	}
	var params struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil {
		return nil, errorResponse(req.ID, errCodeInvalidParams, "Invalid params: "+err.Error())
	}
	var args toolArgs
	if params.Arguments != nil {
		if err := json.Unmarshal(params.Arguments, &args); err != nil {
			return nil, errorResponse(req.ID, errCodeInvalidParams, "Invalid arguments: "+err.Error())
		}
	}
	return args, nil
}

func requireString(args toolArgs, key string, id json.RawMessage) (string, *jsonRPCResponse) {
	v, ok := args[key]
	if !ok {
		return "", errorResponse(id, errCodeInvalidParams, "Missing required argument: "+key)
	}
	s, ok := v.(string)
	if !ok {
		return "", errorResponse(id, errCodeInvalidParams, "Argument "+key+" must be a string")
	}
	return s, nil
}

func optInt(args toolArgs, key string, def int) int {
	v, ok := args[key]
	if !ok {
		return def
	}
	f, ok := v.(float64)
	if !ok {
		return def
	}
	return int(f)
}

func optFloat(args toolArgs, key string, def float64) float64 {
	v, ok := args[key]
	if !ok {
		return def
	}
	f, ok := v.(float64)
	if !ok {
		return def
	}
	return f
}

func optMap(args toolArgs, key string) map[string]any {
	v, ok := args[key]
	if !ok {
		return nil
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil
	}
	return m
}

// ── tools/call dispatcher ────────────────────────────────────────────────

func (s *Server) handleToolsCall(req jsonRPCRequest) *jsonRPCResponse {
	args, errResp := s.parseArgs(req)
	if errResp != nil {
		return errResp
	}

	var params struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil || params.Name == "" {
		return errorResponse(req.ID, errCodeInvalidParams, "Missing tool name")
	}

	switch params.Name {
	case "rag_search":
		return s.callSearch(req, args, ragtypes.SearchSemantic)
	case "rag_bm25_search":
		return s.callSearch(req, args, ragtypes.SearchBM25)
	case "rag_search_hybrid":
		return s.callSearch(req, args, ragtypes.SearchHybrid)
	case "rag_add_document":
		return s.callAddDocument(req, args)
	case "rag_add_file":
		return s.callAddFile(req, args)
	case "rag_add_relation":
		return s.callAddRelation(req, args)
	case "rag_delete_relation":
		return s.callDeleteRelation(req, args)
	case "rag_list_documents":
		return s.callListDocuments(req, args)
	case "rag_get_document":
		return s.callGetDocument(req, args)
	case "rag_update_document":
		return s.callUpdateDocument(req, args)
	case "rag_delete_document":
		return s.callDeleteDocument(req, args)
	case "rag_get_related":
		return s.callGetRelated(req, args)
	case "rag_graph_stats":
		return s.callGraphStats(req)
	case "rag_stats":
		return s.callStats(req)
	case "rag_clear":
		return s.callClear(req)
	case "rag_find_communities":
		return s.callFindCommunities(req, args)
	case "rag_set_community_names":
		return s.callSetCommunityNames(req, args)
	case "rag_get_communities":
		return s.callGetCommunities(req)
	case "rag_add_structured":
		return s.callAddStructured(req, args)
	default:
		return errorResponse(req.ID, errCodeMethodNotFound, "Unknown tool: "+params.Name)
	}
}

// ── Tool call implementations ─────────────────────────────────────────────

// callSearch handles semantic, BM25, and hybrid search.
func (s *Server) callSearch(req jsonRPCRequest, args toolArgs, mode ragtypes.SearchMode) *jsonRPCResponse {
	query, errResp := requireString(args, "query", req.ID)
	if errResp != nil {
		return errResp
	}
	k := optInt(args, "k", 10)
	filter := optMap(args, "filter")

	results, err := s.svc.Search(mode, query, k, filter)
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "Search failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(results))},
	})
}

func (s *Server) callAddDocument(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	text, errResp := requireString(args, "text", req.ID)
	if errResp != nil {
		return errResp
	}
	meta := optMap(args, "metadata")

	result, err := s.svc.AddDocument(text, meta)
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "AddDocument failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(result))},
	})
}

func (s *Server) callAddFile(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	filePath, errResp := requireString(args, "file_path", req.ID)
	if errResp != nil {
		return errResp
	}

	result, err := s.svc.AddFile(filePath)
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "AddFile failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(result))},
	})
}

func (s *Server) callAddRelation(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	source, errResp := requireString(args, "source", req.ID)
	if errResp != nil {
		return errResp
	}
	target, errResp := requireString(args, "target", req.ID)
	if errResp != nil {
		return errResp
	}
	relation, errResp := requireString(args, "relation", req.ID)
	if errResp != nil {
		return errResp
	}
	weight := optFloat(args, "weight", 1.0)

	if err := s.svc.AddRelation(source, target, relation, weight); err != nil {
		return errorResponse(req.ID, errCodeToolError, "AddRelation failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(fmt.Sprintf(`{"status":"ok","source":"%s","target":"%s","relation":"%s"}`, source, target, relation))},
	})
}

func (s *Server) callDeleteRelation(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	source, errResp := requireString(args, "source", req.ID)
	if errResp != nil {
		return errResp
	}
	target, errResp := requireString(args, "target", req.ID)
	if errResp != nil {
		return errResp
	}
	relation, errResp := requireString(args, "relation", req.ID)
	if errResp != nil {
		return errResp
	}

	ok, err := s.svc.DeleteRelation(source, target, relation)
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "DeleteRelation failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(map[string]any{"deleted": ok}))},
	})
}

func (s *Server) callListDocuments(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	limit := optInt(args, "limit", 20)
	offset := optInt(args, "offset", 0)
	filter := optMap(args, "filter")

	docs, total, err := s.svc.ListDocuments(limit, offset, filter)
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "ListDocuments failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(map[string]any{
			"documents": docs,
			"total":     total,
			"limit":     limit,
			"offset":    offset,
		}))},
	})
}

func (s *Server) callGetDocument(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	docID, errResp := requireString(args, "doc_id", req.ID)
	if errResp != nil {
		return errResp
	}

	doc, err := s.svc.GetDocument(ragtypes.DocID(docID))
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "GetDocument failed: "+err.Error())
	}
	if doc == nil {
		return successResponse(req.ID, map[string]any{
			"content": []map[string]any{textContent(`{"error":"document not found"}`)},
		})
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(doc))},
	})
}

func (s *Server) callUpdateDocument(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	docID, errResp := requireString(args, "doc_id", req.ID)
	if errResp != nil {
		return errResp
	}

	var textPtr *string
	if v, ok := args["text"]; ok {
		if s, ok := v.(string); ok {
			textPtr = &s
		}
	}
	meta := optMap(args, "metadata")

	if err := s.svc.UpdateDocument(ragtypes.DocID(docID), textPtr, meta); err != nil {
		return errorResponse(req.ID, errCodeToolError, "UpdateDocument failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(`{"status":"ok"}`)},
	})
}

func (s *Server) callDeleteDocument(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	docID, errResp := requireString(args, "doc_id", req.ID)
	if errResp != nil {
		return errResp
	}

	ok, err := s.svc.DeleteDocument(ragtypes.DocID(docID))
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "DeleteDocument failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(map[string]any{"deleted": ok}))},
	})
}

func (s *Server) callGetRelated(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	nodeID, errResp := requireString(args, "node_id", req.ID)
	if errResp != nil {
		return errResp
	}
	maxDepth := optInt(args, "max_depth", 1)
	filter := optMap(args, "filter")

	edges, err := s.svc.GetRelated(ragtypes.DocID(nodeID), maxDepth, filter)
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "GetRelated failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(edges))},
	})
}

func (s *Server) callGraphStats(req jsonRPCRequest) *jsonRPCResponse {
	stats := s.svc.GraphStats()
	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(stats))},
	})
}

func (s *Server) callStats(req jsonRPCRequest) *jsonRPCResponse {
	stats := s.svc.Stats()
	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(stats))},
	})
}

func (s *Server) callClear(req jsonRPCRequest) *jsonRPCResponse {
	if err := s.svc.Clear(); err != nil {
		return errorResponse(req.ID, errCodeToolError, "Clear failed: "+err.Error())
	}
	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(`{"status":"ok"}`)},
	})
}

func (s *Server) callFindCommunities(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	resolution := optFloat(args, "resolution", 1.0)
	kNN := optInt(args, "knn", 5)

	communities, err := s.svc.FindCommunities(resolution, kNN)
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "FindCommunities failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(communities))},
	})
}

func (s *Server) callSetCommunityNames(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	var names map[int]string

	if m, ok := args["names"].(map[string]any); ok {
		names = make(map[int]string, len(m))
		for k, v := range m {
			id := 0
			if _, err := fmt.Sscanf(k, "%d", &id); err != nil {
				return errorResponse(req.ID, errCodeInvalidParams, "Community ID must be integer: "+k)
			}
			if s, ok := v.(string); ok {
				names[id] = s
			}
		}
	} else {
		// Try parsing the raw string value as JSON
		raw, errResp := requireString(args, "names", req.ID)
		if errResp != nil {
			return errResp
		}
		if err := json.Unmarshal([]byte(raw), &names); err != nil {
			return errorResponse(req.ID, errCodeInvalidParams, "names must be a JSON object mapping int IDs (as strings) to string names")
		}
	}

	if err := s.svc.SetCommunityNames(names); err != nil {
		return errorResponse(req.ID, errCodeToolError, "SetCommunityNames failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(`{"status":"ok"}`)},
	})
}

func (s *Server) callGetCommunities(req jsonRPCRequest) *jsonRPCResponse {
	communities, err := s.svc.GetCommunities()
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "GetCommunities failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(communities))},
	})
}

func (s *Server) callAddStructured(req jsonRPCRequest, args toolArgs) *jsonRPCResponse {
	// The "data" argument comes as an arbitrary JSON object; marshal it to JSON string
	// and feed it to AddStructured as an io.Reader.
	dataRaw, errResp := requireString(args, "data", req.ID)
	if errResp != nil {
		// Maybe it's a raw JSON object we can serialize
		if v, ok := args["data"]; ok {
			serialized, err := json.Marshal(v)
			if err != nil {
				return errorResponse(req.ID, errCodeInvalidParams, "data must be a JSON object or JSON string")
			}
			result, err := s.svc.AddStructured(strings.NewReader(string(serialized)))
			if err != nil {
				return errorResponse(req.ID, errCodeToolError, "AddStructured failed: "+err.Error())
			}
			return successResponse(req.ID, map[string]any{
				"content": []map[string]any{textContent(mustJSON(result))},
			})
		}
		return errorResponse(req.ID, errCodeInvalidParams, "Missing required argument: data")
	}

	result, err := s.svc.AddStructured(strings.NewReader(dataRaw))
	if err != nil {
		return errorResponse(req.ID, errCodeToolError, "AddStructured failed: "+err.Error())
	}

	return successResponse(req.ID, map[string]any{
		"content": []map[string]any{textContent(mustJSON(result))},
	})
}

// ── Helper ────────────────────────────────────────────────────────────────

// mustJSON marshals v to a JSON string, panicking on failure.
func mustJSON(v any) string {
	data, err := json.Marshal(v)
	if err != nil {
		panic("mcp: json marshal failed: " + err.Error())
	}
	return string(data)
}

// Ensure io.Reader interface is referenced (used by AddStructured).
var _ io.Reader = (*strings.Reader)(nil)
