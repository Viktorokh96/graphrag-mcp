// Package httpapi provides an HTTP REST API wrapping the search Service.
// Routes are registered on a standard http.ServeMux with path-based patterns
// (Go 1.22+ method+path routing).
// Supports bearer-token auth (constant-time HMAC comparison) and CORS middleware.
package httpapi

import (
	"crypto/hmac"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
	"github.com/Viktorokh96/graphrag-mcp/internal/search"
)

// ── Server ──────────────────────────────────────────────────────────────────

// Server holds references to the search service and configuration.
type Server struct {
	svc *search.Service
	cfg *config.RAGConfig
}

// NewServer creates an *http.Server with all routes registered and middleware
// applied. Callers should start it with server.ListenAndServe().
func NewServer(svc *search.Service, cfg *config.RAGConfig) *http.Server {
	mux := http.NewServeMux()
	s := &Server{svc: svc, cfg: cfg}

	// ── API endpoints ─────────────────────────────────────────────────────
	// ── Root redirect ────────────────────────────────────────────────────
	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/ui/", http.StatusMovedPermanently)
	})
	mux.HandleFunc("GET /api/health", s.handleHealth)

	mux.HandleFunc("POST /api/documents", s.handleAddDocument)
	mux.HandleFunc("GET /api/documents", s.handleListDocuments)
	mux.HandleFunc("GET /api/documents/{id}", s.handleGetDocument)
	mux.HandleFunc("PUT /api/documents/{id}", s.handleUpdateDocument)
	mux.HandleFunc("DELETE /api/documents/{id}", s.handleDeleteDocument)

	mux.HandleFunc("POST /api/search", s.handleSearch)
	mux.HandleFunc("POST /api/index", s.handleIndexStructured)

	mux.HandleFunc("GET /api/stats", s.handleStats)
	mux.HandleFunc("GET /api/graph", s.handleGraphData)
	mux.HandleFunc("POST /api/reindex", s.handleReindex)
	mux.HandleFunc("GET /api/graph/stats", s.handleGraphStats)
	mux.HandleFunc("GET /api/communities", s.handleCommunities)
	mux.HandleFunc("POST /api/extract", s.handleExtract)

	// ── Static file server ────────────────────────────────────────────────

	mux.Handle("GET /ui/", http.StripPrefix("/ui/", http.FileServer(http.Dir("webui"))))

	// ── Middleware chain (outermost first) ─────────────────────────────────
	var h http.Handler = mux

	if cfg.APIToken != "" {
		h = s.authMiddleware(h)
	}
	if len(cfg.CORSAllowOrigins) > 0 {
		h = s.corsMiddleware(h)
	}

	return &http.Server{
		Addr:         fmt.Sprintf("%s:%d", cfg.APIHost, cfg.APIPort),
		Handler:      h,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}
}

// ── Middleware ──────────────────────────────────────────────────────────────

// authMiddleware checks for a valid Bearer token using constant-time comparison.
// The expected token is stored in cfg.APIToken.
func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Skip auth for CORS preflight — CORS middleware handles those.
		if r.Method == http.MethodOptions {
			next.ServeHTTP(w, r)
			return
		}
		// Skip auth for the UI static files.
		if strings.HasPrefix(r.URL.Path, "/ui/") {
			next.ServeHTTP(w, r)
			return
		}
		auth := r.Header.Get("Authorization")
		if auth == "" || !strings.HasPrefix(auth, "Bearer ") {
			writeError(w, http.StatusUnauthorized, "missing or malformed Authorization header; expected 'Bearer <token>'")
			return
		}
		token := strings.TrimPrefix(auth, "Bearer ")
		if !hmac.Equal([]byte(token), []byte(s.cfg.APIToken)) {
			writeError(w, http.StatusUnauthorized, "invalid API token")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// corsMiddleware sets CORS headers based on cfg.CORSAllowOrigins.
// It handles OPTIONS preflight requests directly by returning 204 No Content.
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		if allowed := matchOrigin(origin, s.cfg.CORSAllowOrigins); allowed != "" {
			w.Header().Set("Access-Control-Allow-Origin", allowed)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			w.Header().Set("Access-Control-Max-Age", "86400")
		}

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// matchOrigin returns the origin to echo back, or "" when no rule matches.
func matchOrigin(origin string, allowedOrigins []string) string {
	if origin == "" {
		return ""
	}
	for _, allowed := range allowedOrigins {
		if allowed == "*" || allowed == origin {
			return allowed
		}
	}
	return ""
}

// ── Helper types ───────────────────────────────────────────────────────────

// apiError is the standard JSON error response body.
type apiError struct {
	Error string `json:"error"`
}

// ── Helper functions ───────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("[httpapi] encode error: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, apiError{Error: msg})
}

// decodeJSON reads and decodes JSON from r, limiting body size to 32 MB.
func decodeJSON(r io.Reader, v any) error {
	return json.NewDecoder(io.LimitReader(r, 32<<20)).Decode(v)
}

// ── Handlers: Health ───────────────────────────────────────────────────────

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.svc.Health())
}

// ── Handlers: Documents ────────────────────────────────────────────────────

type addDocRequest struct {
	Text     string         `json:"text"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type addDocResponse struct {
	DocID     ragtypes.DocID `json:"doc_id"`
	Duplicate bool           `json:"duplicate"`
}

func (s *Server) handleAddDocument(w http.ResponseWriter, r *http.Request) {
	var req addDocRequest
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if strings.TrimSpace(req.Text) == "" {
		writeError(w, http.StatusBadRequest, "'text' is required and must be non-empty")
		return
	}
	if req.Metadata == nil {
		req.Metadata = make(map[string]any)
	}

	result, err := s.svc.AddDocument(req.Text, req.Metadata)
	if err != nil {
		log.Printf("[httpapi] AddDocument error: %v", err)
		writeError(w, http.StatusInternalServerError, "failed to index document")
		return
	}
	writeJSON(w, http.StatusCreated, addDocResponse{
		DocID:     result.DocID,
		Duplicate: result.Duplicate,
	})
}

type listDocsResponse struct {
	Documents []*ragtypes.Document `json:"documents"`
	Total     int                  `json:"total"`
}

func (s *Server) handleListDocuments(w http.ResponseWriter, r *http.Request) {
	limit := 50
	offset := 0

	if l := r.URL.Query().Get("limit"); l != "" {
		if v, err := strconv.Atoi(l); err == nil && v > 0 && v <= 1000 {
			limit = v
		}
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		if v, err := strconv.Atoi(o); err == nil && v >= 0 {
			offset = v
		}
	}

	// Optional JSON filter object passed as a query parameter.
	var filter map[string]any
	if f := r.URL.Query().Get("filter"); f != "" {
		if err := json.Unmarshal([]byte(f), &filter); err != nil {
			writeError(w, http.StatusBadRequest, "invalid 'filter' JSON")
			return
		}
	}

	docs, total, err := s.svc.ListDocuments(limit, offset, filter)
	if err != nil {
		log.Printf("[httpapi] ListDocuments error: %v", err)
		writeError(w, http.StatusInternalServerError, "failed to list documents")
		return
	}
	if docs == nil {
		docs = []*ragtypes.Document{}
	}
	writeJSON(w, http.StatusOK, listDocsResponse{Documents: docs, Total: total})
}

func (s *Server) handleGetDocument(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "missing document ID")
		return
	}

	doc, err := s.svc.GetDocument(id)
	if err != nil {
		writeError(w, http.StatusNotFound, "document not found")
		return
	}
	writeJSON(w, http.StatusOK, doc)
}

type updateDocRequest struct {
	Text     *string        `json:"text,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

func (s *Server) handleUpdateDocument(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "missing document ID")
		return
	}

	var req updateDocRequest
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if req.Text == nil && req.Metadata == nil {
		writeError(w, http.StatusBadRequest, "at least one of 'text' or 'metadata' must be provided")
		return
	}

	if err := s.svc.UpdateDocument(id, req.Text, req.Metadata); err != nil {
		writeError(w, http.StatusNotFound, "document not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "updated"})
}

type deleteDocResponse struct {
	Deleted bool   `json:"deleted"`
	Error   string `json:"error,omitempty"`
}

func (s *Server) handleDeleteDocument(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "missing document ID")
		return
	}

	ok, err := s.svc.DeleteDocument(id)
	if err != nil {
		log.Printf("[httpapi] DeleteDocument error: %v", err)
		writeError(w, http.StatusInternalServerError, "failed to delete document")
		return
	}
	if !ok {
		writeJSON(w, http.StatusNotFound, deleteDocResponse{Deleted: false, Error: "document not found"})
		return
	}
	writeJSON(w, http.StatusOK, deleteDocResponse{Deleted: true})
}

// ── Handlers: Search ───────────────────────────────────────────────────────

type searchRequest struct {
	Query  string            `json:"query"`
	Mode   ragtypes.SearchMode `json:"mode"`
	K      int               `json:"k"`
	Filter map[string]any    `json:"filter,omitempty"`
}

type searchResponse struct {
	Query   string                `json:"query"`
	Results []ragtypes.SearchResult `json:"results"`
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req searchRequest
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		writeError(w, http.StatusBadRequest, "'query' is required")
		return
	}
	if req.Mode == "" {
		req.Mode = ragtypes.SearchHybrid
	}
	if req.Mode != ragtypes.SearchSemantic && req.Mode != ragtypes.SearchBM25 && req.Mode != ragtypes.SearchHybrid {
		writeError(w, http.StatusBadRequest, "'mode' must be 'semantic', 'bm25', or 'hybrid'")
		return
	}
	if req.K <= 0 || req.K > 200 {
		req.K = 10
	}

	results, err := s.svc.Search(req.Mode, req.Query, req.K, req.Filter)
	if err != nil {
		log.Printf("[httpapi] Search error: %v", err)
		writeError(w, http.StatusInternalServerError, "search failed")
		return
	}
	if results == nil {
		results = []ragtypes.SearchResult{}
	}
	writeJSON(w, http.StatusOK, searchResponse{Query: req.Query, Results: results})
}

// ── Handlers: Index (Repomix structured) ────────────────────────────────────

func (s *Server) handleIndexStructured(w http.ResponseWriter, r *http.Request) {
	// Determine content type to support both direct JSON and multipart uploads.
	ct := r.Header.Get("Content-Type")

	var bodyReader io.Reader = r.Body

	if strings.HasPrefix(ct, "multipart/form-data") {
		_, params, err := mime.ParseMediaType(ct)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid Content-Type")
			return
		}
		mr := multipart.NewReader(r.Body, params["boundary"])

		// Read first file part and use it as the JSON body.
		part, err := mr.NextPart()
		if err != nil {
			writeError(w, http.StatusBadRequest, "failed to read multipart data")
			return
		}
		defer part.Close()
		bodyReader = part
	}

	result, err := s.svc.AddStructured(bodyReader)
	if err != nil {
		log.Printf("[httpapi] AddStructured error: %v", err)
		writeError(w, http.StatusInternalServerError, "structured indexing failed")
		return
	}
	writeJSON(w, http.StatusOK, result)
}

// ── Handlers: Stats ────────────────────────────────────────────────────────

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	stats := s.svc.Stats()
	writeJSON(w, http.StatusOK, stats)
}

// ── Handlers: Graph ────────────────────────────────────────────────────────

type graphDataResponse struct {
	Edges []ragtypes.Edge  `json:"edges"`
	Nodes []ragtypes.DocID `json:"nodes"`
}

func (s *Server) handleGraphData(w http.ResponseWriter, r *http.Request) {
	edges, nodes, err := s.svc.GraphData()
	if err != nil {
		log.Printf("[httpapi] GraphData error: %v", err)
		writeError(w, http.StatusInternalServerError, "failed to get graph data")
		return
	}
	if edges == nil {
		edges = []ragtypes.Edge{}
	}
	if nodes == nil {
		nodes = []ragtypes.DocID{}
	}
	writeJSON(w, http.StatusOK, graphDataResponse{Edges: edges, Nodes: nodes})
}

func (s *Server) handleGraphStats(w http.ResponseWriter, r *http.Request) {
	stats := s.svc.GraphStats()
	writeJSON(w, http.StatusOK, stats)
}

// ── Handlers: Communities ──────────────────────────────────────────────────

func (s *Server) handleCommunities(w http.ResponseWriter, r *http.Request) {
	communities, err := s.svc.GetCommunities()
	if err != nil {
		log.Printf("[httpapi] GetCommunities error: %v", err)
		writeError(w, http.StatusInternalServerError, "failed to get communities")
		return
	}
	if communities == nil {
		communities = []ragtypes.Community{}
	}
	writeJSON(w, http.StatusOK, communities)
}

// ── Handlers: Extract ──────────────────────────────────────────────────────

type extractRequest struct {
	DocID ragtypes.DocID `json:"doc_id"`
	Text  string         `json:"text"`
	Mode  string         `json:"mode"`
}

func (s *Server) handleExtract(w http.ResponseWriter, r *http.Request) {
	var req extractRequest
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if strings.TrimSpace(req.Text) == "" {
		writeError(w, http.StatusBadRequest, "'text' is required")
		return
	}
	if req.Mode == "" {
		req.Mode = "llm"
	}
	if req.Mode != "llm" && req.Mode != "ner" {
		writeError(w, http.StatusBadRequest, "'mode' must be 'llm' or 'ner'")
		return
	}

	if err := s.svc.ExtractGraph(req.DocID, req.Text, req.Mode); err != nil {
		log.Printf("[httpapi] ExtractGraph error: %v", err)
		writeError(w, http.StatusInternalServerError, "graph extraction failed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// ── Handlers: Reindex ──────────────────────────────────────────────────────

type reindexResponse struct {
	Status    string `json:"status"`
	Reindexed int    `json:"reindexed"`
}

func (s *Server) handleReindex(w http.ResponseWriter, r *http.Request) {
	count, err := s.svc.Reindex()
	if err != nil {
		log.Printf("[httpapi] Reindex error: %v", err)
		writeError(w, http.StatusInternalServerError, "reindex failed")
		return
	}
	writeJSON(w, http.StatusOK, reindexResponse{Status: "ok", Reindexed: count})
}
