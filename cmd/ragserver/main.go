// Command ragserver is the main entry point for the Go RAG MCP server.
// It supports two modes:
//
//	ragserver       MCP stdio server (default)
//	ragserver -http  HTTP REST API + MCP SSE + WebUI
//
// Configuration is loaded from environment variables (see internal/config).
package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"os"

	_ "github.com/mattn/go-sqlite3"
	"github.com/Viktorokh96/graphrag-mcp/internal/config"
	"github.com/Viktorokh96/graphrag-mcp/internal/docstore"
	"github.com/Viktorokh96/graphrag-mcp/internal/embedding"
	"github.com/Viktorokh96/graphrag-mcp/internal/expander"
	"github.com/Viktorokh96/graphrag-mcp/internal/graphstore"
	"github.com/Viktorokh96/graphrag-mcp/internal/httpapi"
	"github.com/Viktorokh96/graphrag-mcp/internal/mcp"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
	"github.com/Viktorokh96/graphrag-mcp/internal/reranker"
	"github.com/Viktorokh96/graphrag-mcp/internal/search"
	"github.com/Viktorokh96/graphrag-mcp/internal/vecstore"
)

var (
	modeHTTP = flag.Bool("http", false, "Start HTTP REST API + MCP SSE server")
	port     = flag.Int("port", 0, "HTTP port (overrides API_PORT env)")
	host     = flag.String("host", "", "HTTP bind address (overrides API_HOST env)")
)

func main() {
	flag.Parse()

	cfg := config.Load()

	// CLI overrides
	if *port != 0 {
		cfg.APIPort = *port
	}
	if *host != "" {
		cfg.APIHost = *host
	}

	// Ensure store directory
	if err := os.MkdirAll(cfg.StorePath, 0755); err != nil {
		log.Fatalf("cannot create store directory %s: %v", cfg.StorePath, err)
	}


	svc := buildService(cfg)
	defer svc.Close()

	if *modeHTTP {
		runHTTP(svc, cfg)
	} else {
		runMCP(svc)
	}
}

func buildService(cfg *config.RAGConfig) *search.Service {
	dbPath := cfg.StorePath + "/store.db"
	sqlDB, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL")
	if err != nil {
		log.Fatalf("sqlite: %v", err)
	}
	sqlDB.SetMaxOpenConns(1)

	// Document store
	docs, err := docstore.NewSQLiteDocStore(dbPath)
	if err != nil {
		log.Fatalf("docstore: %v", err)
	}

	// Vector store
	vecs, err := vecstore.NewQdrantVecStore(cfg)
	if err != nil {
		log.Fatalf("vecstore: %v", err)
	}

	// Graph store (same store.db, different tables)
	graph, err := graphstore.NewGraphStore(sqlDB, func(docID ragtypes.DocID) (map[string]any, bool) {
		doc, err := docs.Get(docID)
		if err != nil || doc == nil {
			return nil, false
		}
		return doc.Metadata, true
	})
	if err != nil {
		log.Fatalf("graphstore: %v", err)
	}

	// Embedding provider
	emb, err := embedding.NewProvider(cfg)
	if err != nil {
		log.Fatalf("embedding: %v", err)
	}

	// Reranker & expander
	var rerank ragtypes.Reranker = reranker.NewNoopReranker()
	var expand ragtypes.QueryExpander = expander.NewNoopExpander()

	if cfg.RerankerEnabled && cfg.RerankerBaseURL != "" {
		rerank = reranker.NewAPIReranker(cfg.RerankerBaseURL, cfg.RerankerModel, cfg.RerankerAPIKey)
		log.Println("reranker: enabled")
	}
	if cfg.QueryExpansionEnabled {
		baseURL := cfg.ExpanderBaseURL
		if baseURL == "" {
			baseURL = "http://localhost:11434"
		}
		expand = expander.NewOllamaExpander(baseURL, cfg.ExpanderModel, cfg.ExpanderNumVariants)
		log.Println("query expansion: enabled")
	}

	return search.New(docs, vecs, graph, emb, rerank, expand, nil, cfg)
}

func runMCP(svc *search.Service) {
	srv := mcp.New(svc)
	log.SetOutput(os.Stderr)
	log.SetPrefix("[mcp] ")
	fmt.Fprintf(os.Stderr, "MCP server starting (stdio)\n")
	if err := srv.Run(); err != nil {
		log.Fatalf("mcp: %v", err)
	}
}

func runHTTP(svc *search.Service, cfg *config.RAGConfig) {
	srv := httpapi.NewServer(svc, cfg)
	addr := fmt.Sprintf("%s:%d", cfg.APIHost, cfg.APIPort)
	log.SetPrefix("[http] ")
	log.Printf("REST API: http://%s/docs", addr)
	log.Printf("MCP:      http://%s/mcp (Streamable HTTP)", addr)
	log.Printf("WebUI:    http://%s/ui/", addr)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("http: %v", err)
	}
}
