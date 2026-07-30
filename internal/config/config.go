// Package config provides RAGConfig — the central configuration for the RAG system.
// Loaded from environment variables with sensible defaults, mirroring the Python
// pydantic-settings RAGConfig.
package config

import (
	"os"
	"strconv"
	"strings"
)

// EmbeddingProviderKind identifies the embedding backend.
type EmbeddingProviderKind string

const (
	ProviderOllama           EmbeddingProviderKind = "ollama"
	ProviderOpenAICompatible EmbeddingProviderKind = "openai-compatible"
)

// RAGConfig holds every tunable for the RAG system. Zero-value is unusable; always
// construct via Load() or LoadFromEnv().
type RAGConfig struct {
	// -- Paths ----------------------------------------------------------------
	StorePath string // rag_data/ — where SQLite and embedded Qdrant live.

	// -- Embedding ------------------------------------------------------------
	EmbeddingProvider EmbeddingProviderKind
	EmbeddingModel    string
	EmbeddingBaseURL  string
	EmbeddingAPIKey   string
	EmbeddingDim      int

	// -- Qdrant ---------------------------------------------------------------
	QdrantURL string // empty → embedded

	// -- Reranker -------------------------------------------------------------
	RerankerEnabled        bool
	RerankerModel          string
	RerankerBaseURL        string
	RerankerAPIKey         string
	RerankerTopKMultiplier int

	// -- Query expansion ------------------------------------------------------
	QueryExpansionEnabled bool
	ExpanderProvider      string
	ExpanderModel         string
	ExpanderBaseURL       string
	ExpanderAPIKey        string
	ExpanderNumVariants   int

	// -- Hybrid search --------------------------------------------------------
	DefaultAlpha  float64
	CyrillicAlpha float64

	// -- Graph extraction -----------------------------------------------------
	GraphExtractMode    string // llm | ner
	GraphExtractModel   string
	GraphExtractBaseURL string

	// -- HTTP API -------------------------------------------------------------
	APIHost          string
	APIPort          int
	APIToken         string
	CORSAllowOrigins []string

	// -- Misc -----------------------------------------------------------------
	LogLevel      string
	DocumentStore string // sqlite | postgres
	PostgresURL   string

	// Community detection
	CommunityResolution float64
	CommunityKNN        int
}

// Load builds RAGConfig from environment variables with defaults.
func Load() *RAGConfig {
	return &RAGConfig{
		StorePath:            envOr("RAG_STORE_PATH", "./rag_data"),
		EmbeddingProvider: EmbeddingProviderKind(envOr("EMBEDDING_PROVIDER", "ollama")),
		EmbeddingModel:    envOr("EMBEDDING_MODEL_NAME", "nomic-embed-text:latest"),
		EmbeddingBaseURL:  os.Getenv("EMBEDDING_BASE_URL"),
		EmbeddingAPIKey:   os.Getenv("EMBEDDING_API_KEY"),
		EmbeddingDim:      envInt("EMBEDDING_DIMENSION", 768),

		QdrantURL: os.Getenv("QDRANT_URL"),

		RerankerEnabled:        envBool("RERANK_ENABLED", false),
		RerankerModel:          envOr("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
		RerankerBaseURL:        os.Getenv("RERANKER_BASE_URL"),
		RerankerAPIKey:         os.Getenv("RERANKER_API_KEY"),
		RerankerTopKMultiplier: envInt("RERANKER_TOP_K_MULTIPLIER", 3),

		QueryExpansionEnabled: envBool("QUERY_EXPANSION_ENABLED", false),
		ExpanderProvider:      envOr("QUERY_EXPANSION_PROVIDER", "ollama"),
		ExpanderModel:         envOr("QUERY_EXPANSION_MODEL", "qwen2.5:1.5b"),
		ExpanderBaseURL:       os.Getenv("QUERY_EXPANSION_BASE_URL"),
		ExpanderAPIKey:        os.Getenv("QUERY_EXPANSION_API_KEY"),
		ExpanderNumVariants:   envInt("QUERY_EXPANSION_VARIANTS", 3),

		DefaultAlpha:  envFloat("RAG_DEFAULT_ALPHA", 0.5),
		CyrillicAlpha: envFloat("RAG_CYRILLIC_ALPHA", 0.85),

		GraphExtractMode:    envOr("GRAPH_EXTRACT_MODE", "llm"),
		GraphExtractModel:   envOr("GRAPH_EXTRACT_MODEL", "qwen2.5:4b"),
		GraphExtractBaseURL: os.Getenv("GRAPH_EXTRACT_BASE_URL"),

		APIHost:          envOr("API_HOST", "127.0.0.1"),
		APIPort:          envInt("API_PORT", 8765),
		APIToken:         os.Getenv("API_TOKEN"),
		CORSAllowOrigins: parseCSV(os.Getenv("CORS_ALLOW_ORIGINS")),

		LogLevel:      envOr("LOG_LEVEL", "INFO"),
		DocumentStore: envOr("DOCUMENT_STORE", "sqlite"),
		PostgresURL:   os.Getenv("POSTGRES_URL"),
		CommunityResolution: envFloat("COMMUNITY_RESOLUTION", 1.0),
		CommunityKNN:        envInt("COMMUNITY_KNN", 15),
	}
}

// ---------------------------------------------------------------------------
// env helpers

func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func envBool(key string, def bool) bool {
	v := strings.ToLower(os.Getenv(key))
	if v == "1" || v == "true" || v == "yes" {
		return true
	}
	if v == "0" || v == "false" || v == "no" {
		return false
	}
	return def
}

func parseCSV(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}

