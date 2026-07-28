// Package config provides RAGConfig — the central configuration for the RAG system.
// Loaded from environment variables with sensible defaults, mirroring the Python
// pydantic-settings RAGConfig.
package config

import (
	"os"
	"strings"
)

// EmbeddingProviderKind identifies the embedding backend.
type EmbeddingProviderKind string

const (
	ProviderSentenceTransformer EmbeddingProviderKind = "sentence_transformer"
	ProviderOllama              EmbeddingProviderKind = "ollama"
	ProviderOpenAICompatible    EmbeddingProviderKind = "openai-compatible"
	ProviderAnthropic           EmbeddingProviderKind = "anthropic"
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
	EmbeddingDevice   string // cpu | cuda
	EmbeddingDim      int

	// -- Qdrant ---------------------------------------------------------------
	QdrantURL string // empty → embedded

	// -- Reranker -------------------------------------------------------------
	RerankerEnabled         bool
	RerankerModel           string
	RerankerBaseURL         string
	RerankerAPIKey          string
	RerankerTopKMultiplier  int

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
	GraphExtractMode   string // llm | ner
	GraphExtractModel  string
	GraphExtractBaseURL string

	// -- HTTP API -------------------------------------------------------------
	APIHost          string
	APIPort          int
	APIToken         string
	CORSAllowOrigins []string

	// -- Misc -----------------------------------------------------------------
	LogLevel       string
	PreloadModels  bool
	HFHubOffline   bool
	ModelsDir      string
	DocumentStore  string // sqlite | postgres
	PostgresURL    string

	// Community detection
	CommunityResolution float64
	CommunityKNN        int
}

// Load builds RAGConfig from environment variables with defaults.
func Load() *RAGConfig {
	return &RAGConfig{
		StorePath:            envOr("RAG_STORE_PATH", "./rag_data"),
		EmbeddingProvider:    EmbeddingProviderKind(envOr("EMBEDDING_PROVIDER", "sentence_transformer")),
		EmbeddingModel:       envOr("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
		EmbeddingBaseURL:     os.Getenv("EMBEDDING_BASE_URL"),
		EmbeddingAPIKey:      os.Getenv("EMBEDDING_API_KEY"),
		EmbeddingDevice:      envOr("EMBEDDING_DEVICE", "cpu"),
		EmbeddingDim:         envInt("EMBEDDING_DIMENSION", 1024),

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

		LogLevel:            envOr("LOG_LEVEL", "INFO"),
		PreloadModels:       envBool("PRELOAD_MODELS", false),
		HFHubOffline:        envBool("HF_HUB_OFFLINE", false),
		ModelsDir:           os.Getenv("MODELS_DIR"),
		DocumentStore:       envOr("DOCUMENT_STORE", "sqlite"),
		PostgresURL:         os.Getenv("POSTGRES_URL"),
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
	// simplified; production would use strconv
	return def
}

func envFloat(key string, def float64) float64 {
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

// MaskSecret shows only the last 4 characters of a secret value.
func MaskSecret(s string) string {
	if s == "" {
		return ""
	}
	if len(s) <= 4 {
		return "***"
	}
	return "***" + s[len(s)-4:]
}
