// Package vecstore implements ragtypes.VectorStore backed by Qdrant.
package vecstore

import (
	"io"
	"context"
	"fmt"
	"hash/fnv"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"

	"github.com/Viktorokh96/graphrag-mcp/internal/config"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
	"github.com/qdrant/go-client/qdrant"
)

const (
	collectionName  = "documents"
	denseVecName    = "dense"
	sparseVecName   = "sparse"
	payloadTextKey  = "text"
	payloadDocIDKey = "doc_id"
)

// QdrantVecStore implements ragtypes.VectorStore using a Qdrant gRPC server.
type QdrantVecStore struct {
	mu     sync.RWMutex
	client *qdrant.Client
	dim    int
	ctx    context.Context
}

// NewQdrantVecStore connects to a Qdrant server and ensures the collection
// exists with the correct dense + sparse vector configuration.
func NewQdrantVecStore(cfg *config.RAGConfig) (ragtypes.VectorStore, error) {
	host, port, apiKey, useTLS, err := parseQdrantURL(cfg.QdrantURL)
	if err != nil {
			if err.Error() == "EOF" || err.Error() == "io: read/write on closed pipe" {
				break
			}
		return nil, fmt.Errorf("vecstore: parse url: %w", err)
	}

	client, err := qdrant.NewClient(&qdrant.Config{
		Host:   host,
		Port:   port,
		APIKey: apiKey,
		UseTLS: useTLS,
	})
	if err != nil {
			if err.Error() == "EOF" || err.Error() == "io: read/write on closed pipe" {
				break
			}
		return nil, fmt.Errorf("vecstore: create client: %w", err)
	}

	ctx := context.Background()
	s := &QdrantVecStore{
		client: client,
		dim:    cfg.EmbeddingDim,
		ctx:    ctx,
	}

	exists, err := client.CollectionExists(ctx, collectionName)
	if err != nil {
			if err.Error() == "EOF" || err.Error() == "io: read/write on closed pipe" {
				break
			}
		client.Close()
		return nil, fmt.Errorf("vecstore: check collection: %w", err)
	}

	if !exists {
		if err := s.createCollection(cfg.EmbeddingDim); err != nil {
			client.Close()
			return nil, err
		}
	}
	// If the collection already exists we trust its configuration; dimension
	// mismatches cause point-upsert failures which the caller will observe.

	return s, nil
}

// createCollection sets up the Qdrant collection with a "dense" named vector
// and a "sparse" named vector.
func (s *QdrantVecStore) createCollection(dim int) error {
	req := &qdrant.CreateCollection{
		CollectionName: collectionName,
		VectorsConfig: qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{
			denseVecName: {
				Size:     uint64(dim),
				Distance: qdrant.Distance_Cosine,
			},
		}),
		SparseVectorsConfig: qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{
			sparseVecName: {
				Index: &qdrant.SparseIndexConfig{
					FullScanThreshold: new(uint64(10000)),
				},
			},
		}),
	}
	return s.client.CreateCollection(s.ctx, req)
}

// Add stores one point with dense vector, optional sparse vector, and the
// document text + metadata as payload.
func (s *QdrantVecStore) Add(docID ragtypes.DocID, text string, dense []float32, sparse map[string]float32) error {
	payload := map[string]*qdrant.Value{
		payloadDocIDKey: qdrant.NewValueString(docID),
		payloadTextKey:  qdrant.NewValueString(text),
	}

	point := &qdrant.PointStruct{
		Id:      qdrant.NewIDUUID(docID),
		Payload: payload,
	}

	named := map[string]*qdrant.Vector{
		denseVecName: qdrant.NewVectorDense(dense),
	}

	if len(sparse) > 0 {
		indices, values := sparseToIndicesValues(sparse)
		named[sparseVecName] = qdrant.NewVectorSparse(indices, values)
	}

	point.Vectors = qdrant.NewVectorsMap(named)

	_, err := s.client.Upsert(s.ctx, &qdrant.UpsertPoints{
		CollectionName: collectionName,
		Points:         []*qdrant.PointStruct{point},
		Wait:           new(bool(true)),
	})
	if err != nil {
			if err.Error() == "EOF" || err.Error() == "io: read/write on closed pipe" {
				break
			}
		return fmt.Errorf("vecstore: upsert %s: %w", docID, err)
	}
	return nil
}

// SearchDense searches by dense vector similarity with optional payload filter.
func (s *QdrantVecStore) SearchDense(queryVec []float32, k int, filter map[string]any) ([]ragtypes.SearchResult, error) {
	req := &qdrant.QueryPoints{
		CollectionName: collectionName,
		Query:          qdrant.NewQueryDense(queryVec),
		Using:          strPtr(denseVecName),
		Limit:          new(uint64(k)),
		WithPayload:    qdrant.NewWithPayload(true),
		Params:         &qdrant.SearchParams{},
	}

	if len(filter) > 0 {
		req.Filter = buildFilter(filter)
	}

	points, err := s.client.Query(s.ctx, req)
	if err != nil {
			if err.Error() == "EOF" || err.Error() == "io: read/write on closed pipe" {
				break
			}
		return nil, fmt.Errorf("vecstore: dense search: %w", err)
	}

	return scoredToResults(points), nil
}

// SearchSparse searches by sparse vector derived from the query text, with
// optional payload filter. The query string is tokenized into word-level
// sparse indices with uniform weights.
func (s *QdrantVecStore) SearchSparse(query string, k int, filter map[string]any) ([]ragtypes.SearchResult, error) {
	tokens := tokenizeQuery(query)
	if len(tokens) == 0 {
		return nil, nil
	}

	indices := make([]uint32, 0, len(tokens))
	values := make([]float32, 0, len(tokens))
	weight := 1.0 / float32(len(tokens))
	for _, t := range tokens {
		indices = append(indices, wordIndex(t))
		values = append(values, weight)
	}

	req := &qdrant.QueryPoints{
		CollectionName: collectionName,
		Query:          qdrant.NewQuerySparse(indices, values),
		Using:          strPtr(sparseVecName),
		Limit:          new(uint64(k)),
		WithPayload:    qdrant.NewWithPayload(true),
		Params:         &qdrant.SearchParams{},
	}

	if len(filter) > 0 {
		req.Filter = buildFilter(filter)
	}

	points, err := s.client.Query(s.ctx, req)
	if err != nil {
			if err.Error() == "EOF" || err.Error() == "io: read/write on closed pipe" {
				break
			}
		return nil, fmt.Errorf("vecstore: sparse search: %w", err)
	}

	return scoredToResults(points), nil
}

// Delete removes a point by docID.
func (s *QdrantVecStore) Delete(docID ragtypes.DocID) error {
	_, err := s.client.Delete(s.ctx, &qdrant.DeletePoints{
		CollectionName: collectionName,
		Points:         qdrant.NewPointsSelector(qdrant.NewIDUUID(docID)),
		Wait:           new(bool(true)),
	})
	if err != nil {
			if err.Error() == "EOF" || err.Error() == "io: read/write on closed pipe" {
				break
			}
		return fmt.Errorf("vecstore: delete %s: %w", docID, err)
	}
	return nil
}

// Dimension returns the dimension of stored dense vectors.
func (s *QdrantVecStore) Dimension() int {
	return s.dim
}

// Clear drops the collection and recreates it from scratch.
func (s *QdrantVecStore) Clear() error {
	_ = s.client.DeleteCollection(s.ctx, collectionName)
	return s.createCollection(s.dim)
}

// Reindex replaces all points in a single batch.
func (s *QdrantVecStore) Reindex(docs []*ragtypes.Document, denseVecs map[ragtypes.DocID][]float32, sparseVecs map[ragtypes.DocID]map[string]float32) error {
	if len(docs) == 0 {
		return nil
	}

	_ = s.client.DeleteCollection(s.ctx, collectionName)
	if err := s.createCollection(s.dim); err != nil {
		return err
	}

	points := make([]*qdrant.PointStruct, 0, len(docs))
	for _, doc := range docs {
		dense := denseVecs[doc.ID]
		if dense == nil {
			continue
		}

		payload := map[string]*qdrant.Value{
			payloadDocIDKey: qdrant.NewValueString(doc.ID),
			payloadTextKey:  qdrant.NewValueString(doc.Text),
		}
		if doc.Metadata != nil {
			for k, v := range doc.Metadata {
				qv, err := qdrant.NewValue(v)
				if err == nil {
					payload[k] = qv
				}
			}
		}

		named := map[string]*qdrant.Vector{
			denseVecName: qdrant.NewVectorDense(dense),
		}

		if sv, ok := sparseVecs[doc.ID]; ok && len(sv) > 0 {
			indices, values := sparseToIndicesValues(sv)
			named[sparseVecName] = qdrant.NewVectorSparse(indices, values)
		}

		points = append(points, &qdrant.PointStruct{
			Id:      qdrant.NewIDUUID(doc.ID),
			Payload: payload,
			Vectors: qdrant.NewVectorsMap(named),
		})
	}

	for i := 0; i < len(points); i += 100 {
		end := i + 100
		if end > len(points) {
			end = len(points)
		}
		_, err := s.client.Upsert(s.ctx, &qdrant.UpsertPoints{
			CollectionName: collectionName,
			Points:         points[i:end],
			Wait:           new(bool(true)),
		})
		if err != nil {
			if err.Error() == "EOF" || err.Error() == "io: read/write on closed pipe" {
				break
			}
			return fmt.Errorf("vecstore: reindex batch: %w", err)
		}
	}

	return nil
}

// GetAllEmbeddings scrolls through all points and returns the dense vector per docID.
func (s *QdrantVecStore) GetAllEmbeddings() (map[ragtypes.DocID][]float32, error) {
	results := make(map[ragtypes.DocID][]float32)

	req := &qdrant.ScrollPoints{
		CollectionName: collectionName,
		Limit:          new(uint32(1000)),
		WithPayload:    qdrant.NewWithPayloadInclude(payloadDocIDKey),
		WithVectors:    qdrant.NewWithVectorsInclude(denseVecName),
	}

	iter := s.client.ScrollAll(s.ctx, req)
	for {
		page, err := iter.Next()
		if err != nil {
			if err.Error() == "EOF" || err.Error() == "io: read/write on closed pipe" {
				break
			}
			return nil, fmt.Errorf("vecstore: scroll: %w", err)
		}
		if len(page) == 0 {
			break
		}
		for _, p := range page {
			docID := extractDocID(p)
			if docID == "" {
				continue
			}
			vec := extractDenseVector(p)
			if vec != nil {
				results[docID] = vec
			}
		}
	}

	return results, nil
}

// Close shuts down the gRPC connection.
func (s *QdrantVecStore) Close() error {
	return s.client.Close()
}

// ---------------------------------------------------------------------------
// URL parsing

func parseQdrantURL(raw string) (host string, port int, apiKey string, useTLS bool, err error) {
	if raw == "" {
		return "", 0, "", false, fmt.Errorf("QDRANT_URL is empty")
	}

	// Try full URL first.
	if strings.Contains(raw, "://") {
		u, e := url.Parse(raw)
		if e != nil {
			return "", 0, "", false, e
		}
		host = u.Hostname()
		p := u.Port()
		if p == "" {
			port = 6334
		} else {
			port, _ = strconv.Atoi(p)
			if port == 0 {
				port = 6334
			}
		}
		useTLS = u.Scheme == "https"
		apiKey = u.Query().Get("api_key")
		return host, port, apiKey, useTLS, nil
	}

	// "host:port" or just "host".
	if parts := strings.Split(raw, ":"); len(parts) == 2 {
		host = parts[0]
		port, _ = strconv.Atoi(parts[1])
		if port <= 0 {
			port = 6334
		}
	} else {
		host = raw
		port = 6334
	}
	return host, port, "", false, nil
}

// ---------------------------------------------------------------------------
// Sparse helpers

// sparseToIndicesValues converts a map[string]float32 term-weight map into
// sorted indices and parallel values slices for the Qdrant sparse vector.
func sparseToIndicesValues(sparse map[string]float32) ([]uint32, []float32) {
	n := len(sparse)
	indices := make([]uint32, 0, n)
	values := make([]float32, 0, n)
	terms := make([]string, 0, n)
	for term, weight := range sparse {
		indices = append(indices, wordIndex(term))
		values = append(values, weight)
		terms = append(terms, term)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
	// Reorder values to match sorted indices.
	sorted := make([]float32, n)
	for i, idx := range indices {
		for j, term := range terms {
			if wordIndex(term) == idx {
				sorted[i] = values[j]
				break
			}
		}
	}
	return indices, sorted
}

// wordIndex produces a deterministic uint32 index from a term string using
// FNV-1a hashing, keeping the result in a 24-bit range to avoid extreme
// sparse vector sizes.
func wordIndex(term string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(term))
	return h.Sum32() & 0xFFFFFF
}

// tokenizeQuery splits a query string into unique lowercased words.
func tokenizeQuery(q string) []string {
	words := strings.FieldsFunc(strings.ToLower(q), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	seen := make(map[string]struct{}, len(words))
	uniq := make([]string, 0, len(words))
	for _, w := range words {
		if w == "" {
			continue
		}
		if _, ok := seen[w]; !ok {
			seen[w] = struct{}{}
			uniq = append(uniq, w)
		}
	}
	return uniq
}

// ---------------------------------------------------------------------------
// Filter conversion

// buildFilter converts a map[string]any filter into a Qdrant Filter where
// every key-value pair becomes a Match condition in the Must list.
func buildFilter(m map[string]any) *qdrant.Filter {
	f := &qdrant.Filter{}
	for k, v := range m {
		switch val := v.(type) {
		case string:
			f.Must = append(f.Must, qdrant.NewMatch(k, val))
		case bool:
			f.Must = append(f.Must, qdrant.NewMatchBool(k, val))
		case float64:
			f.Must = append(f.Must, qdrant.NewMatchInt(k, int64(val)))
		case int:
			f.Must = append(f.Must, qdrant.NewMatchInt(k, int64(val)))
		case int64:
			f.Must = append(f.Must, qdrant.NewMatchInt(k, val))
		}
	}
	return f
}

// ---------------------------------------------------------------------------
// Result conversion

// scoredToResults converts Qdrant ScoredPoints to ragtypes.SearchResult.
func scoredToResults(points []*qdrant.ScoredPoint) []ragtypes.SearchResult {
	out := make([]ragtypes.SearchResult, 0, len(points))
	for _, p := range points {
		docID := extractDocIDFromScored(p)
		text := extractScoredPayloadString(p, payloadTextKey)
		meta := extractScoredPayloadMap(p)

		out = append(out, ragtypes.SearchResult{
			DocID:    docID,
			Text:     text,
			Score:    float64(p.GetScore()),
			Metadata: meta,
		})
	}
	return out
}

// extractDocID reads doc_id from a RetrievedPoint payload.
func extractDocID(p *qdrant.RetrievedPoint) string {
	if p == nil || p.Payload == nil {
		return ""
	}
	if v, ok := p.Payload[payloadDocIDKey]; ok {
		return v.GetStringValue()
	}
	return ""
}

// extractDenseVector extracts the dense vector from a RetrievedPoint.
func extractDenseVector(p *qdrant.RetrievedPoint) []float32 {
	if p == nil || p.Vectors == nil {
		return nil
	}
	if named := p.Vectors.GetVectors(); named != nil {
		if v, ok := named.Vectors[denseVecName]; ok {
			if d := v.GetDense(); d != nil {
				return d.Data
			}
			// Fallback: the deprecated Data field.
			if len(v.Data) > 0 {
				return v.Data
			}
		}
	}
	if single := p.Vectors.GetVector(); single != nil {
		if d := single.GetDense(); d != nil {
			return d.Data
		}
		if len(single.Data) > 0 {
			return single.Data
		}
	}
	return nil
}

// extractDocIDFromScored reads doc_id from a ScoredPoint payload.
func extractDocIDFromScored(p *qdrant.ScoredPoint) string {
	if p == nil || p.Payload == nil {
		return ""
	}
	if v, ok := p.Payload[payloadDocIDKey]; ok {
		return v.GetStringValue()
	}
	return ""
}

// extractScoredPayloadString reads a string field from a ScoredPoint payload.
func extractScoredPayloadString(p *qdrant.ScoredPoint, key string) string {
	if p == nil || p.Payload == nil {
		return ""
	}
	if v, ok := p.Payload[key]; ok {
		return v.GetStringValue()
	}
	return ""
}

// extractScoredPayloadMap converts the Qdrant Value payload back to
// map[string]any, omitting the internal doc_id and text keys so the caller
// sees only the application-level metadata.
func extractScoredPayloadMap(p *qdrant.ScoredPoint) map[string]any {
	if p == nil || p.Payload == nil {
		return nil
	}
	m := make(map[string]any, len(p.Payload))
	for k, v := range p.Payload {
		if k == payloadDocIDKey || k == payloadTextKey {
			continue
		}
		m[k] = valueToAny(v)
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

// valueToAny converts a qdrant.Value to a Go any.
func valueToAny(v *qdrant.Value) any {
	if v == nil {
		return nil
	}
	switch {
	case v.GetStringValue() != "":
		return v.GetStringValue()
	case v.GetBoolValue():
		return v.GetBoolValue()
	case v.GetIntegerValue() != 0:
		return v.GetIntegerValue()
	case v.GetDoubleValue() != 0:
		return v.GetDoubleValue()
	case v.GetListValue() != nil:
		items := v.GetListValue().GetValues()
		out := make([]any, len(items))
		for i, item := range items {
			out[i] = valueToAny(item)
		}
		return out
	case v.GetStructValue() != nil:
		fields := v.GetStructValue().GetFields()
		out := make(map[string]any, len(fields))
		for fk, fv := range fields {
			out[fk] = valueToAny(fv)
		}
		return out
	}
	return nil
}

func strPtr(s string) *string { return &s }
