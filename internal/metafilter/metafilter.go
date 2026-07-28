// Package metafilter provides metadata filtering helpers for the RAG system.
package metafilter

// Matches checks whether a document's metadata satisfies all filter conditions (AND).
// Each key in the filter must match: scalar → exact match; []any → membership ($in).
func Matches(meta map[string]any, filter map[string]any) bool {
	if filter == nil || len(filter) == 0 {
		return true
	}
	if meta == nil {
		return false
	}
	for key, want := range filter {
		have, ok := meta[key]
		if !ok {
			return false
		}
		if !matchValue(have, want) {
			return false
		}
	}
	return true
}

func matchValue(have, want any) bool {
	switch w := want.(type) {
	case []any:
		for _, item := range w {
			if equals(have, item) {
				return true
			}
		}
		return false
	default:
		return equals(have, want)
	}
}

func equals(a, b any) bool {
	// Simple comparison; production would handle numeric type coercion.
	return a == b
}

// NormalizeFilter ensures the filter is a clean map or nil.
func NormalizeFilter(raw any) map[string]any {
	if raw == nil {
		return nil
	}
	switch v := raw.(type) {
	case map[string]any:
		if len(v) == 0 {
			return nil
		}
		return v
	default:
		return nil
	}
}
