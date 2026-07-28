// Package viz generates graph visualizations: HTML (vis.js), DOT, JSON, ASCII.
package viz

import (
	"encoding/json"
	"fmt"
	"html"
	"sort"
	"strings"

	"github.com/Viktorokh96/graphrag-mcp/internal/color"
	"github.com/Viktorokh96/graphrag-mcp/internal/ragtypes"
)

// RenderHTML generates a self-contained HTML page with an interactive vis.js graph.
func RenderHTML(graph ragtypes.GraphStore, docStore ragtypes.DocumentStore) (string, error) {
	nodes := graph.AllNodes()
	edges := graph.AllEdges()

	// Build node map for labels
	nodeLabels := make(map[ragtypes.DocID]string)
	for _, nid := range nodes {
		doc, err := docStore.Get(nid)
		if err == nil && doc != nil {
			label := firstLine(doc.Text, 30)
			nodeLabels[nid] = html.EscapeString(label)
		} else {
			nodeLabels[nid] = nid[:8]
		}
	}

	// Assign colors via word hues
	type nodeVis struct {
		ID    string `json:"id"`
		Label string `json:"label"`
		Color string `json:"color"`
		Title string `json:"title,omitempty"`
	}
	type edgeVis struct {
		From  string `json:"from"`
		To    string `json:"to"`
		Label string `json:"label"`
	}

	var nodesVis []nodeVis
	for _, nid := range nodes {
		doc, _ := docStore.Get(nid)
		text := ""
		if doc != nil {
			text = doc.Text
		}
		hsl := color.DocColor(text, 0.6, 0.45)
		r, g, b := color.HSLToRGB(hsl)
		nodesVis = append(nodesVis, nodeVis{
			ID:    nid,
			Label: nodeLabels[nid],
			Color: fmt.Sprintf("#%02x%02x%02x", r, g, b),
			Title: html.EscapeString(firstLine(text, 60)),
		})
	}

	relColorMap := map[string]string{
		"related_to":    "#4CAF50",
		"similar_to":    "#2196F3",
		"prerequisite":  "#FF9800",
		"supersedes":    "#E91E63",
		"appears_in":    "#9C27B0",
		"sibling_chunk": "#00BCD4",
	}

	var edgesVis []edgeVis
	for _, e := range edges {
		clr := relColorMap[e.Relation]
		if clr == "" {
			clr = "#607D8B"
		}
		edgesVis = append(edgesVis, edgeVis{
			From:  e.Source,
			To:    e.Target,
			Label: fmt.Sprintf("%s (%.1f)", e.Relation, e.Weight),
		})
	}

	nodesJSON, _ := json.Marshal(nodesVis)
	edgesJSON, _ := json.Marshal(edgesVis)

	return fmt.Sprintf(htmlTemplate, len(nodes), len(edges), nodesJSON, edgesJSON), nil
}

// RenderDOT generates a Graphviz DOT representation.
func RenderDOT(graph ragtypes.GraphStore, docStore ragtypes.DocumentStore) (string, error) {
	var b strings.Builder
	b.WriteString("digraph KnowledgeGraph {\n")
	b.WriteString("  rankdir=LR;\n")
	b.WriteString("  node [shape=box, style=filled, fillcolor=\"#1a1a2e\", fontcolor=\"#e0e0e0\"];\n")

	nodes := graph.AllNodes()
	for _, nid := range nodes {
		doc, err := docStore.Get(nid)
		label := nid[:8]
		if err == nil && doc != nil {
			label = firstLine(doc.Text, 25)
		}
		fmt.Fprintf(&b, "  %q [label=%q];\n", nid, label)
	}

	for _, e := range graph.AllEdges() {
		fmt.Fprintf(&b, "  %q -> %q [label=%q];\n", e.Source, e.Target, e.Relation)
	}

	b.WriteString("}\n")
	return b.String(), nil
}

// RenderJSON exports the graph as JSON.
func RenderJSON(graph ragtypes.GraphStore, docStore ragtypes.DocumentStore) (string, error) {
	nodes := graph.AllNodes()
	nodeData := make(map[ragtypes.DocID]map[string]any)
	for _, nid := range nodes {
		doc, err := docStore.Get(nid)
		if err == nil && doc != nil {
			nodeData[nid] = map[string]any{
				"text":     doc.Text,
				"metadata": doc.Metadata,
			}
		}
	}
	edges := graph.AllEdges()
	result := map[string]any{
		"nodes": nodeData,
		"edges": edges,
	}
	b, _ := json.MarshalIndent(result, "", "  ")
	return string(b), nil
}

// StatsLine returns a human-readable stats summary.
func StatsLine(graph ragtypes.GraphStore) string {
	stats := graph.Stats()
	return fmt.Sprintf("%d nodes · %d edges · %d relation types",
		stats.TotalNodes, stats.TotalEdges, len(stats.RelationTypes))
}

// LegendHTML returns an HTML legend for the graph.
func LegendHTML(graph ragtypes.GraphStore, docStore ragtypes.DocumentStore) string {
	nodes := graph.AllNodes()
	sort.Strings(nodes)

	var b strings.Builder
	b.WriteString(`<div class="legend">`)
	for i, nid := range nodes {
		if i >= 20 {
			b.WriteString(fmt.Sprintf(`<div>... and %d more</div>`, len(nodes)-20))
			break
		}
		doc, _ := docStore.Get(nid)
		label := nid[:8]
		if doc != nil {
			label = firstLine(doc.Text, 40)
		}
		b.WriteString(fmt.Sprintf(`<div>%s</div>`, html.EscapeString(label)))
	}
	b.WriteString(`</div>`)
	return b.String()
}

func firstLine(text string, maxLen int) string {
	line := strings.SplitN(text, "\n", 2)[0]
	line = strings.TrimSpace(line)
	if len(line) > maxLen {
		line = line[:maxLen-3] + "..."
	}
	return line
}

const htmlTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Knowledge Graph — %d nodes, %d edges</title>
<script src="https://unpkg.com/vis-network@9.1.6/dist/vis-network.min.js"></script>
<style>
  body { margin: 0; background: #1a1a2e; font-family: sans-serif; }
  #graph { width: 100vw; height: 100vh; }
</style>
</head>
<body>
<div id="graph"></div>
<script>
  var nodes = new vis.DataSet(%s);
  var edges = new vis.DataSet(%s);
  var container = document.getElementById('graph');
  var data = { nodes: nodes, edges: edges };
  var options = {
    nodes: { font: { color: '#fff', size: 12 } },
    edges: { arrows: 'to', font: { size: 9, color: '#aaa' } },
    physics: { solver: 'forceAtlas2Based' }
  };
  new vis.Network(container, data, options);
</script>
</body>
</html>`
