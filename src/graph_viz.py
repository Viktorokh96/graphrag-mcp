"""Визуализация графа знаний: HTML (vis.js), DOT, JSON, ASCII."""

import http.server
import json
import sys
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Optional

from src.graph_store import GraphStore
from src.rag import RAGSystem

RELATION_COLORS = [
    "#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#795548", "#607D8B", "#3F51B5",
    "#8BC34A", "#FFC107", "#F44336", "#009688", "#673AB7",
]

GROUP_PALETTE = [
    ("#1b4332", "#40916c"), ("#0d47a1", "#42a5f5"), ("#01579b", "#4fc3f7"),
    ("#004d40", "#4db6ac"), ("#bf360c", "#ff7043"), ("#4a148c", "#ce93d8"),
    ("#2d1b69", "#7c3aed"), ("#7f1d1d", "#ef4444"), ("#006064", "#4dd0e1"),
    ("#1a237e", "#7986cb"), ("#33691e", "#8bc34a"), ("#e65100", "#ffa726"),
    ("#4e342e", "#a1887f"), ("#263238", "#90a4ae"), ("#3e2723", "#bcaaa4"),
]

SHAPE_CYCLE = ["box", "diamond", "ellipse", "hexagon", "star", "square",
               "triangle", "triangleDown", "circle", "database"]


class _VizGraph:
    """Адаптер GraphStore → структуры, ожидаемые рендерерами визуализации.

    Новый GraphStore хранит рёбра в SQL + NetworkX-кеше, а тексты/метаданные
    узлов — в DocumentStore, тогда как рендереры графа написаны под старый
    интерфейс (``graph._nodes`` — dict node_id→{text, metadata}; ``graph._edges``
    — dict edge_key→{source, target, relation, weight}). Этот адаптер один раз
    собирает обе структуры из GraphStore + DocumentStore, а всё остальное
    (get_related, get_edges_batch, stats, …) делегирует обёрнутому стору.

    Снимок строится в конструкторе — визуализация read-only, поэтому кеш
    не устаревает в пределах рендера.
    """

    def __init__(self, store: "GraphStore"):
        self._store = store
        docs, _ = store._docs.list(limit=max(store._docs.count(), 1), offset=0)
        self._nodes: dict[str, dict] = {
            d["doc_id"]: {"text": d["text"], "metadata": d["metadata"]} for d in docs
        }
        self._edges: dict[str, dict] = {}
        for source, target, relation, weight in store.get_all_edges():
            self._edges[f"{source}::{relation}::{target}"] = {
                "source": source,
                "target": target,
                "relation": relation,
                "weight": weight,
            }

    def __getattr__(self, name):
        # Делегируем методы обёрнутого стора (get_related, get_edges_batch, ...)
        return getattr(self._store, name)


def _as_viz(graph) -> _VizGraph:
    """Обернуть GraphStore в адаптер (идемпотентно)."""
    return graph if isinstance(graph, _VizGraph) else _VizGraph(graph)


def _get_relation_color(relation: str, color_map: dict[str, str]) -> str:
    if relation not in color_map:
        color_map[relation] = RELATION_COLORS[len(color_map) % len(RELATION_COLORS)]
    return color_map[relation]


def _truncate(text: str, max_len: int = 40) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _make_label(text: str, max_chars: int = 35) -> str:
    first_line = text.split("\n")[0].strip()
    if len(first_line) > max_chars:
        first_line = first_line[:max_chars - 3] + "..."
    if len(first_line) > 18:
        mid = len(first_line) // 2
        split = first_line.rfind(" ", 4, mid + 8)
        if split > 4:
            first_line = first_line[:split] + "\n" + first_line[split + 1:]
    return first_line


def _build_subgraph(
    graph: GraphStore,
    max_nodes: Optional[int] = None,
    relation_type: Optional[list[str]] = None,
    focus_node: Optional[str] = None,
    max_depth: int = 2,
) -> tuple[set[str], list[dict]]:
    if focus_node:
        if focus_node not in graph._nodes:
            raise ValueError(f"Node '{focus_node}' not found in graph")
        included = {focus_node}
        for source, target, _rel, _w, _dir in graph.get_related(focus_node, max_depth=max_depth):
            included.add(source)
            included.add(target)
    else:
        included = set(graph._nodes.keys())

    included &= set(graph._nodes.keys())

    edges = [
        e for e in graph._edges.values()
        if e["source"] in included and e["target"] in included
        and (relation_type is None or e["relation"] in relation_type)
    ]

    if max_nodes is not None and len(included) > max_nodes:
        degree: dict[str, int] = {}
        for e in edges:
            degree[e["source"]] = degree.get(e["source"], 0) + 1
            degree[e["target"]] = degree.get(e["target"], 0) + 1
        kept = set(sorted(included, key=lambda n: degree.get(n, 0), reverse=True)[:max_nodes])
        edges = [e for e in edges if e["source"] in kept and e["target"] in kept]
        included = kept

    return included, edges


def _render_html(
    graph: GraphStore, output_path: str,
    included: set[str], edges: list[dict], layout: str,
    api_base_url: str = "",
) -> None:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    G = nx.Graph()
    for nid in included:
        if nid in graph._nodes:
            G.add_node(nid)
    for e in edges:
        G.add_edge(e["source"], e["target"])

    try:
        communities = list(louvain_communities(G, seed=42))
    except Exception:
        communities = [set(G.nodes())]

    node_comm: dict[str, int] = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node_comm[n] = i

    deg = Counter()
    for e in edges:
        deg[e["source"]] += 1
        deg[e["target"]] += 1

    nodes_js = []
    for nid in included:
        node = graph._nodes.get(nid)
        if node is None:
            continue
        ci = node_comm.get(nid, 0)
        bg, border = GROUP_PALETTE[ci % len(GROUP_PALETTE)]
        shape = SHAPE_CYCLE[ci % len(SHAPE_CYCLE)]

        label = _make_label(node["text"])
        d = deg.get(nid, 0)
        fs = 12 + min(d * 0.5, 8)

        meta_str = json.dumps(node.get("metadata", {}), ensure_ascii=False)
        nid_esc = nid.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        meta_esc = meta_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if api_base_url:
            preview = ""
            title = (
                f"<b>ID:</b> {nid_esc}<br>"
                f"<b>Degree:</b> {d}<br>"
                f"<b>Metadata:</b> {meta_esc}"
            )
        else:
            preview = node["text"][:300].replace("<", "&lt;").replace(">", "&gt;")
            title = (
                f"<b>ID:</b> {nid_esc}<br>"
                f"<b>Preview:</b> {preview}<br>"
                f"<b>Degree:</b> {d}<br>"
                f"<b>Metadata:</b> {meta_esc}"
            )

        node_entry = {
            "id": nid,
            "label": label,
            "group": f"c{ci}",
            "title": title,
            "color": {"background": bg, "border": border},
            "font": {"size": fs, "color": "#fff", "face": "Segoe UI"},
            "shape": shape,
            "borderWidth": 2,
            "margin": {"top": 6, "bottom": 6, "left": 8, "right": 8},
        }
        if not api_base_url:
            node_entry["preview"] = preview
        nodes_js.append(node_entry)

    rel_color_map: dict[str, str] = {}
    edges_js = []
    for e in edges:
        if e["relation"] not in rel_color_map:
            rel_color_map[e["relation"]] = GROUP_PALETTE[
                len(rel_color_map) % len(GROUP_PALETTE)
            ][1]
        edges_js.append({
            "from": e["source"],
            "to": e["target"],
            "label": f"{e['relation']} {e['weight']:.2f}",
            "arrows": "to",
            "color": {"color": rel_color_map[e["relation"]], "opacity": 0.65},
            "width": 0.5 + e["weight"] * 1.5,
            "smooth": {"type": "curvedCW", "roundness": 0.15},
            "font": {"size": 9, "color": "#aaa", "strokeWidth": 0,
                      "align": "middle", "face": "Segoe UI"},
        })

    groups_js = {}
    for i, comm in enumerate(communities):
        bg, border = GROUP_PALETTE[i % len(GROUP_PALETTE)]
        shape = SHAPE_CYCLE[i % len(SHAPE_CYCLE)]
        groups_js[f"c{i}"] = {"shape": shape, "color": {"background": bg, "border": border}}

    legend_rows = []
    for i, comm in enumerate(communities):
        bg, border = GROUP_PALETTE[i % len(GROUP_PALETTE)]
        sample = graph._nodes.get(next(iter(comm)), {})
        lbl = sample.get("text", "").split("\n")[0].strip()[:30]
        sz = len(comm)
        legend_rows.append(
            f'<div class="row"><span class="dot" style="background:{bg};border-color:{border}"></span> '
            f'<span title="{lbl}">C{i+1}: {lbl}</span> <span class="count">{sz}</span></div>'
        )

    nodes_json_raw = json.dumps(nodes_js, ensure_ascii=False)
    edges_json_raw = json.dumps(edges_js, ensure_ascii=False)
    groups_json_raw = json.dumps(groups_js, ensure_ascii=False)
    stats_line = f"<strong>{len(included)}</strong> узлов · <strong>{len(edges)}</strong> связей · <span id='vis-nodes'>0</span> видимых"
    api_base_js = json.dumps(api_base_url or "")

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Knowledge Graph</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,-apple-system,sans-serif; background:#1a1a2e; height:100vh; overflow:hidden; color:#e0e0e0; }}
#mynetwork {{ width:100%; height:100vh; background:#1a1a2e; }}
#legend {{
  position:fixed; bottom:24px; left:24px; z-index:1000;
  background:rgba(15,15,35,0.92); color:#e0e0e0;
  padding:16px 20px; border-radius:14px; font-size:13px; line-height:2;
  border:1px solid rgba(255,255,255,0.08);
  backdrop-filter:blur(12px); box-shadow:0 8px 32px rgba(0,0,0,0.5);
  min-width:180px; user-select:none; max-height:50vh; overflow-y:auto;
}}
#legend .title {{ font-weight:700; font-size:14px; margin-bottom:6px; letter-spacing:0.5px; color:#fff; }}
#legend .row {{ display:flex; align-items:center; gap:10px; }}
#legend .dot {{ display:inline-block; width:14px; height:14px; border-radius:4px; flex-shrink:0; border:2px solid; }}
#legend .count {{ color:#666; font-size:11px; margin-left:auto; }}
#toolbar {{
  position:fixed; top:16px; right:16px; z-index:1000;
  display:flex; gap:8px;
}}
#toolbar button {{
  background:rgba(15,15,35,0.85); color:#e0e0e0; border:1px solid rgba(255,255,255,0.1);
  padding:6px 12px; border-radius:8px; cursor:pointer; font-size:12px;
  backdrop-filter:blur(8px); transition:all .2s; font-family:inherit;
}}
#toolbar button:hover {{ background:rgba(255,255,255,0.12); border-color:rgba(255,255,255,0.2); }}
#stats {{
  position:fixed; top:16px; left:50%; transform:translateX(-50%); z-index:1000;
  background:rgba(15,15,35,0.85); color:#999;
  padding:6px 18px; border-radius:20px; font-size:12px;
  border:1px solid rgba(255,255,255,0.06);
  backdrop-filter:blur(8px); pointer-events:none;
}}
#stats strong {{ color:#e0e0e0; }}
#search-box {{
  position:fixed; top:52px; right:16px; z-index:1000;
  background:rgba(15,15,35,0.85); border:1px solid rgba(255,255,255,0.1);
  border-radius:8px; padding:2px; backdrop-filter:blur(8px); display:none;
}}
#search-box.visible {{ display:flex; }}
#search-box input {{
  background:transparent; border:none; color:#e0e0e0; padding:6px 12px;
  font-size:13px; outline:none; width:200px; font-family:inherit;
}}
#search-box input::placeholder {{ color:#666; }}
#search-box .count {{ padding:6px 10px; font-size:11px; color:#666; white-space:nowrap; }}
#search-results {{
  position:fixed; top:92px; right:16px; z-index:1000;
  background:rgba(15,15,35,0.94); border:1px solid rgba(255,255,255,0.1);
  border-radius:8px; backdrop-filter:blur(8px); display:none;
  max-height:320px; overflow-y:auto; min-width:260px;
  box-shadow:0 8px 32px rgba(0,0,0,0.5);
}}
#search-results.visible {{ display:block; }}
#search-results .sresult {{
  padding:7px 12px; cursor:pointer; font-size:12px; color:#ccc;
  border-bottom:1px solid rgba(255,255,255,0.04); display:flex; align-items:center; gap:8px;
}}
#search-results .sresult:hover {{ background:rgba(255,255,255,0.08); color:#fff; }}
#search-results .sresult .sr-label {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
#search-results .sresult .sr-score {{ color:#666; font-size:11px; white-space:nowrap; }}
#toast {{
  position:fixed; bottom:100px; left:50%; transform:translateX(-50%); z-index:999;
  background:rgba(15,15,35,0.9); color:#e0e0e0;
  padding:8px 20px; border-radius:10px; font-size:13px;
  border:1px solid rgba(255,255,255,0.08);
  backdrop-filter:blur(8px); opacity:0; transition:opacity .3s; pointer-events:none;
}}
#toast.show {{ opacity:1; }}
#sidepanel {{
  position:fixed; top:0; right:-420px; width:400px; height:100vh; z-index:999;
  background:rgba(10,10,30,0.96); color:#e0e0e0;
  padding:20px; font-size:13px; line-height:1.6; overflow-y:auto;
  border-left:1px solid rgba(255,255,255,0.08);
  backdrop-filter:blur(12px); box-shadow:-4px 0 32px rgba(0,0,0,0.6);
  transition:right .3s ease; font-family:'Segoe UI',system-ui,sans-serif;
}}
#sidepanel.open {{ right:0; }}
#sidepanel .close {{
  position:sticky; top:0; float:right; cursor:pointer; font-size:20px;
  color:#888; background:none; border:none; padding:0 0 0 12px; line-height:1;
}}
#sidepanel .close:hover {{ color:#fff; }}
#sidepanel .sid {{
  font-size:11px; color:#666; word-break:break-all; margin-bottom:12px;
}}
#sidepanel .scontent {{
  white-space:pre-wrap; word-break:break-word; font-size:13px; color:#ccc;
}}
#sidepanel .sp-links {{
  margin-top: 16px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.08);
}}
#sidepanel .sp-links .lhead {{
  font-size: 11px; color: #888; margin-bottom: 8px; font-weight: 600; letter-spacing: 0.5px;
}}
#sidepanel .sp-links .lrow {{
  display: flex; align-items: center; gap: 8px; padding: 4px 6px; font-size: 12px;
  cursor:pointer; border-radius:4px; margin:0 -6px;
}}
#sidepanel .sp-links .lrow:hover {{ background:rgba(255,255,255,0.06); }}
#sidepanel .sp-links .lrow .arrow {{ color: #666; font-family: monospace; }}
#sidepanel .sp-links .lrow .rtype {{ color: #666; font-size: 11px; margin-left: auto; }}
</style>
</head>
<body>

<div id="stats">{stats_line}</div>

<div id="toolbar">
  <button onclick="fitGraph()" title="Fit to screen">⟷ Fit</button>
  <button onclick="toggleSearch()" title="Search nodes (Ctrl+F)" id="btn-search">🔍</button>
  <button onclick="resetPhysics()" title="Reset search and show all nodes">⟳ Reset</button>
</div>

<div id="search-box">
  <input type="text" placeholder="Search nodes by label or text..." id="search-input"
         oninput="searchNodes(this.value)" onkeydown="if(event.key==='Escape')hideSearch()">
  <span class="count" id="search-count"></span>
</div>

<div id="search-results"></div>

<div id="mynetwork"></div>

<div id="legend">
  <div class="title">🎯 Community</div>
  {"".join(legend_rows)}
</div>

<div id="toast"></div>

<div id="sidepanel">
  <button class="close" onclick="closeSidepanel()">&times;</button>
  <div class="sid" id="sp-id"></div>
  <div class="scontent" id="sp-content"></div>
  <div class="sp-links" id="sp-links"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<script>
const nodes = new vis.DataSet({nodes_json_raw});
const edges = new vis.DataSet({edges_json_raw});
const groups = {groups_json_raw};
const API_BASE = {api_base_js};

const container = document.getElementById('mynetwork');
const data = {{ nodes, edges }};
const options = {{
  physics: {{
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{
      gravitationalConstant: -80,
      centralGravity: 0.003,
      springLength: 220,
      springConstant: 0.025,
      damping: 0.45,
    }},
    stabilization: {{ iterations: 300, updateInterval: 25 }},
    adaptiveTimestep: true,
  }},
  layout: {{ improvedLayout: true, randomSeed: 42 }},
  groups,
  edges: {{
    font: {{ size: 9, color: '#aaa', strokeWidth: 0, align: 'middle', face: 'Segoe UI' }},
    width: 1.5,
    smooth: {{ type: 'curvedCW', roundness: 0.15 }},
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.6 }} }},
    selectionWidth: 2,
    hoverWidth: 0,
  }},
  nodes: {{
    font: {{ color: '#fff', size: 12, face: 'Segoe UI' }},
    borderWidth: 2,
    borderWidthSelected: 3,
    shadow: {{ enabled: true, size: 6, x: 0, y: 2 }},
    margin: {{ top: 8, bottom: 8, left: 10, right: 10 }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 150,
    navigationButtons: true,
    keyboard: {{ enabled: true, speed: {{ x: 10, y: 10, zoom: 0.02 }} }},
    multiselect: false,
  }},
}};

const network = new vis.Network(container, data, options);

network.on('afterDrawing', function() {{
  document.getElementById('vis-nodes').textContent = nodes.length;
}});

function fitGraph() {{
  network.fit({{ animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }});
}}
function esc(str) {{ return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
function clearAllHighlights() {{
  nodes.forEach(n => {{
    n.hidden = false;
    if (n._origBorder) {{
      n.color.border = n._origBorder;
      delete n._origBorder;
    }}
    if (n._origLabel) {{
      n.label = n._origLabel;
      delete n._origLabel;
    }}
    if (n._origFontSize) {{
      n.font = n.font || {{}};
      n.font.size = n._origFontSize;
      delete n._origFontSize;
    }}
    nodes.update(n);
  }});
  document.getElementById('search-results').innerHTML = '';
  document.getElementById('search-results').classList.remove('visible');
  document.getElementById('search-count').textContent = '';
}}
function resetPhysics() {{
  clearAllHighlights();
  closeSidepanel();
  document.getElementById('search-box').classList.remove('visible');
  document.getElementById('search-input').value = '';
  document.getElementById('search-results').innerHTML = '';
  document.getElementById('search-results').classList.remove('visible');
  document.getElementById('search-count').textContent = '';
  fitGraph();
}}
function toggleSearch() {{
  const box = document.getElementById('search-box');
  box.classList.toggle('visible');
  if (box.classList.contains('visible')) {{
    document.getElementById('search-input').focus();
  }} else {{
    document.getElementById('search-input').value = '';
    document.getElementById('search-results').innerHTML = '';
    document.getElementById('search-results').classList.remove('visible');
    searchNodes('');
  }}
}}
function hideSearch() {{
  document.getElementById('search-box').classList.remove('visible');
  document.getElementById('search-input').value = '';
  document.getElementById('search-results').innerHTML = '';
  document.getElementById('search-results').classList.remove('visible');
  searchNodes('');
}}
let searchTimeout = null;
function searchNodes(query) {{
  clearTimeout(searchTimeout);
  const q = query.trim();
  const count = document.getElementById('search-count');
  const resultsEl = document.getElementById('search-results');

  if (!q) {{
    clearAllHighlights();
    count.textContent = '';
    fitGraph();
    return;
  }}

  function renderResultsList(results, maxShow) {{
    maxShow = maxShow || 10;
    const parts = [];
    for (let i = 0; i < Math.min(results.length, maxShow); i++) {{
      const r = results[i];
      const label = esc(r.label || r.text.split('\\n')[0].trim().slice(0, 40));
      parts.push(
        '<div class="sresult" data-id="' + r.doc_id + '">' +
        '<span class="sr-label">' + label + '</span>' +
        '<span class="sr-score">' + (r.score != null ? r.score.toFixed(3) : '') + '</span></div>'
      );
    }}
    if (parts.length) {{
      resultsEl.innerHTML = parts.join('');
      resultsEl.classList.add('visible');
      resultsEl.querySelectorAll('.sresult').forEach(el => {{
        el.addEventListener('click', function() {{ goToNode(this.dataset.id); }});
      }});
    }} else {{
      resultsEl.innerHTML = '';
      resultsEl.classList.remove('visible');
    }}
  }}

  if (API_BASE) {{
    searchTimeout = setTimeout(() => {{
      fetch(API_BASE + '/api/search?q=' + encodeURIComponent(q))
        .then(r => r.json())
        .then(data => {{
          const matchIds = new Set(data.results.map(r => r.doc_id));
          const scores = data.results.map(r => r.score);
          const maxScore = Math.max(...scores) || 1;
          const scoreMap = {{}};
          for (const r of data.results) scoreMap[r.doc_id] = r.score;
          let matchCount = 0;
          nodes.forEach(n => {{
            if (matchIds.has(n.id)) {{
              n.hidden = false;
              if (!n._origBorder) n._origBorder = n.color.border;
              n.color.border = '#FFD700';
              const sc = scoreMap[n.id];
              const ratio = sc !== undefined ? sc / maxScore : 0;
              if (!n._origLabel) n._origLabel = n.label;
              if (!n._origFontSize) n._origFontSize = (n.font && n.font.size) || 12;
              n.font = n.font || {{}};
              n.font.size = Math.round(10 + ratio * 16);
              n.label = n._origLabel + '\\n' + (sc != null ? sc.toFixed(3) : '');
              matchCount++;
            }} else {{
              n.hidden = true;
              if (n._origBorder) {{
                n.color.border = n._origBorder;
                delete n._origBorder;
              }}
              if (n._origLabel) {{
                n.label = n._origLabel;
                delete n._origLabel;
              }}
              if (n._origFontSize) {{
                n.font = n.font || {{}};
                n.font.size = n._origFontSize;
                delete n._origFontSize;
              }}
            }}
            nodes.update(n);
          }});
          count.textContent = matchCount + ' found';
          renderResultsList(data.results);
          fitGraph();
        }})
        .catch(err => {{
          count.textContent = 'Error: ' + err.message;
        }});
    }}, 300);
  }} else {{
    const ids = nodes.getIds();
    let matchCount = 0;
    const localResults = [];
    ids.forEach(id => {{
      const n = nodes.get(id);
      const label = (n.label || '').toLowerCase();
      const title = (n.title || '').toLowerCase();
      const match = label.includes(q.toLowerCase()) || title.includes(q.toLowerCase());
      if (match) {{
        n.hidden = false;
        if (!n._origBorder) n._origBorder = n.color.border;
        n.color.border = '#FFD700';
        matchCount++;
        localResults.push({{doc_id: n.id, label: n.label || '', score: null}});
      }} else {{
        n.hidden = true;
        if (n._origBorder) {{
          n.color.border = n._origBorder;
          delete n._origBorder;
        }}
      }}
      nodes.update(n);
    }});
    count.textContent = matchCount + ' found';
    renderResultsList(localResults);
    fitGraph();
  }}
}}
function toast(msg) {{
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 2000);
}}

function showDocument(nid) {{
  const spId = document.getElementById('sp-id');
  const spContent = document.getElementById('sp-content');
  const spLinks = document.getElementById('sp-links');
  const panel = document.getElementById('sidepanel');
  if (API_BASE) {{
    spId.textContent = 'ID: ' + nid + ' (loading...)';
    spContent.textContent = '';
    spLinks.innerHTML = '';
    panel.classList.add('open');
    fetch(API_BASE + '/api/document/' + encodeURIComponent(nid))
      .then(r => r.json())
      .then(data => {{
        spId.textContent = 'ID: ' + nid;
        spContent.textContent = data.text || '';
        renderLinks(data.links, spLinks);
      }})
      .catch(err => {{
        spContent.textContent = 'Error: ' + err.message;
      }});
  }} else {{
    const node = nodes.get(nid);
    if (node && node.preview) {{
      spId.textContent = 'ID: ' + nid;
      spContent.textContent = node.preview;
      panel.classList.add('open');
    }}
  }}
}}

function goToNode(nid) {{
  if (!nodes.get(nid)) {{ toast('Node not in current view'); return; }}
  network.focus(nid, {{ scale: 1.8, animation: {{ duration: 400, easingFunction: 'easeInOutQuad' }} }});
  network.selectNodes([nid]);
  showDocument(nid);
}}

document.addEventListener('keydown', e => {{
  if ((e.ctrlKey || e.metaKey) && e.key === 'f') {{
    e.preventDefault();
    toggleSearch();
  }}
}});

network.once('stabilizationIterationsDone', function() {{
  fitGraph();
  setTimeout(fitGraph, 500);
  toast('✅ Graph stabilized — {len(included)} nodes, {len(edges)} edges');
}});

network.on('doubleClick', function(params) {{
  if (params.nodes.length > 0) {{
    network.focus(params.nodes[0], {{ scale: 1.8, animation: {{ duration: 400, easingFunction: 'easeInOutQuad' }} }});
  }}
}});

function renderLinks(links, el) {{
  if (!links || links.length === 0) {{
    el.innerHTML = '';
    return;
  }}
  const parts = ['<div class="lhead">🔗 Related (depth=1)</div>'];
  for (const l of links) {{
    const arrow = l.direction === 'out' ? '→' : '←';
    parts.push(
      '<div class="lrow" data-id="' + l.id + '">' +
      '<span class="arrow">' + arrow + '</span>' +
      '<span>' + esc(l.label || l.id) + '</span>' +
      '<span class="rtype">' + l.relation + '</span>' +
      '</div>'
    );
  }}
  el.innerHTML = parts.join('');
  el.querySelectorAll('.lrow').forEach(row => {{
    row.addEventListener('click', function() {{ goToNode(this.dataset.id); }});
  }});
}}

network.on('click', function(params) {{
  if (params.nodes.length > 0) {{
    goToNode(params.nodes[0]);
  }}
}});

function closeSidepanel() {{
  document.getElementById('sidepanel').classList.remove('open');
  const spContent = document.getElementById('sp-content');
  const spId = document.getElementById('sp-id');
  const spLinks = document.getElementById('sp-links');
  spContent.textContent = '';
  spId.textContent = '';
  spLinks.innerHTML = '';
}}
</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Saved: {output_path}", file=sys.stderr)


def _render_dot(
    graph: GraphStore, output_path: str,
    included: set[str], edges: list[dict], layout: str,
) -> None:
    lines = ["digraph KnowledgeGraph {"]
    lines.append('  rankdir=LR;')
    lines.append('  splines=true;')
    lines.append('  node [style=filled, fillcolor="#ADD8E6"];')
    lines.append('  edge [fontsize=10];')

    for node_id in included:
        node = graph._nodes.get(node_id)
        if node is None:
            continue
        label = _truncate(node["text"]).replace('"', '\\"')
        lines.append(f'  "{node_id}" [label="{label}"];')

    color_map: dict[str, str] = {}
    for e in edges:
        c = _get_relation_color(e["relation"], color_map)
        lines.append(
            f'  "{e["source"]}" -> "{e["target"]}"'
            f' [label="{e["relation"]}", color="{c}", penwidth={e["weight"]:.1f}];'
        )

    lines.append("}")
    Path(output_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"Saved: {output_path}", file=sys.stderr)


def _render_json(
    graph: GraphStore, output_path: str,
    included: set[str], edges: list[dict], layout: str,
) -> None:
    data = {
        "nodes": [
            {"id": nid, "text": graph._nodes[nid]["text"], "metadata": graph._nodes[nid].get("metadata", {})}
            for nid in sorted(included) if nid in graph._nodes
        ],
        "edges": [
            {"source": e["source"], "target": e["target"], "relation": e["relation"], "weight": e["weight"]}
            for e in edges
        ],
    }
    Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved: {output_path}", file=sys.stderr)


def _render_ascii(
    graph: GraphStore, output_path: str,
    included: set[str], edges: list[dict], layout: str,
) -> None:
    lines = [f"Knowledge Graph: {len(included)} nodes, {len(edges)} edges"]

    degree: dict[str, int] = {}
    for e in edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1

    top = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:20]
    lines.append(f"\nTop-{len(top)} nodes by degree:")
    for nid, deg in top:
        node = graph._nodes.get(nid)
        label = _truncate(node["text"], 60) if node else "(unknown)"
        lines.append(f"  {nid[:8]}.. [{deg}] {label}")

    rel_counts: dict[str, int] = {}
    for e in edges:
        rel_counts[e["relation"]] = rel_counts.get(e["relation"], 0) + 1
    lines.append(f"\nRelation types ({len(rel_counts)}):")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {rel}: {cnt}")

    show = min(50, len(edges))
    lines.append(f"\nEdges (showing {show} of {len(edges)}):")
    for e in edges[:show]:
        src = graph._nodes.get(e["source"], {})
        tgt = graph._nodes.get(e["target"], {})
        sl = _truncate(src.get("text", "?"), 30)
        tl = _truncate(tgt.get("text", "?"), 30)
        lines.append(f"  {sl} --[{e['relation']}]--> {tl}")

    content = '\n'.join(lines)
    if output_path == "-":
        print(content)
    else:
        Path(output_path).write_text(content, encoding='utf-8')
        print(f"Saved: {output_path}", file=sys.stderr)


def render_graph_viz(
    graph: GraphStore,
    output_path: str = "graph_viz.html",
    output_format: str = "html",
    max_nodes: Optional[int] = None,
    relation_type: Optional[list[str]] = None,
    focus_node: Optional[str] = None,
    max_depth: int = 2,
    layout: str = "kamada_kawai",
    api_base_url: str = "",
) -> None:
    graph = _as_viz(graph)
    included, edges = _build_subgraph(graph, max_nodes, relation_type, focus_node, max_depth)

    if not included:
        print("Empty graph — nothing to render", file=sys.stderr)
        return

    renderers = {
        "html": lambda g, o, inc, e, lay: _render_html(g, o, inc, e, lay, api_base_url=api_base_url),
        "dot": _render_dot,
        "json": _render_json,
        "ascii": _render_ascii,
    }

    renderer = renderers.get(output_format)
    if renderer is None:
        raise ValueError(f"Unknown format: {output_format}. Choose from {list(renderers.keys())}")

    renderer(graph, output_path, included, edges, layout)


def serve_graph(
    graph: GraphStore,
    rag: Optional[RAGSystem] = None,
    output_path: str = "graph_viz.html",
    port: int = 8090,
    max_nodes: Optional[int] = None,
    relation_type: Optional[list[str]] = None,
    focus_node: Optional[str] = None,
    max_depth: int = 2,
    layout: str = "kamada_kawai",
    open_browser: bool = True,
) -> None:
    """Generate graph HTML with live RAG fetch and serve via HTTP.

    The HTML is generated with ``api_base_url`` set to the local server,
    so clicking a node fetches document text from ``/api/document/<id>``
    instead of embedding it in the HTML.
    """
    graph = _as_viz(graph)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    render_graph_viz(
        graph,
        output_path=output_path,
        output_format="html",
        max_nodes=max_nodes,
        relation_type=relation_type,
        focus_node=focus_node,
        max_depth=max_depth,
        layout=layout,
        api_base_url=f"http://localhost:{port}",
    )

    graph_html = out.read_text(encoding="utf-8")

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/document/"):
                doc_id = self.path[len("/api/document/"):]
                node = graph._nodes.get(doc_id)
                if node is None:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "not found"}, ensure_ascii=False).encode())
                    return
                links_info: list[dict] = []
                edges = graph.get_edges_batch({doc_id}, max_depth=1)
                for neighbour_id, rels in edges.get(doc_id, {}).items():
                    neighbour = graph._nodes.get(neighbour_id, {})
                    nlabel = _make_label(neighbour.get("text", ""))
                    for r in rels:
                        links_info.append({
                            "id": neighbour_id,
                            "label": nlabel,
                            "relation": r["relation"],
                            "weight": r["weight"],
                            "direction": r["direction"],
                        })
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "id": doc_id,
                    "text": node["text"],
                    "metadata": node.get("metadata", {}),
                    "links": links_info,
                }, ensure_ascii=False).encode())
                return
            if self.path.startswith("/api/search"):
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                q = params.get("q", [""])[0]
                if not q:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "query param 'q' is required"}, ensure_ascii=False).encode())
                    return
                if rag is None:
                    self.send_response(501)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "RAG system not available"}, ensure_ascii=False).encode())
                    return
                try:
                    results = rag.search_hybrid(q, k=10)
                    items = [{"doc_id": r[0], "text": r[1], "score": r[2], "label": _make_label(r[1])} for r in results]
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"query": q, "k": len(items), "results": items}, ensure_ascii=False).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode())
                return
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(graph_html.encode("utf-8"))
                return
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "not found"}, ensure_ascii=False).encode())

        def log_message(self, format, *args):
            print(f"[graph] {args[0]} {args[1]} {args[2] if len(args) > 2 else ''}", file=sys.stderr)

    server = http.server.HTTPServer(("", port), _Handler)
    url = f"http://localhost:{port}"

    print(f"🌐 Graph server: {url}", file=sys.stderr)
    print(f"   API:          {url}/api/document/<id>", file=sys.stderr)
    print("   Press Ctrl+C to stop", file=sys.stderr)

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", file=sys.stderr)
        server.server_close()
