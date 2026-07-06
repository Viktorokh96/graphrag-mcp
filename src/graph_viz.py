"""Визуализация графа знаний: HTML (vis.js), DOT, JSON, ASCII."""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from src.graph_store import GraphKnowledgeBase

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
    graph: GraphKnowledgeBase,
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
    graph: GraphKnowledgeBase, output_path: str,
    included: set[str], edges: list[dict], layout: str,
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
        first_line = node["text"].split("\n")[0].strip()[:300].replace("<", "&lt;").replace(">", "&gt;")
        title = (
            f"<b>ID:</b> {nid}<br>"
            f"<b>Text:</b> {first_line}<br>"
            f"<b>Degree:</b> {d}<br>"
            f"<b>Metadata:</b> {meta_str}"
        )

        nodes_js.append({
            "id": nid,
            "label": label,
            "group": f"c{ci}",
            "title": title,
            "color": {"background": bg, "border": border},
            "font": {"size": fs, "color": "#fff", "face": "Segoe UI"},
            "shape": shape,
            "borderWidth": 2,
            "margin": {"top": 6, "bottom": 6, "left": 8, "right": 8},
        })

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
#toast {{
  position:fixed; bottom:100px; left:50%; transform:translateX(-50%); z-index:999;
  background:rgba(15,15,35,0.9); color:#e0e0e0;
  padding:8px 20px; border-radius:10px; font-size:13px;
  border:1px solid rgba(255,255,255,0.08);
  backdrop-filter:blur(8px); opacity:0; transition:opacity .3s; pointer-events:none;
}}
#toast.show {{ opacity:1; }}
</style>
</head>
<body>

<div id="stats">{stats_line}</div>

<div id="toolbar">
  <button onclick="fitGraph()" title="Fit to screen">⟷ Fit</button>
  <button onclick="toggleSearch()" title="Search nodes (Ctrl+F)" id="btn-search">🔍</button>
  <button onclick="resetPhysics()" title="Reset physics">⟳ Reset</button>
</div>

<div id="search-box">
  <input type="text" placeholder="Search nodes by label or text..." id="search-input"
         oninput="searchNodes(this.value)" onkeydown="if(event.key==='Escape')hideSearch()">
  <span class="count" id="search-count"></span>
</div>

<div id="mynetwork"></div>

<div id="legend">
  <div class="title">🎯 Community</div>
  {"".join(legend_rows)}
</div>

<div id="toast"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<script>
const nodes = new vis.DataSet({nodes_json_raw});
const edges = new vis.DataSet({edges_json_raw});
const groups = {groups_json_raw};

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
function resetPhysics() {{
  network.setOptions({{ physics: {{ stabilization: {{ iterations: 100 }} }} }});
  setTimeout(() => network.stopSimulation?.(), 3000);
}}
function toggleSearch() {{
  const box = document.getElementById('search-box');
  box.classList.toggle('visible');
  if (box.classList.contains('visible')) {{
    document.getElementById('search-input').focus();
  }} else {{
    document.getElementById('search-input').value = '';
    searchNodes('');
  }}
}}
function hideSearch() {{
  document.getElementById('search-box').classList.remove('visible');
  document.getElementById('search-input').value = '';
  searchNodes('');
}}
function searchNodes(query) {{
  const q = query.toLowerCase().trim();
  const ids = nodes.getIds();
  const count = document.getElementById('search-count');
  if (!q) {{
    nodes.forEach(n => {{ delete n.hidden; nodes.update(n); nodes.get(n.id); }});
    count.textContent = '';
    fitGraph();
    return;
  }}
  let matchCount = 0;
  ids.forEach(id => {{
    const n = nodes.get(id);
    const label = (n.label || '').toLowerCase();
    const title = (n.title || '').toLowerCase();
    const match = label.includes(q) || title.includes(q);
    n.hidden = !match;
    if (match) matchCount++;
    nodes.update(n);
  }});
  count.textContent = matchCount + ' found';
  fitGraph();
}}
function toast(msg) {{
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 2000);
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
</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Saved: {output_path}", file=sys.stderr)


def _render_dot(
    graph: GraphKnowledgeBase, output_path: str,
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
    graph: GraphKnowledgeBase, output_path: str,
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
    graph: GraphKnowledgeBase, output_path: str,
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
    graph: GraphKnowledgeBase,
    output_path: str = "graph_viz.html",
    output_format: str = "html",
    max_nodes: Optional[int] = None,
    relation_type: Optional[list[str]] = None,
    focus_node: Optional[str] = None,
    max_depth: int = 2,
    layout: str = "kamada_kawai",
) -> None:
    included, edges = _build_subgraph(graph, max_nodes, relation_type, focus_node, max_depth)

    if not included:
        print("Empty graph — nothing to render", file=sys.stderr)
        return

    renderers = {
        "html": _render_html,
        "dot": _render_dot,
        "json": _render_json,
        "ascii": _render_ascii,
    }

    renderer = renderers.get(output_format)
    if renderer is None:
        raise ValueError(f"Unknown format: {output_format}. Choose from {list(renderers.keys())}")

    renderer(graph, output_path, included, edges, layout)
