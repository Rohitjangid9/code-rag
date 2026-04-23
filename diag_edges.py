"""Diagnose actual edge data to understand what's resolving."""
from cce.config import get_settings
from cce.index.db import get_db

s = get_settings()
db = get_db(s.paths.sqlite_db)
conn = db.conn

# Show sample IMPORTS edges
print("=== Sample IMPORTS edges (first 15) ===")
rows = conn.execute(
    "SELECT e.kind, e.file_path, e.line, "
    "ss.qualified_name as src_qn, ds.qualified_name as dst_qn "
    "FROM edges e "
    "LEFT JOIN symbols ss ON ss.id = e.src_id "
    "LEFT JOIN symbols ds ON ds.id = e.dst_id "
    "WHERE e.kind='IMPORTS' LIMIT 15"
).fetchall()
for r in rows:
    print(f"  {r['src_qn'] or '?'} -> {r['dst_qn'] or '?'} @ {r['file_path']}:{r['line']}")

# Check if planner_node has any edges at all
print("\n=== All edges involving planner_node ===")
pn = conn.execute(
    "SELECT id FROM symbols WHERE qualified_name='cce.agents.nodes.planner_node'"
).fetchone()
if pn:
    pn_id = pn["id"]
    edges = conn.execute(
        "SELECT e.kind, e.file_path, e.line, "
        "ss.qualified_name as src_qn, ds.qualified_name as dst_qn "
        "FROM edges e "
        "LEFT JOIN symbols ss ON ss.id = e.src_id "
        "LEFT JOIN symbols ds ON ds.id = e.dst_id "
        "WHERE e.src_id=? OR e.dst_id=?",
        (pn_id, pn_id)
    ).fetchall()
    print(f"  planner_node edges: {len(edges)}")
    for r in edges:
        print(f"  {r['kind']} {r['src_qn']} -> {r['dst_qn']} @ {r['file_path']}:{r['line']}")
else:
    print("  planner_node NOT in DB!")

# Show raw_edges diagnostic from actual resolver (graph.py specifically)
print("\n=== raw edges from resolve_python_file(graph.py) breakdown ===")
from pathlib import Path
from cce.parsers.base import ParsedFile
from cce.graph.schema import Language
from cce.parsers.python_resolver import resolve_python_file

root = Path(".").resolve()
graph_py = root / "cce/agents/graph.py"
source = graph_py.read_text(encoding="utf-8")
parsed = ParsedFile(path=graph_py, rel_path="cce/agents/graph.py",
                    language=Language.PYTHON, source=source)
edges = resolve_python_file(parsed, root)
refs = [(e.kind.value, e.dst_qualified_name, e.line) for e in edges
        if "nodes" in e.dst_qualified_name or "planner" in e.dst_qualified_name]
print(f"  edges referencing planner/nodes: {len(refs)}")
for k, dst, ln in refs:
    resolved = conn.execute("SELECT id FROM symbols WHERE qualified_name=?", (dst,)).fetchone()
    print(f"  {k} -> {dst} line={ln} resolvable={'YES' if resolved else 'NO'}")
