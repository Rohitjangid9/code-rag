"""Trace exactly why REFERENCES edges from graph.py don't reach the DB."""
from pathlib import Path
from cce.config import get_settings
from cce.index.db import get_db
from cce.graph.sqlite_store import SQLiteGraphStore
from cce.parsers.base import ParsedFile
from cce.parsers.tree_sitter_parser import _node_id_from_qname
from cce.graph.schema import Language
from cce.parsers.python_resolver import resolve_python_file

s = get_settings()
db = get_db(s.paths.sqlite_db)
conn = db.conn
graph = SQLiteGraphStore(db)

root = Path(".").resolve()
graph_py = root / "cce/agents/graph.py"
source = graph_py.read_text(encoding="utf-8")

# Build a real ParsedFile with actual nodes (from tree-sitter)
from cce.parsers.tree_sitter_parser import TreeSitterParser
parser = TreeSitterParser()
parsed = parser.parse(graph_py, "cce/agents/graph.py", Language.PYTHON, source)

print(f"ParsedFile has {len(parsed.nodes)} nodes")
for n in parsed.nodes:
    print(f"  {n.qualified_name}  id={n.id}  lines {n.line_start}-{n.line_end}")

print("\n=== Resolving references ===")
edges = resolve_python_file(parsed, root)
ref_edges = [e for e in edges if "nodes" in e.dst_qualified_name or "planner" in e.dst_qualified_name]
print(f"Total raw edges: {len(edges)}, relevant REFERENCES: {len(ref_edges)}")

print("\n=== Attempting edge insertion ===")
for re_ in ref_edges:
    dst_id = graph.resolve_qname(re_.dst_qualified_name)
    # Check if src_id exists in DB
    src_row = conn.execute("SELECT qualified_name FROM symbols WHERE id=?", (re_.src_id,)).fetchone()
    print(f"  kind={re_.kind.value} src_id={re_.src_id[:12]}... ({src_row['qualified_name'] if src_row else 'NOT IN DB'})")
    print(f"    dst={re_.dst_qualified_name} dst_id={dst_id[:12] + '...' if dst_id else 'NONE'}")
    if dst_id and src_row:
        try:
            graph.upsert_edge(
                src_id=re_.src_id,
                dst_id=dst_id,
                kind=re_.kind,
                file_path=re_.file_path,
                line=re_.line,
                confidence=re_.confidence,
            )
            print(f"    -> INSERTED OK")
        except Exception as ex:
            print(f"    -> INSERT FAILED: {ex}")
    elif not src_row:
        print(f"    -> SKIPPED: src_id not in DB!")
    else:
        print(f"    -> SKIPPED: dst not resolvable")

# Verify
conn.commit()
count = conn.execute("SELECT count(*) FROM edges WHERE kind='REFERENCES'").fetchone()[0]
print(f"\nTotal REFERENCES edges in DB now: {count}")
