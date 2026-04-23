"""Simulate _apply_edges for graph.py and trace exactly what happens."""
from pathlib import Path
from cce.config import get_settings
from cce.index.db import get_db
from cce.graph.sqlite_store import SQLiteGraphStore
from cce.index.symbol_store import SymbolStore
from cce.parsers.base import ParsedFile
from cce.graph.schema import Language
from cce.parsers.tree_sitter_parser import TreeSitterParser
from cce.parsers.python_resolver import resolve_python_file

s = get_settings()
db = get_db(s.paths.sqlite_db)
conn = db.conn
graph = SQLiteGraphStore(db)

root = Path(".").resolve()
graph_py = root / "cce/agents/graph.py"
source = graph_py.read_text(encoding="utf-8")
rel = "cce/agents/graph.py"

parser = TreeSitterParser()
parsed = parser.parse(graph_py, rel, Language.PYTHON, source)
print(f"parsed.nodes: {len(parsed.nodes)}")
print(f"parsed.raw_edges: {len(parsed.raw_edges)}")

print("\n=== Running _resolve_references (resolve_python_file) ===")
jedi_edges = resolve_python_file(parsed, root)
print(f"Jedi raw edges: {len(jedi_edges)}")

all_raw = list(parsed.raw_edges) + jedi_edges
print(f"Total raw edges: {len(all_raw)}")

ref_edges = [e for e in all_raw if "nodes" in e.dst_qualified_name]
print(f"Edges referencing nodes: {len(ref_edges)}")

print("\n=== Edge resolution trace ===")
success = 0
skipped_no_dst = 0
skipped_api = 0
for re_ in all_raw:
    if re_.dst_qualified_name.startswith("api:"):
        skipped_api += 1
        continue
    dst_id = graph.resolve_qname(re_.dst_qualified_name)
    if dst_id:
        # check src_id exists
        src_row = conn.execute("SELECT id FROM symbols WHERE id=?", (re_.src_id,)).fetchone()
        if src_row:
            try:
                graph.upsert_edge(
                    src_id=re_.src_id,
                    dst_id=dst_id,
                    kind=re_.kind,
                    file_path=re_.file_path,
                    line=re_.line,
                    confidence=re_.confidence,
                )
                success += 1
                if "nodes" in re_.dst_qualified_name:
                    print(f"  INSERTED: {re_.kind.value} -> {re_.dst_qualified_name} line={re_.line}")
            except Exception as ex:
                print(f"  FAILED: {re_.kind.value} -> {re_.dst_qualified_name}: {ex}")
        else:
            if "nodes" in re_.dst_qualified_name:
                print(f"  SKIP (src_id not in DB): {re_.kind.value} src={re_.src_id[:12]} -> {re_.dst_qualified_name}")
    else:
        skipped_no_dst += 1

conn.commit()
print(f"\nInserted: {success}, skipped_no_dst: {skipped_no_dst}, skipped_api: {skipped_api}")

# Verify
ref_count = conn.execute("SELECT count(*) FROM edges WHERE kind='REFERENCES'").fetchone()[0]
calls_count = conn.execute("SELECT count(*) FROM edges WHERE kind='CALLS'").fetchone()[0]
print(f"REFERENCES in DB: {ref_count}, CALLS in DB: {calls_count}")
