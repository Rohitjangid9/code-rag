"""Quick diagnostic: what's actually in the current index."""
from cce.config import get_settings
from cce.index.db import get_db

settings = get_settings()
db = get_db(settings.paths.sqlite_db)
conn = db.conn

routes = conn.execute("SELECT COUNT(*) as cnt FROM symbols WHERE kind IN ('Route','URLPattern')").fetchone()
print(f"Routes in index: {routes['cnt']}")

total = conn.execute("SELECT COUNT(*) as cnt FROM symbols").fetchone()
print(f"Total symbols: {total['cnt']}")

edges = conn.execute("SELECT COUNT(*) as cnt FROM edges").fetchone()
print(f"Total edges: {edges['cnt']}")

sample_edges = conn.execute("SELECT src_id, dst_id, kind, file_path, line FROM edges LIMIT 10").fetchall()
print("Sample edges:")
for e in sample_edges:
    print(f"  {e['kind']} file={e['file_path']} line={e['line']}")

syms = conn.execute("SELECT qualified_name, line_start, line_end, file_path FROM symbols WHERE name='planner_node'").fetchall()
print("planner_node symbols:")
for s in syms:
    print(f"  {s['qualified_name']} {s['file_path']}:{s['line_start']}-{s['line_end']}")

kinds = conn.execute("SELECT kind, COUNT(*) as cnt FROM symbols GROUP BY kind ORDER BY cnt DESC").fetchall()
print("Symbol kinds:")
for k in kinds:
    print(f"  {k['kind']}: {k['cnt']}")

# Check fast api extractor - does it find router vars?
print("\n--- FastAPI extractor test ---")
from cce.extractors.fastapi_extractor import FastAPIExtractor
from pathlib import Path
extractor = FastAPIExtractor()
sample_file = Path("cce/server/routes/agent.py")
source = sample_file.read_text(encoding="utf-8")
print(f"can_handle: {extractor.can_handle(sample_file, source)}")
from cce.parsers.tree_sitter_parser import _get_parser
from cce.graph.schema import Language
src = source.encode("utf-8")
tree = _get_parser(Language.PYTHON).parse(src)
root = tree.root_node
print(f"Root child types: {[c.type for c in root.children[:10]]}")
router_vars = extractor._collect_router_vars(root, src)
print(f"Router vars found: {router_vars}")
data = extractor.extract(sample_file, "cce/server/routes/agent.py", source)
print(f"Nodes extracted: {len(data.nodes)}")
for n in data.nodes:
    print(f"  {n.kind.value} {n.name} line={n.line_start}")
