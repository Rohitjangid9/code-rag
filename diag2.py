"""Post-fix diagnostic: confirm correct module qnames and edge creation."""
from cce.config import get_settings
from cce.index.db import get_db

s = get_settings()
db = get_db(s.paths.sqlite_db)
conn = db.conn

total = conn.execute("SELECT COUNT(*) as c FROM symbols").fetchone()["c"]
print(f"Total symbols: {total}")

rows = conn.execute("SELECT qualified_name FROM symbols WHERE name='planner_node'").fetchall()
print(f"planner_node records: {[r['qualified_name'] for r in rows]}")

try:
    hashes = conn.execute("SELECT COUNT(*) as c FROM file_hashes").fetchone()["c"]
    print(f"File hashes: {hashes}")
except Exception as e:
    print(f"No file_hashes: {e}")

edges = conn.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]
print(f"Total edges: {edges}")

kinds = conn.execute("SELECT kind, COUNT(*) as c FROM symbols GROUP BY kind ORDER BY c DESC").fetchall()
print("Symbol kinds:")
for k in kinds:
    print(f"  {k['kind']}: {k['c']}")

# Check qnames for nodes.py symbols
nodes_syms = conn.execute(
    "SELECT qualified_name, file_path, line_start FROM symbols WHERE file_path LIKE '%agents%nodes%' LIMIT 5"
).fetchall()
print("\nSymbols from agents/nodes.py:")
for r in nodes_syms:
    print(f"  {r['qualified_name']} @ {r['file_path']}:{r['line_start']}")

# Check for IndexPipeline duplicates
ip = conn.execute("SELECT qualified_name, file_path FROM symbols WHERE name='IndexPipeline'").fetchall()
print(f"\nIndexPipeline records: {len(ip)}")
for r in ip:
    print(f"  {r['qualified_name']} @ {r['file_path']}")
