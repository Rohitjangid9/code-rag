"""Show edge breakdown after re-index."""
from cce.config import get_settings
from cce.index.db import get_db

s = get_settings()
db = get_db(s.paths.sqlite_db)
conn = db.conn

rows = conn.execute("SELECT kind, count(*) cnt FROM edges GROUP BY kind ORDER BY cnt DESC").fetchall()
print("Edge kinds:")
for r in rows:
    print(f"  {r['kind']}: {r['cnt']}")

# Check REFERENCES edges specifically
ref_rows = conn.execute(
    "SELECT e.file_path, e.line, ss.qualified_name src_qn, ds.qualified_name dst_qn "
    "FROM edges e "
    "LEFT JOIN symbols ss ON ss.id = e.src_id "
    "LEFT JOIN symbols ds ON ds.id = e.dst_id "
    "WHERE e.kind='REFERENCES' LIMIT 20"
).fetchall()
print(f"\nSample REFERENCES edges ({len(ref_rows)}):")
for r in ref_rows:
    print(f"  {r['src_qn']} -> {r['dst_qn']} @ {r['file_path']}:{r['line']}")

# Check planner_node edges
pn = conn.execute("SELECT id FROM symbols WHERE qualified_name='cce.agents.nodes.planner_node'").fetchone()
if pn:
    edges = conn.execute(
        "SELECT e.kind, e.file_path, e.line, ss.qualified_name src_qn "
        "FROM edges e LEFT JOIN symbols ss ON ss.id = e.src_id "
        "WHERE e.dst_id=?", (pn["id"],)
    ).fetchall()
    print(f"\nEdges TO planner_node: {len(edges)}")
    for e in edges:
        print(f"  {e['kind']} from {e['src_qn']} @ {e['file_path']}:{e['line']}")
else:
    print("\nplanner_node NOT in DB!")
