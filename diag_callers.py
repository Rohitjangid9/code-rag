"""Diagnose: why do find_callers return line=0, and why IndexPipeline is hard to find."""
from cce.retrieval.tools import _pipeline, find_callers, search_code, list_routes

p = _pipeline()
conn = p.symbol_store._db.conn

# 1. find_callers for planner_node
print("=== find_callers('nodes.planner_node') ===")
callers = find_callers("nodes.planner_node", include_refs=True)
print(f"  {len(callers)} callers")
for c in callers:
    print(f"  {c.qualified_name} {c.file_path}:{c.line_start}")

# 2. What edges point TO planner_node?
pn_node = p.symbol_store.get_by_qname("nodes.planner_node")
if pn_node:
    print(f"\nplanner_node id={pn_node.id}")
    edges = conn.execute(
        "SELECT e.*, s.qualified_name as src_qname, s.line_start as src_line "
        "FROM edges e LEFT JOIN symbols s ON s.id = e.src_id "
        "WHERE e.dst_id = ?",
        (pn_node.id,),
    ).fetchall()
    print(f"Edges pointing to planner_node: {len(edges)}")
    for e in edges:
        print(f"  kind={e['kind']} src_qname={e['src_qname']} file={e['file_path']} edge_line={e['line']}")
        print(f"    src has symbol? {'yes' if e['src_qname'] else 'NO (synthesized module node)'}")
else:
    print("planner_node NOT FOUND in symbol store!")

# 3. search_code for IndexPipeline
print("\n=== search_code('IndexPipeline') top 5 ===")
hits = search_code("IndexPipeline", mode="lexical", k=5)
for h in hits:
    qn = h.node.qualified_name if h.node else "(no node)"
    print(f"  {qn} @ {h.path}:{h.line_start} score={h.score:.2f}")

# 4. Direct symbol lookup for IndexPipeline
print("\n=== symbols named 'IndexPipeline' ===")
rows = conn.execute(
    "SELECT qualified_name, file_path, line_start FROM symbols WHERE name='IndexPipeline'"
).fetchall()
print(f"  {len(rows)} found")
for r in rows:
    print(f"  {r['qualified_name']} @ {r['file_path']}:{r['line_start']}")

# 5. list_routes() result count
print("\n=== list_routes() ===")
routes = list_routes()
print(f"  {len(routes)} routes")
for r in routes[:5]:
    print(f"  [{r.framework}] {' '.join(r.methods)} {r.effective_path} -> {r.handler_qname}")

# 6. Check lex_sym_fts for symbol bodies
print("\n=== lex_sym_fts sample ===")
sym_rows = conn.execute(
    "SELECT path, qualified_name, line_start, line_end, length(content) as content_len "
    "FROM lex_sym_fts LIMIT 5"
).fetchall()
print(f"  lex_sym_fts rows: {len(sym_rows)}")
for r in sym_rows:
    print(f"  {r['qualified_name']} {r['path']}:{r['line_start']}-{r['line_end']} body_len={r['content_len']}")

# Full lex_sym_fts count
total_sym = conn.execute("SELECT COUNT(*) as cnt FROM lex_sym_fts").fetchone()
print(f"  Total lex_sym_fts rows: {total_sym['cnt']}")
