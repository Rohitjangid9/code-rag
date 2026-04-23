"""Verify all fixes after full re-index."""
from cce.retrieval.tools import _pipeline, find_callers, search_code, list_routes

p = _pipeline()
conn = p.symbol_store._db.conn

# 1. planner_node with CORRECT new qname
print("=== find_callers('cce.agents.nodes.planner_node') ===")
callers = find_callers("cce.agents.nodes.planner_node", include_refs=True)
print(f"  {len(callers)} callers")
for c in callers:
    print(f"  {c.qualified_name} @ {c.file_path}:{c.line_start}")

# 2. Check what edges point to planner_node
pn_node = p.symbol_store.get_by_qname("cce.agents.nodes.planner_node")
if pn_node:
    print(f"\nplanner_node id={pn_node.id} line_start={pn_node.line_start}")
    edges = conn.execute(
        "SELECT e.kind, e.file_path, e.line, s.qualified_name as src_qname "
        "FROM edges e LEFT JOIN symbols s ON s.id = e.src_id "
        "WHERE e.dst_id = ?",
        (pn_node.id,),
    ).fetchall()
    print(f"  Edges to planner_node: {len(edges)}")
    for e in edges:
        print(f"  {e['kind']} from={e['src_qname']} @ {e['file_path']}:{e['line']}")
else:
    print("planner_node NOT FOUND!")

# 3. Total edge counts by kind
print("\n=== Edge counts by kind ===")
kinds = conn.execute("SELECT kind, COUNT(*) as c FROM edges GROUP BY kind ORDER BY c DESC").fetchall()
for k in kinds:
    print(f"  {k['kind']}: {k['c']}")

# 4. Search IndexPipeline - should find exactly 1 symbol
print("\n=== search_code('IndexPipeline') ===")
hits = search_code("IndexPipeline", mode="lexical", k=3)
for h in hits:
    print(f"  {h.node.qualified_name if h.node else '?'} @ {h.path}:{h.line_start} score={h.score:.2f}")

# 5. Check has_symbol_body — snippet should be > 50 chars now
print("\n=== Snippet richness check (has_symbol_body) ===")
from cce.retrieval.hybrid import HybridRetriever
hr = HybridRetriever(p.symbol_store, p.lexical_store, p.graph_store, p._settings)
results = hr.retrieve("planner_node agent state", k=5)
for r in results:
    body_ok = len(r.snippet or "") > 50
    print(f"  {r.node.qualified_name if r.node else '?'} snippet_len={len(r.snippet or '')} has_body={body_ok}")

# 6. FastAPI routes from server/routes/
print("\n=== FastAPI routes (cce/server) ===")
routes = list_routes(framework="fastapi")
print(f"  {len(routes)} FastAPI routes")
for r in routes[:8]:
    print(f"  {' '.join(r.methods)} {r.effective_path} -> {r.handler_qname}")
