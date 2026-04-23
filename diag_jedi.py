"""Diagnose Jedi reference resolution on graph.py."""
from pathlib import Path
import jedi

root = Path(".").resolve()
graph_py = root / "cce/agents/graph.py"
source = graph_py.read_text(encoding="utf-8")
project = jedi.Project(path=str(root), smart_sys_path=True)
script = jedi.Script(code=source, path=str(graph_py), project=project)

# Try to resolve 'planner_node' at line 75 (add_node call)
print("Testing Jedi goto on graph.py")
try:
    defs = script.goto(line=75, column=35, follow_imports=True, follow_builtin_imports=False)
    print(f"  goto(75,35): {[(d.name, str(d.module_path)) for d in defs]}")
except Exception as e:
    print(f"  goto failed: {e}")

# Also test infer/complete on planner_node usage
for line_no in range(70, 85):
    line_text = source.split("\n")[line_no - 1]
    if "planner_node" in line_text:
        col = line_text.find("planner_node") + len("planner_node")
        try:
            defs = script.goto(line=line_no, column=col, follow_imports=True)
            print(f"  line {line_no}: '{line_text.strip()}' -> {[(d.name, str(d.module_path)) for d in defs]}")
        except Exception as e:
            print(f"  line {line_no} error: {e}")

# Now run the actual resolver
print("\nRunning resolve_python_file on graph.py...")
from cce.parsers.python_resolver import resolve_python_file
from cce.parsers.base import ParsedFile
from cce.graph.schema import Language

parsed = ParsedFile(
    path=graph_py,
    rel_path="cce/agents/graph.py",
    language=Language.PYTHON,
    source=source,
)
edges = resolve_python_file(parsed, root)
print(f"  {len(edges)} raw edges from resolver")
for e in edges[:20]:
    print(f"  {e.kind.value} {e.dst_qualified_name} via {e.resolver_method} line={e.line}")
