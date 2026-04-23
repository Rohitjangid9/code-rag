import sys
sys.path.insert(0, r'D:\Learn\code-context-extractor')

import importlib
import cce.parsers.python_resolver
importlib.reload(cce.parsers.python_resolver)

from pathlib import Path
import tempfile
from cce.parsers.python_resolver import resolve_python_file, _find_reference_sites, _jedi_def_to_qname
from cce.graph.schema import Language
from cce.parsers.base import ParsedFile
from cce.parsers.tree_sitter_parser import TreeSitterParser, _text
import jedi

with tempfile.TemporaryDirectory() as td:
    src_file = Path(td) / 'refs.py'
    src_file.write_text('def foo():\n    pass\n\nbar = {"x": foo}\nhandler(foo)\n')
    source = src_file.read_text()
    parser = TreeSitterParser()
    pf = parser.parse(src_file, 'refs.py', Language.PYTHON, source)
    print(f'Parsed nodes: {[(n.name, n.kind) for n in pf.nodes]}')

    src_bytes = source.encode('utf-8')
    from cce.parsers.tree_sitter_parser import _get_parser
    tree = _get_parser(Language.PYTHON).parse(src_bytes)
    sites = _find_reference_sites(tree.root_node, src_bytes, pf, Path(td))
    print(f'Ref sites: {len(sites)}')

    project = jedi.Project(path=td, smart_sys_path=True)
    script = jedi.Script(code=source, path=str(src_file), project=project)

    for ref_node, src_qname, confidence in sites:
        line = ref_node.start_point[0] + 1
        col = ref_node.start_point[1]
        print(f'Trying goto line={line} col={col} name={_text(ref_node, src_bytes)}')
        try:
            defs = script.goto(line=line, column=col, follow_imports=True, follow_builtin_imports=False)
        except Exception as exc:
            print(f'  goto failed: {exc}')
            continue
        print(f'  defs={[(d.name, str(d.module_path)) for d in defs]}')
        for d in defs:
            if d.module_path is None:
                print('  module_path is None')
                continue
            dst_qname = _jedi_def_to_qname(d, Path(td))
            print(f'  dst_qname={dst_qname}')

    edges = resolve_python_file(pf, Path(td))
    print(f'Edges: {len(edges)}')
    for e in edges:
        print(f'  {e.kind} {e.src_id} -> {e.dst_qualified_name}')
