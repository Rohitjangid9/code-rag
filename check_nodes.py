import sys
sys.path.insert(0, "src")
from cce.parsers.tree_sitter_parser import _get_parser
from cce.graph.schema import Language

parser = _get_parser(Language.TSX)
source = b'''
export function UserCard({ userId }) {
  return (
    <div className="user-card">
      <Avatar src={user?.avatarUrl} />
      <h2>{user?.name}</h2>
    </div>
  );
}
'''
tree = parser.parse(source)

def visit(node, depth=0):
    if node.type in ("jsx_element", "jsx_self_closing_element", "jsx_opening_element",
                       "jsx_closing_element", "identifier", "member_expression", "call_expression"):
        print("  " * depth + node.type + " | " + source[node.start_byte:node.end_byte].decode("utf-8", "replace").strip().replace("\n", " "))
    for child in node.children:
        visit(child, depth + 1)

visit(tree.root_node)
